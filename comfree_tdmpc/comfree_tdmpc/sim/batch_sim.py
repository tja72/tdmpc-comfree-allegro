"""Batched ComFree-Sim wrapper used as the world model.

Everything TD-MPC normally learns about dynamics is provided by this class:
a GPU-resident bank of `nworld` independent simulations that share one MuJoCo
model but can each carry *different physical parameters*.  That per-world
parameter capability is what makes the SysID gradient and its line search
embarrassingly parallel -- ComFree inherits it from MuJoCo-Warp, whose kernels
index every model array as `arr[worldid % arr.shape[0]]`.

The state of a ComFree simulation with position actuators is exactly
(qpos, qvel): the analytic contact model keeps no solver warm-start and the
actuators have no activation state.  State injection is therefore exact, which
is what lets SysID score short chunks resampled from the replay buffer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

import comfree_warp as cfwarp
from comfree_warp import mujoco_warp as mjwarp

from comfree_tdmpc.sim.params import NUM_PARAMS, PARAM_NAMES

# Cube state layout inside qpos/qvel for the Allegro cube scene: the hand joints
# come first (they are included before the free-floating object in the XML).
HAND_NQ = 16
HAND_NV = 16


def comfree_repo_root() -> Path:
    """Locate the (editable-installed) ComFree repo so we can reuse its assets."""
    env = os.environ.get("COMFREE_REPO")
    if env:
        return Path(env)
    return Path(cfwarp.__file__).resolve().parents[1]


def allegro_scene_path() -> Path:
    return comfree_repo_root() / "benchmark" / "test_data" / "allegro" / "env_allegro_cube.xml"


def load_spec(add_goal_ghost: bool = False, timestep: float | None = None) -> mujoco.MjModel:
    """Compile the Allegro cube scene, optionally with a visual goal marker.

    The scene is edited through `MjSpec` at load time rather than by copying the
    XML, so this repo stays independent of the ComFree checkout.  The goal ghost
    is a *mocap* body: it adds no qpos/qvel entries, so a model compiled with the
    ghost is state-compatible with one compiled without it and can be used purely
    for rendering.
    """
    spec = mujoco.MjSpec.from_file(str(allegro_scene_path()))
    if timestep is not None:
        spec.option.timestep = timestep
    if add_goal_ghost:
        body = spec.worldbody.add_body()
        body.name = "goal_ghost"
        body.pos = [0.0, -0.16, 0.06]
        body.mocap = True
        geom = body.add_geom()
        geom.name = "goal_ghost_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.028, 0.028, 0.028]
        geom.material = "obj_material"
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 0
        geom.mass = 0.0
        geom.rgba = [1.0, 1.0, 1.0, 0.45]
    return spec.compile()


@dataclass
class SimConfig:
    nworld: int = 256
    n_substeps: int = 10
    nconmax: int = 64
    njmax: int = 1000
    timestep: float | None = None
    capture_graph: bool = True
    ccd_iterations: int = 50
    device: str = "cuda:0"


class BatchSim:
    """`nworld` ComFree simulations stepped in lockstep, with per-world params."""

    def __init__(self, cfg: SimConfig, mjm: mujoco.MjModel | None = None):
        self.cfg = cfg
        wp.set_device(cfg.device)
        self.torch_device = torch.device("cuda" if cfg.device.startswith("cuda") else "cpu")

        self.mjm = mjm if mjm is not None else load_spec(timestep=cfg.timestep)
        self.mjm.opt.ccd_iterations = cfg.ccd_iterations
        self.mjd = mujoco.MjData(self.mjm)
        self._load_keyframe()

        self.nworld = cfg.nworld
        self.nq, self.nv, self.nu = self.mjm.nq, self.mjm.nv, self.mjm.nu
        self.n_substeps = cfg.n_substeps
        self.control_dt = self.mjm.opt.timestep * cfg.n_substeps

        self.m = cfwarp.put_model(
            self.mjm,
            comfree_stiffness=np.full(cfg.nworld, 0.1, dtype=np.float32),
            comfree_damping=np.full(cfg.nworld, 1e-3, dtype=np.float32),
        )
        self.d = cfwarp.put_data(
            self.mjm, self.mjd, nworld=cfg.nworld, nconmax=cfg.nconmax, njmax=cfg.njmax
        )

        self._make_params_per_world()
        self._resolve_indices()

        # Zero-copy torch views onto the simulation state.
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)

        self._graph = None
        if cfg.capture_graph:
            self._capture()

    def _load_keyframe(self) -> None:
        if self.mjm.nkey > 0:
            key = self.mjm.key(0)
            self.mjd.qpos[:] = key.qpos
            self.mjd.qvel[:] = key.qvel
            if self.mjm.nu:
                self.mjd.ctrl[:] = key.ctrl
        mujoco.mj_forward(self.mjm, self.mjd)
        self.key_qpos = self.mjd.qpos.copy()
        self.key_qvel = self.mjd.qvel.copy()
        self.key_ctrl = self.mjd.ctrl.copy() if self.mjm.nu else np.zeros(0)

    def _make_params_per_world(self) -> None:
        """Replace the shared (broadcast) model arrays with per-world copies.

        `put_model` builds these with a leading dimension of 1 and a zero stride;
        writing per-world values requires genuinely materialised arrays.  Values
        are tiled from the host model so we never read back a stride-0 array.
        """
        n = self.nworld
        self.m.body_mass = wp.array(
            np.tile(self.mjm.body_mass, (n, 1)).astype(np.float32), dtype=float
        )
        self.m.body_inertia = wp.array(
            np.tile(self.mjm.body_inertia, (n, 1, 1)).astype(np.float32), dtype=wp.vec3
        )
        self.m.geom_friction = wp.array(
            np.tile(self.mjm.geom_friction, (n, 1, 1)).astype(np.float32), dtype=wp.vec3
        )
        self.body_mass = wp.to_torch(self.m.body_mass)
        self.body_inertia = wp.to_torch(self.m.body_inertia)
        self.geom_friction = wp.to_torch(self.m.geom_friction)
        self.stiffness = wp.to_torch(self.m.comfree_stiffness)
        self.damping = wp.to_torch(self.m.comfree_damping)

        dev = self.torch_device
        self.base_body_mass = torch.as_tensor(
            self.mjm.body_mass, dtype=torch.float32, device=dev
        )
        self.base_body_inertia = torch.as_tensor(
            self.mjm.body_inertia, dtype=torch.float32, device=dev
        )

    def _resolve_indices(self) -> None:
        m = self.mjm
        self.cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.cube_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj")
        assert self.cube_body_id >= 0 and self.cube_geom_id >= 0, "cube body/geom not found"
        self.cube_qpos_adr = int(m.jnt_qposadr[m.body_jntadr[self.cube_body_id]])
        self.cube_qvel_adr = int(m.jnt_dofadr[m.body_jntadr[self.cube_body_id]])

        # Hand collision geoms: group 3 in the Allegro XML, excluding cube/floor.
        floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hand_geoms = [
            g
            for g in range(m.ngeom)
            if g not in (self.cube_geom_id, floor_id) and m.geom_contype[g] != 0
        ]
        self.friction_geom_ids = torch.as_tensor(
            [self.cube_geom_id] + hand_geoms, dtype=torch.long, device=self.torch_device
        )

        ctrl_range = torch.as_tensor(
            m.actuator_ctrlrange, dtype=torch.float32, device=self.torch_device
        )
        self.ctrl_low, self.ctrl_high = ctrl_range[:, 0], ctrl_range[:, 1]
        self.ctrl_mid = 0.5 * (self.ctrl_low + self.ctrl_high)
        self.ctrl_half = 0.5 * (self.ctrl_high - self.ctrl_low)

    def _capture(self) -> None:
        # Warm up (kernel compilation + module load) before capturing.
        for _ in range(2):
            cfwarp.step(self.m, self.d)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(self.n_substeps):
                cfwarp.step(self.m, self.d)
        self._graph = capture.graph
        wp.synchronize()

    def set_params(self, theta: torch.Tensor | np.ndarray, recompute_const: bool = True) -> None:
        """Write physical parameters.  `theta` is (P,) or (nworld, P), in *linear* units."""
        if isinstance(theta, np.ndarray):
            theta = torch.as_tensor(theta, dtype=torch.float32, device=self.torch_device)
        theta = theta.to(self.torch_device, torch.float32)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0).expand(self.nworld, -1)
        assert theta.shape == (self.nworld, NUM_PARAMS), theta.shape

        mass_scale = theta[:, PARAM_NAMES.index("cube_mass_scale")]
        friction = theta[:, PARAM_NAMES.index("friction")]

        self.body_mass[:, self.cube_body_id] = (
            self.base_body_mass[self.cube_body_id] * mass_scale
        )
        self.body_inertia[:, self.cube_body_id] = (
            self.base_body_inertia[self.cube_body_id].unsqueeze(0) * mass_scale.unsqueeze(-1)
        )
        # Slide friction only; spin/roll are left at their XML values.
        self.geom_friction[:, self.friction_geom_ids, 0] = friction.unsqueeze(-1)
        self.stiffness.copy_(theta[:, PARAM_NAMES.index("comfree_stiffness")])
        self.damping.copy_(theta[:, PARAM_NAMES.index("comfree_damping")])

        if recompute_const:
            # body_mass feeds body_subtreemass and the invweight0 terms that scale
            # the analytic contact impedance, so they must be refreshed.
            mjwarp.set_const(self.m, self.d)

    def field_view(self, name: str) -> torch.Tensor:
        """A per-world, writable torch view of a model array, materialising it once.

        `put_model` stores model arrays with a leading dimension of 1 and a zero
        stride; giving each world its own value needs a real array.  Values are
        tiled from the host model, never read back from the stride-0 array.
        """
        if not hasattr(self, "_field_views"):
            self._field_views: dict[str, torch.Tensor] = {}
        if name in self._field_views:
            return self._field_views[name]

        if name in ("comfree_stiffness", "comfree_damping"):
            view = wp.to_torch(getattr(self.m, name))
            if view.shape[0] != self.nworld:
                base = float(view[0].item())
                arr = wp.array(
                    np.full(self.nworld, base, dtype=np.float32), dtype=wp.float32
                )
                setattr(self.m, name, arr)
                view = wp.to_torch(arr)
        else:
            host = np.asarray(getattr(self.mjm, name), dtype=np.float32)
            existing = getattr(self.m, name)
            tiled = np.tile(host, (self.nworld,) + (1,) * host.ndim)
            arr = wp.array(tiled, dtype=existing.dtype)
            setattr(self.m, name, arr)
            view = wp.to_torch(arr)
        self._field_views[name] = view
        return view

    def recapture(self) -> None:
        """Rebuild the CUDA graph after model arrays have been replaced.

        `field_view` materialises a per-world array and rebinds it on the Model.
        A graph captured earlier still holds pointers to the arrays that were
        live at capture time, so stepping it would advance the *old* parameters
        while `forward`/`set_const` use the new ones -- a model that contradicts
        itself and blows up within a few steps.
        """
        if self._graph is not None:
            self._capture()

    def host_field(self, name: str) -> np.ndarray:
        if name == "comfree_stiffness":
            return np.array(0.1, dtype=np.float64)
        if name == "comfree_damping":
            return np.array(1e-3, dtype=np.float64)
        return np.asarray(getattr(self.mjm, name), dtype=np.float64)

    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor, forward: bool = True) -> None:
        """Inject state; (nq,)/(nv,) broadcasts to all worlds."""
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.torch_device)
        qvel = torch.as_tensor(qvel, dtype=torch.float32, device=self.torch_device)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).expand(self.nworld, -1)
        if qvel.ndim == 1:
            qvel = qvel.unsqueeze(0).expand(self.nworld, -1)
        self.qpos.copy_(qpos)
        self.qvel.copy_(qvel)
        if forward:
            cfwarp.forward(self.m, self.d)

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qpos.clone(), self.qvel.clone()

    def reset_to_key(self, noise_std: float = 0.0, generator: torch.Generator | None = None) -> None:
        qpos = torch.as_tensor(self.key_qpos, dtype=torch.float32, device=self.torch_device)
        qvel = torch.as_tensor(self.key_qvel, dtype=torch.float32, device=self.torch_device)
        qpos = qpos.unsqueeze(0).expand(self.nworld, -1).contiguous()
        qvel = qvel.unsqueeze(0).expand(self.nworld, -1).contiguous()
        if noise_std > 0:
            qvel = qvel + noise_std * torch.randn(
                qvel.shape, device=qvel.device, generator=generator
            )
        self.set_state(qpos, qvel)
        self.set_ctrl_raw(
            torch.as_tensor(self.key_ctrl, dtype=torch.float32, device=self.torch_device)
            .unsqueeze(0)
            .expand(self.nworld, -1)
        )

    def action_to_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        """Map a normalised action in [-1, 1] to actuator position targets."""
        return self.ctrl_mid + self.ctrl_half * action.clamp(-1.0, 1.0)

    def ctrl_to_action(self, ctrl: torch.Tensor) -> torch.Tensor:
        return ((ctrl - self.ctrl_mid) / self.ctrl_half).clamp(-1.0, 1.0)

    def set_ctrl_raw(self, ctrl: torch.Tensor) -> None:
        self.ctrl.copy_(ctrl)

    def set_action(self, action: torch.Tensor) -> None:
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.nworld, -1)
        self.ctrl.copy_(self.action_to_ctrl(action))

    def step(self) -> None:
        """Advance one *control* step (n_substeps physics steps)."""
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            for _ in range(self.n_substeps):
                cfwarp.step(self.m, self.d)

    def forward(self) -> None:
        cfwarp.forward(self.m, self.d)

    def sync(self) -> None:
        wp.synchronize()

    def to_mjdata(self, mjd: mujoco.MjData, world_id: int = 0) -> mujoco.MjData:
        """Copy one world's state into a host MjData (for rendering)."""
        qpos = self.qpos[world_id].detach().cpu().numpy().astype(np.float64)
        qvel = self.qvel[world_id].detach().cpu().numpy().astype(np.float64)
        mjd.qpos[: self.nq] = qpos
        mjd.qvel[: self.nv] = qvel
        return mjd
