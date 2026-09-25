"""A CPU-MuJoCo drop-in for `comfree_tdmpc.sim.batch_sim.BatchSim`.

Implements exactly the part of the BatchSim interface that the task code
(`VecAllegroReorientEnv`, `reality.probe_stable`) touches -- `qpos/qvel/ctrl`
tensors, `set_action`, `step`, `forward`, `set_state`, `reset_to_key`,
`key_*`, `ctrl_to_action`, `control_dt`, `nu/nq/nv/nworld` -- with one
`MjData` per lane stepped by plain `mujoco.mj_step`, i.e. MuJoCo's
soft-constraint contact solver in float64.

The torch tensors live on the *task* device (cuda by default) in float32,
like the ComFree views, so the unchanged task code computes observations,
rewards and goals from float32 state and draws goals/reset noise from the
same cuda RNG stream as the ComFree env.  MuJoCo keeps its own float64 state
between control steps: a lane is only overwritten from the tensors when the
task code actually wrote a new state (reset, restore), detected against a
shadow copy of what was last read back.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig, load_spec

from sim2sim.solver_configs import DEFAULT, SolverConfig


class MujocoBatchSim(BatchSim):
    """`nworld` independent CPU MuJoCo simulations with the BatchSim interface."""

    def __init__(
        self,
        cfg: SimConfig,
        mjm: mujoco.MjModel | None = None,
        solver: SolverConfig = DEFAULT,
    ):
        # BatchSim.__init__ is deliberately not called: it builds the warp model.
        self.cfg = cfg
        self.torch_device = torch.device(cfg.device if cfg.device.startswith("cuda") else "cpu")
        self.solver = solver

        # Same scene, same edits as BatchSim: load_spec(timestep), ccd_iterations.
        self.mjm = mjm if mjm is not None else load_spec(timestep=cfg.timestep)
        self.mjm.opt.ccd_iterations = cfg.ccd_iterations
        solver.apply(self.mjm)
        self.mjd = mujoco.MjData(self.mjm)
        self._load_keyframe()  # BatchSim's own: key_qpos/key_qvel/key_ctrl

        self.nworld = cfg.nworld
        self.nq, self.nv, self.nu = self.mjm.nq, self.mjm.nv, self.mjm.nu
        self.n_substeps = cfg.n_substeps
        self.control_dt = self.mjm.opt.timestep * cfg.n_substeps
        self._resolve_indices()  # BatchSim's own: cube ids, ctrl range

        self.datas = [mujoco.MjData(self.mjm) for _ in range(self.nworld)]
        dev = self.torch_device
        self.qpos = torch.as_tensor(np.tile(self.key_qpos, (self.nworld, 1)), dtype=torch.float32, device=dev)
        self.qvel = torch.as_tensor(np.tile(self.key_qvel, (self.nworld, 1)), dtype=torch.float32, device=dev)
        self.ctrl = torch.as_tensor(np.tile(self.key_ctrl, (self.nworld, 1)), dtype=torch.float32, device=dev)
        # float32 copies of what the MjData lanes held at the last read-back.
        self._shadow_qpos = np.full((self.nworld, self.nq), np.nan, dtype=np.float32)
        self._shadow_qvel = np.full((self.nworld, self.nv), np.nan, dtype=np.float32)
        # Per-lane count of MuJoCo's automatic resets on bad qacc (divergence).
        self.bad_qacc = np.zeros(self.nworld, dtype=np.int64)
        self._graph = None
        self.model_changed()

    def model_changed(self) -> None:
        """Recompute derived constants after writing model fields (mass, armature...)."""
        mujoco.mj_setConst(self.mjm, self.mjd)
        for d in self.datas:
            mujoco.mj_forward(self.mjm, d)

    def set_params(self, *a, **k):  # pragma: no cover - GPU-only API
        raise NotImplementedError("use sim2sim.params_cpu.CpuHandleWriter")

    def field_view(self, name):  # pragma: no cover - GPU-only API
        raise NotImplementedError("MujocoBatchSim has no per-world warp fields")

    def recapture(self) -> None:
        pass

    def _push(self) -> None:
        """Copy tensors -> MjData; state only for lanes the task code rewrote."""
        qpos = self.qpos.detach().cpu().numpy()
        qvel = self.qvel.detach().cpu().numpy()
        ctrl = self.ctrl.detach().cpu().numpy()
        dirty = ~(
            np.all(qpos == self._shadow_qpos, axis=1) & np.all(qvel == self._shadow_qvel, axis=1)
        )
        for i, d in enumerate(self.datas):
            if dirty[i]:
                d.qpos[:] = qpos[i]
                d.qvel[:] = qvel[i]
                d.qacc_warmstart[:] = 0.0
                d.time = 0.0
            d.ctrl[:] = ctrl[i]

    def _pull(self) -> None:
        qpos = np.stack([d.qpos for d in self.datas]).astype(np.float32)
        qvel = np.stack([d.qvel for d in self.datas]).astype(np.float32)
        self._shadow_qpos, self._shadow_qvel = qpos, qvel
        self.qpos.copy_(torch.from_numpy(qpos))
        self.qvel.copy_(torch.from_numpy(qvel))

    def set_state(self, qpos, qvel, forward: bool = True) -> None:
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.torch_device)
        qvel = torch.as_tensor(qvel, dtype=torch.float32, device=self.torch_device)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).expand(self.nworld, -1)
        if qvel.ndim == 1:
            qvel = qvel.unsqueeze(0).expand(self.nworld, -1)
        self.qpos.copy_(qpos)
        self.qvel.copy_(qvel)
        if forward:
            self.forward()

    def step(self) -> None:
        """Advance one control step (n_substeps mj_step calls) in every lane."""
        self._push()
        warn = mujoco.mjtWarning.mjWARN_BADQACC
        for i, d in enumerate(self.datas):
            before = d.warning[warn].number
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.mjm, d)
            self.bad_qacc[i] += d.warning[warn].number - before
        self._pull()

    def forward(self) -> None:
        self._push()
        for d in self.datas:
            mujoco.mj_forward(self.mjm, d)
        self._pull()

    def sync(self) -> None:
        pass

    def to_mjdata(self, mjd: mujoco.MjData, world_id: int = 0) -> mujoco.MjData:
        mjd.qpos[:] = self.datas[world_id].qpos
        mjd.qvel[:] = self.datas[world_id].qvel
        return mjd
