"""The 'real world': a single ComFree simulation running the true parameters.

Keeping the ground-truth environment and the planner's model in the same physics
engine makes the reality gap *purely parametric*, which is exactly the gap the
SysID stage is meant to close.  `engine="mujoco"` swaps the ground truth for
stock MuJoCo (LCP contact) to measure how the pipeline holds up when the gap is
structural as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from comfree_tdmpc.envs.task import ReorientTask, TaskConfig, sample_goal
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig, load_spec
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace
from comfree_tdmpc.sim.params import ParamVector


@dataclass
class EnvConfig:
    episode_length: int = 150
    n_substeps: int = 10
    timestep: float | None = None
    reset_qvel_noise: float = 1e-3
    engine: str = "comfree"  # "comfree" | "mujoco"
    task: TaskConfig = field(default_factory=TaskConfig)


class AllegroReorientEnv:
    """Single-environment Allegro cube reorientation."""

    def __init__(
        self,
        cfg: EnvConfig,
        params: ParamVector | None = None,
        device: str = "cuda:0",
        seed: int = 0,
        space: ParamSpace | None = None,
        u: np.ndarray | None = None,
    ):
        self.cfg = cfg
        # params=None leaves the model exactly as compiled, which is what the
        # high-dimensional SysID experiments treat as ground truth.
        self.params = params if params is not None else ParamVector()
        self._explicit_params = params is not None
        self.torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
        self.rng = torch.Generator(device=self.torch_device).manual_seed(seed)

        sim_cfg = SimConfig(
            nworld=1,
            n_substeps=cfg.n_substeps,
            timestep=cfg.timestep,
            capture_graph=(cfg.engine == "comfree"),
        )
        self.sim = BatchSim(sim_cfg)
        self.space, self.writer = space, None
        if space is not None:
            # The environment is "reality": its parameters are what SysID has to
            # discover, and they include quantities the compiled XML sets to zero.
            self.writer = HandleWriter(self.sim, space)
            self.writer.write_u(space.true_u if u is None else u)
        elif params is not None:
            self.sim.set_params(self.params.value)
        self.mjm = self.sim.mjm
        self.control_dt = self.sim.control_dt

        self.use_mujoco = cfg.engine == "mujoco"
        if self.use_mujoco:
            self.mjd = mujoco.MjData(self.mjm)
            self._apply_params_to_mjm()

        self.task = ReorientTask(
            cfg.task,
            self.torch_device,
            self.control_dt,
            cube_home=self.sim.key_qpos[16:19],
        )
        # Goals are expressed relative to the cube's rest orientation on the palm.
        self.ref_quat = torch.as_tensor(
            self.sim.key_qpos[19:23], dtype=torch.float32, device=self.torch_device
        )
        self.obs_dim = self.task.obs_dim
        self.action_dim = self.sim.nu

        self.goal_quat = torch.zeros(1, 4, device=self.torch_device)
        self.prev_action = torch.zeros(1, self.action_dim, device=self.torch_device)
        self.prev_angle = torch.zeros(1, device=self.torch_device)
        self.t = 0
        self.goals_reached = 0
        self.success_streak = 0

    def _apply_params_to_mjm(self) -> None:
        """Mirror the parameter vector onto the host MuJoCo model (mujoco engine)."""
        v = self.params.as_dict()
        cube_b = self.sim.cube_body_id
        base_mass = load_spec().body_mass[cube_b]
        base_inertia = load_spec().body_inertia[cube_b].copy()
        self.mjm.body_mass[cube_b] = base_mass * v["cube_mass_scale"]
        self.mjm.body_inertia[cube_b] = base_inertia * v["cube_mass_scale"]
        for g in self.sim.friction_geom_ids.tolist():
            self.mjm.geom_friction[g, 0] = v["friction"]

    def set_params(self, params: ParamVector) -> None:
        self.params = params
        self.sim.set_params(params.value)
        if self.use_mujoco:
            self._apply_params_to_mjm()

    @property
    def qpos(self) -> torch.Tensor:
        if self.use_mujoco:
            return torch.as_tensor(
                self.mjd.qpos, dtype=torch.float32, device=self.torch_device
            ).unsqueeze(0)
        return self.sim.qpos

    @property
    def qvel(self) -> torch.Tensor:
        if self.use_mujoco:
            return torch.as_tensor(
                self.mjd.qvel, dtype=torch.float32, device=self.torch_device
            ).unsqueeze(0)
        return self.sim.qvel

    def observation(self) -> torch.Tensor:
        """Current observation for the agent (batch dimension removed)."""
        return self._obs().squeeze(0)

    def _obs(self) -> torch.Tensor:
        return self.task.observation(self.qpos, self.qvel, self.goal_quat, self.prev_action)

    def _sample_goal(self) -> torch.Tensor:
        return sample_goal(
            1,
            self.cfg.task.goal_mode,
            self.torch_device,
            generator=self.rng,
            ref_quat=self.ref_quat,
            min_angle=self.cfg.task.min_goal_angle,
        )

    def reset(self) -> torch.Tensor:
        if self.use_mujoco:
            mujoco.mj_resetData(self.mjm, self.mjd)
            self.mjd.qpos[:] = self.sim.key_qpos
            self.mjd.qvel[:] = self.sim.key_qvel
            self.mjd.ctrl[:] = self.sim.key_ctrl
            mujoco.mj_forward(self.mjm, self.mjd)
        else:
            self.sim.reset_to_key(noise_std=self.cfg.reset_qvel_noise, generator=self.rng)

        self.goal_quat = self._sample_goal()
        self.prev_action = self.sim.ctrl_to_action(
            torch.as_tensor(self.sim.key_ctrl, dtype=torch.float32, device=self.torch_device)
        ).unsqueeze(0)
        self.prev_angle = self.task.goal_angle(self.qpos, self.goal_quat)
        self.t = 0
        self.goals_reached = 0
        self.success_streak = 0
        return self._obs().squeeze(0)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict]:
        action = action.to(self.torch_device, torch.float32).reshape(1, self.action_dim)
        action = action.clamp(-1.0, 1.0)

        if self.use_mujoco:
            ctrl = self.sim.action_to_ctrl(action)[0].detach().cpu().numpy().astype(np.float64)
            self.mjd.ctrl[:] = ctrl
            for _ in range(self.cfg.n_substeps):
                mujoco.mj_step(self.mjm, self.mjd)
        else:
            self.sim.set_action(action)
            self.sim.step()

        reward, angle, success, dropped = self.task.reward(
            self.qpos, self.qvel, self.goal_quat, action, self.prev_action, self.prev_angle
        )
        self.t += 1
        self.prev_action = action
        self.prev_angle = angle

        at_goal = bool(success.item())
        self.success_streak = self.success_streak + 1 if at_goal else 0
        if self.success_streak >= self.cfg.task.success_hold_steps:
            self.goals_reached += 1
            self.success_streak = 0
            self.goal_quat = self._sample_goal()
            self.prev_angle = self.task.goal_angle(self.qpos, self.goal_quat)

        done = bool(dropped.item()) or self.t >= self.cfg.episode_length
        cube_pos = self.task.cube_pos(self.qpos)[0]
        info = {
            "angle": float(angle.item()),
            "cube_z": float(cube_pos[2].item()),
            "cube_drift": float((cube_pos - self.task.cube_home).norm().item()),
            "success": float(at_goal),
            "success_streak": self.success_streak,
            "hold_steps": self.cfg.task.success_hold_steps,
            "dropped": float(dropped.item()),
            "goals_reached": self.goals_reached,
            "truncated": self.t >= self.cfg.episode_length and not bool(dropped.item()),
        }
        return self._obs().squeeze(0), float(reward.item()), done, info

    def state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qpos.clone(), self.qvel.clone()
