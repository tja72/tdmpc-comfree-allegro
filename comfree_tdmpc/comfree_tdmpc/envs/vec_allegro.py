"""`num_envs` copies of the reorientation task, stepped in lockstep.

The single-environment `AllegroReorientEnv` runs a ComFree bank of exactly one
world, which measures at 1.9 ms per control step -- the same 3.3 ms that 256
worlds cost.  The task is deep (a horizon has to be simulated in order) but not
wide, so almost all of the device sits idle.  Running the environments as extra
worlds in one bank turns that idle width into data rate: `num_envs` transitions
per control step for barely more than the price of one.

Environments finish at different times, so this class auto-resets each lane the
step after it terminates and reports the terminal observation to the caller
first -- the transition written into the buffer must end at the state that
actually terminated, not at the state the lane was reset to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from comfree_tdmpc.envs.task import ReorientTask, TaskConfig, sample_goal
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace


@dataclass
class VecEnvConfig:
    num_envs: int = 16
    episode_length: int = 120
    n_substeps: int = 10
    timestep: float | None = None
    reset_qvel_noise: float = 1e-3
    task: TaskConfig = field(default_factory=TaskConfig)


class VecAllegroReorientEnv:
    """Batched 'reality': one ComFree world per environment, shared parameters."""

    def __init__(
        self,
        cfg: VecEnvConfig,
        device: str = "cuda:0",
        seed: int = 0,
        space: ParamSpace | None = None,
        u: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.torch_device = torch.device("cuda" if device.startswith("cuda") else "cpu")
        self.rng = torch.Generator(device=self.torch_device).manual_seed(seed)

        self.sim = BatchSim(
            SimConfig(
                nworld=cfg.num_envs,
                n_substeps=cfg.n_substeps,
                timestep=cfg.timestep,
                device=device,
            )
        )
        self.space, self.writer = space, None
        if space is not None:
            self.writer = HandleWriter(self.sim, space)
            self.writer.write_u(space.true_u if u is None else u)

        self.control_dt = self.sim.control_dt
        self.task = ReorientTask(
            cfg.task, self.torch_device, self.control_dt, cube_home=self.sim.key_qpos[16:19]
        )
        self.ref_quat = torch.as_tensor(
            self.sim.key_qpos[19:23], dtype=torch.float32, device=self.torch_device
        )
        self.obs_dim = self.task.obs_dim
        self.action_dim = self.sim.nu

        E, A, dev = self.num_envs, self.action_dim, self.torch_device
        self.key_qpos_t = torch.as_tensor(self.sim.key_qpos, dtype=torch.float32, device=dev)
        self.key_qvel_t = torch.as_tensor(self.sim.key_qvel, dtype=torch.float32, device=dev)
        self.key_ctrl_t = torch.as_tensor(self.sim.key_ctrl, dtype=torch.float32, device=dev)
        self.key_action = self.sim.ctrl_to_action(self.key_ctrl_t)

        self.goal_quat = torch.zeros(E, 4, device=dev)
        self.prev_action = torch.zeros(E, A, device=dev)
        self.prev_angle = torch.zeros(E, device=dev)
        self.t = torch.zeros(E, dtype=torch.long, device=dev)
        self.goals_reached = torch.zeros(E, dtype=torch.long, device=dev)
        self.success_streak = torch.zeros(E, dtype=torch.long, device=dev)
        # `needs_t0` tells the planner which lanes have no usable warm start.
        self.needs_t0 = torch.ones(E, dtype=torch.bool, device=dev)
        # A monotonically increasing id per lane; SysID uses it to reject chunks
        # that straddle an episode boundary.
        self.ep_id = torch.arange(E, dtype=torch.long, device=dev)
        self._next_ep = E
        # Episode statistics, accumulated per lane and emitted on termination.
        self.ep_return = torch.zeros(E, device=dev)
        self.ep_angle_sum = torch.zeros(E, device=dev)
        self.finished: list[dict] = []

    @property
    def qpos(self) -> torch.Tensor:
        return self.sim.qpos

    @property
    def qvel(self) -> torch.Tensor:
        return self.sim.qvel

    def state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sim.qpos.clone(), self.sim.qvel.clone()

    def observation(self) -> torch.Tensor:
        return self.task.observation(
            self.sim.qpos, self.sim.qvel, self.goal_quat, self.prev_action
        )

    def _sample_goals(self, n: int) -> torch.Tensor:
        return sample_goal(
            n,
            self.cfg.task.goal_mode,
            self.torch_device,
            generator=self.rng,
            ref_quat=self.ref_quat,
            min_angle=self.cfg.task.min_goal_angle,
        )

    def reset_idx(self, idx: torch.Tensor) -> None:
        """Reset the lanes listed in `idx` (a long tensor of world indices)."""
        if idx.numel() == 0:
            return
        n = idx.numel()
        qpos = self.key_qpos_t.unsqueeze(0).expand(n, -1).clone()
        qvel = self.key_qvel_t.unsqueeze(0).expand(n, -1).clone()
        if self.cfg.reset_qvel_noise > 0:
            qvel = qvel + self.cfg.reset_qvel_noise * torch.randn(
                qvel.shape, device=qvel.device, generator=self.rng
            )
        # Written through the zero-copy warp views, then re-derived once for all
        # worlds: `forward` is a single kernel launch, cheaper than masking it.
        self.sim.qpos[idx] = qpos
        self.sim.qvel[idx] = qvel
        self.sim.ctrl[idx] = self.key_ctrl_t.unsqueeze(0).expand(n, -1)
        self.sim.forward()

        self.goal_quat[idx] = self._sample_goals(n)
        self.prev_action[idx] = self.key_action.unsqueeze(0).expand(n, -1)
        self.prev_angle[idx] = self.task.goal_angle(self.sim.qpos[idx], self.goal_quat[idx])
        self.t[idx] = 0
        self.goals_reached[idx] = 0
        self.success_streak[idx] = 0
        self.needs_t0[idx] = True
        self.ep_return[idx] = 0.0
        self.ep_angle_sum[idx] = 0.0
        self.ep_id[idx] = torch.arange(
            self._next_ep, self._next_ep + n, device=self.torch_device
        )
        self._next_ep += n

    def reset(self) -> torch.Tensor:
        self.reset_idx(torch.arange(self.num_envs, device=self.torch_device))
        self.finished = []
        return self.observation()

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance every lane; returns (next_obs, reward, done, info).

        `next_obs` is the observation *before* the auto-reset, so a terminal
        transition stored in the replay buffer ends where the episode really
        ended.  Read `observation()` afterwards for the state the next action
        should be chosen from.
        """
        action = action.to(self.torch_device, torch.float32).reshape(
            self.num_envs, self.action_dim
        ).clamp(-1.0, 1.0)

        self.sim.set_action(action)
        self.sim.step()

        reward, angle, success, dropped = self.task.reward(
            self.sim.qpos, self.sim.qvel, self.goal_quat, action, self.prev_action,
            self.prev_angle,
        )
        self.t += 1
        self.prev_action = action
        self.prev_angle = angle
        self.ep_return += reward
        self.ep_angle_sum += angle

        # A goal counts once the cube has been held inside the cone long enough;
        # the lane then continues with a fresh goal, which is what makes
        # `goals_reached` a rate rather than a binary.
        self.success_streak = torch.where(
            success, self.success_streak + 1, torch.zeros_like(self.success_streak)
        )
        solved = self.success_streak >= self.cfg.task.success_hold_steps
        if bool(solved.any()):
            sidx = solved.nonzero(as_tuple=True)[0]
            self.goals_reached[sidx] += 1
            self.success_streak[sidx] = 0
            self.goal_quat[sidx] = self._sample_goals(sidx.numel())
            self.prev_angle[sidx] = self.task.goal_angle(
                self.sim.qpos[sidx], self.goal_quat[sidx]
            )

        truncated = self.t >= self.cfg.episode_length
        done = dropped | truncated
        next_obs = self.observation()

        # Cloned, not aliased: the auto-reset below zeroes the live counters, and
        # what the caller needs is the episode that just ended.
        info = {
            "angle": angle,
            "success": success.float(),
            "dropped": dropped.float(),
            "truncated": truncated.float(),
            "goals_reached": self.goals_reached.clone(),
            "success_streak": self.success_streak.clone(),
        }

        if bool(done.any()):
            didx = done.nonzero(as_tuple=True)[0]
            length = self.t[didx].float()
            for i in range(didx.numel()):
                self.finished.append({
                    "episode_reward": float(self.ep_return[didx[i]].item()),
                    "episode_length": int(length[i].item()),
                    "goals_reached": int(self.goals_reached[didx[i]].item()),
                    "dropped": float(dropped[didx[i]].item()),
                    "mean_angle": float(
                        (self.ep_angle_sum[didx[i]] / length[i].clamp(min=1)).item()
                    ),
                })
            self.reset_idx(didx)
        self.needs_t0[~done] = False

        return next_obs, reward, done, info

    def snapshot(self) -> dict:
        """Everything needed to put the bank back exactly as it was."""
        return {
            "qpos": self.sim.qpos.clone(),
            "qvel": self.sim.qvel.clone(),
            "ctrl": self.sim.ctrl.clone(),
            "goal_quat": self.goal_quat.clone(),
            "prev_action": self.prev_action.clone(),
            "prev_angle": self.prev_angle.clone(),
            "t": self.t.clone(),
            "goals_reached": self.goals_reached.clone(),
            "success_streak": self.success_streak.clone(),
            "needs_t0": self.needs_t0.clone(),
            "ep_id": self.ep_id.clone(),
            "ep_return": self.ep_return.clone(),
            "ep_angle_sum": self.ep_angle_sum.clone(),
            "next_ep": self._next_ep,
            "finished": list(self.finished),
            "rng": self.rng.get_state(),
        }

    def restore(self, snap: dict) -> None:
        self.sim.qpos.copy_(snap["qpos"])
        self.sim.qvel.copy_(snap["qvel"])
        self.sim.ctrl.copy_(snap["ctrl"])
        self.sim.forward()
        self.goal_quat = snap["goal_quat"].clone()
        self.prev_action = snap["prev_action"].clone()
        self.prev_angle = snap["prev_angle"].clone()
        self.t = snap["t"].clone()
        self.goals_reached = snap["goals_reached"].clone()
        self.success_streak = snap["success_streak"].clone()
        self.needs_t0 = snap["needs_t0"].clone()
        self.ep_id = snap["ep_id"].clone()
        self.ep_return = snap["ep_return"].clone()
        self.ep_angle_sum = snap["ep_angle_sum"].clone()
        self._next_ep = snap["next_ep"]
        self.finished = list(snap["finished"])
        self.rng.set_state(snap["rng"])

    def pop_finished(self) -> list[dict]:
        out, self.finished = self.finished, []
        return out
