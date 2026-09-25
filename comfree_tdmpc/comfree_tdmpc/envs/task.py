"""Allegro in-hand cube reorientation: observation, reward and goal logic.

Everything here is a pure batched torch function of (qpos, qvel, goal, action).
The same code therefore scores the real environment and the thousands of MPPI
rollouts inside the simulator, which is what makes the planner's objective
*exactly* the task objective (no learned reward model required).

MuJoCo quaternion convention is (w, x, y, z).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

HAND_NQ = 16
HAND_NV = 16
CUBE_NQ = 7  # 3 position + 4 quaternion
CUBE_NV = 6  # 3 linear + 3 angular


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_angle(q: torch.Tensor) -> torch.Tensor:
    """Rotation angle in [0, pi] of a (batched) unit quaternion."""
    w = quat_normalize(q)[..., 0].abs().clamp(max=1.0 - 1e-7)
    return 2.0 * torch.acos(w)


def quat_error(q_cube: torch.Tensor, q_goal: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two orientations, in [0, pi]."""
    return quat_angle(quat_mul(q_goal, quat_conj(quat_normalize(q_cube))))


def quat_from_z(angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def sample_goal(
    n: int,
    mode: str,
    device: torch.device,
    generator: torch.Generator | None = None,
    ref_quat: torch.Tensor | None = None,
    min_angle: float = 0.6,
) -> torch.Tensor:
    """Sample a target orientation.

    In `z_axis` mode the goal is the reference orientation yawed by an angle of
    at least `min_angle`, i.e. the task is 'spin the cube about the palm normal
    by this much'.  Goals closer than the success threshold are excluded so that
    every goal requires actual manipulation.
    """
    if mode == "z_axis":
        mag = min_angle + torch.rand(n, device=device, generator=generator) * (
            torch.pi - min_angle
        )
        sign = torch.where(
            torch.rand(n, device=device, generator=generator) < 0.5, -1.0, 1.0
        )
        goal = quat_from_z(mag * sign)
        if ref_quat is not None:
            goal = quat_mul(goal, quat_normalize(ref_quat.reshape(-1, 4)))
        return goal
    if mode == "random":
        u = torch.rand(n, 3, device=device, generator=generator)
        return torch.stack(
            [
                torch.sqrt(1 - u[:, 0]) * torch.sin(2 * torch.pi * u[:, 1]),
                torch.sqrt(1 - u[:, 0]) * torch.cos(2 * torch.pi * u[:, 1]),
                torch.sqrt(u[:, 0]) * torch.sin(2 * torch.pi * u[:, 2]),
                torch.sqrt(u[:, 0]) * torch.cos(2 * torch.pi * u[:, 2]),
            ],
            dim=-1,
        )
    raise ValueError(f"unknown goal mode {mode}")


@dataclass
class TaskConfig:
    goal_mode: str = "z_axis"
    success_angle: float = 0.4  # rad
    # A goal counts as solved only after the cube is held inside the success cone
    # for this many consecutive control steps (0.2 s).  Without the hold the goal
    # is resampled the instant the cone is touched, so the cube never visibly
    # settles on the target -- and the planner is rewarded for a fly-by.
    success_hold_steps: int = 10
    min_goal_angle: float = 0.6  # never sample a goal that is already solved
    drop_height: float = -0.01  # cube centre height below which it has fallen
    max_cube_drift: float = 0.15  # cube displacement from its home on the palm

    w_progress: float = 2.0  # dense: reduction of the goal angle per second
    w_angle: float = 0.5  # residual penalty on the remaining angle
    w_success: float = 8.0  # bonus while inside the success cone
    # Dropping must cost more than the rotation reward it would buy: the planner
    # will happily throw the cube away for a burst of angular progress otherwise.
    w_drop: float = 50.0
    w_pos: float = 80.0  # keep the cube near its home position on the palm
    # The cube legitimately shifts around the palm while being regrasped, so the
    # position penalty only starts outside a deadband; a penalty that bites from
    # zero displacement just makes the planner hold still.
    pos_deadband: float = 0.03
    w_action: float = 0.05  # action magnitude
    w_smooth: float = 3.0  # action rate (large: keeps the sampled plans smooth)
    w_vel: float = 0.002  # hand joint velocity


class ReorientTask:
    """Observation / reward / termination for the Allegro cube task."""

    def __init__(
        self,
        cfg: TaskConfig,
        device: torch.device,
        control_dt: float,
        cube_home: torch.Tensor | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self.control_dt = control_dt
        # Where the cube rests on the palm in the scene keyframe.
        self.cube_home = (
            torch.zeros(3, device=device)
            if cube_home is None
            else torch.as_tensor(cube_home, dtype=torch.float32, device=device)
        )

    @staticmethod
    def cube_pos(qpos: torch.Tensor) -> torch.Tensor:
        return qpos[..., HAND_NQ : HAND_NQ + 3]

    @staticmethod
    def cube_quat(qpos: torch.Tensor) -> torch.Tensor:
        return qpos[..., HAND_NQ + 3 : HAND_NQ + 7]

    @staticmethod
    def hand_qpos(qpos: torch.Tensor) -> torch.Tensor:
        return qpos[..., :HAND_NQ]

    @staticmethod
    def hand_qvel(qvel: torch.Tensor) -> torch.Tensor:
        return qvel[..., :HAND_NV]

    @staticmethod
    def cube_linvel(qvel: torch.Tensor) -> torch.Tensor:
        return qvel[..., HAND_NV : HAND_NV + 3]

    @staticmethod
    def cube_angvel(qvel: torch.Tensor) -> torch.Tensor:
        return qvel[..., HAND_NV + 3 : HAND_NV + 6]

    def goal_angle(self, qpos: torch.Tensor, goal_quat: torch.Tensor) -> torch.Tensor:
        return quat_error(self.cube_quat(qpos), goal_quat)

    def dropped(self, qpos: torch.Tensor) -> torch.Tensor:
        pos = self.cube_pos(qpos)
        too_low = pos[..., 2] < self.cfg.drop_height
        too_far = (pos - self.cube_home).norm(dim=-1) > self.cfg.max_cube_drift
        return too_low | too_far

    def observation(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        goal_quat: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        cube_q = quat_normalize(self.cube_quat(qpos))
        rel = quat_mul(goal_quat, quat_conj(cube_q))
        ang = quat_angle(rel).unsqueeze(-1)
        return torch.cat(
            [
                self.hand_qpos(qpos),
                0.1 * self.hand_qvel(qvel),
                self.cube_pos(qpos) - self.cube_home,
                cube_q,
                self.cube_linvel(qvel),
                0.1 * self.cube_angvel(qvel),
                goal_quat,
                rel,
                ang,
                prev_action,
            ],
            dim=-1,
        )

    @property
    def obs_dim(self) -> int:
        return HAND_NQ + HAND_NV + 3 + 4 + 3 + 3 + 4 + 4 + 1 + 16

    def reward(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        goal_quat: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        prev_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reward, angle, success, dropped).

        The dominant term is *progress*: the per-second reduction of the goal
        angle.  Summed over a planning horizon it telescopes to the total angle
        reduction, so a short-horizon planner still receives a signal about a
        goal that is many horizons away, while the terminal value supplies the
        rest of the long-range guidance.
        """
        cfg = self.cfg
        angle = self.goal_angle(qpos, goal_quat)
        dropped = self.dropped(qpos)
        success = angle < cfg.success_angle

        progress = (prev_angle - angle) / self.control_dt
        pos_err = ((self.cube_pos(qpos) - self.cube_home).norm(dim=-1) - cfg.pos_deadband).clamp(min=0.0)

        r = cfg.w_progress * progress
        r = r - cfg.w_angle * angle
        r = r + cfg.w_success * success.float()
        r = r - cfg.w_pos * pos_err
        r = r - cfg.w_action * action.pow(2).mean(dim=-1)
        r = r - cfg.w_smooth * (action - prev_action).pow(2).mean(dim=-1)
        r = r - cfg.w_vel * self.hand_qvel(qvel).pow(2).mean(dim=-1)
        r = r - cfg.w_drop * dropped.float()
        # A physically unstable parameter setting can drive the state to inf; the
        # reward must stay finite or one such step poisons the replay buffer and
        # every gradient computed from it.
        r = torch.nan_to_num(r, nan=-cfg.w_drop, posinf=0.0, neginf=-cfg.w_drop)
        return r.clamp(-10.0 * cfg.w_drop, 10.0 * cfg.w_drop), angle, success, dropped
