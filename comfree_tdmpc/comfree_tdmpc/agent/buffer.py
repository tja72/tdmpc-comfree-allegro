"""GPU-resident replay buffer.

Besides what TD-MPC needs (obs, action, planner mu/std, reward), the buffer also
keeps the full simulator state (qpos, qvel) of every transition.  SysID resamples
short chunks of those states: because a ComFree simulation's entire state is
(qpos, qvel), a chunk can be replayed inside the simulator *exactly*, with no
warm-start or actuator activation to reconstruct.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        nq: int,
        nv: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.obs = z(capacity, obs_dim)
        self.next_obs = z(capacity, obs_dim)
        self.action = z(capacity, action_dim)
        self.mu = z(capacity, action_dim)
        self.std = z(capacity, action_dim)
        self.reward = z(capacity)
        self.done = z(capacity)  # 1.0 on terminal (dropped), 0.0 on truncation
        self.qpos = z(capacity, nq)
        self.qvel = z(capacity, nv)
        self.ep_id = torch.zeros(capacity, dtype=torch.long, device=device)
        self.idx = 0
        self.full = False
        self._ep = 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        reward: float,
        done: float,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
    ) -> None:
        i = self.idx
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = action
        self.mu[i] = mu
        self.std[i] = std
        self.reward[i] = reward
        self.done[i] = done
        self.qpos[i] = qpos
        self.qvel[i] = qvel
        self.ep_id[i] = self._ep
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Insert many transitions at once (used for planner rollout data).

        Wraps around the ring in at most two contiguous writes; qpos/qvel are
        left untouched because simulated transitions are never used for SysID.
        """
        n = obs.shape[0]
        assert n <= self.capacity
        idx = (self.idx + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.action[idx] = action
        self.mu[idx] = mu
        self.std[idx] = std
        self.reward[idx] = reward
        self.done[idx] = done
        self.ep_id[idx] = self._ep
        self.full = self.full or (self.idx + n) >= self.capacity
        self.idx = int((self.idx + n) % self.capacity)

    def end_episode(self) -> None:
        self._ep += 1

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "action": self.action[idx],
            "mu": self.mu[idx],
            "std": self.std[idx],
            "reward": self.reward[idx],
            "done": self.done[idx],
        }

    def sample_chunks(self, num_chunks: int, length: int) -> dict[str, torch.Tensor] | None:
        """Sample contiguous within-episode chunks for SysID.

        Returns qpos0/qvel0 (C, nq/nv), actions (L, C, A) and the reference
        trajectory qpos/qvel (L, C, ...) that the simulator must reproduce.
        """
        n = len(self)
        if n < length + 2:
            return None
        starts = torch.randint(0, n - length - 1, (num_chunks * 4,), device=self.device)
        # Keep only chunks that stay inside one episode.
        same_ep = self.ep_id[starts] == self.ep_id[starts + length]
        starts = starts[same_ep][:num_chunks]
        if starts.numel() < num_chunks:
            return None

        offs = torch.arange(1, length + 1, device=self.device).view(-1, 1)
        idx = starts.view(1, -1) + offs  # (L, C)
        act_idx = starts.view(1, -1) + torch.arange(0, length, device=self.device).view(-1, 1)
        return {
            "qpos0": self.qpos[starts],
            "qvel0": self.qvel[starts],
            "actions": self.action[act_idx],
            "qpos_ref": self.qpos[idx],
            "qvel_ref": self.qvel[idx],
        }
