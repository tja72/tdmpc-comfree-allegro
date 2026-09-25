"""Replay buffer for `num_envs` lanes that advance in lockstep.

The flat `ReplayBuffer` stores transitions in arrival order, which is what lets
SysID resample a *contiguous* chunk of one trajectory.  With several
environments interleaved, arrival order no longer follows any one trajectory, so
this buffer keeps a separate ring per lane: index `(e, i)` and `(e, i+1)` are
consecutive control steps of environment `e`, exactly as SysID requires, while
the agent samples uniformly over the flattened whole.

All lanes share one write cursor because all lanes step together.
"""

from __future__ import annotations

import torch


class VecReplayBuffer:
    def __init__(
        self,
        capacity_per_env: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        nq: int,
        nv: int,
        device: torch.device,
    ):
        self.capacity = capacity_per_env
        self.num_envs = num_envs
        self.device = device
        E, C = num_envs, capacity_per_env
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.obs = z(E, C, obs_dim)
        self.next_obs = z(E, C, obs_dim)
        self.action = z(E, C, action_dim)
        self.mu = z(E, C, action_dim)
        self.std = z(E, C, action_dim)
        self.reward = z(E, C)
        self.done = z(E, C)  # 1.0 on terminal (dropped), 0.0 on truncation
        self.qpos = z(E, C, nq)
        self.qvel = z(E, C, nv)
        self.ep_id = torch.zeros(E, C, dtype=torch.long, device=device)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        """Number of transitions, counting every lane."""
        return self.num_envs * (self.capacity if self.full else self.idx)

    @property
    def per_lane(self) -> int:
        return self.capacity if self.full else self.idx

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ep_id: torch.Tensor,
    ) -> None:
        """One control step of every lane; all arguments have a leading E."""
        i = self.idx
        self.obs[:, i] = obs
        self.next_obs[:, i] = next_obs
        self.action[:, i] = action
        self.mu[:, i] = mu
        self.std[:, i] = std
        self.reward[:, i] = reward
        self.done[:, i] = done
        self.qpos[:, i] = qpos
        self.qvel[:, i] = qvel
        self.ep_id[:, i] = ep_id
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = self.per_lane
        e = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        i = torch.randint(0, n, (batch_size,), device=self.device)
        return {
            "obs": self.obs[e, i],
            "next_obs": self.next_obs[e, i],
            "action": self.action[e, i],
            "mu": self.mu[e, i],
            "std": self.std[e, i],
            "reward": self.reward[e, i],
            "done": self.done[e, i],
        }

    @torch.no_grad()
    def sample_chunks(self, num_chunks: int, length: int) -> dict[str, torch.Tensor] | None:
        """Contiguous within-episode chunks, for SysID.

        Returns qpos0/qvel0 (C, nq/nv), actions (L, C, A) and the reference
        trajectory (L, C, ...) that a candidate parameter vector must reproduce.
        """
        n = self.per_lane
        if n < length + 2:
            return None
        # Oversample and filter: a chunk is usable only if it stays inside one
        # episode, which also rejects the chunks that straddle the ring's cursor.
        m = num_chunks * 4
        e = torch.randint(0, self.num_envs, (m,), device=self.device)
        s = torch.randint(0, n - length - 1, (m,), device=self.device)
        ok = self.ep_id[e, s] == self.ep_id[e, s + length]
        e, s = e[ok][:num_chunks], s[ok][:num_chunks]
        if s.numel() < num_chunks:
            return None

        offs = torch.arange(1, length + 1, device=self.device).view(-1, 1)
        idx = s.view(1, -1) + offs  # (L, C)
        act_idx = s.view(1, -1) + torch.arange(length, device=self.device).view(-1, 1)
        ee = e.view(1, -1).expand(length, -1)
        return {
            "qpos0": self.qpos[e, s],
            "qvel0": self.qvel[e, s],
            "actions": self.action[ee, act_idx],
            "qpos_ref": self.qpos[ee, idx],
            "qvel_ref": self.qvel[ee, idx],
        }
