"""Shared data collection for the SysID experiments."""

from __future__ import annotations

import torch

from comfree_tdmpc.agent.buffer import ReplayBuffer
from comfree_tdmpc.envs.allegro import AllegroReorientEnv


def collect_excitation(
    env: AllegroReorientEnv, buffer: ReplayBuffer, steps: int, seed: int,
    smoothing: float = 0.8, noise: float = 0.35,
) -> None:
    """Excite the hand with smoothed random actions and record every state.

    The actions are temporally smoothed so the cube stays in the hand long
    enough to produce contact-rich data; independent per-step noise mostly
    produces trajectories where the cube has already been flung away.
    """
    g = torch.Generator(device=env.torch_device).manual_seed(seed)
    env.reset()
    action = env.prev_action.clone().squeeze(0)
    for _ in range(steps):
        qpos, qvel = env.state()
        eps = torch.randn(env.action_dim, device=env.torch_device, generator=g)
        action = (smoothing * action + noise * eps).clamp(-1, 1)
        obs = env.observation()
        next_obs, reward, done, info = env.step(action)
        buffer.add(
            obs, next_obs, action, action, torch.full_like(action, 0.1), reward,
            float(info["dropped"]), qpos.squeeze(0), qvel.squeeze(0),
        )
        if done:
            buffer.end_episode()
            env.reset()
            action = env.prev_action.clone().squeeze(0)
