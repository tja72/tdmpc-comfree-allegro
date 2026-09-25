"""
Allegro in-hand cube reorientation from ../comfree_tdmpc, as a TD-MPC2 env.

A thin adapter, not a reimplementation: the dynamics (ComFree-Sim), observation,
reward, goal sequencing and drop detection all come from
`comfree_tdmpc.envs.vec_allegro.VecAllegroReorientEnv`, run with a single lane, in
the same "reality" (perturbed physical parameters) that every comfree_tdmpc ablation
trains and evaluates in (`comfree_tdmpc.reality.apply_reality`). This module only
converts torch-on-cuda to numpy float32 and the vec-env API to the gym 5-tuple.

    python -m tdmpc_square.train task=comfree-allegro-cube model_size=5 actor_mode=sac

Needs `comfree_tdmpc` and `comfree_warp` importable.
Tunables live in `comfree_allegro_assets/comfree_allegro.yaml`, merged with
`cfg.comfree_allegro`, e.g. `+comfree_allegro.reality_perturb=0.0`.

Episode semantics:
  - 70-dim observation, 16-dim action in [-1, 1] mapped to absolute joint targets,
    50 Hz control, 120 steps per episode.
  - terminated = cube dropped (reward includes the -w_drop penalty on that step),
    truncated = time limit. Pearl's trainer treats both alike: no terminal masking in
    the TD target, the same as for HumanoidBench falls.
  - Goals are sequential: a goal counts once it has been held for
    `success_hold_steps`, then a new goal is drawn and the episode continues.
    `info["success"]` is the running goal count `goals_reached`, so pearl's
    last-step `success` metric is goals solved per episode, the number
    comfree_tdmpc logs as `goals_reached` / prints as "solved".
"""

import os

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.reality import apply_reality
from comfree_tdmpc.sim.batch_sim import load_spec
from comfree_tdmpc.sim.param_space import ParamSpace

_DEFAULTS_PATH = os.path.join(
    os.path.dirname(__file__), "comfree_allegro_assets", "comfree_allegro.yaml"
)
TASKS = {"comfree-allegro-cube"}


def _load_defaults(overrides=None):
    defaults = OmegaConf.load(_DEFAULTS_PATH)
    if overrides is None:
        return defaults
    return OmegaConf.merge(defaults, overrides)


class ComFreeAllegroEnv:
    """Single-lane VecAllegroReorientEnv in the shared comfree_tdmpc reality."""

    def __init__(self, cfg=None, seed=0):
        self.cfg = _load_defaults(cfg)
        seed = int(self.cfg.seed if self.cfg.seed is not None else seed)
        self.space = ParamSpace.realistic(load_spec())
        # Constructed with the nominal model, then switched to reality: the same
        # order as comfree_tdmpc's VecTrainer.
        self.env = VecAllegroReorientEnv(
            VecEnvConfig(num_envs=1, episode_length=int(self.cfg.episode_length)),
            str(self.cfg.device),
            seed,
            space=self.space,
            u=self.space.true_u.copy(),
        )
        self.u_reality = apply_reality(
            self.env, self.space, int(self.cfg.reality_seed), float(self.cfg.reality_perturb)
        )
        self.max_episode_steps = int(self.cfg.episode_length)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.env.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.env.action_dim,), dtype=np.float32
        )
        self.action_space.seed(seed)
        self._renderer = None
        self._last_info = None

    @staticmethod
    def _np(x):
        return x[0].detach().cpu().numpy().astype(np.float32)

    def reset(self):
        # The vec env already auto-reset the lane when the previous episode ended;
        # resetting again only draws a fresh goal, which keeps reset() self-contained.
        obs = self.env.reset()
        self._last_info = None
        return self._np(obs), {}

    def step(self, action):
        a = torch.as_tensor(np.asarray(action, dtype=np.float32)).reshape(1, -1)
        # next_obs is the pre-auto-reset observation, i.e. where the episode ended.
        next_obs, reward, _, info = self.env.step(a)
        dropped = bool(info["dropped"][0].item())
        goals = int(info["goals_reached"][0].item())
        out = dict(
            success=float(goals),
            success_subtasks=0.0,
            goals_reached=float(goals),
            dropped=float(dropped),
            angle=float(info["angle"][0].item()),
            success_streak=int(info["success_streak"][0].item()),
        )
        # On the terminal step: comfree_tdmpc's own episode record (episode_reward,
        # episode_length, goals_reached, dropped, mean_angle), used by the trainer
        # to log the same train/eval statistics as comfree_tdmpc.
        finished = self.env.pop_finished()
        if finished:
            out["episode"] = finished[-1]
        self._last_info = out
        return (
            self._np(next_obs),
            float(reward[0].item()),
            dropped,
            bool(info["truncated"][0].item()),
            out,
        )

    def render(self, *args, **kwargs):
        if self._renderer is None:
            from comfree_tdmpc.render import SceneRenderer

            self._renderer = SceneRenderer()
        goals = 0 if self._last_info is None else int(self._last_info["goals_reached"])
        return self._renderer.frame(
            self.env.qpos[0].cpu().numpy(),
            self.env.goal_quat[0].cpu().numpy(),
            label=f"TD-MPC2 baseline   goals solved={goals}",
        )


def make_env(cfg):
    """
    Make the comfree_tdmpc Allegro cube-reorientation env.
    Task string: "comfree-allegro-cube".
    """
    if cfg.task not in TASKS:
        raise ValueError(f"Task {cfg.task} not supported")
    return ComFreeAllegroEnv(cfg=cfg.get("comfree_allegro", None), seed=cfg.get("seed", 0))
