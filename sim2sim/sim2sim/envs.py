"""Build the ComFree and the MuJoCo version of the *same* task env.

Both run the unchanged `comfree_tdmpc.envs.vec_allegro.VecAllegroReorientEnv`
(observation, action clipping, ctrl mapping, reset, goals, reward, success
hold, drop, truncation).  The MuJoCo one only differs in the sim object it
holds: its `__init__` runs with `vec_allegro.BatchSim` swapped for
`MujocoBatchSim` for the duration of that one call (see `_cpu_batchsim`).
"""

from __future__ import annotations

import contextlib
import functools

import numpy as np

import comfree_tdmpc.envs.vec_allegro as vec_allegro
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.reality import apply_reality
from comfree_tdmpc.sim.batch_sim import load_spec
from comfree_tdmpc.sim.param_space import ParamSpace

from sim2sim.mujoco_backend import MujocoBatchSim
from sim2sim.params_cpu import CpuHandleWriter
from sim2sim.solver_configs import DEFAULT, SolverConfig


@contextlib.contextmanager
def _cpu_batchsim(solver: SolverConfig):
    """Scoped swap of the BatchSim symbol `VecAllegroReorientEnv.__init__` resolves."""
    orig = vec_allegro.BatchSim
    vec_allegro.BatchSim = functools.partial(MujocoBatchSim, solver=solver)
    try:
        yield
    finally:
        vec_allegro.BatchSim = orig


class MujocoAllegroEnv(VecAllegroReorientEnv):
    """VecAllegroReorientEnv whose physics is CPU MuJoCo (`mj_step`)."""

    def __init__(
        self,
        cfg: VecEnvConfig,
        device: str = "cuda:0",
        seed: int = 0,
        space: ParamSpace | None = None,
        u: np.ndarray | None = None,
        solver: SolverConfig = DEFAULT,
    ):
        with _cpu_batchsim(solver):
            super().__init__(cfg, device, seed, space=None, u=None)
        assert isinstance(self.sim, MujocoBatchSim)
        self.space = space
        if space is not None:
            self.writer = CpuHandleWriter(self.sim, space)
            self.writer.write_u(space.true_u if u is None else u)


def env_config_from_run(run_cfg: dict | None, num_envs: int) -> VecEnvConfig:
    """VecEnvConfig from a comfree_tdmpc run's config.json['env'] (or defaults)."""
    if run_cfg is None:
        return VecEnvConfig(num_envs=num_envs)
    e = dict(run_cfg["env"])
    e["task"] = TaskConfig(**e["task"])
    e["num_envs"] = num_envs
    return VecEnvConfig(**e)


def make_comfree_env(env_cfg: VecEnvConfig, seed: int, reality_seed: int, reality_perturb: float):
    """The training-time reality env, built exactly like VecTrainer does."""
    space = ParamSpace.realistic(load_spec())
    env = VecAllegroReorientEnv(env_cfg, "cuda:0", seed, space=space, u=space.true_u.copy())
    u = apply_reality(env, space, reality_seed, reality_perturb)
    return env, space, u


def make_mujoco_env(env_cfg: VecEnvConfig, seed: int, space: ParamSpace, u: np.ndarray,
                    solver: SolverConfig = DEFAULT) -> MujocoAllegroEnv:
    return MujocoAllegroEnv(env_cfg, "cuda:0", seed, space=space, u=u, solver=solver)


def keyframe_hold(env: VecAllegroReorientEnv, u: np.ndarray) -> dict:
    """`reality.probe_stable` (40 steps holding key_ctrl), plus the numbers behind it."""
    from comfree_tdmpc.reality import probe_stable

    ok = probe_stable(env, u)  # writes u, resets to key, holds key_ctrl for 40 steps
    sim = env.sim
    qpos = sim.qpos[0].detach().cpu().numpy()
    key = sim.key_qpos
    rec = {
        "stable": bool(ok),
        "cube_z": float(qpos[18]),
        "drop_height": float(env.cfg.task.drop_height),
        "max_abs_qvel": float(sim.qvel.abs().max()),
        "cube_pos_shift": float(np.linalg.norm(qpos[16:19] - key[16:19])),
        "hand_qpos_max_shift": float(np.abs(qpos[:16] - key[:16]).max()),
    }
    if isinstance(sim, MujocoBatchSim):
        rec["bad_qacc_resets"] = int(sim.bad_qacc.sum())
    return rec
