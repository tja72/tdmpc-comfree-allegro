"""The shared 'reality' every run trains and evaluates in.

Reality is the nominal `ParamSpace.realistic` model with every parameter
displaced by up to `reality_perturb` in u-space, drawn from a fixed seed and
rejection-sampled for stability.  It lives here rather than inside the
trainer so that code outside this package (the TD-MPC2 baseline in
`../tdmpc-square-pearl`) builds exactly the same plant.
"""

from __future__ import annotations

import numpy as np
import torch

from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv
from comfree_tdmpc.sim.param_space import ParamSpace


def probe_stable(env: VecAllegroReorientEnv, u: np.ndarray) -> bool:
    """Does this parameter vector simulate quietly from the scene keyframe?"""
    sim = env.sim
    env.writer.write_u(u)
    sim.reset_to_key()
    hold = torch.as_tensor(sim.key_ctrl, dtype=torch.float32, device=sim.torch_device)
    for _ in range(40):
        sim.set_ctrl_raw(hold.unsqueeze(0).expand(sim.nworld, -1))
        sim.step()
    sim.sync()
    finite = bool(torch.isfinite(sim.qpos).all() and torch.isfinite(sim.qvel).all())
    settled = finite and float(sim.qvel.abs().max()) < 50.0
    return bool(settled and float(sim.qpos[0, 18]) > env.cfg.task.drop_height)


def apply_reality(
    env: VecAllegroReorientEnv,
    space: ParamSpace,
    reality_seed: int = 12345,
    reality_perturb: float = 0.25,
) -> np.ndarray:
    """Draw the reality parameters, write them into `env`, and return them.

    The stability probe steps `env.sim` from the keyframe, so the env must be
    reset before it is used.
    """
    rng = np.random.default_rng(reality_seed)
    u = (
        space.sample_stable(rng, reality_perturb, lambda u: probe_stable(env, u))
        if reality_perturb > 0
        else space.true_u.copy()
    )
    env.writer.write_u(u)
    return u
