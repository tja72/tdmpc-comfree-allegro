"""Is the parameter-to-trajectory map actually smooth enough for first order?

That is the premise of the whole SysID stage, so it is worth measuring rather
than asserting.  Two checks, both of which the world bank answers in a single
batched rollout each:

  1. One-dimensional slices of the multi-step prediction loss around the truth.
     A complementarity-based contact solver produces a piecewise map with kinks
     where contact modes switch; ComFree's analytic penalty law should not.
  2. Stability of the central-difference derivative as the step size sweeps two
     orders of magnitude.  A smooth map gives a plateau; a noisy or kinked one
     gives an estimate that drifts with the step size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.buffer import ReplayBuffer
from comfree_tdmpc.data_collection import collect_excitation
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.logger import new_figure
from comfree_tdmpc.sim.params import NUM_PARAMS, PARAM_NAMES, ParamVector
from comfree_tdmpc.sysid.fd_optimizer import SysIDConfig, SysIDOptimizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--num_chunks", type=int, default=8)
    ap.add_argument("--chunk_length", type=int, default=5)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--span", type=float, default=0.7, help="+/- span in log-space")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/report")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    theta_true = ParamVector()
    env = AllegroReorientEnv(EnvConfig(episode_length=200), theta_true, seed=args.seed)
    buffer = ReplayBuffer(
        args.steps + 10, env.obs_dim, env.action_dim, env.sim.nq, env.sim.nv, env.torch_device
    )
    collect_excitation(env, buffer, args.steps, args.seed)
    del env
    torch.cuda.empty_cache()

    cfg = SysIDConfig(chunk_length=args.chunk_length, num_chunks=args.num_chunks)
    # The world bank has to hold `grid` candidates at once for the slice sweep.
    opt = SysIDOptimizer(cfg)
    opt.nworld_needed = args.grid * args.num_chunks
    if opt.nworld < opt.nworld_needed:
        from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig

        del opt.sim
        torch.cuda.empty_cache()
        opt.sim = BatchSim(SimConfig(nworld=opt.nworld_needed, n_substeps=10))
        opt.nworld = opt.nworld_needed
    print(f"world bank: {opt.nworld}")

    chunks = buffer.sample_chunks(cfg.num_chunks, cfg.chunk_length)
    log_true = theta_true.log
    offsets = np.linspace(-args.span, args.span, args.grid)

    # 1. loss slices
    slices = {}
    for p, name in enumerate(PARAM_NAMES):
        cand = np.tile(log_true, (args.grid, 1))
        cand[:, p] = log_true[p] + offsets
        thetas = torch.as_tensor(np.exp(cand), dtype=torch.float32, device=opt.torch_device)
        slices[name] = opt._evaluate(thetas, chunks).cpu().numpy()
        print(f"slice {name}: min at offset {offsets[int(np.argmin(slices[name]))]:+.3f}")

    # 2. finite-difference step-size stability
    eps_grid = np.geomspace(3e-3, 3e-1, 16)
    derivs = np.zeros((NUM_PARAMS, len(eps_grid)))
    for p in range(NUM_PARAMS):
        cand = []
        for e in eps_grid:
            for s in (+1.0, -1.0):
                q = log_true.copy()
                q[p] += s * e
                cand.append(q)
        thetas = torch.as_tensor(
            np.exp(np.stack(cand)), dtype=torch.float32, device=opt.torch_device
        )
        losses = opt._evaluate(thetas, chunks).cpu().numpy()
        for i, e in enumerate(eps_grid):
            derivs[p, i] = (losses[2 * i] - losses[2 * i + 1]) / (2 * e)

    # figure
    fig, axes = new_figure(1, 2, figsize=(11, 3.6))
    for name, s in slices.items():
        axes[0].plot(offsets, s / s.max(), marker=".", ms=3, label=name)
    axes[0].axvline(0.0, color="k", ls="--", lw=1)
    axes[0].set_xlabel(r"$\log\theta_p - \log\theta_p^*$")
    axes[0].set_ylabel("prediction loss (normalised)")
    axes[0].set_title("Loss slices through the true parameters")
    axes[0].legend(fontsize=7)

    for p, name in enumerate(PARAM_NAMES):
        d = derivs[p]
        scale = np.abs(d).max() + 1e-12
        axes[1].semilogx(eps_grid, d / scale, marker="o", ms=3, label=name)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("central-difference step in log-space")
    axes[1].set_ylabel("derivative estimate (normalised)")
    axes[1].set_title("First-order estimate vs. step size")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig5_smoothness.png", dpi=140)
    print(f"wrote {out}/fig5_smoothness.png")

    np.savez(out / "smoothness.npz", offsets=offsets, eps=eps_grid, derivs=derivs, **slices)


if __name__ == "__main__":
    main()
