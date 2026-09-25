"""Can stochastic projected gradient identify ~100 parameters at once?

The four-parameter demonstration used a coordinate-wise stencil and an Armijo
line search.  Neither survives the jump to the number of parameters a real hand
leaves unknown, so this validates the replacement: random directional
derivatives averaged over the world bank, a decaying step, and a clip to the box.

Ground truth is the compiled model; the optimiser starts from a displaced
coordinate vector and has to walk back.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.buffer import ReplayBuffer
from comfree_tdmpc.data_collection import collect_excitation
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.logger import new_figure
from comfree_tdmpc.sim.batch_sim import load_spec
from comfree_tdmpc.sim.param_space import ParamSpace
from comfree_tdmpc.sysid.stochastic import StochasticSysID, StochasticSysIDConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--directions", type=int, default=24)
    ap.add_argument("--num_chunks", type=int, default=6)
    ap.add_argument("--chunk_length", type=int, default=5)
    ap.add_argument("--perturb", type=float, default=0.25, help="initial error in u units")
    ap.add_argument("--step0", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/05_sysid_highdim")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    space = ParamSpace.realistic(load_spec())
    print(space.describe())
    u_true = space.true_u.copy()
    u0 = space.perturb(rng, args.perturb)
    print(f"initial mean |u - u*| = {space.error(u0).mean():.4f}")

    env = AllegroReorientEnv(EnvConfig(episode_length=200), None, seed=args.seed)
    buffer = ReplayBuffer(args.steps + 10, env.obs_dim, env.action_dim,
                          env.sim.nq, env.sim.nv, env.torch_device)
    t0 = time.perf_counter()
    collect_excitation(env, buffer, args.steps, args.seed)
    print(f"collected {len(buffer)} ground-truth steps in {time.perf_counter()-t0:.1f}s")
    del env
    torch.cuda.empty_cache()

    cfg = StochasticSysIDConfig(
        chunk_length=args.chunk_length, num_chunks=args.num_chunks,
        num_directions=args.directions, step0=args.step0,
    )
    opt = StochasticSysID(cfg, space)
    print(f"world bank: {opt.nworld} = {opt.n_probe} probes x {cfg.num_chunks} chunks")

    def sampler():
        return buffer.sample_chunks(cfg.num_chunks, cfg.chunk_length)

    eval_chunks = sampler()
    # How much of the loss is reducible at all: the ground-truth coordinates are
    # the floor, anything above it is model error the optimiser could still remove.
    ref = float(opt._evaluate(np.repeat(u_true[None], opt.n_probe, 0), eval_chunks)[0])
    start = float(opt._evaluate(np.repeat(u0[None], opt.n_probe, 0), eval_chunks)[0])
    print(f"held-out loss: at truth {ref:.5f}, at start {start:.5f}")
    t0 = time.perf_counter()
    u, hist = opt.optimize(u0.copy(), sampler, num_iters=args.iters, seed=args.seed,
                           verbose=False, eval_chunks=eval_chunks)
    dt = time.perf_counter() - t0
    print(f"{len(hist)} iterations in {dt:.1f}s ({dt/max(len(hist),1)*1e3:.0f} ms/iter)")

    err0, err1 = space.error(u0), space.error(u)
    print(f"\nmean |u - u*|: {err0.mean():.4f} -> {err1.mean():.4f}")
    final = hist[-1]["val_loss"]
    closed = 100 * (start - final) / max(start - ref, 1e-12)
    print(f"held-out loss: {start:.5f} -> {final:.5f}  (truth {ref:.5f}; "
          f"{closed:.0f}% of the reducible gap closed)")
    print("\nper group (mean |u - u*|):")
    for g, idx in sorted(space.groups().items()):
        print(f"  {g:<10} n={len(idx):>3}   {err0[idx].mean():.4f} -> {err1[idx].mean():.4f}")

    np.savez(out / "result.npz", u0=u0, u=u, u_true=u_true,
             val_loss=[h["val_loss"] for h in hist],
             err=[h["param_err_mean"] for h in hist])

    fig, axes = new_figure(1, 3, figsize=(15, 3.7))
    it = np.arange(len(hist))
    axes[0].semilogy(it, [h["val_loss"] for h in hist], label="held-out prediction loss")
    axes[0].axhline(ref, color="k", ls="--", lw=1, label="loss at the true parameters")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Multi-step prediction on held-out chunks")
    axes[1].plot(it, [h["param_err_mean"] for h in hist], color="tab:red")
    axes[1].set_ylabel(r"mean $|u-u^*|$")
    axes[1].set_title(f"Parameter error ({space.n} parameters)")
    groups = space.groups()
    w = 0.35
    xs = np.arange(len(groups))
    axes[2].bar(xs - w / 2, [err0[i].mean() for i in groups.values()], w, label="initial")
    axes[2].bar(xs + w / 2, [err1[i].mean() for i in groups.values()], w, label="identified")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([f"{g}\nn={len(i)}" for g, i in groups.items()], fontsize=7)
    axes[2].set_ylabel(r"mean $|u-u^*|$")
    axes[2].set_title("Which parameter groups are identifiable")
    for ax in (axes[0], axes[1]):
        ax.set_xlabel("iteration")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "highdim_sysid.png", dpi=140)
    print(f"wrote {out}/highdim_sysid.png")


if __name__ == "__main__":
    main()
