"""M2: can first-order projected gradient recover the physical parameters?

A ground-truth simulation with parameters theta* is driven by a fixed random
action sequence; the resulting (qpos, qvel, action) trajectory is the only data.
Starting from a deliberately wrong theta_0, we run projected gradient descent
with a batched Armijo line search and check that theta -> theta*.

This is the unit test for the module that replaces world-model learning.
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
from comfree_tdmpc.sim.params import PARAM_NAMES, ParamVector
from comfree_tdmpc.sysid.fd_optimizer import SysIDConfig, SysIDOptimizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600, help="ground-truth data steps")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--num_chunks", type=int, default=8)
    ap.add_argument("--chunk_length", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--theta_init",
        type=float,
        nargs="+",
        default=[2.5, 0.35, 0.35, 0.008],
        help="wrong initial parameters (mass_scale, friction, stiffness, damping)",
    )
    ap.add_argument(
        "--theta_true", type=float, nargs="+", default=[1.0, 1.0, 0.1, 1e-3]
    )
    ap.add_argument("--out", type=str, default="outputs/02_sysid")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    theta_true = ParamVector(args.theta_true)
    theta_init = ParamVector(args.theta_init)
    print(f"true : {theta_true}")
    print(f"init : {theta_init}")

    env = AllegroReorientEnv(EnvConfig(episode_length=200), theta_true, seed=args.seed)
    buffer = ReplayBuffer(
        args.steps + 10, env.obs_dim, env.action_dim, env.sim.nq, env.sim.nv, env.torch_device
    )
    t0 = time.perf_counter()
    collect_excitation(env, buffer, args.steps, args.seed)
    print(f"collected {len(buffer)} ground-truth steps in {time.perf_counter()-t0:.1f}s")
    del env
    torch.cuda.empty_cache()

    cfg = SysIDConfig(chunk_length=args.chunk_length, num_chunks=args.num_chunks)
    opt = SysIDOptimizer(cfg)
    print(f"sysid worlds: {opt.nworld} (grad {opt.num_grad_worlds}, line search {opt.num_ls_worlds})")

    torch.manual_seed(args.seed)

    def sampler():
        return buffer.sample_chunks(cfg.num_chunks, cfg.chunk_length)

    # A fixed held-out chunk set, so the reported loss is comparable across
    # iterations (the training loss uses fresh chunks every step).
    eval_chunks = buffer.sample_chunks(cfg.num_chunks, cfg.chunk_length)

    t0 = time.perf_counter()
    params, history = opt.optimize(
        theta_init.copy(), sampler, num_iters=args.iters, verbose=True, eval_chunks=eval_chunks
    )
    dt = time.perf_counter() - t0
    print(f"\n{args.iters} iterations in {dt:.1f}s ({dt/args.iters*1e3:.0f} ms/iter)")
    print(f"final: {params}")
    print(f"true : {theta_true}")

    err0 = np.abs(theta_init.log - theta_true.log)
    err1 = np.abs(params.log - theta_true.log)
    print("\nper-parameter |log error| (multiplicative):")
    for i, n in enumerate(PARAM_NAMES):
        print(f"  {n:20s} init {err0[i]:.3f} -> final {err1[i]:.3f}   "
              f"({theta_init.value[i]:.4g} -> {params.value[i]:.4g}, true {theta_true.value[i]:.4g})")

    np.save(out / "history.npy", np.array([h["loss"] for h in history]))

    # figure: parameter traces + loss
    fig, axes = new_figure(1, 2, figsize=(11, 3.6))
    it = np.arange(len(history))
    for i, n in enumerate(PARAM_NAMES):
        vals = np.array([h[f"param/{n}"] for h in history])
        line, = axes[0].plot(it, vals / theta_true.value[i], label=n)
        axes[0].axhline(1.0, color=line.get_color(), ls=":", lw=0.8)
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("projected-gradient iteration")
    axes[0].set_ylabel(r"estimate / truth")
    axes[0].set_title("SysID parameter convergence")
    axes[0].legend(fontsize=7)

    axes[1].semilogy(it, [h["val_loss"] for h in history], label="held-out prediction loss")
    axes[1].semilogy(it, [h["loss"] for h in history], lw=0.8, alpha=0.45,
                     label="training loss (fresh chunks)")
    err = np.array([[abs(np.log(h[f"param/{n}"]) - np.log(theta_true.value[i]))
                     for i, n in enumerate(PARAM_NAMES)] for h in history]).mean(1)
    axes[1].semilogy(it, err, color="tab:red", lw=1.5, ls="--",
                     label=r"mean $|\log \hat\theta - \log \theta^*|$")
    axes[1].set_xlabel("projected-gradient iteration")
    axes[1].set_ylabel("loss / parameter error")
    axes[1].set_title("Descent with batched Armijo line search")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sysid_convergence.png", dpi=140)
    print(f"wrote {out}/sysid_convergence.png")


if __name__ == "__main__":
    main()
