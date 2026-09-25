"""Tune the planner on its own, before any learning is involved.

Every training run pays the planner's cost at every environment step, so the
planner's compute/quality trade-off sets the budget for everything else.  This
sweeps the MPPI knobs with no agent attached (no terminal value, no policy
prior) and the true physics, so what it measures is purely the planner.

Each configuration is scored on matched seeds and reported together with its
wall-clock cost per control step, so a setting can be judged on quality *per
second* rather than quality alone.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import ParamVector

BASE = dict(
    horizon=16, iterations=4, num_samples=256, num_elites=32,
    temperature=0.5, noise_beta=2.0, max_std=0.5, min_std=0.05,
)

# One axis at a time around the base configuration.
AXES: dict[str, list] = {
    "horizon": [8, 12, 16, 24],
    "num_samples": [64, 128, 256, 512],
    "iterations": [1, 2, 4, 6],
    "temperature": [0.1, 0.5, 2.0],
    "num_elites": [8, 32, 64],
    "noise_beta": [0.0, 1.0, 2.0, 3.0],
    "max_std": [0.3, 0.5, 0.8],
}


class SimPool:
    """One world bank per sample count, reused across configurations."""

    def __init__(self, n_substeps: int):
        self.n_substeps = n_substeps
        self._sims: dict[int, BatchSim] = {}

    def get(self, n: int) -> BatchSim:
        if n not in self._sims:
            sim = BatchSim(SimConfig(nworld=n, n_substeps=self.n_substeps))
            sim.set_params(ParamVector().value)
            self._sims[n] = sim
        return self._sims[n]


def run_config(pool: SimPool, cfg_dict: dict, steps: int, seeds: list[int]) -> dict:
    env_cfg = EnvConfig(episode_length=steps, task=TaskConfig())
    rets, goals, drops, lens, rates, times = [], [], [], [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        env = AllegroReorientEnv(env_cfg, ParamVector(), seed=seed)
        planner = MPPIPlanner(
            pool.get(cfg_dict["num_samples"]),
            env.task,
            PlannerConfig(num_pi_trajs=0, use_terminal_value=False, **cfg_dict),
        )
        env.reset()
        planner.reset()
        R, prev, t = 0.0, None, 0
        t0 = time.perf_counter()
        while t < steps:
            qpos, qvel = env.state()
            a, _, _ = planner.plan(
                qpos, qvel, env.goal_quat, env.prev_action, t0=(t == 0), eval_mode=True
            )
            if prev is not None:
                rates.append((a - prev).abs().mean().item())
            prev = a.clone()
            _, r, done, info = env.step(a)
            R += r
            t += 1
            if done:
                break
        times.append((time.perf_counter() - t0) / t)
        rets.append(R)
        goals.append(info["goals_reached"])
        drops.append(info["dropped"])
        lens.append(t)
        del env
    return {
        "return": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "goals": float(np.mean(goals)),
        "dropped": float(np.mean(drops)),
        "length": float(np.mean(lens)),
        "action_rate": float(np.mean(rates)),
        "sec_per_step": float(np.mean(times)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--axes", type=str, nargs="+", default=list(AXES))
    ap.add_argument("--out", type=str, default="outputs/03_tune_mppi")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    pool = SimPool(EnvConfig().n_substeps)

    rows: list[dict] = []
    print(f"{'axis':<13} {'value':>8} | {'return':>8} {'+/-':>6} {'goals':>6} {'drop':>5} "
          f"{'len':>5} {'jitter':>7} {'ms/step':>8} {'ret/s':>7}")

    seen: set[tuple] = set()
    for axis in args.axes:
        for value in AXES[axis]:
            cfg_dict = dict(BASE)
            cfg_dict[axis] = value
            if axis == "num_elites" and value > cfg_dict["num_samples"]:
                continue
            key = tuple(sorted(cfg_dict.items()))
            if key in seen:
                continue
            seen.add(key)
            res = run_config(pool, cfg_dict, args.steps, seeds)
            res.update(axis=axis, value=value, **cfg_dict)
            res["return_per_sec"] = res["return"] / max(res["sec_per_step"], 1e-9) / 1000
            rows.append(res)
            print(f"{axis:<13} {str(value):>8} | {res['return']:>8.0f} {res['return_std']:>6.0f} "
                  f"{res['goals']:>6.2f} {res['dropped']:>5.2f} {res['length']:>5.0f} "
                  f"{res['action_rate']:>7.3f} {res['sec_per_step']*1e3:>8.0f} "
                  f"{res['return_per_sec']:>7.1f}", flush=True)

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from comfree_tdmpc.logger import new_figure

    axes_used = [a for a in args.axes if any(r["axis"] == a for r in rows)]
    fig, axs = new_figure(1, len(axes_used), figsize=(3.2 * len(axes_used), 3.2), squeeze=False)
    for ax, axis in zip(axs[0], axes_used):
        sel = [r for r in rows if r["axis"] == axis]
        x = [r["value"] for r in sel]
        ax.errorbar(x, [r["return"] for r in sel], yerr=[r["return_std"] for r in sel],
                    marker="o", capsize=3)
        ax.set_xlabel(axis)
        ax2 = ax.twinx()
        ax2.plot(x, [r["sec_per_step"] * 1e3 for r in sel], color="tab:orange", ls=":", marker="s", ms=3)
        ax2.set_ylabel("ms / control step", color="tab:orange", fontsize=7)
        ax2.tick_params(labelsize=6)
    axs[0][0].set_ylabel("episode return")
    fig.suptitle("MPPI hyperparameters: quality (blue) and cost (orange)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "mppi_sweep.png", dpi=140)
    print(f"\nwrote {out}/sweep.csv and {out}/mppi_sweep.png")

    best = max(rows, key=lambda r: r["return"])
    cheap = max(rows, key=lambda r: r["return_per_sec"])
    print(f"best return : {best['axis']}={best['value']}  R={best['return']:.0f} "
          f"({best['sec_per_step']*1e3:.0f} ms/step)")
    print(f"best per-sec: {cheap['axis']}={cheap['value']}  R={cheap['return']:.0f} "
          f"({cheap['sec_per_step']*1e3:.0f} ms/step)")


if __name__ == "__main__":
    main()
