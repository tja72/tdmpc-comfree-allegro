#!/usr/bin/env python
"""A2 figures/rollout_budget.svg: planner goals/ep vs plan time for smaller MPPI budgets at evaluation.

Data: comfree_tdmpc/outputs/rollout_budget{,_rep2,_rep3}/results.json (eval-only sweep, 32 episodes per
cell). Rep k compares vec/dagger_b30[_repk] with ablation[_repk]/d_termq; checkpoint labels are the
run-dir names (dagger_b30* -> dagger_b30). Mean ± sample std over n=3 checkpoints per arm; rep 1 faint.
"""
import json
from collections import defaultdict

import numpy as np

import style as S

cells = defaultdict(lambda: defaultdict(list))   # arm -> (iters, samples) -> [(rep, goals, ms)]
for rep, sub in ((1, "rollout_budget"), (2, "rollout_budget_rep2"), (3, "rollout_budget_rep3")):
    for r in json.loads((S.OUT / sub / "results.json").read_text()):
        arm = "dagger_b30" if r["checkpoint"].startswith("dagger") else "d_termq"
        cells[arm][(r["iterations"], r["num_samples"])].append((rep, r["goals_reached"], r["plan_ms_mean"]))

STY = {"d_termq": (S.C["D"], "o", "-", "rung D (d_termq)"),
       "dagger_b30": (S.C["dagger"], "s", "--", "DAgger β=0.3 (dagger_b30)")}
fig, ax = S.subplots(S.FULL, 3.4)
for arm, cs in cells.items():
    col, mk, ls, lab = STY[arm]
    keys = sorted(cs, key=lambda k: -np.mean([m for _, _, m in cs[k]]))
    ms = np.array([np.mean([m for _, _, m in cs[k]]) for k in keys])
    g = np.array([[v for _, v, _ in sorted(cs[k])] for k in keys])   # (cells, reps)
    n = g.shape[1]
    m, s = g.mean(1), g.std(1, ddof=1)
    ax.errorbar(ms, m, s, color=col, marker=mk, ls=ls, lw=2, ms=6, capsize=3, markeredgecolor="none",
                label=f"{lab}, mean ± std, n={n}")
    r1 = np.array([[v for rp, v, _ in cs[k] if rp == 1][0] for k in keys])
    ms1 = np.array([[t for rp, _, t in cs[k] if rp == 1][0] for k in keys])
    ax.plot(ms1, r1, color=col, marker=mk, ls=ls, lw=1, ms=4, alpha=0.65, markerfacecolor="none",
            label=f"{lab.split(' (')[0]}, rep 1 only")
    if arm == "d_termq":
        for k, xx, yy in zip(keys, ms, m):
            ax.text(xx, 1.15, f"{k[0]}×{k[1]}", ha="center", va="bottom", fontsize=7.5, color=S.INK)
            ax.plot([xx, xx], [1.45, 1.7], color=S.INK, lw=0.6)
ax.set_xscale("log")
ax.set_xticks([25, 50, 100, 200], ["25", "50", "100", "200"])
ax.minorticks_off()
ax.set_xlabel("ms per plan call (log scale); labels = MPPI iterations × samples")
ax.set_ylabel("planner goals / episode")
ax.set_ylim(1.0, 6.0)
ax.legend(loc="upper left", ncol=2, fontsize=7.5)
S.save(fig, "rollout_budget")
