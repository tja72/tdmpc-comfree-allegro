#!/usr/bin/env python
"""F6 figures/planner_vs_policy.svg: planner vs policy tail goals/ep per ladder rung, + e_bc@300k + TD-MPC2 slot.

Data: blogpost/results.json groups (tail = last 3 planner / last 6 policy evals per run; bars = mean,
error bars = sample std over reps, n=3). Drift ratio = policy imitation error on its own states /
on planner states, from each rung's rep-1 diagnosis.json (single run).
Baseline: --baseline_run <pearl dir> (repeatable) and --max_step (default 150000), parsed with
aggregate_results.load_baseline; without it the TD-MPC2 bars are a hatched PENDING slot.
"""
import argparse

import numpy as np
from matplotlib.patches import Patch

import runs as R
import style as S

ap = R.add_baseline_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
args = ap.parse_args()
MS = args.max_step or 150000
G = R.RESULTS["groups"]


gstat = R.gstat
cols = []   # (label, planner (m,s), policy (m,s), n, drift)
for k in "ABCDEF":
    grp = G["ablation"][S.RUNG_DIR[k]]
    pm, ps, n = gstat(grp, "mpc/goals_reached/tail")
    qm, qs, _ = gstat(grp, "pi/goals_reached/tail")
    cols.append((k, (pm, ps), (qm, qs), n, R.run_entry(f"ablation/{S.RUNG_DIR[k]}")["drift_ratio"]))
grp = G["ablation_300k"]["e_bc@300k"]
pm, ps, n = gstat(grp, "mpc/goals_reached/tail")
qm, qs, _ = gstat(grp, "pi/goals_reached/tail")
cols.append(("E\n@300k", (pm, ps), (qm, qs), n, None))

base = None
if args.baseline_run:
    B = [R.AR.load_baseline(p, MS) for p in args.baseline_run]
    pv = [b["mpc"]["goals_reached"]["tail"] for b in B]
    qv = [b["pi"]["goals_reached"]["tail"] for b in B]
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0  # noqa: E731
    base = ((float(np.mean(pv)), sd(pv)), (float(np.mean(qv)), sd(qv)), len(B))
    for b in B:
        print(f"[baseline] {b['source']} @ {MS}: reached {b['reached_step']}, mpc tail "
              f"{b['mpc']['goals_reached']['tail']:.2f}, pi tail {b['pi']['goals_reached']['tail']:.2f}")

fig, ax = S.subplots(S.FULL, 3.3)
w = 0.38
x = np.arange(len(cols) + 1, dtype=float)
x[-2:] += 0.35   # gap between the ladder and the extra columns
x[-1] += 0.15
for i, (lab, p, q, n, drift) in enumerate(cols):
    for j, (v, role) in enumerate(((p, "planner"), (q, "policy"))):
        xx = x[i] + (j - 0.5) * (w + 0.03)
        ax.bar(xx, v[0], w, color=S.C[role], edgecolor="none")
        ax.errorbar(xx, v[0], v[1], color=S.INK, lw=1, capsize=2.5)
    if drift is not None:
        ax.text(x[i] - (w + 0.03) / 2, p[0] + p[1] + 0.12, f"drift {drift:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=S.INK, style="italic")
xb = x[-1]
if base:
    for j, (v, role, hatch) in enumerate(((base[0], "planner", None), (base[1], "policy", None))):
        xx = xb + (j - 0.5) * (w + 0.03)
        ax.bar(xx, v[0], w, color=S.C["baseline"], edgecolor="none", alpha=1.0 if j == 0 else 0.55)
        if base[2] > 1:
            ax.errorbar(xx, v[0], v[1], color=S.INK, lw=1, capsize=2.5)
    blab = f"TD-MPC2\n@{MS // 1000}k" + ("\n(single run)" if base[2] == 1 else f"\n(n={base[2]})")
else:
    for j in (0, 1):
        ax.bar(xb + (j - 0.5) * (w + 0.03), 4.0, w, fill=False, hatch="///", edgecolor=S.PENDING,
               lw=0.8, ls="--")
    ax.text(xb, 4.2, "PENDING\nrun in\nprogress", ha="center", va="bottom", fontsize=7.5,
            color=S.PENDING, style="italic")
    blab = "TD-MPC2\n(baseline)"
ax.set_xticks(x, [c[0] for c in cols] + [blab])
ax.grid(axis="x", visible=False)
ax.set_ylim(0, 7.0)
ax.set_ylabel("tail goals / episode")
ax.axvline((x[5] + x[6]) / 2, color=S.INK, lw=0.6, ls=":")
ax.text(x[0] - 0.45, 6.85, "ablation ladder, 150k steps, n=3 each", fontsize=S.FS["annot"], color=S.INK, va="top")
ax.text(x[6], 6.85, "300k, n=3", fontsize=S.FS["annot"], color=S.INK, va="top", ha="center")
h = [Patch(color=S.C["planner"], label="planner (MPPI through ComFree)"),
     Patch(color=S.C["policy"], label="policy alone")]
if base:
    h += [Patch(color=S.C["baseline"], label="TD-MPC2 planner"),
          Patch(color=S.C["baseline"], alpha=0.55, label="TD-MPC2 policy")]
fig.legend(handles=h, loc="outside lower center", ncol=len(h))
ax.text(x[0] - 0.45, 6.4, "drift = policy imitation error on its own states / on planner states\n(rep 1 of each rung, single run)",
        fontsize=7.5, color=S.INK, va="top", style="italic")
S.save(fig, "planner_vs_policy", args.out_dir)
