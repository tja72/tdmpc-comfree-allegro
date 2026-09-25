#!/usr/bin/env python
"""F4 figures/long_runs.svg: 300k-step runs (+ TD-MPC2 baseline slot), goals/ep vs env steps and vs wall-clock.

Data: comfree_tdmpc/outputs/ablation_300k{,_rep2,_rep3}/{e_bc,d_termq}/{eval,train}.csv
(d_termq rep 3 is degenerate and excluded -> n=2); hero = ablation_300k_rep2/e_bc (thin line).
Baseline: pearl run dir(s) via --baseline_run, parsed with aggregate_results' reader; without it
the baseline region is a grey PENDING slot.

  python scripts/fig_f4_long_runs.py                                   # PENDING slot
  python scripts/fig_f4_long_runs.py --baseline_run ../tdmpc-square-pearl/logs/comfree-allegro-cube/0/residual_1m \
      --max_step 1000000
"""
import argparse

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import runs as R
import style as S

ap = R.add_baseline_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
args = ap.parse_args()
XMAX = args.max_step or 1000000
HERO = R.RESULTS["hero"]["hero"]
BREAK = 300000


def xscale(ax):
    """0-300k linear on 70% of the width, 300k-XMAX compressed into the remaining 30%."""
    if XMAX <= BREAK:
        ax.set_xlim(0, XMAX * 1.02)
        return
    k = (0.3 / 0.7) * BREAK / (XMAX - BREAK)

    def fwd(x):
        x = np.asarray(x, float)
        return np.where(x <= BREAK, x, BREAK + (x - BREAK) * k)

    def inv(y):
        y = np.asarray(y, float)
        return np.where(y <= BREAK, y, BREAK + (y - BREAK) / k)

    ax.set_xscale("function", functions=(fwd, inv))
    ax.set_xlim(0, XMAX * 1.005)
    ticks = [0, 100000, 200000, 300000, XMAX]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.axvspan(BREAK, XMAX * 1.005, color=S.GRID, alpha=0.06, lw=0, zorder=0)


CFG = {"e_bc@300k": ("e_bc", "-"), "d_termq@300k": ("d_termq", "--")}

fig, (ax, aw) = S.subplots(S.FULL, 3.3, ncols=2)
for name, (d, ls) in CFG.items():
    rels = R.rep_dirs("ablation_300k", d)
    for mode in ("mpc", "pi"):
        x, G = R.curves(rels, mode)
        col = S.C["planner" if mode == "mpc" else "policy"]
        S.band(ax, x, G, col, None, ls=ls, marker=None, lw=1.8, alpha=0.15)
        # wall-clock: per run (reps ran on different GPUs, so their clocks don't average)
        for rel, g in zip(rels, G):
            aw.plot(R.wall_hours_at(rel, x), g, color=col, ls=ls, lw=1.1, alpha=0.8)
    if d == "e_bc":
        for mode in ("mpc", "pi"):
            x, g = R.curve(HERO, mode)
            ax.plot(x, g, color=S.C["planner" if mode == "mpc" else "policy"], lw=0.7, zorder=4)

# ---- baseline
if args.baseline_run:
    B = [R.baseline_curves(p, XMAX) for p in args.baseline_run]
    for mode, ls in (("mpc", "-"), ("pi", "--")):
        x0 = B[0][mode][0]
        Y = np.stack([np.interp(x0, b[mode][0], b[mode][1], right=np.nan) for b in B])
        S.band(ax, x0, Y, S.C["baseline"], None, ls=ls, lw=1.8, alpha=0.15)
        for b in B:
            if b[mode + "_wall_h"] is not None:
                aw.plot(b[mode + "_wall_h"], b[mode][1], color=S.C["baseline"], ls=ls, lw=1.1)
    for b in B:
        s = b["summary"]
        print(f"[baseline] {s['source']}: reached {s['reached_step']} / {XMAX}, "
              f"mpc tail {s['mpc']['goals_reached']['tail']}, pi tail {s['pi']['goals_reached']['tail']}")
    nb = len(B)
else:
    lo = 0.72 if XMAX > BREAK else 0.35
    S.pending(ax, "TD-MPC2\nbaseline,\n1M-step run\nin progress", xy=((lo + 1) / 2, 0.5),
              rect=(lo, 0.04, 0.99 - lo, 0.9), fontsize=7)
    S.pending(aw, "TD-MPC2 baseline: run in progress", xy=(0.68, 0.45), fontsize=7.5)
    nb = 0

xscale(ax)
for xv, lab in ((150000, "150k"), (300000, "300k")):
    if xv <= XMAX:
        ax.axvline(xv, color=S.INK, lw=0.8, ls=":", zorder=1)
        ax.text(xv, 6.55, lab, ha="center", va="bottom", fontsize=S.FS["annot"], color=S.INK)
S.kfmt(ax)
ax.set_xlabel("env steps")
aw.set_xlabel("wall-clock hours (each run on its own GPU)")
for a in (ax, aw):
    a.set_ylim(-0.1, 6.5)
ax.set_ylabel("goals / episode")
aw.set_xlim(left=0)

h = [Line2D([], [], color=S.C["planner"], lw=2, label="planner (MPPI through ComFree)"),
     Line2D([], [], color=S.C["policy"], lw=2, label="policy alone"),
     Line2D([], [], color=S.INK, lw=1.8, ls="-", label="e_bc@300k (n=3)"),
     Line2D([], [], color=S.INK, lw=1.8, ls="--", label="d_termq@300k (n=2, 1 degenerate run excl.)"),
     Line2D([], [], color=S.INK, lw=0.7, label="hero run (e_bc, rep 2)")]
if nb:
    h.append(Line2D([], [], color=S.C["baseline"], lw=2,
                    label=f"TD-MPC2 ({'single run' if nb == 1 else f'n={nb}'}): planner —, policy - -"))
fig.legend(handles=h, loc="outside lower center", ncol=3, handlelength=2.6, columnspacing=1.2)
S.save(fig, "long_runs", args.out_dir)
