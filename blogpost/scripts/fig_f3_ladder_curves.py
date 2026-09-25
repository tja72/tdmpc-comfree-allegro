#!/usr/bin/env python
"""F3 figures/ladder_curves.svg: planner goals/ep and drop rate vs env steps for ladder rungs A-F.

Data: comfree_tdmpc/outputs/ablation{,_rep2,_rep3}/<rung>/eval.csv (mpc rows every 25k),
and vec/pure_rl_v1{,_rep2,_rep3}/eval.csv (policy rows; no planner). Mean ± sample std, n=3 each.
"""
import numpy as np

import runs as R
import style as S

fig, (ag, ad) = S.subplots(S.FULL, 3.1, ncols=2)
ends = {}
for k in "ABCDEF":
    rels = R.rep_dirs("ablation", S.RUNG_DIR[k])
    kw = dict(marker=S.MARKER[k], ls="--" if k in "ABC" else "-", lw=1.6 if k in "ABC" else 2.0,
              alpha=0.12 if k in "ABC" else 0.18)
    x, G = R.curves(rels, "mpc", "goals_reached")
    ln = S.band(ag, x, G, S.C[k], f"{S.RUNG_NAME[k]} (n={len(rels)})", **kw)
    x, D = R.curves(rels, "mpc", "dropped")
    S.band(ad, x, D, S.C[k], None, **kw)
    ends[k] = (x[-1], np.mean(G[:, -1]))

rels = R.vec_reps("pure_rl_v1")
x, G = R.curves(rels, "pi", "goals_reached")
S.band(ag, x, G, S.C["pure_rl"], f"pure RL, policy only (n={len(rels)})", ls=":", lw=1.8)
x, D = R.curves(rels, "pi", "dropped")
S.band(ad, x, D, S.C["pure_rl"], None, ls=":", lw=1.8)

# direct labels: rung letter at the end of each planner curve (legend carries the full names)
ys = sorted(ends.items(), key=lambda kv: kv[1][1])
last = -1e9
for k, (xe, ye) in ys:
    ye = max(ye, last + 0.36)
    S.direct_label(ag, xe, ye, k, S.C[k], dx=6)
    last = ye
ag.set_xlim(0, 160000)
ad.set_xlim(0, 160000)
ag.set_ylim(-0.2, 6.2)
ag.annotate("pure RL: 0 goals", (100000, 0), xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=S.FS["annot"], color=S.INK)
ad.set_ylim(-0.03, 1.05)
ag.set_ylabel("planner goals / episode")
ad.set_ylabel("planner drop rate")
for ax in (ag, ad):
    ax.set_xlabel("env steps")
    S.kfmt(ax)
ad.annotate("D–F: terminal Q on", (40000, 0.40), ha="left", fontsize=S.FS["annot"], color=S.INK)
fig.legend(loc="outside lower center", ncol=4, handlelength=2.6, columnspacing=1.2)
S.save(fig, "ladder_curves")
