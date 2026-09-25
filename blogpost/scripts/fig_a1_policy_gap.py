#!/usr/bin/env python
"""A1 figures/policy_gap.svg: planner and policy tail goals/ep for every policy-gap intervention.

Data: blogpost/results.json groups/pooled (tail means per run; dot = mean over reps, bar = sample std,
n per row). Degenerate runs excluded (e_no_q: n=2). Reference: rung D (dotted vertical line).
"""
import numpy as np

import runs as R
import style as S

G, P = R.RESULTS["groups"], R.RESULTS["pooled"]
ROWS = [  # (label, group entry, section)
    ("D (reference)", G["ablation"]["d_termq"], "ladder"),
    ("E  +bc actor", G["ablation"]["e_bc"], "ladder"),
    ("F  +mean target", G["ablation"]["f_meantarget"], "ladder"),
    ("E without Q (pure BC)", G["no_q"]["e_no_q"], "actor loss"),
    ("F without Q (pure BC)", G["no_q"]["f_no_q"], "actor loss"),
    ("net small", G["netsize"]["netsize_small"], "policy net size"),
    ("net default", G["netsize"]["netsize_default"], "policy net size"),
    ("net large", G["netsize"]["netsize_large"], "policy net size"),
    ("DAgger β=0 (bc_mu)", G["dagger"]["dagger_b0"], "DAgger"),
    ("DAgger β=0.3", G["dagger"]["dagger_b30"], "DAgger"),
    ("DAgger β=0.6", G["dagger"]["dagger_b60"], "DAgger"),
    ("D, 128 samples", G["dagger"]["d_termq_s128"], "train at 128 samples"),
    ("DAgger β=0.3, 128 samples", G["dagger"]["dagger_b30_s128"], "train at 128 samples"),
    ("offline phase v1 (pooled)", next(v for k, v in P.items() if k.startswith("offline_v1 threshold 0.1 (")),
     "offline phase"),
]

fig, ax = S.subplots(S.FULL, 5.2)
y = []
yy, prev = 0.0, None
for lab, grp, sec in ROWS:
    if sec != prev:
        yy += 0.9 if prev is not None else 0.4
    y.append(yy)
    yy += 1
    prev = sec
y = np.array(y)
for key, role, dy, mk in (("mpc/goals_reached/tail", "planner", -0.13, "o"),
                          ("pi/goals_reached/tail", "policy", 0.13, "s")):
    ref = R.gstat(ROWS[0][1], key)[0]
    ax.axvline(ref, color=S.C[role], lw=0.9, ls=":", zorder=1)
    for i, (yi, (lab, grp, sec)) in enumerate(zip(y, ROWS)):
        m, s_, n = R.gstat(grp, key)
        ax.errorbar(m, yi + dy, xerr=s_, fmt=mk, color=S.C[role], ms=5.5, lw=1.4, capsize=2.5,
                    markeredgecolor="none", label=f"{role} (MPPI through ComFree)" if role == "planner" and i == 0
                    else ("policy alone" if i == 0 else None))
ax.set_yticks(y, [f"{lab}  (n={R.gstat(g, 'mpc/goals_reached/tail')[2]})" for lab, g, _ in ROWS])
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 6)
ax.set_ylim(yy - 0.3, -0.4)
ax.set_xlabel("tail goals / episode (mean ± std over runs); dotted = rung D")
prev = None
for yi, (_, _, sec) in zip(y, ROWS):
    if sec != prev:
        ax.text(0.05, yi - 0.62, sec, fontsize=7.5, color=S.INK, style="italic", va="center", ha="left")
    prev = sec
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
S.save(fig, "policy_gap")
