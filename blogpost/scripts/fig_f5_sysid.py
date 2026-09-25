#!/usr/bin/env python
"""F5 figures/sysid.svg: SysID error per parameter group, first vs last call, and ComFree-group error vs steps.

Data: comfree_tdmpc/outputs/ablation{,_rep2,_rep3}/{c_sysid,d_termq,e_bc,f_meantarget}/sysid.csv
(err/<group> = mean |u_hat - u*| over that group's parameters, normalised coordinates).
Left: pooled over rungs C-F at 150k, n = 12 runs (3 reps x 4 rungs), mean ± sample std.
Right: comfree group vs env steps, rungs C and D, n=3 each.
"""
import numpy as np
from matplotlib.patches import Patch

import runs as R
import style as S

GROUPS = [("joint", 48), ("actuator", 8), ("contact", 7), ("inertial", 6), ("comfree", 2)]
rels = [r for k in "CDEF" for r in R.rep_dirs("ablation", S.RUNG_DIR[k])]
first = {g: [] for g, _ in GROUPS}
last = {g: [] for g, _ in GROUPS}
for rel in rels:
    sy = R.AR.read_csv(S.OUT / rel / "sysid.csv")
    for g, _ in GROUPS:
        first[g].append(sy[f"err/{g}"][0])
        last[g].append(sy[f"err/{g}"][-1])

fig, (ab, ac) = S.subplots(S.FULL, 3.0, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(GROUPS))
w = 0.36
for i, (g, npar) in enumerate(GROUPS):
    col = S.C[g]
    for j, (vals, kw) in enumerate(((first[g], dict(fill=False, hatch="////", edgecolor=col, lw=1.2)),
                                    (last[g], dict(color=col, edgecolor="none")))):
        m, s = np.mean(vals), np.std(vals, ddof=1)
        xx = x[i] + (j - 0.5) * (w + 0.04)
        ab.bar(xx, m, w, **kw)
        ab.errorbar(xx, m, s, color=S.INK, lw=1, capsize=2.5)
    d = np.mean(last[g]) - np.mean(first[g])
    top = max(np.mean(first[g]) + np.std(first[g], ddof=1), np.mean(last[g]) + np.std(last[g], ddof=1))
    ab.text(x[i], top + 0.008, f"{d:+.3f}", ha="center", va="bottom", fontsize=S.FS["annot"], color=S.INK)
ab.set_xticks(x, [f"{g}\n({n})" for g, n in GROUPS])
ab.grid(axis="x", visible=False)
ab.set_ylabel("mean |û − u*| (normalised)")
ab.set_ylim(0, 0.24)
ab.set_title(f"per group, first → last call (rungs C–F, n={len(rels)})", fontsize=S.FS["small"])
ab.legend(handles=[Patch(fill=False, hatch="////", edgecolor=S.INK, label="first call (5k steps)"),
                   Patch(color=S.INK, label="last call (150k)")], loc="upper left")

for k in "CD":
    rr = R.rep_dirs("ablation", S.RUNG_DIR[k])
    ss = [R.AR.read_csv(S.OUT / r / "sysid.csv") for r in rr]
    T = min(len(s["step"]) for s in ss)
    Y = np.stack([s["err/comfree"][:T] for s in ss])
    S.band(ac, ss[0]["step"][:T], Y, S.C[k], f"{S.RUNG_NAME[k]} (n={len(rr)})",
           ls="--" if k == "C" else "-", marker=None)
ac.set_ylim(0, 0.24)
ac.set_xlim(0, 152000)
S.kfmt(ac)
ac.set_xlabel("env steps")
ac.set_ylabel("ComFree-group error")
ac.set_title("ComFree contact constants over training", fontsize=S.FS["small"])
ac.legend(loc="upper right")
S.save(fig, "sysid")
