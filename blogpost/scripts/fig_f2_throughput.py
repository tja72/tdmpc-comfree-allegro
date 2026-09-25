#!/usr/bin/env python
"""F2 figures/throughput.svg: ComFree-Sim cost per control step and physics steps/s vs number of worlds.

Data: comfree_tdmpc/outputs/06_throughput/throughput.json ("sim" rows). Machine not recorded in the JSON.
"""
import json

import numpy as np

import style as S

d = json.loads((S.OUT / "06_throughput" / "throughput.json").read_text())["sim"]
nw = np.array([r["nworld"] for r in d])
ms = np.array([r["ctrl_step_ms"] for r in d])
pps = np.array([r["phys_steps_per_s"] for r in d])

fig, (a1, a2) = S.subplots(S.FULL, 2.7, ncols=2)
for ax, y, lab in ((a1, ms, "ms per control step (all worlds)"),
                   (a2, pps, "physics steps / s")):
    ax.plot(nw, y, color=S.C["planner"], marker="o", ms=5, markeredgecolor="none")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
    ax.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
    ax.minorticks_off()
    ax.set_xlabel("parallel ComFree worlds")
    ax.set_ylabel(lab)
    ax.axvline(4096, color=S.INK, lw=0.8, ls=":", zorder=1)
a1.set_ylim(0, ms.max() * 1.22)
a2.set_yscale("log")
a2.set_ylim(2e3, 5e6)
i = int(np.where(nw == 4096)[0][0])
a1.annotate(f"planner bank\n16 envs × 256 samples\n{ms[i]:.1f} ms", (4096, ms[i]), xytext=(-14, -4),
            textcoords="offset points", ha="right", va="top", fontsize=S.FS["annot"], color=S.INK)
a2.annotate(f"{pps[0]/1e3:.1f}k", (1, pps[0]), xytext=(4, -10), textcoords="offset points",
            fontsize=S.FS["annot"], color=S.INK)
a1.annotate(f"1 world: {ms[0]:.2f} ms", (1, ms[0]), xytext=(4, 10), textcoords="offset points",
            fontsize=S.FS["annot"], color=S.INK)
a2.annotate(f"{pps[i]/1e6:.2f}M", (4096, pps[i]), xytext=(-6, 6), textcoords="offset points",
            ha="right", fontsize=S.FS["annot"], color=S.INK)
S.save(fig, "throughput")
