#!/usr/bin/env python
"""F1 figures/architecture.svg: what is learned vs given in the agent (own diagram, no data).

Content follows the method section: reality = ComFree at the perturbed u*;
SysID fits û by batched finite differences; TD learning trains encoder + Q ensemble + policy;
MPPI rolls out 16 envs x 256 samples = 4096 ComFree worlds at û, scored by the analytic reward
plus the gamma^H min-Q terminal value.
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

import style as S

L, G = S.C["learned"], S.C["given"]
fig, ax = S.subplots(S.FULL, 3.7)
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, title, sub, col, fill_alpha=0.13, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=col if col else "none", alpha=fill_alpha if col else 1, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="none", ec=col or S.INK, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=9, color=S.INK, weight="bold")
    ax.text(x + w / 2, y + h - 5.0, sub, ha="center", va="top", fontsize=7.5, color=S.INK, linespacing=1.25)


def arrow(p0, p1, text=None, tpos=None, cs="arc3", ha="center"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=S.INK,
                                 connectionstyle=cs, shrinkA=2, shrinkB=2))
    if text:
        ax.text(*tpos, text, fontsize=7.5, color=S.INK, ha=ha, va="center", style="italic")


# row 1: reality and data
box(2, 31, 25, 13, "“Reality”", "same ComFree simulator at\nperturbed parameters u*\n(71 params, unknown to agent)", G)
box(38, 31, 22, 13, "Replay buffer", "real transitions\n(+ harvested planner elites)", None)
# row 2: learned components
box(24, 15.5, 30, 11.5, "SysID → û", "fit 71 sim parameters on replayed\nchunks, batched finite differences", L)
box(60, 15.5, 30, 11.5, "TD learning", "encoder (featuriser) → Q ensemble,\npolicy π (planner prior + actor)", L)
# row 3: the planner
box(14, 1.5, 62, 10.5, "MPPI planner", "16 envs × 256 samples = 4096 ComFree worlds at û, horizon 8\n"
    "score = Σ γᵗ r(analytic)  +  γᴴ · min Q(s_H, π(s_H))", G)
box(82, 1.5, 16, 10.5, "Task reward", "analytic, same\nfunction as reality", G)

arrow((27.3, 37.5), (37.7, 37.5), "transitions", (32.5, 39.3))
arrow((42, 30.7), (39, 27.3))
arrow((56, 30.7), (70, 27.3))
arrow((39, 15.2), (39, 12.3), "û: written into all worlds", (40.5, 13.8), ha="left")
arrow((75, 15.2), (66, 12.3), "Q, π", (73, 13.6), ha="left")
arrow((81.7, 6.8), (76.3, 6.8))
arrow((13.7, 6.8), (8, 30.7), "action", (4.2, 18), cs="angle,angleA=180,angleB=90,rad=0", ha="center")

ax.legend(handles=[Patch(fc=L, alpha=0.35, ec=L, label="learned"),
                   Patch(fc=G, alpha=0.35, ec=G, label="given / analytic (simulator, reward)"),
                   Patch(fc="none", ec=S.INK, label="data")],
          loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=7.5, ncol=1, handlelength=1.4)
S.save(fig, "architecture")
