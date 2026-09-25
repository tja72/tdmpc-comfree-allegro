#!/usr/bin/env python
"""F7 figures/sim2sim.svg: goals/ep and drop rate in ComFree vs CPU MuJoCo for the sim2sim shortlist.

Data: sim2sim/outputs/<agent>__<run>/default/summary.json (MuJoCo default solver; the ComFree
reference in the same file comes from the same call). Tags follow sim2sim/scripts/evaluate.py's
default_tag; pearl tags are set explicitly (pearl_{mpc,pi}__residual_1m_<ckpt>) (the sim2sim shortlist).
Per row: single run -> bar = mean over episodes, error bar = bootstrap 95% CI from summary.json;
several runs -> bar = mean of the per-run means, error bar = sample std over runs (n in the label).
With no summary.json at all the figure is a PENDING placeholder with the planned rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

import style as S

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--sim_dir", type=Path, default=S.WS / "sim2sim" / "outputs")
ap.add_argument("--out_dir", type=Path, default=None, help="write here instead of figures/ (for tests)")
args = ap.parse_args()
SIM = args.sim_dir


def cf(mode, rels):
    return [f"comfree_{mode}__" + r.replace("/", "_") for r in rels]


E300 = ["ablation_300k/e_bc", "ablation_300k_rep2/e_bc", "ablation_300k_rep3/e_bc"]
E150 = ["ablation/e_bc", "ablation_rep2/e_bc", "ablation_rep3/e_bc"]
D150 = ["ablation/d_termq", "ablation_rep2/d_termq", "ablation_rep3/d_termq"]
DAG30 = ["vec/dagger_b30", "vec/dagger_b30_rep2", "vec/dagger_b30_rep3"]
DAG60 = ["vec/dagger_b60", "vec/dagger_b60_rep2", "vec/dagger_b60_rep3"]
# (row label, role, tags) in shortlist priority order (#1-#5)
ROWS = [
    ("hero E@300k", "planner", cf("mpc", E300[1:2])),
    ("hero E@300k", "policy", cf("pi", E300[1:2])),
    ("E@300k", "planner", cf("mpc", E300)),
    ("E@300k", "policy", cf("pi", E300)),
    ("E@150k", "planner", cf("mpc", E150)),
    ("E@150k", "policy", cf("pi", E150)),
    ("TD-MPC2@150k", "baseline", ["pearl_mpc__residual_1m_150000"]),
    ("TD-MPC2@150k π", "baseline_pi", ["pearl_pi__residual_1m_150000"]),
    ("TD-MPC2@1M", "baseline", ["pearl_mpc__residual_1m_final"]),
    ("TD-MPC2@1M π", "baseline_pi", ["pearl_pi__residual_1m_final"]),
    ("D@150k", "policy", cf("pi", D150)),
    ("DAgger β=0.3", "policy", cf("pi", DAG30)),
    ("DAgger β=0.6", "policy", cf("pi", DAG60)),
]
COLOR = {"planner": S.C["planner"], "policy": S.C["policy"], "baseline": S.C["baseline"],
         "baseline_pi": S.C["baseline"]}


def load(tag):
    p = SIM / tag / "default" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def stat(sums, backend, metric):
    vals = [s[backend][metric] for s in sums]
    if len(vals) == 1:
        v = vals[0]
        return v["mean"], (v["mean"] - v["ci95"][0], v["ci95"][1] - v["mean"])
    m = [v["mean"] for v in vals]
    sd = float(np.std(m, ddof=1))
    return float(np.mean(m)), (sd, sd)


rows = []
for lab, role, tags in ROWS:
    sums = [s for s in (load(t) for t in tags) if s is not None and "comfree" in s]
    if sums:
        rows.append((lab, role, sums))

fig, axes = S.subplots(S.FULL, 3.4, ncols=2)
if not rows:
    labels = [f"{lab}{' π' if role == 'policy' else (' mpc' if role == 'planner' else '')}"
              for lab, role, _ in ROWS if not lab.startswith("TD-MPC2") or role == "baseline"]
    for ax, title in zip(axes, ("goals / episode", "drop rate")):
        y = np.arange(len(labels))
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_ylim(len(labels) - 0.4, -0.6)
        ax.set_xlim(0, 6 if ax is axes[0] else 1)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        S.pending(ax, "sim2sim evaluation\n(ComFree vs MuJoCo)\nnot run yet")
    print("[sim2sim] no summary.json found -> PENDING placeholder")
else:
    y = np.arange(len(rows))
    h = 0.38
    for ax, metric, xl in ((axes[0], "goals_reached", "goals / episode"), (axes[1], "dropped", "drop rate")):
        for i, (lab, role, sums) in enumerate(rows):
            col = COLOR[role]
            for j, backend in enumerate(("comfree", "mujoco")):
                m, err = stat(sums, backend, metric)
                yy = y[i] + (j - 0.5) * (h + 0.03)
                kw = (dict(fill=False, hatch="////", edgecolor=col, lw=1.1) if backend == "comfree"
                      else dict(color=col, edgecolor="none"))
                ax.barh(yy, m, h, **kw)
                ax.errorbar(m, yy, xerr=[[err[0]], [err[1]]], color=S.INK, lw=1, capsize=2)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows) - 0.4, -0.6)
    axes[1].set_xlim(0, 1)
    axes[0].set_yticks(y, [f"{lab}{' π' if role == 'policy' else (' mpc' if role == 'planner' else '')} (n={len(s)})" for lab, role, s in rows])
    axes[1].set_yticks(y, [])
    print("[sim2sim] rows:", ", ".join(f"{lab}/{role} n={len(s)}" for lab, role, s in rows))
h = [Patch(fill=False, hatch="////", edgecolor=S.INK, label="ComFree (training simulator)"),
     Patch(color=S.INK, label="MuJoCo CPU, default solver"),
     Patch(color=S.C["planner"], label="planner"), Patch(color=S.C["policy"], label="policy"),
     Patch(color=S.C["baseline"], label="TD-MPC2")]
fig.legend(handles=h, loc="outside lower center", ncol=5, fontsize=7.5)
S.save(fig, "sim2sim", args.out_dir)
