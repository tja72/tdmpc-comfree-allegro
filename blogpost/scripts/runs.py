"""Read per-run curves from comfree_tdmpc/outputs (and pearl logs) for the figure scripts.

Reuses the CSV parser and baseline loader of scripts/aggregate_results.py, so the
figures and results.md read the data the same way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import aggregate_results as AR  # noqa: F401  (re-exported for the fig scripts)
from style import BLOG, OUT

RESULTS = json.loads((BLOG / "results.json").read_text())
DEGENERATE = set(RESULTS["degenerate_listed"])
REPS = ("", "_rep2", "_rep3")


def rep_dirs(top: str, name: str, excl_degenerate=True) -> list[str]:
    """e.g. rep_dirs('ablation', 'd_termq') -> ['ablation/d_termq', 'ablation_rep2/d_termq', ...]"""
    out = [f"{top}{r}/{name}" for r in REPS]
    out = [d for d in out if (OUT / d / "eval.csv").exists()]
    return [d for d in out if not (excl_degenerate and d in DEGENERATE)]


def vec_reps(name: str, excl_degenerate=True) -> list[str]:
    out = [f"vec/{name}{r}" for r in REPS]
    out = [d for d in out if (OUT / d / "eval.csv").exists()]
    return [d for d in out if not (excl_degenerate and d in DEGENERATE)]


def curve(rel: str, mode: str, metric: str = "goals_reached"):
    ev = AR.read_csv(OUT / rel / "eval.csv")
    return AR.mode_series(ev, mode, metric)


def curves(rels: list[str], mode: str, metric: str = "goals_reached"):
    """Stack per-run eval series; runs have the same eval steps (checked), truncated to the shortest."""
    ser = [curve(r, mode, metric) for r in rels]
    T = min(len(s) for s, _ in ser)
    steps = ser[0][0][:T]
    for s, _ in ser:
        if np.any(np.abs(s[:T] - steps) > 300):
            raise ValueError(f"eval steps differ across {rels}")
    return steps, np.stack([v[:T] for _, v in ser])


def wall_hours_at(rel: str, steps):
    """train.csv wall_time interpolated at the given env steps, in hours."""
    tr = AR.read_csv(OUT / rel / "train.csv")
    return np.interp(steps, tr["step"], tr["wall_time"]) / 3600.0


def run_entry(rel: str) -> dict:
    for r in RESULTS["runs"]:
        if r["rel"] == rel:
            return r
    raise KeyError(rel)


def tail(rel: str, mode: str, metric: str = "goals_reached") -> float:
    return run_entry(rel)[mode][metric]["tail"]


def add_baseline_args(p):
    p.add_argument("--baseline_run", action="append", default=[], type=Path,
                   help="pearl TD-MPC2 run dir (eval.csv/train.csv); repeatable. Omit -> PENDING slot.")
    p.add_argument("--max_step", type=int, default=None, choices=(150000, 300000, 1000000),
                   help="x-range / baseline truncation budget")
    p.add_argument("--out_dir", type=Path, default=None, help="write here instead of figures/ (for tests)")
    return p


def baseline_curves(run: Path, max_step: int):
    """pearl eval.csv series truncated at max_step (same parser as aggregate_results.load_baseline)."""
    ev = AR.read_csv(Path(run) / "eval.csv")
    tr = AR.read_csv(Path(run) / "train.csv")
    keep = ev["step"] <= max_step + AR.BASELINE_STEP_SLACK
    ev = {k: v[keep] for k, v in ev.items()}
    out = {"summary": AR.load_baseline(Path(run), max_step)}
    for mode in ("mpc", "pi"):
        s, v = AR.mode_series(ev, mode, "goals_reached")
        out[mode] = (s, v)
        out[mode + "_wall_h"] = (np.interp(s, tr["step"], tr["wall_time"]) / 3600.0 if tr else None)
    return out


def gstat(grp: dict, key: str):
    """(mean, std, n) of a results.json group entry, excluding degenerate runs where that variant exists."""
    a = grp.get("excl_degenerate", grp["all"])
    return a[key]["mean"], a[key]["std"], a["n"]
