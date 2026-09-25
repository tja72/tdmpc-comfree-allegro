#!/usr/bin/env python
"""Aggregate every comfree_tdmpc training run into blogpost/results.{md,json}.

Read-only over ../comfree_tdmpc/outputs (and, for the TD-MPC2 baseline hooks,
../tdmpc-square-pearl/logs). Writes only blogpost/results.md and
blogpost/results.json. Needs numpy only (no pandas in the comfree-tdmpc env).

Conventions (see results.md header for the rendered version):
  * final  = the last eval row that has a value for that mode (pi every 5k
             steps, mpc every 25k steps in comfree_tdmpc).
  * tail   = mean of the last 3 mpc evals / last 6 pi evals, the convention of
             comfree_tdmpc/scripts/report/make_ablation_figures.py
             (fig_final_bars -> outputs/report/figures/summary.json).
  * across reps: mean +- SAMPLE std (ddof=1) of the per-run value, n stated.

Usage (run from blogpost/, NOT from the workspace root):
  ~/miniconda3/envs/comfree-tdmpc/bin/python scripts/aggregate_results.py
  # TD-MPC2 baseline, one or more seeds, one or both budgets:
  ... scripts/aggregate_results.py --baseline_run <pearl run dir> [--baseline_run <dir2> ...] \
        --max_step 150000 [300000] [1000000]
  # parse the baseline and print it, but keep PENDING in results.md/json:
  ... --baseline_print_only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLOG = HERE.parent
WS = BLOG.parent
OUT = WS / "comfree_tdmpc" / "outputs"

PI_TAIL, MPC_TAIL = 6, 3
METRICS = ("goals_reached", "dropped", "return")
# EXPERIMENTS.md section 8: these three dropped every episode from the first eval on.
LISTED_DEGENERATE = {"ablation_300k_rep3/d_termq", "vec/e_no_q_rep3", "vec/offline_v2_baseline_r1"}
SKIP = ("_crashed_partial", "smoke_", "pilot", "offline_v2_logs")
HERO_MAX_DROP = 0.10      # tail mpc drop rate must be <= this
HERO_TIE_BAND = 0.25      # tail mpc goals within this of the best count as tied
BASELINE_STEP_SLACK = 500  # pearl logs evals a few env steps past the nominal step


# ----------------------------------------------------------------------------- io
def read_csv(p: Path) -> dict[str, np.ndarray]:
    if not p.exists():
        return {}
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        out[k] = np.array([_f(r.get(k)) for r in rows], dtype=float)
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def mode_series(ev: dict, mode: str, metric: str):
    """(steps, values) of the eval rows where <mode>/<metric> is present."""
    k = f"{mode}/{metric}"
    if k not in ev:
        return np.array([]), np.array([])
    m = np.isfinite(ev[k])
    return ev["step"][m], ev[k][m]


def final_and_tail(ev: dict, mode: str, win: int) -> dict:
    res = {}
    for met in METRICS:
        s, v = mode_series(ev, mode, met)
        if len(v) == 0:
            res[met] = {"final": None, "tail": None, "tail_within_std": None}
            continue
        res[met] = {
            "final": float(v[-1]),
            "tail": float(v[-win:].mean()),
            "tail_within_std": float(v[-win:].std()),  # population std, as in summary.json
        }
    s, _ = mode_series(ev, mode, "goals_reached")
    res["n_evals"] = int(len(s))
    res["final_step"] = int(s[-1]) if len(s) else None
    res["tail_steps"] = [int(x) for x in s[-win:]] if len(s) else []
    return res


# ------------------------------------------------------------------ run discovery
def classify(rel: str) -> tuple[str, str, str]:
    """rel like 'ablation_rep2/c_sysid' -> (family, config, rep)."""
    top, name = rel.split("/", 1)
    if top.startswith("ablation_300k"):
        rep = top[len("ablation_300k"):].lstrip("_") or "rep1"
        return "ablation_300k", f"{name}@300k", rep
    if top.startswith("ablation"):
        rep = top[len("ablation"):].lstrip("_") or "rep1"
        return "ablation", name, rep
    m = re.match(r"(.*?)(?:_(rep\d|r\d))?$", name)
    cfg, rep = m.group(1), m.group(2) or "rep1"
    if cfg.startswith("offline_v2"):
        fam = "offline_v2"
    elif cfg.startswith("offline_v1"):
        fam = "offline_v1"
    elif cfg.startswith("netsize"):
        fam = "netsize"
    elif cfg.startswith("dagger") or cfg.endswith("_s128"):
        fam = "dagger"
    elif cfg in ("e_no_q", "f_no_q"):
        fam = "no_q"
    elif cfg.startswith("pure_rl"):
        fam = "pure_rl"
    else:
        fam = "vec_other"
    return fam, cfg, rep


def discover() -> list[Path]:
    dirs = []
    for top in sorted(OUT.iterdir()):
        if not top.is_dir() or not (top.name.startswith("ablation") or top.name == "vec"):
            continue
        for d in sorted(top.iterdir()):
            if d.is_dir() and not any(s in d.name for s in SKIP) and (d / "eval.csv").exists():
                dirs.append(d)
    return dirs


def load_run(d: Path) -> dict:
    rel = str(d.relative_to(OUT))
    fam, cfg, rep = classify(rel)
    ev, tr = read_csv(d / "eval.csv"), read_csv(d / "train.csv")
    conf = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    r = {"rel": rel, "source": f"comfree_tdmpc/outputs/{rel}", "family": fam, "config": cfg, "rep": rep,
         "pi": final_and_tail(ev, "pi", PI_TAIL), "mpc": final_and_tail(ev, "mpc", MPC_TAIL),
         "eval_final_step": int(ev["step"][-1])}
    # wall clock: train.csv wall_time (seconds since trainer start) -> else log -> NA
    if tr and "wall_time" in tr:
        r["train_final_step"] = int(tr["step"][-1])
        r["wall_s"] = float(tr["wall_time"][-1])
        r["wall_source"] = "train.csv:wall_time"
        r["env_steps_per_s_last"] = float(tr["env_steps_per_s"][-1]) if "env_steps_per_s" in tr else None
        upd = float(tr["updates"][-1]) if "updates" in tr else math.nan
    else:
        r["train_final_step"], r["wall_s"], r["wall_source"], upd = None, None, "NA", math.nan
        r["env_steps_per_s_last"] = None
    # offline phase detection: updates beyond updates_per_env_step*(steps-seed_steps)
    if conf:
        expected = conf.get("updates_per_env_step", 1.5) * (conf["total_env_steps"] - conf.get("seed_env_steps", 2000))
        r["updates"] = None if math.isnan(upd) else int(upd)
        r["extra_updates"] = None if math.isnan(upd) else int(round(upd - expected))
        r["offline_cfg"] = {k: conf.get(k) for k in ("offline_start_val_loss", "offline_force_step", "offline_mode")
                            if conf.get(k) is not None}
        r["actor_mode"] = conf.get("agent", {}).get("actor_mode")
        r["num_samples"] = conf.get("planner", {}).get("num_samples")
        r["total_env_steps"] = conf.get("total_env_steps")
    # degenerate: every pi AND mpc eval drops every episode
    _, pd_ = mode_series(ev, "pi", "dropped")
    _, md_ = mode_series(ev, "mpc", "dropped")
    r["degenerate_auto"] = bool(len(md_) and np.all(md_ == 1.0) and np.all(pd_ == 1.0))
    r["degenerate_listed"] = rel in LISTED_DEGENERATE
    r["degenerate"] = r["degenerate_auto"] or r["degenerate_listed"]
    # diagnosis
    if (d / "diagnosis.json").exists():
        dg = json.loads((d / "diagnosis.json").read_text())
        a, b = dg.get("imitation_error/policy_states"), dg.get("imitation_error/planner_states")
        r["drift_ratio"] = a / b if a and b else None
    else:
        r["drift_ratio"] = None
    # sysid first vs last
    sy = read_csv(d / "sysid.csv")
    if sy:
        r["sysid"] = {k.replace("err/", ""): [float(sy[k][0]), float(sy[k][-1])]
                      for k in sy if k.startswith("err/") or k == "param_err_mean"}
        r["sysid"]["n_calls"] = int(len(sy["step"]))
    else:
        r["sysid"] = None
    # checkpoint + videos
    r["agent_pt"] = (d / "agent.pt").exists()
    vd = d / "videos"
    r["videos"] = sorted([{"path": str((vd / f).relative_to(WS)), "bytes": (vd / f).stat().st_size,
                           "step": int(re.search(r"s(\d+)", f).group(1)) if re.search(r"s(\d+)", f) else None,
                           "mode": "mpc" if "mpc" in f else ("pi" if "pi" in f else "?")}
                          for f in os.listdir(vd)], key=lambda x: (x["step"] or 0, x["mode"])) if vd.is_dir() else []
    return r


# ----------------------------------------------------------------------- stats
def agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v))}


def fmt_agg(a, nd=2, show_n=False):
    if a is None or a["n"] == 0:
        return "NA"
    if a["n"] == 1:
        return f"{a['mean']:.{nd}f} (single run)"
    s = f"{a['mean']:.{nd}f} ± {a['std']:.{nd}f}"
    return s + (f" (n={a['n']})" if show_n else "")


def f2(x, nd=2):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


AGG_FIELDS = [("mpc", "goals_reached", "final"), ("mpc", "goals_reached", "tail"),
              ("mpc", "dropped", "final"), ("mpc", "dropped", "tail"), ("mpc", "return", "tail"),
              ("pi", "goals_reached", "final"), ("pi", "goals_reached", "tail"),
              ("pi", "dropped", "tail"), ("pi", "return", "tail")]


def aggregate_group(runs):
    out = {"n": len(runs), "sources": [r["source"] for r in runs]}
    for mode, met, kind in AGG_FIELDS:
        out[f"{mode}/{met}/{kind}"] = agg([r[mode][met][kind] for r in runs])
    out["wall_h"] = agg([r["wall_s"] / 3600 if r["wall_s"] else None for r in runs])
    out["final_step"] = sorted({r["eval_final_step"] for r in runs})
    return out


# ------------------------------------------------------------- pearl baseline
def load_baseline(run: Path, max_step: int) -> dict:
    """Pearl TD-MPC2 run: <run>/eval.csv (pi/* mpc/* at every eval) + <run>/train.csv."""
    run = run.resolve()
    ev, tr = read_csv(run / "eval.csv"), read_csv(run / "train.csv")
    if not ev:
        raise SystemExit(f"no eval.csv in {run}")
    lim = max_step + BASELINE_STEP_SLACK
    keep = ev["step"] <= lim
    ev = {k: v[keep] for k, v in ev.items()}
    reached = int(ev["step"][-1]) if len(ev["step"]) else 0
    pi = final_and_tail(ev, "pi", PI_TAIL)
    # pearl evaluates mpc at every eval (~5k); comfree only every 25k. For a
    # matched tail window, subsample pearl's mpc evals to the eval nearest each
    # 25k multiple (within half an eval interval) and take the last 3 of those.
    steps = ev["step"]
    idx = []
    for k in range(1, max_step // 25000 + 1):
        j = int(np.argmin(np.abs(steps - 25000 * k)))
        if abs(steps[j] - 25000 * k) <= 2500:
            idx.append(j)
    sub = {k: v[idx] for k, v in ev.items()}
    mpc = final_and_tail(sub, "mpc", MPC_TAIL)
    mpc_all = final_and_tail(ev, "mpc", MPC_TAIL)
    for met in METRICS:  # final = last eval of any cadence
        mpc[met]["final"] = mpc_all[met]["final"]
    mpc["final_step"] = mpc_all["final_step"]
    mpc["tail_note"] = "last 3 evals on the 25k grid (comfree cadence)"
    wall = None
    if tr and "wall_time" in tr:
        m = tr["step"] <= lim
        wall = float(tr["wall_time"][m][-1]) if m.any() else None
    return {"source": str(run.relative_to(WS)) if run.is_relative_to(WS) else str(run), "max_step": max_step,
            "reached_step": reached, "complete": reached >= max_step - BASELINE_STEP_SLACK,
            "pi": pi, "mpc": mpc, "wall_s": wall, "n_eval_rows": int(len(steps))}


# -------------------------------------------------------------------- markdown
def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def group_rows(groups, order, label_fn=lambda c: f"`{c}`"):
    rows = []
    for cfg in order:
        if cfg not in groups:
            continue
        for variant, g in groups[cfg].items():
            lab = label_fn(cfg) + ("" if variant == "all" else " **excl. degenerate**")
            rows.append([lab, g["n"],
                         fmt_agg(g["mpc/goals_reached/final"]), fmt_agg(g["mpc/goals_reached/tail"]),
                         fmt_agg(g["mpc/dropped/tail"]),
                         fmt_agg(g["pi/goals_reached/final"]), fmt_agg(g["pi/goals_reached/tail"]),
                         fmt_agg(g["pi/dropped/tail"]), fmt_agg(g["wall_h"], 1),
                         ", ".join(f"`{s.replace('comfree_tdmpc/outputs/', '')}`" for s in g["sources"])])
    return rows


GROUP_HEADER = ["config", "n", "mpc goals/ep final", "mpc goals/ep tail", "mpc drop tail",
                "pi goals/ep final", "pi goals/ep tail", "pi drop tail", "wall h", "source (outputs/)"]


def run_rows(runs):
    rows = []
    for r in runs:
        flag = " ⚠degenerate" if r["degenerate"] else ""
        rows.append([f"`{r['rel']}`{flag}", r["eval_final_step"],
                     f2(r["mpc"]["goals_reached"]["final"]), f2(r["mpc"]["goals_reached"]["tail"]),
                     f2(r["mpc"]["dropped"]["final"]), f2(r["mpc"]["dropped"]["tail"]),
                     f2(r["mpc"]["return"]["tail"], 0),
                     f2(r["pi"]["goals_reached"]["final"]), f2(r["pi"]["goals_reached"]["tail"]),
                     f2(r["pi"]["dropped"]["final"]), f2(r["pi"]["dropped"]["tail"]),
                     f2(r["pi"]["return"]["tail"], 0),
                     f2(r["wall_s"] / 3600 if r["wall_s"] else None, 2),
                     r.get("extra_updates", "NA"), f2(r["drift_ratio"])])
    return rows


RUN_HEADER = ["run", "env steps", "mpc G final", "mpc G tail", "mpc drop final", "mpc drop tail",
              "mpc R tail", "pi G final", "pi G tail", "pi drop final", "pi drop tail", "pi R tail",
              "wall h", "extra updates", "drift ratio"]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_run", action="append", default=[], type=Path,
                    help="pearl TD-MPC2 run dir (eval.csv, train.csv); repeat for seeds")
    ap.add_argument("--max_step", nargs="+", type=int, default=[150000, 300000, 1000000], choices=[150000, 300000, 1000000],
                    help="baseline budget(s) to truncate at")
    ap.add_argument("--baseline_print_only", action="store_true",
                    help="parse+print the baseline but keep PENDING rows in results.md/json")
    ap.add_argument("--out_md", type=Path, default=BLOG / "results.md")
    ap.add_argument("--out_json", type=Path, default=BLOG / "results.json")
    args = ap.parse_args()

    runs = [load_run(d) for d in discover()]
    by_rel = {r["rel"]: r for r in runs}

    # -------- group by (family, config)
    groups: dict[str, dict[str, dict[str, dict]]] = {}
    fam_cfgs: dict[str, list] = {}
    for r in runs:
        fam_cfgs.setdefault(r["family"], [])
        if r["config"] not in fam_cfgs[r["family"]]:
            fam_cfgs[r["family"]].append(r["config"])
    for fam, cfgs in fam_cfgs.items():
        groups[fam] = {}
        for cfg in cfgs:
            members = sorted([r for r in runs if r["family"] == fam and r["config"] == cfg], key=lambda r: r["rep"])
            g = {"all": aggregate_group(members)}
            clean = [r for r in members if not r["degenerate"]]
            if len(clean) < len(members):
                g["excl_degenerate"] = aggregate_group(clean)
            groups[fam][cfg] = g
    # pooled groups with byte-identical configs (checked: config.json diff only
    # in fields added later with inert defaults)
    pooled = {
        "rung D config @150k (ablation d_termq x3 + netsize_default x3)":
            [by_rel[k] for k in ("ablation/d_termq", "ablation_rep2/d_termq", "ablation_rep3/d_termq",
                                  "vec/netsize_default", "vec/netsize_default_rep2", "vec/netsize_default_rep3")],
        "offline_v1 threshold 0.1 (offline_v1_01 x3 + offline_v1_02 x3)":
            [by_rel[f"vec/offline_v1_0{i}{s}"] for i in (1, 2) for s in ("", "_rep2", "_rep3")],
        "offline_v1 threshold 0.1, only runs where the offline phase fired":
            [by_rel[f"vec/offline_v1_0{i}{s}"] for i in (1, 2) for s in ("", "_rep2", "_rep3")
             if by_rel[f"vec/offline_v1_0{i}{s}"]["extra_updates"] > 0],
    }
    pooled_agg = {k: {"all": aggregate_group(v)} for k, v in pooled.items()}

    # -------- hero pick
    elig = [r for r in runs if r["agent_pt"] and r["videos"] and not r["degenerate"]
            and r["mpc"]["goals_reached"]["tail"] is not None
            and r["mpc"]["dropped"]["tail"] <= HERO_MAX_DROP and r["mpc"]["dropped"]["final"] <= HERO_MAX_DROP]
    best = max(r["mpc"]["goals_reached"]["tail"] for r in elig)
    band = [r for r in elig if r["mpc"]["goals_reached"]["tail"] >= best - HERO_TIE_BAND]
    band.sort(key=lambda r: (r["pi"]["goals_reached"]["tail"], r["mpc"]["goals_reached"]["tail"]), reverse=True)
    rest = sorted([r for r in elig if r not in band], key=lambda r: r["mpc"]["goals_reached"]["tail"], reverse=True)
    ranking = band + rest
    hero, runner = ranking[0], ranking[1]
    # best 150k-step run too (the main ablation budget)
    elig150 = [r for r in ranking if r["eval_final_step"] == 150000]
    polbest = max([r for r in elig if r["mpc"]["goals_reached"]["tail"] >= 5.0], key=lambda r: r["pi"]["goals_reached"]["tail"])

    # -------- baseline
    baseline = {}
    for ms in args.max_step:
        if args.baseline_run:
            seeds = [load_baseline(p, ms) for p in args.baseline_run]
            baseline[str(ms)] = {"status": "parsed", "seeds": seeds,
                                 "agg": {f"{m}/{met}/{k}": agg([s[m][met][k] for s in seeds])
                                         for m in ("mpc", "pi") for met in METRICS for k in ("final", "tail")}}
            print(f"\n[baseline] max_step={ms}")
            for s in seeds:
                print(f"  {s['source']}: reached {s['reached_step']} (complete={s['complete']}), "
                      f"{s['n_eval_rows']} eval rows kept, wall {s['wall_s']}")
                for m in ("mpc", "pi"):
                    print(f"    {m}: " + ", ".join(
                        f"{met} final={f2(s[m][met]['final'])} tail={f2(s[m][met]['tail'])}" for met in METRICS)
                          + f"  (final eval step {s[m]['final_step']}, tail steps {s[m]['tail_steps']})")
    emit_baseline = bool(args.baseline_run) and not args.baseline_print_only
    if not emit_baseline:
        baseline_out = {str(ms): {"status": "PENDING"} for ms in (150000, 300000, 1000000)}
    else:
        baseline_out = {str(ms): baseline.get(str(ms), {"status": "PENDING"}) for ms in (150000, 300000, 1000000)}

    # -------- rollout budget / benchmarks
    rb = json.loads((OUT / "rollout_budget/results.json").read_text())
    # repetitions 2/3 of the eval-only sweep (outputs/rollout_budget_rep{2,3}/, same script, same seed 77)
    rb_reps = {1: rb}
    for k in (2, 3):
        f = OUT / f"rollout_budget_rep{k}/results.json"
        if f.exists():
            rb_reps[k] = json.loads(f.read_text())
    def _arm(name):  # checkpoint label -> arm (the script labels by run dir name)
        return "dagger_b30" if name.startswith("dagger") else "d_termq"
    rb_pooled = {}
    for k, rows_k in rb_reps.items():
        for x in rows_k:
            rb_pooled.setdefault((_arm(x["checkpoint"]), x["iterations"], x["num_samples"]), []).append(x)
    thr = json.loads((OUT / "06_throughput/throughput.json").read_text())
    ver = json.loads((OUT / "07_verify/vec.json").read_text())
    tune = json.loads((OUT / "08_tune/planner.json").read_text())

    G = groups
    def ga(fam, cfg, key, variant="all"):
        return G[fam][cfg][variant][key]

    # ======================================================== write markdown
    cmd = " ".join(shlex.quote(a) for a in [Path(sys.executable).as_posix(), "scripts/aggregate_results.py", *sys.argv[1:]])
    md = []
    md.append("# Aggregated results (comfree_tdmpc)\n")
    md.append(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by\n\n```bash\ncd blogpost && {cmd}\n```\n")
    md.append(
        "Source of every row: `comfree_tdmpc/outputs/<dir>` (listed per row). Machine-readable copy: "
        "`blogpost/results.json`.\n\n"
        "**Conventions.** *final* = the last evaluation row of that mode (pi = policy alone, evaluated every 5k env "
        "steps; mpc = planner, every 25k). *tail* = mean of the last 6 pi / last 3 mpc evaluations, the convention of "
        "`comfree_tdmpc/scripts/report/make_ablation_figures.py` (→ `outputs/report/figures/summary.json`, which "
        "the README table quotes). Each evaluation is 16 episodes of 120 control steps. For the 60k-step offline_v2 "
        "runs the mpc tail covers all 3 mpc evals. *G* = goals reached per episode, *drop* = fraction of episodes "
        "where the cube fell, *R* = return. Across repetitions: **mean ± sample std (ddof=1)**; n is the number of "
        "runs; `(single run)` where n=1. Repetitions are same-seed reruns on a non-deterministic GPU simulator, not "
        "independent seeds (EXPERIMENTS.md, conventions). *wall h* = `train.csv` `wall_time` of the last row; rep-1 "
        "runs of several vec configs ran on a slower/shared GPU (RTX 3070 vs 4090, EXPERIMENTS.md) so wall-clock is "
        "not comparable across reps. *extra updates* = `updates` − 1.5·(steps − 2000 seed steps): >0 means an "
        "offline phase actually ran (50000 for a full once-phase, 8000 per periodic phase).\n\n"
        "**Degenerate runs** (EXPERIMENTS.md §8; auto-check: every pi and mpc evaluation drops all episodes, "
        f"agrees = {all(r['degenerate_auto'] == r['degenerate_listed'] for r in runs)}): "
        + ", ".join(f"`{x}`" for x in sorted(LISTED_DEGENERATE))
        + ". Groups containing one are reported twice: all runs, and **excl. degenerate**.\n")

    # ---- ablation ladder
    md.append("## 1. Ablation ladder, 150k steps (`ablation/`, `ablation_rep2/`, `ablation_rep3/`)\n")
    md.append(table(GROUP_HEADER, group_rows(G["ablation"], ["a_base", "b_distill", "c_sysid", "d_termq", "e_bc", "f_meantarget"])))
    md.append("\nPooled, identical configuration (config.json diff is empty apart from later-added inert defaults):\n\n")
    md.append(table(GROUP_HEADER, group_rows(pooled_agg, list(pooled_agg), label_fn=lambda c: c)))
    md.append("\n## 2. Rungs D and E at 300k steps (`ablation_300k{,_rep2,_rep3}/`)\n")
    md.append(table(GROUP_HEADER, group_rows(G["ablation_300k"], ["d_termq@300k", "e_bc@300k"])))
    md.append("\n## 3. Policy-net size (`vec/netsize_*`, all with rung-D flags)\n")
    md.append(table(GROUP_HEADER, group_rows(G["netsize"], ["netsize_small", "netsize_default", "netsize_large"])))
    md.append("\n## 4. Actor loss without Q (`vec/e_no_q*`, `vec/f_no_q*`) — compare with rungs E/F in §1\n")
    md.append(table(GROUP_HEADER, group_rows(G["no_q"], ["e_no_q", "f_no_q"])))
    md.append("\n## 5. DAgger and training-time planner budget (`vec/dagger_*`, `vec/d_termq_s128*`)\n")
    md.append(table(GROUP_HEADER, group_rows(G["dagger"], ["dagger_b0", "dagger_b30", "dagger_b60", "dagger_b30_s128", "d_termq_s128"])))
    md.append("\n## 6. Offline consolidation phase v1, 150k steps (`vec/offline_v1*`)\n")
    md.append("`offline_v1` = trigger val_loss<0.15; `offline_v1_01` and `offline_v1_02` = identical configs, trigger 0.1 "
              "(pooled n=6 in §1's pooled table).\n\n")
    md.append(table(GROUP_HEADER, group_rows(G["offline_v1"], ["offline_v1", "offline_v1_01", "offline_v1_02"])))
    ofl = [by_rel[k] for k in by_rel if k.startswith("vec/offline_v")]
    md.append("\nDid the offline phase fire? (`extra updates` from train.csv)\n\n")
    md.append(table(["run", "offline config", "extra updates", "fired"],
                    [[f"`{r['rel']}`", json.dumps(r["offline_cfg"]) if r["offline_cfg"] else "off", r["extra_updates"],
                      "yes" if r["extra_updates"] > 0 else "**no**"] for r in sorted(ofl, key=lambda r: r["rel"])]))
    md.append("\n## 7. Offline phase v2 trigger sweep, 60k steps (`vec/offline_v2_*_r{1,2,3}`)\n")
    md.append(table(GROUP_HEADER, group_rows(G["offline_v2"], ["offline_v2_baseline", "offline_v2_forced_early",
                                                              "offline_v2_forced_late", "offline_v2_periodic"])))
    md.append("\n## 8. Pure RL, no planner (`vec/pure_rl_v1*`)\n")
    md.append(table(GROUP_HEADER, group_rows(G["pure_rl"], ["pure_rl_v1"])))
    md.append("\n(no mpc columns: the planner is never used, so mpc is NA.)\n")

    # ---- TD-MPC2 baseline
    md.append("\n## 9. TD-MPC2 baseline (pearl, `comfree-allegro-cube`)\n")
    if not emit_baseline:
        md.append("**PENDING** — no finished baseline run exists; the partial `residual_1m` run "
                  "(tdmpc-square-pearl/logs/comfree-allegro-cube/0/residual_1m) is not a result (the run was still in progress when this table was generated). Fill with "
                  "`--baseline_run <pearl run dir> [--baseline_run ...] --max_step 150000 300000 1000000`.\n\n")
        md.append(table(["budget", "mpc G final", "mpc G tail", "mpc drop tail", "pi G final", "pi G tail", "n seeds", "source"],
                        [[b, *["PENDING"] * 6, "PENDING"] for b in ("150k (= ladder budget)", "300k (= 300k runs / hero)", "1M (full run)")]))
    else:
        rows = []
        for ms in ("150000", "300000", "1000000"):
            b = baseline_out[ms]
            if b.get("status") != "parsed":
                rows.append([ms, *["PENDING"] * 6, "PENDING"]); continue
            a = b["agg"]
            inc = [s for s in b["seeds"] if not s["complete"]]
            src = ", ".join(f"`{s['source']}`" + ("" if s["complete"] else f" (INCOMPLETE, reached {s['reached_step']})")
                            for s in b["seeds"])
            rows.append([ms, fmt_agg(a["mpc/goals_reached/final"]), fmt_agg(a["mpc/goals_reached/tail"]),
                         fmt_agg(a["mpc/dropped/tail"]), fmt_agg(a["pi/goals_reached/final"]),
                         fmt_agg(a["pi/goals_reached/tail"]), len(b["seeds"]), src])
        md.append(table(["budget", "mpc G final", "mpc G tail", "mpc drop tail", "pi G final", "pi G tail", "n seeds", "source"], rows))
        md.append("\nBaseline mpc tail uses pearl's eval nearest each 25k multiple (comfree cadence); pearl has no "
                  "terminal masking and different hyperparameters (see the baseline caveats in the post).\n")

    # ---- rollout budget
    md.append("\n## 10. Rollout-budget sweep, eval only (`outputs/rollout_budget/results.json`)\n")
    md.append("One checkpoint each (rep 1 of `vec/dagger_b30` and `ablation/d_termq`), 32 episodes per cell, seed 77, "
              "`num_pi_trajs`=24.\n\n")
    cells = sorted({(x["iterations"], x["num_samples"], x["total_budget"]) for x in rb}, key=lambda t: -t[2])
    rows = []
    for it, ns, tb in cells:
        c = {x["checkpoint"]: x for x in rb if x["iterations"] == it and x["num_samples"] == ns}
        rows.append([f"{it}x{ns} ({tb})",
                     f2(c["dagger_b30"]["goals_reached"]), f2(c["dagger_b30"]["dropped"]), f2(c["dagger_b30"]["plan_ms_mean"], 0),
                     f2(c["d_termq"]["goals_reached"]), f2(c["d_termq"]["dropped"]), f2(c["d_termq"]["plan_ms_mean"], 0)])
    md.append(table(["iters x samples (budget)", "dagger_b30 G", "dagger_b30 drop", "dagger_b30 ms/plan",
                     "d_termq G", "d_termq drop", "d_termq ms/plan"], rows))
    if len(rb_reps) > 1:
        reps_used = sorted(rb_reps)
        md.append(f"\nAll repetitions pooled (reps {reps_used}: rep k = `vec/dagger_b30[_repk]` vs "
                  "`ablation[_repk]/d_termq`; reps 2/3 in `outputs/rollout_budget_rep{2,3}/results.json`), "
                  "mean ± sample std over checkpoints, n per cell:\n\n")
        prow = []
        wins = 0; ncell = 0
        for it, ns, tb in cells:
            cd = rb_pooled[("dagger_b30", it, ns)]; ct = rb_pooled[("d_termq", it, ns)]
            gd = np.array([x["goals_reached"] for x in cd]); gt = np.array([x["goals_reached"] for x in ct])
            sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
            prow.append([f"{it}x{ns} ({tb})", f"{gd.mean():.2f} ± {sd(gd):.2f}",
                         f"{np.mean([x['dropped'] for x in cd]):.2f}", f"{np.mean([x['plan_ms_mean'] for x in cd]):.0f}",
                         f"{gt.mean():.2f} ± {sd(gt):.2f}", f"{np.mean([x['dropped'] for x in ct]):.2f}",
                         f"{np.mean([x['plan_ms_mean'] for x in ct]):.0f}", f"{len(gd)}/{len(gt)}",
                         f"{int((gd > gt).sum())}/{min(len(gd), len(gt))}"])
        md.append(table(["iters x samples (budget)", "dagger_b30 G", "drop", "ms/plan", "d_termq G", "drop", "ms/plan",
                         "n (dagger/d_termq)", "reps where dagger > d_termq"], prow))

    # ---- benchmarks
    md.append("\n## 11. Benchmarks (`06_throughput/`, `07_verify/`, `08_tune/`)\n")
    sim512 = next(x for x in thr["sim"] if x["nworld"] == 512)
    simmax = max(thr["sim"], key=lambda x: x["phys_steps_per_s"])
    md.append(f"- `06_throughput/throughput.json`: {sim512['ctrl_steps_per_s']/1e3:.1f}k control steps/s "
              f"({sim512['phys_steps_per_s']/1e6:.2f}M physics steps/s) at 512 worlds; peak "
              f"{simmax['phys_steps_per_s']/1e6:.2f}M physics steps/s at {simmax['nworld']} worlds; 1 world: "
              f"{thr['sim'][0]['ctrl_steps_per_s']:.0f} control steps/s. Plan call: "
              + "; ".join(f"{p['nworld']} worlds {p['plan_ms']:.0f} ms (learned-net overhead {p['learned_overhead_ms']:.1f} ms)"
                          for p in thr["plan"])
              + ". Update: " + "; ".join(f"batch {u['batch_size']} {u['update_ms']:.1f} ms" for u in thr["update"]) + ".\n")
    ok = all(v["ok"] for v in ver["correctness"].values())
    t16 = next(x for x in ver["throughput"] if x["num_envs"] == 16)
    md.append(f"- `07_verify/vec.json`: all {len(ver['correctness'])} correctness checks ok={ok} "
              f"(vec env vs single env after 1 step: {ver['correctness']['vec env reproduces single env']['step1']:.1e}); "
              f"speedup at 16 envs {t16['speedup']:.2f}x ({t16['env_steps_per_s']:.1f} env steps/s), at 32 envs "
              f"{ver['throughput'][-1]['speedup']:.2f}x.\n")
    md.append("- `08_tune/planner.json` (single-env MPPI, no learning — planner alone drops the cube in most episodes):\n\n")
    md.append(table(["config", "H", "iters", "samples", "goals/1k steps", "drop rate", "goals/GPU-min", "episodes"],
                    [[x["config"], x["horizon"], x["iterations"], x["num_samples"], f2(x["goals_per_1k_steps"], 1),
                      f2(x["drop_rate"]), f2(x["goals_per_gpu_min"], 1), x["episodes"]] for x in tune]))

    # ---- sysid
    md.append("\n## 12. SysID error, first vs last SysID call (`<run>/sysid.csv`, mean |u_hat − u*| per group)\n")
    groups_sy = ["joint", "actuator", "contact", "inertial", "comfree", "param_err_mean"]
    sy_runs = [r for r in runs if r["sysid"]]
    md.append("Per configuration, mean ± sample std over reps of first → last:\n\n")
    rows = []
    for fam in G:
        for cfg in G[fam]:
            mem = [r for r in sy_runs if r["family"] == fam and r["config"] == cfg]
            if not mem:
                continue
            cells = []
            for gname in groups_sy:
                a0, a1 = agg([r["sysid"][gname][0] for r in mem]), agg([r["sysid"][gname][1] for r in mem])
                cells.append(f"{a0['mean']:.3f}→{a1['mean']:.3f}" + (f" (±{a1['std']:.3f})" if a1["std"] is not None else ""))
            rows.append([f"`{cfg}`", len(mem), *cells])
    md.append(table(["config", "n", *groups_sy], rows))
    md.append("\nPer run (first → last):\n\n")
    md.append(table(["run", "calls", *groups_sy],
                    [[f"`{r['rel']}`", r["sysid"]["n_calls"],
                      *[f"{r['sysid'][g][0]:.3f}→{r['sysid'][g][1]:.3f}" for g in groups_sy]] for r in sy_runs]))

    # ---- contradiction check
    md.append("\n## 13. Contradiction check: claims in EXPERIMENTS.md / README.md / earlier project notes vs multi-rep data\n")
    claims = contradiction_checks(G, pooled_agg, by_rel, rb, rb_reps, rb_pooled)
    md.append(table(["claim (source)", "claimed", "multi-rep data", "verdict"], claims))

    # ---- hero
    md.append("\n## 14. Hero checkpoint\n")
    md.append(f"**Criterion.** Eligible: `agent.pt` and `videos/` on disk, not degenerate, planner drop rate ≤ {HERO_MAX_DROP} "
              f"both in the final eval and in the tail. Rank by tail mpc goals/ep; every run within {HERO_TIE_BAND} goals/ep "
              "of the best counts as tied (well inside one 16-episode eval's noise), and among those the highest pi "
              f"tail goals/ep wins, so the policy also works. {len(elig)} of {len(runs)} runs are eligible (all runs have "
              "`agent.pt` and videos on disk).\n\n")
    md.append(table(["rank", "run", "env steps", "mpc G tail", "mpc drop tail", "pi G tail", "mpc G final", "pi G final"],
                    [[i + 1, f"`{r['rel']}`" + (" (tie band)" if r in band else ""), r["eval_final_step"],
                      f2(r["mpc"]["goals_reached"]["tail"]), f2(r["mpc"]["dropped"]["tail"]),
                      f2(r["pi"]["goals_reached"]["tail"]), f2(r["mpc"]["goals_reached"]["final"]),
                      f2(r["pi"]["goals_reached"]["final"])] for i, r in enumerate(ranking[:8])]))
    md.append(f"\n**Hero: `{hero['rel']}`**. Runner-up: `{runner['rel']}`. Best at the 150k main budget: "
              f"`{elig150[0]['rel']}`. Note: `{polbest['rel']}` has the best policy among runs with mpc tail ≥ 5 "
              f"(pi tail {polbest['pi']['goals_reached']['tail']:.2f}, mpc tail {polbest['mpc']['goals_reached']['tail']:.2f}) but falls just "
              "outside the tie band; it is the pick if the policy video matters more than the planner video.\n\n")
    for lab, r in (("Hero", hero), ("Runner-up", runner)):
        md.append(f"{lab} videos (GIF, recorded at the eval of that env step; checkpoint `{r['source']}/agent.pt`, "
                  f"{(WS / r['source'] / 'agent.pt').stat().st_size/1e6:.1f} MB = final weights):\n\n")
        md.append(table(["file", "step", "mode", "size MB"],
                        [[f"`{v['path']}`", v["step"], v["mode"], f"{v['bytes']/1e6:.2f}"] for v in r["videos"]]))
        md.append("\n")

    # ---- per-run appendix
    md.append("\n## Appendix A. Every run\n")
    for fam in G:
        md.append(f"\n### {fam}\n\n")
        md.append(table(RUN_HEADER, run_rows(sorted([r for r in runs if r["family"] == fam], key=lambda r: r["rel"]))))

    text = re.sub(r"\n*(^#{1,3} [^\n]*)\n*", r"\n\n\1\n\n", "".join(md), flags=re.M).lstrip()
    args.out_md.write_text(re.sub(r"\n{3,}", "\n\n", text))
    js = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "command": cmd,
          "conventions": {"tail_pi": PI_TAIL, "tail_mpc": MPC_TAIL, "std_across_reps": "sample (ddof=1)",
                          "tail_within_std": "population (ddof=0), as summary.json"},
          "degenerate_listed": sorted(LISTED_DEGENERATE), "runs": runs, "groups": G, "pooled": pooled_agg,
          "hero": {"criterion": {"max_drop": HERO_MAX_DROP, "tie_band": HERO_TIE_BAND},
                   "hero": hero["rel"], "runner_up": runner["rel"], "best_150k": elig150[0]["rel"],
                   "ranking": [r["rel"] for r in ranking]},
          "baseline": baseline_out, "rollout_budget": rb, "rollout_budget_reps": {str(k): v for k, v in rb_reps.items()},
          "benchmarks": {"throughput": thr, "verify": ver, "tune": tune},
          "contradictions": [dict(zip(["claim", "claimed", "data", "verdict"], c)) for c in claims]}
    args.out_json.write_text(json.dumps(js, indent=1, default=str))
    print(f"wrote {args.out_md} and {args.out_json}: {len(runs)} runs")
    print(f"hero={hero['rel']} runner_up={runner['rel']} best150k={elig150[0]['rel']}")


# ------------------------------------------------------------ claims checking
def contradiction_checks(G, P, R, rb, rb_reps, rb_pooled):
    def g(fam, cfg, key, v="all"):
        return G[fam][cfg][v][key]
    def a(fam, cfg, key, v="all", nd=2):
        return fmt_agg(g(fam, cfg, key, v), nd, show_n=True)
    def fin(rel, mode="mpc", met="goals_reached", kind="final"):
        return R[rel][mode][met][kind]
    def m(fam, cfg, key, v="all"):
        return g(fam, cfg, key, v)["mean"]
    rows = []
    L = "ablation"
    # --- terminal value
    cD, cC = m(L, "d_termq", "mpc/goals_reached/tail"), m(L, "c_sysid", "mpc/goals_reached/tail")
    rows.append(["Terminal value is the largest effect: C→D drop 0.88→0.02, goals/ep more than doubled (README; project notes: 1.62→4.56)",
                 "tail, rep 1", f"mpc G tail C {a(L,'c_sysid','mpc/goals_reached/tail')} → D {a(L,'d_termq','mpc/goals_reached/tail')}; "
                 f"mpc drop tail C {a(L,'c_sysid','mpc/dropped/tail')} → D {a(L,'d_termq','mpc/dropped/tail')}; "
                 f"A→D {m(L,'a_base','mpc/goals_reached/tail'):.2f}→{cD:.2f}",
                 f"**Holds (n=3)** as the largest single step and for the drop rate. 'More than doubles' holds vs A "
                 f"({cD/m(L,'a_base','mpc/goals_reached/tail'):.1f}x) but only {cD/cC:.1f}x vs C on the 3-rep tail mean: C reps 2/3 "
                 f"reach {fin('ablation_rep2/c_sysid'):.2f}/{fin('ablation_rep3/c_sysid'):.2f} goals/ep in the final eval, rep 1 was a low draw."])
    # --- distill
    rows.append(["Distillation helps MPPI: A→B 1.62→2.02 goals/ep (project notes)", "tail, rep 1",
                 f"A {a(L,'a_base','mpc/goals_reached/tail')} → B {a(L,'b_distill','mpc/goals_reached/tail')}",
                 "Direction holds on the mean, but the gap is about one std; weak evidence with n=3."])
    rows.append(["SysID (C) vs B: C's planner no better than B (README table 1.94 vs 2.02)", "tail, rep 1",
                 f"B {a(L,'b_distill','mpc/goals_reached/tail')} → C {a(L,'c_sysid','mpc/goals_reached/tail')}",
                 "**Rep-1-specific.** Over 3 reps C is above B on the mean (large spread)."])
    rows.append(["BC (E) no real gain over D (project notes, policy 0.65±0.39)", "tail, rep 1",
                 f"policy tail D {a(L,'d_termq','pi/goals_reached/tail')}, E {a(L,'e_bc','pi/goals_reached/tail')}; "
                 f"mpc tail D {a(L,'d_termq','mpc/goals_reached/tail')}, E {a(L,'e_bc','mpc/goals_reached/tail')}",
                 "Holds: differences are inside the rep spread."])
    # --- policy does not catch up
    ratios = [m(L, c, "pi/goals_reached/tail") / m(L, c, "mpc/goals_reached/tail") for c in G[L]]
    rows.append(["The policy does not catch up with the planner (README, project notes)", "all rungs",
                 f"pi/mpc tail goals ratio across rungs (3-rep means): {min(ratios):.2f}–{max(ratios):.2f}; best policy tail "
                 f"of any 150k config ≤ {max(m(f,c,'pi/goals_reached/tail') for f in G for c in G[f] if G[f][c]['all']['final_step']==[150000]):.2f}",
                 "**Holds (n=3 everywhere).** Drift-ratio numbers (1.05…2.35) come from single-run `diagnosis.json` (rep 1 only) and cannot be checked over reps."])
    # --- 300k
    rows.append(["d_termq 300k planner 5.62 goals/ep (EXPERIMENTS §3)", "5.62 final, rep 1",
                 f"final: {a('ablation_300k','d_termq@300k','mpc/goals_reached/final')}; excl. degenerate rep3: "
                 f"{a('ablation_300k','d_termq@300k','mpc/goals_reached/final','excl_degenerate')}; tail excl.: "
                 f"{a('ablation_300k','d_termq@300k','mpc/goals_reached/tail','excl_degenerate')}",
                 "Rep-1 value reproduced; holds with n=2 non-degenerate reps (rep 3 degenerate, 0 goals). Longer training helps the planner vs 150k rung D."])
    rows.append(["e_bc 300k policy 0.50 goals/ep (EXPERIMENTS §3)", "0.50 final, rep 1",
                 f"pi final {a('ablation_300k','e_bc@300k','pi/goals_reached/final')}; pi tail {a('ablation_300k','e_bc@300k','pi/goals_reached/tail')}; "
                 f"mpc final {a('ablation_300k','e_bc@300k','mpc/goals_reached/final')}",
                 "**Rep 1 is the low outlier**; the other reps' policy is 1.1–1.9. Don't quote 0.50 as the 300k result."])
    # --- netsize
    ns = {c: (m("netsize", c, "pi/goals_reached/final"), m("netsize", c, "pi/goals_reached/tail")) for c in ("netsize_small", "netsize_default", "netsize_large")}
    mono_f = ns["netsize_small"][0] < ns["netsize_default"][0] < ns["netsize_large"][0]
    mono_t = ns["netsize_small"][1] < ns["netsize_default"][1] < ns["netsize_large"][1]
    rows.append(["Net size: policy 0.19 → 0.75 → 0.75 (small/default/large), monotonic secondary lever; planner unaffected (EXPERIMENTS §4, project notes)",
                 "final, rep 1",
                 "pi final " + ", ".join(f"{c.split('_')[1]} {a('netsize',c,'pi/goals_reached/final')}" for c in ns)
                 + "; pi tail " + ", ".join(f"{c.split('_')[1]} {a('netsize',c,'pi/goals_reached/tail')}" for c in ns)
                 + "; mpc tail " + ", ".join(f"{c.split('_')[1]} {a('netsize',c,'mpc/goals_reached/tail')}" for c in ns),
                 f"Ordering small<default<large on 3-rep means: final {'holds' if mono_f else 'does NOT hold'}, tail {'holds' if mono_t else 'does NOT hold'}; "
                 "but reps overlap (std ≈ gaps), so 'monotonic' is suggestive, not established. Planner unaffected: holds. Drift ratios single-run only."])
    rows.append(["netsize_default planner 5.75 (project notes)", "final, rep 1",
                 f"netsize_default mpc final {a('netsize','netsize_default','mpc/goals_reached/final')}; rung-D config pooled n=6 final "
                 f"{fmt_agg(P[list(P)[0]]['all']['mpc/goals_reached/final'], show_n=True)}, tail {fmt_agg(P[list(P)[0]]['all']['mpc/goals_reached/tail'], show_n=True)}",
                 "5.75 is the top draw of 6 identical-config runs; the config's typical planner value is the pooled mean."])
    # --- e_no_q
    rows.append(["e_no_q: result blank (EXPERIMENTS §4); 'did not finish, CUDA OOM at 112000, no config.json, excluded' (project notes)",
                 "—",
                 f"On disk: 3 complete 150k runs with config.json (`vec/e_no_q.log` ends with `[trainer] done: 150000 env steps`). "
                 f"rep1 mpc final {f2(fin('vec/e_no_q'))}, pi final {f2(fin('vec/e_no_q','pi'))}; all: mpc tail {a('no_q','e_no_q','mpc/goals_reached/tail')}, "
                 f"pi tail {a('no_q','e_no_q','pi/goals_reached/tail')}; excl. degenerate rep3: mpc tail {a('no_q','e_no_q','mpc/goals_reached/tail','excl_degenerate')}, "
                 f"pi tail {a('no_q','e_no_q','pi/goals_reached/tail','excl_degenerate')}. Compare E: pi tail {a('ablation','e_bc','pi/goals_reached/tail')}",
                 "**The notes are stale**: the crashed run was evidently rerun to completion. Result: dropping Q from the BC actor "
                 "loss does not help the policy (lower than E), planner similar; rep 3 degenerate."])
    rows.append(["f_no_q worse than F for the policy (0.875 vs 1.06), 'no improvement over F' (project notes, EXPERIMENTS)", "final, rep 1",
                 f"pi final F {a('ablation','f_meantarget','pi/goals_reached/final')} vs f_no_q {a('no_q','f_no_q','pi/goals_reached/final')}; "
                 f"pi tail F {a('ablation','f_meantarget','pi/goals_reached/tail')} vs {a('no_q','f_no_q','pi/goals_reached/tail')}",
                 "**Holds (n=3)**, and more clearly than in rep 1."])
    # --- dagger
    rows.append(["dagger_b30 policy 1.38 goals/ep, 'strongest policy-catching-up result' (EXPERIMENTS §4, project notes)", "final, rep 1",
                 "pi final " + ", ".join(f"β{c.split('_b')[1]} {a('dagger',c,'pi/goals_reached/final')}" for c in ("dagger_b0", "dagger_b30", "dagger_b60"))
                 + "; pi tail " + ", ".join(f"β{c.split('_b')[1]} {a('dagger',c,'pi/goals_reached/tail')}" for c in ("dagger_b0", "dagger_b30", "dagger_b60")),
                 "**Does not hold with n=3.** 1.38 is rep 1 only (reps 2/3: "
                 f"{f2(fin('vec/dagger_b30_rep2','pi'))}, {f2(fin('vec/dagger_b30_rep3','pi'))}); no β is distinguishable from the others on the policy. "
                 f"Planner cost: mpc tail β0 {a('dagger','dagger_b0','mpc/goals_reached/tail')}, β30 {a('dagger','dagger_b30','mpc/goals_reached/tail')}, "
                 f"β60 {a('dagger','dagger_b60','mpc/goals_reached/tail')} — higher β lowers the planner."])
    rows.append(["Trained at 128 samples, default beats DAgger: d_termq_s128 pi 1.25 / mpc 4.44 vs dagger_b30_s128 0.69 / 3.25 (project notes)", "final, rep 1",
                 f"pi final d_termq_s128 {a('dagger','d_termq_s128','pi/goals_reached/final')} vs dagger_b30_s128 {a('dagger','dagger_b30_s128','pi/goals_reached/final')}; "
                 f"mpc final {a('dagger','d_termq_s128','mpc/goals_reached/final')} vs {a('dagger','dagger_b30_s128','mpc/goals_reached/final')}; "
                 f"mpc tail {a('dagger','d_termq_s128','mpc/goals_reached/tail')} vs {a('dagger','dagger_b30_s128','mpc/goals_reached/tail')}",
                 "**Partly.** Planner side holds (n=3). Policy side reverses on the mean: DAgger's s128 policy is not worse over 3 reps."])
    # --- rollout budget
    c = {(x["checkpoint"], x["iterations"], x["num_samples"]): x for x in rb}
    rows.append(["Rollout budget: dagger_b30 at 3x128 reaches 4.81 > d_termq full-budget 4.69 (EXPERIMENTS §7, project notes)", "eval-only, 32 ep",
                 f"{c[('dagger_b30',3,128)]['goals_reached']:.2f} vs {c[('d_termq',3,256)]['goals_reached']:.2f} "
                 f"({c[('dagger_b30',3,128)]['plan_ms_mean']:.0f} vs {c[('d_termq',3,256)]['plan_ms_mean']:.0f} ms); dagger_b30 above d_termq in all "
                 f"{sum(c[('dagger_b30',i,s)]['goals_reached']>c[('d_termq',i,s)]['goals_reached'] for (_,i,s) in [k for k in c if k[0]=='d_termq'])}/6 cells",
                 ("Numbers reproduce, but it is one checkpoint per arm, and the dagger_b30 checkpoint is the lucky rep 1 (see above). "
                  "The 0.12 goals/ep gap is far below one eval's rep-to-rep spread; treat as 'no loss at half budget', not a gain. "
                  if len(rb_reps) == 1 else
                  "**Does not hold with n=3.** Rep 1 reproduces, but reps 2/3 reverse it: on the 3-rep mean d_termq is at or above "
                  "dagger_b30 at every budget, so DAgger does not buy cheaper planning. What holds for both arms: planning degrades "
                  "gracefully with fewer samples. ")
                 + ("No multi-rep data exists." if len(rb_reps) == 1 else
                    f"Pooled over reps {sorted(rb_reps)}: dagger_b30 3x128 "
                    f"{np.mean([x['goals_reached'] for x in rb_pooled[('dagger_b30',3,128)]]):.2f} vs d_termq 3x256 "
                    f"{np.mean([x['goals_reached'] for x in rb_pooled[('d_termq',3,256)]]):.2f}; dagger_b30 > d_termq in "
                    f"{sum(np.mean([x['goals_reached'] for x in rb_pooled[('dagger_b30',i,s)]]) > np.mean([x['goals_reached'] for x in rb_pooled[('d_termq',i,s)]]) for (_,i,s) in [k for k in rb_pooled if k[0]=='d_termq'])}/6 cells on the mean (see §10).")])
    # --- offline v1
    ofl = P[list(P)[1]]["all"]; fired = P[list(P)[2]]["all"]
    rows.append(["offline_v1_01 planner 5.63; offline_v1_02 (identical rerun) 4.81; offline_v1 4.69 (EXPERIMENTS §4)", "final, rep 1",
                 f"thr 0.1 pooled mpc final {fmt_agg(ofl['mpc/goals_reached/final'], show_n=True)}, tail {fmt_agg(ofl['mpc/goals_reached/tail'], show_n=True)}; "
                 f"only fired runs tail {fmt_agg(fired['mpc/goals_reached/tail'], show_n=True)}; offline_v1 (0.15) tail {a('offline_v1','offline_v1','mpc/goals_reached/tail')}; "
                 f"rung-D config (no offline) tail {fmt_agg(P[list(P)[0]]['all']['mpc/goals_reached/tail'], show_n=True)}",
                 "5.63 is the top draw. The offline phase gives no clear planner or policy gain over the rung-D config; "
                 "`offline_v1_02_rep3` never triggered the phase (extra updates 0)."])
    # --- offline v2
    rows.append(["offline_v2 periodic planner 3.94 and 4.44 for r1/r2 (EXPERIMENTS §5)", "final",
                 f"periodic {a('offline_v2','offline_v2_periodic','mpc/goals_reached/final')}; baseline {a('offline_v2','offline_v2_baseline','mpc/goals_reached/final')} "
                 f"(excl. degenerate r1 {a('offline_v2','offline_v2_baseline','mpc/goals_reached/final','excl_degenerate')}); extra updates periodic r1/r2/r3 = "
                 f"{R['vec/offline_v2_periodic_r1']['extra_updates']}/{R['vec/offline_v2_periodic_r2']['extra_updates']}/{R['vec/offline_v2_periodic_r3']['extra_updates']}",
                 "Numbers reproduce, but the periodic phase **never fired in r2 and r3** and fired once (8000 updates) in r1, so 'periodic' is mostly the baseline. "
                 "No condition is separable from the others with n=3."])
    # --- pure rl
    rows.append(["pure_rl_v1: all three reps 0 goals/ep, drop 1.0 (EXPERIMENTS §6)", "all reps",
                 f"pi final {a('pure_rl','pure_rl_v1','pi/goals_reached/final')}, drop {a('pure_rl','pure_rl_v1','pi/dropped/tail')}", "Holds (n=3)."])
    # --- sysid
    s = {k: [R[f"{d}/c_sysid"]["sysid"][k] for d in ("ablation", "ablation_rep2", "ablation_rep3")] for k in ("comfree", "joint", "inertial")}
    sd = [R[f"{d}/d_termq"]["sysid"]["comfree"] for d in ("ablation", "ablation_rep2", "ablation_rep3")]
    rows.append(["SysID fixes the contact model: comfree err 0.19→0.09 (C), 0.19→0.11 (D); joint 0.085→0.079; inertial worse 0.092→0.107 (C) (README)",
                 "rep 1",
                 "C reps comfree " + ", ".join(f"{x[0]:.3f}→{x[1]:.3f}" for x in s["comfree"]) + "; joint " + ", ".join(f"{x[0]:.3f}→{x[1]:.3f}" for x in s["joint"])
                 + "; inertial " + ", ".join(f"{x[0]:.3f}→{x[1]:.3f}" for x in s["inertial"]) + "; D comfree " + ", ".join(f"{x[0]:.3f}→{x[1]:.3f}" for x in sd),
                 "**Holds (n=3)** for C and D: the comfree group roughly halves in every rep, joint stays flat, inertial gets worse in every C rep. "
                 "Same pattern in every 150k/300k config (§12). Exception: in the 60k offline_v2 baseline/forced_early runs the comfree error has not moved yet."])
    return rows


if __name__ == "__main__":
    main()
