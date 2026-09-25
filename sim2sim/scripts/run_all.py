"""Run every sensible sim2sim evaluation, resumably, then aggregate.

From the workspace root (paths are resolved from the script location; jobs run
with the workspace root as cwd):

    nohup setsid python sim2sim/scripts/run_all.py > sim2sim/outputs/run_all.log 2>&1 &
    ... --mpc all          # MPC on every run instead of the core set (~+3 h)
    ... --dry_run          # print the plan + time estimate only
    ... --aggregate_only   # rebuild the summary tables from what exists

Stages (each call is skipped if its summary.json already exists, so the script
can be killed and restarted; failures are logged and skipped):
  A  checks:   model parity + keyframe hold for every solver config
  B  pi:       policy-only, default solver, every loadable comfree_tdmpc run
  C  mpc:      with ComFree planner, default solver; core = ablation ladder x3 reps,
               ablation_300k x3, dagger_b{0,30,60} x3 (priority order: all rep1s
               first, then rep2s, then rep3s); --mpc all adds the rest
  D  sweep:    solver sweep -- MPC on ablation/{e_bc,d_termq} (+4 videos for e_bc),
               pi on the rep1 ladder + 300k + dagger_b30
  E  openloop: open-loop trajectory gap, e_bc pi actions (sweep) + e_bc MPC actions
  F  pearl:    every non-smoke tdmpc-square-pearl comfree-allegro-cube checkpoint
               (mpc + pi), if any exist
  G  aggregate: outputs/summary_table.csv + outputs/summary_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # sim2sim/
WS = ROOT.parent  # workspace
CF_OUT = WS / "comfree_tdmpc" / "outputs"
PEARL_LOGS = WS / "tdmpc-square-pearl" / "logs" / "comfree-allegro-cube"
PY = sys.executable
EPISODES = 64

# Rough per-call wall times on an RTX 4090 (16 lanes, 64 episodes, both backends);
# MPC is planner-bound at ~55 s per 16-episode round and backend pair.
EST_S = {"checks": 90, "pi": 20, "pi_sweep": 60, "mpc": 240, "mpc_s128": 150,
         "mpc_sweep": 6 * 115, "openloop_pi": 90, "openloop_mpc": 150, "pearl": 480}


def default_tag(agent: str, run: Path) -> str:
    """Same scheme as evaluate.py:default_tag (kept import-free: no GPU in the orchestrator)."""
    run = run.resolve()
    for base in (CF_OUT, WS / "tdmpc-square-pearl" / "logs"):
        try:
            return f"{agent}__" + "_".join(run.relative_to(base).parts)
        except ValueError:
            pass
    return f"{agent}__{run.name}"


def comfree_runs() -> list[Path]:
    runs = []
    for pt in sorted(CF_OUT.rglob("agent.pt")):
        d = pt.parent
        if "_crashed_partial" in d.parts or not (d / "config.json").exists():
            continue
        c = json.loads((d / "config.json").read_text())
        if not all(k in c for k in ("env", "agent")) or c.get("total_env_steps", 0) < 60_000:
            continue  # legacy pilot (8k steps) etc.
        runs.append(d)
    return runs


def is_pure_rl(run: Path) -> bool:
    return bool(json.loads((run / "config.json").read_text()).get("pure_rl", False))


def rep_of(run: Path) -> int:
    s = "/".join(run.relative_to(CF_OUT).parts)
    m = re.search(r"_rep(\d)|_r(\d)\b", s)
    return int(next(g for g in m.groups() if g)) if m else 1


def mpc_runs(all_runs: list[Path], which: str) -> list[Path]:
    runs = [r for r in all_runs if not is_pure_rl(r)]
    if which == "none":
        return []
    if which == "core":
        def core(r: Path) -> bool:
            rel = r.relative_to(CF_OUT).parts
            return (rel[0].startswith("ablation")
                    or (rel[0] == "vec" and re.fullmatch(r"dagger_b(0|30|60)(_rep\d)?", rel[1]) is not None))
        runs = [r for r in runs if core(r)]
    return sorted(runs, key=lambda r: (rep_of(r), str(r)))  # all rep1s first


def pearl_ckpts() -> list[tuple[Path, str]]:
    out = []
    for models in sorted(PEARL_LOGS.glob("*/*/models")):
        run = models.parent
        if run.name.startswith("smoke"):
            continue
        cks = sorted(models.glob("*.pt"))
        if (models / "final.pt").exists():
            out.append((run, "final.pt"))
        elif cks:  # killed run: newest step checkpoint
            out.append((run, max(cks, key=lambda p: int(p.stem) if p.stem.isdigit() else -1).name))
    return out


class Job:
    def __init__(self, stage: str, name: str, argv: list[str], done_files: list[Path], est: str):
        self.stage, self.name, self.argv, self.done_files, self.est = stage, name, argv, done_files, est

    def done(self) -> bool:
        return all(f.exists() for f in self.done_files)


def build_jobs(mpc_which: str) -> list[Job]:
    from_solver = ["default", "cone_elliptic", "solref_tc0.01", "solref_tc0.04",
                   "solimp_hard", "solimp_soft", "noslip10"]
    out = ROOT / "outputs"
    ev = [PY, str(HERE / "evaluate.py")]
    jobs: list[Job] = []
    runs = comfree_runs()

    jobs.append(Job("A", "checks", [PY, str(HERE / "check_parity.py"), "--run",
                                     str(CF_OUT / "ablation" / "e_bc"), "--solver", "all",
                                     "--out", str(out / "checks" / "parity.json")],
                    [out / "checks" / "parity.json"], "checks"))

    for r in runs:
        tag = default_tag("comfree_pi", r)
        jobs.append(Job("B", tag, ev + ["--agent", "comfree_pi", "--run", str(r), "--episodes", str(EPISODES)],
                        [out / tag / "default" / "summary.json"], "pi"))

    for r in mpc_runs(runs, mpc_which):
        tag = default_tag("comfree_mpc", r)
        s128 = json.loads((r / "config.json").read_text())["planner"]["num_samples"] <= 128
        jobs.append(Job("C", tag, ev + ["--agent", "comfree_mpc", "--run", str(r), "--episodes", str(EPISODES)],
                        [out / tag / "default" / "summary.json"], "mpc_s128" if s128 else "mpc"))

    sweep_done = lambda tag: [out / tag / s / "summary.json" for s in from_solver]  # noqa: E731
    for rel in ("ablation/e_bc", "ablation/d_termq"):
        r = CF_OUT / rel
        tag = default_tag("comfree_mpc", r)
        extra = ["--video", "4"] if rel.endswith("e_bc") else []
        jobs.append(Job("D", tag + " --sweep", ev + ["--agent", "comfree_mpc", "--run", str(r), "--episodes",
                                                    str(EPISODES), "--sweep", "--skip_comfree"] + extra,
                        sweep_done(tag), "mpc_sweep"))
    pi_sweep = [CF_OUT / "ablation" / x for x in ("a_base", "b_distill", "c_sysid", "d_termq", "e_bc", "f_meantarget")]
    pi_sweep += [CF_OUT / "ablation_300k" / "d_termq", CF_OUT / "ablation_300k" / "e_bc", CF_OUT / "vec" / "dagger_b30"]
    for r in pi_sweep:
        if not (r / "agent.pt").exists():
            continue
        tag = default_tag("comfree_pi", r)
        jobs.append(Job("D", tag + " --sweep", ev + ["--agent", "comfree_pi", "--run", str(r), "--episodes",
                                                    str(EPISODES), "--sweep", "--skip_comfree"],
                        sweep_done(tag), "pi_sweep"))

    ol = [PY, str(HERE / "open_loop_gap.py")]
    e_bc = CF_OUT / "ablation" / "e_bc"
    for agent, extra, est in (("comfree_pi", ["--sweep"], "openloop_pi"), ("comfree_mpc", [], "openloop_mpc")):
        tag = "openloop__" + default_tag(agent, e_bc)
        jobs.append(Job("E", tag, ol + ["--agent", agent, "--run", str(e_bc), "--episodes", str(EPISODES)] + extra,
                        [out / tag / "summary.json"], est))

    for run, ck in pearl_ckpts():
        for agent in ("pearl_mpc", "pearl_pi"):
            tag = default_tag(agent, run)
            jobs.append(Job("F", f"{tag} ({ck})", ev + ["--agent", agent, "--run", str(run), "--pearl_ckpt", ck,
                                                        "--episodes", str(EPISODES)],
                            [out / tag / "default" / "summary.json"], "pearl"))
    return jobs


def _group(tag: str) -> str:
    """Collapse replicate suffixes so reps of one configuration share a group."""
    g = re.sub(r"(ablation(?:_300k)?)_rep\d", r"\1", tag)
    g = re.sub(r"_rep\d$", "", g)
    return re.sub(r"_r\d$", "", g)


def aggregate() -> None:
    out = ROOT / "outputs"
    rows = []
    for summ in sorted(out.glob("*/*/summary.json")):
        solver, tag = summ.parent.name, summ.parent.parent.name
        if solver == "comfree" or tag.startswith("openloop__") or tag == "checks":
            continue
        s = json.loads(summ.read_text())
        cf_s = out / tag / "comfree" / "summary.json"
        sc = json.loads(cf_s.read_text()).get("self_check", {}) if cf_s.exists() else {}
        row = {"tag": tag, "group": _group(tag), "solver": solver, "n": s["mujoco"]["n"]}
        for side in ("comfree", "mujoco"):
            for k in ("return", "goals_reached", "success", "dropped", "time_to_success_s"):
                row[f"{side}/{k}"] = s.get(side, {}).get(k, {}).get("mean", float("nan"))
        for k in ("return", "goals_reached", "success", "dropped"):
            g = s.get("gap_mujoco_minus_comfree", {}).get(k, {})
            row[f"gap/{k}"] = g.get("diff", float("nan"))
            row[f"gap/{k}_ci_lo"], row[f"gap/{k}_ci_hi"] = (g.get("ci95") or [float("nan")] * 2)
        row["self_check_ok"] = sc.get("ok")
        row["nonfinite_actions"] = (s.get("comfree", {}).get("nonfinite_action_steps_total", 0)
                                    + s["mujoco"].get("nonfinite_action_steps_total", 0))
        row["mujoco_bad_qacc"] = s["mujoco"].get("bad_qacc_resets_total", 0)
        rows.append(row)
    if not rows:
        print("[aggregate] nothing to aggregate yet")
        return
    keys = list(rows[0])
    with open(out / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Markdown: default solver, replicates pooled per group (mean over reps of per-run means).
    import statistics as st

    def fmt(x, p=2):
        return "nan" if x != x else f"{x:.{p}f}"

    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["solver"] == "default":
            groups.setdefault(r["group"], []).append(r)
    lines = ["# sim2sim summary (default MuJoCo solver; mean over replicates)", "",
             "| group | reps | ComFree return | MuJoCo return | Δreturn | ComFree goals | MuJoCo goals | "
             "ComFree drop | MuJoCo drop | self-check | flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, rs in sorted(groups.items()):
        m = lambda k: st.fmean([r[k] for r in rs])  # noqa: E731
        flags = []
        if any(r["nonfinite_actions"] for r in rs):
            flags.append("NaN actions")
        if any(r["mujoco_bad_qacc"] for r in rs):
            flags.append("MuJoCo bad-qacc")
        checks = [r["self_check_ok"] for r in rs]
        chk = "ok" if all(c is True for c in checks) else ("n/a" if all(c is None for c in checks) else
                                                           f"{sum(c is False for c in checks)} fail")
        lines.append(f"| {g} | {len(rs)} | {fmt(m('comfree/return'), 1)} | {fmt(m('mujoco/return'), 1)} | "
                     f"{fmt(m('gap/return'), 1)} | {fmt(m('comfree/goals_reached'))} | "
                     f"{fmt(m('mujoco/goals_reached'))} | {fmt(m('comfree/dropped'))} | "
                     f"{fmt(m('mujoco/dropped'))} | {chk} | {', '.join(flags)} |")
    sweep = [r for r in rows if r["solver"] != "default"]
    if sweep:
        lines += ["", "## Solver sweep (per run)", "",
                  "| tag | solver | MuJoCo return | Δreturn [95% CI] | MuJoCo goals | MuJoCo drop |",
                  "|---|---|---|---|---|---|"]
        by_tag = {}
        for r in rows:
            by_tag.setdefault(r["tag"], []).append(r)
        for tag, rs in sorted(by_tag.items()):
            if len(rs) < 2:
                continue
            for r in sorted(rs, key=lambda r: (r["solver"] != "default", r["solver"])):
                lines.append(f"| {tag} | {r['solver']} | {fmt(r['mujoco/return'], 1)} | {fmt(r['gap/return'], 1)} "
                             f"[{fmt(r['gap/return_ci_lo'], 1)}, {fmt(r['gap/return_ci_hi'], 1)}] | "
                             f"{fmt(r['mujoco/goals_reached'])} | {fmt(r['mujoco/dropped'])} |")
    (out / "summary_table.md").write_text("\n".join(lines) + "\n")
    print(f"[aggregate] {len(rows)} rows -> {out/'summary_table.csv'}, {out/'summary_table.md'}")


def hms(s: float) -> str:
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpc", choices=["core", "all", "none"], default="core")
    ap.add_argument("--stages", default="ABCDEF", help="subset of stages to run, e.g. 'BC'")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate()
        return 0

    jobs = [j for j in build_jobs(args.mpc) if j.stage in args.stages]
    todo = [j for j in jobs if not j.done()]
    total_est = sum(EST_S[j.est] for j in todo)
    print(f"[run_all] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run, "
          f"estimated {hms(total_est)}")
    for st_ in sorted({j.stage for j in jobs}):
        sj = [j for j in todo if j.stage == st_]
        print(f"  stage {st_}: {len(sj):3d} jobs  ~{hms(sum(EST_S[j.est] for j in sj))}")
    if not any(j.stage == "F" for j in jobs) and "F" in args.stages:
        print("  stage F: no non-smoke pearl comfree-allegro-cube checkpoint found -- skipped")
    if args.dry_run:
        for j in todo:
            print(f"    [{j.stage}] {j.name}")
        return 0

    logs = ROOT / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL="egl")
    t_start, spent_est, failed = time.time(), 0.0, []
    for i, j in enumerate(todo):
        log = logs / (re.sub(r"[^\w.-]+", "_", f"{j.stage}_{j.name}") + ".log")
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run(j.argv, cwd=WS, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        spent_est += EST_S[j.est]
        elapsed = time.time() - t_start
        eta = (total_est - spent_est) * elapsed / max(spent_est, 1)
        status = "ok" if rc == 0 and j.done() else f"FAILED rc={rc} (see {log})"
        if status != "ok":
            failed.append(j.name)
        print(f"[{i + 1}/{len(todo)}] [{j.stage}] {j.name}: {status}  ({time.time() - t0:.0f}s, "
              f"elapsed {hms(elapsed)}, ETA {hms(eta)})", flush=True)
        if j.stage in "BCD" and (i + 1) % 10 == 0:
            aggregate()
    aggregate()
    print(f"[run_all] finished in {hms(time.time() - t_start)}; {len(failed)} failed"
          + (": " + ", ".join(failed) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
