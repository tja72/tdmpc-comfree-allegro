"""Declarative experiment runner.

Every training run in the report is defined here, grouped into suites of
matched pairs that differ in exactly one thing.  Runs go to `outputs/runs/<name>`
and their console output to `outputs/runs/<name>.log`.

    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --suite all --jobs 4
    python scripts/run_experiments.py --suite terminal --jobs 2 --force

Concurrency does not create GPU throughput out of nothing -- the planner
saturates the device -- but it does keep several comparisons advancing at once,
and it fills the gaps while each process is busy in Python.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs" / "runs"

# How far "reality" sits from the compiled model, in normalised units.
REALITY_PERTURB = "0.25"


@dataclass
class Job:
    name: str
    args: list[str]
    suite: str
    note: str = ""
    priority: int = 0


def common(episodes: int, horizon: int, extra: list[str] | None = None) -> list[str]:
    return [
        "--episodes", str(episodes),
        "--episode_length", "120",
        "--horizon", str(horizon),
        "--eval_every", "5",
        "--eval_episodes_pi", "3",
        "--video_every", "15",
        "--updates_per_step", "2",
        *(extra or []),
    ]


def build_jobs(episodes: int, horizon: int, term_horizon: int) -> list[Job]:
    jobs: list[Job] = []

    # 1. system identification: the reality gap is purely parametric
    jobs += [
        Job("sysid_on", common(episodes, horizon, [
            "--distill_rollouts", "--reality_perturb", REALITY_PERTURB,
            "--sysid_every_steps", "300", "--sysid_iters", "50",
            "--sysid_warmup_steps", "240"]),
            suite="sysid", priority=1,
            note="71 unknown parameters, identified online"),
        Job("sysid_off", common(episodes, horizon, [
            "--distill_rollouts", "--reality_perturb", REALITY_PERTURB, "--no_sysid"]),
            suite="sysid", priority=1,
            note="same 71 unknowns, never identified"),
    ]

    # 2. terminal value: can a learned Q replace planning horizon?
    # Deliberately run at a shorter horizon: at the full horizon the planner
    # already solves the task and leaves the value nothing to contribute.
    jobs += [
        Job("termQ_on", common(episodes, term_horizon,
            ["--distill_rollouts", "--reality_perturb", "0.0", "--no_sysid"]),
            suite="terminal", priority=2,
            note=f"H={term_horizon} planner + learned terminal Q"),
        Job("termQ_off", common(episodes, term_horizon, [
            "--distill_rollouts", "--reality_perturb", "0.0", "--no_sysid",
            "--no_terminal_value"]),
            suite="terminal", priority=2,
            note=f"H={term_horizon} planner, no terminal value"),
    ]

    # 3. distillation: does harvesting MPPI rollouts build a policy?
    jobs += [
        Job("distill_on", common(episodes, horizon,
            ["--distill_rollouts", "--reality_perturb", "0.0", "--no_sysid"]),
            suite="distill", priority=0,
            note="policy and Q trained on top-k MPPI elite rollouts"),
        Job("distill_off", common(episodes, horizon,
            ["--reality_perturb", "0.0", "--no_sysid"]),
            suite="distill", priority=0,
            note="policy and Q trained on executed transitions only"),
    ]
    return jobs


def launch(job: Job, out_root: Path) -> subprocess.Popen:
    log = out_root / f"{job.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "train.py"),
        "--name", job.name, "--out_root", str(out_root), *job.args,
    ]
    handle = open(log, "w")
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="all",
                    help="all | sysid | terminal | distill | <run name>")
    # Each process holds ~2 GB of host RSS (warp modules, CUDA context, torch).
    # Four of them exhausts a 15 GB WSL2 allocation and the kernel OOM-kills one
    # mid-run, which is silent apart from an exit code of -9.
    ap.add_argument("--jobs", type=int, default=3, help="concurrent training processes")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--term_horizon", type=int, default=8)
    ap.add_argument("--out_root", type=str, default=str(RUNS))
    ap.add_argument("--force", action="store_true", help="rerun even if finished")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    jobs = build_jobs(args.episodes, args.horizon, args.term_horizon)
    if args.suite != "all":
        jobs = [j for j in jobs if j.suite == args.suite or j.name == args.suite]
    if not jobs:
        print(f"no jobs match '{args.suite}'")
        return

    if args.list:
        for j in sorted(jobs, key=lambda j: j.priority):
            done = (out_root / j.name / "agent.pt").exists()
            print(f"[{'done' if done else '    '}] {j.suite:<9} {j.name:<14} {j.note}")
        return

    if not args.force:
        jobs = [j for j in jobs if not (out_root / j.name / "agent.pt").exists()]
        if not jobs:
            print("everything in this suite is already finished (use --force to rerun)")
            return

    jobs.sort(key=lambda j: j.priority)
    print(f"{len(jobs)} run(s), {args.jobs} at a time -> {out_root}")
    for j in jobs:
        print(f"  {j.suite:<9} {j.name:<14} {j.note}")
    if args.dry_run:
        return

    pending, running = list(jobs), {}
    start = time.time()
    while pending or running:
        while pending and len(running) < args.jobs:
            job = pending.pop(0)
            running[job.name] = (launch(job, out_root), job, time.time())
            print(f"[{time.time()-start:7.0f}s] start  {job.name}", flush=True)
        time.sleep(10)
        for name, (proc, job, t0) in list(running.items()):
            if proc.poll() is not None:
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                print(f"[{time.time()-start:7.0f}s] finish {name} {status} "
                      f"after {(time.time()-t0)/60:.1f} min", flush=True)
                del running[name]
    print(f"all runs complete in {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
