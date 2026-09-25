"""The ablation ladder: start from nothing, switch on one component at a time.

Every rung differs from the one below it in exactly one thing, and every rung
shares the planner configuration, the physics gap, the seed and the step budget,
so the difference between two rows is attributable to the switch between them.

    A  base      MPPI over a wrong model; Q and pi learn from executed actions
    B  +distill  the MPPI elite rollouts are harvested as training data too
    C  +sysid    the model's 71 unknown parameters are identified online
    D  +termQ    the planner adds gamma^H Q(s_H) to every sampled rollout
    E  +BC       same as D, but the policy is fitted to the elite actions
                 directly instead of through TD-M(PC)^2's residual prior

Runs are sequential on purpose: at 16 environments the planner already fills the
device, so two processes would split one GPU's throughput rather than add to it.

    python scripts/run_ablation.py --steps 150000
    python scripts/run_ablation.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LADDER = [
    ("a_base", [], "MPPI on a wrong model, TD from executed actions only"),
    ("b_distill", ["--distill"], "+ harvest MPPI elite rollouts"),
    ("c_sysid", ["--distill", "--sysid"], "+ identify the 71 unknown parameters"),
    ("d_termq", ["--distill", "--sysid", "--terminal_value"],
     "+ learned terminal value in the planner"),
    ("e_bc", ["--distill", "--sysid", "--terminal_value",
              "--actor_mode", "bc", "--sim_data_ratio", "0.75"],
     "full stack, policy fitted to elite actions by regression"),
    # Rung F is not part of the ladder: it is the fix suggested by the
    # measurement in Part 5.  The elite actions the policy is asked to copy
    # disagree with each other about as much as the policy's own error, so it is
    # given the plan's weighted mean instead -- one consistent target per state.
    ("f_meantarget", ["--distill", "--sysid", "--terminal_value",
                      "--actor_mode", "bc_mu", "--elite_target", "mean",
                      "--sim_data_ratio", "0.75", "--prior_coef", "2.0"],
     "full stack, policy fitted to the plan mean (single-mode target)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="run only these rungs by name")
    ap.add_argument("--out_root", type=str, default="outputs/ablation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rungs = LADDER if args.only is None else [r for r in LADDER if r[0] in args.only]

    if args.list:
        for name, flags, note in rungs:
            done = (out_root / name / "agent.pt").exists()
            print(f"[{'done' if done else '    '}] {name:<11} {note}")
            print(f"{'':<15}{' '.join(flags) or '(no switches)'}")
        return

    if not args.force:
        rungs = [r for r in rungs if not (out_root / r[0] / "config.json").exists()]
    if not rungs:
        print("every rung is already finished (use --force to rerun)")
        return

    print(f"{len(rungs)} run(s) x {args.steps} env steps -> {out_root}", flush=True)
    start = time.time()
    for name, flags, note in rungs:
        log = out_root / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(REPO / "scripts" / "train_vec.py"),
            "--name", name, "--out_root", str(out_root),
            "--steps", str(args.steps), "--num_envs", str(args.num_envs),
            "--horizon", str(args.horizon), "--iterations", str(args.iterations),
            "--seed", str(args.seed), "--tag", name, *flags,
        ]
        print(f"[{(time.time()-start)/60:6.1f} min] start  {name}: {note}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
        status = "ok" if rc == 0 else f"FAILED ({rc})"
        print(f"[{(time.time()-start)/60:6.1f} min] finish {name} {status} "
              f"after {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"ablation complete in {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
