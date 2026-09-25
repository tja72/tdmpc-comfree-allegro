"""Validation checks 1 + 2: model parity CPU-MuJoCo vs ComFree, and keyframe hold.

Run from the workspace root:

    python sim2sim/scripts/check_parity.py
    ... --run comfree_tdmpc/outputs/ablation/e_bc   # also asserts u == params_reality.npy
    ... --solver all                                # hold check for every sweep config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sim2sim.envs import keyframe_hold, make_comfree_env, make_mujoco_env  # noqa: E402
from comfree_tdmpc.envs.vec_allegro import VecEnvConfig  # noqa: E402
from sim2sim.params_cpu import check_parity  # noqa: E402
from sim2sim.solver_configs import BY_NAME, SWEEP, model_solver_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None, help="comfree_tdmpc run dir (for params_reality.npy)")
    ap.add_argument("--reality_seed", type=int, default=12345)
    ap.add_argument("--reality_perturb", type=float, default=0.25)
    ap.add_argument("--solver", default="default", help="sweep config name, or 'all'")
    ap.add_argument("--out", type=Path, default=None, help="optional json output path")
    args = ap.parse_args()

    env_cfg = VecEnvConfig(num_envs=1)
    cf_env, space, u = make_comfree_env(env_cfg, 0, args.reality_seed, args.reality_perturb)
    report: dict = {"reality_seed": args.reality_seed, "reality_perturb": args.reality_perturb}
    if args.run is not None:
        ref = np.load(args.run / "params_reality.npy")
        report["u_matches_params_reality"] = bool(np.allclose(u, ref, atol=1e-12))
        print(f"u == {args.run}/params_reality.npy: {report['u_matches_params_reality']}")

    # Check 2 (ComFree side, for reference): the same probe on the ComFree env.
    report["hold_comfree"] = keyframe_hold(cf_env, u)
    cf_env.writer.write_u(u)  # probe_stable already leaves u written; rewrite explicitly

    ok = True
    report["solvers"] = {}
    for sc in (SWEEP if args.solver == "all" else [BY_NAME[args.solver]]):
        mj_env = make_mujoco_env(env_cfg, 0, space, u, solver=sc)
        rec = {"solver": sc.to_dict(), "model": model_solver_summary(mj_env.sim.mjm),
               "skipped_params": mj_env.writer.skipped}
        # Check 1: every field the handles touch, whole array, vs ComFree world 0.
        rec["parity"] = parity = check_parity(mj_env.sim, cf_env.sim, mj_env.writer.written_fields)
        rec["hold"] = keyframe_hold(mj_env, u)
        report["solvers"][sc.name] = rec
        ok &= parity["ok"] and rec["hold"]["stable"]

        print(f"\n== solver {sc.name}")
        for group in ("fields", "derived"):
            for name, r in parity[group].items():
                if "skipped" in r:
                    print(f"  [{group}] {name:18s} skipped ({r['skipped']})")
                else:
                    flag = "OK " if r["ok"] else ("DIFF" if group == "fields" else "info")
                    print(f"  [{group}] {name:18s} {flag} max_abs={r['max_abs']:.2e} max_rel={r['max_rel']:.2e}"
                          + (f"  first_bad={r['first_bad']}" if not r["ok"] else ""))
        print(f"  hold: {rec['hold']}")
    print(f"\nComFree hold: {report['hold_comfree']}")
    print(f"skipped (no MuJoCo equivalent): {report['solvers'][next(iter(report['solvers']))]['skipped_params']}")
    print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
