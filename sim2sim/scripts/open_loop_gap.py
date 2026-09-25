"""Validation check 4: policy-independent open-loop trajectory gap.

Records action sequences from agent rollouts in ComFree (the training plant),
then replays each one open loop from the rollout's initial state in
  * ComFree, twice  -> "comfree_rerun": the noise floor (ComFree is not bitwise
    deterministic, ~1e-6/step, amplified by contact chaos), and
  * CPU MuJoCo, per solver config -> "mujoco_<solver>",
and logs, per control step, cube position error, cube orientation error
(geodesic angle) and hand-qpos RMS error, each vs the first ComFree replay.
Steps after the source episode ended (drop / time limit) are masked: the
recorded actions there belong to the auto-reset episode.

Run from the workspace root:

    python sim2sim/scripts/open_loop_gap.py \
        --agent comfree_pi --run comfree_tdmpc/outputs/ablation/e_bc --episodes 64 [--sweep]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from comfree_tdmpc.envs.task import quat_error  # noqa: E402
from sim2sim import harness  # noqa: E402
from sim2sim.agents import ComfreeController, load_run_config  # noqa: E402
from sim2sim.envs import env_config_from_run, make_comfree_env, make_mujoco_env  # noqa: E402
from sim2sim.solver_configs import BY_NAME, SWEEP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("cube_pos_err_m", "cube_rot_err_rad", "hand_qpos_rms_rad")
REPORT_STEPS = (5, 10, 25, 50, 100)  # control steps (x 0.02 s)


@torch.no_grad()
def replay(sim, roll: dict) -> np.ndarray:
    """Open-loop replay of one round on a BatchSim-like sim; returns qpos (T, E, nq)."""
    dev = sim.torch_device
    sim.set_state(torch.as_tensor(roll["qpos0"], device=dev), torch.as_tensor(roll["qvel0"], device=dev))
    sim.set_ctrl_raw(torch.as_tensor(roll["ctrl0"], device=dev))
    out = []
    for a in roll["actions"]:
        sim.set_action(torch.as_tensor(a, device=dev))
        sim.step()
        out.append(sim.qpos.detach().cpu().numpy().copy())
    return np.stack(out)


def errors(ref: np.ndarray, other: np.ndarray, live: np.ndarray) -> dict[str, np.ndarray]:
    """Per (t, lane) errors, NaN where the source episode had already ended."""
    r, o = torch.as_tensor(ref, dtype=torch.float64), torch.as_tensor(other, dtype=torch.float64)
    pos = (r[..., 16:19] - o[..., 16:19]).norm(dim=-1).numpy()
    rot = quat_error(o[..., 19:23], r[..., 19:23]).numpy()
    hand = (r[..., :16] - o[..., :16]).pow(2).mean(-1).sqrt().numpy()
    mask = np.where(live, 1.0, np.nan)
    return {"cube_pos_err_m": pos * mask, "cube_rot_err_rad": rot * mask, "hand_qpos_rms_rad": hand * mask}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="comfree_pi", choices=["comfree_pi", "comfree_mpc"])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default="default")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from evaluate import default_tag  # same tag scheme as evaluate.py

    rcfg = load_run_config(args.run)
    env_cfg = env_config_from_run(rcfg, args.num_envs)
    cf_env, space, u = make_comfree_env(env_cfg, args.seed, int(rcfg.get("reality_seed", 12345)),
                                        float(rcfg.get("reality_perturb", 0.25)))
    ref = args.run / "params_reality.npy"
    if ref.exists() and not np.array_equal(u, np.load(ref)):
        raise SystemExit(f"apply_reality u != {ref}")
    ctrl = ComfreeController(args.run, args.agent.split("_")[1], args.num_envs, cf_env.task, space)
    _, rollouts = harness.run_vec_episodes(cf_env, ctrl, args.episodes, args.seed, record_rollouts=True)
    print(f"[open-loop] recorded {len(rollouts)} rounds x {args.num_envs} lanes from {args.agent}")

    solvers = SWEEP if args.sweep else [BY_NAME[args.solver]]
    mj_sims = {s.name: make_mujoco_env(env_cfg, args.seed, space, u, s).sim for s in solvers}

    pairs: dict[str, dict[str, list[np.ndarray]]] = {}
    for roll in rollouts:
        live = roll["live"]
        base = replay(cf_env.sim, roll)
        runs = {"comfree_rerun": replay(cf_env.sim, roll)}
        for name, sim in mj_sims.items():
            runs[f"mujoco_{name}"] = replay(sim, roll)
        for pname, q in runs.items():
            for k, v in errors(base, q, live).items():
                pairs.setdefault(pname, {}).setdefault(k, []).append(v)

    T = env_cfg.episode_length
    dt = cf_env.control_dt
    stats = {p: {k: np.concatenate(v, axis=1) for k, v in d.items()} for p, d in pairs.items()}  # (T, lanes)
    tag = args.tag or ("openloop__" + default_tag(args.agent, args.run))
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "open_loop_gap.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = [f"{p}/{k}/{s}" for p in stats for k in METRICS for s in ("median", "p25", "p75", "mean")]
        w.writerow(["t", "time_s", "n_valid"] + cols)
        for t in range(T):
            n_valid = int(np.isfinite(stats["comfree_rerun"]["cube_pos_err_m"][t]).sum())
            row = [t + 1, (t + 1) * dt, n_valid]
            for p in stats:
                for k in METRICS:
                    x = stats[p][k][t]
                    x = x[np.isfinite(x)]
                    row += ([float(np.median(x)), float(np.quantile(x, .25)), float(np.quantile(x, .75)),
                             float(x.mean())] if len(x) else [float("nan")] * 4)
            w.writerow(row)

    summary = {"agent": args.agent, "run": str(args.run), "episodes": len(rollouts) * args.num_envs,
               "dt": dt, "solvers": [s.to_dict() for s in solvers], "mujoco_bad_qacc_resets": {n: int(s.bad_qacc.sum()) for n, s in mj_sims.items()},
               "median_error_at_step": {}}
    for p in stats:
        summary["median_error_at_step"][p] = {
            k: {str(t): (float(np.nanmedian(stats[p][k][t - 1])) if np.isfinite(stats[p][k][t - 1]).any()
                         else None) for t in REPORT_STEPS if t <= T}
            for k in METRICS}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ts = np.arange(1, T + 1) * dt
        for ax, k in zip(axes, METRICS):
            for p in stats:
                with np.errstate(all="ignore"):
                    med = np.nanmedian(stats[p][k], axis=1)
                    lo, hi = np.nanquantile(stats[p][k], .25, axis=1), np.nanquantile(stats[p][k], .75, axis=1)
                ax.plot(ts, med, label=p)
                ax.fill_between(ts, lo, hi, alpha=0.15)
            ax.set_yscale("log")
            ax.set_xlabel("time [s]")
            ax.set_title(k + " vs ComFree replay (median, IQR)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "open_loop_gap.png", dpi=120)
    except Exception as e:  # plotting is optional
        print(f"[open-loop] no plot: {e}")

    for p in stats:
        m = summary["median_error_at_step"][p]
        print(f"  {p:24s} " + "  ".join(
            f"{k.split('_err')[0].split('_rms')[0]}@{t}={m[k][str(t)]:.2e}" for k in METRICS for t in (10, 50)
            if m[k].get(str(t)) is not None))
    print(f"[open-loop] outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
