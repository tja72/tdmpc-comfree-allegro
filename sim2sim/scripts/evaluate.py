"""Evaluate a trained agent in ComFree (training sim) and in CPU MuJoCo (sim2sim).

Run from the workspace root (relative --run paths resolve against the cwd):

    python sim2sim/scripts/evaluate.py \
        --agent comfree_mpc --run comfree_tdmpc/outputs/ablation/e_bc --episodes 64 [--sweep] [--video 4]

    --agent comfree_pi | comfree_mpc   --run comfree_tdmpc/outputs/<...>/<run>
    --agent pearl_mpc  | pearl_pi      --run tdmpc-square-pearl/logs/comfree-allegro-cube/<seed>/<exp_name>

Writes sim2sim/outputs/<tag>/comfree/ (harness on the ComFree backend, incl. the
eval.csv self-check) and sim2sim/outputs/<tag>/<solver>/ (MuJoCo, incl. model
parity + keyframe hold for that solver config, and the gap vs ComFree).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import os

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp  # noqa: E402

from comfree_tdmpc.envs.vec_allegro import VecEnvConfig  # noqa: E402
from sim2sim import harness  # noqa: E402
from sim2sim.agents import ComfreeController, load_pearl, load_run_config  # noqa: E402
from sim2sim.envs import env_config_from_run, keyframe_hold, make_comfree_env, make_mujoco_env  # noqa: E402
from sim2sim.params_cpu import check_parity  # noqa: E402
from sim2sim.solver_configs import BY_NAME, SWEEP, model_solver_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def default_tag(agent: str, run: Path) -> str:
    run = run.resolve()
    try:
        rel = run.relative_to(WORKSPACE / "comfree_tdmpc" / "outputs")
    except ValueError:
        try:
            rel = run.relative_to(WORKSPACE / "tdmpc-square-pearl" / "logs")
        except ValueError:
            rel = Path(run.name)
    return f"{agent}__" + "_".join(rel.parts)


class FrameRecorder:
    """Short gifs of the first `k` lanes of the first round (MuJoCo side)."""

    def __init__(self, k: int, title: str):
        from comfree_tdmpc.render import SceneRenderer

        self.k, self.title = k, title
        self.renderer = SceneRenderer()
        self.frames: list[list[np.ndarray]] = [[] for _ in range(k)]

    def __call__(self, env, t, info, live):
        for i in range(min(self.k, env.num_envs)):
            if live is not None and not bool(live[i]):
                continue
            self.frames[i].append(self.renderer.frame(
                env.qpos[i].detach().cpu().numpy(), env.goal_quat[i].detach().cpu().numpy(),
                label=f"{self.title}  t={t + 1}  goals={int(env.goals_reached[i])}",
                at_goal=bool(env.success_streak[i] > 0),
            ))

    def save(self, out_dir: Path) -> list[str]:
        from comfree_tdmpc.render import save_gif

        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, fr in enumerate(self.frames):
            if fr:
                paths.append(str(save_gif(fr, out_dir / f"ep{i}.gif")))
        self.renderer.close()
        return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, choices=["comfree_pi", "comfree_mpc", "pearl_mpc", "pearl_pi"])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--num_envs", type=int, default=16, help="parallel lanes (comfree agents; pearl is always 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default="default", help=f"one of {list(BY_NAME)}")
    ap.add_argument("--sweep", action="store_true", help="run every solver config in solver_configs.SWEEP")
    ap.add_argument("--reality_seed", type=int, default=None, help="default: from the run's config")
    ap.add_argument("--reality_perturb", type=float, default=None, help="default: from the run's config")
    ap.add_argument("--num_samples", type=int, default=None, help="override planner samples (comfree_mpc)")
    ap.add_argument("--video", type=int, default=0, help="gifs of the first K MuJoCo episodes")
    ap.add_argument("--skip_comfree", action="store_true", help="reuse an existing <tag>/comfree/episodes.csv")
    ap.add_argument("--no_comfree", action="store_true", help="MuJoCo only (no ComFree reference / gap)")
    ap.add_argument("--pearl_ckpt", default="final.pt")
    ap.add_argument("--pearl_hydra_dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    solvers = SWEEP if args.sweep else [BY_NAME[args.solver]]
    tag = args.tag or default_tag(args.agent, args.run)
    out = args.out / tag
    versions = {"mujoco": mujoco.__version__, "warp": warp.config.version, "torch": torch.__version__}
    assert mujoco.__version__ == "3.6.0" and warp.config.version == "1.14.0", versions
    is_pearl = args.agent.startswith("pearl")

    if is_pearl:
        pcfg, genv, pagent, hydra_dir = load_pearl(args.run, args.pearl_ckpt, args.pearl_hydra_dir)
        cf_wrapper = genv.env  # ComFreeAllegroEnv, already in apply_reality(...)'s plant
        cf_vec, space, u = cf_wrapper.env, cf_wrapper.space, cf_wrapper.u_reality
        rs, rp = int(cf_wrapper.cfg.reality_seed), float(cf_wrapper.cfg.reality_perturb)
        if args.reality_seed is not None or args.reality_perturb is not None:
            raise SystemExit("pearl: set the reality via the run's +comfree_allegro.* config, not CLI")
        env_cfg = VecEnvConfig(num_envs=1, episode_length=int(cf_wrapper.max_episode_steps))
        use_pi = args.agent == "pearl_pi"
        agent_desc = {"run": str(args.run), "ckpt": args.pearl_ckpt, "hydra_dir": str(hydra_dir),
                      "mode": "pi" if use_pi else "mpc", "horizon": int(pcfg.horizon),
                      "model_size": pcfg.get("model_size"), "actor_mode": pcfg.get("actor_mode")}
        eval_prefix = "pi" if use_pi else "mpc"
        stem = Path(args.pearl_ckpt).stem
        ckpt_step = int(stem) if stem.isdigit() else int(pcfg.steps)

        def run_eps(vec_env, frame_fn=None):
            orig = cf_wrapper.env
            cf_wrapper.env = vec_env
            try:
                return harness.run_gym_episodes(genv, pagent, use_pi, args.episodes, args.seed, frame_fn)
            finally:
                cf_wrapper.env = orig
    else:
        rcfg = load_run_config(args.run)
        rs = args.reality_seed if args.reality_seed is not None else int(rcfg.get("reality_seed", 12345))
        rp = args.reality_perturb if args.reality_perturb is not None else float(rcfg.get("reality_perturb", 0.25))
        env_cfg = env_config_from_run(rcfg, args.num_envs)
        cf_vec, space, u = make_comfree_env(env_cfg, args.seed, rs, rp)
        ref = args.run / "params_reality.npy"
        if ref.exists() and args.reality_seed is None and args.reality_perturb is None:
            if not np.array_equal(u, np.load(ref)):
                raise SystemExit(f"apply_reality u != {ref} -- reality plant not reproduced")
        mode = "pi" if args.agent == "comfree_pi" else "mpc"
        ctrl = ComfreeController(args.run, mode, args.num_envs, cf_vec.task, space, args.num_samples)
        agent_desc = ctrl.describe()
        eval_prefix = mode
        ckpt_step = int(rcfg.get("total_env_steps", 0)) or None

        def run_eps(vec_env, frame_fn=None):
            return harness.run_vec_episodes(vec_env, ctrl, args.episodes, args.seed, frame_fn)[0]

    common = {"agent_type": args.agent, "tag": tag, "episodes": args.episodes, "seed": args.seed,
              "num_envs": env_cfg.num_envs, "reality_seed": rs, "reality_perturb": rp,
              "u_reality": np.asarray(u).tolist(), "agent": agent_desc, "versions": versions,
              "argv": sys.argv}

    # ComFree reference
    cf_dir = out / "comfree"
    cf_recs = None
    if not args.no_comfree:
        if args.skip_comfree and (cf_dir / "episodes.csv").exists():
            cf_recs = harness.read_episodes(cf_dir / "episodes.csv")
            print(f"[sim2sim] reusing {cf_dir/'episodes.csv'} ({len(cf_recs)} episodes)")
        else:
            t0 = time.time()
            cf_recs = run_eps(cf_vec)
            s = harness.summarize(cf_recs)
            eval_csv = args.run / "eval.csv"
            s["self_check"] = (harness.self_check(s, eval_csv, eval_prefix, ckpt_step=ckpt_step) if eval_csv.exists()
                               else {"available": False, "reason": "no eval.csv"})
            harness.write_episodes(cf_dir / "episodes.csv", cf_recs)
            _json(cf_dir / "summary.json", s)
            _json(cf_dir / "config.json", {**common, "backend": "comfree", "wall_s": time.time() - t0})
            sc = s["self_check"]
            if s["nonfinite_action_steps_total"]:
                print(f"[sim2sim] WARNING: agent emitted non-finite actions on "
                      f"{s['nonfinite_action_steps_total']} steps -- diverged checkpoint, gap is meaningless")
            print(f"[comfree] return {s['return']['mean']:.1f}  goals {s['goals_reached']['mean']:.2f}  "
                  f"drop {s['dropped']['mean']:.2f}  ({time.time()-t0:.0f}s)")
            if sc.get("available"):
                print(f"[self-check vs eval.csv step {sc['eval_csv_step']}] "
                      + "  ".join(f"{k}: {m['harness']:.2f} vs {m['eval_csv']:.2f}" + (f" (z={m['z']:.1f})" if "z" in m else "")
                                  for k, m in sc["metrics"].items())
                      + {True: "  OK", False: "  MISMATCH", None: f"  N/A (eval row step {sc['eval_csv_step']} != checkpoint step {sc.get('ckpt_step')})"}[sc["ok"]])

    # MuJoCo, once per solver config
    for solver in solvers:
        t0 = time.time()
        mj_env = make_mujoco_env(env_cfg, args.seed, space, u, solver)
        parity = check_parity(mj_env.sim, cf_vec.sim, mj_env.writer.written_fields)
        hold = keyframe_hold(mj_env, u)
        if not parity["ok"]:
            bad = [k for k, r in parity["fields"].items() if not r.get("ok", True)]
            raise SystemExit(f"model parity failed for {bad}")
        rec = FrameRecorder(args.video, f"MuJoCo[{solver.name}]") if args.video > 0 else None
        mj_recs = run_eps(mj_env, rec)
        d = out / solver.name
        harness.write_episodes(d / "episodes.csv", mj_recs)
        videos = rec.save(d / "videos") if rec else []
        summary = {"mujoco": harness.summarize(mj_recs)}
        if cf_recs is not None:
            summary["comfree"] = harness.summarize(cf_recs)
            summary["gap_mujoco_minus_comfree"] = harness.gap(cf_recs, mj_recs)
        _json(d / "summary.json", summary)
        _json(d / "config.json", {
            **common, "backend": "mujoco", "solver": solver.to_dict(),
            "model": model_solver_summary(mj_env.sim.mjm), "skipped_params": mj_env.writer.skipped,
            "parity": parity, "keyframe_hold": hold, "videos": videos, "wall_s": time.time() - t0,
        })
        m = summary["mujoco"]
        line = (f"[mujoco:{solver.name}] return {m['return']['mean']:.1f}  goals {m['goals_reached']['mean']:.2f}  "
                f"drop {m['dropped']['mean']:.2f}  hold={'ok' if hold['stable'] else 'UNSTABLE'}  "
                f"bad_qacc={m['bad_qacc_resets_total']}  ({time.time()-t0:.0f}s)")
        if cf_recs is not None:
            g = summary["gap_mujoco_minus_comfree"]
            line += (f"\n    gap: return {g['return']['diff']:+.1f} [{g['return']['ci95'][0]:+.1f}, "
                     f"{g['return']['ci95'][1]:+.1f}]  goals {g['goals_reached']['diff']:+.2f}  "
                     f"drop {g['dropped']['diff']:+.2f}")
        print(line, flush=True)
    print(f"[sim2sim] outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
