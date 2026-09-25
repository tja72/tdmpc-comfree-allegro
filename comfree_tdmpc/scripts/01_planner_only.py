"""M1: MPPI planning through ComFree, with no learning at all.

If the simulator cannot serve as the world model for a sampling planner, nothing
downstream matters -- so this runs the planner with the true parameters, no value
function and no policy prior, and reports how much of the cube reorientation it
solves on its own.  It also renders the resulting trajectory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import ParamVector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--episode_length", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--num_elites", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_std", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/01_planner")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    env_cfg = EnvConfig(episode_length=args.episode_length, task=TaskConfig())
    env = AllegroReorientEnv(env_cfg, ParamVector(), seed=args.seed)

    plan_cfg = PlannerConfig(
        horizon=args.horizon,
        iterations=args.iterations,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_pi_trajs=0,
        temperature=args.temperature,
        max_std=args.max_std,
        use_terminal_value=False,
    )
    plan_sim = BatchSim(
        SimConfig(nworld=args.num_samples, n_substeps=env_cfg.n_substeps)
    )
    plan_sim.set_params(ParamVector().value)
    planner = MPPIPlanner(plan_sim, env.task, plan_cfg, agent=None)

    renderer = None
    if not args.no_render:
        from comfree_tdmpc.render import SceneRenderer, save_gif

        renderer = SceneRenderer()

    all_stats = []
    for ep in range(args.episodes):
        env.reset()
        planner.reset()
        frames, angles, rewards = [], [], []
        t0 = time.perf_counter()
        for t in range(args.episode_length):
            qpos, qvel = env.state()
            action, _, _ = planner.plan(
                qpos, qvel, env.goal_quat, env.prev_action, t0=(t == 0), eval_mode=True
            )
            _, reward, done, info = env.step(action)
            angles.append(info["angle"])
            rewards.append(reward)
            if renderer is not None:
                at_goal = info["success"] > 0
                status = (
                    f"AT GOAL  holding {info['success_streak']}/{info['hold_steps']}"
                    if at_goal
                    else f"goal err={info['angle']:.2f} rad"
                )
                frames.append(
                    renderer.frame(
                        env.qpos[0].cpu().numpy(),
                        env.goal_quat[0].cpu().numpy(),
                        label=f"MPPI planner (no learning)   t={t:3d}\n"
                        f"{status}   solved={info['goals_reached']}",
                        at_goal=at_goal,
                    )
                )
            if done:
                break
        dt = time.perf_counter() - t0
        stats = {
            "episode": ep,
            "steps": t + 1,
            "return": float(np.sum(rewards)),
            "goals_reached": info["goals_reached"],
            "final_angle": angles[-1],
            "min_angle": float(np.min(angles)),
            "mean_angle": float(np.mean(angles)),
            "dropped": info["dropped"],
            "sec_per_step": dt / (t + 1),
        }
        all_stats.append(stats)
        print(
            f"[ep {ep}] steps={stats['steps']:3d} return={stats['return']:8.1f} "
            f"solved={stats['goals_reached']} min_err={stats['min_angle']:.3f} "
            f"final_err={stats['final_angle']:.3f} dropped={stats['dropped']:.0f} "
            f"({stats['sec_per_step']*1e3:.0f} ms/step)"
        )
        if renderer is not None and frames:
            save_gif(frames, out / f"ep{ep}.gif", fps=25)

        np.save(out / f"ep{ep}_angles.npy", np.array(angles))

    solved = float(np.mean([s["goals_reached"] for s in all_stats]))
    print(f"\nmean goals solved per episode: {solved:.2f}")
    print(f"mean return: {np.mean([s['return'] for s in all_stats]):.1f}")

    # Angle-vs-time plot, one line per episode.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for ep in range(len(all_stats)):
        a = np.load(out / f"ep{ep}_angles.npy")
        ax.plot(a, label=f"episode {ep}")
    ax.axhline(env_cfg.task.success_angle, ls="--", c="k", lw=1, label="success threshold")
    ax.set_xlabel("control step")
    ax.set_ylabel("goal orientation error (rad)")
    ax.set_title("MPPI through ComFree, no learning")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "angle_vs_time.png", dpi=140)
    print(f"wrote {out}/angle_vs_time.png")


if __name__ == "__main__":
    main()
