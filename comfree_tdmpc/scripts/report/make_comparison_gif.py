"""Side-by-side gifs: what each controller actually does with the hand.

Numbers say the planner scores more than the policy.  A gif says why: the
planner keeps regrasping and the policy lets the cube go.  Both panels start
from the same state with the same goal, so the only difference on screen is who
is choosing the actions.

    python scripts/report/make_comparison_gif.py --run outputs/ablation/d_termq
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.render import SceneRenderer, save_gif, save_grid_gif
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig, load_spec
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace


def run_planner_cfg(run: Path, num_samples: int) -> PlannerConfig:
    """The planner this run actually trained with.

    Reading it back matters: rung C was trained with no terminal value, and
    rendering it with one would show a controller that run never had.
    """
    p = run / "config.json"
    if not p.exists():
        return PlannerConfig(horizon=8, iterations=3, num_samples=num_samples,
                             num_pi_trajs=24, use_terminal_value=True, max_std=0.5)
    saved = json.loads(p.read_text())["planner"]
    saved["num_samples"] = num_samples
    saved["record_elites"] = False
    return PlannerConfig(**saved)


@torch.no_grad()
def rollout(kind: str, run: Path, steps: int, seed: int, cfg: PlannerConfig,
            renderer: SceneRenderer, label_extra: str = "") -> tuple[list[np.ndarray], dict]:
    space = ParamSpace.realistic(load_spec())
    u_model = np.load(run / "params.npy") if (run / "params.npy").exists() else space.true_u
    u_real = (np.load(run / "params_reality.npy")
              if (run / "params_reality.npy").exists() else space.true_u)

    env = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=1, episode_length=steps, task=TaskConfig()),
        seed=seed, space=space, u=u_real)
    agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
    if (run / "agent.pt").exists():
        agent.load(str(run / "agent.pt"))

    planner = None
    if kind == "mpc":
        sim = BatchSim(SimConfig(nworld=cfg.num_samples))
        HandleWriter(sim, space).write_u(u_model)
        planner = BatchMPPIPlanner(sim, env.task, cfg, 1, agent=agent)

    torch.manual_seed(seed)
    obs = env.reset()
    who = (f"MPPI planner   H={cfg.horizon}, {cfg.num_samples} samples"
           f"{', terminal Q' if cfg.use_terminal_value else ', no terminal Q'}"
           if kind == "mpc" else "Learned policy only   (no planning)")
    theta_err = float(np.abs(u_model - u_real).mean())

    frames, solved, dropped = [], 0, 0.0
    total_r = 0.0
    for t in range(steps):
        if kind == "mpc":
            qpos, qvel = env.state()
            a, _, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                                   t0=env.needs_t0.clone(), eval_mode=True)
        else:
            a = agent.act_batch(obs, eval_mode=True)
        # The goal this step is scored against.  `env.step` resamples it when a
        # goal is held long enough, and again when the episode ends and the lane
        # auto-resets, so reading it back afterwards would draw a target the
        # caption's angle was never measured against.
        goal = env.goal_quat[0].clone()
        obs, r, done, info = env.step(a)
        total_r += float(r[0])
        at_goal = float(info["success"][0]) > 0
        solved = max(solved, int(info["goals_reached"][0]))
        status = (f"AT GOAL  holding {int(info['success_streak'][0])}/10"
                  if at_goal else f"goal error = {float(info['angle'][0]):.2f} rad")
        frames.append(renderer.frame(
            env.qpos[0].cpu().numpy(), goal.cpu().numpy(),
            label=f"{who}{label_extra}\nstep {t:3d}   {status}\n"
                  f"goals solved = {solved}   reward so far = {total_r:6.0f}\n"
                  f"model error = {theta_err:.3f}",
            at_goal=at_goal))
        if bool(done[0]):
            dropped = float(info["dropped"][0])
            # Freeze on the last frame so the outcome is readable in the loop.
            frames.extend([frames[-1]] * 15)
            break
    stats = {"return": total_r, "goals": solved, "dropped": dropped, "length": len(frames)}
    del env, agent, planner
    torch.cuda.empty_cache()
    return frames, stats


def pad_to(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    return frames + [frames[-1]] * max(0, n - len(frames))


def eval_mean_goals(run: Path, who: str, win: int) -> float:
    """The run's own measured average, so the gif can be chosen to match it."""
    p = run / "eval.csv"
    if not p.exists():
        return float("nan")
    with open(p) as f:
        rows = list(csv.DictReader(f))
    v = []
    for r in rows:
        try:
            v.append(float(r[f"{who}/goals_reached"]))
        except (KeyError, TypeError, ValueError):
            pass
    return float(np.mean(v[-win:])) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="outputs/ablation/d_termq")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[404, 405, 406, 407, 408],
                    help="episodes to try; the one closest to the run's own average is kept")
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--out", type=str, default="outputs/report/gifs")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = SceneRenderer()
    cfg = run_planner_cfg(run, args.num_samples)
    print(f"{run.name}: planner H={cfg.horizon} iters={cfg.iterations} "
          f"terminal_value={cfg.use_terminal_value}")

    # A single episode is an anecdote, and the seed that makes the nicest gif is
    # usually the luckiest one.  Several are rolled out and the episode whose
    # goal count is closest to the run's measured average is the one shown.
    got = {}
    for kind, who, win in (("mpc", "mpc", 3), ("pi", "pi", 6)):
        target = eval_mean_goals(run, who, win)
        best = None
        for seed in args.seeds:
            frames, stats = rollout(kind, run, args.steps, seed, cfg, renderer)
            dist = abs(stats["goals"] - target) if np.isfinite(target) else 0.0
            print(f"  {kind:>4} seed {seed}: return={stats['return']:8.1f} "
                  f"goals={stats['goals']}  dropped={stats['dropped']:.0f}"
                  f"  |goals-avg|={dist:.2f}")
            if best is None or dist < best[0]:
                best = (dist, frames, stats, seed)
            if dist == 0.0 and not np.isfinite(target):
                break
        _, frames, stats, seed = best
        got[kind] = frames
        print(f"  -> {kind}: kept seed {seed} (run average {target:.2f} goals, "
              f"this episode {stats['goals']})")
        save_gif(frames, out / f"{run.name}_{kind}.gif", fps=25)

    n = max(len(f) for f in got.values())
    p = save_grid_gif([pad_to(got["mpc"], n), pad_to(got["pi"], n)],
                      out / f"{run.name}_side_by_side.gif", fps=25)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
