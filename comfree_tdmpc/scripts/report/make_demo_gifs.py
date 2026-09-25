"""Purpose-built demo clips, each answering one question.

  sysid_effect.gif      MPPI with the wrong physics  |  MPPI after SysID
  planner_vs_policy.gif MPPI planner                 |  distilled policy alone
  terminal_value.gif    short-horizon MPPI, no value |  with the learned value

Every clip is captioned with which controller is driving, so a planner rollout
can never be mistaken for a policy rollout.  Both panels of a clip start from the
same state and see the same goal sequence, so the comparison is like-for-like.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig
from comfree_tdmpc.render import SceneRenderer, save_gif
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import ParamVector


def run_episode(
    env: AllegroReorientEnv,
    renderer: SceneRenderer,
    caption: str,
    steps: int,
    planner: MPPIPlanner | None = None,
    agent: TDMPCAgent | None = None,
    seed: int = 0,
) -> tuple[list[np.ndarray], dict]:
    """One evaluation episode driven by either the planner or the policy alone."""
    torch.manual_seed(seed)
    env.rng.manual_seed(seed)
    obs = env.reset()
    if planner is not None:
        planner.reset()
    frames, t = [], 0
    while t < steps:
        if planner is not None:
            qpos, qvel = env.state()
            a, _, _ = planner.plan(
                qpos, qvel, env.goal_quat, env.prev_action, t0=(t == 0), eval_mode=True
            )
        else:
            a = agent.act(obs, eval_mode=True)
        obs, r, done, info = env.step(a)
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
                label=f"{caption}\nt={t:3d}  {status}  solved={info['goals_reached']}",
                at_goal=at_goal,
            )
        )
        t += 1
        if done:
            # Freeze on the failure so it is visible in the clip.
            frames += [frames[-1]] * 15
            break
    return frames, info


def side_by_side(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    n = max(len(a), len(b))
    a = a + [a[-1]] * (n - len(a))
    b = b + [b[-1]] * (n - len(b))
    return [np.concatenate([x, y], axis=1) for x, y in zip(a, b)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--sysid_run", type=str, default="outputs/m4_sysid")
    ap.add_argument("--policy_run", type=str, default="outputs/m5_distill")
    ap.add_argument("--shorth_run", type=str, default="outputs/m3_true")
    ap.add_argument("--wrong_params", type=float, nargs=4, default=[2.5, 0.35, 0.35, 0.008])
    ap.add_argument("--out", type=str, default="outputs/report")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = SceneRenderer(width=420, height=320)
    env_cfg = EnvConfig(episode_length=args.steps, task=TaskConfig())
    env = AllegroReorientEnv(env_cfg, ParamVector(), seed=args.seed)

    def make_planner(params: ParamVector, horizon: int, agent=None, terminal=True):
        sim = BatchSim(SimConfig(nworld=256, n_substeps=env_cfg.n_substeps))
        sim.set_params(params.value)
        cfg = PlannerConfig(
            horizon=horizon, iterations=4, num_samples=256, max_std=0.5,
            num_pi_trajs=24 if agent is not None else 0,
            use_terminal_value=terminal and agent is not None,
        )
        return MPPIPlanner(sim, env.task, cfg, agent=agent), sim

    # 1. what SysID buys the planner
    sysid_dir = Path(args.sysid_run)
    if (sysid_dir / "params.npy").exists():
        identified = ParamVector(np.load(sysid_dir / "params.npy"))
        wrong = ParamVector(args.wrong_params)
        print(f"wrong      {wrong}\nidentified {identified}")
        pl, sim = make_planner(wrong, 16)
        fa, ia = run_episode(env, renderer, "MPPI  |  WRONG physics (no SysID)",
                             args.steps, planner=pl, seed=args.seed)
        del pl, sim
        torch.cuda.empty_cache()
        pl, sim = make_planner(identified, 16)
        fb, ib = run_episode(env, renderer, "MPPI  |  physics after SysID",
                             args.steps, planner=pl, seed=args.seed)
        del pl, sim
        torch.cuda.empty_cache()
        save_gif(side_by_side(fa, fb), out / "demo_sysid_effect.gif", fps=25)
        print(f"sysid effect: goals {ia['goals_reached']} -> {ib['goals_reached']}, "
              f"dropped {ia['dropped']} -> {ib['dropped']}")

    # 2. planner vs the policy distilled from it
    pol_dir = Path(args.policy_run)
    if (pol_dir / "agent.pt").exists():
        agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
        agent.load(str(pol_dir / "agent.pt"))
        pl, sim = make_planner(ParamVector(), 16, agent=agent)
        fa, ia = run_episode(env, renderer, "MPPI planner (256 samples, H=16)",
                             args.steps, planner=pl, seed=args.seed)
        del pl, sim
        torch.cuda.empty_cache()
        fb, ib = run_episode(env, renderer, "LEARNED POLICY alone (no planning)",
                             args.steps, agent=agent, seed=args.seed)
        save_gif(side_by_side(fa, fb), out / "demo_planner_vs_policy.gif", fps=25)
        print(f"planner goals {ia['goals_reached']} vs policy goals {ib['goals_reached']}")

    # 3. terminal value on a short-horizon planner
    sh_dir = Path(args.shorth_run)
    if (sh_dir / "agent.pt").exists():
        agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
        agent.load(str(sh_dir / "agent.pt"))
        pl, sim = make_planner(ParamVector(), 6, agent=agent, terminal=False)
        fa, ia = run_episode(env, renderer, "MPPI H=6  |  NO terminal value",
                             args.steps, planner=pl, seed=args.seed)
        del pl, sim
        torch.cuda.empty_cache()
        pl, sim = make_planner(ParamVector(), 6, agent=agent, terminal=True)
        fb, ib = run_episode(env, renderer, "MPPI H=6  |  + learned terminal Q",
                             args.steps, planner=pl, seed=args.seed)
        del pl, sim
        torch.cuda.empty_cache()
        save_gif(side_by_side(fa, fb), out / "demo_terminal_value.gif", fps=25)
        print(f"short horizon goals {ia['goals_reached']} -> {ib['goals_reached']}")

    print(f"demo clips in {out}")


if __name__ == "__main__":
    main()
