"""Does the learned value actually buy anything, and how close is the policy?

Training curves cannot answer this: they mix the planner, the value, the policy
and the physics into one number.  This script freezes one trained checkpoint and
evaluates controllers built from its pieces on matched seeds:

  * MPPI at several horizons, with and without the terminal value
  * the policy alone, with no planning at all
  * MPPI without the policy-prior trajectories

The terminal value's job is to stand in for lookahead the planner does not have,
so the place to look for it is at short horizons: if `gamma^H Q(s_H)` is doing
its job, MPPI at H=3 with the value should approach MPPI at H=12 without it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.logger import new_figure
from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import ParamVector


@torch.no_grad()
def evaluate(env, planner, agent, episodes, steps, seeds) -> dict:
    rets, goals, drops, lens = [], [], [], []
    for s in seeds[:episodes]:
        torch.manual_seed(s)
        env.rng.manual_seed(s)
        obs = env.reset()
        if planner is not None:
            planner.reset()
        R, t = 0.0, 0
        while t < steps:
            if planner is not None:
                qpos, qvel = env.state()
                a, _, _ = planner.plan(
                    qpos, qvel, env.goal_quat, env.prev_action, t0=(t == 0), eval_mode=True
                )
            else:
                a = agent.act(obs, eval_mode=True)
            obs, r, done, info = env.step(a)
            R += r
            t += 1
            if done:
                break
        rets.append(R)
        goals.append(info["goals_reached"])
        drops.append(info["dropped"])
        lens.append(t)
    return {
        "return": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "goals": float(np.mean(goals)),
        "dropped": float(np.mean(drops)),
        "length": float(np.mean(lens)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="outputs/m4_sysid")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 12])
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--use_run_params", action="store_true",
                    help="plan with the parameters SysID identified in that run")
    ap.add_argument("--out", type=str, default="outputs/report")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(100, 100 + args.episodes))

    env_cfg = EnvConfig(episode_length=args.steps, task=TaskConfig())
    env = AllegroReorientEnv(env_cfg, ParamVector(), seed=0)
    agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
    agent.load(str(run / "agent.pt"))
    print(f"loaded {run}/agent.pt")

    params = ParamVector()
    if args.use_run_params and (run / "params.npy").exists():
        params = ParamVector(np.load(run / "params.npy"))
    print(f"planner physics: {params}")

    sim = BatchSim(SimConfig(nworld=args.num_samples, n_substeps=env_cfg.n_substeps))
    sim.set_params(params.value)

    def planner(horizon, terminal, num_pi=24):
        cfg = PlannerConfig(
            horizon=horizon, iterations=4, num_samples=args.num_samples,
            max_std=0.5, noise_beta=2.0, num_pi_trajs=num_pi,
            use_terminal_value=terminal,
        )
        return MPPIPlanner(sim, env.task, cfg, agent=agent)

    results: dict[str, dict] = {}

    print(f"\n{'controller':<38} {'return':>9} {'+/-':>7} {'goals':>6} {'drop':>5} {'len':>5}")

    def report(name, res):
        results[name] = res
        print(f"{name:<38} {res['return']:>9.1f} {res['return_std']:>7.1f} "
              f"{res['goals']:>6.2f} {res['dropped']:>5.2f} {res['length']:>5.0f}", flush=True)

    for h in args.horizons:
        for terminal in (False, True):
            name = f"MPPI H={h:<2} {'+ terminal Q' if terminal else 'no value   '}"
            report(name, evaluate(env, planner(h, terminal), agent, args.episodes, args.steps, seeds))

    report("policy only (no planning)",
           evaluate(env, None, agent, args.episodes, args.steps, seeds))
    h = max(args.horizons)
    report(f"MPPI H={h} + Q, no policy prior",
           evaluate(env, planner(h, True, num_pi=0), agent, args.episodes, args.steps, seeds))

    (out / "ablation.json").write_text(json.dumps(results, indent=2))

    # figure: value vs horizon
    fig, axes = new_figure(1, 2, figsize=(11, 3.6))
    hs = args.horizons
    for terminal, style in ((False, "--o"), (True, "-o")):
        key = lambda h: f"MPPI H={h:<2} {'+ terminal Q' if terminal else 'no value   '}"  # noqa: E731
        axes[0].plot(hs, [results[key(h)]["return"] for h in hs], style,
                     label="with terminal Q" if terminal else "no terminal value")
        axes[1].plot(hs, [results[key(h)]["goals"] for h in hs], style,
                     label="with terminal Q" if terminal else "no terminal value")
    pi = results["policy only (no planning)"]
    for ax, k in ((axes[0], "return"), (axes[1], "goals")):
        ax.axhline(pi[k], color="tab:red", ls=":", lw=1.5, label="policy alone (no planning)")
        ax.set_xlabel("planning horizon (control steps)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("episode return")
    axes[0].set_title("Terminal value substitutes for planning horizon")
    axes[1].set_ylabel("goals reached")
    axes[1].set_title("Task progress")
    fig.tight_layout()
    fig.savefig(out / "fig7_controller_ablation.png", dpi=140)
    print(f"\nwrote {out}/fig7_controller_ablation.png")


if __name__ == "__main__":
    main()
