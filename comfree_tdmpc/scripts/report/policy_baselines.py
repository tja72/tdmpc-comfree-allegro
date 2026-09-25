"""Is the learned policy actually doing anything?

A return of -300 means nothing on its own.  This measures the trivial baselines
on the same task and seeds -- hold the initial posture, act randomly -- so the
learned policy can be placed against "do nothing" rather than against zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.allegro import AllegroReorientEnv, EnvConfig
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.sim.params import ParamVector


@torch.no_grad()
def evaluate(env, act_fn, episodes: int, steps: int, seeds: list[int]) -> dict:
    rets, goals, drops, lens = [], [], [], []
    for s in seeds[:episodes]:
        torch.manual_seed(s)
        env.rng.manual_seed(s)
        obs = env.reset()
        hold = env.prev_action.clone().squeeze(0)
        R, t, a = 0.0, 0, hold.clone()
        while t < steps:
            a = act_fn(obs, hold, a)
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
        "return": float(np.mean(rets)), "std": float(np.std(rets)),
        "goals": float(np.mean(goals)), "dropped": float(np.mean(drops)),
        "length": float(np.mean(lens)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="+", default=["outputs/runs/distill_on"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    seeds = list(range(200, 200 + args.episodes))
    env = AllegroReorientEnv(EnvConfig(episode_length=args.steps, task=TaskConfig()),
                             ParamVector(), seed=0)

    print(f"{'controller':<34} {'return':>9} {'+/-':>7} {'goals':>6} {'drop':>5} {'len':>5}")

    def row(name, res):
        print(f"{name:<34} {res['return']:>9.1f} {res['std']:>7.1f} {res['goals']:>6.2f} "
              f"{res['dropped']:>5.2f} {res['length']:>5.0f}", flush=True)

    # Hold the posture the scene starts in: the cube stays in the hand and
    # nothing is attempted.  This is the floor any policy must beat.
    row("hold initial posture", evaluate(env, lambda o, h, a: h, args.episodes, args.steps, seeds))
    row("zero action (mid ctrl range)",
        evaluate(env, lambda o, h, a: torch.zeros_like(h), args.episodes, args.steps, seeds))

    g = torch.Generator(device=env.torch_device).manual_seed(0)
    row("smoothed random actions", evaluate(
        env, lambda o, h, a: (0.8 * a + 0.35 * torch.randn(
            a.shape, device=a.device, generator=g)).clamp(-1, 1),
        args.episodes, args.steps, seeds))

    for run in args.runs:
        p = Path(run) / "agent.pt"
        if not p.exists():
            print(f"{run}: no checkpoint yet")
            continue
        agent = TDMPCAgent(AgentConfig(), env.obs_dim, env.action_dim, env.torch_device)
        agent.load(str(p))
        row(f"learned policy ({Path(run).name})",
            evaluate(env, lambda o, h, a: agent.act(o, eval_mode=True),
                     args.episodes, args.steps, seeds))


if __name__ == "__main__":
    main()
