"""Does a better-distilled policy let MPPI plan with a smaller rollout budget?

Compares `outputs/vec/dagger_b30` (dagger_beta=0.3) against
`outputs/ablation/d_termq` (rung D).  Both were trained with the same
full-budget planner config: horizon=8, iterations=3, num_samples=256,
num_pi_trajs=24, num_elites=32, use_terminal_value=true.

Hypothesis: a better-distilled policy means MPPI's `num_pi_trajs=24`
policy-proposed candidates already start closer to optimal, so the planner
should need fewer total sampled action sequences (`iterations * num_samples`)
to reach the same goals/ep -- directly relevant to real-robot deployment,
where planning compute is budgeted per control step. `num_pi_trajs` is held
fixed at 24 across the sweep; only `iterations`/`num_samples` shrink around it.

Uses the vectorized path (`VecAllegroReorientEnv`, `ParamSpace.realistic()`,
`HandleWriter`, `BatchMPPIPlanner`) since that is what both checkpoints were
trained under.  The legacy single-env `AllegroReorientEnv`/`MPPIPlanner`/
`ParamVector` path (`ablate_controllers.py`) uses a different parameter space
and would silently misload a 71-parameter checkpoint.  Each checkpoint's
`AgentConfig` is rebuilt from its `config.json["agent"]` rather than
hardcoded, since `actor_mode` differs between the two runs.

    python scripts/report/ablate_rollout_budget.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.task import TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.logger import new_figure
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig, load_spec
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace

# (iterations, num_samples), full budget first, shrinking around a fixed
# num_pi_trajs=24.
BUDGETS = [
    (3, 256),
    (3, 128),
    (3, 64),
    (2, 64),
    (1, 64),
    (1, 32),
]


def load(run: Path, num_envs: int, horizon: int, iterations: int, num_samples: int,
         num_pi_trajs: int, seed: int):
    config = json.loads((run / "config.json").read_text())
    space = ParamSpace.realistic(load_spec())
    u_model = np.load(run / "params.npy") if (run / "params.npy").exists() else space.true_u
    u_real = (np.load(run / "params_reality.npy")
              if (run / "params_reality.npy").exists() else space.true_u)
    env = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=num_envs, episode_length=120, task=TaskConfig()),
        seed=seed, space=space, u=u_real)
    agent = TDMPCAgent(AgentConfig(**config["agent"]), env.obs_dim, env.action_dim,
                       env.torch_device)
    agent.load(str(run / "agent.pt"))
    cfg = PlannerConfig(horizon=horizon, iterations=iterations, num_samples=num_samples,
                        num_elites=min(32, num_samples), num_pi_trajs=min(num_pi_trajs, num_samples),
                        use_terminal_value=True, max_std=0.5, record_elites=False)
    sim = BatchSim(SimConfig(nworld=num_envs * num_samples))
    HandleWriter(sim, space).write_u(u_model)
    planner = BatchMPPIPlanner(sim, env.task, cfg, num_envs, agent=agent)
    return env, agent, planner


@torch.no_grad()
def evaluate(run: Path, num_envs: int, horizon: int, iterations: int, num_samples: int,
            num_pi_trajs: int, episodes: int, max_steps: int, seed: int) -> dict:
    env, agent, planner = load(run, num_envs, horizon, iterations, num_samples,
                               num_pi_trajs, seed)
    env.reset()
    planner.reset()
    plan_times = []
    t = 0
    while len(env.finished) < episodes and t < max_steps:
        qpos, qvel = env.state()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a, _, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                               t0=env.needs_t0.clone(), eval_mode=True)
        torch.cuda.synchronize()
        plan_times.append(time.perf_counter() - t0)
        env.step(a)
        t += 1
    finished = env.finished[:episodes]

    result = {
        "checkpoint": run.name,
        "iterations": iterations,
        "num_samples": num_samples,
        "num_pi_trajs": num_pi_trajs,
        "total_budget": iterations * num_samples,
        "episodes": len(finished),
        "control_steps": t,
        "goals_reached": float(np.mean([e["goals_reached"] for e in finished])) if finished else float("nan"),
        "dropped": float(np.mean([e["dropped"] for e in finished])) if finished else float("nan"),
        "return": float(np.mean([e["episode_reward"] for e in finished])) if finished else float("nan"),
        "plan_ms_mean": float(np.mean(plan_times) * 1e3),
        "plan_ms_std": float(np.std(plan_times) * 1e3),
    }
    del env, agent, planner
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs=2,
                    default=["outputs/vec/dagger_b30", "outputs/ablation/d_termq"])
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--num_pi_trajs", type=int, default=24)
    ap.add_argument("--episodes", type=int, default=32,
                    help="episodes to average per (checkpoint, budget) cell")
    ap.add_argument("--max_steps", type=int, default=480,
                    help="safety cap on control steps if `episodes` never fills")
    ap.add_argument("--seed", type=int, default=77, help="matched across both checkpoints")
    ap.add_argument("--out", type=str, default="outputs/rollout_budget")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'checkpoint':<20} {'iters':>5} {'samples':>7} {'goals/ep':>9} "
          f"{'dropped':>8} {'return':>9} {'plan ms':>9}")
    print("-" * 72)
    for run_str in args.runs:
        run = Path(run_str)
        for iterations, num_samples in BUDGETS:
            r = evaluate(run, args.num_envs, args.horizon, iterations, num_samples,
                        args.num_pi_trajs, args.episodes, args.max_steps, args.seed)
            rows.append(r)
            print(f"{run.name:<20} {iterations:>5} {num_samples:>7} "
                  f"{r['goals_reached']:>9.3f} {r['dropped']:>8.3f} "
                  f"{r['return']:>9.1f} {r['plan_ms_mean']:>9.2f}", flush=True)

    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/results.json")

    # figure: goals/ep and plan time vs. total sampled sequences
    fig, axes = new_figure(1, 2, figsize=(11, 4.2))
    for run_str, style in zip(args.runs, ("-o", "--s")):
        name = Path(run_str).name
        rs = [r for r in rows if r["checkpoint"] == name]
        total = [r["total_budget"] for r in rs]
        axes[0].plot(total, [r["goals_reached"] for r in rs], style, label=name)
        axes[1].plot(total, [r["plan_ms_mean"] for r in rs], style, label=name)
    for ax in axes:
        ax.set_xlabel("iterations x num_samples (total sequences / plan() call)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("goals reached / episode")
    axes[0].set_title("Task performance vs. rollout budget")
    axes[1].set_ylabel("plan() wall-clock (ms)")
    axes[1].set_title("Planning cost vs. rollout budget")
    fig.tight_layout()
    fig.savefig(out / "rollout_budget.png", dpi=140)
    print(f"wrote {out}/rollout_budget.png")


if __name__ == "__main__":
    main()
