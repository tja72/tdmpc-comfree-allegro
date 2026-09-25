"""What is the planner's cheapest configuration that still solves the task?

Every environment step in training pays for one `plan()` call, so the planner's
cost is the training loop's budget.  The question is not "which configuration
plans best" but "which configuration plans best *per second*" -- time saved here
is time the learner can spend on gradient steps, and the pilot run showed the
learner idle 90% of the time.

Configurations are scored with no agent attached (no terminal value, no policy
prior) on the true physics, one axis at a time around a base setting, using the
vectorised environment so 16 episodes run at once.

    python scripts/08_tune_planner_vec.py --steps 240
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.envs.task import ReorientTask, TaskConfig
from comfree_tdmpc.envs.vec_allegro import VecAllegroReorientEnv, VecEnvConfig
from comfree_tdmpc.planner.batch_mppi import BatchMPPIPlanner
from comfree_tdmpc.planner.mppi import PlannerConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig

BASE = dict(horizon=8, iterations=4, num_samples=256, num_elites=32,
            temperature=0.5, noise_beta=2.0, max_std=0.5, min_std=0.05,
            num_pi_trajs=0, use_terminal_value=False)

AXES: dict[str, list] = {
    "iterations": [1, 2, 3, 4],
    "num_samples": [128, 256, 512],
    "horizon": [4, 6, 8, 12],
}


class BankPool:
    """One world bank per size, reused across configurations."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._banks: dict[int, BatchSim] = {}

    def get(self, num_samples: int) -> BatchSim:
        n = self.num_envs * num_samples
        if n not in self._banks:
            self._banks[n] = BatchSim(SimConfig(nworld=n))
        return self._banks[n]


@torch.no_grad()
def score(cfg_kwargs: dict, env: VecAllegroReorientEnv, pool: BankPool,
          steps: int, seed: int) -> dict:
    cfg = PlannerConfig(**cfg_kwargs)
    sim = pool.get(cfg.num_samples)
    task = ReorientTask(TaskConfig(), sim.torch_device, sim.control_dt,
                        cube_home=sim.key_qpos[16:19])
    planner = BatchMPPIPlanner(sim, task, cfg, env.num_envs)

    torch.manual_seed(seed)
    env.rng.manual_seed(seed)
    env.reset()

    # Warm up: the first call after a bank switch pays kernel/graph setup.
    for _ in range(2):
        qpos, qvel = env.state()
        a, _, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                               t0=env.needs_t0.clone(), eval_mode=True)
        env.step(a)
    env.reset()
    env.pop_finished()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    angles = []
    for _ in range(steps):
        qpos, qvel = env.state()
        a, _, _ = planner.plan(qpos, qvel, env.goal_quat, env.prev_action,
                               t0=env.needs_t0.clone(), eval_mode=True)
        _, _, _, info = env.step(a)
        angles.append(float(info["angle"].mean()))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    eps = env.pop_finished()
    n_env_steps = steps * env.num_envs
    goals = sum(e["goals_reached"] for e in eps)
    # Goals are counted over completed episodes only; normalise by the steps
    # those episodes actually consumed so partial episodes do not distort it.
    used = sum(e["episode_length"] for e in eps) or 1
    return {
        **{k: cfg_kwargs[k] for k in ("horizon", "iterations", "num_samples")},
        "goals_per_1k_steps": 1000.0 * goals / used,
        "drop_rate": float(np.mean([e["dropped"] for e in eps])) if eps else float("nan"),
        "episodes": len(eps),
        "mean_angle": float(np.mean(angles)),
        "ms_per_env_step": 1e3 * wall / n_env_steps,
        "env_steps_per_s": n_env_steps / wall,
        "goals_per_gpu_min": 60.0 * goals / wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/08_tune/planner.json")
    args = ap.parse_args()

    env = VecAllegroReorientEnv(
        VecEnvConfig(num_envs=args.num_envs, episode_length=120, task=TaskConfig()),
        seed=args.seed)
    pool = BankPool(args.num_envs)

    configs = [("base", dict(BASE))]
    for axis, values in AXES.items():
        for v in values:
            if v == BASE[axis]:
                continue
            configs.append((f"{axis}={v}", {**BASE, axis: v}))

    print(f"{'config':<18} {'H':>3} {'it':>3} {'N':>5} {'goals/1k':>9} "
          f"{'drop':>6} {'angle':>6} {'ms/step':>8} {'steps/s':>8} {'goals/gpu-min':>14}")
    rows = []
    for name, kw in configs:
        try:
            r = score(kw, env, pool, args.steps, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<18} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        r["config"] = name
        rows.append(r)
        print(f"{name:<18} {r['horizon']:>3} {r['iterations']:>3} {r['num_samples']:>5} "
              f"{r['goals_per_1k_steps']:>9.2f} {r['drop_rate']:>6.2f} {r['mean_angle']:>6.2f} "
              f"{r['ms_per_env_step']:>8.1f} {r['env_steps_per_s']:>8.1f} "
              f"{r['goals_per_gpu_min']:>14.1f}", flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {p}")
    if rows:
        best = max(rows, key=lambda r: r["goals_per_gpu_min"])
        print(f"best goals per GPU-minute: {best['config']} "
              f"({best['goals_per_gpu_min']:.1f}, quality {best['goals_per_1k_steps']:.2f}/1k)")


if __name__ == "__main__":
    main()
