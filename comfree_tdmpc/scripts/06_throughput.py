"""Where does the wall clock actually go?

Three measurements, in the order they matter for this project:

1. `BatchSim.step` against `nworld` -- if the curve is flat the GPU is idle at
   the planner's 256 worlds and running several environments at once is nearly
   free, which is the single biggest lever on data rate.
2. one `MPPIPlanner.plan` call, split into simulator time, policy-prior time and
   the python/torch glue around them.
3. one `TDMPCAgent.update`, to see whether learning or planning is the bottleneck.

    python scripts/06_throughput.py --nworlds 64 256 1024 4096
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from comfree_tdmpc.agent.tdmpc_agent import AgentConfig, TDMPCAgent
from comfree_tdmpc.envs.task import ReorientTask, TaskConfig
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig


def time_sim(nworld: int, n_substeps: int, steps: int = 40) -> dict:
    sim = BatchSim(SimConfig(nworld=nworld, n_substeps=n_substeps))
    sim.reset_to_key()
    action = torch.zeros(sim.nworld, sim.nu, device=sim.torch_device)
    for _ in range(5):  # warm up
        sim.set_action(action)
        sim.step()
    sim.sync()

    t0 = time.perf_counter()
    for _ in range(steps):
        sim.set_action(action)
        sim.step()
    sim.sync()
    dt = (time.perf_counter() - t0) / steps

    # Same loop without the launch, to separate GPU work from python overhead.
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.set_action(action)
    sim.sync()
    overhead = (time.perf_counter() - t0) / steps

    out = {
        "nworld": nworld,
        "ctrl_step_ms": dt * 1e3,
        "python_ms": overhead * 1e3,
        "phys_steps_per_s": nworld * n_substeps / dt,
        "ctrl_steps_per_s": nworld / dt,
        "finite": bool(torch.isfinite(sim.qpos).all().item()),
    }
    del sim
    torch.cuda.empty_cache()
    return out


def time_plan(nworld: int, horizon: int, iterations: int, num_pi: int,
              n_substeps: int, reps: int = 10) -> dict:
    """One MPPI plan call, with and without the learned pieces."""
    from comfree_tdmpc.planner.mppi import MPPIPlanner, PlannerConfig

    sim = BatchSim(SimConfig(nworld=nworld, n_substeps=n_substeps))
    task = ReorientTask(TaskConfig(), sim.torch_device, sim.control_dt,
                        cube_home=sim.key_qpos[16:19])
    agent = TDMPCAgent(AgentConfig(), task.obs_dim, sim.nu, sim.torch_device)
    cfg = PlannerConfig(horizon=horizon, iterations=iterations, num_samples=nworld,
                        num_pi_trajs=num_pi)
    planner = MPPIPlanner(sim, task, cfg, agent=agent)

    sim.reset_to_key()
    qpos, qvel = sim.qpos[0].clone(), sim.qvel[0].clone()
    goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.torch_device)
    prev = torch.zeros(1, sim.nu, device=sim.torch_device)

    def run(reps_):
        t0 = time.perf_counter()
        for _ in range(reps_):
            planner.plan(qpos, qvel, goal, prev, t0=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps_

    run(2)  # warm up
    full = run(reps)

    planner.agent = None  # no policy prior, no terminal value
    cfg.use_terminal_value = False
    run(2)
    sim_only = run(reps)

    del sim, planner, agent
    torch.cuda.empty_cache()
    return {
        "nworld": nworld,
        "plan_ms": full * 1e3,
        "plan_ms_sim_only": sim_only * 1e3,
        "learned_overhead_ms": (full - sim_only) * 1e3,
        "sim_steps_in_plan": horizon * iterations,
        "env_steps_per_s": 1.0 / full,
    }


def time_update(batch_size: int, reps: int = 50) -> dict:
    dev = torch.device("cuda")
    cfg = AgentConfig(batch_size=batch_size)
    agent = TDMPCAgent(cfg, 54, 16, dev)
    batch = {
        "obs": torch.randn(batch_size, 54, device=dev),
        "next_obs": torch.randn(batch_size, 54, device=dev),
        "action": torch.randn(batch_size, 16, device=dev).clamp(-1, 1),
        "mu": torch.randn(batch_size, 16, device=dev).clamp(-1, 1),
        "std": torch.full((batch_size, 16), 0.3, device=dev),
        "reward": torch.randn(batch_size, device=dev),
        "done": torch.zeros(batch_size, device=dev),
    }
    for _ in range(5):
        agent.update(batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        agent.update(batch)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    del agent
    torch.cuda.empty_cache()
    return {"batch_size": batch_size, "update_ms": dt * 1e3, "updates_per_s": 1.0 / dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworlds", type=int, nargs="+",
                    default=[1, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--n_substeps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--num_pi", type=int, default=24)
    ap.add_argument("--plan_worlds", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--out", type=str, default="outputs/06_throughput/throughput.json")
    args = ap.parse_args()

    report: dict[str, list] = {"sim": [], "plan": [], "update": []}

    print("=== simulator: one control step ===")
    print(f"{'nworld':>7} {'ctrl step ms':>13} {'python ms':>10} "
          f"{'phys steps/s':>13} {'ctrl steps/s':>13}")
    for n in args.nworlds:
        try:
            r = time_sim(n, args.n_substeps)
            report["sim"].append(r)
            print(f"{r['nworld']:>7} {r['ctrl_step_ms']:>13.2f} {r['python_ms']:>10.2f} "
                  f"{r['phys_steps_per_s']:>13.3e} {r['ctrl_steps_per_s']:>13.3e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{n:>7}  FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n=== planner: one plan() call (H={args.horizon}, "
          f"{args.iterations} iters) ===")
    print(f"{'nworld':>7} {'plan ms':>10} {'sim only':>10} {'learned':>9} {'env steps/s':>12}")
    for n in args.plan_worlds:
        try:
            r = time_plan(n, args.horizon, args.iterations, args.num_pi, args.n_substeps)
            report["plan"].append(r)
            print(f"{r['nworld']:>7} {r['plan_ms']:>10.1f} {r['plan_ms_sim_only']:>10.1f} "
                  f"{r['learned_overhead_ms']:>9.1f} {r['env_steps_per_s']:>12.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{n:>7}  FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n=== agent: one gradient update ===")
    print(f"{'batch':>7} {'update ms':>11} {'updates/s':>11}")
    for bs in (256, 512, 1024):
        r = time_update(bs)
        report["update"].append(r)
        print(f"{r['batch_size']:>7} {r['update_ms']:>11.2f} {r['updates_per_s']:>11.0f}",
              flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {p}")

    # The headline number: at the planner's own world count, how many environment
    # steps per second does one sequential environment produce?
    if report["plan"]:
        base = report["plan"][0]
        print(f"\none environment at {base['nworld']} worlds: "
              f"{base['env_steps_per_s']:.1f} env steps/s "
              f"({base['env_steps_per_s']*3600/1000:.0f}k steps/hour)")


if __name__ == "__main__":
    main()
