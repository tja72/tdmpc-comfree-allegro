"""M0: does the stack run at all, and how fast?

Checks the ComFree bank steps, per-world parameters take effect, state injection
is exact, and offscreen rendering works.  Prints a throughput table used to size
the planner.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import PARAM_NAMES, ParamVector


def bench(nworld: int, n_substeps: int, steps: int = 30) -> dict:
    sim = BatchSim(SimConfig(nworld=nworld, n_substeps=n_substeps))
    sim.set_params(ParamVector().value)
    sim.reset_to_key()
    sim.step()
    sim.sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    sim.sync()
    dt = (time.perf_counter() - t0) / steps
    out = {
        "nworld": nworld,
        "control_step_ms": dt * 1e3,
        "sim_steps_per_s": nworld * n_substeps / dt,
        "cube_z": float(sim.qpos[0, 18].item()),
        "finite": bool(torch.isfinite(sim.qpos).all().item()),
    }
    del sim
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworlds", type=int, nargs="+", default=[1, 64, 128, 256, 512])
    ap.add_argument("--n_substeps", type=int, default=10)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    print("=== model ===")
    sim = BatchSim(SimConfig(nworld=8, n_substeps=args.n_substeps))
    print(
        f"nq={sim.nq} nv={sim.nv} nu={sim.nu} nbody={sim.mjm.nbody} ngeom={sim.mjm.ngeom} "
        f"timestep={sim.mjm.opt.timestep} control_dt={sim.control_dt:.4f}"
    )
    print(f"cube body={sim.cube_body_id} geom={sim.cube_geom_id} qpos_adr={sim.cube_qpos_adr}")
    print(f"friction geoms: {len(sim.friction_geom_ids)}")
    print(f"key qpos cube: {sim.key_qpos[16:19]}")

    print("\n=== per-world parameters take effect ===")
    theta = torch.tensor(ParamVector().value, dtype=torch.float32, device=sim.torch_device)
    thetas = theta.unsqueeze(0).repeat(8, 1)
    thetas[:, PARAM_NAMES.index("cube_mass_scale")] = torch.linspace(0.3, 3.0, 8, device=thetas.device)
    sim.set_params(thetas)
    print("body_mass per world:", sim.body_mass[:, sim.cube_body_id].tolist())
    sim.reset_to_key()
    for _ in range(20):
        sim.set_action(torch.zeros(sim.nu, device=sim.torch_device))
        sim.step()
    sim.sync()
    spread = sim.qpos[:, 16:19].std(dim=0).sum().item()
    print(f"cube position spread across mass settings after 20 control steps: {spread:.3e}")
    assert spread > 0, "per-world parameters had no effect"

    print("\n=== state injection is exact ===")
    sim.reset_to_key()
    for _ in range(5):
        sim.set_action(torch.zeros(sim.nu, device=sim.torch_device))
        sim.step()
    qpos_a, qvel_a = sim.get_state()
    sim.set_state(qpos_a[0], qvel_a[0])
    sim.set_action(torch.zeros(sim.nu, device=sim.torch_device))
    sim.step()
    qpos_b = sim.qpos.clone()
    print(f"max deviation across worlds after re-injection: {(qpos_b - qpos_b[0]).abs().max().item():.3e}")
    del sim
    torch.cuda.empty_cache()

    print("\n=== throughput ===")
    print(f"{'nworld':>8} {'ctrl step (ms)':>16} {'sim steps/s':>14} {'finite':>8}")
    for n in args.nworlds:
        try:
            r = bench(n, args.n_substeps)
            print(
                f"{r['nworld']:>8} {r['control_step_ms']:>16.2f} {r['sim_steps_per_s']:>14.3e} {str(r['finite']):>8}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"{n:>8}  FAILED: {type(e).__name__}: {e}")

    if args.render:
        print("\n=== render ===")
        from comfree_tdmpc.render import SceneRenderer, save_gif

        sim = BatchSim(SimConfig(nworld=1, n_substeps=args.n_substeps))
        sim.set_params(ParamVector().value)
        sim.reset_to_key()
        r = SceneRenderer()
        frames = []
        for t in range(60):
            a = 0.3 * np.sin(0.15 * t) * np.ones(sim.nu, dtype=np.float32)
            sim.set_action(torch.as_tensor(a, device=sim.torch_device))
            sim.step()
            frames.append(
                r.frame(sim.qpos[0].cpu().numpy(), label=f"passive rollout t={t}")
            )
        p = save_gif(frames, "outputs/00_smoke/passive_rollout.gif", fps=20)
        print(f"wrote {p} ({len(frames)} frames, {frames[0].shape})")


if __name__ == "__main__":
    main()
