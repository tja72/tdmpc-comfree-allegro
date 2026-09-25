"""Side-by-side rollout of the same actions under three parameter settings.

Left: the real environment (theta*).  Middle: the planner's model before SysID.
Right: the planner's model after SysID.  All three are driven by the *same*
action sequence from the same initial state, so the visible divergence is exactly
the model error that SysID removes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from comfree_tdmpc.render import SceneRenderer, save_gif
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import ParamVector


def rollout(params: ParamVector, actions: torch.Tensor, qpos0, qvel0) -> np.ndarray:
    sim = BatchSim(SimConfig(nworld=1, n_substeps=10))
    sim.set_params(params.value)
    sim.set_state(
        torch.as_tensor(qpos0, dtype=torch.float32, device=sim.torch_device),
        torch.as_tensor(qvel0, dtype=torch.float32, device=sim.torch_device),
    )
    traj = []
    for t in range(actions.shape[0]):
        sim.set_action(actions[t].to(sim.torch_device))
        sim.step()
        traj.append(sim.qpos[0].cpu().numpy().copy())
    del sim
    torch.cuda.empty_cache()
    return np.stack(traj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="outputs/m4_sysid")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--true_params", type=float, nargs=4, default=[1.0, 1.0, 0.1, 1e-3])
    ap.add_argument("--init_params", type=float, nargs=4, default=[2.5, 0.35, 0.35, 0.008])
    ap.add_argument("--out", type=str, default="outputs/report/model_comparison.gif")
    args = ap.parse_args()

    identified = ParamVector(np.load(Path(args.run) / "params.npy"))
    true_p, init_p = ParamVector(args.true_params), ParamVector(args.init_params)
    print(f"true       {true_p}\ninitial    {init_p}\nidentified {identified}")

    # Excite the hand with a smooth action sequence from the scene keyframe.
    probe = BatchSim(SimConfig(nworld=1, n_substeps=10))
    qpos0, qvel0 = probe.key_qpos.copy(), probe.key_qvel.copy()
    a0 = probe.ctrl_to_action(
        torch.as_tensor(probe.key_ctrl, dtype=torch.float32, device=probe.torch_device)
    )
    del probe
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    actions = torch.zeros(args.steps, 1, a0.numel())
    a = a0.cpu().clone()
    for t in range(args.steps):
        a = (0.8 * a + 0.35 * torch.randn_like(a)).clamp(-1, 1)
        actions[t, 0] = a

    trajs = {
        "real (theta*)": rollout(true_p, actions, qpos0, qvel0),
        "model before SysID": rollout(init_p, actions, qpos0, qvel0),
        "model after SysID": rollout(identified, actions, qpos0, qvel0),
    }

    ref = trajs["real (theta*)"]
    errs = {}
    for k, tr in trajs.items():
        errs[k] = np.linalg.norm(tr[:, 16:19] - ref[:, 16:19], axis=-1)
        print(f"{k:22s} final cube position error {errs[k][-1]*100:.2f} cm")

    renderer = SceneRenderer(width=380, height=300, show_goal=False)
    frames = []
    for t in range(args.steps):
        row = []
        for k, tr in trajs.items():
            row.append(
                renderer.frame(tr[t], label=f"{k}\nt={t}  cube err={errs[k][t]*100:5.2f} cm")
            )
        frames.append(np.concatenate(row, axis=1))
    out = Path(args.out)
    save_gif(frames, out, fps=20)
    print(f"wrote {out}")

    from comfree_tdmpc.logger import new_figure

    fig, ax = new_figure(figsize=(6, 3.2))
    for k, e in errs.items():
        if k.startswith("real"):
            continue
        ax.plot(np.arange(len(e)) * 0.02, e * 100, label=k)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cube position error vs. real (cm)")
    ax.set_title("Open-loop prediction error of the world model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.parent / "model_error.png", dpi=140)
    print(f"wrote {out.parent}/model_error.png")


if __name__ == "__main__":
    main()
