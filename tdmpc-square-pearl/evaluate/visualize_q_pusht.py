import os
import sys

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from termcolor import colored

from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.envs import make_env
from tdmpc_square.tdmpc_square import TDMPC2
from tdmpc_square.envs.pusht import PIXEL_SCALE

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def q_action_grid(agent, z, task, action_low, action_high, grid_size, return_type):
    """
    Evaluates Q(z, a) for a dense grid spanning the full 2D action space.
    pusht's action *is* a pixel coordinate (the PD-controller target), so sweeping
    the whole action space is equivalent to sweeping the whole image space.

    Returns:
        q_grid (np.ndarray): (grid_size, grid_size) array of Q-values.
        xs, ys (np.ndarray): the two 1D action-space axes the grid was built from.
    """
    xs = np.linspace(action_low[0], action_high[0], grid_size)
    ys = np.linspace(action_low[1], action_high[1], grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    actions = torch.tensor(
        np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1),
        dtype=torch.float32,
        device=agent.device,
    )
    z_batch = z.repeat(actions.shape[0], 1)
    q = agent.model.Q(z_batch, actions, task, return_type=return_type)
    q_grid = q.squeeze(-1).reshape(grid_size, grid_size).cpu().numpy()
    return q_grid, xs, ys

def action_to_pixel(action, action_low, action_high, overlay_extent):
    left, right, bottom, top = overlay_extent
    frac = (np.asarray(action) - action_low) / (action_high - action_low)
    px = left + frac[0] * (right-left)
    py = top + frac[1] * (bottom-top)
    return px, py

def plot_q_heatmap(frame, q_grid, action_low, action_high, action_taken, overlay_extent, save_path,
                   vmin=None, vmax=None, cmap='turbo'):
    """Renders the env frame next to (and overlaid with) the Q-value heatmap.

    `overlay_extent` places the heatmap onto the rendered frame in pixel
    coordinates: with absolute pixel-target actions the swept grid covers the
    whole frame; with delta actions it only covers a small window around the
    agent's current position, since the grid sweeps relative displacements.
    """
    extent = (action_low[0], action_high[0], action_high[1], action_low[1])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(frame)
    axes[0].set_title("Observation")
    axes[0].axis("off")

    im = axes[1].imshow(q_grid, extent=extent, origin="upper", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("Q(s, a) over action space")
    axes[1].set_xlabel("action[0]")
    axes[1].set_ylabel("action[1]")
    if action_taken is not None:
        axes[1].scatter(*action_taken, c="red", marker="x", s=80, label="current action")
        axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    h, w = frame.shape[:2]
    axes[2].imshow(frame, extent=(0, w, h, 0))
    axes[2].imshow(
        q_grid,
        extent=overlay_extent,
        origin="upper",
        cmap=cmap,
        alpha=0.5,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    if action_taken is not None:
        px, py = action_to_pixel(action_taken, action_low, action_high, overlay_extent)
        axes[2].scatter(px, py, c="red", marker="x", s=80, label="current action")
        axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Q overlaid on observation")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@hydra.main(config_name="config", config_path="../tdmpc_square/tdmpc_square")
def visualize(cfg: dict):
    """
    Script for visualizing the learned Q-function of a trained TD-MPC2 agent on the
    `pusht` task.

    In pusht, an action is either a pixel coordinate (the PD-controller's
    absolute target position) or, if `delta_actions` is set, a displacement
    relative to the agent's current position. Either way it's a 2D space, so for
    any fixed state we can sweep the entire action space and render Q(s, a) as a
    heatmap directly over the scene -- showing which pixel (absolute mode) or
    which direction/distance (delta mode) the agent's critic currently considers
    best.

    Most relevant args:
            `checkpoint`: path to model checkpoint to load
            `q_grid_size`: resolution of the action-space sweep (default: 64)
            `q_return_type`: which Q reduction to visualize: min/avg/max (default: avg)
            `viz_every`: save a heatmap frame every N env steps (default: 5)
            `seed`: random seed (default: 1)

    Example usage:
    ```
            $ python evaluate/visualize_q_pusht.py task=pusht checkpoint=logs/pusht/0/<exp_name>/models/final.pt
    ```
    """
    assert torch.cuda.is_available()
    assert "pusht" in cfg.task, f"Expected a pusht task, got task={cfg.task}."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))

    env = make_env(cfg)
    assert cfg.action_dim == 2, (
        "This visualization sweeps a 2D action space; "
        f"got action_dim={cfg.action_dim}."
    )
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    delta_mode = cfg.get("delta_actions", False)
    delta_max = cfg.get("delta_action_max", 50.0)
    if delta_mode:
        print(colored(f"Delta actions enabled (max_delta={delta_max} px)", "blue"))
    path = cfg.checkpoint.rsplit('/', 2)[0]
    out_dir = os.path.join(path, "q_viz")
    os.makedirs(out_dir, exist_ok=True)

    print(colored("Rolling out trained agent...", "yellow", attrs=["bold"]))
    obs, done, t = env.reset()[0], False, 0
    snapshots = []
    while not done:
        frame = env.render()
        action, _, _ = agent.act(obs, t0=t == 0, eval_mode=True)

        if t % cfg.viz_every == 0 or t < 20:
            z = agent.model.encode(obs.to(agent.device).unsqueeze(0), task=None)
            q_grid, _, _ = q_action_grid(
                agent, z, None, action_low, action_high, cfg.q_grid_size, cfg.q_return_type
            )
            h, w = frame.shape[:2]
            if delta_mode:
                # the swept grid covers a displacement, not an absolute pixel
                # coordinate, so place it in a window around the agent's position
                agent_px = obs[:2].numpy() * PIXEL_SCALE
                scale_x, scale_y = w / PIXEL_SCALE, h / PIXEL_SCALE
                agent_render = agent_px[0] * scale_x, agent_px[1] * scale_y
                delta_render = delta_max * scale_x, delta_max * scale_y
                overlay_extent = (
                    agent_render[0] - delta_render[0], agent_render[0] + delta_render[0],
                    agent_render[1] + delta_render[1], agent_render[1] - delta_render[1],
                )
            else:
                overlay_extent = (0, w, h, 0)
            snapshots.append((t, frame, q_grid, action.numpy(), overlay_extent))

        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        t += 1

    vmin = min(q_grid.min() for _, _, q_grid, _, _ in snapshots)
    vmax = max(q_grid.max() for _, _, q_grid, _, _ in snapshots)
    print(colored(f"Q range over rollout: [{vmin:.3f}, {vmax:.3f}]", "blue"))

    q_frames = []
    for t, frame, q_grid, action, overlay_extent in snapshots:
        save_path = os.path.join(out_dir, f"step_{t:04d}.png")
        plot_q_heatmap(frame, q_grid, action_low, action_high, action, overlay_extent, save_path, vmin, vmax, cmap=cfg.get('q_cmap', 'turbo'),)
        q_frames.append(imageio.imread(save_path))
        print(colored(f"   saved {save_path}", "yellow"))
    if q_frames:
        gif_path = os.path.join(out_dir, "q_viz.gif")
        imageio.mimsave(gif_path, q_frames, fps=4)
        print(colored(f"Saved animation to {gif_path}", "green", attrs=["bold"]))


if __name__ == "__main__":
    visualize()
