import os
import sys

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

import warnings

warnings.filterwarnings("ignore")

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from termcolor import colored

from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.envs import make_env
from tdmpc_square.envs.pusht import PIXEL_SCALE

torch.backends.cudnn.benchmark = True


def collect_seed_positions(env, seed_steps):
    """
    Replays exactly the seed-phase exploration policy used in OnlineTrainer
    (`env.rand_act()` for `seed_steps` env steps, resetting on episode end) and
    records the agent's and block's pixel position at every step.
    """
    agent_positions = np.empty((seed_steps, 2), dtype=np.float32)
    block_positions = np.empty((seed_steps, 2), dtype=np.float32)

    env.reset()
    for step in range(seed_steps):
        action = env.rand_act()
        obs, reward, done, truncated, info = env.step(action)
        pos = obs[:4].numpy() * PIXEL_SCALE
        agent_positions[step] = pos[:2]
        block_positions[step] = pos[2:4]
        if done or truncated:
            env.reset()

    return agent_positions, block_positions


def corner_concentration(positions, corner_size):
    """
    Fraction of `positions` falling inside any of the 4 `corner_size`x`corner_size`
    pixel squares at the corners of the [0, PIXEL_SCALE]^2 board, compared to the
    fraction a uniform distribution would put there -- a ratio >> 1 indicates the
    random walk is disproportionately stuck at corners.
    """
    x, y = positions[:, 0], positions[:, 1]
    in_corner = (
        ((x < corner_size) | (x > PIXEL_SCALE - corner_size))
        & ((y < corner_size) | (y > PIXEL_SCALE - corner_size))
    )
    observed_frac = float(in_corner.mean())
    uniform_frac = 4 * (corner_size**2) / (PIXEL_SCALE**2)
    return observed_frac, uniform_frac, observed_frac / uniform_frac


def plot_position_histograms(agent_positions, block_positions, bins, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, positions, title in [
        (axes[0], agent_positions, "Agent position"),
        (axes[1], block_positions, "Block position"),
    ]:
        h = ax.hist2d(
            positions[:, 0], positions[:, 1],
            bins=bins, range=[[0, PIXEL_SCALE], [0, PIXEL_SCALE]], cmap="turbo",
        )
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{title} visited during seed phase")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.invert_yaxis()  # match image pixel coordinates (origin top-left)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@hydra.main(config_name="config", config_path="../tdmpc_square/tdmpc_square")
def analyze(cfg: dict):
    """
    Script for diagnosing whether the seed-phase (pure random) exploration on
    `pusht` gets disproportionately stuck in the corners of the board -- a
    bounded random walk (which is what `delta_actions` reduces `rand_act()` to)
    is statistically biased toward corners, since deltas pointing further into a
    wall get clipped back to the same position.

    Replays `cfg.seed_steps` steps of `env.rand_act()` (identical to the seed
    phase in OnlineTrainer) and plots a 2D histogram of visited agent/block
    positions, plus a quantitative corner-concentration ratio.

    Most relevant args:
            `seed_steps`: number of random seed steps to replay (uses the training config's value by default)
            `corner_size`: side length (px) of the corner squares used for the concentration stat (default: 64)
            `hist_bins`: number of histogram bins per axis (default: 32)
            `seed`: random seed (default: 1)

    Example usage:
    ```
            $ python evaluate/analyze_seed_exploration.py task=pusht delta_actions=true delta_action_max=40 seed_steps=20000
    ```
    """
    assert "pusht" in cfg.task, f"Expected a pusht task, got task={cfg.task}."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    corner_size = cfg.get("corner_size", 64)
    hist_bins = cfg.get("hist_bins", 32)

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"delta_actions={cfg.get('delta_actions', False)}, "
                  f"delta_action_max={cfg.get('delta_action_max', None)}, "
                  f"max_speed={cfg.get('max_speed', None)}", "blue"))
    print(colored(f"Replaying {cfg.seed_steps} seed steps (env.rand_act())...", "yellow", attrs=["bold"]))

    env = make_env(cfg)
    agent_positions, block_positions = collect_seed_positions(env, cfg.seed_steps)

    out_dir = os.path.join(cfg.work_dir, "seed_exploration")
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "position_histogram.png")
    plot_position_histograms(agent_positions, block_positions, hist_bins, hist_path)
    print(colored(f"Saved histogram to {hist_path}", "green", attrs=["bold"]))

    lines = [f"Task: {cfg.task}", f"seed_steps: {cfg.seed_steps}", f"corner_size: {corner_size} px", ""]
    for name, positions in [("Agent", agent_positions), ("Block", block_positions)]:
        observed, uniform, ratio = corner_concentration(positions, corner_size)
        lines.append(
            f"{name}: {observed*100:.1f}% of steps in a corner "
            f"(uniform-random baseline: {uniform*100:.1f}%, ratio={ratio:.2f}x)"
        )
    summary = "\n".join(lines)
    print(colored(summary, "yellow"))

    stats_path = os.path.join(out_dir, "seed_exploration_stats.txt")
    with open(stats_path, "w") as f:
        f.write(summary + "\n")
    print(colored(f"Saved stats to {stats_path}", "green", attrs=["bold"]))


if __name__ == "__main__":
    analyze()
