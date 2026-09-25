import os
import sys

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from termcolor import colored

from tdmpc_square.common import math
from tdmpc_square.common.buffer import Buffer
from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.envs import make_env
from tdmpc_square.tdmpc_square import TDMPC2

torch.backends.cudnn.benchmark = True


def to_td(env, obs, action=None, mu=None, std=None, reward=None):
    """Creates a TensorDict for a single environment step (mirrors OnlineTrainer.to_td)."""
    obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full_like(env.rand_act(), float("nan"))
    if mu is None:
        mu = torch.full_like(action, float("nan"))
    if std is None:
        std = torch.full_like(action, float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan"))
    return TensorDict(
        dict(
            obs=obs,
            action=action.unsqueeze(0),
            mu=mu.unsqueeze(0),
            std=std.unsqueeze(0),
            reward=reward.unsqueeze(0),
        ),
        batch_size=(1,),
    )


@torch.no_grad()
def prediction_loss(agent, obs, action, reward, task):
    """
    Computes the world-model prediction losses (consistency, reward, value) for a
    sampled batch, exactly as in TDMPC2.update(), but without any backward pass or
    optimizer step.
    """
    next_z = agent.model.encode(obs[1:], task)
    td_targets = agent._td_target(next_z, reward, task)

    zs = torch.empty(
        agent.cfg.horizon + 1,
        obs.shape[1],
        agent.cfg.latent_dim,
        device=agent.device,
    )
    z = agent.model.encode(obs[0], task)
    zs[0] = z
    consistency_loss = 0
    for t in range(agent.cfg.horizon):
        z = agent.model.next(z, action[t], task)
        consistency_loss += F.mse_loss(z, next_z[t]) * agent.cfg.rho**t
        zs[t + 1] = z

    _zs = zs[:-1]
    qs = agent.model.Q(_zs, action, task, return_type="all")
    reward_preds = agent.model.reward(_zs, action, task)

    reward_loss, value_loss = 0, 0
    for t in range(agent.cfg.horizon):
        reward_loss += (
            math.soft_ce(reward_preds[t], reward[t], agent.cfg).mean() * agent.cfg.rho**t
        )
        for q in range(agent.cfg.num_q):
            value_loss += (
                math.soft_ce(qs[q][t], td_targets[t], agent.cfg).mean() * agent.cfg.rho**t
            )
    consistency_loss *= 1 / agent.cfg.horizon
    reward_loss *= 1 / agent.cfg.horizon
    value_loss *= 1 / (agent.cfg.horizon * agent.cfg.num_q)

    total_loss = (
        agent.cfg.consistency_coef * consistency_loss
        + agent.cfg.reward_coef * reward_loss
        + agent.cfg.value_coef * value_loss
    )
    return {
        "consistency_loss": float(consistency_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(total_loss.item()),
    }


def collect_rollouts(env, agent, buffer, cfg, random=False):
    """
    Collects `cfg.eval_episodes` episodes into `buffer`.
    If `random` is True, actions (and thus the visited states) come from the
    environment's action space instead of the trained agent's policy.
    """
    for _ in range(cfg.eval_episodes):
        obs, done, t = env.reset()[0], False, 0
        tds = [to_td(env, obs)]
        while not done:
            if random:
                action = env.rand_act()
            else:
                action, _, _ = agent.act(obs, t0=t == 0, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            tds.append(to_td(env, obs, action, reward=reward))
            t += 1
        buffer.add(torch.cat(tds))


def evaluate_buffer(agent, buffer, loss_eval_batches):
    """Averages `prediction_loss` over `loss_eval_batches` sampled batches from `buffer`."""
    losses = []
    for _ in range(loss_eval_batches):
        obs, action, mu, std, reward, task = buffer.sample()
        losses.append(prediction_loss(agent, obs, action, reward, task))


    return {
        k: {
            "mean": float(np.mean([loss[k] for loss in losses])),
            "std": float(np.std([loss[k] for loss in losses])),
        } for k in losses[0]
    }



@hydra.main(config_name="config", config_path="../tdmpc_square/tdmpc_square")
def evaluate(cfg: dict):
    """
    Script for evaluating the world-model prediction loss (latent consistency,
    reward, and value losses) of a trained TD-MPC2 checkpoint.

    Evaluates two settings:
      1. On-policy: states/actions visited by the trained agent itself.
      2. Random: states/actions visited by taking random actions from the
         environment's action space (i.e. an out-of-distribution rollout).
    For each, transitions are stored in a replay buffer and the average
    prediction losses are reported over sampled subsequences -- the same
    quantities optimized in TDMPC2.update(), but computed without any
    gradient step.

    Most relevant args:
            `task`: task name
            `checkpoint`: path to model checkpoint to load
            `eval_episodes`: number of rollout episodes to collect (default: 10)
            `loss_eval_batches`: number of batches to average losses over (default: 50)
            `seed`: random seed (default: 1)

    Example usage:
    ```
            $ python evaluate/eval_prediction_loss.py task=pusht checkpoint=logs/pusht/0/<exp_name>/models/final.pt eval_episodes=10
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    loss_eval_batches = cfg.loss_eval_batches

    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))

    # Make environment and load agent
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)

    results = {}
    for name, random in [("On-policy", False), ("Random", True)]:
        print(
            colored(
                f"Collecting {cfg.eval_episodes} {name.lower()} episode(s) on {cfg.task}...",
                "yellow",
                attrs=["bold"],
            )
        )
        buffer = Buffer(cfg)
        collect_rollouts(env, agent, buffer, cfg, random=random)

        print(colored(f"Evaluating {name.lower()} prediction loss...", "yellow", attrs=["bold"]))
        avg_losses = evaluate_buffer(agent, buffer, loss_eval_batches)
        for k, v in avg_losses.items():
            print(colored(f"   {k:<20}\t{v['mean']:.4f} +- {v['std']:.4f}", "yellow"))
        results[name] = avg_losses

    out_dir = cfg.checkpoint.rsplit('/', 2)[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prediction_losses.txt")
    with open(out_path, "w") as f:
        f.write(f"Task: {cfg.task}\n")
        f.write(f"Checkpoint: {cfg.checkpoint}\n")
        f.write(f"eval_episodes: {cfg.eval_episodes}\n")
        f.write(f"loss_eval_batches: {loss_eval_batches}\n")
        for name, avg_losses in results.items():
            f.write(f"{name}:\n")
            for k, v in avg_losses.items():
                f.write(f"   {k:<20}\t{v['mean']:.4f} +- {v['std']:.4f}\n")
            f.write("\n")

    return results


if __name__ == "__main__":
    evaluate()
