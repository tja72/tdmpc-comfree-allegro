from time import time

import math
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from tdmpc_square.trainer.base import Trainer

# comfree-allegro-cube only: mirror ../comfree_tdmpc's train.csv/eval.csv metrics.
COMFREE_LOG_EVERY = 2000  # comfree_tdmpc VecTrainConfig.log_every_steps
COMFREE_EPISODE_KEYS = ["episode_reward", "episode_length", "goals_reached", "dropped", "mean_angle"]


def comfree_eval_stats(records, prefix):
    """Same statistics as comfree_tdmpc's VecTrainer.evaluate(), from its own
    per-episode records (VecAllegroReorientEnv.finished)."""
    if not records:
        return {}
    ret = np.array([e["episode_reward"] for e in records])
    return {
        prefix + "return": float(ret.mean()),
        prefix + "return_std": float(ret.std(ddof=1)) if len(ret) > 1 else np.nan,
        prefix + "goals_reached": float(np.mean([e["goals_reached"] for e in records])),
        prefix + "dropped": float(np.mean([e["dropped"] for e in records])),
        prefix + "length": float(np.mean([e["episode_length"] for e in records])),
    }


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        # Per-episode records from envs that emit them (comfree-allegro-cube). `info`
        # is a defaultdict, so test membership rather than index into it.
        ep_records, ep_records_pi = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if "episode" in info:
                ep_records.append(info["episode"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        
        if self.cfg.eval_pi:
            # Evaluate nominal policy pi
            ep_rewards_pi, ep_successes_pi = [], []
            for i in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
                while not done:
                    action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards_pi.append(ep_reward)
                ep_successes_pi.append(info["success"])
                if "episode" in info:
                    ep_records_pi.append(info["episode"])
            
        metrics = dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_reward_pi=np.nanmean(ep_rewards_pi) if self.cfg.eval_pi else np.nan,
            episode_success_pi=np.nanmean(ep_successes_pi) if self.cfg.eval_pi else np.nan,
        )
        if ep_records:
            metrics.update(comfree_eval_stats(ep_records, "mpc/"))
            metrics.update(comfree_eval_stats(ep_records_pi, "pi/"))
        return metrics

    def _log_comfree_train(self, train_metrics):
        """comfree_tdmpc's train.csv row: rolling means over the last 64 episodes."""
        rec = self._comfree_recent[-64:]
        wall = time() - self._start_time
        row = dict(
            step=self._step,
            wall_time=wall,
            env_steps_per_s=self._step / max(wall, 1e-9),
            episodes=len(self._comfree_recent),
            **{k: float(np.mean([e[k] for e in rec])) for k in COMFREE_EPISODE_KEYS},
        )
        # Pearl's own losses etc.; its step/episode_reward duplicate the above.
        for k, v in train_metrics.items():
            row.setdefault(k, float(v))
        self.logger.log_comfree_train(row)

    def eval_value(self, n_samples=100):
        """evaluate value approximation."""
        # MC value estimation
        mc_ep_rewards = []
        for i in range(n_samples):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            while not done:
                action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward * self.agent.discount ** t
                t += 1
            mc_ep_rewards.append(ep_reward)

        # Value function approximation
        q_values = []
        for i in range(n_samples):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
            action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
            task = None
            q_value = self.agent.model.Q(self.agent.model.encode(obs.to(self.agent.device), task), 
                                         action.to(self.agent.device), 
                                         task, return_type="avg")
            q_values.append(q_value.detach().cpu().numpy())
        
        return dict(
            mc_value= np.nanmean(mc_ep_rewards),
            q_value= np.nanmean(q_values),
        )

    def to_td(self, obs, action=None, mu=None, std=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if mu is None:
            mu = torch.full_like(action, float("nan"))
        if std is None:
            std = torch.full_like(action, float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                mu=mu.unsqueeze(0),
                std=std.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True
        self._comfree_recent, self._comfree_last_log = [], 0
        save_agent_steps = {int(s) for s in (self.cfg.get("save_agent_steps", None) or [])}

        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()

                    if self.cfg.eval_value:
                        eval_metrics.update(self.eval_value())
                        
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {'return': train_metrics['episode_reward'],
                                       'episode_length': len(self._tds[1:]),
                                       'success': train_metrics['episode_success'],
                                       'success_subtasks': info['success_subtasks'],
                                       'step': self._step,}
                
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))
                    if "episode" in info:
                        self._comfree_recent.append(info["episode"])

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                t0 = len(self._tds) == 1
                action, mu, std = self.agent.act(obs, t0=t0)
            else:
                action = self.env.rand_act()
                mu, std = action.detach().clone(), torch.full_like(action, math.exp(self.cfg.log_std_max)) # torch.full_like(action, float('nan')), torch.full_like(action, float('nan')) #  # noqa
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, mu, std, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            # Same agent state a run with steps=<this step> saves as final.pt, since
            # the loop runs self._step = 0..cfg.steps inclusive.
            if self._step in save_agent_steps:
                self.logger.save_agent(self.agent, identifier=self._step)

            self._step += 1
            if self._comfree_recent and self._step - self._comfree_last_log >= COMFREE_LOG_EVERY:
                self._comfree_last_log = self._step
                self._log_comfree_train(train_metrics)

        self.logger.finish(self.agent)
