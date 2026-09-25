"""MPPI over ComFree rollouts.

This is TD-MPC2's planner with the learned latent dynamics and learned reward
replaced by the simulator and the analytic task reward.  What survives from
TD-MPC is the part that carries long-horizon information: the terminal value
`gamma^H * Q(z_H, pi(z_H))`, and the policy prior that seeds a fraction of the
sampled trajectories.

One rollout of `num_samples` trajectories is one pass over the world bank: every
sample is a world, so a horizon-H rollout costs H control steps regardless of how
many samples are drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from comfree_tdmpc.envs.task import ReorientTask
from comfree_tdmpc.sim.batch_sim import BatchSim


def colored_noise(beta: float, horizon: int, shape: tuple[int, ...], device) -> torch.Tensor:
    """Temporally correlated noise with a 1/f^beta power spectrum (iCEM).

    Sampling each timestep independently makes the *plan itself* discontinuous in
    time: consecutive replans then command unrelated postures and the hand
    chatters even though every individual plan looks fine.  Correlated noise
    samples smooth action sequences, so the elite set contains smooth motions.
    beta=0 recovers white noise.
    """
    if beta <= 0:
        return torch.randn(horizon, *shape, device=device)
    freqs = torch.fft.rfftfreq(horizon, device=device)
    freqs[0] = freqs[1] if freqs.numel() > 1 else 1.0
    scale = freqs.pow(-beta / 2.0)
    spectrum = torch.randn(
        freqs.numel(), *shape, dtype=torch.cfloat, device=device
    ) * scale.view(-1, *([1] * len(shape)))
    noise = torch.fft.irfft(spectrum, n=horizon, dim=0)
    return noise / (noise.std(dim=0, keepdim=True) + 1e-6)


@dataclass
class PlannerConfig:
    horizon: int = 16
    iterations: int = 4
    num_samples: int = 256
    num_elites: int = 32
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 0.8
    temperature: float = 0.5
    discount: float = 0.95
    # Exponent of the sampling noise spectrum; 2.0 = smooth (Brownian), 0 = white.
    noise_beta: float = 2.0
    use_terminal_value: bool = True
    # Harvest the elite rollouts themselves as training data.  Every planning
    # call already simulates num_samples x horizon transitions and then throws
    # all but the first action away; recording the elites turns one environment
    # step into `record_topk * horizon` extra supervised transitions.
    record_elites: bool = False
    record_topk: int = 8
    # What the harvested rollouts offer the policy as its imitation target.
    # "elite": the elite action itself -- but MPPI is multi-modal, so equally
    #   good elites disagree, and a deterministic policy fitted to all of them
    #   lands between the modes rather than on any of them.
    # "mean": the MPPI-weighted mean of the plan at that timestep -- one
    #   consistent target per state, and the action the planner itself executes
    #   at evaluation time.
    elite_target: str = "elite"


class MPPIPlanner:
    def __init__(
        self,
        sim: BatchSim,
        task: ReorientTask,
        cfg: PlannerConfig,
        agent=None,
        sim_buffer=None,
    ):
        self.sim = sim
        self.sim_buffer = sim_buffer
        self.task = task
        self.cfg = cfg
        self.agent = agent
        self.device = sim.torch_device
        self.action_dim = sim.nu
        assert sim.nworld == cfg.num_samples, (
            f"planner sim must have one world per sample "
            f"({sim.nworld} worlds vs {cfg.num_samples} samples)"
        )
        self._prev_mean = torch.zeros(cfg.horizon, self.action_dim, device=self.device)
        self._tape = None
        self.last_stats: dict[str, float] = {}

    @torch.no_grad()
    def _rollout(
        self,
        qpos0: torch.Tensor,
        qvel0: torch.Tensor,
        goal_quat: torch.Tensor,
        prev_action0: torch.Tensor,
        actions: torch.Tensor,
        num_pi: int,
        record: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll `actions` (H, N, A) through the simulator; returns (value, actions).

        Worlds `[:num_pi]` ignore their sampled actions and follow the policy
        prior instead; the executed actions are written back so the elite update
        sees what was really applied.
        """
        cfg = self.cfg
        sim, task = self.sim, self.task
        n = sim.nworld

        sim.set_state(qpos0.expand(n, -1), qvel0.expand(n, -1))
        goal = goal_quat.expand(n, -1)
        prev_action = prev_action0.expand(n, -1).clone()
        prev_angle = task.goal_angle(sim.qpos, goal)

        value = torch.zeros(n, device=self.device)
        alive = torch.ones(n, device=self.device)
        discount = 1.0
        self._tape = [] if record else None

        for t in range(cfg.horizon):
            a = actions[t]
            need_obs = record or (num_pi > 0 and self.agent is not None)
            if need_obs:
                obs = task.observation(sim.qpos, sim.qvel, goal, prev_action)
            if num_pi > 0 and self.agent is not None:
                a = a.clone()
                a[:num_pi] = self.agent.act_batch(obs[:num_pi], eval_mode=False)
                actions[t] = a

            sim.set_action(a)
            sim.step()

            r, angle, success, dropped = task.reward(
                sim.qpos, sim.qvel, goal, a, prev_action, prev_angle
            )
            if record:
                next_obs = task.observation(sim.qpos, sim.qvel, goal, a)
                self._tape.append((obs, next_obs, a.clone(), r, dropped.float(), alive.clone()))
            value = value + discount * alive * r
            alive = alive * (~dropped).float()
            discount *= cfg.discount
            prev_action = a
            prev_angle = angle

        if cfg.use_terminal_value and self.agent is not None:
            obs = task.observation(sim.qpos, sim.qvel, goal, prev_action)
            value = value + discount * alive * self.agent.terminal_value(obs)
        return value, actions

    @torch.no_grad()
    def plan(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        goal_quat: torch.Tensor,
        prev_action: torch.Tensor,
        t0: bool = False,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        qpos = qpos.reshape(1, -1)
        qvel = qvel.reshape(1, -1)
        goal_quat = goal_quat.reshape(1, -1)
        prev_action = prev_action.reshape(1, -1)

        mean = torch.zeros(cfg.horizon, self.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        else:
            mean[:] = prev_action  # start from the current posture, not zero
        std = cfg.max_std * torch.ones(cfg.horizon, self.action_dim, device=self.device)

        num_pi = cfg.num_pi_trajs if self.agent is not None else 0
        actions = torch.empty(
            cfg.horizon, cfg.num_samples, self.action_dim, device=self.device
        )

        for it in range(cfg.iterations):
            noise = colored_noise(
                cfg.noise_beta, cfg.horizon,
                (cfg.num_samples, self.action_dim), self.device,
            )
            actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1, 1)

            # Only the last iteration is worth harvesting: its samples come from
            # the converged distribution.
            last = it == cfg.iterations - 1
            value, actions = self._rollout(
                qpos, qvel, goal_quat, prev_action, actions, num_pi,
                record=cfg.record_elites and last and self.sim_buffer is not None,
            )
            value = value.nan_to_num(-1e4)

            elite_idxs = torch.topk(value, cfg.num_elites, dim=0).indices
            elite_value = value[elite_idxs]
            elite_actions = actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(cfg.temperature * (elite_value - max_value))
            score = score / (score.sum(0) + 1e-9)

            mean = (score.view(1, -1, 1) * elite_actions).sum(1)
            var = (score.view(1, -1, 1) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(1)
            std = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        if self._tape is not None:
            self._store_elites(elite_idxs, std, mean)

        # At evaluation time execute the MPPI-weighted mean; sampling a single
        # elite is an exploration device and adds jitter that the mean does not.
        if eval_mode:
            plan_actions = mean
        else:
            idx = torch.multinomial(score, 1).item()
            plan_actions = elite_actions[:, idx]
        self._prev_mean = mean
        self.last_stats = {
            "plan_value": float(elite_value.mean().item()),
            "plan_value_max": float(elite_value.max().item()),
            "plan_std": float(std.mean().item()),
        }

        a, a_std = plan_actions[0], std[0]
        if not eval_mode:
            a = a + a_std * torch.randn(self.action_dim, device=self.device)
        return a.clamp(-1, 1), mean[0], a_std

    @torch.no_grad()
    def _store_elites(self, elite_idxs: torch.Tensor, std: torch.Tensor,
                      mean: torch.Tensor) -> None:
        """Write the best rollouts of the final iteration into the sim buffer.

        Every stored transition is a real simulator transition scored by the real
        task reward, and its action is one the planner judged elite -- so the
        policy gets `topk * horizon` distillation targets per environment step
        instead of the single executed action, and Q gets the same number of
        Dyna-style TD transitions.
        """
        k = min(self.cfg.record_topk, elite_idxs.numel())
        sel = elite_idxs[:k]
        obs, nobs, act, rew, done, alive = [], [], [], [], [], []
        for t, (o, no, a, r, d, al) in enumerate(self._tape):
            obs.append(o[sel])
            nobs.append(no[sel])
            act.append(a[sel])
            rew.append(r[sel])
            done.append(d[sel])
            alive.append(al[sel])
        obs = torch.cat(obs)
        keep = torch.cat(alive) > 0  # drop transitions after the cube was lost
        if keep.sum() == 0:
            self._tape = None
            return
        a = torch.cat(act)[keep]
        if self.cfg.elite_target == "mean":
            # One consistent target per timestep instead of each elite's own draw.
            target = torch.cat([
                mean[t].unsqueeze(0).expand(k, -1) for t in range(len(self._tape))
            ])[keep]
        else:
            target = a  # the elite action is itself the distillation target
        self.sim_buffer.add_batch(
            obs[keep],
            torch.cat(nobs)[keep],
            a,
            target,
            std[0].unsqueeze(0).expand(a.shape[0], -1),
            torch.cat(rew)[keep],
            torch.cat(done)[keep],
        )
        self._tape = None

    def reset(self) -> None:
        self._prev_mean.zero_()
