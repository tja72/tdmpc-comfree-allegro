"""MPPI for `num_envs` environments at once, over one shared ComFree bank.

`MPPIPlanner` maps one sampled action sequence to one world.  This class maps
one *(environment, sample)* pair to one world, laying them out env-major so
world `e * num_samples + i` is sample `i` of environment `e`.  Everything the
single-environment planner does per call is then done for all environments in
the same kernel launches: the horizon is still simulated in order, but the
device is filled in the sample direction, which is the direction that was empty.

The measured consequence on an RTX 4080: 256 worlds cost 3.3 ms per control step
and 4096 cost 10.0 ms, so 16 environments planning together produce 16 times the
data for 3 times the wall clock.
"""

from __future__ import annotations

import torch

from comfree_tdmpc.envs.task import ReorientTask
from comfree_tdmpc.planner.mppi import PlannerConfig, colored_noise
from comfree_tdmpc.sim.batch_sim import BatchSim


class BatchMPPIPlanner:
    def __init__(
        self,
        sim: BatchSim,
        task: ReorientTask,
        cfg: PlannerConfig,
        num_envs: int,
        agent=None,
        sim_buffer=None,
    ):
        self.sim = sim
        self.task = task
        self.cfg = cfg
        self.agent = agent
        self.sim_buffer = sim_buffer
        self.device = sim.torch_device
        self.action_dim = sim.nu
        self.E = num_envs
        self.N = cfg.num_samples
        assert sim.nworld == self.E * self.N, (
            f"the bank must hold one world per (env, sample): "
            f"{sim.nworld} worlds vs {self.E} x {self.N}"
        )
        self._prev_mean = torch.zeros(cfg.horizon, self.E, self.action_dim, device=self.device)
        self._tape = None
        self.last_stats: dict[str, float] = {}
        self.last_elites: tuple[torch.Tensor, torch.Tensor] | None = None

        # Worlds whose sample index is below `num_pi_trajs` follow the policy
        # prior instead of their sampled action.  Precomputed because it is a
        # gather over 4096 worlds executed `horizon * iterations` times per step.
        w = torch.arange(self.E * self.N, device=self.device)
        self._is_pi = (w % self.N) < max(cfg.num_pi_trajs, 0)
        self._pi_idx = self._is_pi.nonzero(as_tuple=True)[0]
        self._env_ar = torch.arange(self.E, device=self.device)

    @torch.no_grad()
    def _rollout(
        self,
        qpos0: torch.Tensor,
        qvel0: torch.Tensor,
        goal_quat: torch.Tensor,
        prev_action0: torch.Tensor,
        actions: torch.Tensor,
        use_pi: bool,
        record: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll `actions` (H, E, N, A); returns value (E, N) and the executed actions."""
        cfg, sim, task = self.cfg, self.sim, self.task
        E, N = self.E, self.N
        W = E * N

        sim.set_state(
            qpos0.repeat_interleave(N, dim=0), qvel0.repeat_interleave(N, dim=0)
        )
        goal = goal_quat.repeat_interleave(N, dim=0)
        prev_action = prev_action0.repeat_interleave(N, dim=0).clone()
        prev_angle = task.goal_angle(sim.qpos, goal)

        value = torch.zeros(W, device=self.device)
        alive = torch.ones(W, device=self.device)
        discount = 1.0
        self._tape = [] if record else None

        for t in range(cfg.horizon):
            a = actions[t].reshape(W, self.action_dim)
            need_obs = record or (use_pi and self.agent is not None)
            if need_obs:
                obs = task.observation(sim.qpos, sim.qvel, goal, prev_action)
            if use_pi and self.agent is not None:
                a = a.clone()
                a[self._pi_idx] = self.agent.act_batch(
                    obs[self._pi_idx], eval_mode=False
                )
                actions[t] = a.view(E, N, self.action_dim)

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
        return value.view(E, N), actions

    @torch.no_grad()
    def plan(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        goal_quat: torch.Tensor,
        prev_action: torch.Tensor,
        t0: torch.Tensor | bool = False,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One planning step for every environment.

        `t0` is a per-environment boolean: lanes that have just been reset have
        no usable warm start and begin from the posture they are holding.
        Returns (action, mean, std), each (E, A).
        """
        cfg = self.cfg
        E, N, A = self.E, self.N, self.action_dim
        H = cfg.horizon

        mean = torch.zeros(H, E, A, device=self.device)
        mean[:-1] = self._prev_mean[1:]
        if isinstance(t0, bool):
            t0 = torch.full((E,), t0, dtype=torch.bool, device=self.device)
        if bool(t0.any()):
            mean = torch.where(
                t0.view(1, E, 1), prev_action.view(1, E, A).expand(H, -1, -1), mean
            )
        std = cfg.max_std * torch.ones(H, E, A, device=self.device)

        use_pi = cfg.num_pi_trajs > 0 and self.agent is not None
        elite_idxs = elite_value = elite_actions = score = None

        for it in range(cfg.iterations):
            noise = colored_noise(cfg.noise_beta, H, (E, N, A), self.device)
            actions = (mean.unsqueeze(2) + std.unsqueeze(2) * noise).clamp(-1, 1)

            last = it == cfg.iterations - 1
            value, actions = self._rollout(
                qpos, qvel, goal_quat, prev_action, actions, use_pi,
                record=cfg.record_elites and last and self.sim_buffer is not None,
            )
            value = value.nan_to_num(-1e4)

            elite_idxs = torch.topk(value, cfg.num_elites, dim=1).indices  # (E, K)
            elite_value = value.gather(1, elite_idxs)  # (E, K)
            # (H, E, K, A): pick the elite samples out of every environment's block.
            elite_actions = actions[:, self._env_ar.unsqueeze(1), elite_idxs]

            max_value = elite_value.max(dim=1, keepdim=True).values
            score = torch.exp(cfg.temperature * (elite_value - max_value))
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)

            w = score.view(1, E, -1, 1)
            mean = (w * elite_actions).sum(2)
            var = (w * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2)
            std = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        # The converged elite set, kept for diagnostics: how far apart the
        # actions the planner considers equally good actually are is the error
        # floor for any deterministic policy fitted to them.
        self.last_elites = (elite_actions[0].clone(), elite_value.clone())

        if self._tape is not None:
            self._store_elites(elite_idxs, std, mean)

        if eval_mode:
            plan_actions = mean[0]  # (E, A)
        else:
            pick = torch.multinomial(score, 1)  # (E, 1)
            plan_actions = elite_actions[0].gather(
                1, pick.unsqueeze(-1).expand(-1, -1, A)
            ).squeeze(1)
        self._prev_mean = mean
        self.last_stats = {
            "plan_value": float(elite_value.mean().item()),
            "plan_value_max": float(elite_value.max().item()),
            "plan_std": float(std.mean().item()),
        }

        a, a_std = plan_actions, std[0]
        if not eval_mode:
            a = a + a_std * torch.randn(E, A, device=self.device)
        return a.clamp(-1, 1), mean[0], a_std

    @torch.no_grad()
    def _store_elites(self, elite_idxs: torch.Tensor, std: torch.Tensor,
                      mean: torch.Tensor) -> None:
        """Harvest the top-k rollouts of every environment into the sim buffer.

        Each planning step already simulated `E * N * H` transitions and is about
        to discard all but `E` first actions.  Recording the elites turns one
        vectorised environment step into `E * topk * H` extra supervised
        transitions -- with 16 environments and topk=8 that is 2048 transitions
        per step instead of 16.
        """
        N = self.N
        k = min(self.cfg.record_topk, elite_idxs.shape[1])
        # Elite sample index -> flat world index within this environment's block.
        sel = (self._env_ar.unsqueeze(1) * N + elite_idxs[:, :k]).reshape(-1)  # (E*k,)

        obs, nobs, act, rew, done, alive = [], [], [], [], [], []
        for o, no, a, r, d, al in self._tape:
            obs.append(o[sel])
            nobs.append(no[sel])
            act.append(a[sel])
            rew.append(r[sel])
            done.append(d[sel])
            alive.append(al[sel])
        obs = torch.cat(obs)
        keep = torch.cat(alive) > 0  # drop transitions after the cube was lost
        if int(keep.sum()) == 0:
            self._tape = None
            return
        a = torch.cat(act)[keep]
        # Every elite belongs to some environment; give each its own std row.
        std_rows = std[0].repeat_interleave(k, dim=0).repeat(len(self._tape), 1)[keep]
        if self.cfg.elite_target == "mean":
            # One target per (state, timestep): the plan's own weighted mean, so
            # every elite visiting that timestep is told to reproduce the same
            # action rather than its own draw from a multi-modal set.
            target = torch.cat([
                mean[t].repeat_interleave(k, dim=0) for t in range(len(self._tape))
            ])[keep]
        else:
            target = a  # the elite action is itself the distillation target
        self.sim_buffer.add_batch(
            obs[keep],
            torch.cat(nobs)[keep],
            a,
            target,
            std_rows,
            torch.cat(rew)[keep],
            torch.cat(done)[keep],
        )
        self._tape = None

    def reset(self) -> None:
        self._prev_mean.zero_()

    def reset_idx(self, idx: torch.Tensor) -> None:
        if idx.numel():
            self._prev_mean[:, idx] = 0.0

    def snapshot(self) -> torch.Tensor:
        return self._prev_mean.clone()

    def restore(self, prev_mean: torch.Tensor) -> None:
        self._prev_mean = prev_mean.clone()
