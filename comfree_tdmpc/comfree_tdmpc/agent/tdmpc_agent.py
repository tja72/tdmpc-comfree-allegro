"""The learning half of TD-M(PC)^2, with the world model removed.

What TD-MPC2 learns is (encoder, latent dynamics, reward, Q, pi).  Here the
dynamics are the simulator and the reward is analytic, so the consistency loss
disappears and what remains is:

  * an encoder + Q ensemble trained by TD on transitions collected by the planner,
  * a policy trained against Q, optionally constrained towards the planner's own
    action distribution (TD-M(PC)^2's `residual` mode) because *all* of the data
    is off-policy planner data,
  * an optional reward head, kept so the planner can be run with a learned reward
    for ablation.

The Q function is what the planner queries as its terminal value, so improving it
directly lengthens the planner's effective horizon.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from comfree_tdmpc.agent.nets import (
    EnsembleMLP,
    RunningScale,
    SimNorm,
    TwoHot,
    gaussian_logprob,
    mlp,
    squash,
    weight_init,
)


@dataclass
class AgentConfig:
    latent_dim: int = 256
    mlp_dim: int = 384
    enc_dim: int = 256
    num_q: int = 5
    dropout: float = 0.01
    simnorm_dim: int = 8

    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    tau: float = 0.01
    discount: float = 0.95
    batch_size: int = 256

    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1e-4

    # Policy objective:
    # residual: TD-M(PC)^2's Gaussian prior towards the planner's distribution
    # sac:      maximise Q alone
    # bc:       regress onto the action that was taken
    # bc_mu:    regress onto the planner's *mean*, which is one consistent
    #           target per state; `bc` regresses onto the individual elite
    #           action, and equally good elites disagree with each other.
    # bc_no_q / bc_mu_no_q: same targets as bc / bc_mu, but q_loss is left
    #           out of the policy's backward pass (still computed and logged).
    #           Q itself still trains and still serves as the terminal value.
    actor_mode: str = "residual"  # residual | sac | bc | bc_mu | bc_no_q | bc_mu_no_q
    prior_coef: float = 0.5
    # Offline phase only (train_offline): explicit pull of pi(z_real) towards the
    # real cohort's mu, same order of magnitude as prior_coef since it plays the
    # same regularizing role, just against a real- instead of a batch-local target.
    real_reg_coef: float = 1.0
    min_std: float = 0.05
    scale_threshold: float = 1.0

    # How the terminal value is read out for the planner.  MPPI maximises over
    # hundreds of sampled endpoints, so it actively seeks states where Q is
    # over-estimated; taking the min of two Q heads makes that exploitation
    # cost the planner something instead of paying it a bonus.
    terminal_value_type: str = "min"
    learn_reward: bool = True
    reward_coef: float = 0.1
    value_coef: float = 1.0


class TDMPCAgent(nn.Module):
    def __init__(self, cfg: AgentConfig, obs_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim

        self._encoder = mlp(obs_dim, [cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg.simnorm_dim))
        self._pi = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2 * action_dim)
        self._Qs = EnsembleMLP(
            cfg.num_q,
            cfg.latent_dim + action_dim,
            [cfg.mlp_dim, cfg.mlp_dim],
            cfg.num_bins,
            dropout=cfg.dropout,
        )
        self._reward = mlp(cfg.latent_dim + action_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.num_bins)
        self.apply(weight_init)
        self._Qs.zero_last_layer_()
        nn.init.zeros_(self._reward[-1].weight)
        nn.init.zeros_(self._reward[-1].bias)
        self.to(device)
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

        self.two_hot = TwoHot(cfg.num_bins, cfg.vmin, cfg.vmax, device)
        self.scale = RunningScale(cfg.tau, device)
        self.log_std_min = torch.tensor(cfg.log_std_min, device=device)
        self.log_std_dif = torch.tensor(cfg.log_std_max - cfg.log_std_min, device=device)

        self.optim = torch.optim.Adam(
            [
                {"params": self._encoder.parameters(), "lr": cfg.lr * cfg.enc_lr_scale},
                {"params": self._Qs.parameters()},
                {"params": self._reward.parameters()},
            ],
            lr=cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(self._pi.parameters(), lr=cfg.lr, eps=1e-5)
        self.eval()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self._encoder(obs)

    def pi(self, z: torch.Tensor):
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mu)
        log_pi = gaussian_logprob(eps, log_std)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def Q(self, z: torch.Tensor, a: torch.Tensor, return_type: str = "min", target: bool = False):
        x = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(x)
        if return_type == "all":
            return out
        idx = np.random.choice(self.cfg.num_q, 2, replace=False)
        q1, q2 = self.two_hot.decode(out[idx[0]]), self.two_hot.decode(out[idx[1]])
        if return_type == "min":
            return torch.min(q1, q2)
        return (q1 + q2) / 2

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._reward(torch.cat([z, a], dim=-1))

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """Actions for the policy-prior rollouts inside MPPI."""
        mu, pi, _, _ = self.pi(self.encode(obs))
        return mu if eval_mode else pi

    @torch.no_grad()
    def act(self, obs: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        return self.act_batch(obs.unsqueeze(0), eval_mode)[0]

    @torch.no_grad()
    def terminal_value(self, obs: torch.Tensor) -> torch.Tensor:
        """gamma-discounted value used as MPPI's terminal cost."""
        z = self.encode(obs)
        _, pi, _, _ = self.pi(z)
        return self.Q(z, pi, return_type=self.cfg.terminal_value_type).squeeze(-1)

    @torch.no_grad()
    def predicted_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.two_hot.decode(self.reward(self.encode(obs), action)).squeeze(-1)

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for p, pt in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                pt.data.lerp_(p.data, self.cfg.tau)

    def track_q_grad(self, mode: bool) -> None:
        for p in self._Qs.parameters():
            p.requires_grad_(mode)

    def update_pi(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        z_real: torch.Tensor | None = None,
        mu_real: torch.Tensor | None = None,
    ):
        cfg = self.cfg
        self.pi_optim.zero_grad(set_to_none=True)
        self.track_q_grad(False)

        _, pis, log_pis, _ = self.pi(z)
        qs = self.Q(z, pis, return_type="avg")
        self.scale.update(qs)
        qs = self.scale(qs)

        q_loss = (cfg.entropy_coef * log_pis - qs).mean()
        if cfg.actor_mode == "sac":
            prior_loss = torch.zeros_like(q_loss)
            pi_loss = q_loss
        elif cfg.actor_mode == "residual":
            # TD-M(PC)^2: keep pi close to the planner's action distribution,
            # since every transition in the buffer was produced by the planner.
            std_c = torch.clamp(std, min=cfg.min_std)
            eps = (pis - mu) / std_c
            log_pis_prior = gaussian_logprob(eps, std_c.log())
            log_pis_prior = (
                self.scale(log_pis_prior)
                if self.scale.value > cfg.scale_threshold
                else torch.zeros_like(log_pis_prior)
            )
            prior_loss = -log_pis_prior.mean()
            pi_loss = q_loss + (cfg.prior_coef * self.action_dim / 61) * prior_loss
        elif cfg.actor_mode == "bc":
            prior_loss = (pis - action).pow(2).sum(-1).mean()
            pi_loss = q_loss + cfg.prior_coef * prior_loss
        elif cfg.actor_mode == "bc_mu":
            prior_loss = (pis - mu).pow(2).sum(-1).mean()
            pi_loss = q_loss + cfg.prior_coef * prior_loss
        elif cfg.actor_mode == "bc_no_q":
            # Same target as bc; only the policy's own objective drops q_loss.
            prior_loss = (pis - action).pow(2).sum(-1).mean()
            pi_loss = cfg.prior_coef * prior_loss
        elif cfg.actor_mode == "bc_mu_no_q":
            # Same target as bc_mu, q_loss excluded as above.
            prior_loss = (pis - mu).pow(2).sum(-1).mean()
            pi_loss = cfg.prior_coef * prior_loss
        else:
            raise NotImplementedError(cfg.actor_mode)

        # Offline real-rollout regularization (train_offline only): pull pi
        # towards the planner's recorded mu on real transitions, independent
        # of the real/synthetic mix that `z`/`mu` above came from.
        if z_real is not None:
            _, pis_real, _, _ = self.pi(z_real)
            real_reg = (pis_real - mu_real).pow(2).sum(-1).mean()
            pi_loss = pi_loss + cfg.real_reg_coef * real_reg

        pi_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self._pi.parameters(), cfg.grad_clip_norm)
        self.pi_optim.step()
        self.track_q_grad(True)
        return float(pi_loss.item()), float(q_loss.item()), float(prior_loss.item()), float(grad)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        obs, next_obs = batch["obs"], batch["next_obs"]
        action, reward = batch["action"], batch["reward"].unsqueeze(-1)
        done = batch["done"].unsqueeze(-1)

        with torch.no_grad():
            next_z = self.encode(next_obs)
            _, next_pi, _, _ = self.pi(next_z)
            next_q = self.Q(next_z, next_pi, return_type="min", target=True)
            td_target = reward + cfg.discount * (1.0 - done) * next_q

        self.train()
        self.optim.zero_grad(set_to_none=True)
        z = self.encode(obs)
        qs = self.Q(z, action, return_type="all")
        value_loss = self.two_hot.soft_ce(qs, td_target.expand(cfg.num_q, -1, -1)).mean()

        if cfg.learn_reward:
            reward_loss = self.two_hot.soft_ce(self.reward(z, action), reward).mean()
        else:
            reward_loss = torch.zeros((), device=self.device)

        total = cfg.value_coef * value_loss + cfg.reward_coef * reward_loss
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self._encoder.parameters()) + list(self._Qs.parameters()) + list(self._reward.parameters()),
            cfg.grad_clip_norm,
        )
        self.optim.step()

        pi_loss, pi_q, pi_prior, pi_grad = self.update_pi(
            z.detach(), action.detach(), batch["mu"].detach(), batch["std"].detach()
        )
        self.soft_update_target()
        self.eval()
        return {
            "value_loss": float(value_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "pi_loss": pi_loss,
            "pi_loss_q": pi_q,
            "pi_loss_prior": pi_prior,
            "grad_norm": float(grad_norm),
            "pi_grad_norm": pi_grad,
            "q_scale": self.scale.value,
            "td_target_mean": float(td_target.mean().item()),
        }

    def update_offline(
        self,
        batch_sim: dict[str, torch.Tensor],
        batch_real_for_q: dict[str, torch.Tensor],
        batch_real_for_pi: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Offline consolidation update (docs/offline_training_design.md SS2).

        value_loss/reward_loss train on `batch_real_for_q` (real-heavy, per
        `offline_value_real_ratio`) so Q's own TD loss is already a real
        anchor; pi's q_loss/prior_loss train on the large synthetic
        `batch_sim`, with an explicit real_reg pull towards `batch_real_for_pi`
        added on top -- pi has no TD loss of its own to anchor it to real data,
        unlike Q.
        """
        cfg = self.cfg
        obs, next_obs = batch_real_for_q["obs"], batch_real_for_q["next_obs"]
        action, reward = batch_real_for_q["action"], batch_real_for_q["reward"].unsqueeze(-1)
        done = batch_real_for_q["done"].unsqueeze(-1)

        with torch.no_grad():
            next_z = self.encode(next_obs)
            _, next_pi, _, _ = self.pi(next_z)
            next_q = self.Q(next_z, next_pi, return_type="min", target=True)
            td_target = reward + cfg.discount * (1.0 - done) * next_q

        self.train()
        self.optim.zero_grad(set_to_none=True)
        z = self.encode(obs)
        qs = self.Q(z, action, return_type="all")
        value_loss = self.two_hot.soft_ce(qs, td_target.expand(cfg.num_q, -1, -1)).mean()

        if cfg.learn_reward:
            reward_loss = self.two_hot.soft_ce(self.reward(z, action), reward).mean()
        else:
            reward_loss = torch.zeros((), device=self.device)

        total = cfg.value_coef * value_loss + cfg.reward_coef * reward_loss
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self._encoder.parameters()) + list(self._Qs.parameters()) + list(self._reward.parameters()),
            cfg.grad_clip_norm,
        )
        self.optim.step()

        with torch.no_grad():
            z_sim = self.encode(batch_sim["obs"])
            z_real_pi = self.encode(batch_real_for_pi["obs"])
        pi_loss, pi_q, pi_prior, pi_grad = self.update_pi(
            z_sim, batch_sim["action"], batch_sim["mu"], batch_sim["std"],
            z_real=z_real_pi, mu_real=batch_real_for_pi["mu"],
        )
        self.soft_update_target()
        self.eval()
        return {
            "value_loss": float(value_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "pi_loss": pi_loss,
            "pi_loss_q": pi_q,
            "pi_loss_prior": pi_prior,
            "grad_norm": float(grad_norm),
            "pi_grad_norm": pi_grad,
            "q_scale": self.scale.value,
            "td_target_mean": float(td_target.mean().item()),
        }

    def save(self, fp: str) -> None:
        torch.save({"model": self.state_dict()}, fp)

    def load(self, fp: str) -> None:
        sd = torch.load(fp, map_location=self.device)
        self.load_state_dict(sd["model"])
