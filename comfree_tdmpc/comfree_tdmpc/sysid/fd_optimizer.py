"""System identification: first-order projected gradient with a line search.

This replaces "learning the world model".  Instead of fitting a black-box latent
dynamics network, we fit a handful of physical parameters of a simulator whose
contact model is *analytic and smooth* -- ComFree resolves contacts with a
differentiable penalty law instead of a complementarity problem, so the map
theta -> trajectory has no combinatorial mode switching and a first-order method
is well posed.

Gradients are obtained by central differences, which the per-world parameter
support turns into a single batched rollout: with P parameters and M chunks, all
2P+1 parameter settings x M chunks are simulated *simultaneously* as
(2P+1)*M worlds.  The Armijo line search is batched the same way -- every
candidate step length is another block of worlds, so a full descent iteration
costs exactly two batched rollouts regardless of P or the ladder length.

Optimisation is done in log-space with box projection, so parameters stay
positive and the projection is a clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from comfree_tdmpc.envs.task import HAND_NQ, quat_error
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.params import NUM_PARAMS, ParamVector, log_bounds


@dataclass
class SysIDConfig:
    chunk_length: int = 5
    num_chunks: int = 8
    fd_eps: float = 0.05  # central-difference step, in log-space
    num_iters: int = 15
    ls_steps: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.1, 0.03)
    init_step: float = 0.5
    armijo_c: float = 1e-4
    grad_tol: float = 1e-4
    w_hand_qpos: float = 1.0
    w_cube_pos: float = 20.0
    w_cube_rot: float = 1.0
    w_qvel: float = 0.02
    recompute_const: bool = True


class SysIDOptimizer:
    """Projected-gradient SysID over ComFree's physical parameters."""

    def __init__(
        self,
        cfg: SysIDConfig,
        device: str = "cuda:0",
        n_substeps: int = 10,
        timestep: float | None = None,
    ):
        self.cfg = cfg
        self.num_grad_worlds = (2 * NUM_PARAMS + 1) * cfg.num_chunks
        self.num_ls_worlds = len(cfg.ls_steps) * cfg.num_chunks
        nworld = max(self.num_grad_worlds, self.num_ls_worlds)
        self.sim = BatchSim(
            SimConfig(
                nworld=nworld,
                n_substeps=n_substeps,
                timestep=timestep,
                capture_graph=True,
                device=device,
            )
        )
        self.torch_device = self.sim.torch_device
        self.nworld = nworld
        self.last_history: list[dict[str, float]] = []

    @torch.no_grad()
    def _evaluate(self, thetas: torch.Tensor, chunks: dict[str, torch.Tensor]) -> torch.Tensor:
        """Multi-step prediction loss for K parameter settings.

        `thetas` is (K, P) in linear units; returns (K,) losses averaged over the
        M chunks.  World w = k*M + j holds candidate k replaying chunk j.
        """
        cfg = self.cfg
        k = thetas.shape[0]
        m = chunks["qpos0"].shape[0]
        n = k * m
        assert n <= self.nworld, f"need {n} worlds, have {self.nworld}"

        sim = self.sim
        # World w replays chunk (w mod M) under candidate (w div M).  The world
        # bank has a fixed size, so any leftover worlds repeat the last candidate
        # and are simply discarded when the losses are gathered.
        w = torch.arange(self.nworld, device=self.torch_device)
        w_chunk = w % m
        w_cand = torch.clamp(torch.div(w, m, rounding_mode="floor"), max=k - 1)

        sim.set_params(thetas[w_cand], recompute_const=cfg.recompute_const)
        sim.set_state(chunks["qpos0"][w_chunk], chunks["qvel0"][w_chunk])

        loss = torch.zeros(self.nworld, device=self.torch_device)
        length = chunks["actions"].shape[0]
        for t in range(length):
            sim.set_action(chunks["actions"][t][w_chunk])
            sim.step()
            loss = loss + self._state_error(
                sim.qpos, sim.qvel, chunks["qpos_ref"][t][w_chunk], chunks["qvel_ref"][t][w_chunk]
            )

        loss = loss[:n].view(k, m).mean(dim=1) / length
        return loss

    def _state_error(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ref_qpos: torch.Tensor,
        ref_qvel: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        hand = (qpos[:, :HAND_NQ] - ref_qpos[:, :HAND_NQ]).pow(2).sum(-1)
        cube_pos = (qpos[:, HAND_NQ : HAND_NQ + 3] - ref_qpos[:, HAND_NQ : HAND_NQ + 3]).pow(2).sum(-1)
        cube_rot = quat_error(qpos[:, HAND_NQ + 3 : HAND_NQ + 7], ref_qpos[:, HAND_NQ + 3 : HAND_NQ + 7]).pow(2)
        vel = (qvel - ref_qvel).pow(2).sum(-1)
        return (
            cfg.w_hand_qpos * hand
            + cfg.w_cube_pos * cube_pos
            + cfg.w_cube_rot * cube_rot
            + cfg.w_qvel * vel
        )

    @torch.no_grad()
    def gradient(
        self, params: ParamVector, chunks: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, float]:
        """Central-difference gradient w.r.t. log-parameters, one batched rollout."""
        eps = self.cfg.fd_eps
        log_theta = params.log
        lo, hi = log_bounds()

        cand_log = [log_theta]
        for p in range(NUM_PARAMS):
            for sign in (+1.0, -1.0):
                q = log_theta.copy()
                q[p] = np.clip(q[p] + sign * eps, lo[p], hi[p])
                cand_log.append(q)
        cand_log_arr = np.stack(cand_log)  # (2P+1, P)

        thetas = torch.as_tensor(
            np.exp(cand_log_arr), dtype=torch.float32, device=self.torch_device
        )
        losses = self._evaluate(thetas, chunks).cpu().numpy().astype(np.float64)

        grad = np.zeros(NUM_PARAMS)
        for p in range(NUM_PARAMS):
            lp, lm = losses[1 + 2 * p], losses[2 + 2 * p]
            # Use the *actual* (possibly clipped) separation so that a parameter
            # sitting on its bound still yields a one-sided derivative.
            h = cand_log_arr[1 + 2 * p][p] - cand_log_arr[2 + 2 * p][p]
            grad[p] = (lp - lm) / h if abs(h) > 1e-12 else 0.0
        return grad, float(losses[0])

    @torch.no_grad()
    def descent_step(
        self, params: ParamVector, chunks: dict[str, torch.Tensor], step_scale: float
    ) -> tuple[ParamVector, dict[str, float]]:
        """One projected-gradient iteration with a batched Armijo line search."""
        cfg = self.cfg
        grad, loss0 = self.gradient(params, chunks)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < cfg.grad_tol:
            return params, {"loss": loss0, "grad_norm": gnorm, "step": 0.0, "accepted": 0.0}

        direction = -grad / max(gnorm, 1e-12)
        lo, hi = log_bounds()
        cand_log = []
        alphas = []
        for s in cfg.ls_steps:
            alpha = step_scale * s
            cand_log.append(np.clip(params.log + alpha * direction, lo, hi))
            alphas.append(alpha)
        cand_log_arr = np.stack(cand_log)

        thetas = torch.as_tensor(
            np.exp(cand_log_arr), dtype=torch.float32, device=self.torch_device
        )
        losses = self._evaluate(thetas, chunks).cpu().numpy().astype(np.float64)

        best = int(np.argmin(losses))
        # Armijo sufficient decrease along the normalised direction.
        armijo = losses[best] <= loss0 - cfg.armijo_c * alphas[best] * gnorm
        if losses[best] < loss0:
            new_params = params.copy().set_log(cand_log_arr[best])
            accepted = 1.0
        else:
            new_params = params
            accepted = 0.0
        info = {
            "loss": loss0,
            "loss_new": float(losses[best]),
            "grad_norm": gnorm,
            "step": float(alphas[best]),
            "accepted": accepted,
            "armijo": float(bool(armijo)),
        }
        return new_params, info

    @torch.no_grad()
    def validation_loss(self, params: ParamVector, chunks: dict[str, torch.Tensor]) -> float:
        theta = torch.as_tensor(
            params.value[None], dtype=torch.float32, device=self.torch_device
        )
        return float(self._evaluate(theta, chunks)[0].item())

    @torch.no_grad()
    def optimize(
        self,
        params: ParamVector,
        chunk_sampler,
        num_iters: int | None = None,
        verbose: bool = False,
        eval_chunks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[ParamVector, list[dict[str, float]]]:
        """Run projected gradient descent; `chunk_sampler()` returns fresh chunks.

        The training loss is measured on freshly sampled chunks every iteration,
        so it is a noisy estimate; pass `eval_chunks` to also track a held-out
        loss on a fixed set, which is what actually shows convergence.
        """
        cfg = self.cfg
        iters = num_iters if num_iters is not None else cfg.num_iters
        history: list[dict[str, float]] = []
        step_scale = cfg.init_step
        for it in range(iters):
            chunks = chunk_sampler()
            if chunks is None:
                break
            params, info = self.descent_step(params, chunks, step_scale)
            info["iter"] = it
            if eval_chunks is not None:
                info["val_loss"] = self.validation_loss(params, eval_chunks)
            info.update({f"param/{n}": float(v) for n, v in params.as_dict().items()})
            history.append(info)
            if info["accepted"] > 0:
                # Grow slowly on success, shrink hard on failure (trust region-ish).
                step_scale = min(step_scale * 1.3, 1.0)
            else:
                step_scale = max(step_scale * 0.4, 1e-3)
            if verbose:
                print(
                    f"  [sysid {it:02d}] loss={info['loss']:.5f} -> {info.get('loss_new', float('nan')):.5f} "
                    f"|g|={info['grad_norm']:.4f} a={info['step']:.3f} {params}"
                )
        self.last_history = history
        return params, history
