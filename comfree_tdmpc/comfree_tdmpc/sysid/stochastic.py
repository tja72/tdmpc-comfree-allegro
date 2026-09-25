"""Stochastic projected gradient descent over the physical parameters.

With four parameters a coordinate-wise central difference plus a line search is
affordable.  With ~100 -- the number a real hand actually leaves unknown -- it is
not: the stencil alone costs 2P+1 rollouts, and the line search adds another
batched pass whose step length has to be re-derived every iteration.

This replaces both with a randomised gradient estimate and a fixed step schedule:

    g = (1/K) sum_k  [L(u + eps d_k) - L(u - eps d_k)] / (2 eps)  * d_k ,  d_k ~ N(0, I)
    u <- clip(u - alpha_t * g / ||g||, 0, 1)

Two evaluations per direction, K directions, independent of P.  Both the K
directions and the M data chunks are laid out across the world bank, so one
iteration is still exactly two batched rollouts no matter how many parameters
there are.  The estimator is unbiased for the Gaussian-smoothed objective, and
the smoothing is a feature here: it averages over the numerical noise that makes
coordinate-wise differences unreliable for weakly identifiable parameters.

Everything happens in the normalised u-coordinates of `ParamSpace`, so the box
projection is a clip to the unit cube and one step size fits all parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from comfree_tdmpc.envs.task import HAND_NQ, quat_error
from comfree_tdmpc.sim.batch_sim import BatchSim, SimConfig
from comfree_tdmpc.sim.handle_writer import HandleWriter
from comfree_tdmpc.sim.param_space import ParamSpace


@dataclass
class StochasticSysIDConfig:
    chunk_length: int = 5
    num_chunks: int = 16
    num_directions: int = 16  # K
    fd_eps: float = 0.02  # probe size in u-space
    step0: float = 0.03  # initial step length in u-space
    step_decay: float = 40.0  # alpha_t = step0 / (1 + t / step_decay)
    num_iters: int = 20
    momentum: float = 0.5
    grad_clip: float = 1.0
    w_hand_qpos: float = 1.0
    w_cube_pos: float = 20.0
    w_cube_rot: float = 1.0
    w_qvel: float = 0.02
    # A parameter setting can be physically unstable (a stiff servo with little
    # damping diverges under explicit integration).  Such a probe must score
    # badly but *finitely*, or the gradient estimate becomes NaN and the
    # optimiser cannot walk back out of the unstable region.
    error_cap: float = 1e3
    # Joint dry friction adds a constraint row per DOF on top of the contacts, so
    # this must be larger than a contact-only estimate would suggest.
    njmax: int = 800
    recompute_const: bool = True


class StochasticSysID:
    def __init__(
        self,
        cfg: StochasticSysIDConfig,
        space: ParamSpace,
        device: str = "cuda:0",
        n_substeps: int = 10,
        timestep: float | None = None,
    ):
        self.cfg = cfg
        self.space = space
        # World layout: (2K + 1) probes x M chunks.  The +1 evaluates the current
        # point, which is what the reported loss and any divergence check use.
        self.n_probe = 2 * cfg.num_directions + 1
        nworld = self.n_probe * cfg.num_chunks
        self.sim = BatchSim(
            SimConfig(nworld=nworld, n_substeps=n_substeps, timestep=timestep,
                      njmax=cfg.njmax, device=device)
        )
        self.writer = HandleWriter(self.sim, space)
        self.torch_device = self.sim.torch_device
        self.nworld = nworld
        self.velocity = np.zeros(space.n)

        w = torch.arange(nworld, device=self.torch_device)
        self._w_chunk = w % cfg.num_chunks
        self._w_probe = torch.div(w, cfg.num_chunks, rounding_mode="floor")

    @torch.no_grad()
    def _evaluate(self, u_probes: np.ndarray, chunks: dict[str, torch.Tensor]) -> np.ndarray:
        """Multi-step prediction loss for each of the `n_probe` coordinate vectors."""
        cfg, sim = self.cfg, self.sim
        vals = np.stack([self.space.values(u) for u in u_probes])
        vals_t = torch.as_tensor(vals, dtype=torch.float32, device=self.torch_device)
        self.writer.write(vals_t[self._w_probe], recompute_const=cfg.recompute_const)

        sim.set_state(chunks["qpos0"][self._w_chunk], chunks["qvel0"][self._w_chunk])
        loss = torch.zeros(self.nworld, device=self.torch_device)
        length = chunks["actions"].shape[0]
        for t in range(length):
            sim.set_action(chunks["actions"][t][self._w_chunk])
            sim.step()
            loss = loss + self._state_error(
                sim.qpos, sim.qvel,
                chunks["qpos_ref"][t][self._w_chunk], chunks["qvel_ref"][t][self._w_chunk],
            )
        # Chunks resampled from real rollouts include states from which a
        # mismatched model diverges outright, and a plain mean over chunks is
        # then dominated by two or three catastrophic ones -- the very chunks
        # that carry no gradient because they have saturated.  Compressing each
        # chunk's error logarithmically keeps every chunk's contribution
        # comparable while leaving the ordering and the gradient intact.
        per_chunk = torch.log1p(loss.view(self.n_probe, cfg.num_chunks) / length)
        return per_chunk.mean(dim=1).nan_to_num(
            float(np.log1p(cfg.error_cap))
        ).cpu().numpy().astype(np.float64)

    def _state_error(self, qpos, qvel, ref_qpos, ref_qvel) -> torch.Tensor:
        cfg = self.cfg
        hand = (qpos[:, :HAND_NQ] - ref_qpos[:, :HAND_NQ]).pow(2).sum(-1)
        cpos = (qpos[:, HAND_NQ:HAND_NQ + 3] - ref_qpos[:, HAND_NQ:HAND_NQ + 3]).pow(2).sum(-1)
        crot = quat_error(qpos[:, HAND_NQ + 3:HAND_NQ + 7],
                          ref_qpos[:, HAND_NQ + 3:HAND_NQ + 7]).pow(2)
        vel = (qvel - ref_qvel).pow(2).sum(-1)
        err = (cfg.w_hand_qpos * hand + cfg.w_cube_pos * cpos
               + cfg.w_cube_rot * crot + cfg.w_qvel * vel)
        cap = cfg.error_cap
        return torch.nan_to_num(err, nan=cap, posinf=cap, neginf=cap).clamp(max=cap)

    @torch.no_grad()
    def step(
        self, u: np.ndarray, chunks: dict[str, torch.Tensor], iteration: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, float]]:
        cfg, space = self.cfg, self.space
        K, P = cfg.num_directions, space.n

        dirs = rng.standard_normal((K, P))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

        probes = np.empty((self.n_probe, P))
        probes[0] = u
        probes[1::2] = space.project(u + cfg.fd_eps * dirs)
        probes[2::2] = space.project(u - cfg.fd_eps * dirs)

        losses = self._evaluate(probes, chunks)
        base = losses[0]
        deriv = (losses[1::2] - losses[2::2]) / (2.0 * cfg.fd_eps)
        grad = (deriv[:, None] * dirs).mean(axis=0) * P  # P: Gaussian-smoothing scale

        gnorm = float(np.linalg.norm(grad))
        if not np.isfinite(gnorm) or gnorm < 1e-12:
            return u, {"loss": float(base), "grad_norm": 0.0, "step": 0.0}

        self.velocity = cfg.momentum * self.velocity + grad / gnorm
        alpha = cfg.step0 / (1.0 + iteration / cfg.step_decay)
        direction = self.velocity / (np.linalg.norm(self.velocity) + 1e-12)
        u_new = space.project(u - alpha * direction)
        return u_new, {
            "loss": float(base),
            "grad_norm": gnorm,
            "step": float(alpha),
            "moved": float(np.linalg.norm(u_new - u)),
        }

    @torch.no_grad()
    def optimize(
        self, u: np.ndarray, chunk_sampler, num_iters: int | None = None,
        seed: int = 0, verbose: bool = False, eval_chunks=None,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        history: list[dict[str, float]] = []
        self.velocity = np.zeros(self.space.n)
        # Chunks resampled from a live replay buffer are wildly heterogeneous --
        # some start beside a drop, some are quiet -- so a single SGD iterate can
        # be much worse than an earlier one.  Keeping the best iterate by
        # held-out loss is the usual remedy and, unlike a line search, costs
        # nothing extra: that evaluation is already being made for logging.
        best_u, best_loss = u.copy(), float("inf")
        for it in range(num_iters if num_iters is not None else cfg.num_iters):
            chunks = chunk_sampler()
            if chunks is None:
                break
            u, info = self.step(u, chunks, it, rng)
            info["iter"] = it
            info["param_err_mean"] = float(self.space.error(u).mean())
            if eval_chunks is not None:
                info["val_loss"] = float(self._evaluate(np.repeat(u[None], self.n_probe, 0),
                                                        eval_chunks)[0])
                if info["val_loss"] < best_loss:
                    best_loss, best_u = info["val_loss"], u.copy()
            history.append(info)
            if verbose:
                print(f"  [sysid {it:03d}] loss={info['loss']:.5f} |g|={info['grad_norm']:.3f} "
                      f"a={info['step']:.4f} err={info['param_err_mean']:.4f}", flush=True)
        if eval_chunks is not None and history:
            u = best_u
            history[-1]["val_loss"] = best_loss
            history[-1]["param_err_mean"] = float(self.space.error(u).mean())
        return u, history
