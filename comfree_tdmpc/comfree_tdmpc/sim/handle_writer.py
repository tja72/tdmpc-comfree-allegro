"""Write a whole parameter vector into the world bank, per world, in one pass."""

from __future__ import annotations

import numpy as np
import torch

from comfree_tdmpc.sim.batch_sim import BatchSim
from comfree_tdmpc.sim.param_space import ParamSpace


class HandleWriter:
    """Scatters u-coordinates into the simulator's per-world model arrays.

    All the index arithmetic is precomputed, so writing 71 parameters for 1000
    worlds is a handful of scatter operations rather than a Python loop.
    """

    def __init__(self, sim: BatchSim, space: ParamSpace):
        self.sim = sim
        self.space = space
        self.device = sim.torch_device

        # target_field -> (row indices into the flattened per-world view,
        #                  parameter index each row comes from,
        #                  sign, nominal value)
        plans: dict[str, list[tuple[int, int, float, float]]] = {}
        self.mass_bodies: list[tuple[int, np.ndarray]] = []

        for p, h in enumerate(space.handles):
            targets = [(h.field, h.index, 1.0)] + list(h.aliases)
            for fname, idx, sign in targets:
                view = sim.field_view(fname)
                dims = tuple(view.shape[1:])
                flat = int(np.ravel_multi_index(idx, dims)) if dims else 0
                nominal = float(sim.host_field(fname)[idx]) if dims else h.nominal
                plans.setdefault(fname, []).append((flat, p, sign, nominal))
            if h.extra_bodies:
                self.mass_bodies.append((p, np.asarray(h.extra_bodies)))

        self.plans = {}
        for fname, entries in plans.items():
            rows = torch.tensor([e[0] for e in entries], dtype=torch.long, device=self.device)
            pidx = torch.tensor([e[1] for e in entries], dtype=torch.long, device=self.device)
            sign = torch.tensor([e[2] for e in entries], dtype=torch.float32, device=self.device)
            nom = torch.tensor([e[3] for e in entries], dtype=torch.float32, device=self.device)
            self.plans[fname] = (rows, pidx, sign, nom)

        self.multiplicative = torch.tensor(
            space.multiplicative, dtype=torch.bool, device=self.device
        )
        # Inertia of the bodies whose mass is scaled must follow the mass.
        self.base_inertia = torch.as_tensor(
            sim.mjm.body_inertia, dtype=torch.float32, device=self.device
        )
        self.base_mass = torch.as_tensor(
            sim.mjm.body_mass, dtype=torch.float32, device=self.device
        )
        self.needs_const = any(
            f in self.plans for f in ("body_mass", "body_inertia", "dof_armature")
        )
        # Every field is materialised by now; the step graph must be rebuilt so it
        # points at the arrays this writer will actually write to.
        sim.recapture()

    @torch.no_grad()
    def write(self, values: torch.Tensor, recompute_const: bool = True) -> None:
        """`values` is (nworld, P) in physical units (scale factors where multiplicative)."""
        sim = self.sim
        assert values.shape == (sim.nworld, self.space.n), values.shape
        values = values.to(self.device, torch.float32)

        for fname, (rows, pidx, sign, nom) in self.plans.items():
            view = sim.field_view(fname)
            flat = view.reshape(sim.nworld, -1) if view.dim() > 1 else view.reshape(sim.nworld, 1)
            v = values[:, pidx]
            mul = self.multiplicative[pidx]
            out = torch.where(mul.unsqueeze(0), v * nom.unsqueeze(0), v)
            # No extra sign is applied: an alias target carries its own nominal
            # value, which already has the right sign (biasprm[1] is -kp), so
            # scaling it by the same factor preserves the relationship.
            flat[:, rows] = out

        # Scaling a body's mass without its inertia changes the model in a way no
        # physical change can: keep the two together.
        if self.mass_bodies:
            inertia = sim.field_view("body_inertia")
            mass = sim.field_view("body_mass")
            for p, bodies in self.mass_bodies:
                scale = values[:, p].unsqueeze(-1)
                bidx = torch.as_tensor(bodies, dtype=torch.long, device=self.device)
                mass[:, bidx] = self.base_mass[bidx].unsqueeze(0) * scale
                inertia[:, bidx] = self.base_inertia[bidx].unsqueeze(0) * scale.unsqueeze(-1)

        if recompute_const and self.needs_const:
            from comfree_warp import mujoco_warp as mjwarp

            mjwarp.set_const(sim.m, sim.d)

    @torch.no_grad()
    def write_u(self, u: np.ndarray | torch.Tensor, recompute_const: bool = True) -> None:
        """Convenience: write a (nworld, P) or (P,) coordinate vector."""
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        vals = np.stack([self.space.values(row) for row in u])
        if vals.shape[0] == 1:
            vals = np.repeat(vals, self.sim.nworld, axis=0)
        self.write(torch.as_tensor(vals, dtype=torch.float32, device=self.device), recompute_const)
