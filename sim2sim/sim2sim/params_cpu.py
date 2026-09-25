"""Write a ParamSpace u-vector into a CPU `MjModel`, and check it against ComFree.

Mirrors `comfree_tdmpc.sim.handle_writer.HandleWriter.write` step for step,
on the host model instead of the per-world warp arrays:

  1. `space.values(u)` (ParamSpace's own u -> physical mapping);
  2. every handle target *and alias* gets `value * own_nominal` (multiplicative)
     or `value` (additive), nominals read from the freshly compiled model,
     fields processed in the same order as HandleWriter's plan dict;
  3. afterwards, for handles with `extra_bodies`, mass and inertia of those
     bodies are reset to `base * scale` (this runs last in HandleWriter too,
     so e.g. the cube's inertia ends up following cube/mass);
  4. `mj_setConst`, the CPU counterpart of `mjwarp.set_const`.

The two ComFree contact-model handles have no MuJoCo equivalent and are
skipped.  `check_parity` then compares every written field, whole array,
against world 0 of a ComFree env holding the same u.
"""

from __future__ import annotations

import numpy as np

from comfree_tdmpc.sim.batch_sim import load_spec
from comfree_tdmpc.sim.param_space import ParamSpace

COMFREE_ONLY_FIELDS = ("comfree_stiffness", "comfree_damping")


class CpuHandleWriter:
    """HandleWriter for one host MjModel.  `write_u` matches HandleWriter.write_u."""

    def __init__(self, sim, space: ParamSpace, verbose: bool = True):
        # `sim` is a MujocoBatchSim (needs .mjm and .model_changed()).
        self.sim = sim
        self.space = space
        self.base = load_spec(timestep=sim.cfg.timestep)  # nominal values, never written
        self.skipped: list[str] = []
        self.plans: dict[str, list[tuple[tuple[int, ...], int]]] = {}
        self.mass_bodies: list[tuple[int, np.ndarray]] = []
        for p, h in enumerate(space.handles):
            if h.field in COMFREE_ONLY_FIELDS:
                self.skipped.append(h.name)
                continue
            for fname, idx, _sign in [(h.field, h.index, 1.0)] + list(h.aliases):
                self.plans.setdefault(fname, []).append((tuple(idx), p))
            if h.extra_bodies:
                self.mass_bodies.append((p, np.asarray(h.extra_bodies)))
        self.written_fields = sorted(set(self.plans) | ({"body_mass", "body_inertia"} if self.mass_bodies else set()))
        if verbose and self.skipped:
            print(f"[sim2sim] skipping {len(self.skipped)} ComFree-only params with no MuJoCo "
                  f"equivalent: {', '.join(self.skipped)}", flush=True)

    def write(self, values: np.ndarray) -> None:
        m, base = self.sim.mjm, self.base
        mul = self.space.multiplicative
        for fname, entries in self.plans.items():
            arr, nom = getattr(m, fname), getattr(base, fname)
            for idx, p in entries:
                arr[idx] = values[p] * nom[idx] if mul[p] else values[p]
        for p, bodies in self.mass_bodies:
            m.body_mass[bodies] = base.body_mass[bodies] * values[p]
            m.body_inertia[bodies] = base.body_inertia[bodies] * values[p]
        self.sim.model_changed()

    def write_u(self, u: np.ndarray, recompute_const: bool = True) -> None:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 2:  # (nworld, P) -- shared parameters only
            assert np.all(u == u[:1]), "CPU backend holds one parameter set for all lanes"
            u = u[0]
        self.write(self.space.values(u))


def _gpu_field(comfree_sim, name: str) -> np.ndarray:
    import warp as wp

    arr = wp.to_torch(getattr(comfree_sim.m, name)).detach().cpu().numpy().astype(np.float64)
    host = np.asarray(getattr(comfree_sim.mjm, name))
    if arr.ndim == host.ndim + 1:  # per-world (or broadcast leading dim 1)
        arr = arr[0]
    return arr.reshape(host.shape)


def check_parity(
    cpu_sim,
    comfree_sim,
    fields: list[str],
    rtol: float = 2e-6,
    atol: float = 1e-9,
    derived: tuple[str, ...] = ("body_subtreemass", "dof_invweight0", "body_invweight0"),
) -> dict:
    """Compare every field in `fields` (and set_const-derived ones) CPU vs ComFree world 0.

    ComFree stores float32, MuJoCo float64, hence the relative tolerance.
    Returns {"ok": bool, "fields": {name: {"max_abs": .., "max_rel": .., "ok": ..}}}.
    """
    out: dict = {"ok": True, "fields": {}, "derived": {}}
    for group, names in (("fields", fields), ("derived", derived)):
        for name in names:
            try:
                g = _gpu_field(comfree_sim, name)
            except (AttributeError, ValueError) as e:
                out[group][name] = {"skipped": str(e)}
                continue
            c = np.asarray(getattr(cpu_sim.mjm, name), dtype=np.float64)
            diff = np.abs(c - g)
            rel = diff / np.maximum(np.abs(g), atol)
            ok = bool(np.all(diff <= atol + rtol * np.abs(g)))
            rec = {"max_abs": float(diff.max(initial=0.0)), "max_rel": float(rel.max(initial=0.0)), "ok": ok}
            if not ok:
                bad = np.argwhere(diff > atol + rtol * np.abs(g))[:5]
                rec["first_bad"] = [
                    {"index": b.tolist(), "cpu": float(c[tuple(b)]), "comfree": float(g[tuple(b)])} for b in bad
                ]
            out[group][name] = rec
            # set_const-derived arrays are informative: ComFree may define them differently.
            if group == "fields":
                out["ok"] &= ok
    return out
