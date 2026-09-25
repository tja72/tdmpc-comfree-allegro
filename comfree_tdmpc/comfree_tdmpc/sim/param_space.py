"""Addressing arbitrary scalar physical parameters, and the coordinates SysID uses.

Four hand-picked parameters are a demonstration; a real Allegro would come with
roughly a hundred genuinely unknown numbers.  This module names them and gives
the optimiser a single well-scaled coordinate system to work in.

Every parameter is mapped to a normalised coordinate u in [0, 1]:

  * multiplicative parameters (a scale on the compiled model's value) are
    log-uniform in u, so a step of 0.1 means the same relative change whether the
    quantity is a mass of 0.14 kg or a contact damping of 1e-3;
  * additive parameters whose nominal value is exactly zero -- joint armature and
    dry friction, which the XML omits but a real geared hand certainly has -- are
    linear in u, because a scale factor on zero carries no information.

Working in u makes the projection a clip to the unit box and makes one step size
meaningful across all parameters (71 handles for the current Allegro-cube model,
via realistic_handles() below), which is what lets the line search go away.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Handle:
    """One scalar physical parameter, and where to write it in the model."""

    field: str  # MuJoCo model array, e.g. "body_mass"
    index: tuple[int, ...]  # index into that array, without the world dimension
    name: str
    group: str
    nominal: float
    multiplicative: bool = True
    low: float = 0.1  # bounds on the scale (multiplicative) or the value (additive)
    high: float = 10.0
    # Some physical quantities live in more than one model entry: a MuJoCo
    # position actuator stores kp in gainprm[0] *and* -kp in biasprm[1], and
    # writing only one of them silently produces an inconsistent actuator.
    # Each alias is scaled by this handle's factor applied to its own nominal
    # value, so signs take care of themselves.
    aliases: tuple[tuple[str, tuple[int, ...], float], ...] = ()
    # Bodies whose mass and inertia move together with this handle.
    extra_bodies: tuple[int, ...] = ()

    @property
    def true_u(self) -> float:
        """Coordinate of the model as compiled (i.e. the value to recover)."""
        if self.multiplicative:
            return _to_u_log(1.0, self.low, self.high)
        return _to_u_lin(self.nominal, self.low, self.high)


def _to_u_log(scale: float, lo: float, hi: float) -> float:
    return float((np.log(scale) - np.log(lo)) / (np.log(hi) - np.log(lo)))


def _to_u_lin(v: float, lo: float, hi: float) -> float:
    return float((v - lo) / (hi - lo))


def _name(m: mujoco.MjModel, objtype, i: int) -> str:
    n = mujoco.mj_id2name(m, objtype, i)
    return n if n else f"#{i}"


def realistic_handles(m: mujoco.MjModel) -> list[Handle]:
    """The parameters a real Allegro-hand cube setup would actually not know."""
    H: list[Handle] = []
    BODY, GEOM = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
    nu = m.nu

    # Per-joint transmission: damping, rotor inertia, dry friction.
    for d in range(nu):
        H.append(Handle("dof_damping", (d,), f"damping/dof{d}", "joint",
                        float(m.dof_damping[d]), True, 0.3, 3.0))
        # The XML omits both of these; a geared finger joint has them.
        H.append(Handle("dof_armature", (d,), f"armature/dof{d}", "joint",
                        float(m.dof_armature[d]), False, 0.0, 2e-3))
        H.append(Handle("dof_frictionloss", (d,), f"dryfric/dof{d}", "joint",
                        float(m.dof_frictionloss[d]), False, 0.0, 1e-2))

    # Servo gains, grouped by joint index within a finger.
    # A real hand is tuned per joint *class*, not per joint: the four fingers are
    # mechanically identical, so 32 independent gains are 24 spurious degrees of
    # freedom in the least identifiable part of the parameter set.
    for j in range(4):
        same = [u for u in range(nu) if u % 4 == j]
        head, *rest = same
        H.append(Handle("actuator_gainprm", (head, 0), f"kp/joint{j}", "actuator",
                        float(m.actuator_gainprm[head, 0]), True, 0.5, 2.0,
                        aliases=(("actuator_biasprm", (head, 1), -1.0),)
                        + tuple(("actuator_gainprm", (u, 0), 1.0) for u in rest)
                        + tuple(("actuator_biasprm", (u, 1), -1.0) for u in rest)))
        H.append(Handle("actuator_biasprm", (head, 2), f"kv/joint{j}", "actuator",
                        float(m.actuator_biasprm[head, 2]), True, 0.5, 2.0,
                        aliases=tuple(("actuator_biasprm", (u, 2), 1.0) for u in rest)))

    # Contact friction, grouped the way a real hand is characterised.
    groups: dict[str, list[int]] = {"ff": [], "mf": [], "rf": [], "th": [], "palm": []}
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        gname = _name(m, GEOM, g)
        if gname in ("obj", "floor"):
            continue
        bname = _name(m, BODY, m.geom_bodyid[g])
        key = next((k for k in ("ff", "mf", "rf", "th") if bname.startswith(k)), "palm")
        groups[key].append(g)
    for key, gids in groups.items():
        if not gids:
            continue
        head, *rest = gids
        H.append(Handle("geom_friction", (head, 0), f"friction/{key}", "contact",
                        float(m.geom_friction[head, 0]), True, 0.3, 3.0,
                        aliases=tuple(("geom_friction", (g, 0), 1.0) for g in rest)))

    # Link masses, one scale per finger.
    finger_bodies: dict[str, list[int]] = {"ff": [], "mf": [], "rf": [], "th": []}
    for b in range(1, m.nbody):
        bname = _name(m, BODY, b)
        for k in finger_bodies:
            if bname.startswith(k) and m.body_mass[b] > 0:
                finger_bodies[k].append(b)
    for key, bids in finger_bodies.items():
        if not bids:
            continue
        head, *rest = bids
        H.append(Handle("body_mass", (head,), f"linkmass/{key}", "inertial",
                        float(m.body_mass[head]), True, 0.7, 1.4,
                        extra_bodies=tuple(bids)))

    # The manipulated object.
    obj_b = mujoco.mj_name2id(m, BODY, "obj")
    obj_g = mujoco.mj_name2id(m, GEOM, "obj")
    H.append(Handle("body_mass", (obj_b,), "cube/mass", "inertial",
                    float(m.body_mass[obj_b]), True, 0.4, 2.5, extra_bodies=(obj_b,)))
    # One isotropic scale, not three independent principal moments.  The cube's
    # inertia *is* isotropic, and scaling the compiled tensor keeps it positive
    # definite and inside the triangle inequality by construction -- three free
    # moments can leave the set of physically realisable rigid bodies entirely.
    H.append(Handle("body_inertia", (obj_b, 0), "cube/inertia", "inertial",
                    float(m.body_inertia[obj_b, 0]), True, 0.5, 2.0,
                    aliases=(("body_inertia", (obj_b, 1), 1.0),
                             ("body_inertia", (obj_b, 2), 1.0))))
    H.append(Handle("geom_friction", (obj_g, 0), "cube/friction", "contact",
                    float(m.geom_friction[obj_g, 0]), True, 0.3, 3.0))
    H.append(Handle("geom_friction", (obj_g, 1), "cube/torsion", "contact",
                    float(m.geom_friction[obj_g, 1]), True, 0.3, 3.0))

    # The contact model itself.
    H.append(Handle("comfree_stiffness", (), "comfree/stiffness", "comfree",
                    0.1, True, 0.3, 3.0))
    H.append(Handle("comfree_damping", (), "comfree/damping", "comfree",
                    1e-3, True, 0.3, 3.0))
    return H


class ParamSpace:
    """A list of handles plus the u-coordinates the optimiser moves in."""

    def __init__(self, handles: list[Handle]):
        self.handles = handles
        self.n = len(handles)
        self.true_u = np.array([h.true_u for h in handles], dtype=np.float64)
        self.multiplicative = np.array([h.multiplicative for h in handles])
        self.low = np.array([h.low for h in handles], dtype=np.float64)
        self.high = np.array([h.high for h in handles], dtype=np.float64)
        self.nominal = np.array([h.nominal for h in handles], dtype=np.float64)

    @classmethod
    def realistic(cls, m: mujoco.MjModel) -> "ParamSpace":
        return cls(realistic_handles(m))

    def values(self, u: np.ndarray) -> np.ndarray:
        """Physical values for a coordinate vector (scale factors where multiplicative)."""
        u = np.clip(u, 0.0, 1.0)
        out = np.empty(self.n)
        mul = self.multiplicative
        out[mul] = np.exp(
            np.log(self.low[mul]) + u[mul] * (np.log(self.high[mul]) - np.log(self.low[mul]))
        )
        out[~mul] = self.low[~mul] + u[~mul] * (self.high[~mul] - self.low[~mul])
        return out

    def project(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, 0.0, 1.0)

    def perturb(self, rng: np.random.Generator, scale: float = 0.25) -> np.ndarray:
        """A wrong initial guess: displace every coordinate in u-space."""
        return self.project(self.true_u + rng.uniform(-scale, scale, self.n))

    def sample_stable(
        self, rng: np.random.Generator, scale: float, is_stable, max_tries: int = 25
    ) -> np.ndarray:
        """Draw a parameter vector that actually simulates.

        A real plant is stable; a uniform draw in the box is not necessarily so
        (a stiff servo with little damping diverges under explicit integration).
        Rejection sampling keeps the physical realism without hand-tuning the
        bounds until nothing interesting is left in them.
        """
        for attempt in range(max_tries):
            u = self.perturb(rng, scale * (0.85 ** (attempt // 5)))
            if is_stable(u):
                return u
        return self.true_u.copy()

    def error(self, u: np.ndarray) -> np.ndarray:
        """Per-parameter error, in the same normalised units for all parameters."""
        return np.abs(np.asarray(u) - self.true_u)

    def groups(self) -> dict[str, np.ndarray]:
        out: dict[str, list[int]] = {}
        for i, h in enumerate(self.handles):
            out.setdefault(h.group, []).append(i)
        return {k: np.array(v) for k, v in out.items()}

    def describe(self) -> str:
        counts: dict[str, int] = {}
        for h in self.handles:
            counts[h.group] = counts.get(h.group, 0) + 1
        return f"{self.n} parameters: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
