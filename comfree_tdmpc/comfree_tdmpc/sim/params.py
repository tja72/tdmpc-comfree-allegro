"""Physical parameters exposed to system identification.

Every parameter is strictly positive and optimised in log-space, so the
projected-gradient step is well conditioned across parameters that differ by
orders of magnitude (contact damping ~1e-3 vs. friction ~1e0) and positivity is
guaranteed by construction.  Box bounds are therefore also stored in log-space.

A note on friction: MuJoCo resolves the friction of a contact as the elementwise
max of the two colliding geoms' friction.  The Allegro hand geoms inherit the
default slide friction of 1.0 while the cube XML sets 0.5, so perturbing only the
cube would leave every hand/cube contact at 1.0 and the parameter would be
unidentifiable.  `friction` therefore writes the slide friction of the cube
*and* of every hand collision geom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    low: float
    high: float
    # Human-readable description of what the parameter does physically.
    doc: str


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("cube_mass_scale", 1.0, 0.25, 4.0, "multiplies cube mass and inertia"),
    ParamSpec("friction", 1.0, 0.1, 2.0, "slide friction of cube and hand collision geoms"),
    ParamSpec("comfree_stiffness", 0.1, 0.01, 1.0, "ComFree contact stiffness"),
    ParamSpec("comfree_damping", 1e-3, 1e-4, 5e-2, "ComFree contact damping"),
)

PARAM_NAMES: tuple[str, ...] = tuple(s.name for s in PARAM_SPECS)
NUM_PARAMS = len(PARAM_SPECS)


def default_params() -> np.ndarray:
    return np.array([s.default for s in PARAM_SPECS], dtype=np.float64)


def log_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.log(np.array([s.low for s in PARAM_SPECS], dtype=np.float64))
    hi = np.log(np.array([s.high for s in PARAM_SPECS], dtype=np.float64))
    return lo, hi


class ParamVector:
    """A parameter vector held in log-space with box projection."""

    def __init__(self, values: Sequence[float] | np.ndarray | None = None):
        v = default_params() if values is None else np.asarray(values, dtype=np.float64)
        assert v.shape == (NUM_PARAMS,), f"expected {NUM_PARAMS} params, got {v.shape}"
        self._log = np.log(v)
        self.project_()

    @property
    def log(self) -> np.ndarray:
        return self._log.copy()

    @property
    def value(self) -> np.ndarray:
        return np.exp(self._log)

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(PARAM_NAMES, self.value)}

    def set_log(self, log_values: np.ndarray) -> "ParamVector":
        self._log = np.asarray(log_values, dtype=np.float64).copy()
        self.project_()
        return self

    def project_(self) -> "ParamVector":
        lo, hi = log_bounds()
        self._log = np.clip(self._log, lo, hi)
        return self

    def copy(self) -> "ParamVector":
        out = ParamVector()
        out._log = self._log.copy()
        return out

    def __repr__(self) -> str:
        items = ", ".join(f"{n}={v:.4g}" for n, v in self.as_dict().items())
        return f"ParamVector({items})"
