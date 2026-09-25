"""MuJoCo contact/constraint solver settings for the sim2sim sweep.

`default` is the scene exactly as compiled (Newton, pyramidal cone, MuJoCo's
default solref (0.02, 1) / solimp (0.9, 0.95, 0.001, 0.5, 2) on every geom,
no noslip) and gives the headline number.  The sweep moves one knob at a time
away from it to show how sensitive the gap is to the solver configuration --
it is deliberately *not* a calibration of MuJoCo towards ComFree.

solref/solimp overrides touch contact parameters only (`geom_solref`,
`geom_solimp`, which MuJoCo mixes per contact pair); joint-limit and
equality constraints keep their compiled values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco


@dataclass(frozen=True)
class SolverConfig:
    name: str
    cone: str | None = None  # "pyramidal" | "elliptic"; None = as compiled
    solref_timeconst: float | None = None  # geom_solref[:, 0]
    solimp: tuple[float, float, float] | None = None  # geom_solimp[:, :3]
    noslip_iterations: int | None = None

    def apply(self, m: mujoco.MjModel) -> None:
        if self.cone is not None:
            m.opt.cone = {
                "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
            }[self.cone]
        if self.solref_timeconst is not None:
            m.geom_solref[:, 0] = self.solref_timeconst
        if self.solimp is not None:
            m.geom_solimp[:, :3] = self.solimp
        if self.noslip_iterations is not None:
            m.opt.noslip_iterations = self.noslip_iterations

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = SolverConfig("default")

SWEEP: list[SolverConfig] = [
    DEFAULT,
    SolverConfig("cone_elliptic", cone="elliptic"),
    SolverConfig("solref_tc0.01", solref_timeconst=0.01),  # stiffer (>= 2*dt = 0.004)
    SolverConfig("solref_tc0.04", solref_timeconst=0.04),  # softer
    SolverConfig("solimp_hard", solimp=(0.95, 0.99, 0.001)),
    SolverConfig("solimp_soft", solimp=(0.8, 0.9, 0.001)),
    SolverConfig("noslip10", noslip_iterations=10),
]

BY_NAME = {c.name: c for c in SWEEP}


def model_solver_summary(m: mujoco.MjModel) -> dict:
    """What the compiled model actually uses, for config.json."""
    import numpy as np

    return {
        "timestep": float(m.opt.timestep),
        "cone": mujoco.mjtCone(m.opt.cone).name,
        "solver": mujoco.mjtSolver(m.opt.solver).name,
        "integrator": mujoco.mjtIntegrator(m.opt.integrator).name,
        "iterations": int(m.opt.iterations),
        "noslip_iterations": int(m.opt.noslip_iterations),
        "ccd_iterations": int(m.opt.ccd_iterations),
        "impratio": float(m.opt.impratio),
        "geom_solref_unique": np.unique(m.geom_solref, axis=0).tolist(),
        "geom_solimp_unique": np.unique(m.geom_solimp, axis=0).tolist(),
    }
