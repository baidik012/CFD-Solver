"""P1 correctness fixes layered on top of the merged v0.3.2 P0 solver.

P1 fixes:
- preserve physical simulation time across checkpoints;
- apply an initial condition only once across repeated solve() calls;
- keep the P0 periodic pressure/velocity fixes without inheriting the P0
  constructor's over-restrictive CN/Inlet/Outlet guard;
- preserve the legacy PeriodicPressureSolver(mesh) API while allowing the
  generalized factory to return the same compatible class;
- make parabolic inlet profiles use the actual domain height.
"""

from __future__ import annotations

import os
import sys
import numpy as np

from . import solver as solver_module
from . import pressure as pressure_module
from .p0_fixes import (
    P0Solver,
    GeneralPeriodicPressureSolver,
    _periodic_flags,
    _wrap_periodic_advection,
    create_pressure_solver_p0,
)


class LegacyPeriodicPressureSolver(GeneralPeriodicPressureSolver):
    """Backward-compatible periodic pressure solver API."""

    def __init__(self, mesh, periodic_x=True, periodic_y=False):
        super().__init__(mesh, periodic_x=periodic_x, periodic_y=periodic_y)


def create_pressure_solver_p1(mesh, bc=None):
    """P0 pressure factory with legacy-class compatibility."""
    if bc is not None:
        px, py = _periodic_flags(bc)
        if px or py:
            if bc.has_outlet():
                raise ValueError(
                    "OutletWall cannot be combined with periodic pressure topology"
                )
            return LegacyPeriodicPressureSolver(mesh, px, py)
    return create_pressure_solver_p0(mesh, bc)


pressure_module.PeriodicPressureSolver = LegacyPeriodicPressureSolver
pressure_module.create_pressure_solver = create_pressure_solver_p1
solver_module.create_pressure_solver = create_pressure_solver_p1


def _p1_inlet_profile(self, wall, Ny, axis="x"):
    """Return an inlet profile using the physical domain height."""
    if wall.profile == "uniform":
        return np.full(Ny, wall.U_max, dtype=float)
    if wall.profile != "parabolic":
        raise ValueError(f"Unknown inlet profile: {wall.profile!r}")

    H = float(getattr(self, "_domain_height", 1.0))
    if H <= 0.0:
        raise ValueError(f"Domain height must be positive, got {H}")
    y = (np.arange(Ny, dtype=float) + 0.5) * H / Ny
    eta = y / H
    return 4.0 * wall.U_max * eta * (1.0 - eta)


from . import bc as bc_module
bc_module.BoundaryConditions._inlet_profile = _p1_inlet_profile


# p0_fixes has replaced solver_module.Solver with P0Solver. Its immediate base
# is the original Solver implementation, which is what P1 needs in order to
# bypass only the accidental P0 CN/Inlet/Outlet constructor guard.
_BaseSolver = P0Solver.__mro__[1]


class P1Solver(P0Solver):
    """P0 solver plus restart/time and initial-condition lifecycle fixes."""

    def __init__(self, *args, **kwargs):
        boundary_config = kwargs.get("boundary_config")
        if boundary_config is not None:
            px, py = _periodic_flags(boundary_config)
        else:
            px = py = False

        _BaseSolver.__init__(self, *args, **kwargs)

        self._periodic_x = px
        self._periodic_y = py
        if px or py:
            self._advection_fn = _wrap_periodic_advection(
                self._advection_fn, px, py
            )
        self._initial_condition_applied = False

    def solve(self, *args, **kwargs):
        """Run from the current state without re-applying an initial condition."""
        initial_condition = self._initial_condition_fn
        result = super().solve(*args, **kwargs)
        if initial_condition is not None:
            self._initial_condition_fn = None
            self._initial_condition_applied = True
        return result

    def checkpoint(self, path):
        """Save the solver state including the physical simulation time."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            path,
            u=self.u,
            v=self.v,
            p=self.p,
            Nx=self.Nx,
            Ny=self.Ny,
            Lx=self.Lx,
            Ly=self.Ly,
            dt=self.dt,
            nu=self.nu,
            time=self.time,
            lid_speed=self.bc.top,
            smooth_lid=self.bc.smooth_lid,
            advection_scheme=self.advection_scheme,
            diffusion_scheme=self.diffusion_scheme,
        )

    @classmethod
    def from_checkpoint(cls, path, force=False, body_force=None,
                        initial_condition=None, boundary_config=None):
        """Restore a checkpoint and preserve its physical simulation time."""
        solver = super().from_checkpoint(
            path,
            force=force,
            body_force=body_force,
            initial_condition=initial_condition,
            boundary_config=boundary_config,
        )

        with np.load(path) as data:
            if "time" in data:
                time_value = float(data["time"])
                if not np.isfinite(time_value) or time_value < 0.0:
                    raise ValueError(
                        f"Checkpoint contains invalid simulation time: {time_value!r}"
                    )
                solver.time = time_value
            elif not force:
                print(
                    "  [WARNING] from_checkpoint: checkpoint has no 'time' field.\n"
                    "            This is a pre-P1 checkpoint, so physical time is\n"
                    "            restored as 0.0 because the original time is unavailable.",
                    file=sys.stderr,
                )

        solver._initial_condition_fn = None
        solver._initial_condition_applied = True
        return solver


solver_module.Solver = P1Solver