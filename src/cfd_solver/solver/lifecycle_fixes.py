"""P1 correctness fixes layered on top of the merged v0.3.2 P0 solver.

P1 fixes:
- preserve physical simulation time across checkpoints;
- apply an initial condition only once across repeated solve() calls;
- preserve the P0 CN/Inlet/Outlet safety guard;
- preserve the legacy PeriodicPressureSolver(mesh) API;
- make inlet profiles follow the actual inlet direction and domain size.
"""

from __future__ import annotations

import os
import sys
import numpy as np

from . import solver as solver_module
from . import pressure as pressure_module
from .numerical_fixes import (
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


def _p1_inlet_profile(self, wall, N, axis="x"):
    """Return an inlet profile at face positions along the inlet."""
    if wall.profile == "uniform":
        return np.full(N, wall.U_max, dtype=float)
    if wall.profile != "parabolic":
        raise ValueError(f"Unknown inlet profile: {wall.profile!r}")

    if axis == "x":
        length = float(getattr(self, "_domain_height", 1.0))
    elif axis == "y":
        length = float(getattr(self, "_domain_width", 1.0))
    else:
        raise ValueError(f"Unknown inlet axis: {axis!r}")
    if length <= 0.0:
        raise ValueError(f"Inlet span must be positive, got {length}")

    coordinate = (np.arange(N, dtype=float) + 0.5) * length / N
    eta = coordinate / length
    return 4.0 * wall.U_max * eta * (1.0 - eta)


from . import bc as bc_module
from .bc import InletWall, OutletWall


def _inlet_top(self, u, v, Nx, Ny, bc):
    v[1:-1, Ny] = bc._inlet_profile(self, Nx, axis="y")
    u[:, -1] = -u[:, -2]


def _inlet_bottom(self, u, v, Nx, Ny, bc):
    v[1:-1, 0] = bc._inlet_profile(self, Nx, axis="y")
    u[:, 0] = -u[:, 1]


def _inlet_left(self, u, v, Nx, Ny, bc):
    u[0, 1:-1] = bc._inlet_profile(self, Ny, axis="x")
    v[0, :] = -v[1, :]


def _inlet_right(self, u, v, Nx, Ny, bc):
    u[Nx, 1:-1] = bc._inlet_profile(self, Ny, axis="x")
    v[-1, :] = -v[-2, :]


InletWall.apply_top = _inlet_top
InletWall.apply_bottom = _inlet_bottom
InletWall.apply_left = _inlet_left
InletWall.apply_right = _inlet_right
bc_module.BoundaryConditions._inlet_profile = _p1_inlet_profile


# P0Solver's immediate base is the original Solver implementation. P1 uses it
# to retain the P0 numerical changes while adding lifecycle fixes.
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
