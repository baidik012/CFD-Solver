"""Post-merge P0 compatibility corrections.

These corrections keep the P0 numerical fixes while restoring functionality
that the first P0 implementation accidentally regressed:

* Crank-Nicolson remains available for InletWall/OutletWall because the
  audited wall ghost-cell hooks already encode the supported implicit BCs.
* The historical PeriodicPressureSolver(mesh) public type remains usable and
  the generalized factory returns a subclass of GeneralPeriodicPressureSolver
  so existing isinstance checks continue to work.
"""

from __future__ import annotations

from . import pressure as pressure_module
from . import solver as solver_module
from .p0_fixes import (
    P0Solver,
    GeneralPeriodicPressureSolver,
    _periodic_flags,
    _wrap_periodic_advection,
    create_pressure_solver_p0,
)

# p0_fixes replaces solver_module.Solver with P0Solver. Recover the original
# numerical Solver implementation through P0Solver's immediate base class so
# this compatibility layer can bypass only the erroneous P0 constructor guard.
_BaseSolver = P0Solver.__mro__[1]


class CompatiblePeriodicPressureSolver(GeneralPeriodicPressureSolver):
    """General periodic pressure solver with the legacy one-argument API."""

    def __init__(self, mesh, periodic_x=True, periodic_y=False):
        super().__init__(mesh, periodic_x=periodic_x, periodic_y=periodic_y)


def create_pressure_solver_compat(mesh, bc=None):
    if bc is not None:
        px, py = _periodic_flags(bc)
        if px or py:
            if bc.has_outlet():
                raise ValueError(
                    "OutletWall cannot be combined with periodic pressure topology"
                )
            return CompatiblePeriodicPressureSolver(mesh, px, py)
    return create_pressure_solver_p0(mesh, bc)


class CorrectedP0Solver(P0Solver):
    """P0 solver with the accidental CN/Inlet/Outlet rejection removed."""

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


pressure_module.PeriodicPressureSolver = CompatiblePeriodicPressureSolver
pressure_module.create_pressure_solver = create_pressure_solver_compat
solver_module.create_pressure_solver = create_pressure_solver_compat
solver_module.Solver = CorrectedP0Solver
