"""P1 correctness fixes layered on top of the merged v0.3.2 P0 solver.

P1-1: Solver restart state must preserve physical time.  The legacy
checkpoint format stored velocity/pressure and numerical parameters but not
``Solver.time``; resuming therefore reset the clock to zero.

P1-2: An ``initial_condition`` callable must be applied once, not every time
``Solver.solve()`` is called.  Repeated ``solve()`` calls are a continuation of
the current state; re-applying the initial condition silently discarded the
progress from the previous run.

The fixes are isolated here so they can be reviewed and merged independently
from the numerical P0 changes.
"""

from __future__ import annotations

import os
import sys
import numpy as np

from . import solver as solver_module
from .p0_fixes import P0Solver


class P1Solver(P0Solver):
    """P0 solver plus restart/time and initial-condition lifecycle fixes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_condition_applied = False

    def solve(self, *args, **kwargs):
        """Run from the current state without re-applying an initial condition.

        The base solver owns the actual time-stepping implementation. We let
        it perform the initial-condition application on the first call, then
        detach the callable so later calls continue from the current state.
        """
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
        """Restore a checkpoint and preserve its physical simulation time.

        Checkpoints created before P1 did not contain ``time``. Those files
        remain readable, but are restored at ``t=0`` with a warning because the
        historical physical time cannot be reconstructed from the stored data.
        """
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

        # A restored state is already initialized. Never let a supplied
        # initial-condition callable overwrite the checkpoint on the next
        # solve() call.
        solver._initial_condition_fn = None
        solver._initial_condition_applied = True
        return solver


# Keep direct submodule imports consistent with the package-level Solver.
solver_module.Solver = P1Solver
