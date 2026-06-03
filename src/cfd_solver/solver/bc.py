"""Boundary conditions for staggered incompressible flow.

Supports constant wall velocities and optional smooth lid profiles
(sinusoidal) to avoid corner singularities in lid-driven cavity problems.
"""

import numpy as np


class BoundaryConditions:
    """Velocity boundary conditions on a staggered grid.

    Parameters
    ----------
    top, bottom, left, right : float
        Tangential u-velocity on each wall.
    smooth_lid : bool
        If True, apply a sinusoidal lid profile u(x) = U * sin(pi*x/L)
        instead of a uniform step function. Removes the corner singularity.
    """

    def __init__(self, top: float = 1.0, bottom: float = 0.0,
                 left: float = 0.0, right: float = 0.0,
                 smooth_lid: bool = False):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.smooth_lid = smooth_lid

        self._lid_profile = None

    def _get_lid_profile(self, Nx: int):
        """Return the sinusoidal lid profile array of length Nx+1."""
        if self._lid_profile is None or len(self._lid_profile) != Nx + 1:
            self._lid_profile = self.top * np.sin(np.pi * np.arange(Nx + 1) / Nx)
        return self._lid_profile

    def apply(self, u, v, Nx: int, Ny: int):
        """Set boundary values on u (Nx+1, Ny) and v (Nx, Ny+1).

        u is defined on all walls. v is only defined on top/bottom walls
        (j=0, j=Ny); it is not defined on left/right walls.
        """
        # u on all walls
        u[0, :] = self.left
        u[Nx, :] = self.right
        u[:, 0] = self.bottom

        if self.smooth_lid:
            u[:, Ny - 1] = self._get_lid_profile(Nx)
        else:
            u[:, Ny - 1] = self.top

        # v on top and bottom only
        v[:, 0] = self.bottom
        v[:, Ny] = 0.0

    def lid_values(self, Nx: int):
        """Return the lid u-values at u-face positions (length Nx+1).

        Used by the diffusion solver for Dirichlet boundary contributions.
        """
        if self.smooth_lid:
            return self._get_lid_profile(Nx)
        return self.top
