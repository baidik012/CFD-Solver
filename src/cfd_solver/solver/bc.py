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
        """Set boundary values on u (Nx+1, Ny+2) and v (Nx+2, Ny+1).

        Physical boundaries:
        - u-normal (left/right walls) is defined at u[0, 1:-1] and u[Nx, 1:-1].
        - v-normal (bottom/top walls) is defined at v[1:-1, 0] and v[1:-1, Ny].
        - u-tangential (bottom/top walls) is set via ghost cells u[:, 0] and u[:, -1].
        - v-tangential (left/right walls) is set via ghost cells v[0, :] and v[-1, :].
        """
        # 1. Normal velocities (no penetration: zero at walls)
        u[0, 1:-1] = 0.0
        u[Nx, 1:-1] = 0.0
        v[1:-1, 0] = 0.0
        v[1:-1, Ny] = 0.0

        # 2. Tangential velocity u at top/bottom walls (via ghost cells)
        u[:, 0] = 2.0 * self.bottom - u[:, 1]
        if self.smooth_lid:
            u[:, -1] = 2.0 * self._get_lid_profile(Nx) - u[:, -2]
        else:
            u[:, -1] = 2.0 * self.top - u[:, -2]

        # 3. Tangential velocity v at left/right walls (via ghost cells)
        v[0, :] = 2.0 * self.left - v[1, :]
        v[-1, :] = 2.0 * self.right - v[-2, :]

    def apply_pressure(self, p):
        """Set pressure ghost cells using Neumann BC (zero gradient).

        p shape is (Nx+2, Ny+2).
        """
        p[0, :] = p[1, :]    # left wall
        p[-1, :] = p[-2, :]  # right wall
        p[:, 0] = p[:, 1]    # bottom wall
        p[:, -1] = p[:, -2]  # top wall

    def lid_values(self, Nx: int):
        """Return the lid u-values at u-face positions (length Nx+1)."""
        if self.smooth_lid:
            return self._get_lid_profile(Nx)
        return self.top

