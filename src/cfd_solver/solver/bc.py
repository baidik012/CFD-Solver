"""Boundary conditions for staggered incompressible flow.

This module handles the enforcement of physical boundary conditions on the
staggered Arakawa C-grid. Because velocity components are stored at different
locations, different strategies are used for normal and tangential components.

Velocity BCs:
-------------
1. Normal Velocity: Specified directly at the boundary faces. For a no-penetration
   wall, the velocity component normal to the wall is set to zero.
2. Tangential Velocity: Specified at the wall, but because the staggered
   component (e.g., 'u' at a top/bottom wall) is stored half a cell away, we
   use "ghost cells" and linear interpolation to enforce the value.
   To set velocity V_wall at the boundary:
       (V_interior + V_ghost) / 2 = V_wall  =>  V_ghost = 2*V_wall - V_interior

Pressure BCs:
-------------
We typically use Neumann (zero-gradient) boundary conditions for pressure,
implying that the pressure at the ghost cell is equal to the pressure at the
adjacent interior cell: p_ghost = p_interior.
"""

import numpy as np


class BoundaryConditions:
    """Velocity boundary conditions on a staggered grid.

    Parameters
    ----------
    top : float, optional
        Tangential u-velocity on the top wall (default 1.0).
    bottom : float, optional
        Tangential u-velocity on the bottom wall (default 0.0).
    left : float, optional
        Tangential v-velocity on the left wall (default 0.0).
    right : float, optional
        Tangential v-velocity on the right wall (default 0.0).
    smooth_lid : bool, optional
        If True, apply a sinusoidal lid profile u(x) = U * sin(pi*x/L)
        instead of a uniform step function. This avoids the numerical
        singularities (infinite stress) at the top corners of a lid-driven
        cavity (default False).
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
        self._lid_profile_key = None

    def _get_lid_profile(self, Nx: int):
        """Return the sinusoidal lid profile array.

        Parameters
        ----------
        Nx : int
            Number of cells in x.

        Returns
        -------
        ndarray
            Sinusoidal velocity profile of length Nx+1.
        """
        key = (Nx, self.top)
        if self._lid_profile is None or self._lid_profile_key != key:
            self._lid_profile = self.top * np.sin(np.pi * np.arange(Nx + 1) / Nx)
            self._lid_profile_key = key
        return self._lid_profile

    def apply(self, u, v, Nx: int, Ny: int):
        """Set boundary values on velocity components.

        Parameters
        ----------
        u : ndarray, shape (Nx+1, Ny+2)
            u-velocity array including ghost cells.
        v : ndarray, shape (Nx+2, Ny+1)
            v-velocity array including ghost cells.
        Nx, Ny : int
            Number of cells in each direction.
        """
        # 1. Normal velocities: Set to 0 at the physical boundary faces
        # u-normal at left (i=0) and right (i=Nx) walls
        u[0, 1:-1] = 0.0
        u[Nx, 1:-1] = 0.0
        # v-normal at bottom (j=0) and top (j=Ny) walls
        v[1:-1, 0] = 0.0
        v[1:-1, Ny] = 0.0

        # 2. Tangential velocity u at top/bottom walls (via ghost cells)
        # u[i, 0] is the bottom ghost cell, u[i, 1] is the first interior cell.
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

        Parameters
        ----------
        p : ndarray, shape (Nx+2, Ny+2)
            Pressure array including ghost cells.
        """
        p[0, :] = p[1, :]    # left wall
        p[-1, :] = p[-2, :]  # right wall
        p[:, 0] = p[:, 1]    # bottom wall
        p[:, -1] = p[:, -2]  # top wall

    def lid_values(self, Nx: int):
        """Return the tangential lid u-values at face positions.

        Returns
        -------
        float or ndarray
            The velocity values assigned to the top lid.
        """
        if self.smooth_lid:
            return self._get_lid_profile(Nx)
        return self.top

