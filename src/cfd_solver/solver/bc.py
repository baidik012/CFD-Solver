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


# ---------------------------------------------------------------------------
# Wall type classes for flexible boundary condition specification.
# ---------------------------------------------------------------------------

class WallType:
    """Base class for all wall boundary condition types."""
    pass


class NoSlipWall(WallType):
    """No-slip wall with specified tangential velocity.

    Parameters
    ----------
    u : float, optional
        Tangential u-velocity at the wall (for top/bottom walls). Default 0.0.
    v : float, optional
        Tangential v-velocity at the wall (for left/right walls). Default 0.0.
    """

    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

    def __repr__(self):
        return f"NoSlipWall(u={self.u}, v={self.v})"


class FreeSlipWall(WallType):
    """Free-slip (symmetry) wall: zero normal gradient for tangential velocity.

    Parameters
    ----------
    u : float, optional
        Tangential u-velocity at the wall. Default 0.0.
    v : float, optional
        Tangential v-velocity at the wall. Default 0.0.
    """

    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

    def __repr__(self):
        return f"FreeSlipWall(u={self.u}, v={self.v})"


class InletWall(WallType):
    """Inlet boundary with a specified velocity profile.

    Parameters
    ----------
    profile : str, optional
        Velocity profile type: ``"uniform"`` (flat) or ``"parabolic"``
        (fully-developed channel profile). Default ``"uniform"``.
    U_max : float, optional
        Maximum velocity at the inlet. Default 1.0.
    """

    def __init__(self, profile="uniform", U_max=1.0):
        self.profile = profile
        self.U_max = U_max

    def __repr__(self):
        return f"InletWall(profile={self.profile!r}, U_max={self.U_max})"


class OutletWall(WallType):
    """Outlet boundary with outflow treatment.

    Parameters
    ----------
    method : str, optional
        Outflow method: ``"zero_gradient"`` (extrapolate from interior) or
        ``"convective"`` (convective outflow). Default ``"zero_gradient"``.
    """

    def __init__(self, method="zero_gradient"):
        self.method = method

    def __repr__(self):
        return f"OutletWall(method={self.method!r})"


class PeriodicWall(WallType):
    """Periodic boundary: wraps to the opposite wall."""

    def __repr__(self):
        return "PeriodicWall()"


# ---------------------------------------------------------------------------
# Main boundary-conditions class
# ---------------------------------------------------------------------------

class BoundaryConditions:
    """Velocity boundary conditions on a staggered grid.

    Supports both a legacy scalar-based API and a newer per-wall-type API.
    Legacy parameters (top, bottom, left, right as floats) are automatically
    wrapped in :class:`NoSlipWall` objects and stored in the ``walls`` dict.

    Parameters
    ----------
    top : float or WallType, optional
        Tangential u-velocity on the top wall (default 1.0), or a WallType
        object for full control.
    bottom : float or WallType, optional
        Tangential u-velocity on the bottom wall (default 0.0).
    left : float or WallType, optional
        Tangential v-velocity on the left wall (default 0.0).
    right : float or WallType, optional
        Tangential v-velocity on the right wall (default 0.0).
    smooth_lid : bool, optional
        If True, apply a sinusoidal lid profile u(x) = U * sin(pi*x/L)
        instead of a uniform step function (default False).

    Attributes
    ----------
    walls : dict
        Per-wall configuration keyed by ``'top'``, ``'bottom'``, ``'left'``,
        ``'right'``.  Values are WallType instances.
    """

    def __init__(self, top=1.0, bottom=0.0, left=0.0, right=0.0,
                 smooth_lid: bool = False):
        # Legacy scalar attributes — preserved for backward compatibility
        # with diffusion.py and checkpoint serialization.
        self.smooth_lid = smooth_lid

        # Build per-wall dict, converting scalars to NoSlipWall
        self.walls = {
            'top': self._to_wall(top, NoSlipWall, u=1.0),
            'bottom': self._to_wall(bottom, NoSlipWall, u=0.0),
            'left': self._to_wall(left, NoSlipWall, v=0.0),
            'right': self._to_wall(right, NoSlipWall, v=0.0),
        }

        # Sync legacy float attributes from the wall objects
        self._sync_legacy_attrs()

        self._lid_profile = None
        self._lid_profile_key = None

    @staticmethod
    def _to_wall(value, default_type, **defaults):
        """Convert a constructor argument to a WallType object.

        Accepts a WallType instance directly, a scalar (wrapped in
        ``default_type``), or ``None`` (uses the default).

        For backward compatibility, a scalar ``value`` is interpreted as
        the primary tangential velocity of the wall (``u`` for top/bottom,
        ``v`` for left/right).  The secondary component keeps its default.
        """
        if isinstance(value, WallType):
            return value
        if value is None:
            return default_type(**defaults)
        if isinstance(value, (int, float)):
            # Legacy scalar — determine which component it maps to
            # by checking which component the default_type uses
            params = dict(defaults)
            if 'u' in defaults:
                params['u'] = value
            elif 'v' in defaults:
                params['v'] = value
            return default_type(**params)
        raise TypeError(
            f"Expected WallType, scalar, or None, got {type(value).__name__}"
        )

    def _sync_legacy_attrs(self):
        """Write float values from wall objects back to self.top / bottom / etc."""
        top_w = self.walls['top']
        self.top = top_w.u if isinstance(top_w, (NoSlipWall, FreeSlipWall)) else 0.0
        bottom_w = self.walls['bottom']
        self.bottom = bottom_w.u if isinstance(bottom_w, (NoSlipWall, FreeSlipWall)) else 0.0
        left_w = self.walls['left']
        self.left = left_w.v if isinstance(left_w, (NoSlipWall, FreeSlipWall)) else 0.0
        right_w = self.walls['right']
        self.right = right_w.v if isinstance(right_w, (NoSlipWall, FreeSlipWall)) else 0.0

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

        Dispatches to per-wall-type handlers.  For backward compatibility
        all walls default to :class:`NoSlipWall` and produce identical
        results to the original implementation.

        Parameters
        ----------
        u : ndarray, shape (Nx+1, Ny+2)
            u-velocity array including ghost cells.
        v : ndarray, shape (Nx+2, Ny+1)
            v-velocity array including ghost cells.
        Nx, Ny : int
            Number of cells in each direction.
        """
        # --- Top wall (j = Ny) ---
        wall = self.walls['top']
        if isinstance(wall, InletWall):
            # Normal velocity (v) at top face
            v[1:-1, Ny] = self._inlet_profile(wall, Ny, axis='y')
            # Tangential u: ghost-cell Dirichlet
            u[:, -1] = 2.0 * 0.0 - u[:, -2]
        elif isinstance(wall, OutletWall):
            v[1:-1, Ny] = v[1:-1, Ny - 1]  # zero-gradient
            u[:, -1] = u[:, -2]              # zero-gradient
        elif isinstance(wall, FreeSlipWall):
            v[1:-1, Ny] = 0.0
            u[:, -1] = u[:, -2]              # free-slip: ghost = interior
        else:
            # NoSlipWall (default)
            v[1:-1, Ny] = 0.0
            if self.smooth_lid:
                u[:, -1] = 2.0 * self._get_lid_profile(Nx) - u[:, -2]
            else:
                u[:, -1] = 2.0 * self.top - u[:, -2]

        # --- Bottom wall (j = 0) ---
        wall = self.walls['bottom']
        if isinstance(wall, InletWall):
            v[1:-1, 0] = self._inlet_profile(wall, Ny, axis='y')
            u[:, 0] = 2.0 * 0.0 - u[:, 1]
        elif isinstance(wall, OutletWall):
            v[1:-1, 0] = v[1:-1, 1]
            u[:, 0] = u[:, 1]
        elif isinstance(wall, FreeSlipWall):
            v[1:-1, 0] = 0.0
            u[:, 0] = u[:, 1]
        else:
            v[1:-1, 0] = 0.0
            u[:, 0] = 2.0 * self.bottom - u[:, 1]

        # --- Left wall (i = 0) ---
        wall = self.walls['left']
        if isinstance(wall, InletWall):
            u[0, 1:-1] = self._inlet_profile(wall, Ny, axis='x')
            v[0, :] = 2.0 * 0.0 - v[1, :]
        elif isinstance(wall, OutletWall):
            u[0, 1:-1] = u[1, 1:-1]
            v[0, :] = v[1, :]
        elif isinstance(wall, FreeSlipWall):
            u[0, 1:-1] = 0.0
            v[0, :] = v[1, :]
        else:
            u[0, 1:-1] = 0.0
            v[0, :] = 2.0 * self.left - v[1, :]

        # --- Right wall (i = Nx) ---
        wall = self.walls['right']
        if isinstance(wall, InletWall):
            u[Nx, 1:-1] = self._inlet_profile(wall, Ny, axis='x')
            v[-1, :] = 2.0 * 0.0 - v[-2, :]
        elif isinstance(wall, OutletWall):
            u[Nx, 1:-1] = u[Nx - 1, 1:-1]
            v[-1, :] = v[-2, :]
        elif isinstance(wall, FreeSlipWall):
            u[Nx, 1:-1] = 0.0
            v[-1, :] = v[-2, :]
        else:
            u[Nx, 1:-1] = 0.0
            v[-1, :] = 2.0 * self.right - v[-2, :]

    def _inlet_profile(self, wall, N, axis='x'):
        """Compute inlet velocity profile at boundary face positions.

        Parameters
        ----------
        wall : InletWall
            The inlet wall configuration.
        N : int
            Number of cells in the wall-normal direction (Ny for top/bottom,
            Ny for left/right since u-faces span Ny interior points).
        axis : str
            ``'x'`` for left/right walls (sets u at i=0 or i=Nx),
            ``'y'`` for top/bottom walls (sets v at j=0 or j=Ny).

        Returns
        -------
        ndarray
            Profile values of length N (one per interior face).
        """
        if wall.profile == "uniform":
            return np.full(N, wall.U_max)

        # Parabolic profile: u(y) = 4 * U_max * y * (H - y) / H^2
        # Evaluated at face positions.  For a channel of height H, the
        # interior faces are at y_j = (j + 0.5) * dy, j = 0 .. N-1.
        H = 1.0  # normalised; caller rescales via U_max
        y = (np.arange(N) + 0.5) / N  # y/H in [0, 1]
        return 4.0 * wall.U_max * y * (1.0 - y)

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

