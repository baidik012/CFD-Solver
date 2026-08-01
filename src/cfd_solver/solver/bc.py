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

Design (Audit refactoring P0-1):
---------------------------------
Each :class:`WallType` subclass implements four ``apply_*`` methods
(``apply_top``, ``apply_bottom``, ``apply_left``, ``apply_right``) that
mutate the (u, v) arrays in place.  :meth:`BoundaryConditions.apply`
simply iterates the four walls and delegates.  Adding a new wall type
no longer requires editing ``apply`` — the new subclass just implements
the four methods.

The diffusion-solver dispatch (``is_noslip`` / ``ghost_cell_coeffs``)
is also moved onto WallType, eliminating the isinstance chains that
previously lived in ``diffusion.py``.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Wall type classes for flexible boundary condition specification.
#
# Each wall type implements:
#   - apply_top(u, v, Nx, Ny, bc)     — set BCs at the top wall (j = Ny)
#   - apply_bottom(u, v, Nx, Ny, bc)  — set BCs at the bottom wall (j = 0)
#   - apply_left(u, v, Nx, Ny, bc)    — set BCs at the left wall (i = 0)
#   - apply_right(u, v, Nx, Ny, bc)   — set BCs at the right wall (i = Nx)
#   - is_noslip() -> bool             — used by FFTCrankNicolson to pick DST/DCT
#   - ghost_cell_coeffs(component)    — used by CrankNicolson to build its matrix
#
# Adding a new wall type means subclassing WallType and implementing these
# six methods.  No edits to BoundaryConditions.apply, diffusion.py, or
# pressure.py are required (audit finding P0-1).
# ---------------------------------------------------------------------------

class WallType:
    """Base class for all wall boundary condition types.

    Subclasses override the four ``apply_*`` methods to enforce their
    specific boundary condition.  The default implementations are
    no-ops, so a subclass only needs to override the walls it actually
    affects (though in practice all four are usually overridden).

    The ``bc`` parameter passed to each ``apply_*`` method is the
    owning :class:`BoundaryConditions` instance, giving wall types
    access to shared helpers like ``_inlet_profile`` and
    ``_get_lid_profile``.
    """

    # String identifier used by the CLI / YAML schema.  Subclasses set this
    # so that the parser can look up the class via a single registry
    # (see ``WALL_TYPE_REGISTRY`` below).
    type_name = "wall"

    def apply_top(self, u, v, Nx, Ny, bc):
        """Enforce BC at the top wall (j = Ny). Default: no-op."""
        pass

    def apply_bottom(self, u, v, Nx, Ny, bc):
        """Enforce BC at the bottom wall (j = 0). Default: no-op."""
        pass

    def apply_left(self, u, v, Nx, Ny, bc):
        """Enforce BC at the left wall (i = 0). Default: no-op."""
        pass

    def apply_right(self, u, v, Nx, Ny, bc):
        """Enforce BC at the right wall (i = Nx). Default: no-op."""
        pass

    # ── Diffusion-solver hooks ──────────────────────────────────────────

    def is_noslip(self):
        """Return True if this wall behaves as a no-slip wall for the
        implicit Laplacian (i.e. tangential velocity is fixed, requiring
        the ghost-cell Dirichlet treatment).

        Used by :class:`~cfd_solver.solver.diffusion.FFTCrankNicolson`
        to decide between DST-II (no-slip) and DCT-II (free-slip) along
        the relevant axis.
        """
        return False

    def ghost_cell_coeffs(self, component='u'):
        """Return ``(has_dirichlet, wall_value)`` for the implicit Laplacian.

        - For Dirichlet (no-slip) walls: ghost = 2*u_wall - interior,
          so returns ``(True, u_wall)``.
        - For Neumann (free-slip) walls: ghost = interior,
          so returns ``(False, 0.0)``.

        ``component`` is ``'u'`` for top/bottom walls (tangential u) or
        ``'v'`` for left/right walls (tangential v).
        """
        return (False, 0.0)


class NoSlipWall(WallType):
    """No-slip wall with specified tangential velocity.

    Parameters
    ----------
    u : float, optional
        Tangential u-velocity at the wall (for top/bottom walls). Default 0.0.
    v : float, optional
        Tangential v-velocity at the wall (for left/right walls). Default 0.0.
    """

    type_name = "wall"

    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

    def __repr__(self):
        return f"NoSlipWall(u={self.u}, v={self.v})"

    def apply_top(self, u, v, Nx, Ny, bc):
        v[1:-1, Ny] = 0.0
        if bc.smooth_lid:
            u[:, -1] = 2.0 * bc._get_lid_profile(Nx) - u[:, -2]
        else:
            u[:, -1] = 2.0 * self.u - u[:, -2]

    def apply_bottom(self, u, v, Nx, Ny, bc):
        v[1:-1, 0] = 0.0
        u[:, 0] = 2.0 * self.u - u[:, 1]

    def apply_left(self, u, v, Nx, Ny, bc):
        u[0, 1:-1] = 0.0
        v[0, :] = 2.0 * self.v - v[1, :]

    def apply_right(self, u, v, Nx, Ny, bc):
        u[Nx, 1:-1] = 0.0
        v[-1, :] = 2.0 * self.v - v[-2, :]

    def is_noslip(self):
        return True

    def ghost_cell_coeffs(self, component='u'):
        val = self.u if component == 'u' else self.v
        return (True, val)


class FreeSlipWall(WallType):
    """Free-slip (symmetry) wall: zero normal gradient for tangential velocity.

    Parameters
    ----------
    u : float, optional
        Tangential u-velocity at the wall. Default 0.0.
    v : float, optional
        Tangential v-velocity at the wall. Default 0.0.
    """

    type_name = "free_slip"

    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

    def __repr__(self):
        return f"FreeSlipWall(u={self.u}, v={self.v})"

    def apply_top(self, u, v, Nx, Ny, bc):
        v[1:-1, Ny] = 0.0
        u[:, -1] = u[:, -2]   # free-slip: ghost = interior

    def apply_bottom(self, u, v, Nx, Ny, bc):
        v[1:-1, 0] = 0.0
        u[:, 0] = u[:, 1]

    def apply_left(self, u, v, Nx, Ny, bc):
        u[0, 1:-1] = 0.0
        v[0, :] = v[1, :]

    def apply_right(self, u, v, Nx, Ny, bc):
        u[Nx, 1:-1] = 0.0
        v[-1, :] = v[-2, :]

    def is_noslip(self):
        return False

    def ghost_cell_coeffs(self, component='u'):
        return (False, 0.0)


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

    type_name = "inlet"

    def __init__(self, profile="uniform", U_max=1.0):
        self.profile = profile
        self.U_max = U_max

    def __repr__(self):
        return f"InletWall(profile={self.profile!r}, U_max={self.U_max})"

    def apply_top(self, u, v, Nx, Ny, bc):
        v[1:-1, Ny] = bc._inlet_profile(self, Ny, axis='y')
        u[:, -1] = 2.0 * 0.0 - u[:, -2]

    def apply_bottom(self, u, v, Nx, Ny, bc):
        v[1:-1, 0] = bc._inlet_profile(self, Ny, axis='y')
        u[:, 0] = 2.0 * 0.0 - u[:, 1]

    def apply_left(self, u, v, Nx, Ny, bc):
        u[0, 1:-1] = bc._inlet_profile(self, Ny, axis='x')
        v[0, :] = 2.0 * 0.0 - v[1, :]

    def apply_right(self, u, v, Nx, Ny, bc):
        u[Nx, 1:-1] = bc._inlet_profile(self, Ny, axis='x')
        v[-1, :] = 2.0 * 0.0 - v[-2, :]

    def is_noslip(self):
        # Tangential component is zero (Dirichlet) → treated as NoSlip for DST.
        return True

    def ghost_cell_coeffs(self, component='u'):
        # Same as NoSlipWall with u_wall = 0 (tangential velocity is zero).
        return (True, 0.0)


class OutletWall(WallType):
    """Outlet boundary with outflow treatment.

    Parameters
    ----------
    method : str, optional
        Outflow method: ``"zero_gradient"`` (extrapolate from interior) or
        ``"convective"`` (convective outflow). Default ``"zero_gradient"``.
    """

    type_name = "outlet"

    def __init__(self, method="zero_gradient"):
        self.method = method

    def __repr__(self):
        return f"OutletWall(method={self.method!r})"

    def apply_top(self, u, v, Nx, Ny, bc):
        v[1:-1, Ny] = v[1:-1, Ny - 1]   # zero-gradient
        u[:, -1] = u[:, -2]              # zero-gradient

    def apply_bottom(self, u, v, Nx, Ny, bc):
        v[1:-1, 0] = v[1:-1, 1]
        u[:, 0] = u[:, 1]

    def apply_left(self, u, v, Nx, Ny, bc):
        u[0, 1:-1] = u[1, 1:-1]
        v[0, :] = v[1, :]

    def apply_right(self, u, v, Nx, Ny, bc):
        u[Nx, 1:-1] = u[Nx - 1, 1:-1]
        v[-1, :] = v[-2, :]

    def is_noslip(self):
        # Tangential component is zero (Dirichlet) → treated as NoSlip for DST.
        return True

    def ghost_cell_coeffs(self, component='u'):
        return (True, 0.0)


class PeriodicWall(WallType):
    """Periodic boundary: wraps to the opposite wall."""

    type_name = "periodic"

    def __repr__(self):
        return "PeriodicWall()"

    def apply_top(self, u, v, Nx, Ny, bc):
        # y-periodic: top ghost = bottom interior
        u[:, -1] = u[:, 1]

    def apply_bottom(self, u, v, Nx, Ny, bc):
        # y-periodic: bottom ghost = top interior
        u[:, 0] = u[:, Ny]
        v[1:-1, 0] = v[1:-1, Ny]

    def apply_left(self, u, v, Nx, Ny, bc):
        # x-periodic: left ghost = right interior
        v[0, :] = v[Nx, :]
        u[0, 1:-1] = u[Nx, 1:-1]

    def apply_right(self, u, v, Nx, Ny, bc):
        # x-periodic: right ghost = left interior
        v[-1, :] = v[1, :]
        u[Nx, 1:-1] = u[0, 1:-1]

    def is_noslip(self):
        return False

    def ghost_cell_coeffs(self, component='u'):
        return (False, 0.0)


# Single source of truth for wall-type-name → class lookup.
# The CLI parser and the YAML schema both consult this map rather than
# maintaining their own copies.  (Audit finding P0-1 / P0-3.)
WALL_TYPE_REGISTRY = {
    NoSlipWall.type_name:    NoSlipWall,
    FreeSlipWall.type_name:  FreeSlipWall,
    InletWall.type_name:     InletWall,
    OutletWall.type_name:    OutletWall,
    PeriodicWall.type_name:  PeriodicWall,
}


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
            'top':    self._to_wall(top,    NoSlipWall, u=1.0),
            'bottom': self._to_wall(bottom, NoSlipWall, u=0.0),
            'left':   self._to_wall(left,   NoSlipWall, v=0.0),
            'right':  self._to_wall(right,  NoSlipWall, v=0.0),
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

        Delegates to each wall's ``apply_*`` method.  Adding a new wall
        type no longer requires modifying this method — the new WallType
        subclass just implements the four ``apply_*`` methods.

        Parameters
        ----------
        u : ndarray, shape (Nx+1, Ny+2)
            u-velocity array including ghost cells.
        v : ndarray, shape (Nx+2, Ny+1)
            v-velocity array including ghost cells.
        Nx, Ny : int
            Number of cells in each direction.
        """
        self.walls['top'].apply_top(u, v, Nx, Ny, self)
        self.walls['bottom'].apply_bottom(u, v, Nx, Ny, self)
        self.walls['left'].apply_left(u, v, Nx, Ny, self)
        self.walls['right'].apply_right(u, v, Nx, Ny, self)

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

    def has_periodic(self):
        """Return True if any wall is a :class:`PeriodicWall`."""
        return any(isinstance(w, PeriodicWall) for w in self.walls.values())

    def has_outlet(self):
        """Return True if any wall is an :class:`OutletWall`."""
        return any(isinstance(w, OutletWall) for w in self.walls.values())
