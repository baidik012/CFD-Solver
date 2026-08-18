"""Boundary conditions for staggered incompressible flow.

Staggered-grid convention:
- u-velocity: stored at vertical (x-)faces, with y-ghost cells
- v-velocity: stored at horizontal (y-)faces, with x-ghost cells
- pressure:  stored at cell centres, Neumann (zero-gradient) BCs

Boundary enforcement uses ghost cells: for a Dirichlet value V_wall,
the ghost-cell entry is V_ghost = 2*V_wall − V_interior.
"""

import numpy as np


class WallType:
    """Base class for all wall boundary-condition types.

    Subclasses override the four ``apply_*`` methods to enforce their
    specific boundary condition.
    """

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

    def is_noslip(self):
        """Return True if this wall is Dirichlet for the implicit Laplacian.

        Used by FFTCrankNicolson to pick DST-II (no-slip) vs DCT-II
        (free-slip) along the relevant axis.
        """
        return False

    def ghost_cell_coeffs(self, component='u'):
        """Return (has_dirichlet, wall_value) for the implicit Laplacian.

        - Dirichlet (no-slip): returns (True, u_wall).
        - Neumann (free-slip / periodic): returns (False, 0.0).
        """
        return (False, 0.0)


class NoSlipWall(WallType):
    """No-slip wall with specified tangential velocity."""

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
    """Free-slip (symmetry) wall: zero normal gradient for tangential velocity."""

    type_name = "free_slip"

    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

    def __repr__(self):
        return f"FreeSlipWall(u={self.u}, v={self.v})"

    def apply_top(self, u, v, Nx, Ny, bc):
        v[1:-1, Ny] = 0.0
        u[:, -1] = u[:, -2]

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
    """Inlet boundary with a specified velocity profile."""

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
        # Tangential velocity is zero at an inlet.
        return (True, 0.0)


class OutletWall(WallType):
    """Outlet boundary with zero-gradient (or convective) outflow."""

    type_name = "outlet"

    def __init__(self, method="zero_gradient"):
        self.method = method

    def __repr__(self):
        return f"OutletWall(method={self.method!r})"

    def apply_top(self, u, v, Nx, Ny, bc):
        v[1:-1, Ny] = v[1:-1, Ny - 1]
        u[:, -1] = u[:, -2]

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
        # Outlet uses zero-gradient (Neumann) BCs, not Dirichlet.
        # The implicit Laplacian must use Neumann (ghost = interior) too
        # to stay consistent; otherwise an unphysical boundary layer forms.
        return False

    def ghost_cell_coeffs(self, component='u'):
        # Neumann: ghost = interior
        return (False, 0.0)


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
# The CLI parser and the YAML schema both consult this map.
WALL_TYPE_REGISTRY = {
    NoSlipWall.type_name:    NoSlipWall,
    FreeSlipWall.type_name:  FreeSlipWall,
    InletWall.type_name:     InletWall,
    OutletWall.type_name:    OutletWall,
    PeriodicWall.type_name:  PeriodicWall,
}


class BoundaryConditions:
    """Velocity boundary conditions on a staggered grid.

    Supports both a legacy scalar-based API and a per-wall-type API.
    Legacy scalar arguments (top, bottom, left, right) are automatically
    wrapped in NoSlipWall objects and stored in the ``walls`` dict.
    """

    def __init__(self, top=1.0, bottom=0.0, left=0.0, right=0.0,
                 smooth_lid: bool = False):
        self.smooth_lid = smooth_lid

        self.walls = {
            'top':    self._to_wall(top,    NoSlipWall, u=1.0),
            'bottom': self._to_wall(bottom, NoSlipWall, u=0.0),
            'left':   self._to_wall(left,   NoSlipWall, v=0.0),
            'right':  self._to_wall(right,  NoSlipWall, v=0.0),
        }

        self._sync_legacy_attrs()

        self._lid_profile = None
        self._lid_profile_key = None

    @staticmethod
    def _to_wall(value, default_type, **defaults):
        """Convert a constructor argument to a WallType object."""
        if isinstance(value, WallType):
            return value
        if value is None:
            return default_type(**defaults)
        if isinstance(value, (int, float)):
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
        """Return the sinusoidal lid profile array (cached by Nx, top speed)."""
        key = (Nx, self.top)
        if self._lid_profile is None or self._lid_profile_key != key:
            self._lid_profile = self.top * np.sin(np.pi * np.arange(Nx + 1) / Nx)
            self._lid_profile_key = key
        return self._lid_profile

    def apply(self, u, v, Nx: int, Ny: int):
        """Set boundary values on velocity components by delegating to each wall."""
        self.walls['top'].apply_top(u, v, Nx, Ny, self)
        self.walls['bottom'].apply_bottom(u, v, Nx, Ny, self)
        self.walls['left'].apply_left(u, v, Nx, Ny, self)
        self.walls['right'].apply_right(u, v, Nx, Ny, self)

    def _inlet_profile(self, wall, N, axis='x'):
        """Compute inlet velocity profile at boundary face positions."""
        if wall.profile == "uniform":
            return np.full(N, wall.U_max)

        # Parabolic: u(y) = 4 * U_max * y * (H - y) / H^2
        # Evaluated at face positions y_j = (j + 0.5) * H / N
        H = self._domain_height if hasattr(self, '_domain_height') else 1.0
        y = (np.arange(N) + 0.5) / N * H
        return 4.0 * wall.U_max * y * (H - y) / (H ** 2)

    def lid_values(self, Nx: int):
        """Return the tangential lid u-values at face positions."""
        if self.smooth_lid:
            return self._get_lid_profile(Nx)
        return self.top

    def has_periodic(self):
        """Return True if any wall is a PeriodicWall."""
        return any(isinstance(w, PeriodicWall) for w in self.walls.values())

    def has_outlet(self):
        """Return True if any wall is an OutletWall."""
        return any(isinstance(w, OutletWall) for w in self.walls.values())