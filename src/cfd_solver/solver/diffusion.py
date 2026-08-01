"""Diffusion schemes for staggered incompressible flow with ghost cells.

This module provides methods to solve the viscous diffusion term (nu * ∇²u)
in the Navier-Stokes equations.

Viscous Term:
-------------
The diffusion term represents the internal friction within the fluid.
In 2D, for velocity components (u, v), it is:
    ∂u/∂t = nu * (∂²u/∂x² + ∂²u/∂y²)
    ∂v/∂t = nu * (∂²v/∂x² + ∂²v/∂y²)

Chorin's Projection Method:
---------------------------
In the context of the fractional step method, diffusion and advection are
combined to calculate an intermediate "star" velocity field (u*, v*) that
does not yet satisfy the incompressibility constraint.

Available schemes:
------------------
1. Explicit: A Forward Euler discretization. It is simple to implement but has
   a strict stability requirement: dt <= (min(dx, dy)² / (4 * nu)).
2. Crank-Nicolson: A semi-implicit scheme that averages the Laplacian at the
   current and next time steps. It is second-order accurate in time and
   unconditionally stable, allowing for larger time steps.
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.fft import dstn, idstn

from .bc import PeriodicWall


def explicit(u, v, adv_u, adv_v, dx, dy, dt, nu, bc, Nx, Ny,
             u_out=None, v_out=None):
    """Forward Euler diffusion + advection predictor with ghost cells.

    Calculates the intermediate velocity (u*, v*) by explicitly stepping
    forward in time using the current advection and diffusion.

    Parameters
    ----------
    u, v : ndarray
        Current velocity components.
    adv_u, adv_v : ndarray
        Advective terms calculated by an advection scheme.
    dx, dy : float
        Grid spacing.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.
    bc : BoundaryConditions
        Object to enforce boundary values.
    Nx, Ny : int
        Number of grid cells.
    u_out, v_out : ndarray, optional
        Pre-allocated output arrays.  If provided the result is written
        here instead of allocating new arrays (zero-copy path).

    Returns
    -------
    u_star, v_star : ndarray
        Intermediate velocity fields with boundary conditions applied.
    """
    dx2, dy2 = dx**2, dy**2

    if u_out is not None:
        u_out[:] = u
        v_out[:] = v
        u_star, v_star = u_out, v_out
    else:
        u_star = u.copy()
        v_star = v.copy()

    # Laplacian of u at active physical interior faces: i=1..Nx-1, j=1..Ny
    lap_u = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2 + \
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2

    # Laplacian of v at active physical interior faces: i=1..Nx, j=1..Ny-1
    lap_v = (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx2 + \
            (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy2

    # Explicit step: u* = u + dt * (-advection + nu * Laplacian)
    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (-adv_u[1:-1, 1:-1] + nu * lap_u)
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (-adv_v[1:-1, 1:-1] + nu * lap_v)

    bc.apply(u_star, v_star, Nx, Ny)
    return u_star, v_star


class CrankNicolson:
    """Semi-implicit Crank-Nicolson diffusion solver.

    The Crank-Nicolson method treats the diffusion term implicitly by solving
    the linear system: (I - 0.5*dt*nu*L) u* = (I + 0.5*dt*nu*L) u - dt * advection.
    This allows for much larger time steps than the explicit method while
    maintaining stability and second-order accuracy.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    bc : BoundaryConditions
        Object to enforce boundary values.

    Attributes
    ----------
    A_u, A_v : csc_matrix
        The implicit operators for u and v velocities.
    """

    def __init__(self, mesh, nu, dt, bc):
        self.nu = nu
        self.dt = dt
        self.bc = bc
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        # Build and pre-factorize the matrices once to speed up the solve() calls
        self.A_u = self._build_u_matrix().tocsc()
        self.A_v = self._build_v_matrix().tocsc()

        self._solve_u = splu(self.A_u).solve
        self._solve_v = splu(self.A_v).solve

    @staticmethod
    def _ghost_cell_coeffs(wall, component='u'):
        """Return (has_dirichlet, wall_value) for the implicit Laplacian.

        .. deprecated:: 
            This static method is preserved for backward compatibility but
            now simply delegates to ``wall.ghost_cell_coeffs(component)``.
            New code should call the wall object's method directly.
        """
        return wall.ghost_cell_coeffs(component)

    def _build_u_matrix(self):
        """Build the (I - 0.5*dt*nu*L) operator for u unknowns.

        Unknowns are at active interior u-faces: i=1..Nx-1, j=1..Ny.
        Boundary conditions are incorporated into the operator where they
        affect the implicit calculation.
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        # i-direction (x): (Nx-1) unknowns. Dirichlet boundaries for u are
        # handled explicitly in the RHS for the x-direction.
        n_i = Nx - 1
        e_i = np.ones(n_i)
        off_i = np.full(n_i - 1, -rx)
        Lx_1d = diags([off_i, np.full(n_i, 2.0 * rx), off_i], [-1, 0, 1], shape=(n_i, n_i), format="csr")

        # j-direction (y): Ny unknowns. Bottom and top walls use ghost cells.
        # Enforcing u_wall via ghost cells modifies the diagonal of the Laplacian.
        n_j = Ny
        diag_j = np.full(n_j, 2.0 * ry)
        off_j = np.full(n_j - 1, -ry)

        # Bottom wall (j=1, matrix index 0)
        bottom_wall = self.bc.walls.get('bottom')
        if bottom_wall is not None and not isinstance(bottom_wall, PeriodicWall):
            has_noslip, val = bottom_wall.ghost_cell_coeffs('u')
            if has_noslip:
                diag_j[0] += ry  # ghost cell adds +ry to diagonal

        # Top wall (j=Ny, matrix index -1)
        top_wall = self.bc.walls.get('top')
        if top_wall is not None and not isinstance(top_wall, PeriodicWall):
            has_noslip, val = top_wall.ghost_cell_coeffs('u')
            if has_noslip:
                diag_j[-1] += ry  # ghost cell adds +ry to diagonal

        Ly_1d = diags([off_j, diag_j, off_j], [-1, 0, 1], shape=(n_j, n_j), format="csr")

        I_ni = eye(n_i, format="csr")
        I_nj = eye(n_j, format="csr")

        A = eye(n_i * n_j, format="csr") + kron(I_nj, Lx_1d) + kron(Ly_1d, I_ni)
        return A

    def _build_v_matrix(self):
        """Build the (I - 0.5*dt*nu*L) operator for v unknowns.

        Unknowns are at active interior v-faces: i=1..Nx, j=1..Ny-1.
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        # i-direction (x): Nx unknowns. Left and right walls use ghost cells.
        n_i = Nx
        diag_i = np.full(n_i, 2.0 * rx)
        off_i = np.full(n_i - 1, -rx)

        # Left wall (i=1, matrix index 0)
        left_wall = self.bc.walls.get('left')
        if left_wall is not None and not isinstance(left_wall, PeriodicWall):
            has_noslip, val = left_wall.ghost_cell_coeffs('v')
            if has_noslip:
                diag_i[0] += rx

        # Right wall (i=Nx, matrix index -1)
        right_wall = self.bc.walls.get('right')
        if right_wall is not None and not isinstance(right_wall, PeriodicWall):
            has_noslip, val = right_wall.ghost_cell_coeffs('v')
            if has_noslip:
                diag_i[-1] += rx

        Lx_1d = diags([off_i, diag_i, off_i], [-1, 0, 1], shape=(n_i, n_i), format="csr")

        # j-direction (y): (Ny-1) unknowns. Bottom and top boundaries are exact.
        n_j = Ny - 1
        off_j = np.full(n_j - 1, -ry)
        Ly_1d = diags([off_j, np.full(n_j, 2.0 * ry), off_j], [-1, 0, 1], shape=(n_j, n_j), format="csr")

        I_ni = eye(n_i, format="csr")
        I_nj = eye(n_j, format="csr")

        A = eye(n_i * n_j, format="csr") + kron(I_nj, Lx_1d) + kron(Ly_1d, I_ni)
        return A

    def update_matrices(self):
        """Rebuild and re-factorize the implicit operators after a BC change.

        Call this whenever boundary conditions change type (e.g. switching
        from NoSlip to FreeSlip) to keep the implicit solve consistent.
        """
        self.A_u = self._build_u_matrix().tocsc()
        self.A_v = self._build_v_matrix().tocsc()
        self._solve_u = splu(self.A_u).solve
        self._solve_v = splu(self.A_v).solve

    def solve(self, u, v, adv_u, adv_v, u_out=None, v_out=None):
        """Advance diffusion + advection to get intermediate velocity (u*, v*).

        Parameters
        ----------
        u, v : ndarray
            Current velocity components.
        adv_u, adv_v : ndarray
            Advective terms.
        u_out, v_out : ndarray, optional
            Pre-allocated output arrays.  If provided the result is written
            here instead of allocating new arrays (zero-copy path).

        Returns
        -------
        u_star, v_star : ndarray
            Intermediate velocity field.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx**2, dy**2
        nu, dt = self.nu, self.dt
        rx = 0.5 * nu * dt / dx2
        ry = 0.5 * nu * dt / dy2

        if u_out is not None:
            u_out[:] = u
            v_out[:] = v
            u_star, v_star = u_out, v_out
        else:
            u_star = u.copy()
            v_star = v.copy()

        # Explicit Laplacian of u (part of the Crank-Nicolson RHS)
        lap_u = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2 + \
                (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2

        # Explicit Laplacian of v (part of the Crank-Nicolson RHS)
        lap_v = (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx2 + \
                (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy2

        # --- Solve for u ---
        # RHS = u + dt*(-advection + 0.5*nu*lap_u)
        rhs_u = u[1:-1, 1:-1] - dt * adv_u[1:-1, 1:-1] + 0.5 * dt * nu * lap_u

        # Add boundary contributions for the implicit part.
        # x-direction Dirichlet: the unknowns adjacent to the side walls
        # couple to u at the wall faces (i=0, i=Nx), which is the wall-NORMAL
        # velocity and is exactly 0 for an impermeable cavity (bc.apply sets
        # those faces to 0). NOTE: bc.left / bc.right are the TANGENTIAL
        # v-velocities on those walls and belong to the v-equation only, so
        # the u-RHS receives no contribution here.
        
        # y-direction (via ghost cells)
        bottom_wall = self.bc.walls.get('bottom')
        if bottom_wall is not None:
            has_noslip, val = bottom_wall.ghost_cell_coeffs('u')
            if has_noslip:
                rhs_u[:, 0] += 2.0 * ry * val

        top_wall = self.bc.walls.get('top')
        if top_wall is not None:
            has_noslip, val = top_wall.ghost_cell_coeffs('u')
            if has_noslip:
                if self.bc.smooth_lid:
                    rhs_u[:, -1] += 2.0 * ry * self.bc._get_lid_profile(Nx)[1:-1]
                else:
                    rhs_u[:, -1] += 2.0 * ry * val

        u_flat = self._solve_u(rhs_u.flatten(order="F"))
        u_star[1:-1, 1:-1] = u_flat.reshape((Nx - 1, Ny), order="F")

        # --- Solve for v ---
        # RHS = v + dt*(-advection + 0.5*nu*lap_v)
        rhs_v = v[1:-1, 1:-1] - dt * adv_v[1:-1, 1:-1] + 0.5 * dt * nu * lap_v

        # x-direction (via ghost cells)
        left_wall = self.bc.walls.get('left')
        if left_wall is not None:
            has_noslip, val = left_wall.ghost_cell_coeffs('v')
            if has_noslip:
                rhs_v[0, :] += 2.0 * rx * val

        right_wall = self.bc.walls.get('right')
        if right_wall is not None:
            has_noslip, val = right_wall.ghost_cell_coeffs('v')
            if has_noslip:
                rhs_v[-1, :] += 2.0 * rx * val

        v_flat = self._solve_v(rhs_v.flatten(order="F"))
        v_star[1:-1, 1:-1] = v_flat.reshape((Nx, Ny - 1), order="F")

        self.bc.apply(u_star, v_star, Nx, Ny)
        return u_star, v_star


class FFTCrankNicolson:
    """Spectral Crank-Nicolson diffusion solver using DST / DCT.

    Solves the same (I - 0.5*dt*nu*L) u* = (I + 0.5*dt*nu*L) u - dt*adv
    system as :class:`CrankNicolson`, but diagonalises the 2-D Laplacian
    analytically in the frequency domain.

    Boundary conventions (staggered Arakawa C-grid with ghost cells):

    * **u-equation** — unknowns at interior u-faces (i=1..Nx-1, j=1..Ny):
      - x: Dirichlet walls (u=0) → DST-I  (Nx-1 points)
      - y: ghost-cell walls      → DST-II (Ny points) for NoSlip,
                                  DCT-II (Ny points) for FreeSlip

    * **v-equation** — unknowns at interior v-faces (i=1..Nx, j=1..Ny-1):
      - x: ghost-cell walls      → DST-II (Nx points) for NoSlip,
                                  DCT-II (Nx points) for FreeSlip
      - y: Dirichlet walls (v=0) → DST-I  (Ny-1 points)

    Each solve is O(Nx*Ny*log(Nx*Ny)) with no factorisation step.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    bc : BoundaryConditions
        Ghost-cell boundary conditions.
    """

    def __init__(self, mesh, nu, dt, bc):
        self.nu = nu
        self.dt = dt
        self.bc = bc
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        self._build_eigenvalues()

    def _is_noslip(self, wall):
        """Check if a wall behaves as NoSlip for the implicit Laplacian.

        Delegates to ``wall.is_noslip()``.  PeriodicWall returns False
        (handled separately by the caller).
        """
        if wall is None:
            return True  # default: treat as NoSlip
        return wall.is_noslip()

    def _build_eigenvalues(self):
        """Compute DST/DCT eigenvalues based on current wall types."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        nu, dt = self.nu, self.dt
        rx = 0.5 * nu * dt / dx2
        ry = 0.5 * nu * dt / dy2

        # u-equation: x is always DST-I (Dirichlet at left/right)
        ni_u = Nx - 1
        nj_u = Ny
        kx_u = np.arange(ni_u)
        eig_x_u = 2.0 * (1.0 - np.cos(np.pi * (kx_u + 1) / (ni_u + 1)))

        # u-equation: y — check wall types
        ky_u = np.arange(nj_u)
        bottom_noslip = self._is_noslip(self.bc.walls.get('bottom'))
        top_noslip = self._is_noslip(self.bc.walls.get('top'))
        if bottom_noslip and top_noslip:
            # Both NoSlip → DST-II
            eig_y_u = 2.0 * (1.0 - np.cos(np.pi * (ky_u + 1) / nj_u))
            self._u_y_type = 2  # DST-II
        elif not bottom_noslip and not top_noslip:
            # Both FreeSlip → DCT-II
            eig_y_u = 2.0 * (1.0 - np.cos(np.pi * ky_u / nj_u))
            self._u_y_type = 3  # DCT-II (type=3 in scipy is DCT-II with ortho)
        else:
            # Mixed: fall back to sparse solver
            raise ValueError(
                "FFTCrankNicolson does not support mixed NoSlip/FreeSlip "
                "on y-boundaries for the u-equation. Use CrankNicolson."
            )

        self._eig_u = 1.0 + rx * eig_x_u[:, np.newaxis] + ry * eig_y_u[np.newaxis, :]

        # v-equation: y is always DST-I (Dirichlet at bottom/top)
        ni_v = Nx
        nj_v = Ny - 1
        ky_v = np.arange(nj_v)
        eig_y_v = 2.0 * (1.0 - np.cos(np.pi * (ky_v + 1) / (nj_v + 1)))

        # v-equation: x — check wall types
        kx_v = np.arange(ni_v)
        left_noslip = self._is_noslip(self.bc.walls.get('left'))
        right_noslip = self._is_noslip(self.bc.walls.get('right'))
        if left_noslip and right_noslip:
            # Both NoSlip → DST-II
            eig_x_v = 2.0 * (1.0 - np.cos(np.pi * (kx_v + 1) / ni_v))
            self._v_x_type = 2  # DST-II
        elif not left_noslip and not right_noslip:
            # Both FreeSlip → DCT-II
            eig_x_v = 2.0 * (1.0 - np.cos(np.pi * kx_v / ni_v))
            self._v_x_type = 3  # DCT-II
        else:
            raise ValueError(
                "FFTCrankNicolson does not support mixed NoSlip/FreeSlip "
                "on x-boundaries for the v-equation. Use CrankNicolson."
            )

        self._eig_v = 1.0 + rx * eig_x_v[:, np.newaxis] + ry * eig_y_v[np.newaxis, :]

    def update_matrices(self):
        """Recompute eigenvalues after a BC change."""
        self._build_eigenvalues()

    def solve(self, u, v, adv_u, adv_v, u_out=None, v_out=None):
        """Advance diffusion + advection to get intermediate velocity (u*, v*).

        Parameters
        ----------
        u, v : ndarray
            Current velocity components.
        adv_u, adv_v : ndarray
            Advective terms.
        u_out, v_out : ndarray, optional
            Pre-allocated output arrays.  If provided the result is written
            here instead of allocating new arrays (zero-copy path).

        Returns
        -------
        u_star, v_star : ndarray
            Intermediate velocity field.
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        nu, dt = self.nu, self.dt
        rx = 0.5 * nu * dt / dx2
        ry = 0.5 * nu * dt / dy2

        if u_out is not None:
            u_out[:] = u
            v_out[:] = v
            u_star, v_star = u_out, v_out
        else:
            u_star = u.copy()
            v_star = v.copy()

        # --- u-equation ---
        lap_u = ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2
                 + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2)
        rhs_u = u[1:-1, 1:-1] - dt * adv_u[1:-1, 1:-1] + 0.5 * dt * nu * lap_u

        # y-direction RHS boundary contributions
        bottom_wall = self.bc.walls.get('bottom')
        if bottom_wall is not None:
            has_noslip, val = CrankNicolson._ghost_cell_coeffs(bottom_wall, 'u')
            if has_noslip:
                rhs_u[:, 0] += 2.0 * ry * val

        top_wall = self.bc.walls.get('top')
        if top_wall is not None:
            has_noslip, val = CrankNicolson._ghost_cell_coeffs(top_wall, 'u')
            if has_noslip:
                if self.bc.smooth_lid:
                    rhs_u[:, -1] += 2.0 * ry * self.bc._get_lid_profile(Nx)[1:-1]
                else:
                    rhs_u[:, -1] += 2.0 * ry * val

        # Solve in spectral domain: DST-I(axis=0, x) then DST/DCT(axis=1, y)
        spec_u = dstn(rhs_u, type=1, norm='ortho', axes=(0,))
        spec_u = dstn(spec_u, type=self._u_y_type, norm='ortho', axes=(1,))
        spec_u /= self._eig_u
        sol_u = idstn(spec_u, type=self._u_y_type, norm='ortho', axes=(1,))
        sol_u = idstn(sol_u, type=1, norm='ortho', axes=(0,))
        u_star[1:-1, 1:-1] = sol_u

        # --- v-equation ---
        lap_v = ((v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx2
                 + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy2)
        rhs_v = v[1:-1, 1:-1] - dt * adv_v[1:-1, 1:-1] + 0.5 * dt * nu * lap_v

        # x-direction RHS boundary contributions
        left_wall = self.bc.walls.get('left')
        if left_wall is not None:
            has_noslip, val = CrankNicolson._ghost_cell_coeffs(left_wall, 'v')
            if has_noslip:
                rhs_v[0, :] += 2.0 * rx * val

        right_wall = self.bc.walls.get('right')
        if right_wall is not None:
            has_noslip, val = CrankNicolson._ghost_cell_coeffs(right_wall, 'v')
            if has_noslip:
                rhs_v[-1, :] += 2.0 * rx * val

        # Solve in spectral domain: DST/DCT(axis=0, x) then DST-I(axis=1, y)
        spec_v = dstn(rhs_v, type=self._v_x_type, norm='ortho', axes=(0,))
        spec_v = dstn(spec_v, type=1, norm='ortho', axes=(1,))
        spec_v /= self._eig_v
        sol_v = idstn(spec_v, type=1, norm='ortho', axes=(1,))
        sol_v = idstn(sol_v, type=self._v_x_type, norm='ortho', axes=(0,))
        v_star[1:-1, 1:-1] = sol_v

        self.bc.apply(u_star, v_star, Nx, Ny)
        return u_star, v_star


def create_diffusion_solver(mesh, nu, dt, bc, threshold=128):
    """Create the optimal diffusion solver for the given grid size.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    bc : BoundaryConditions
        Boundary conditions.
    threshold : int
        Grid size threshold.  FFT is used when max(Nx, Ny) >= threshold.

    Returns
    -------
    solver
        Either :class:`CrankNicolson` or :class:`FFTCrankNicolson`.
    """
    if max(mesh.Nx, mesh.Ny) >= threshold:
        return FFTCrankNicolson(mesh, nu, dt, bc)
    return CrankNicolson(mesh, nu, dt, bc)
