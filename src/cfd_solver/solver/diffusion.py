"""Diffusion schemes for staggered incompressible flow with ghost cells.

Available schemes:
  - explicit: Forward Euler (stable for dt <= dx^2 / (4*nu))
  - crank_nicolson: Semi-implicit Crank-Nicolson (unconditionally stable)
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import cg, splu


def explicit(u, v, adv_u, adv_v, dx, dy, dt, nu, bc, Nx, Ny):
    """Forward Euler diffusion + advection predictor with ghost cells.

    Returns u_star, v_star with BCs applied.
    """
    dx2, dy2 = dx**2, dy**2

    u_star = u.copy()
    v_star = v.copy()

    # Laplacian of u at active physical interior faces: i=1..Nx-1, j=1..Ny
    lap_u = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2 + \
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2

    # Laplacian of v at active physical interior faces: i=1..Nx, j=1..Ny-1
    lap_v = (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx2 + \
            (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy2

    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (-adv_u[1:-1, 1:-1] + nu * lap_u)
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (-adv_v[1:-1, 1:-1] + nu * lap_v)

    bc.apply(u_star, v_star, Nx, Ny)
    return u_star, v_star


class CrankNicolson:
    """Semi-implicit Crank-Nicolson diffusion solver.

    Builds the implicit matrices once at construction time. Each call to
    ``solve()`` assembles the RHS (including the explicit Laplacian of the
    current velocity) and performs a CG solve.

    Parameters
    ----------
    mesh : Mesh
    nu, dt : float
    bc : BoundaryConditions
    cg_maxiter : int
    cg_rtol : float
    """

    def __init__(self, mesh, nu, dt, bc, cg_maxiter=1000, cg_rtol=1e-5):
        self.nu = nu
        self.dt = dt
        self.bc = bc
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.cg_maxiter = cg_maxiter
        self.cg_rtol = cg_rtol

        self.A_u = self._build_u_matrix().tocsc()
        self.A_v = self._build_v_matrix().tocsc()

        # Pre-factorize for speed (matrix is constant)
        self._solve_u = splu(self.A_u).solve
        self._solve_v = splu(self.A_v).solve

    def _build_u_matrix(self):
        """Build (I - 0.5*dt*nu*L) for active u unknowns: i=1..Nx-1, j=1..Ny."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        # i-direction: (Nx-1) unknowns, no boundary modifications needed for Dirichlet walls
        n_i = Nx - 1
        e_i = np.ones(n_i)
        off_i = np.full(n_i - 1, -rx)
        Lx_1d = diags([off_i, np.full(n_i, 2.0 * rx), off_i], [-1, 0, 1], shape=(n_i, n_i), format="csr")

        # j-direction: Ny unknowns. Bottom and top walls use ghost cells to enforce BCs.
        n_j = Ny
        diag_j = np.full(n_j, 2.0 * ry)
        diag_j[0] += ry   # bottom ghost cell boundary: diag becomes 3*ry
        diag_j[-1] += ry  # top ghost cell boundary: diag becomes 3*ry
        off_j = np.full(n_j - 1, -ry)
        Ly_1d = diags([off_j, diag_j, off_j], [-1, 0, 1], shape=(n_j, n_j), format="csr")

        I_ni = eye(n_i, format="csr")
        I_nj = eye(n_j, format="csr")

        A = eye(n_i * n_j, format="csr") + kron(I_nj, Lx_1d) + kron(Ly_1d, I_ni)
        return A

    def _build_v_matrix(self):
        """Build (I - 0.5*dt*nu*L) for active v unknowns: i=1..Nx, j=1..Ny-1."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        # i-direction: Nx unknowns. Left and right walls use ghost cells for no-slip.
        n_i = Nx
        diag_i = np.full(n_i, 2.0 * rx)
        diag_i[0] += rx   # left no-slip wall: diag becomes 3*rx
        diag_i[-1] += rx  # right no-slip wall: diag becomes 3*rx
        off_i = np.full(n_i - 1, -rx)
        Lx_1d = diags([off_i, diag_i, off_i], [-1, 0, 1], shape=(n_i, n_i), format="csr")

        # j-direction: (Ny-1) unknowns. Bottom and top are exactly on boundaries (v=0).
        n_j = Ny - 1
        off_j = np.full(n_j - 1, -ry)
        Ly_1d = diags([off_j, np.full(n_j, 2.0 * ry), off_j], [-1, 0, 1], shape=(n_j, n_j), format="csr")

        I_ni = eye(n_i, format="csr")
        I_nj = eye(n_j, format="csr")

        A = eye(n_i * n_j, format="csr") + kron(I_nj, Lx_1d) + kron(Ly_1d, I_ni)
        return A

    def solve(self, u, v, adv_u, adv_v):
        """Advance diffusion + advection to get u_star, v_star.

        Returns u_star, v_star with BCs applied.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx**2, dy**2
        nu, dt = self.nu, self.dt
        rx = 0.5 * nu * dt / dx2
        ry = 0.5 * nu * dt / dy2

        u_star = u.copy()
        v_star = v.copy()

        # Explicit Laplacian of u
        lap_u = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2 + \
                (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2

        # Explicit Laplacian of v
        lap_v = (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx2 + \
                (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy2

        # --- Solve for u ---
        rhs_u = u[1:-1, 1:-1] - dt * adv_u[1:-1, 1:-1] + 0.5 * dt * nu * lap_u

        # Dirichlet boundary contributions for implicit part of u
        # Left/Right walls
        rhs_u[0, :] += rx * self.bc.left
        rhs_u[-1, :] += rx * self.bc.right
        
        # Bottom wall (via ghost cell at j=0)
        rhs_u[:, 0] += 2.0 * ry * self.bc.bottom
        
        # Top wall (via ghost cell at j=Ny+1)
        if self.bc.smooth_lid:
            rhs_u[:, -1] += 2.0 * ry * self.bc._get_lid_profile(Nx)[1:-1]
        else:
            rhs_u[:, -1] += 2.0 * ry * self.bc.top

        u_flat = self._solve_u(rhs_u.flatten())
        u_star[1:-1, 1:-1] = u_flat.reshape((Nx - 1, Ny))

        # --- Solve for v ---
        rhs_v = v[1:-1, 1:-1] - dt * adv_v[1:-1, 1:-1] + 0.5 * dt * nu * lap_v

        # Dirichlet boundary contributions for implicit part of v
        # Left wall (via ghost cell at i=0)
        rhs_v[0, :] += 2.0 * rx * self.bc.left
        # Right wall (via ghost cell at i=Nx+1)
        rhs_v[-1, :] += 2.0 * rx * self.bc.right

        # Note: bottom and top walls for v are exactly 0, so no contributions to add

        v_flat = self._solve_v(rhs_v.flatten())
        v_star[1:-1, 1:-1] = v_flat.reshape((Nx, Ny - 1))

        self.bc.apply(u_star, v_star, Nx, Ny)
        return u_star, v_star
