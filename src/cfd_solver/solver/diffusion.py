"""Diffusion schemes for staggered incompressible flow.

Available schemes:
  - explicit: Forward Euler (stable for dt <= dx^2 / (4*nu))
  - crank_nicolson: Semi-implicit Crank-Nicolson (unconditionally stable)
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import cg


def explicit(u, v, adv_u, adv_v, dx, dy, dt, nu, bc, Nx, Ny):
    """Forward Euler diffusion + advection predictor.

    Returns u_star, v_star with BCs applied.
    """
    dx2, dy2 = dx**2, dy**2

    # Laplacian of u
    lap_u = np.zeros_like(u)
    lap_u[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx2
    if u.shape[1] > 2:
        lap_u[:, 1:-1] += (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dy2

    # Laplacian of v
    lap_v = np.zeros_like(v)
    lap_v[1:-1, :] = (v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]) / dx2
    lap_v[0, :] += (v[1, :] - 2 * v[0, :] + (-v[0, :])) / dx2
    lap_v[-1, :] += (v[-2, :] - 2 * v[-1, :] + (-v[-1, :])) / dx2
    if v.shape[1] > 2:
        lap_v[:, 1:-1] += (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / dy2

    u_star = u.copy()
    v_star = v.copy()

    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
        -adv_u[1:-1, 1:-1] + nu * lap_u[1:-1, 1:-1]
    )
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
        -adv_v[1:-1, 1:-1] + nu * lap_v[1:-1, 1:-1]
    )

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

        self.A_u = self._build_u_matrix()
        self.A_v = self._build_v_matrix()

    def _build_u_matrix(self):
        """Build (I - 0.5*dt*nu*L) for u unknowns: interior faces i=1..Nx-1, j=1..Ny-2.

        Uses Kronecker products — no Python loops.
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        # 1D operators for the (Nx-1) × (Ny-2) unknown grid
        # i-direction: (Nx-1) unknowns, no BC modifications needed
        n_i = Nx - 1
        e_i = np.ones(n_i)
        Ix = diags([e_i], [0], shape=(n_i, n_i), format="csr")

        # j-direction: (Ny-2) unknowns, Dirichlet BCs already incorporated
        n_j = Ny - 2
        e_j = np.ones(n_j)
        Iy = diags([e_j], [0], shape=(n_j, n_j), format="csr")

        # Identity matrices for Kronecker
        I_ni = eye(n_i, format="csr")
        I_nj = eye(n_j, format="csr")

        # Stencil: 1 + 2*rx + 2*ry on diagonal, -rx on i-neighbors, -ry on j-neighbors
        # Since all unknowns are interior (no boundary reductions), it's a simple 5-point stencil
        Lx_1d = diags([np.full(n_i - 1, -rx), np.full(n_i, 2 * rx),
                        np.full(n_i - 1, -rx)],
                       [-1, 0, 1], shape=(n_i, n_i), format="csr")
        Ly_1d = diags([np.full(n_j - 1, -ry), np.full(n_j, 2 * ry),
                        np.full(n_j - 1, -ry)],
                       [-1, 0, 1], shape=(n_j, n_j), format="csr")

        A = eye(n_i * n_j, format="csr") + kron(I_nj, Lx_1d) + kron(Ly_1d, I_ni)
        return A

    def _build_v_matrix(self):
        """Build (I - 0.5*dt*nu*L) for v unknowns: all i, interior j=1..Ny-1.

        Uses Kronecker products — no Python loops.
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        # i-direction: Nx unknowns, reflecting BCs at i=0 and i=Nx-1
        n_i = Nx
        diag_i = np.full(n_i, 2.0 * rx)
        diag_i[0] = rx   # reflecting BC: reduce by rx
        diag_i[-1] = rx  # reflecting BC: reduce by rx
        off_i = np.full(n_i - 1, -rx)
        Lx_1d = diags([off_i, diag_i, off_i], [-1, 0, 1],
                       shape=(n_i, n_i), format="csr")

        # j-direction: (Ny-1) unknowns, no boundary reductions
        n_j = Ny - 1
        diag_j = np.full(n_j, 2.0 * ry)
        off_j = np.full(n_j - 1, -ry)
        Ly_1d = diags([off_j, diag_j, off_j], [-1, 0, 1],
                       shape=(n_j, n_j), format="csr")

        I_ni = eye(n_i, format="csr")
        I_nj = eye(n_j, format="csr")

        A = eye(n_i * n_j, format="csr") + kron(I_nj, Lx_1d) + kron(Ly_1d, I_ni)
        return A

    def solve(self, u, v, adv_u, adv_v):
        """Advance diffusion + advection to get u_star, v_star.

        Returns u_star, v_star with BCs applied. If the CG solver fails,
        u is set to NaN to signal blowup.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx**2, dy**2
        nu, dt = self.nu, self.dt

        u_star = u.copy()
        v_star = v.copy()

        # Explicit Laplacian of u
        lap_u = np.zeros_like(u)
        lap_u[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx2
        if u.shape[1] > 2:
            lap_u[:, 1:-1] += (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dy2

        # Explicit Laplacian of v
        lap_v = np.zeros_like(v)
        lap_v[1:-1, :] = (v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]) / dx2
        lap_v[0, :] += (v[1, :] - 2 * v[0, :] + (-v[0, :])) / dx2
        lap_v[-1, :] += (v[-2, :] - 2 * v[-1, :] + (-v[-1, :])) / dx2
        if v.shape[1] > 2:
            lap_v[:, 1:-1] += (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / dy2

        # u: RHS = u - dt*adv + 0.5*dt*nu*L(u) + Dirichlet contributions
        rhs_u = u[1:-1, 1:-1] - dt * adv_u[1:-1, 1:-1] + 0.5 * dt * nu * lap_u[1:-1, 1:-1]

        rhs_u[0, :] += 0.5 * nu * dt * self.bc.left / dx2
        rhs_u[-1, :] += 0.5 * nu * dt * self.bc.right / dx2
        rhs_u[:, 0] += 0.5 * nu * dt * self.bc.bottom / dy2

        lid = self.bc.lid_values(Nx)
        if self.bc.smooth_lid:
            rhs_u[:, -1] += 0.5 * nu * dt * lid[1:-1] / dy2
        else:
            rhs_u[:, -1] += 0.5 * nu * dt * lid / dy2

        u_flat, info = cg(self.A_u, rhs_u.flatten(), maxiter=self.cg_maxiter, rtol=self.cg_rtol)
        if info != 0:
            u_star[:] = np.nan
            return u_star, v_star
        u_star[1:-1, 1:-1] = u_flat.reshape((Nx - 1, Ny - 2))

        # v: RHS = v - dt*adv + 0.5*dt*nu*L(v)
        rhs_v = v[:, 1:-1] - dt * adv_v[:, 1:-1] + 0.5 * dt * nu * lap_v[:, 1:-1]

        v_flat, info = cg(self.A_v, rhs_v.flatten(), maxiter=self.cg_maxiter, rtol=self.cg_rtol)
        if info != 0:
            u_star[:] = np.nan
            return u_star, v_star
        v_star[:, 1:-1] = v_flat.reshape((Nx, Ny - 1))

        self.bc.apply(u_star, v_star, Nx, Ny)
        return u_star, v_star
