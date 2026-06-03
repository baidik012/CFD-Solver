"""Diffusion schemes for staggered incompressible flow.

Available schemes:
  - explicit: Forward Euler (stable for dt <= dx^2 / (4*nu))
  - crank_nicolson: Semi-implicit Crank-Nicolson (unconditionally stable)
"""

import numpy as np
from scipy.sparse import lil_matrix
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
        """Build (I - 0.5*dt*nu*L) for u unknowns: interior faces i=1..Nx-1, j=1..Ny-2."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        n_u = (Nx - 1) * (Ny - 2)
        A = lil_matrix((n_u, n_u))

        for i in range(1, Nx):
            for j in range(1, Ny - 1):
                k = (i - 1) * (Ny - 2) + (j - 1)
                diag = 1.0 + 2.0 * rx + 2.0 * ry

                if i > 1:
                    A[k, k - (Ny - 2)] = -rx
                if i < Nx - 1:
                    A[k, k + (Ny - 2)] = -rx
                if j > 1:
                    A[k, k - 1] = -ry
                if j < Ny - 2:
                    A[k, k + 1] = -ry

                A[k, k] = diag

        return A.tocsr()

    def _build_v_matrix(self):
        """Build (I - 0.5*dt*nu*L) for v unknowns: all i, interior j=1..Ny-1."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2
        rx = 0.5 * self.nu * self.dt / dx2
        ry = 0.5 * self.nu * self.dt / dy2

        n_v = Nx * (Ny - 1)
        A = lil_matrix((n_v, n_v))

        for i in range(Nx):
            for j in range(1, Ny):
                k = i * (Ny - 1) + (j - 1)
                diag = 1.0 + 2.0 * ry

                if i == 0:
                    diag += rx
                elif i > 0:
                    A[k, k - (Ny - 1)] = -rx

                if i == Nx - 1:
                    diag += rx
                elif i < Nx - 1:
                    A[k, k + (Ny - 1)] = -rx

                if j > 1:
                    A[k, k - 1] = -ry
                if j < Ny - 1:
                    A[k, k + 1] = -ry

                A[k, k] = diag

        return A.tocsr()

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
