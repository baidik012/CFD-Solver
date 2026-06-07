"""Pressure Poisson solver for staggered incompressible flow with ghost cells.

Builds a positive-Laplacian operator with Neumann walls and a pinned
reference pressure, then solves via conjugate gradient.
"""

import numpy as np
import numba


@numba.njit(cache=True)
def _pressure_matvec_kernel(p_active_flat, out_active_flat, p_2d, Nx, Ny, inv_dx2, inv_dy2):
    """Apply the pressure Poisson operator to active interior cells.

    Uses pressure ghost cells inside p_2d (1D array of size (Nx+2)*(Ny+2)) 
    to automatically enforce Neumann boundary conditions.
    """
    stride = Ny + 2

    # 1. Populate interior of p_2d from active flat array
    for i in range(Nx):
        base_2d = (i + 1) * stride
        base_active = i * Ny
        for j in range(Ny):
            p_2d[base_2d + j + 1] = p_active_flat[base_active + j]

    # 2. Enforce Neumann boundary conditions on p_2d ghost cells
    # Left and right walls
    for j in range(Ny + 2):
        p_2d[0 * stride + j] = p_2d[1 * stride + j]
        p_2d[(Nx + 1) * stride + j] = p_2d[Nx * stride + j]
    # Bottom and top walls
    for i in range(Nx + 2):
        base_i = i * stride
        p_2d[base_i + 0] = p_2d[base_i + 1]
        p_2d[base_i + Ny + 1] = p_2d[base_i + Ny]

    # 3. Pin the first physical cell (i=0, j=0, active index 0)
    out_active_flat[0] = p_active_flat[0]

    # 4. Apply uniform 5-point stencil for all other active cells
    # Active index: k = i * Ny + j
    for i in range(Nx):
        i_2d = i + 1
        base_2d = i_2d * stride
        base_active = i * Ny
        for j in range(Ny):
            k = base_active + j
            if k == 0:
                continue

            j_2d = j + 1

            left = p_2d[(i_2d - 1) * stride + j_2d]
            right = p_2d[(i_2d + 1) * stride + j_2d]
            bottom = p_2d[base_2d + j_2d - 1]
            top = p_2d[base_2d + j_2d + 1]
            center = p_2d[base_2d + j_2d]

            # Column zeroing: omit connections to pinned cell (active index 0 at interior coordinates (1,1))
            if i_2d - 1 == 1 and j_2d == 1:
                left = 0.0
            if i_2d + 1 == 1 and j_2d == 1:
                right = 0.0
            if i_2d == 1 and j_2d - 1 == 1:
                bottom = 0.0
            if i_2d == 1 and j_2d + 1 == 1:
                top = 0.0

            ddx2 = (2.0 * center - left - right) * inv_dx2
            ddy2 = (2.0 * center - bottom - top) * inv_dy2

            out_active_flat[k] = ddx2 + ddy2


@numba.njit(cache=True)
def _norm2(v):
    s = 0.0
    for i in range(len(v)):
        s += v[i] * v[i]
    return np.sqrt(s)


@numba.njit(cache=True)
def _cg_solve(x, b, p_2d, Nx, Ny, inv_dx2, inv_dy2, maxiter, rtol):
    n = Nx * Ny
    r = np.empty(n, dtype=np.float64)
    # r = b - A x_0
    _pressure_matvec_kernel(x, r, p_2d, Nx, Ny, inv_dx2, inv_dy2)
    for i in range(n):
        r[i] = b[i] - r[i]

    bnrm2 = _norm2(b)
    if bnrm2 == 0.0:
        x.fill(0.0)
        return 0

    tol = rtol * bnrm2
    rnrm2 = _norm2(r)
    if rnrm2 < tol:
        return 0

    p = r.copy()
    rho = rnrm2 * rnrm2
    w = np.empty(n, dtype=np.float64)

    for k in range(maxiter):
        _pressure_matvec_kernel(p, w, p_2d, Nx, Ny, inv_dx2, inv_dy2)

        p_dot_w = 0.0
        for i in range(n):
            p_dot_w += p[i] * w[i]

        if p_dot_w == 0.0:
            return -1  # Failure to avoid division by zero

        alpha = rho / p_dot_w

        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * w[i]

        rnrm2 = _norm2(r)
        if rnrm2 < tol:
            return 0

        rho_new = rnrm2 * rnrm2
        beta = rho_new / rho
        rho = rho_new

        for i in range(n):
            p[i] = r[i] + beta * p[i]

    return maxiter


class PressureSolver:
    """Pressure Poisson solver with Neumann boundary conditions.

    Parameters
    ----------
    mesh : Mesh
    cg_maxiter : int
    cg_rtol : float
    """

    def __init__(self, mesh, cg_maxiter=1000, cg_rtol=1e-5):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.cg_maxiter = cg_maxiter
        self.cg_rtol = cg_rtol

        # Preallocate work arrays to avoid JIT allocation overhead
        self._work_2d = np.empty((self.Nx + 2) * (self.Ny + 2), dtype=np.float64)

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation and return p (Nx+2, Ny+2).

        Parameters
        ----------
        u_star, v_star : ndarray
            Intermediate velocity fields from the predictor step.
        dt : float
            Time step.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        # Compute divergence over active cells
        div = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
        rhs = (-div / dt).flatten()
        rhs[0] = 0.0

        p_flat = np.zeros(Nx * Ny, dtype=np.float64)
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        info = _cg_solve(
            p_flat, rhs, self._work_2d, Nx, Ny, inv_dx2, inv_dy2, self.cg_maxiter, self.cg_rtol
        )
        if info < 0 or info == self.cg_maxiter:
            raise RuntimeError(f"Pressure CG failed to converge (info={info})")

        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        # Populate interior physical cells
        for i in range(Nx):
            p[i + 1, 1:-1] = p_flat[i * Ny : (i + 1) * Ny]

        p[1:-1, 1:-1] -= np.mean(p[1:-1, 1:-1])

        # Enforce Neumann BCs on pressure ghost cells
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        return p
