"""Pressure Poisson solver for staggered incompressible flow.

Builds a positive-Laplacian operator with Neumann walls and a pinned
reference pressure, then solves via conjugate gradient.

The operator is applied as a matrix-free ``LinearOperator`` backed by a
numba-JIT stencil kernel. This avoids the sparse-matrix build cost and
removes the per-iteration Python wrapper overhead of scipy's
``csr_matvec``, both of which dominated runtime for small grids.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

import numba


@numba.njit(cache=True)
def _pressure_matvec_kernel(p_flat, out, Nx, Ny, inv_dx2, inv_dy2):
    """Apply the pressure Poisson operator to ``p_flat`` (1D, length Nx*Ny).

    Layout: ``k = i * Ny + j``.

    The operator is the Kronecker sum ``Lx (x) I_Ny + I_Nx (x) Ly`` where
    ``Lx`` and ``Ly`` are the 1D positive-definite Neumann Laplacians
    (diagonal halved at the ends). The stencil form is
    ``(2*c - left - right) * inv_h2`` in the interior and
    ``(c - neighbor) * inv_h2`` at the boundary. Cell ``(0, 0)`` is
    pinned: its row is the unit vector, and its column is zeroed (so
    the stencil at adjacent cells omits the ``(0, 0)`` term).
    """
    for i in range(Nx):
        base_i = i * Ny
        for j in range(Ny):
            k = base_i + j
            if k == 0:
                out[k] = p_flat[k]
                continue

            if i == 0:
                ddx2 = (p_flat[k] - p_flat[k + Ny]) * inv_dx2
            elif i == Nx - 1:
                ddx2 = (p_flat[k] - p_flat[k - Ny]) * inv_dx2
            elif k - Ny == 0:
                ddx2 = (2.0 * p_flat[k] - p_flat[k + Ny]) * inv_dx2
            else:
                ddx2 = (2.0 * p_flat[k] - p_flat[k - Ny] - p_flat[k + Ny]) * inv_dx2

            if j == 0:
                ddy2 = (p_flat[k] - p_flat[k + 1]) * inv_dy2
            elif j == Ny - 1:
                ddy2 = (p_flat[k] - p_flat[k - 1]) * inv_dy2
            elif k - 1 == 0:
                ddy2 = (2.0 * p_flat[k] - p_flat[k + 1]) * inv_dy2
            else:
                ddy2 = (2.0 * p_flat[k] - p_flat[k - 1] - p_flat[k + 1]) * inv_dy2

            out[k] = ddx2 + ddy2


@numba.njit(cache=True)
def _norm2(v):
    s = 0.0
    for i in range(len(v)):
        s += v[i] * v[i]
    return np.sqrt(s)


@numba.njit(cache=True)
def _cg_solve(x, b, Nx, Ny, inv_dx2, inv_dy2, maxiter, rtol):
    n = Nx * Ny
    r = np.empty(n, dtype=np.float64)
    # r = b - A x_0
    _pressure_matvec_kernel(x, r, Nx, Ny, inv_dx2, inv_dy2)
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
        _pressure_matvec_kernel(p, w, Nx, Ny, inv_dx2, inv_dy2)

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

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation and return p (Nx, Ny).

        Parameters
        ----------
        u_star, v_star : ndarray
            Intermediate velocity fields from the predictor step.
        dt : float
            Time step.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        div = (u_star[1:, :] - u_star[:-1, :]) / dx + (v_star[:, 1:] - v_star[:, :-1]) / dy
        rhs = (-div / dt).flatten()
        rhs[0] = 0.0

        p_flat = np.zeros(Nx * Ny, dtype=np.float64)
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        info = _cg_solve(
            p_flat, rhs, Nx, Ny, inv_dx2, inv_dy2, self.cg_maxiter, self.cg_rtol
        )
        if info < 0 or info == self.cg_maxiter:
            raise RuntimeError(f"Pressure CG failed to converge (info={info})")

        p = p_flat.reshape((Nx, Ny))
        p -= np.mean(p)
        return p
