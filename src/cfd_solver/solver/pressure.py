"""Pressure Poisson solver for staggered incompressible flow.

Builds the positive-Laplacian matrix with Neumann walls and a pinned
reference pressure, then solves via conjugate gradient.
"""

import numpy as np
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import cg


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
        self.A = self._build_matrix()

    def _build_matrix(self):
        """Build the positive-Laplacian pressure matrix with Neumann walls.

        Uses Kronecker products of 1D Neumann Laplacians — no Python loops.
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        # 1D negative Laplacian with Neumann BCs
        def _neumann_1d(n, h2):
            diag = np.full(n, 2.0 / h2)
            diag[0] = 1.0 / h2
            diag[-1] = 1.0 / h2
            off = np.full(n - 1, -1.0 / h2)
            return diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr")

        Lx = _neumann_1d(Nx, dx2)
        Ly = _neumann_1d(Ny, dy2)

        # 2D Laplacian: k = i*Ny + j ordering
        A = kron(Lx, eye(Ny, format="csr"), format="csr") + \
            kron(eye(Nx, format="csr"), Ly, format="csr")

        # Pin one pressure value to remove the nullspace
        A = A.tolil()
        A[0, :] = 0
        A[:, 0] = 0
        A[0, 0] = 1.0

        return A.tocsr()

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

        p_flat, info = cg(self.A, rhs, maxiter=self.cg_maxiter, rtol=self.cg_rtol)
        if info != 0:
            raise RuntimeError(f"Pressure CG failed to converge (info={info})")

        p = p_flat.reshape((Nx, Ny))
        p -= np.mean(p)
        return p
