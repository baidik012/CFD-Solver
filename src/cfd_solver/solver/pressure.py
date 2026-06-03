"""Pressure Poisson solver for staggered incompressible flow.

Builds the positive-Laplacian matrix with Neumann walls and a pinned
reference pressure, then solves via conjugate gradient.
"""

import numpy as np
from scipy.sparse import lil_matrix
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
        """Build the positive-Laplacian pressure matrix with Neumann walls."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        N = Nx * Ny
        A = lil_matrix((N, N))

        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                diag = 2.0 / dx2 + 2.0 / dy2

                if i > 0:
                    A[idx, idx - Ny] = -1.0 / dx2
                else:
                    diag -= 1.0 / dx2

                if i < Nx - 1:
                    A[idx, idx + Ny] = -1.0 / dx2
                else:
                    diag -= 1.0 / dx2

                if j > 0:
                    A[idx, idx - 1] = -1.0 / dy2
                else:
                    diag -= 1.0 / dy2

                if j < Ny - 1:
                    A[idx, idx + 1] = -1.0 / dy2
                else:
                    diag -= 1.0 / dy2

                A[idx, idx] = diag

        # Pin one pressure value to remove the nullspace
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
