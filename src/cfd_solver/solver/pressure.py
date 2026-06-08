"""Pressure Poisson solver for staggered incompressible flow with ghost cells.

Builds a positive-Laplacian operator with Neumann walls and a pinned
reference pressure, then solves via conjugate gradient.
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


class PressureSolver:
    """Pressure Poisson solver with Neumann boundary conditions.

    Uses a pre-factorized sparse Laplacian operator for speed.
    """

    def __init__(self, mesh, cg_maxiter=1000, cg_rtol=1e-5):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        # Build constant Poisson matrix
        self.A = self._build_matrix()
        self._solve = splu(self.A).solve

    def _build_matrix(self):
        Nx, Ny = self.Nx, self.Ny
        inv_dx2 = 1.0 / (self.dx**2)
        inv_dy2 = 1.0 / (self.dy**2)

        # 1D Laplacian with Neumann boundaries in x
        diag_x = np.full(Nx, 2.0 * inv_dx2)
        diag_x[0] = inv_dx2
        diag_x[-1] = inv_dx2
        off_x = np.full(Nx - 1, -inv_dx2)
        Lx = diags([off_x, diag_x, off_x], [-1, 0, 1], shape=(Nx, Nx), format="csr")

        # 1D Laplacian with Neumann boundaries in y
        diag_y = np.full(Ny, 2.0 * inv_dy2)
        diag_y[0] = inv_dy2
        diag_y[-1] = inv_dy2
        off_y = np.full(Ny - 1, -inv_dy2)
        Ly = diags([off_y, diag_y, off_y], [-1, 0, 1], shape=(Ny, Ny), format="csr")

        I_Nx = eye(Nx, format="csr")
        I_Ny = eye(Ny, format="csr")

        # Combine into 2D Laplacian: A = I_Ny \otimes Lx + Ly \otimes I_Nx
        # Note: kron(I_Ny, Lx) handles x-derivatives, kron(Ly, I_Nx) handles y-derivatives
        A = kron(I_Ny, Lx) + kron(Ly, I_Nx)

        # Pin pressure at first cell (0,0) to ensure a unique solution
        A = A.tolil()
        # Zero out the first row and set diagonal to 1
        A[0, :] = 0
        A[0, 0] = 1.0
        # To maintain symmetry (optional for splu but good practice), zero the column too
        # But we'd need to adjust the RHS. For splu, just zeroing the row is enough.
        
        return A.tocsc()

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation and return p (Nx+2, Ny+2)."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        # Compute divergence over active cells
        div = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
        rhs = (-div / dt).ravel(order="F")
        
        # Pin pressure at first cell to 0
        rhs[0] = 0.0

        p_flat = self._solve(rhs)

        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        # Populate interior physical cells
        p[1:-1, 1:-1] = p_flat.reshape((Nx, Ny), order="F")

        # Normalize to zero mean
        p[1:-1, 1:-1] -= np.mean(p[1:-1, 1:-1])

        # Enforce Neumann BCs on pressure ghost cells
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        return p

