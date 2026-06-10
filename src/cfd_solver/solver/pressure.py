"""Pressure Poisson solver for staggered incompressible flow.

This module implements the pressure projection step of Chorin's method.
After an intermediate velocity field (u*, v*) is calculated (accounting for
advection and diffusion), it generally will not satisfy the incompressibility
constraint (∇·u = 0).

Pressure Poisson Equation (PPE):
--------------------------------
The projection step finds a pressure field 'p' such that the corrected
velocity u = u* - dt*∇p is divergence-free. Taking the divergence of this
equation leads to the PPE:
    ∇²p = (∇·u*) / dt

Solving this Poisson equation with Neumann boundary conditions (∂p/∂n = 0)
allows us to project the intermediate velocity onto the space of
divergence-free fields.
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


class PressureSolver:
    """Pressure Poisson solver with Neumann boundary conditions.

    The solver builds a discrete Laplacian operator (A) and uses a direct
    sparse solver (superLU via splu) for efficiency. Since the Poisson
    problem with pure Neumann BCs is singular (defined only up to an
    additive constant), we pin one pressure value to zero to ensure a
    unique solution.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    cg_maxiter : int, optional
        Maximum iterations for the iterative solver (if used).
    cg_rtol : float, optional
        Relative tolerance for the iterative solver (if used).

    Attributes
    ----------
    A : csc_matrix
        The discrete Laplacian operator.
    """

    def __init__(self, mesh, cg_maxiter=1000, cg_rtol=1e-5):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        # Build the constant Poisson matrix once during initialization
        self.A = self._build_matrix()
        self._solve = splu(self.A).solve

    def _build_matrix(self):
        """Build the discrete 2D Laplacian operator with Neumann BCs."""
        Nx, Ny = self.Nx, self.Ny
        inv_dx2 = 1.0 / (self.dx**2)
        inv_dy2 = 1.0 / (self.dy**2)

        # 1D Laplacian with Neumann boundaries in x
        # ∂²p/∂x² ≈ (p[i+1] - 2p[i] + p[i-1]) / dx²
        # At boundaries, p[ghost] = p[neighbor] => (p[neighbor] - p[i]) / dx²
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

        # Combine into 2D Laplacian via Kronecker product: A = Ly ⊕ Lx
        A = kron(I_Ny, Lx) + kron(Ly, I_Nx)

        # Pin pressure at the first cell (0,0) to ensure a unique solution
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1.0
        
        return A.tocsc()

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation.

        Parameters
        ----------
        u_star, v_star : ndarray
            The intermediate velocity field.
        dt : float
            Time step.

        Returns
        -------
        p : ndarray, shape (Nx+2, Ny+2)
            The calculated pressure field including ghost cells.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        # Compute divergence of intermediate velocity over active cells
        div = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
        
        # RHS of Poisson eq: ∇²p = (∇·u*) / dt.
        # The assembled operator A has a positive diagonal / negative
        # off-diagonals, i.e. A = -∇² (positive-definite form). Solving
        # A·p = rhs therefore yields -∇²p = rhs. To recover ∇²p = (∇·u*)/dt
        # we must negate the RHS, otherwise the projection ADDS divergence
        # instead of removing it and the simulation blows up.
        rhs = (-div / dt).ravel(order="F")
        
        # Pin pressure at first cell to 0
        rhs[0] = 0.0

        p_flat = self._solve(rhs)

        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        # Populate interior physical cells
        p[1:-1, 1:-1] = p_flat.reshape((Nx, Ny), order="F")

        # Normalize to zero mean for consistency
        p[1:-1, 1:-1] -= np.mean(p[1:-1, 1:-1])

        # Enforce Neumann BCs on pressure ghost cells (zero gradient)
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        return p

