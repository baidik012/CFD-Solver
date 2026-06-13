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

Two solver backends are provided:

* :class:`PressureSolver` — direct sparse solver (SuperLU via ``splu``).
  Factorization is O(N^1.5) but each back-substitution is O(N). Best for
  small-to-medium grids where the factorization cost is amortized.

* :class:`FFTPressureSolver` — spectral solver using the Discrete Cosine
  Transform (DCT-II). The Neumann-Laplacian eigenvectors on a uniform
  rectangular grid are known analytically as DCT-II basis functions, so
  the Poisson equation diagonalises in the frequency domain. Each solve
  is O(N log N) with no factorization step. Best for large grids (≥ 128²).

The factory function :func:`create_pressure_solver` picks the appropriate
backend automatically based on grid size.
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.fft import dctn, idctn


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

    Attributes
    ----------
    A : csc_matrix
        The discrete Laplacian operator.
    """

    def __init__(self, mesh):
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


class FFTPressureSolver:
    """Spectral pressure Poisson solver using DCT-II.

    On a uniform rectangular grid the discrete Neumann Laplacian eigenvectors
    are the DCT-II basis functions cos(πk(i+½)/N). Dividing the DCT of the
    right-hand side by the corresponding eigenvalues yields the solution in
    O(Nx·Ny·log(Nx·Ny)) time per solve, with no matrix factorization step.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.

    Attributes
    ----------
    eig_2d : ndarray, shape (Nx, Ny)
        Precomputed eigenvalues of the 2D discrete Laplacian.  The zero-mode
        (0,0) is set to ``inf`` so that dividing by it produces zero,
        effectively pinning the mean pressure to zero.
    """

    def __init__(self, mesh):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        # Precompute eigenvalues: λ = λx + λy
        # λx_k = 2(1 - cos(πk/Nx)) / dx²   (k = 0 … Nx-1)
        # These are the eigenvalues of the 1D Neumann Laplacian that arise
        # from the Kronecker-product assembly of the 2D operator.
        kx = np.arange(self.Nx)
        ky = np.arange(self.Ny)
        eig_x = 2.0 * (1.0 - np.cos(np.pi * kx / self.Nx)) / (self.dx ** 2)
        eig_y = 2.0 * (1.0 - np.cos(np.pi * ky / self.Ny)) / (self.dy ** 2)
        eig_2d = eig_x[:, np.newaxis] + eig_y[np.newaxis, :]

        # Pin the zero (constant) mode to avoid division by zero.
        eig_2d[0, 0] = np.inf
        self.eig_2d = eig_2d

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
        div = (
            (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx
            + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
        )

        # RHS: -∇²p = -(∇·u*)/dt  (the assembled operator A = -∇² is
        # positive-definite, so we negate the divergence to match).
        rhs = (-div / dt)

        # Forward DCT-II (orthonormal): transforms into the eigenbasis
        rhs_hat = dctn(rhs, type=2, norm="ortho", axes=(0, 1))

        # Solve in frequency domain: P̂ = R̂ / λ
        # The zero-mode (λ=∞) is automatically zeroed by floating-point
        # division, but we set it explicitly for clarity.
        p_hat = rhs_hat / self.eig_2d
        p_hat[0, 0] = 0.0

        # Inverse DCT-II to return to physical space
        p_interior = idctn(p_hat, type=2, norm="ortho", axes=(0, 1))

        # Pack into ghost-cell array
        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        p[1:-1, 1:-1] = p_interior

        # Enforce Neumann BCs on pressure ghost cells (zero gradient)
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        return p


# Threshold above which the FFT solver is used.
# For grids larger than FFT_THRESHOLD × FFT_THRESHOLD, the O(N log N) FFT
# solver is faster than the O(N^1.5) splu back-substitution.
FFT_THRESHOLD = 128


def create_pressure_solver(mesh):
    """Create the appropriate pressure solver for the given mesh.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.

    Returns
    -------
    PressureSolver or FFTPressureSolver
        For grids with min(Nx, Ny) ≤ :data:`FFT_THRESHOLD`, returns a
        :class:`PressureSolver` (direct sparse).  For larger grids returns
        an :class:`FFTPressureSolver` (spectral DCT-II).
    """
    if mesh.Nx <= FFT_THRESHOLD and mesh.Ny <= FFT_THRESHOLD:
        return PressureSolver(mesh)
    return FFTPressureSolver(mesh)

