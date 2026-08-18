"""Pressure Poisson solver for staggered incompressible flow.

Solves ∇²p = (∇·u*)/dt with Neumann (∂p/∂n = 0) or periodic boundary
conditions. The intermediate velocity is then projected onto the
divergence-free space via u = u* − dt·∇p.

Backends:
- PressureSolver: direct sparse (splu); handles arbitrary BCs.
- FFTPressureSolver: DCT-II spectral; pure Neumann only.
- PeriodicPressureSolver: DFT (x) + DCT-II (y); periodic x + Neumann y.
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.fft import dctn, idctn


class PressureSolver:
    """Pressure Poisson solver with Neumann boundary conditions."""

    def __init__(self, mesh, bc=None):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.bc = bc

        self._outlet_cols = self._find_outlet_columns()
        self.A = self._build_matrix()
        self._solve = splu(self.A).solve

    def _find_outlet_columns(self):
        """Return pressure unknown indices adjacent to outlet walls."""
        cols = set()
        if self.bc is None:
            return cols
        from .bc import OutletWall

        if isinstance(self.bc.walls.get('left'), OutletWall):
            for j in range(self.Ny):
                cols.add(j * self.Nx)
        if isinstance(self.bc.walls.get('right'), OutletWall):
            for j in range(self.Ny):
                cols.add(j * self.Nx + self.Nx - 1)
        if isinstance(self.bc.walls.get('bottom'), OutletWall):
            for i in range(self.Nx):
                cols.add(i)
        if isinstance(self.bc.walls.get('top'), OutletWall):
            top_row = (self.Ny - 1) * self.Nx
            for i in range(self.Nx):
                cols.add(top_row + i)
        return cols

    def _build_matrix(self):
        """Build the discrete 2D Laplacian operator with Neumann BCs."""
        Nx, Ny = self.Nx, self.Ny
        inv_dx2 = 1.0 / (self.dx**2)
        inv_dy2 = 1.0 / (self.dy**2)

        diag_x = np.full(Nx, 2.0 * inv_dx2)
        diag_x[0] = inv_dx2
        diag_x[-1] = inv_dx2
        off_x = np.full(Nx - 1, -inv_dx2)
        Lx = diags([off_x, diag_x, off_x], [-1, 0, 1], shape=(Nx, Nx), format="csr")

        diag_y = np.full(Ny, 2.0 * inv_dy2)
        diag_y[0] = inv_dy2
        diag_y[-1] = inv_dy2
        off_y = np.full(Ny - 1, -inv_dy2)
        Ly = diags([off_y, diag_y, off_y], [-1, 0, 1], shape=(Ny, Ny), format="csr")

        I_Nx = eye(Nx, format="csr")
        I_Ny = eye(Ny, format="csr")
        A = kron(I_Ny, Lx) + kron(Ly, I_Nx)
        A = A.tolil()

        if self._outlet_cols:
            for col in self._outlet_cols:
                A[col, :] = 0
                A[:, col] = 0
                A[col, col] = 1.0
        else:
            A[0, :] = 0
            A[0, 0] = 1.0

        return A.tocsc()

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        div = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
        rhs = (-div / dt).ravel(order="F")

        if self._outlet_cols:
            for col in self._outlet_cols:
                rhs[col] = 0.0
        else:
            rhs[0] = 0.0

        p_flat = self._solve(rhs)

        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        p[1:-1, 1:-1] = p_flat.reshape((Nx, Ny), order="F")

        if not self._outlet_cols:
            p[1:-1, 1:-1] -= np.mean(p[1:-1, 1:-1])

        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        return p


class FFTPressureSolver:
    """Spectral pressure Poisson solver using DCT-II (pure Neumann BCs)."""

    def __init__(self, mesh):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy

        kx = np.arange(self.Nx)
        ky = np.arange(self.Ny)
        eig_x = 2.0 * (1.0 - np.cos(np.pi * kx / self.Nx)) / (self.dx ** 2)
        eig_y = 2.0 * (1.0 - np.cos(np.pi * ky / self.Ny)) / (self.dy ** 2)
        eig_2d = eig_x[:, np.newaxis] + eig_y[np.newaxis, :]
        eig_2d[0, 0] = np.inf
        self.eig_2d = eig_2d

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        div = (
            (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx
            + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
        )
        rhs = (-div / dt)
        rhs_hat = dctn(rhs, type=2, norm="ortho", axes=(0, 1))
        p_hat = rhs_hat / self.eig_2d
        p_hat[0, 0] = 0.0
        p_interior = idctn(p_hat, type=2, norm="ortho", axes=(0, 1))
        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        p[1:-1, 1:-1] = p_interior
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        return p


class PeriodicPressureSolver:
    """Spectral pressure Poisson solver for periodic x + Neumann y."""

    def __init__(self, mesh):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy
        Nx, Ny = self.Nx, self.Ny
        kx = np.arange(Nx)
        eig_x = 2.0 * (1.0 - np.cos(2.0 * np.pi * kx / Nx)) / (self.dx ** 2)
        ky = np.arange(Ny)
        eig_y = 2.0 * (1.0 - np.cos(np.pi * ky / Ny)) / (self.dy ** 2)
        eig_2d = eig_x[:, np.newaxis] + eig_y[np.newaxis, :]
        eig_2d[0, 0] = np.inf
        self.eig_2d = eig_2d

    def solve(self, u_star, v_star, dt):
        """Solve the pressure Poisson equation."""
        Nx, Ny = self.Nx, self.Ny
        div = (
            (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / self.dx
            + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / self.dy
        )
        rhs = (-div / dt)
        from scipy.fft import fft, ifft, dctn, idctn
        rhs_hat = dctn(rhs, type=2, norm="ortho", axes=(1,))
        rhs_hat = fft(rhs_hat, axis=0)
        p_hat = rhs_hat / self.eig_2d
        p_hat[0, 0] = 0.0
        p_interior = np.real(ifft(p_hat, axis=0))
        p_interior = idctn(p_interior, type=2, norm="ortho", axes=(1,))
        p = np.zeros((Nx + 2, Ny + 2), dtype=np.float64)
        p[1:-1, 1:-1] = p_interior
        p[0, :] = p[Nx, :]
        p[-1, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        return p


FFT_THRESHOLD = 128


def create_pressure_solver(mesh, bc=None):
    """Create the appropriate pressure solver for the given mesh."""
    if bc is not None:
        if bc.has_periodic():
            return PeriodicPressureSolver(mesh)
        if bc.has_outlet() or (mesh.Nx <= FFT_THRESHOLD and mesh.Ny <= FFT_THRESHOLD):
            return PressureSolver(mesh, bc=bc)

    if mesh.Nx <= FFT_THRESHOLD and mesh.Ny <= FFT_THRESHOLD:
        return PressureSolver(mesh, bc=bc)
    return FFTPressureSolver(mesh)
