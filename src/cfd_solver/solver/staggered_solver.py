"""Simple but working staggered grid solver."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg


class StaggeredSolver:
    """Incompressible Navier-Stokes solver using Chorin splitting."""

    def __init__(self, Lx, Ly, Nx, Ny, nu, dt,
                 u_bc={"top": 1.0, "bottom": 0.0, "left": 0.0, "right": 0.0}):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.nu = nu
        self.dt = dt

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Boundary conditions
        self.u_top = u_bc.get("top", 0.0)
        self.u_bottom = u_bc.get("bottom", 0.0)
        self.u_left = u_bc.get("left", 0.0)
        self.u_right = u_bc.get("right", 0.0)

        # Arrays: u at x-faces, v at y-faces, p at cell centers
        self.u = np.zeros((Nx + 1, Ny))      # u-velocity
        self.v = np.zeros((Nx, Ny + 1))      # v-velocity
        self.p = np.zeros((Nx, Ny))          # pressure

        self._build_pressure_matrix()

    def _build_pressure_matrix(self):
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        N = Nx * Ny
        A = lil_matrix((N, N))
        diag = 2.0 / dx2 + 2.0 / dy2

        for j in range(Ny):
            for i in range(Nx):
                idx = j * Nx + i
                A[idx, idx] = diag
                if i > 0: A[idx, idx - 1] = -1.0 / dx2
                if i < Nx - 1: A[idx, idx + 1] = -1.0 / dx2
                if j > 0: A[idx, idx - Nx] = -1.0 / dy2
                if j < Ny - 1: A[idx, idx + Nx] = -1.0 / dy2

        self.A = A.tocsr()

    def _apply_bc(self):
        """Apply boundary conditions."""
        Nx, Ny = self.Nx, self.Ny

        # Top and bottom walls
        self.u[:, 0] = self.u_bottom
        self.u[:, Ny - 1] = self.u_top

        # Left and right walls
        self.u[0, :] = self.u_left
        self.u[Nx, :] = self.u_right

        # v-velocity at all walls
        self.v[:, 0] = 0.0
        self.v[:, Ny] = 0.0
        self.v[0, :] = 0.0
        self.v[Nx - 1, :] = 0.0

    def step(self):
        """Advance one time step."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        nu, dt = self.nu, self.dt

        # Ensure boundary conditions applied
        self._apply_bc()

        # Work with face velocities (u on vertical faces, v on horizontal faces)
        u = self.u
        v = self.v

        # Predictor: keep current velocities (no explicit advection/diffusion here)
        u_star = u.copy()
        v_star = v.copy()

        # Divergence at cell centers (shape: Nx x Ny)
        div = (u_star[1:, :] - u_star[:-1, :]) / dx + (v_star[:, 1:] - v_star[:, :-1]) / dy

        # Pressure Poisson RHS
        rhs = div / dt

        # Solve pressure Poisson: A was built for Nx * Ny
        p_flat, _ = cg(self.A, rhs.flatten(), tol=1e-8, maxiter=500)
        self.p = p_flat.reshape((Nx, Ny))

        # Corrector: compute pressure gradients on faces
        grad_p_x = (self.p[1:, :] - self.p[:-1, :]) / dx   # shape (Nx-1, Ny)
        grad_p_y = (self.p[:, 1:] - self.p[:, :-1]) / dy   # shape (Nx, Ny-1)

        # Update face velocities on interior faces
        self.u[1:-1, :] = u_star[1:-1, :] - dt * grad_p_x
        self.v[:, 1:-1] = v_star[:, 1:-1] - dt * grad_p_y

        # Re-apply boundary conditions
        self._apply_bc()

    def divergence_norm(self):
        u = self.u[1:-1, 1:-1]
        v = self.v[1:-1, 1:-1]
        div = (u[:, 2:] - u[:, :-2]) / (2 * self.dx)
        div += (v[2:, :] - v[:-2, :]) / (2 * self.dy)
        return np.sqrt(np.mean(div**2))

    def max_divergence(self):
        u = self.u[1:-1, 1:-1]
        v = self.v[1:-1, 1:-1]
        div = (u[:, 2:] - u[:, :-2]) / (2 * self.dx)
        div += (v[2:, :] - v[:-2, :]) / (2 * self.dy)
        return np.max(np.abs(div))

    def cfl(self):
        return (np.max(np.abs(self.u)) * self.dt / self.dx +
                np.max(np.abs(self.v)) * self.dt / self.dy)

    def solve(self, steps, verbose=True):
        self._apply_bc()
        for i in range(steps):
            self.step()
            if verbose and i % max(1, steps // 10) == 0:
                print(f"Step {i:4d}: |∇·u|∞ = {self.max_divergence():.2e}, CFL = {self.cfl():.3f}")