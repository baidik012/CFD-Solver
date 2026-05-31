"""Chorin projection method solver."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
from .boundaries import BoundaryConditions
from .grid import Grid


class Solver:
    """Incompressible Navier-Stokes solver using Chorin's method."""

    def __init__(self, grid: Grid, nu: float, dt: float, bc: BoundaryConditions):
        self.grid = grid
        self.nu = nu
        self.dt = dt
        self.bc = bc

        # Use staggered grid dimensions from grid
        self.u = np.zeros(grid.shape_u)  # (Nx+1, Ny)
        self.v = np.zeros(grid.shape_v)  # (Nx, Ny+1)
        self.p = np.zeros(grid.shape_p)  # (Nx, Ny)

        self._build_pressure_matrix()
        bc.apply(self.u, self.v)

    def _build_pressure_matrix(self):
        """Build positive -Laplacian pressure matrix with Neumann walls."""
        Nx, Ny = self.grid.Nx, self.grid.Ny
        dx2, dy2 = self.grid.dx**2, self.grid.dy**2
        A = lil_matrix((Nx * Ny, Nx * Ny))

        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                diag = 0.0

                if i > 0:
                    A[idx, idx - Ny] = -1.0 / dx2
                    diag += 1.0 / dx2
                if i < Nx - 1:
                    A[idx, idx + Ny] = -1.0 / dx2
                    diag += 1.0 / dx2
                if j > 0:
                    A[idx, idx - 1] = -1.0 / dy2
                    diag += 1.0 / dy2
                if j < Ny - 1:
                    A[idx, idx + 1] = -1.0 / dy2
                    diag += 1.0 / dy2

                A[idx, idx] = diag

        # Fix one pressure value to remove the Neumann nullspace while
        # keeping the matrix symmetric for conjugate gradient.
        A[0, :] = 0.0
        A[:, 0] = 0.0
        A[0, 0] = 1.0
        self.pressure_matrix = A.tocsr()

    def _advection(self, u, v, dx, dy):
        """Compute advection terms (u*du/dx + v*du/dy) for u-momentum and
        (u*dv/dx + v*dv/dy) for v-momentum at cell centers.

        On a staggered grid, u is at x-faces (Nx+1, Ny) and v is at y-faces (Nx, Ny+1).
        Advection must be computed at cell centers, requiring interpolation from faces.
        """
        # For u-momentum at u-faces (i+1/2, j), interior is [1:-1, 1:-1] -> shape (Nx-1, Ny-2)
        # Note: u has shape (Nx+1, Ny), so interior [1:-1, 1:-1] removes 2 rows in x and 2 in y,
        # giving (Nx+1-2, Ny-2) = (Nx-1, Ny-2). We compute advection_u with this shape.
        
        # dudx at u-faces using central difference
        # u is at (i+1/2, j), so dudx at (i+1/2, j) uses u[i+3/2] - u[i-1/2] / (2*dx)
        # In array indices: dudx[i,j] = (u[i+2,j] - u[i,j]) / (2*dx) for i in [1:-1]
        dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)  # shape (Nx-1, Ny-2)
        
        # dudy at u-faces: need du/dy at (i+1/2, j)
        # Use central diff: dudy[i,j] = (u[i,j+1] - u[i,j-1]) / (2*dy)
        dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)  # shape (Nx-1, Ny-2)
        
        # Interpolate v to u-faces from the four surrounding v-face values.
        v_at_u = 0.25 * (
            v[:-1, 1:-2] + v[1:, 1:-2] +
            v[:-1, 2:-1] + v[1:, 2:-1]
        )
        
        # u velocity at u interior points
        u_adv = u[1:-1, 1:-1]  # (Nx-1, Ny-2)
        
        advection_u = u_adv * dudx + v_at_u * dudy

        # For v-momentum at v-faces (i, j+1/2)
        # v shape: (Nx, Ny+1), interior [1:-1, 1:-1] -> (Nx-2, Ny-1)
        
        # dvdx at v interior
        dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)  # (Nx-2, Ny-1)
        # dvdy at v interior
        dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)  # (Nx-2, Ny-1)
        
        # Interpolate u to v-faces from the four surrounding u-face values.
        u_at_v = 0.25 * (
            u[1:-2, :-1] + u[2:-1, :-1] +
            u[1:-2, 1:] + u[2:-1, 1:]
        )
        
        v_adv = v[1:-1, 1:-1]  # (Nx-2, Ny-1)
        advection_v = u_at_v * dvdx + v_adv * dvdy

        return advection_u, advection_v

    def _laplacian(self, u, dx, dy):
        """Compute Laplacian using central differences with boundary conditions."""
        Nx, Ny = self.grid.Nx, self.grid.Ny
        lap = np.zeros_like(u)
        
        # Interior points: standard 5-point stencil
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
        return lap

    def step(self):
        """Advance one time step."""
        u, v = self.u, self.v
        dx, dy, dt = self.grid.dx, self.grid.dy, self.dt
        nu = self.nu

        # --- Predictor: momentum without pressure ---
        u_star = u.copy()
        v_star = v.copy()

        # Advection
        d_u, d_v = self._advection(u, v, dx, dy)

        # Laplacian
        lap_u = self._laplacian(u, dx, dy)
        lap_v = self._laplacian(v, dx, dy)

        # Interior update for u (at faces i+1/2, j)
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            -d_u + nu * lap_u[1:-1, 1:-1]
        )
        # Interior update for v (at faces i, j+1/2)
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            -d_v + nu * lap_v[1:-1, 1:-1]
        )
        
        # Apply boundary conditions to predictor velocities for consistent advection in next iteration
        # This ensures boundary conditions are enforced at the same time level for the divergence calculation
        self.bc.apply(u_star, v_star)

        # --- Poisson: pressure correction ---
        # Compute divergence at cell centers from face velocities
        # u_star shape: (Nx+1, Ny), v_star shape: (Nx, Ny+1)
        # Divergence at cell centers: du/dx + dv/dy
        div = np.zeros(self.grid.shape_p)  # (Nx, Ny)
        # du/dx at cell centers using u at faces
        div[:, :] = (
            (u_star[1:, :] - u_star[:-1, :]) / dx +  # du/dx
            (v_star[:, 1:] - v_star[:, :-1]) / dy    # dv/dy
        )

        # Pressure Poisson equation: nabla^2 p = div / dt.
        # Pure Neumann pressure BCs need a compatible zero-mean RHS.
        rhs = div / dt
        rhs -= np.mean(rhs)
        
        rhs_flat = (-rhs).ravel()
        rhs_flat[0] = 0.0
        p_flat, info = cg(self.pressure_matrix, rhs_flat, x0=self.p.ravel())
        if info != 0:
            raise RuntimeError(f"Pressure solve failed to converge (info={info})")
        self.p = p_flat.reshape(self.grid.shape_p)
        self.p -= np.mean(self.p)

        # --- Corrector: apply pressure gradient ---
        # Pressure gradient at face locations
        # dp/dx at u-faces: p[i+1,j] - p[i,j], shape (Nx-1, Ny)
        grad_p_x = (self.p[1:, :] - self.p[:-1, :]) / dx
        # dp/dy at v-faces: p[i,j+1] - p[i,j], shape (Nx, Ny-1)
        grad_p_y = (self.p[:, 1:] - self.p[:, :-1]) / dy

        # Update interior u and v
        # u[1:-1, 1:-1] has shape (Nx-1, Ny-2) = (9, 8)
        # grad_p_x is (Nx, Ny) = (10, 10), wait no: p is (10,10), so grad_p_x is (9, 10)
        # grad_p_x[:, 1:-1] -> (9, 8) ✓
        u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x[:, 1:-1]
        
        # v[1:-1, 1:-1] has shape (Nx-2, Ny-1) = (8, 9)
        # grad_p_y is (Nx, Ny-1) = (10, 9)
        # grad_p_y[1:-1, :] -> (8, 9) ✓
        v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y[1:-1, :]

        # Re-apply boundaries
        self.bc.apply(u, v)

    def solve(self, steps: int, verbose: bool = True):
        """Run the simulation."""
        for i in range(steps):
            self.step()
            if verbose and i % 100 == 0:
                div = self.divergence(interior_only=True)
                print(f"Step {i}: max |∇·u| = {div:.6e}")

    def divergence(self, interior_only: bool = False):
        """Check mass conservation. Should be ~0."""
        dx, dy = self.grid.dx, self.grid.dy
        # Divergence at cell centers: du/dx + dv/dy
        # u is at x-faces: shape (Nx+1, Ny), so du/dx uses u[i+1,j] - u[i,j] / dx
        # v is at y-faces: shape (Nx, Ny+1), so dv/dy uses v[i,j+1] - v[i,j] / dy
        div = (
            (self.u[1:, :] - self.u[:-1, :]) / dx +  # du/dx at cell centers
            (self.v[:, 1:] - self.v[:, :-1]) / dy    # dv/dy at cell centers
        )
        if interior_only and div.shape[0] > 2 and div.shape[1] > 2:
            div = div[1:-1, 1:-1]
        return np.max(np.abs(div))
