"""Chorin projection method solver."""

import numpy as np
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

        bc.apply(self.u, self.v)

    def _advection(self, u, v, dx, dy):
        """Compute advection terms (u*du/dx + v*du/dy) for u-momentum and 
        (u*dv/dx + v*dv/dy) for v-momentum."""
        Nx, Ny = self.grid.Nx, self.grid.Ny

        # Compute derivatives at cell centers [1:-1, 1:-1]
        # For u-momentum: u*du/dx + v*du/dy
        dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)  # ∂u/∂x
        dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)  # ∂u/∂y
        advection_u = u[1:-1, 1:-1] * dudx + v[1:-1, 1:-1] * dudy

        # For v-momentum: u*dv/dx + v*dv/dy
        dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)  # ∂v/∂x
        dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)  # ∂v/∂y
        advection_v = u[1:-1, 1:-1] * dvdx + v[1:-1, 1:-1] * dvdy

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

        # --- Poisson: pressure correction ---
        # Compute divergence at cell centers from face velocities
        div_u = (
            (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2 * dx) +  # du/dx at cell centers
            (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2 * dy)    # dv/dy at cell centers
        )

        # Simple Jacobi iteration for pressure
        for _ in range(50):
            p_new = self.p.copy()
            p_new[1:-1, 1:-1] = (
                (self.p[2:, 1:-1] + self.p[:-2, 1:-1]) / dx**2 +
                (self.p[1:-1, 2:] + self.p[1:-1, :-2]) / dy**2 -
                div_u / dt
            ) / (2 / dx**2 + 2 / dy**2)
            self.p = p_new

        # --- Corrector: apply pressure gradient ---
        # Pressure gradient at face locations
        grad_p_x = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2 * dx)  # dp/dx at u-faces
        grad_p_y = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2 * dy)  # dp/dy at v-faces

        u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
        v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y

        # Re-apply boundaries
        self.bc.apply(u, v)

    def solve(self, steps: int, verbose: bool = True):
        """Run the simulation."""
        for i in range(steps):
            self.step()
            if verbose and i % 100 == 0:
                div = self.divergence()
                print(f"Step {i}: max |∇·u| = {div:.6e}")

    def divergence(self):
        """Check mass conservation. Should be ~0."""
        dx, dy = self.grid.dx, self.grid.dy
        # Divergence at cell centers: du/dx + dv/dy
        # u is at x-faces: shape (Nx+1, Ny), so du/dx uses u[i+1,j] - u[i-1,j]
        dv_dx = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * dx)  # du/dx at cell centers
        # v is at y-faces: shape (Nx, Ny+1), so dv/dy uses v[i,j+1] - v[i,j-1]
        dv_dy = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * dy)  # dv/dy at cell centers
        return np.max(np.abs(dv_dx + dv_dy))