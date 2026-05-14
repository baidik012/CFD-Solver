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

        Nx, Ny = grid.Nx, grid.Ny
        self.u = np.zeros((Nx, Ny))
        self.v = np.zeros((Nx, Ny))
        self.p = np.zeros((Nx, Ny))

        bc.apply(self.u, self.v)

    def _advection(self, u, v, dx, dy):
        """Compute advection terms (u*du/dx + v*du/dy)."""
        Nx, Ny = self.grid.Nx, self.grid.Ny

        # Use centered differences for advection
        # du^2/dx
        u_sq = u**2
        d_u = (u_sq[1:-1, 2:] - u_sq[1:-1, :-2]) / (2 * dx)

        # dv^2/dy
        v_sq = v**2
        d_v = (v_sq[2:, 1:-1] - v_sq[:-2, 1:-1]) / (2 * dy)

        return d_u, d_v

    def _laplacian(self, u, dx, dy):
        """Compute Laplacian using central differences."""
        lap = (
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 4*u
        ) / (dx**2)
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

        # Interior update
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            -d_u - d_v + nu * lap_u[1:-1, 1:-1]
        )
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            -d_u - d_v + nu * lap_v[1:-1, 1:-1]
        )

        # --- Poisson: pressure correction ---
        div_u = (
            (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx) +
            (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy)
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
        grad_p_x = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2 * dx)
        grad_p_y = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2 * dy)

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
        dv_dx = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * dx)
        dv_dy = (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2 * dy)
        return np.max(np.abs(dv_dx + dv_dy))