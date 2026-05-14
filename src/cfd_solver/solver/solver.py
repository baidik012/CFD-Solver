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

    def step(self):
        """Advance one time step."""
        u, v = self.u, self.v
        p = self.p
        dx, dy, dt = self.grid.dx, self.grid.dy, self.dt
        nu = self.nu

        # --- Predictor: momentum without pressure ---
        u_star = u.copy()
        v_star = v.copy()

        # Advection (explicit, upwind)
        # u * du/dx
        ua = np.roll(u, 1, axis=1)  # u[i-1] for upwind
        ub = u                       # u[i] for downwind
        u_advec_x = np.where(u > 0,
                             u * (u - ua) / dx,
                             u * (ub - np.roll(u, -1, axis=1)) / dx)

        va = np.roll(v, 1, axis=0)   # v[j-1] for upwind
        vb = v                       # v[j] for downwind
        u_advec_y = np.where(v > 0,
                             v * (u - va) / dy,
                             v * (ub - np.roll(u, -1, axis=0)) / dy)

        # du^2/dx + dv^2/dy (full form)
        u_sq = u[1:-1, 1:-1]**2
        v_sq = v[1:-1, 1:-1]**2

        d_u_sq = np.gradient(u_sq, dx, axis=1)
        d_v_sq = np.gradient(v_sq, dy, axis=0)

        # Laplacian
        lap_u = (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
                 np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 4*u) / (dx**2)
        lap_v = (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) +
                 np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 4*v) / (dy**2)

        # Interior update
        u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*(
            -d_u_sq[1:-1, 1:-1] - d_v_sq[1:-1, 1:-1] + nu*lap_u[1:-1, 1:-1]
        )
        v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt*(
            -d_u_sq[1:-1, 1:-1] - d_v_sq[1:-1, 1:-1] + nu*lap_v[1:-1, 1:-1]
        )

        # --- Poisson: pressure correction ---
        div_u = (np.gradient(u_star, dx, axis=1) +
                 np.gradient(v_star, dy, axis=0))

        # Simple Jacobi iteration
        p_new = p.copy()
        for _ in range(50):
            p_interior = p[1:-1, 1:-1]
            p_poisson = (
                (p[2:, 1:-1] + p[:-2, 1:-1]) / dx**2 +
                (p[1:-1, 2:] + p[1:-1, :-2]) / dy**2 -
                div_u[1:-1, 1:-1] / dt
            ) / (2 / dx**2 + 2 / dy**2)
            p[1:-1, 1:-1] = p_poisson

        # --- Corrector: apply pressure gradient ---
        grad_p_x = np.gradient(p, dx, axis=1)
        grad_p_y = np.gradient(p, dy, axis=0)

        u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x[1:-1, 1:-1]
        v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y[1:-1, 1:-1]

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
        dv_dx = np.gradient(self.u, self.grid.dx, axis=1)
        dv_dy = np.gradient(self.v, self.grid.dy, axis=0)
        return np.max(np.abs(dv_dx + dv_dy))