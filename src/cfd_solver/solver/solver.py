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
        
        # Interpolate v to u-faces: v is at (i, j+1/2), we need it at (i+1/2, j)
        # Strategy: average v in x-direction at interior y-points
        # v shape: (Nx, Ny+1), select interior y: v[:, 1:-1] -> (Nx, Ny-1)
        # Average in x: (v[1:, 1:-1] + v[:-1, 1:-1]) / 2 -> (Nx-1, Ny-1)
        # But we only need interior u points at [1:-1, 1:-1] -> (Nx-1, Ny-2)
        # So we trim y: [:, :-1] -> (Nx-1, Ny-2)
        v_at_u = (v[1:, 1:-1] + v[:-1, 1:-1]) / 2.0  # Average in x: (Nx-1, Ny-1)
        v_at_u = v_at_u[:, :-1]  # Select interior y points: (Nx-1, Ny-2)
        
        # u velocity at u interior points
        u_adv = u[1:-1, 1:-1]  # (Nx-1, Ny-2)
        
        advection_u = u_adv * dudx + v_at_u * dudy

        # For v-momentum at v-faces (i, j+1/2)
        # v shape: (Nx, Ny+1), interior [1:-1, 1:-1] -> (Nx-2, Ny-1)
        
        # dvdx at v interior
        dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)  # (Nx-2, Ny-1)
        # dvdy at v interior
        dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)  # (Nx-2, Ny-1)
        
        # Interpolate u to v-faces: u is at (i+1/2, j), we need it at (i, j+1/2)
        # Strategy: average u in y-direction at interior x-points
        # u shape: (Nx+1, Ny), select interior x: u[1:-1, :] -> (Nx-1, Ny)
        # Average in y: (u[1:-1, 1:] + u[1:-1, :-1]) / 2 -> (Nx-1, Ny-1)
        # But we only need interior v points at [1:-1, 1:-1] -> (Nx-2, Ny-1)
        # So we trim x: [:-1, :] -> (Nx-2, Ny-1)
        u_at_v = (u[1:-1, 1:] + u[1:-1, :-1]) / 2.0  # Average in y: (Nx-1, Ny-1)
        u_at_v = u_at_v[:-1, :]  # Select interior x points: (Nx-2, Ny-1)
        
        # But wait, dvdx is (6, 9), not (8, 9). Need to fix dvdx computation
        
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

        # Pressure Poisson equation: nabla^2 p = rho * div / dt
        # Using standard 5-point stencil with proper RHS scaling
        # The RHS should be div/dt at ALL cell centers, not just interior
        rhs = div / dt
        
        # SOR (Successive Over-Relaxation) for pressure with better convergence
        # Neumann boundary conditions: dp/dn = 0 at all walls
        omega = 1.5  # over-relaxation parameter (1 < omega < 2)
        for _ in range(300):
            p_old = self.p.copy()
            # Update interior points using 5-point stencil
            p_new = self.p.copy()
            p_new[1:-1, 1:-1] = (
                (self.p[2:, 1:-1] + self.p[:-2, 1:-1]) / dx**2 +
                (self.p[1:-1, 2:] + self.p[1:-1, :-2]) / dy**2 -
                rhs[1:-1, 1:-1]
            ) / (2 / dx**2 + 2 / dy**2)
            
            # Apply Neumann boundary conditions (zero normal derivative)
            # Extrapolate pressure at boundaries
            p_new[0, :] = p_new[1, :]    # left boundary
            p_new[-1, :] = p_new[-2, :]  # right boundary
            p_new[:, 0] = p_new[:, 1]    # bottom boundary
            p_new[:, -1] = p_new[:, -2]  # top boundary
            
            # Apply SOR relaxation
            self.p = omega * p_new + (1 - omega) * self.p
            
            # Check convergence: if pressure change is below tolerance, early termination
            residual = np.max(np.abs(self.p - p_old))
            if residual < 1e-6:
                break

        # --- Corrector: apply pressure gradient ---
        # Pressure gradient at face locations
        # dp/dx at u-faces: use p[i+1,j] - p[i,j], shape (Nx, Ny) -> (9, 10) for Nx=10,Ny=10
        grad_p_x = (self.p[1:, :] - self.p[:-1, :]) / dx  # shape (Nx, Ny)
        # dp/dy at v-faces: use p[i,j+1] - p[i,j], shape (Nx, Ny) -> (10, 9) for Nx=10,Ny=10
        grad_p_y = (self.p[:, 1:] - self.p[:, :-1]) / dy  # shape (Nx, Ny)

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
                div = self.divergence()
                print(f"Step {i}: max |∇·u| = {div:.6e}")

    def divergence(self):
        """Check mass conservation. Should be ~0."""
        dx, dy = self.grid.dx, self.grid.dy
        # Divergence at cell centers: du/dx + dv/dy
        # u is at x-faces: shape (Nx+1, Ny), so du/dx uses u[i+1,j] - u[i,j] / dx
        # v is at y-faces: shape (Nx, Ny+1), so dv/dy uses v[i,j+1] - v[i,j] / dy
        div = (
            (self.u[1:, :] - self.u[:-1, :]) / dx +  # du/dx at cell centers
            (self.v[:, 1:] - self.v[:, :-1]) / dy    # dv/dy at cell centers
        )
        return np.max(np.abs(div))