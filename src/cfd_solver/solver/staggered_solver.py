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
        """Build pressure Poisson matrix with Neumann BCs."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        N = Nx * Ny
        A = lil_matrix((N, N))

        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                
                # Start with standard 5-point Laplacian coefficients
                diag = 2.0 / dx2 + 2.0 / dy2
                
                # Left neighbor (i-1, j)
                if i > 0:
                    A[idx, idx - Ny] = -1.0 / dx2
                else:
                    # Neumann BC at left: dp/dx = 0 => ghost cell p[-1,j] = p[1,j]
                    # This modifies the diagonal: add 1/dx2 instead of connecting to left
                    diag += 1.0 / dx2
                
                # Right neighbor (i+1, j)
                if i < Nx - 1:
                    A[idx, idx + Ny] = -1.0 / dx2
                else:
                    # Neumann BC at right: dp/dx = 0 => ghost cell p[Nx,j] = p[Nx-2,j]
                    # This modifies the diagonal: add 1/dx2 instead of connecting to right
                    diag += 1.0 / dx2
                
                # Bottom neighbor (i, j-1)
                if j > 0:
                    A[idx, idx - 1] = -1.0 / dy2
                else:
                    # Neumann BC at bottom: dp/dy = 0 => ghost cell p[i,-1] = p[i,1]
                    # This modifies the diagonal: add 1/dy2 instead of connecting to bottom
                    diag += 1.0 / dy2
                
                # Top neighbor (i, j+1)
                if j < Ny - 1:
                    A[idx, idx + 1] = -1.0 / dy2
                else:
                    # Neumann BC at top: dp/dy = 0 => ghost cell p[i,Ny] = p[i,Ny-2]
                    # This modifies the diagonal: add 1/dy2 instead of connecting to top
                    diag += 1.0 / dy2
                
                A[idx, idx] = diag

        # Fix one point to remove nullspace (constant pressure = no physics)
        # This makes the matrix non-singular so CG can converge
        A[0, :] = 0
        A[0, 0] = 1.0

        self.A = A.tocsr()

    def _apply_bc(self):
        """Apply boundary conditions."""
        Nx, Ny = self.Nx, self.Ny

        # Top and bottom walls (u-velocity on vertical faces at j=0 and j=Ny-1)
        # Only set interior vertical faces, not corners which are handled by side walls
        self.u[1:Nx, 0] = self.u_bottom  # bottom wall, exclude left corner
        self.u[1:Nx, Ny - 1] = self.u_top  # top wall, exclude left corner
        
        # Left and right walls (u-velocity on vertical faces at i=0 and i=Nx)
        self.u[0, :] = self.u_left   # left wall (all j including corners)
        self.u[Nx, :] = self.u_right  # right wall (all j including corners)

        # v-velocity at all walls (v lives on horizontal faces)
        self.v[:, 0] = 0.0      # bottom wall (j=0)
        self.v[:, Ny] = 0.0     # top wall (j=Ny)
        self.v[0, :] = 0.0      # left wall (i=0)
        self.v[Nx - 1, :] = 0.0  # right wall (i=Nx-1)

        # u-velocity: apply in correct order for lid-driven cavity
        # Corners should follow lid velocity (top wall) for mass conservation
        self.u[:, 0] = self.u_bottom  # bottom wall
        self.u[0, :] = self.u_left    # left wall
        self.u[Nx, :] = self.u_right  # right wall
        # Top wall (lid) last - overwrites corners with lid velocity
        self.u[:, Ny - 1] = self.u_top

        # Fix corners: for lid-driven cavity, corners move with lid
        # This prevents artificial mass sources at corners
        self.u[0, Ny - 1] = self.u_top     # left-top corner
        self.u[Nx, Ny - 1] = self.u_top    # right-top corner

    def _advection(self, u, v):
        """Compute advection term - placeholder returning zeros."""
        # TODO: Implement proper advection
        return np.zeros_like(u), np.zeros_like(v)

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

        # Predictor: momentum without pressure
        u_star = u.copy()
        v_star = v.copy()

        # Advection (simplified first-order upwind)
        adv_u, adv_v = self._advection(u, v)
        u_star -= dt * adv_u
        v_star -= dt * adv_v

        # Viscous diffusion (centered second differences)
        # u: shape (Nx+1, Ny) - apply diffusion to interior faces i=1..Nx-1
        lap_u = np.zeros_like(u)
        # second derivative in x (interior faces i=1..Nx-1)
        lap_u_x = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dx**2
        # second derivative in y: only for interior j=1..Ny-2
        lap_u_y = np.zeros_like(lap_u_x)
        if u.shape[1] > 2:
            lap_u_y[:, 1:-1] = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        lap_u[1:-1, :] = lap_u_x + lap_u_y
        u_star[1:-1, :] += nu * dt * lap_u[1:-1, :]

        # v: shape (Nx, Ny+1) - apply diffusion to interior faces j=1..Ny-1 and i=1..Nx-2
        lap_v = np.zeros_like(v)
        # interior in x is 1:-1 (i=1..Nx-2), interior in y is 1:-1 (j=1..Ny-1)
        if v.shape[0] > 2 and v.shape[1] > 2:
            lap_v[1:-1, 1:-1] = (
                (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2 +
                (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
            )
            v_star[1:-1, 1:-1] += nu * dt * lap_v[1:-1, 1:-1]

        # Divergence at cell centers (shape: Nx x Ny)
        div = (u_star[1:, :] - u_star[:-1, :]) / dx + (v_star[:, 1:] - v_star[:, :-1]) / dy

        # Pressure Poisson RHS (sign convention)
        rhs = -div / dt

        # Modify RHS to account for pressure fix at point 0
        rhs_flat = rhs.flatten()
        rhs_flat[0] = 0.0  # p[0] = 0 (Dirichlet condition)

        # Solve pressure Poisson using CG
        p_flat, info = cg(self.A, rhs_flat)
        if info != 0:
            raise RuntimeError(f"CG solver failed to converge (info={info})")
        self.p = p_flat.reshape((Nx, Ny))
        # Enforce zero-mean pressure to remove numerical drift
        self.p -= np.mean(self.p)

        # Corrector: compute pressure gradients on faces
        grad_p_x = (self.p[1:, :] - self.p[:-1, :]) / dx   # shape (Nx-1, Ny)
        grad_p_y = (self.p[:, 1:] - self.p[:, :-1]) / dy   # shape (Nx, Ny-1)

        # Update face velocities on interior faces
        self.u[1:-1, :] = u_star[1:-1, :] - dt * grad_p_x
        self.v[:, 1:-1] = v_star[:, 1:-1] - dt * grad_p_y

        # Re-apply boundary conditions
        self._apply_bc()

    def divergence_norm(self):
        # compute divergence at cell centers using face velocities (shape: Nx x Ny)
        div = (self.u[1:, :] - self.u[:-1, :]) / self.dx + (self.v[:, 1:] - self.v[:, :-1]) / self.dy
        return np.sqrt(np.mean(div**2))

    def max_divergence(self):
        # compute divergence at cell centers using face velocities (shape: Nx x Ny)
        div = (self.u[1:, :] - self.u[:-1, :]) / self.dx + (self.v[:, 1:] - self.v[:, :-1]) / self.dy
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
