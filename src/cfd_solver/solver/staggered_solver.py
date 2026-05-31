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
                    # Neumann BC at left: dp/dx = 0 => ghost p[-1,j] = p[0,j]
                    # (p[-1] - 2p[0] + p[1])/dx² = (p[0] - 2p[0] + p[1])/dx² = (p[1] - p[0])/dx²
                    # In A@p form: -(p[1]-p[0])/dx² = (p[0] - p[1])/dx², so diag -= 1/dx²
                    diag -= 1.0 / dx2
                
                # Right neighbor (i+1, j)
                if i < Nx - 1:
                    A[idx, idx + Ny] = -1.0 / dx2
                else:
                    # Neumann BC at right: dp/dx = 0 => ghost p[Nx,j] = p[Nx-1,j]
                    diag -= 1.0 / dx2
                
                # Bottom neighbor (i, j-1)
                if j > 0:
                    A[idx, idx - 1] = -1.0 / dy2
                else:
                    # Neumann BC at bottom: dp/dy = 0 => ghost p[i,-1] = p[i,0]
                    diag -= 1.0 / dy2
                
                # Top neighbor (i, j+1)
                if j < Ny - 1:
                    A[idx, idx + 1] = -1.0 / dy2
                else:
                    # Neumann BC at top: dp/dy = 0 => ghost p[i,Ny] = p[i,Ny-1]
                    diag -= 1.0 / dy2
                
                A[idx, idx] = diag

        # Fix one point to remove nullspace (constant pressure = no physics)
        # This makes the matrix non-singular so CG can converge
        A[0, :] = 0
        A[0, 0] = 1.0

        self.A = A.tocsr()

    def _apply_bc(self):
        """Apply boundary conditions for lid-driven cavity.

        Grid layout:
          u: shape (Nx+1, Ny) — u[i,j] on vertical face at x=i*dx, y=(j+0.5)*dy
          v: shape (Nx, Ny+1) — v[i,j] on horizontal face at x=(i+0.5)*dx, y=j*dy
          p: shape (Nx, Ny)   — p[i,j] at cell center (i+0.5)*dx, (j+0.5)*dy

        u is directly on left/right walls (i=0, i=Nx) and on top/bottom walls (j=0, j=Ny-1).
        v is directly on top/bottom walls (j=0, j=Ny) but NOT on left/right walls.
        """
        Nx, Ny = self.Nx, self.Ny

        # u-velocity: all walls (u lives on vertical faces)
        self.u[0, :] = self.u_left       # left wall (x=0)
        self.u[Nx, :] = self.u_right     # right wall (x=Lx)
        self.u[:, 0] = self.u_bottom     # bottom wall (y=0)
        self.u[:, Ny - 1] = self.u_top   # top wall (y=Ly) — lid, set last so corners = lid

        # v-velocity: top and bottom walls (v lives on horizontal faces)
        # v is NOT defined on left/right walls (those are at cell-center x positions)
        self.v[:, 0] = 0.0               # bottom wall (y=0)
        self.v[:, Ny] = 0.0              # top wall (y=Ly)

    def _advection(self, u, v):
        """First-order upwind advection on a staggered grid.

        Returns adv_u (Nx+1, Ny) and adv_v (Nx, Ny+1) at interior faces.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)

        # --- u-advection at interior u-faces (i=1..Nx-1, j=1..Ny-2) ---
        ui = slice(1, Nx)     # i = 1 .. Nx-1
        uj = slice(1, Ny-1)   # j = 1 .. Ny-2

        u_ij = u[ui, uj]

        # u * ∂u/∂x (upwind)
        du_dx = np.where(
            u_ij > 0,
            (u_ij - u[ui.start-1:ui.stop-1, uj]) / dx,
            (u[ui.start+1:ui.stop+1, uj] - u_ij) / dx,
        )

        # v interpolated to u-face: average of 4 surrounding v points
        v_at_u = 0.25 * (
            v[ui.start-1:ui.stop-1, uj]                     # v[i-1, j]
            + v[ui.start-1:ui.stop-1, uj.start+1:uj.stop+1]  # v[i-1, j+1]
            + v[ui.start:ui.stop, uj]                       # v[i, j]
            + v[ui.start:ui.stop, uj.start+1:uj.stop+1]     # v[i, j+1]
        )

        # v * ∂u/∂y (upwind)
        du_dy = np.where(
            v_at_u > 0,
            (u_ij - u[ui, uj.start-1:uj.stop-1]) / dy,
            (u[ui, uj.start+1:uj.stop+1] - u_ij) / dy,
        )

        adv_u[ui, uj] = u_ij * du_dx + v_at_u * du_dy

        # --- v-advection at interior v-faces (i=1..Nx-2, j=1..Ny-1) ---
        vi = slice(1, Nx-1)  # i = 1 .. Nx-2
        vj = slice(1, Ny)    # j = 1 .. Ny-1

        v_ij = v[vi, vj]

        # u interpolated to v-face: average of 4 surrounding u points
        u_at_v = 0.25 * (
            u[vi.start:vi.stop, vj.start-1:vj.stop-1]         # u[i, j-1]
            + u[vi.start+1:vi.stop+1, vj.start-1:vj.stop-1]  # u[i+1, j-1]
            + u[vi.start:vi.stop, vj.start:vj.stop]           # u[i, j]
            + u[vi.start+1:vi.stop+1, vj.start:vj.stop]       # u[i+1, j]
        )

        # u * ∂v/∂x (upwind)
        dv_dx = np.where(
            u_at_v > 0,
            (v_ij - v[vi.start-1:vi.stop-1, vj]) / dx,
            (v[vi.start+1:vi.stop+1, vj] - v_ij) / dx,
        )

        # v * ∂v/∂y (upwind)
        dv_dy = np.where(
            v_ij > 0,
            (v_ij - v[vi, vj.start-1:vj.stop-1]) / dy,
            (v[vi, vj.start+1:vj.stop+1] - v_ij) / dy,
        )

        adv_v[vi, vj] = u_at_v * dv_dx + v_ij * dv_dy

        return adv_u, adv_v

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

    def _divergence(self):
        return ((self.u[1:, :] - self.u[:-1, :]) / self.dx +
                (self.v[:, 1:] - self.v[:, :-1]) / self.dy)

    def divergence_norm(self):
        div = self._divergence()
        return np.sqrt(np.mean(div**2))

    def max_divergence(self):
        return np.max(np.abs(self._divergence()))

    def cfl(self):
        return (np.max(np.abs(self.u)) * self.dt / self.dx +
                np.max(np.abs(self.v)) * self.dt / self.dy)

    def solve(self, steps, verbose=True):
        for i in range(steps):
            self.step()
            if verbose and i % max(1, steps // 10) == 0:
                print(f"Step {i:4d}: |∇·u|∞ = {self.max_divergence():.2e}, CFL = {self.cfl():.3f}")
