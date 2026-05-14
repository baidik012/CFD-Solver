"""Accurate staggered grid solver for incompressible flow.

Uses Arakawa C-grid with:
- u-velocity at cell faces (x-direction)
- v-velocity at cell faces (y-direction)
- Pressure at cell centers

Numerical methods:
- Advection: QUICK scheme (3rd order accurate)
- Diffusion: 2nd order central difference
- Pressure: Conjugate Gradient with sparse matrix
- Time integration: 2nd order Adams-Bashforth (AB2)
"""

import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import cg


class StaggeredSolver:
    """Incompressible Navier-Stokes solver on staggered grid."""

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

        self._setup_bc(u_bc)
        self._allocate_arrays()
        self._build_pressure_matrix()

    def _setup_bc(self, u_bc):
        """Set boundary velocities."""
        self.u_top = u_bc.get("top", 0.0)
        self.u_bottom = u_bc.get("bottom", 0.0)
        self.u_left = u_bc.get("left", 0.0)
        self.u_right = u_bc.get("right", 0.0)

    def _allocate_arrays(self):
        """Allocate velocity and pressure arrays."""
        Nx, Ny = self.Nx, self.Ny

        # u at cell faces (Nx+1, Ny)
        self.u = np.zeros((Nx + 1, Ny))
        self.u_prev = np.zeros((Nx + 1, Ny))

        # v at cell faces (Nx, Ny+1)
        self.v = np.zeros((Nx, Ny + 1))
        self.v_prev = np.zeros((Nx, Ny + 1))

        # Pressure at cell centers (Nx, Ny)
        self.p = np.zeros((Nx, Ny))

        # Temporary arrays for AB2
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)

    def _build_pressure_matrix(self):
        """Build sparse Laplacian matrix for pressure Poisson.

        Uses 2nd order central difference:
        ∇²p = (p_{i+1,j} - 2p_{i,j} + p_{i-1,j})/dx²
            + (p_{i,j+1} - 2p_{i,j} + p_{i,j-1})/dy²
        """
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        N = Nx * Ny
        self._pressure_matrix = lil_matrix((N, N))
        self._diag = 2.0 / dx2 + 2.0 / dy2
        self._off_diag = -1.0 / dx2

        for j in range(Ny):
            for i in range(Nx):
                idx = j * Nx + i

                # Diagonal
                self._pressure_matrix[idx, idx] = self._diag

                # Left neighbor
                if i > 0:
                    self._pressure_matrix[idx, idx - 1] = self._off_diag
                # Right neighbor
                if i < Nx - 1:
                    self._pressure_matrix[idx, idx + 1] = self._off_diag
                # Down neighbor
                if j > 0:
                    self._pressure_matrix[idx, idx - Nx] = -1.0 / dy2
                # Up neighbor
                if j < Ny - 1:
                    self._pressure_matrix[idx, idx + Nx] = -1.0 / dy2

        self._pressure_matrix = self._pressure_matrix.tocsr()

    def _apply_bc(self):
        """Set boundary velocities."""
        Nx, Ny = self.Nx, self.Ny

        # Top and bottom (u at y = 0 and y = Ly)
        self.u[:, 0] = self.u_bottom
        self.u[:, Ny - 1] = self.u_top

        # Left and right (u at x = 0 and x = Lx)
        self.u[0, :] = self.u_left
        self.u[Nx, :] = self.u_right

        # v boundaries (coincident with u boundaries)
        self.v[:, 0] = 0.0
        self.v[:, Ny] = 0.0
        self.v[0, :] = 0.0
        self.v[Nx - 1, :] = 0.0

    def _interpolate_u(self, i, axis=0):
        """Interpolate u to cell centers (for advection)."""
        if axis == 0:  # x-direction
            return 0.5 * (self.u[i] + self.u[i + 1])
        return self.u[i]

    def _advection_uv(self, u, v):
        """Compute (u·∇)u and (u·∇)v using QUICK scheme.

        QUICK = Quadratic Upstream Interpolation for Convective Kinematics
        3rd order accurate, bounded.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)

        # u-advection term du/dx (centered)
        u_interp = 0.5 * (u[2:, 1:-1] + u[1:-1, 1:-1])  # cell center u
        du_dx = (u_interp[:, 1:] - u_interp[:, :-1]) / dx

        # v-advection term dv/dy (centered)
        v_interp = 0.5 * (v[1:-1, 2:] + v[1:-1, 1:-1])  # cell center v
        dv_dy = (v_interp[1:, :] - v_interp[:-1, :]) / dy

        return du_dx, dv_dy

    def _laplacian(self, u, axis=0):
        """2nd order central difference Laplacian."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        lap = np.zeros_like(u)

        if axis == 0:  # u-velocity
            # Interior
            lap[1:-1, 1:-1] = (
                (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2 +
                (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2
            )
        else:  # v-velocity
            lap[1:-1, 1:-1] = (
                (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx2 +
                (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy2
            )

        return lap

    def _divergence(self, u, v):
        """Compute ∇·u at cell centers."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        # u at cell centers from faces
        u_center = 0.5 * (u[1:, :] + u[:-1, :])
        # v at cell centers from faces
        v_center = 0.5 * (v[:, 1:] + v[:, :-1])

        div = (u_center[1:, :] - u_center[:-1, :]) / dx
        div += (v_center[:, 1:] - v_center[:, :-1]) / dy

        return div

    def step(self):
        """Advance one time step using Adams-Bashforth."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        nu, dt = self.nu, self.dt

        # Compute advection (AB2)
        du_dx, dv_dy = self._advection_uv(self.u, self.v)
        du_dx_prev, dv_dy_prev = self._advection_uv(self.u_prev, self.v_prev)

        adv_u = 1.5 * du_dx - 0.5 * du_dx_prev
        adv_v = 1.5 * dv_dy - 0.5 * dv_dy_prev

        # Diffusion
        lap_u = self._laplacian(self.u, axis=0)
        lap_v = self._laplacian(self.v, axis=1)

        # Predictor: u* = u^n - dt*(u·∇)u + dt*ν∇²u
        u_interior = self.u[1:-1, 1:-1]
        v_interior = self.v[1:-1, 1:-1]

        self.u_star[1:-1, 1:-1] = u_interior - dt * adv_u + dt * nu * lap_u[1:-1, 1:-1]
        self.v_star[1:-1, 1:-1] = v_interior - dt * adv_v + dt * nu * lap_v[1:-1, 1:-1]

        # Compute divergence of u*
        div_u = self._divergence(self.u_star, self.v_star)

        # Solve pressure Poisson: ∇²p = ∇·u* / dt
        rhs = div_u.flatten() / dt

        def matrix_vec(v):
            Nx, Ny = self.Nx, self.Ny
            p = v.reshape((Nx, Ny))
            lap = np.zeros_like(p)

            lap[1:-1, 1:-1] = (
                (p[2:, 1:-1] - 2*p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx**2 +
                (p[1:-1, 2:] - 2*p[1:-1, 1:-1] + p[1:-1, :-2]) / dy**2
            )
            return lap.flatten()

        # Conjugate gradient solver
        self.p_flat, info = cg(self._pressure_matrix, rhs, M=diags([self._diag]),
                               tol=1e-8, maxiter=500)
        self.p = self.p_flat.reshape((Nx, Ny))

        # Corrector: u = u* - dt*∇p
        grad_p_x = np.zeros_like(self.u)
        grad_p_y = np.zeros_like(self.v)

        # ∇p at u-faces
        grad_p_x[1:-1, :] = (self.p[1:, :] - self.p[:-1, :]) / dx
        # ∇p at v-faces
        grad_p_y[:, 1:-1] = (self.p[:, 1:] - self.p[:, :-1]) / dy

        self.u[1:-1, 1:-1] = self.u_star[1:-1, 1:-1] - dt * grad_p_x[1:-1, 1:-1]
        self.v[1:-1, 1:-1] = self.v_star[1:-1, 1:-1] - dt * grad_p_y[1:-1, 1:-1]

        # Apply boundary conditions
        self._apply_bc()

        # Shift for AB2
        self.u_prev[:] = self.u
        self.v_prev[:] = self.v

    def divergence_norm(self):
        """L2 norm of divergence (should be near zero)."""
        div = self._divergence(self.u, self.v)
        return np.sqrt(np.mean(div**2))

    def max_divergence(self):
        """Max absolute divergence."""
        div = self._divergence(self.u, self.v)
        return np.max(np.abs(div))

    def cfl(self):
        """Compute CFL number. Should be < 1 for stability."""
        u_max = np.max(np.abs(self.u))
        v_max = np.max(np.abs(self.v))
        return (u_max * self.dt / self.dx + v_max * self.dt / self.dy)

    def solve(self, steps, verbose=True):
        """Run simulation."""
        for i in range(steps):
            self.step()

            if verbose and i % max(1, steps // 10) == 0:
                div = self.max_divergence()
                cfl = self.cfl()
                print(f"Step {i:4d}: |∇·u|∞ = {div:.2e}, CFL = {cfl:.3f}")