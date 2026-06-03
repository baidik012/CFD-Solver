"""Staggered-grid incompressible Navier-Stokes solver (Chorin projection)."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg


class StaggeredSolver:
    """Incompressible Navier-Stokes solver using Chorin splitting.

    Staggered grid layout (Arakawa C-grid):
      u: (Nx+1, Ny) — vertical faces at x = i*dx
      v: (Nx, Ny+1) — horizontal faces at y = j*dy
      p: (Nx, Ny)   — cell centers
    """

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

        self.u_top = u_bc.get("top", 0.0)
        self.u_bottom = u_bc.get("bottom", 0.0)
        self.u_left = u_bc.get("left", 0.0)
        self.u_right = u_bc.get("right", 0.0)

        self.u = np.zeros((Nx + 1, Ny))
        self.v = np.zeros((Nx, Ny + 1))
        self.p = np.zeros((Nx, Ny))

        self._build_pressure_matrix()

    def _build_pressure_matrix(self):
        """Build the positive-Laplacian pressure matrix with Neumann walls."""
        Nx, Ny = self.Nx, self.Ny
        dx2, dy2 = self.dx**2, self.dy**2

        N = Nx * Ny
        A = lil_matrix((N, N))

        for i in range(Nx):
            for j in range(Ny):
                idx = i * Ny + j
                diag = 2.0 / dx2 + 2.0 / dy2

                if i > 0:
                    A[idx, idx - Ny] = -1.0 / dx2
                else:
                    diag -= 1.0 / dx2  # Neumann left

                if i < Nx - 1:
                    A[idx, idx + Ny] = -1.0 / dx2
                else:
                    diag -= 1.0 / dx2  # Neumann right

                if j > 0:
                    A[idx, idx - 1] = -1.0 / dy2
                else:
                    diag -= 1.0 / dy2  # Neumann bottom

                if j < Ny - 1:
                    A[idx, idx + 1] = -1.0 / dy2
                else:
                    diag -= 1.0 / dy2  # Neumann top

                A[idx, idx] = diag

        # Pin one pressure value to remove the nullspace; zero row and
        # column to keep the matrix symmetric for CG.
        A[0, :] = 0
        A[:, 0] = 0
        A[0, 0] = 1.0

        self.A = A.tocsr()

    def _set_bc(self, u, v):
        """Set velocity boundary conditions on the given arrays.

        u (Nx+1, Ny): fixed on all walls — left/right (i=0, i=Nx),
        top/bottom (j=0, j=Ny-1).  v (Nx, Ny+1): fixed on top/bottom
        walls only (j=0, j=Ny); not defined on left/right walls.
        """
        Nx, Ny = self.Nx, self.Ny

        u[0, :] = self.u_left
        u[Nx, :] = self.u_right
        u[:, 0] = self.u_bottom
        u[:, Ny - 1] = self.u_top

        v[:, 0] = 0.0
        v[:, Ny] = 0.0

    def _apply_bc(self):
        """Set boundary conditions on self.u and self.v."""
        self._set_bc(self.u, self.v)

    def _advection(self, u, v):
        """First-order upwind advection.

        Returns adv_u (Nx+1, Ny) and adv_v (Nx, Ny+1) — non-zero only
        at interior faces.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy

        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)

        # --- u-advection at interior u-faces (i=1..Nx-1, j=1..Ny-2) ---
        ui = slice(1, Nx)
        uj = slice(1, Ny - 1)
        u_ij = u[ui, uj]

        du_dx = np.where(
            u_ij > 0,
            (u_ij - u[ui.start - 1:ui.stop - 1, uj]) / dx,
            (u[ui.start + 1:ui.stop + 1, uj] - u_ij) / dx,
        )

        v_at_u = 0.25 * (
            v[ui.start - 1:ui.stop - 1, uj]
            + v[ui.start - 1:ui.stop - 1, uj.start + 1:uj.stop + 1]
            + v[ui.start:ui.stop, uj]
            + v[ui.start:ui.stop, uj.start + 1:uj.stop + 1]
        )

        du_dy = np.where(
            v_at_u > 0,
            (u_ij - u[ui, uj.start - 1:uj.stop - 1]) / dy,
            (u[ui, uj.start + 1:uj.stop + 1] - u_ij) / dy,
        )

        adv_u[ui, uj] = u_ij * du_dx + v_at_u * du_dy

        # --- v-advection at interior v-faces (i=1..Nx-2, j=1..Ny-1) ---
        vi = slice(1, Nx - 1)
        vj = slice(1, Ny)
        v_ij = v[vi, vj]

        u_at_v = 0.25 * (
            u[vi.start:vi.stop, vj.start - 1:vj.stop - 1]
            + u[vi.start + 1:vi.stop + 1, vj.start - 1:vj.stop - 1]
            + u[vi.start:vi.stop, vj.start:vj.stop]
            + u[vi.start + 1:vi.stop + 1, vj.start:vj.stop]
        )

        dv_dx = np.where(
            u_at_v > 0,
            (v_ij - v[vi.start - 1:vi.stop - 1, vj]) / dx,
            (v[vi.start + 1:vi.stop + 1, vj] - v_ij) / dx,
        )

        dv_dy = np.where(
            v_ij > 0,
            (v_ij - v[vi, vj.start - 1:vj.stop - 1]) / dy,
            (v[vi, vj.start + 1:vj.stop + 1] - v_ij) / dy,
        )

        adv_v[vi, vj] = u_at_v * dv_dx + v_ij * dv_dy

        return adv_u, adv_v

    def step(self):
        """Advance one time step (Chorin projection)."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        nu, dt = self.nu, self.dt

        self._apply_bc()
        u = self.u
        v = self.v

        # --- Predictor: u* = u + dt*(advection + diffusion) ---
        u_star = u.copy()
        v_star = v.copy()

        adv_u, adv_v = self._advection(u, v)
        u_star -= dt * adv_u
        v_star -= dt * adv_v

        # u diffusion — interior faces i=1..Nx-1, all j
        lap_u_x = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx**2
        lap_u_y = np.zeros_like(lap_u_x)
        if u.shape[1] > 2:
            lap_u_y[:, 1:-1] = (
                (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
            )
        u_star[1:-1, :] += nu * dt * (lap_u_x + lap_u_y)

        # v diffusion — all i, interior j=1..Ny-1
        if v.shape[0] > 2 and v.shape[1] > 2:
            lap_v_x = np.zeros_like(v)
            lap_v_x[1:-1, :] = (
                (v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]) / dx**2
            )
            lap_v_x[0, :] = (v[1, :] - 3 * v[0, :]) / dx**2
            lap_v_x[-1, :] = (v[-2, :] - 3 * v[-1, :]) / dx**2

            lap_v_y = np.zeros_like(v)
            lap_v_y[:, 1:-1] = (
                (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / dy**2
            )

            v_star[:, 1:-1] += nu * dt * (lap_v_x + lap_v_y)[:, 1:-1]

        # Enforce BCs on predictor before divergence
        self._set_bc(u_star, v_star)

        # --- Pressure Poisson: div(u*)/dt = Lap(p) ---
        div = (
            (u_star[1:, :] - u_star[:-1, :]) / dx
            + (v_star[:, 1:] - v_star[:, :-1]) / dy
        )
        rhs = (-div / dt).flatten()
        rhs[0] = 0.0

        p_flat, info = cg(self.A, rhs)
        if info != 0:
            raise RuntimeError(f"CG solver failed to converge (info={info})")
        self.p = p_flat.reshape((Nx, Ny))
        self.p -= np.mean(self.p)

        # --- Corrector: u = u* - dt*grad(p) ---
        grad_p_x = (self.p[1:, :] - self.p[:-1, :]) / dx
        grad_p_y = (self.p[:, 1:] - self.p[:, :-1]) / dy

        self.u[1:-1, :] = u_star[1:-1, :] - dt * grad_p_x
        self.v[:, 1:-1] = v_star[:, 1:-1] - dt * grad_p_y

        self._apply_bc()

    def _divergence(self):
        """Pointwise divergence du/dx + dv/dy at cell centers."""
        return (
            (self.u[1:, :] - self.u[:-1, :]) / self.dx
            + (self.v[:, 1:] - self.v[:, :-1]) / self.dy
        )

    def divergence_norm(self):
        """RMS divergence."""
        return np.sqrt(np.mean(self._divergence()**2))

    def max_divergence(self, interior_only=False):
        """Max absolute divergence, optionally excluding boundary cells."""
        div = self._divergence()
        if interior_only and div.shape[0] > 2 and div.shape[1] > 2:
            div = div[1:-1, 1:-1]
        return np.max(np.abs(div))

    def cfl(self):
        """Maximum CFL number."""
        return (
            np.max(np.abs(self.u)) * self.dt / self.dx
            + np.max(np.abs(self.v)) * self.dt / self.dy
        )

    def solve(self, steps, verbose=True):
        """Run the simulation for the given number of steps."""
        for i in range(steps):
            self.step()
            if verbose and i % max(1, steps // 10) == 0:
                print(
                    f"Step {i:4d}: "
                    f"|div|_inf_int = {self.max_divergence(interior_only=True):.2e}, "
                    f"|div|_inf_all = {self.max_divergence():.2e}, "
                    f"CFL = {self.cfl():.3f}"
                )
