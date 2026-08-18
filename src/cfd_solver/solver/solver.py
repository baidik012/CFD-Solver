"""Incompressible Navier-Stokes solver using Chorin's projection method.

The solver advances velocity on a staggered Arakawa C-grid using three
fractional steps: an advection/diffusion predictor, a pressure Poisson
solve, and a velocity correction (projection). Schemes for advection
(upwind/central) and diffusion (Crank-Nicolson/explicit) are pluggable.
"""

import os
import sys
import time
import numpy as np

from .mesh import Mesh
from .bc import BoundaryConditions, NoSlipWall, OutletWall, InletWall, FreeSlipWall, PeriodicWall
from . import advection
from .diffusion import CrankNicolson, create_diffusion_solver
from .pressure import create_pressure_solver
from .diagnostics import (
    divergence_norm,
    max_divergence,
    cfl,
    is_blowup,
)
from .viz import save_quiver as _save_quiver, save_contour, save_streamlines


class Solver:
    """Incompressible Navier-Stokes solver on a staggered C-grid.

    Attributes
    ----------
    u, v : ndarray
        Velocity fields on the x-/y-faces (include ghost cells).
    p : ndarray
        Pressure field at cell centers (include ghost cells).
    """

    def __init__(self, grid_size, nu, dt, lid_speed=1.0, smooth_lid=True,
                 advection_scheme="upwind", diffusion_scheme="crank_nicolson",
                 Lx=1.0, Ly=1.0, force=False,
                 boundary_config=None,
                 body_force=None,
                 initial_condition=None):
        Nx, Ny = grid_size
        if Nx < 2 or Ny < 2:
            raise ValueError(f"grid_size must be at least (2, 2), got ({Nx}, {Ny})")
        if nu < 0:
            raise ValueError(f"viscosity (nu) must be non-negative, got {nu}")
        if dt <= 0:
            raise ValueError(f"time step (dt) must be positive, got {dt}")
        if Lx <= 0 or Ly <= 0:
            raise ValueError(f"domain size (Lx, Ly) must be positive, got ({Lx}, {Ly})")

        # Resource guardrails: refuse grids that would exhaust memory
        total_cells = Nx * Ny
        if total_cells > 4_000_000 and not force:
            raise ValueError(
                f"Grid size {Nx}x{Ny} ({total_cells:,} cells) exceeds the safety limit of 4,000,000 cells. "
                "Use force=True to override."
            )

        # Conservative memory estimate: ~15 arrays at 8 bytes each
        est_mem_bytes = total_cells * 8 * 15
        est_mem_gb = est_mem_bytes / (1024**3)

        if est_mem_gb > 4.0 and not force:
            raise MemoryError(
                f"Estimated memory usage ({est_mem_gb:.1f} GB) exceeds the safety limit of 4.0 GB. "
                "Use force=True to override."
            )
        elif est_mem_gb > 1.0:
            print(f"  [warning] Large grid detected. Estimated memory: {est_mem_gb:.1f} GB", file=sys.stderr)

        # Reject physically nonsensical time steps up front
        dt_limit = 10.0 * (Lx + Ly) / max(abs(lid_speed), 1e-10)
        if dt > dt_limit and not force:
            raise ValueError(
                f"dt={dt} is nonsensically large for this domain (limit: {dt_limit:.2f}). "
                "Check your units or use force=True to override."
            )

        self.mesh = Mesh(Lx, Ly, Nx, Ny)

        # Accept pre-built BC object or legacy scalar API
        if boundary_config is not None:
            self.bc = boundary_config
        else:
            self.bc = BoundaryConditions(top=lid_speed, smooth_lid=smooth_lid)

        # Inlet profiles use the real domain height, not a hardcoded H=1
        self.bc._domain_width = Lx
        self.bc._domain_height = Ly

        # Time-step limits: CFL and explicit-diffusion stability.
        # Recirculation can reach 2-3x the lid speed, so keep a
        # conservative margin.
        dx, dy = self.mesh.dx, self.mesh.dy
        dx_min = min(dx, dy)

        effective_lid_speed = max(abs(self.bc.top), abs(self.bc.bottom), 1e-10)
        expected_max_speed = 3.0 * effective_lid_speed
        dt_advection = 0.5 * dx_min / expected_max_speed

        if nu > 1e-10:
            dt_diffusion = 0.25 * dx_min**2 / nu
        else:
            dt_diffusion = float('inf')

        dt_max = min(dt_advection, dt_diffusion)

        if dt > dt_max and not force:
            raise ValueError(
                f"dt={dt} exceeds the numerical stability limit ({dt_max:.4g}). "
                f"This limit accounts for advection (CFL < 0.5, assuming peak speed ~{expected_max_speed:.3f}) "
                f"and diffusion stability. "
                f"Suggested dt <= {dt_max:.4g}. Use force=True to override."
            )
        # Central differencing has no numerical dissipation and produces
        # grid-scale oscillations when the cell Peclet number Pe = |u|*dx/nu
        # exceeds 2. Warn independently of the stability limits above.
        if advection_scheme == "central" and nu > 1e-10:
            cell_peclet = expected_max_speed * dx_min / nu
            if cell_peclet > 2.0 and not force:
                print(
                    f"  [warning] Central advection with cell Peclet number "
                    f"{cell_peclet:.1f} > 2 may produce grid-scale oscillations. "
                    f"Use advection_scheme='upwind', refine the grid, or increase nu.",
                    file=sys.stderr,
                )

        self.dt = dt
        self.nu = nu
        self.time = 0.0
        self._body_force_fn = body_force
        self._initial_condition_fn = initial_condition
        self.advection_scheme = advection_scheme
        self.diffusion_scheme = diffusion_scheme

        # Periodic is only active when the opposite wall pair uses it
        self._periodic_x = (isinstance(self.bc.walls.get('left'), PeriodicWall) and
                            isinstance(self.bc.walls.get('right'), PeriodicWall))

        # Velocity/pressure arrays and pre-allocated buffers
        self.u = np.zeros(self.mesh.shape_u)
        self.v = np.zeros(self.mesh.shape_v)
        self.p = np.zeros(self.mesh.shape_p)

        self._u_star = np.zeros(self.mesh.shape_u)
        self._v_star = np.zeros(self.mesh.shape_v)

        # Contiguous buffers for diagnostics (avoid per-call allocation)
        self._u_phys = np.empty((Nx + 1, Ny), dtype=np.float64)
        self._v_phys = np.empty((Nx, Ny + 1), dtype=np.float64)
        self._div_buf = np.empty((Nx, Ny), dtype=np.float64)

        if advection_scheme == "upwind":
            self._advection_fn = advection.upwind
        elif advection_scheme == "central":
            self._advection_fn = advection.central
        else:
            raise ValueError(f"Unknown advection scheme: {advection_scheme}")

        if diffusion_scheme == "crank_nicolson":
            # CN matrices encode the BC type at construction time; the
            # periodic case needs a circulant structure that the current
            # solvers don't support, so fall back to explicit Euler loudly.
            if self.bc.has_periodic():
                self._diffusion = None
                if not force:
                    print(
                        "  [WARNING] PeriodicWall detected — Crank-Nicolson diffusion\n"
                        "            is not yet supported with periodic boundary conditions.\n"
                        "            Falling back to explicit Euler (first-order in time,\n"
                        "            requires dt < dx^2 / (4*nu) for stability).\n"
                        "            To silence this warning: pass force=True or set\n"
                        "            diffusion_scheme='explicit' explicitly.",
                        file=sys.stderr,
                    )
            else:
                self._diffusion = create_diffusion_solver(self.mesh, nu, dt, self.bc)
        elif diffusion_scheme == "explicit":
            self._diffusion = None
        else:
            raise ValueError(f"Unknown diffusion scheme: {diffusion_scheme}")

        self._pressure = create_pressure_solver(self.mesh, self.bc)

        self.bc.apply(self.u, self.v, Nx, Ny)

    @property
    def Nx(self):
        """Number of cells in x."""
        return self.mesh.Nx

    @property
    def Ny(self):
        """Number of cells in y."""
        return self.mesh.Ny

    @property
    def dx(self):
        """Grid spacing in x."""
        return self.mesh.dx

    @property
    def dy(self):
        """Grid spacing in y."""
        return self.mesh.dy

    @property
    def Lx(self):
        """Domain length in x."""
        return self.mesh.Lx

    @property
    def Ly(self):
        """Domain length in y."""
        return self.mesh.Ly

    def step(self):
        """Advance the simulation by one time step."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt

        self.bc.apply(self.u, self.v, Nx, Ny)

        # Predictor: advection + diffusion
        adv_u, adv_v = self._advection_fn(self.u, self.v, dx, dy)

        if self._diffusion is not None:
            u_star, v_star = self._diffusion.solve(
                self.u, self.v, adv_u, adv_v,
                u_out=self._u_star, v_out=self._v_star,
            )
        else:
            from .diffusion import explicit
            u_star, v_star = explicit(
                self.u, self.v, adv_u, adv_v, dx, dy, dt,
                self.nu, self.bc, Nx, Ny,
                u_out=self._u_star, v_out=self._v_star,
            )

        # Body force
        if self._body_force_fn is not None:
            fu, fv = self._body_force_fn(self.u, self.v, self.time)
            u_star += dt * fu
            v_star += dt * fv

        # Periodic x: the boundary face (i=0=i=Nx) is a real degree of
        # freedom; compute its full prediction with the wrapping Laplacian.
        if self._periodic_x:
            dx2, dy2 = dx**2, dy**2
            lap_u_0 = (self.u[1, 1:-1] - 2.0 * self.u[0, 1:-1] + self.u[-2, 1:-1]) / dx2 + \
                      (self.u[0, 2:] - 2.0 * self.u[0, 1:-1] + self.u[0, :-2]) / dy2
            u_star[0, 1:-1] = self.u[0, 1:-1] + dt * (-adv_u[0, 1:-1] + self.nu * lap_u_0)
            u_star[-1, 1:-1] = u_star[0, 1:-1]

        # Pressure solve
        self.p[:] = self._pressure.solve(u_star, v_star, dt)

        # Projection: u^{n+1} = u* - dt * ∇p (gradients at face locations)
        grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dx
        grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dy

        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y

        # Periodic x: correct the periodic face too
        if self._periodic_x:
            grad_p_x_per = (self.p[1, 1:-1] - self.p[-2, 1:-1]) / dx
            self.u[0, 1:-1] = u_star[0, 1:-1] - dt * grad_p_x_per
            self.u[-1, 1:-1] = self.u[0, 1:-1]

        self.bc.apply(self.u, self.v, Nx, Ny)

        self.time += dt

    def solve(self, steps=None, verbose=True, simulation_time=None,
              convergence_tol=None, convergence_window=100):
        """Run the simulation.

        Either ``steps`` or ``simulation_time`` must be provided.  When
        ``simulation_time`` is given, the step count is computed as
        ``ceil(simulation_time / dt)`` and ``steps`` is ignored.

        Returns True if all steps completed, False if the velocity field
        blew up (NaN/Inf) and the run was aborted.
        """
        if simulation_time is not None:
            if simulation_time <= 0:
                raise ValueError(f"simulation_time must be positive, got {simulation_time}")
            steps = int(np.ceil(simulation_time / self.dt))
        if steps is None or steps <= 0:
            raise ValueError("Provide either steps > 0 or simulation_time > 0")
        if steps > 1_000_000:
            print(f"  [warning] Requesting {steps:,} steps. This may take a long time.", file=sys.stderr)

        # Apply initial condition if provided
        if self._initial_condition_fn is not None:
            self.u[:], self.v[:], self.p[:] = self._initial_condition_fn(self.mesh)
            self.bc.apply(self.u, self.v, self.Nx, self.Ny)

        t0 = time.time()
        div = 0.0
        c = 0.0
        converged_count = 0
        u_old = self.u.copy() if convergence_tol is not None else None
        v_old = self.v.copy() if convergence_tol is not None else None
        for i in range(steps):
            self.step()

            # Abort on NaN/Inf velocities, with a suggested safe dt
            if is_blowup(self.u, self.v):
                if verbose:
                    c = self.cfl()
                    if c == np.inf:
                        safe_dt = self.dt * 0.1
                    else:
                        safe_dt = self.dt / max(c, 1.0) * 0.5
                    cfl_str = f"{c:.2e}" if c != np.inf else "Inf"
                    print(
                        f"\nStep {i:4d}: BLOWUP — velocity is NaN/Inf. "
                        f"CFL at last step: {cfl_str}\n"
                        f"  Try: dt <= {safe_dt:.4g}  (currently {self.dt}), "
                        f"smaller lid_speed, or smooth_lid=True."
                    )
                return False

            # Steady-state check: require both velocity components to
            # satisfy the convergence criterion.
            if convergence_tol is not None:
                delta_u = np.max(np.abs(self.u - u_old))
                delta_v = np.max(np.abs(self.v - v_old))
                delta = max(delta_u, delta_v)
                if delta < convergence_tol:
                    converged_count += 1
                    if converged_count >= convergence_window:
                        if verbose:
                            sys.stdout.write(
                                f"\n  Converged at step {i+1}: "
                                f"max|du|={delta_u:.2e}, max|dv|={delta_v:.2e} "
                                f"< {convergence_tol:.2e} "
                                f"for {convergence_window} steps\n"
                            )
                        return True
                else:
                    converged_count = 0
                u_old[:] = self.u
                v_old[:] = self.v

            if verbose:
                bar_len = 30
                filled = int(bar_len * (i + 1) / steps)
                bar = "=" * filled + "-" * (bar_len - filled)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (steps - i - 1) / rate if rate > 0 else 0
                # Diagnostics every 10 steps cut the overhead of the norm calls
                if i % 10 == 0 or i == steps - 1:
                    div = self.max_divergence()
                    c = self.cfl()
                cfl_str = f"{c:.3f}" if c != np.inf else "Inf"
                sys.stdout.write(
                    f"\r  [{bar}] {i+1:4d}/{steps}  "
                    f"|div|={div:.2e}  CFL={cfl_str}  "
                    f"ETA={eta:.0f}s"
                )
                sys.stdout.flush()

        if verbose:
            elapsed = time.time() - t0
            sys.stdout.write(f"\n  Done in {elapsed:.1f}s\n")
        return True

    def divergence_norm(self):
        """RMS divergence of the current velocity field."""
        self._u_phys[:] = self.u[:, 1:-1]
        self._v_phys[:] = self.v[1:-1, :]
        self._div_buf[:] = (self._u_phys[1:, :] - self._u_phys[:-1, :]) / self.dx + \
                           (self._v_phys[:, 1:] - self._v_phys[:, :-1]) / self.dy
        return float(np.sqrt(np.mean(self._div_buf ** 2)))

    def max_divergence(self, interior_only=False):
        """Maximum absolute divergence of the current velocity field."""
        self._u_phys[:] = self.u[:, 1:-1]
        self._v_phys[:] = self.v[1:-1, :]
        self._div_buf[:] = (self._u_phys[1:, :] - self._u_phys[:-1, :]) / self.dx + \
                           (self._v_phys[:, 1:] - self._v_phys[:, :-1]) / self.dy
        d = self._div_buf
        if interior_only and d.shape[0] > 2 and d.shape[1] > 2:
            d = d[1:-1, 1:-1]
        return float(np.max(np.abs(d)))

    def cfl(self):
        """Current maximum CFL number."""
        if not (np.all(np.isfinite(self.u)) and np.all(np.isfinite(self.v))):
            return np.inf
        self._u_phys[:] = self.u[:, 1:-1]
        self._v_phys[:] = self.v[1:-1, :]
        u_max = np.max(np.abs(self._u_phys))
        v_max = np.max(np.abs(self._v_phys))
        return u_max * self.dt / self.dx + v_max * self.dt / self.dy

    def save(self, path, skip=None, scale=None):
        """Save a visualization of pressure and velocity magnitude."""
        save_contour(self.mesh, self.u, self.v, self.p, path, skip, scale)

    def save_quiver(self, path, skip=None, scale=None):
        """Save a velocity vector (quiver) plot."""
        _save_quiver(self.mesh, self.u, self.v, path, skip, scale)

    def save_streamlines(self, path, density=2.0):
        """Save a streamline plot of the velocity field and pressure contours."""
        save_streamlines(self.mesh, self.u, self.v, self.p, path, density)

    def checkpoint(self, path):
        """Save the current solver state to a compressed .npz file."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            path,
            u=self.u, v=self.v, p=self.p,
            Nx=self.Nx, Ny=self.Ny,
            Lx=self.Lx, Ly=self.Ly,
            dt=self.dt, nu=self.nu,
            lid_speed=self.bc.top,
            smooth_lid=self.bc.smooth_lid,
            advection_scheme=self.advection_scheme,
            diffusion_scheme=self.diffusion_scheme,
        )

    @classmethod
    def from_checkpoint(cls, path, force=False, body_force=None,
                        initial_condition=None, boundary_config=None):
        """Load a solver instance from a checkpoint file.

        Python callables (body_force, initial_condition) and non-default
        wall types are not serialized, so they MUST be re-supplied by the
        caller; a warning is printed when they are missing.
        """
        if not os.path.exists(path):
            if not path.endswith(".npz") and os.path.exists(path + ".npz"):
                path += ".npz"
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {path}")

        data = np.load(path)

        required_keys = ["u", "v", "p", "Nx", "Ny", "Lx", "Ly", "dt", "nu"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Corrupted checkpoint: missing key '{key}'")

        Nx, Ny = int(data["Nx"]), int(data["Ny"])
        Lx, Ly = float(data["Lx"]), float(data["Ly"])
        dt, nu = float(data["dt"]), float(data["nu"])
        lid_speed = float(data["lid_speed"]) if "lid_speed" in data else 1.0
        smooth_lid = bool(data["smooth_lid"]) if "smooth_lid" in data else True
        advection_scheme = str(data["advection_scheme"]) if "advection_scheme" in data else "upwind"
        diffusion_scheme = str(data["diffusion_scheme"]) if "diffusion_scheme" in data else "crank_nicolson"

        # Warn about unserialized state: the legacy checkpoint format only
        # stores lid_speed + smooth_lid, so resuming without the original
        # BC object silently turns non-default walls (e.g. a channel-flow
        # inlet/outlet) back into a lid-driven cavity.
        if boundary_config is None and not force:
            print(
                "  [WARNING] from_checkpoint: boundary_config not supplied.\n"
                "            Checkpoints only serialise lid_speed and smooth_lid;\n"
                "            any non-default wall types (InletWall, OutletWall,\n"
                "            PeriodicWall, FreeSlipWall) from the original run are\n"
                "            LOST.  Pass boundary_config=<original BC> to restore.\n"
                "            (Set force=True to silence this warning.)",
                file=sys.stderr,
            )
        if body_force is None and not force:
            # Cannot detect whether the original had a body force (it is
            # not serialized), so warn unconditionally.
            print(
                "  [WARNING] from_checkpoint: body_force not supplied.\n"
                "            If the original simulation used a body force, it is\n"
                "            LOST on checkpoint round-trip (Python callables are\n"
                "            not serialised).  Pass body_force=<original fn> to\n"
                "            restore, or force=True to silence.",
                file=sys.stderr,
            )

        solver = cls(grid_size=(Nx, Ny), nu=nu, dt=dt, Lx=Lx, Ly=Ly,
                     lid_speed=lid_speed, smooth_lid=smooth_lid,
                     advection_scheme=advection_scheme, diffusion_scheme=diffusion_scheme,
                     force=force,
                     boundary_config=boundary_config,
                     body_force=body_force,
                     initial_condition=initial_condition)

        # Validate data integrity
        for name, arr in [("u", data["u"]), ("v", data["v"]), ("p", data["p"])]:
            expected_shape = getattr(solver, name).shape
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Checkpoint array '{name}' shape mismatch. "
                    f"Expected {expected_shape}, got {arr.shape}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"Checkpoint array '{name}' contains NaN or Inf values")

        solver.u[:] = data["u"]
        solver.v[:] = data["v"]
        solver.p[:] = data["p"]
        return solver