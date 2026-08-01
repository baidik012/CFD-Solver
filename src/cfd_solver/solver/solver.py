"""Incompressible Navier-Stokes solver using Chorin's projection method.

This module provides the main Solver class, which implements a fractional-step
method (Chorin's projection method) to solve the incompressible Navier-Stokes
equations on a staggered Arakawa C-grid.

Chorin's Projection Method Steps:
---------------------------------
1. Intermediate Velocity (Advection + Diffusion):
   Calculate an intermediate velocity field u* by solving the momentum
   equations without the pressure gradient:
       (u* - u^n) / dt = -(u^n · ∇)u^n + nu * ∇²u^n
   u* does not necessarily satisfy the incompressibility constraint (∇·u* = 0).

2. Pressure Poisson Equation:
   Solve for a pressure field 'p' that will project u* onto a divergence-free
   space. This is derived from the requirement that ∇·u^{n+1} = 0:
       ∇²p = (∇·u*) / dt

3. Velocity Correction (Projection):
   Correct the intermediate velocity using the calculated pressure gradient to
   obtain the divergence-free velocity field at the next time step:
       u^{n+1} = u* - dt * ∇p

Example
-------
>>> from cfd_solver.solver import Solver
>>> s = Solver(grid_size=(64, 64), nu=0.01, dt=0.001, lid_speed=1.0)
>>> s.solve(200)
>>> s.save("output/result.png")
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
    """Incompressible Navier-Stokes solver.

    Coordinates the computational mesh, boundary conditions, and numerical
    schemes to advance the fluid simulation in time.

    Parameters
    ----------
    grid_size : tuple[int, int]
        (Nx, Ny) number of computational cells.
    nu : float
        Kinematic viscosity of the fluid.
    dt : float
        Time step size.
    lid_speed : float, optional
        Tangential u-velocity on the top wall (default 1.0).
    smooth_lid : bool, optional
        Use a sinusoidal lid profile to avoid corner singularities (default True).
    advection_scheme : str, optional
        Numerical scheme for advection: "upwind" or "central" (default "upwind").
    diffusion_scheme : str, optional
        Numerical scheme for diffusion: "crank_nicolson" or "explicit"
        (default "crank_nicolson").
    Lx, Ly : float, optional
        Physical dimensions of the domain (default 1.0, 1.0).
    force : bool, optional
        If True, bypass safety checks on grid size and memory (default False).

    Attributes
    ----------
    u : ndarray
        u-velocity field (horizontal faces).
    v : ndarray
        v-velocity field (vertical faces).
    p : ndarray
        Pressure field (cell centers).
    """

    def __init__(self, grid_size, nu, dt, lid_speed=1.0, smooth_lid=True,
                 advection_scheme="upwind", diffusion_scheme="crank_nicolson",
                 Lx=1.0, Ly=1.0, force=False,
                 boundary_config=None,
                 body_force=None,
                 initial_condition=None):
        # --- Input Validation ---
        Nx, Ny = grid_size
        if Nx < 2 or Ny < 2:
            raise ValueError(f"grid_size must be at least (2, 2), got ({Nx}, {Ny})")
        if nu < 0:
            raise ValueError(f"viscosity (nu) must be non-negative, got {nu}")
        if dt <= 0:
            raise ValueError(f"time step (dt) must be positive, got {dt}")
        if Lx <= 0 or Ly <= 0:
            raise ValueError(f"domain size (Lx, Ly) must be positive, got ({Lx}, {Ly})")
        
        # --- Resource Guardrails ---
        total_cells = Nx * Ny
        if total_cells > 4_000_000 and not force:
            raise ValueError(
                f"Grid size {Nx}x{Ny} ({total_cells:,} cells) exceeds the safety limit of 4,000,000 cells. "
                "Use force=True to override."
            )

        # Estimate memory usage for main arrays and operators
        # conservative multiplier for sparse matrices and intermediate buffers
        est_mem_bytes = total_cells * 8 * 15 
        est_mem_gb = est_mem_bytes / (1024**3)

        if est_mem_gb > 4.0 and not force:
            raise MemoryError(
                f"Estimated memory usage ({est_mem_gb:.1f} GB) exceeds the safety limit of 4.0 GB. "
                "Use force=True to override."
            )
        elif est_mem_gb > 1.0:
            print(f"  [warning] Large grid detected. Estimated memory: {est_mem_gb:.1f} GB", file=sys.stderr)

        # Ensure dt is physically sensible
        dt_limit = 10.0 * (Lx + Ly) / max(abs(lid_speed), 1e-10)
        if dt > dt_limit and not force:
            raise ValueError(
                f"dt={dt} is nonsensically large for this domain (limit: {dt_limit:.2f}). "
                "Check your units or use force=True to override."
            )
        # ------------------------

        self.mesh = Mesh(Lx, Ly, Nx, Ny)

        # Build boundary conditions — accept pre-built object or legacy scalars
        if boundary_config is not None:
            self.bc = boundary_config
        else:
            self.bc = BoundaryConditions(top=lid_speed, smooth_lid=smooth_lid)

        # Stability check: CFL and diffusion constraints.
        # IMPORTANT: Flow velocities can reach 2-3x the lid speed due to recirculation.
        # We use a conservative limit to prevent blowup.
        dx, dy = self.mesh.dx, self.mesh.dy
        dx_min = min(dx, dy)

        # Advection stability: CFL = (|u|*dt/dx) + (|v|*dt/dy) < 1
        # Conservatively assume max speed reaches 3x lid_speed during simulation
        effective_lid_speed = max(abs(self.bc.top), abs(self.bc.bottom), 1e-10)
        expected_max_speed = 3.0 * effective_lid_speed
        dt_advection = 0.5 * dx_min / expected_max_speed
        
        # Diffusion stability (mainly for explicit schemes): dt < dx²/(4*nu)
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
        # Cell Peclet check: central differencing has no numerical dissipation
        # and produces grid-scale oscillations ("wiggles") when the cell Peclet
        # number Pe = |u|*dx/nu exceeds 2. This is independent of the CFL/diffusion
        # time-step limits above, so warn explicitly here.
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

        # Detect periodic directions
        self._periodic_x = any(isinstance(w, PeriodicWall) for w in self.bc.walls.values())

        # Initialize velocity and pressure arrays
        self.u = np.zeros(self.mesh.shape_u)
        self.v = np.zeros(self.mesh.shape_v)
        self.p = np.zeros(self.mesh.shape_p)

        # Pre-allocated intermediate buffers (avoids per-step allocation)
        self._u_star = np.zeros(self.mesh.shape_u)
        self._v_star = np.zeros(self.mesh.shape_v)

        # Pre-allocated contiguous buffers for diagnostics
        self._u_phys = np.empty((Nx + 1, Ny), dtype=np.float64)
        self._v_phys = np.empty((Nx, Ny + 1), dtype=np.float64)
        self._div_buf = np.empty((Nx, Ny), dtype=np.float64)

        # Configure numerical schemes
        if advection_scheme == "upwind":
            self._advection_fn = advection.upwind
        elif advection_scheme == "central":
            self._advection_fn = advection.central
        else:
            raise ValueError(f"Unknown advection scheme: {advection_scheme}")

        if diffusion_scheme == "crank_nicolson":
            # Crank-Nicolson matrices encode the BC type at construction time.
            # Periodic walls require a fundamentally different matrix structure
            # (circulant) which the current CrankNicolson / FFTCrankNicolson
            # solvers do not yet support.  Fall back to explicit Euler in that
            # case, but emit a LOUD warning so the user knows their simulation
            # is running with a different (less stable, first-order in time)
            # diffusion scheme than they asked for.
            # (Audit finding P0-2 — previously this downgrade was silent.)
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

        # Apply initial boundary conditions
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
        """Advance the simulation by one time step (dt).

        Executes the three steps of Chorin's projection method:
        1. Calculate intermediate velocity (u*, v*) via advection + diffusion.
        2. Solve the Pressure Poisson Equation for 'p'.
        3. Correct (project) the velocity field using the pressure gradient.
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt

        # Ensure BCs are up to date
        self.bc.apply(self.u, self.v, Nx, Ny)

        # 1. Prediction Step: Advection + Diffusion
        adv_u, adv_v = self._advection_fn(self.u, self.v, dx, dy)

        if self._diffusion is not None:
            # Semi-implicit Crank-Nicolson
            u_star, v_star = self._diffusion.solve(
                self.u, self.v, adv_u, adv_v,
                u_out=self._u_star, v_out=self._v_star,
            )
        else:
            # Explicit Forward Euler
            from .diffusion import explicit
            u_star, v_star = explicit(
                self.u, self.v, adv_u, adv_v, dx, dy, dt,
                self.nu, self.bc, Nx, Ny,
                u_out=self._u_star, v_out=self._v_star,
            )

        # Body force: add external forces to intermediate velocity
        if self._body_force_fn is not None:
            fu, fv = self._body_force_fn(self.u, self.v, self.time)
            u_star += dt * fu
            v_star += dt * fv

        # Periodic x: the advection/diffusion step doesn't update u_star
        # at the boundary face (i=0/i=Nx), so we compute the diffusion
        # contribution there using the wrapping Laplacian.
        if self._periodic_x:
            dx2, dy2 = dx**2, dy**2
            lap_u_0 = (u_star[1, 1:-1] - 2.0 * self.u[0, 1:-1] + u_star[-2, 1:-1]) / dx2 + \
                      (self.u[0, 2:] - 2.0 * self.u[0, 1:-1] + self.u[0, :-2]) / dy2
            u_star[0, 1:-1] = self.u[0, 1:-1] + dt * self.nu * lap_u_0
            u_star[-1, 1:-1] = u_star[0, 1:-1]

        # 2. Pressure Step: Solve ∇²p = (∇·u*) / dt
        self.p[:] = self._pressure.solve(u_star, v_star, dt)

        # 3. Correction Step: u^{n+1} = u* - dt * ∇p
        # On the staggered C-grid, pressure gradients are computed at face locations.
        grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dx
        grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dy

        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y

        # Periodic x: update u at the periodic face (i=0 = i=Nx)
        if self._periodic_x:
            grad_p_x_per = (self.p[1, 1:-1] - self.p[-2, 1:-1]) / dx
            self.u[0, 1:-1] = u_star[0, 1:-1] - dt * grad_p_x_per
            self.u[-1, 1:-1] = self.u[0, 1:-1]

        # Finalize BCs for the new velocity field
        self.bc.apply(self.u, self.v, Nx, Ny)

        # Advance physical time
        self.time += dt

    def solve(self, steps=None, verbose=True, simulation_time=None,
              convergence_tol=None, convergence_window=100):
        """Run the simulation.

        Either ``steps`` or ``simulation_time`` must be provided.  When
        ``simulation_time`` is given, the number of steps is computed
        automatically as ``ceil(simulation_time / dt)`` and the ``steps``
        argument is ignored.

        Parameters
        ----------
        steps : int, optional
            Number of time steps to advance.  Required if
            ``simulation_time`` is not set.
        verbose : bool, optional
            If True, print a progress bar and diagnostics (default True).
        simulation_time : float, optional
            Physical time (in seconds) to simulate.  When set, overrides
            ``steps``.
        convergence_tol : float, optional
            If set, stop early when max|u^{n+1} - u^n| < convergence_tol
            for ``convergence_window`` consecutive steps.
        convergence_window : int, optional
            Number of consecutive steps below tolerance before declaring
            convergence (default 100).

        Returns
        -------
        bool
            True if all steps completed, False if the simulation blew up
            (velocity field became NaN/Inf) and was aborted early.
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
        for i in range(steps):
            self.step()

            # Stability check: ensure velocities haven't exploded
            if is_blowup(self.u, self.v):
                if verbose:
                    c = self.cfl()
                    # If CFL is inf, the field has NaN/Inf; suggest much smaller dt
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

            # Steady-state convergence check
            if convergence_tol is not None:
                delta = np.max(np.abs(self.u - u_old))
                if delta < convergence_tol:
                    converged_count += 1
                    if converged_count >= convergence_window:
                        if verbose:
                            sys.stdout.write(
                                f"\n  Converged at step {i+1}: "
                                f"max|du|={delta:.2e} < {convergence_tol:.2e} "
                                f"for {convergence_window} steps\n"
                            )
                        return True
                else:
                    converged_count = 0
                u_old[:] = self.u

            if verbose:
                bar_len = 30
                filled = int(bar_len * (i + 1) / steps)
                bar = "=" * filled + "-" * (bar_len - filled)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (steps - i - 1) / rate if rate > 0 else 0
                # Compute diagnostics every 10 steps to reduce overhead
                if i % 10 == 0 or i == steps - 1:
                    div = self.max_divergence()
                    c = self.cfl()
                # Handle inf CFL gracefully in display
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
        """Calculate the RMS divergence of the current velocity field."""
        self._u_phys[:] = self.u[:, 1:-1]
        self._v_phys[:] = self.v[1:-1, :]
        self._div_buf[:] = (self._u_phys[1:, :] - self._u_phys[:-1, :]) / self.dx + \
                           (self._v_phys[:, 1:] - self._v_phys[:, :-1]) / self.dy
        return float(np.sqrt(np.mean(self._div_buf ** 2)))

    def max_divergence(self, interior_only=False):
        """Calculate the maximum absolute divergence."""
        self._u_phys[:] = self.u[:, 1:-1]
        self._v_phys[:] = self.v[1:-1, :]
        self._div_buf[:] = (self._u_phys[1:, :] - self._u_phys[:-1, :]) / self.dx + \
                           (self._v_phys[:, 1:] - self._v_phys[:, :-1]) / self.dy
        d = self._div_buf
        if interior_only and d.shape[0] > 2 and d.shape[1] > 2:
            d = d[1:-1, 1:-1]
        return float(np.max(np.abs(d)))

    def cfl(self):
        """Calculate the current maximum CFL number."""
        if not (np.all(np.isfinite(self.u)) and np.all(np.isfinite(self.v))):
            return np.inf
        self._u_phys[:] = self.u[:, 1:-1]
        self._v_phys[:] = self.v[1:-1, :]
        u_max = np.max(np.abs(self._u_phys))
        v_max = np.max(np.abs(self._v_phys))
        return u_max * self.dt / self.dx + v_max * self.dt / self.dy

    def save(self, path, skip=None, scale=None):
        """Save a visualization of pressure and velocity magnitude.

        Parameters
        ----------
        path : str
            File path to save the image.
        skip : int, optional
            Number of points to skip for quiver plot (if included).
        scale : float, optional
            Scaling factor for vectors.
        """
        save_contour(self.mesh, self.u, self.v, self.p, path, skip, scale)

    def save_quiver(self, path, skip=None, scale=None):
        """Save a velocity vector (quiver) plot.

        Parameters
        ----------
        path : str
            File path to save the image.
        skip : int, optional
            Number of vectors to skip in each direction for clarity.
        scale : float, optional
            Scaling factor for vector lengths.
        """
        _save_quiver(self.mesh, self.u, self.v, path, skip, scale)

    def save_streamlines(self, path, density=2.0):
        """Save a streamline plot of the velocity field and pressure contours.

        Parameters
        ----------
        path : str
            File path to save the image.
        density : float, optional
            Density of streamlines in the plot.
        """
        save_streamlines(self.mesh, self.u, self.v, self.p, path, density)

    def checkpoint(self, path):
        """Save the current solver state to a compressed .npz file.

        The checkpoint includes all velocity and pressure data, as well
        as solver configuration, allowing the simulation to be resumed.

        Parameters
        ----------
        path : str
            Destination file path. Parent directories are created automatically.
        """
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

        Parameters
        ----------
        path : str
            Path to the .npz checkpoint file.
        force : bool, optional
            If True, bypass safety checks during initialization.
        body_force : callable, optional
            Re-attach a body-force function to the restored solver.
            Checkpoints do not serialize Python callables, so any
            body force from the original simulation MUST be re-supplied
            here.  A loud warning is printed if this is None and the
            caller is expected to have had one.  (Audit finding P2-12.)
        initial_condition : callable, optional
            Re-attach an initial-condition function.  Same caveat as
            ``body_force`` — not serialized, must be re-supplied.
        boundary_config : BoundaryConditions, optional
            Re-attach a full boundary-conditions object.  Checkpoints
            only serialize ``lid_speed`` and ``smooth_lid`` (the legacy
            scalar attributes); any non-default wall types (InletWall,
            OutletWall, PeriodicWall, FreeSlipWall) are lost on
            checkpoint round-trip.  Pass the original BC object here
            to restore it.  A warning is printed if this is None and
            the checkpoint appears to have used non-default BCs.

        Returns
        -------
        Solver
            A new Solver instance with state restored from the file.
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

        # ── Audit finding P2-12 — warn about unserialised state ───────
        # The legacy checkpoint format only stores lid_speed + smooth_lid.
        # Any non-default BC (InletWall, OutletWall, PeriodicWall,
        # FreeSlipWall) and any body_force / initial_condition are
        # silently lost.  Warn loudly so users do not accidentally
        # resume, e.g., a channel-flow sim as a lid-driven cavity.
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
            # We cannot detect whether the original had a body force
            # (it is not serialised), so we warn unconditionally.
            print(
                "  [WARNING] from_checkpoint: body_force not supplied.\n"
                "            If the original simulation used a body force, it is\n"
                "            LOST on checkpoint round-trip (Python callables are\n"
                "            not serialised).  Pass body_force=<original fn> to\n"
                "            restore, or force=True to silence.",
                file=sys.stderr,
            )
        # ────────────────────────────────────────────────────────────────

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
