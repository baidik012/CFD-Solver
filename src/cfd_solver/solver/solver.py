"""Incompressible Navier-Stokes solver using Chorin's projection method.

This is the main public API. It wires together the mesh, boundary conditions,
advection, diffusion, pressure solver, and diagnostics into a single Solver
class with a clean interface.

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
from .bc import BoundaryConditions
from . import advection
from .diffusion import CrankNicolson
from .pressure import PressureSolver
from .diagnostics import (
    divergence as _divergence,
    divergence_norm,
    max_divergence,
    cfl,
    is_blowup,
)
from .viz import save_quiver as _save_quiver, save_contour


class Solver:
    """Incompressible Navier-Stokes solver.

    Parameters
    ----------
    grid_size : tuple[int, int]
        (Nx, Ny) number of cells.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    lid_speed : float, optional
        u-velocity on the top wall. Default 1.0.
    smooth_lid : bool, optional
        Use sinusoidal lid profile to avoid corner singularity. Default True.
    advection_scheme : str, optional
        "upwind" (default) or "central".
    diffusion_scheme : str, optional
        "crank_nicolson" (default) or "explicit".
    Lx, Ly : float, optional
        Domain size. Default 1.0 (unit square).
    """

    def __init__(self, grid_size, nu, dt, lid_speed=1.0, smooth_lid=True,
                 advection_scheme="upwind", diffusion_scheme="crank_nicolson",
                 Lx=1.0, Ly=1.0):
        Nx, Ny = grid_size

        self.mesh = Mesh(Lx, Ly, Nx, Ny)
        self.bc = BoundaryConditions(top=lid_speed, smooth_lid=smooth_lid)

        # Auto-scale dt to keep CFL < 1 if the user's dt is too large
        dx, dy = self.mesh.dx, self.mesh.dy
        dt_max = min(dx, dy) / max(lid_speed, 1e-10) * 0.1
        if dt > dt_max:
            dt = dt_max

        self.dt = dt
        self.nu = nu
        self.advection_scheme = advection_scheme
        self.diffusion_scheme = diffusion_scheme

        # Velocity and pressure arrays
        self.u = np.zeros(self.mesh.shape_u)
        self.v = np.zeros(self.mesh.shape_v)
        self.p = np.zeros(self.mesh.shape_p)

        # Select advection scheme
        if advection_scheme == "upwind":
            self._advection_fn = advection.upwind
        elif advection_scheme == "central":
            self._advection_fn = advection.central
        else:
            raise ValueError(f"Unknown advection scheme: {advection_scheme}")

        # Build diffusion solver
        if diffusion_scheme == "crank_nicolson":
            self._diffusion = CrankNicolson(self.mesh, nu, dt, self.bc)
        elif diffusion_scheme == "explicit":
            self._diffusion = None
        else:
            raise ValueError(f"Unknown diffusion scheme: {diffusion_scheme}")

        # Build pressure solver
        self._pressure = PressureSolver(self.mesh)

        # Apply initial BCs
        self.bc.apply(self.u, self.v, Nx, Ny)

    @property
    def Nx(self):
        return self.mesh.Nx

    @property
    def Ny(self):
        return self.mesh.Ny

    @property
    def dx(self):
        return self.mesh.dx

    @property
    def dy(self):
        return self.mesh.dy

    @property
    def Lx(self):
        return self.mesh.Lx

    @property
    def Ly(self):
        return self.mesh.Ly

    def step(self):
        """Advance one time step."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt

        self.bc.apply(self.u, self.v, Nx, Ny)

        adv_u, adv_v = self._advection_fn(self.u, self.v, dx, dy)

        if self._diffusion is not None:
            u_star, v_star = self._diffusion.solve(self.u, self.v, adv_u, adv_v)
        else:
            from .diffusion import explicit
            u_star, v_star = explicit(
                self.u, self.v, adv_u, adv_v, dx, dy, dt,
                self.nu, self.bc, Nx, Ny,
            )

        self.p[:] = self._pressure.solve(u_star, v_star, dt)

        # Pressure gradient must align with staggered C-grid:
        # u lives on x-faces => du = -dt * (p[i] - p[i-1]) / dx
        # v lives on y-faces => dv = -dt * (p[j] - p[j-1]) / dy
        # Interior u faces are i=1..Nx-1. Gradient at face i uses cells i and i+1.
        # In p array, cells 1..Nx are physical. Face i=1 is between p[1] and p[2].
        grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dx  # (Nx-1, Ny)
        grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dy  # (Nx, Ny-1)

        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y

        self.bc.apply(self.u, self.v, Nx, Ny)

    def solve(self, steps, verbose=True):
        """Run the simulation for the given number of steps."""
        t0 = time.time()
        for i in range(steps):
            self.step()

            if is_blowup(self.u, self.v):
                if verbose:
                    c = cfl(self.u, self.v, self.dx, self.dy, self.dt)
                    safe_dt = self.dt / max(c, 1.0) * 0.8
                    print(
                        f"\nStep {i:4d}: BLOWUP — velocity is NaN/Inf."
                        f"  CFL at last step: {c:.2f}\n"
                        f"  Try: dt <= {safe_dt:.4g}  (currently {self.dt}),"
                        f" smaller lid_speed, or smooth_lid=True."
                    )
                return

            if verbose:
                bar_len = 30
                filled = int(bar_len * (i + 1) / steps)
                bar = "=" * filled + "-" * (bar_len - filled)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (steps - i - 1) / rate if rate > 0 else 0
                div = max_divergence(self.u, self.v, self.dx, self.dy)
                c = cfl(self.u, self.v, self.dx, self.dy, self.dt)
                sys.stdout.write(
                    f"\r  [{bar}] {i+1:4d}/{steps}  "
                    f"|div|={div:.2e}  CFL={c:.3f}  "
                    f"ETA={eta:.0f}s"
                )
                sys.stdout.flush()

        if verbose:
            elapsed = time.time() - t0
            sys.stdout.write(f"\n  Done in {elapsed:.1f}s\n")

    def divergence_norm(self):
        """RMS divergence."""
        return divergence_norm(self.u, self.v, self.dx, self.dy)

    def max_divergence(self, interior_only=False):
        """Max absolute divergence."""
        return max_divergence(self.u, self.v, self.dx, self.dy, interior_only)

    def cfl(self):
        """Maximum CFL number."""
        return cfl(self.u, self.v, self.dx, self.dy, self.dt)

    def save(self, path, skip=None, scale=None):
        """Save pressure contour + velocity magnitude plot."""
        save_contour(self.mesh, self.u, self.v, self.p, path, skip, scale)

    def save_quiver(self, path, skip=None, scale=None):
        """Save velocity vector plot."""
        _save_quiver(self.mesh, self.u, self.v, path, skip, scale)

    def checkpoint(self, path):
        """Save solver state to a .npz file for resume later.

        Parameters
        ----------
        path : str
            Destination file path. Parent directories are created automatically.
            A `.npz` extension is added by NumPy if not already present.
        """
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            path,
            u=self.u, v=self.v, p=self.p,
            Nx=self.Nx, Ny=self.Ny,
            Lx=self.Lx, Ly=self.Ly,
            dt=self.dt, nu=self.nu,
            lid_speed=self.bc.top,  # Save lid_speed explicitly
            smooth_lid=self.bc.smooth_lid,
            advection_scheme=self.advection_scheme,
            diffusion_scheme=self.diffusion_scheme,
        )

    @classmethod
    def from_checkpoint(cls, path):
        """Load a solver from a checkpoint file.

        Returns the Solver instance with u, v, p restored.
        The diffusion/pressure solvers are rebuilt from saved parameters.
        """
        data = np.load(path)
        Nx, Ny = int(data["Nx"]), int(data["Ny"])
        Lx, Ly = float(data["Lx"]), float(data["Ly"])
        dt, nu = float(data["dt"]), float(data["nu"])
        lid_speed = float(data["lid_speed"]) if "lid_speed" in data else 1.0
        smooth_lid = bool(data["smooth_lid"]) if "smooth_lid" in data else True
        advection_scheme = str(data["advection_scheme"]) if "advection_scheme" in data else "upwind"
        diffusion_scheme = str(data["diffusion_scheme"]) if "diffusion_scheme" in data else "crank_nicolson"

        solver = cls(grid_size=(Nx, Ny), nu=nu, dt=dt, Lx=Lx, Ly=Ly,
                     lid_speed=lid_speed, smooth_lid=smooth_lid,
                     advection_scheme=advection_scheme, diffusion_scheme=diffusion_scheme)
        solver.u[:] = data["u"]
        solver.v[:] = data["v"]
        solver.p[:] = data["p"]
        return solver
