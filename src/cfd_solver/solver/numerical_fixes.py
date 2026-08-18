"""P0 numerical correctness fixes for v0.3.2.

This module is intentionally isolated so the critical fixes can be reviewed and
merged independently. It is imported by ``solver.__init__`` and replaces the
public Solver class with ``P0Solver`` while preserving the existing public API.

P0 fixes implemented here:

1. Periodic pressure topology is derived from the actual x/y periodic pairs.
   The old solver used a periodic-x/Neumann-y spectral operator for every
   configuration containing any PeriodicWall.
2. Inlet/outlet normal-velocity cases are rejected when Crank-Nicolson is
   requested. The existing CN matrix does not consistently incorporate those
   normal-velocity boundary equations, so silently solving that problem would
   produce a physically inconsistent predictor. Failing fast is safer than
   returning a wrong solution until a dedicated mixed-BC CN operator is added.
3. Periodic staggered velocity faces are advanced as real PDE degrees of
   freedom. The periodic u face in x and periodic v face in y receive the
   same advection, diffusion, body-force, and pressure-correction treatment as
   their duplicated physical faces.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, fft, idct, ifft

from . import diffusion as diffusion_module
from . import pressure as pressure_module
from . import solver as solver_module
from .bc import InletWall, OutletWall, PeriodicWall


class GeneralPeriodicPressureSolver:
    """FFT/DCT pressure solver for arbitrary periodic directions.

    The pressure unknowns are cell-centered. A periodic direction uses a DFT
    and the standard periodic discrete Laplacian eigenvalues. A non-periodic
    direction uses the existing homogeneous-Neumann DCT-II operator.
    """

    def __init__(self, mesh, periodic_x: bool, periodic_y: bool):
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.periodic_x = bool(periodic_x)
        self.periodic_y = bool(periodic_y)

        kx = np.arange(self.Nx)
        ky = np.arange(self.Ny)
        if self.periodic_x:
            eig_x = 2.0 * (1.0 - np.cos(2.0 * np.pi * kx / self.Nx)) / self.dx**2
        else:
            eig_x = 2.0 * (1.0 - np.cos(np.pi * kx / self.Nx)) / self.dx**2
        if self.periodic_y:
            eig_y = 2.0 * (1.0 - np.cos(2.0 * np.pi * ky / self.Ny)) / self.dy**2
        else:
            eig_y = 2.0 * (1.0 - np.cos(np.pi * ky / self.Ny)) / self.dy**2

        self.eig_2d = eig_x[:, None] + eig_y[None, :]
        self.eig_2d[0, 0] = np.inf

    def _forward(self, rhs):
        out = rhs
        if self.periodic_x:
            out = fft(out, axis=0)
        else:
            out = dct(out, type=2, norm="ortho", axis=0)
        if self.periodic_y:
            out = fft(out, axis=1)
        else:
            out = dct(out, type=2, norm="ortho", axis=1)
        return out

    def _inverse(self, transformed):
        out = transformed
        if self.periodic_y:
            out = ifft(out, axis=1)
        else:
            out = idct(out, type=2, norm="ortho", axis=1)
        if self.periodic_x:
            out = ifft(out, axis=0)
        else:
            out = idct(out, type=2, norm="ortho", axis=0)
        return out

    def solve(self, u_star, v_star, dt):
        div = (
            (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / self.dx
            + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / self.dy
        )
        rhs = -div / dt
        rhs_hat = self._forward(rhs)
        p_hat = rhs_hat / self.eig_2d
        p_hat[0, 0] = 0.0
        p_interior = np.real(self._inverse(p_hat))

        p = np.zeros((self.Nx + 2, self.Ny + 2), dtype=np.float64)
        p[1:-1, 1:-1] = p_interior

        if self.periodic_x:
            p[0, :] = p[self.Nx, :]
            p[-1, :] = p[1, :]
        else:
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
        if self.periodic_y:
            p[:, 0] = p[:, self.Ny]
            p[:, -1] = p[:, 1]
        else:
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
        return p


_ORIGINAL_PRESSURE_FACTORY = pressure_module.create_pressure_solver


def _periodic_flags(bc):
    px_l = isinstance(bc.walls.get("left"), PeriodicWall)
    px_r = isinstance(bc.walls.get("right"), PeriodicWall)
    py_b = isinstance(bc.walls.get("bottom"), PeriodicWall)
    py_t = isinstance(bc.walls.get("top"), PeriodicWall)
    if px_l != px_r:
        raise ValueError("Periodic x boundary must be specified on both left and right walls")
    if py_b != py_t:
        raise ValueError("Periodic y boundary must be specified on both bottom and top walls")
    return px_l, py_b


def create_pressure_solver_p0(mesh, bc=None):
    if bc is not None:
        px, py = _periodic_flags(bc)
        if (px or py) and bc.has_outlet():
            raise ValueError("OutletWall cannot be combined with periodic pressure topology")
        if px or py:
            return GeneralPeriodicPressureSolver(mesh, px, py)
    return _ORIGINAL_PRESSURE_FACTORY(mesh, bc)


# Patch the module-level reference used by Solver.__init__ as well as the
# public pressure factory. This is done once at package import time.
pressure_module.PeriodicPressureSolver = GeneralPeriodicPressureSolver
pressure_module.create_pressure_solver = create_pressure_solver_p0
solver_module.create_pressure_solver = create_pressure_solver_p0


def _wrap_periodic_advection(original, periodic_x, periodic_y):
    """Add advection terms on the duplicated physical periodic faces."""

    def wrapped(u, v, dx, dy):
        adv_u, adv_v = original(u, v, dx, dy)
        Nx = u.shape[0] - 1
        Ny = u.shape[1] - 2

        if periodic_x:
            u0 = u[0, 1:-1]
            v_at_u = 0.25 * (
                v[0, 0:Ny] + v[0, 1:Ny + 1]
                + v[1, 0:Ny] + v[1, 1:Ny + 1]
            )
            if original is getattr(__import__(__name__.rsplit('.', 1)[0] + '.advection', fromlist=['central']), 'central', None):
                du_dx = (u[1, 1:-1] - u[-2, 1:-1]) / (2.0 * dx)
                du_dy = (u[0, 2:] - u[0, :-2]) / (2.0 * dy)
            else:
                du_dx = np.where(
                    u0 > 0.0,
                    (u0 - u[-2, 1:-1]) / dx,
                    (u[1, 1:-1] - u0) / dx,
                )
                du_dy = np.where(
                    v_at_u > 0.0,
                    (u0 - u[0, :-2]) / dy,
                    (u[0, 2:] - u0) / dy,
                )
            adv_u[0, 1:-1] = u0 * du_dx + v_at_u * du_dy
            adv_u[-1, 1:-1] = adv_u[0, 1:-1]

        if periodic_y:
            v0 = v[1:-1, 0]
            u_at_v = 0.25 * (
                u[0:Nx, 0] + u[1:Nx + 1, 0]
                + u[0:Nx, 1] + u[1:Nx + 1, 1]
            )
            if original is getattr(__import__(__name__.rsplit('.', 1)[0] + '.advection', fromlist=['central']), 'central', None):
                dv_dx = (v[2:, 0] - v[:-2, 0]) / (2.0 * dx)
                dv_dy = (v[1:-1, 1] - v[1:-1, -2]) / (2.0 * dy)
            else:
                dv_dx = np.where(
                    u_at_v > 0.0,
                    (v0 - v[:-2, 0]) / dx,
                    (v[2:, 0] - v0) / dx,
                )
                dv_dy = np.where(
                    v0 > 0.0,
                    (v0 - v[1:-1, -2]) / dy,
                    (v[1:-1, 1] - v0) / dy,
                )
            adv_v[1:-1, 0] = u_at_v * dv_dx + v0 * dv_dy
            adv_v[1:-1, -1] = adv_v[1:-1, 0]

        return adv_u, adv_v

    return wrapped


_ORIGINAL_EXPLICIT = diffusion_module.explicit


def explicit_p0(u, v, adv_u, adv_v, dx, dy, dt, nu, bc, Nx, Ny, u_out=None, v_out=None):
    """Explicit predictor with periodic face diffusion updates."""
    u_star, v_star = _ORIGINAL_EXPLICIT(
        u, v, adv_u, adv_v, dx, dy, dt, nu, bc, Nx, Ny, u_out=u_out, v_out=v_out
    )
    px, py = _periodic_flags(bc)
    dx2, dy2 = dx * dx, dy * dy

    if py:
        lap_v0 = (
            (v[2:, 0] - 2.0 * v[1:-1, 0] + v[:-2, 0]) / dx2
            + (v[1:-1, 1] - 2.0 * v[1:-1, 0] + v[1:-1, -2]) / dy2
        )
        v_star[1:-1, 0] = v[1:-1, 0] + dt * (
            -adv_v[1:-1, 0] + nu * lap_v0
        )
        v_star[1:-1, -1] = v_star[1:-1, 0]

    if px:
        lap_u0 = (
            (u[1, 1:-1] - 2.0 * u[0, 1:-1] + u[-2, 1:-1]) / dx2
            + (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]) / dy2
        )
        u_star[0, 1:-1] = u[0, 1:-1] + dt * (
            -adv_u[0, 1:-1] + nu * lap_u0
        )
        u_star[-1, 1:-1] = u_star[0, 1:-1]

    return u_star, v_star


diffusion_module.explicit = explicit_p0


_BaseSolver = solver_module.Solver


class P0Solver(_BaseSolver):
    """Solver with the v0.3.2 P0 numerical-correctness fixes."""

    def __init__(self, *args, **kwargs):
        boundary_config = kwargs.get("boundary_config")
        diffusion_scheme = kwargs.get("diffusion_scheme", "crank_nicolson")

        if boundary_config is not None:
            px, py = _periodic_flags(boundary_config)
            if diffusion_scheme == "crank_nicolson":
                if any(isinstance(w, (InletWall, OutletWall)) for w in boundary_config.walls.values()):
                    raise ValueError(
                        "Crank-Nicolson is not currently valid for InletWall/OutletWall "
                        "normal-velocity boundaries. Use diffusion_scheme='explicit' "
                        "until the mixed-BC implicit operator is implemented."
                    )
        else:
            px = py = False

        super().__init__(*args, **kwargs)
        self._periodic_x = px
        self._periodic_y = py
        if px or py:
            self._advection_fn = _wrap_periodic_advection(
                self._advection_fn, px, py
            )

    def step(self):
        """Advance one step with periodic-face predictor/correction updates."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt

        self.bc.apply(self.u, self.v, Nx, Ny)
        adv_u, adv_v = self._advection_fn(self.u, self.v, dx, dy)

        if self._diffusion is not None:
            u_star, v_star = self._diffusion.solve(
                self.u, self.v, adv_u, adv_v,
                u_out=self._u_star, v_out=self._v_star,
            )
        else:
            u_star, v_star = explicit_p0(
                self.u, self.v, adv_u, adv_v, dx, dy, dt,
                self.nu, self.bc, Nx, Ny,
                u_out=self._u_star, v_out=self._v_star,
            )

        if self._body_force_fn is not None:
            fu, fv = self._body_force_fn(self.u, self.v, self.time)
            u_star += dt * fu
            v_star += dt * fv
        else:
            fu = fv = 0.0

        dx2, dy2 = dx * dx, dy * dy
        if self._periodic_x:
            lap_u0 = (
                (self.u[1, 1:-1] - 2.0 * self.u[0, 1:-1] + self.u[-2, 1:-1]) / dx2
                + (self.u[0, 2:] - 2.0 * self.u[0, 1:-1] + self.u[0, :-2]) / dy2
            )
            force_u0 = fu[0, 1:-1] if not np.isscalar(fu) else 0.0
            u_star[0, 1:-1] = self.u[0, 1:-1] + dt * (
                -adv_u[0, 1:-1] + self.nu * lap_u0 + force_u0
            )
            u_star[-1, 1:-1] = u_star[0, 1:-1]

        if self._periodic_y:
            lap_v0 = (
                (self.v[2:, 0] - 2.0 * self.v[1:-1, 0] + self.v[:-2, 0]) / dx2
                + (self.v[1:-1, 1] - 2.0 * self.v[1:-1, 0] + self.v[1:-1, -2]) / dy2
            )
            force_v0 = fv[1:-1, 0] if not np.isscalar(fv) else 0.0
            v_star[1:-1, 0] = self.v[1:-1, 0] + dt * (
                -adv_v[1:-1, 0] + self.nu * lap_v0 + force_v0
            )
            v_star[1:-1, -1] = v_star[1:-1, 0]

        self.p[:] = self._pressure.solve(u_star, v_star, dt)

        grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dx
        grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dy
        self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
        self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y

        if self._periodic_x:
            grad_px_per = (self.p[1, 1:-1] - self.p[-2, 1:-1]) / dx
            self.u[0, 1:-1] = u_star[0, 1:-1] - dt * grad_px_per
            self.u[-1, 1:-1] = self.u[0, 1:-1]

        if self._periodic_y:
            grad_py_per = (self.p[1:-1, 1] - self.p[1:-1, -2]) / dy
            self.v[1:-1, 0] = v_star[1:-1, 0] - dt * grad_py_per
            self.v[1:-1, -1] = self.v[1:-1, 0]

        self.bc.apply(self.u, self.v, Nx, Ny)
        self.time += dt


# Make both the package-level import and direct submodule import resolve to
# the fixed implementation after the package is initialized.
solver_module.Solver = P0Solver