"""Validate Taylor-Green vortex against exact analytical solution.

Compares both u and v velocity components against the known decaying vortex
solution. Also reports kinetic energy decay.

Usage:
    python -m examples.taylor_green.validate
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, FreeSlipWall, PeriodicWall
from cfd_solver.validation import (
    compute_l2_error, compute_linf_error, print_error_report,
)
from examples.taylor_green.run import taylor_green_ic


def taylor_green_analytical(mesh, t, nu, U0=1.0):
    """Compute exact u and v at time t.

    Parameters
    ----------
    mesh : Mesh
    t : float
        Current simulation time.
    nu : float
        Kinematic viscosity.
    U0 : float
        Initial velocity amplitude.

    Returns
    -------
    u_exact : ndarray, shape (Nx+1, Ny+2)
    v_exact : ndarray, shape (Nx+2, Ny+1)
    """
    Lx, Ly = mesh.Lx, mesh.Ly
    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    d = nu * (kx**2 + ky**2)

    Xf, Yf = mesh.u_face_grid()
    u_exact = np.zeros((mesh.Nx + 1, mesh.Ny + 2))
    u_exact[:, 1:-1] = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf) * np.exp(-d * t)

    Xv, Yv = mesh.v_face_grid()
    v_exact = np.zeros((mesh.Nx + 2, mesh.Ny + 1))
    v_exact[1:-1, :] = U0 * np.cos(kx * Xv) * np.sin(ky * Yv) * np.exp(-d * t)

    return u_exact, v_exact


def kinetic_energy(solver):
    """Compute domain-averaged kinetic energy 0.5*(u^2+v^2) at cell centers."""
    # u at cell centers: average of left and right faces
    u_cc = 0.5 * (solver.u[:-1, 1:-1] + solver.u[1:, 1:-1])
    # v at cell centers: average of bottom and top faces
    v_cc = 0.5 * (solver.v[1:-1, :-1] + solver.v[1:-1, 1:])
    return 0.5 * np.mean(u_cc**2 + v_cc**2)


def validate():
    Lx = 2.0 * np.pi
    Ly = 2.0 * np.pi
    nu = 0.01
    dt = 0.001
    sim_time = 2.0

    bc = BoundaryConditions(
        top=FreeSlipWall(u=0.0),
        bottom=FreeSlipWall(u=0.0),
        left=PeriodicWall(),
        right=PeriodicWall(),
    )
    s = Solver(
        grid_size=(64, 64),
        nu=nu,
        dt=dt,
        Lx=Lx,
        Ly=Ly,
        lid_speed=0.0,
        smooth_lid=False,
        boundary_config=bc,
        initial_condition=taylor_green_ic,
        force=True,
    )
    s.solve(simulation_time=sim_time, verbose=True)

    u_exact, v_exact = taylor_green_analytical(s.mesh, s.time, nu)

    # Compare interior cells only (strip ghost cells)
    u_num = s.u[:, 1:-1]
    v_num = s.v[1:-1, :]

    u_l2 = compute_l2_error(u_num, u_exact[:, 1:-1])
    u_linf = compute_linf_error(u_num, u_exact[:, 1:-1])
    v_l2 = compute_l2_error(v_num, v_exact[1:-1, :])
    v_linf = compute_linf_error(v_num, v_exact[1:-1, :])

    ke = kinetic_energy(s)
    d = nu * ((2 * np.pi / Lx) ** 2 + (2 * np.pi / Ly) ** 2)
    ke_exact = 0.25 * np.exp(-2 * d * s.time)

    print_error_report(
        "Taylor-Green Vortex (u-component)",
        l2=u_l2,
        linf=u_linf,
        divergence=s.max_divergence(),
        grid=f"{s.Nx}x{s.Ny}",
        extra={
            "Time": f"{s.time:.3f}s",
            "KE": f"{ke:.6e}",
            "KE_exact": f"{ke_exact:.6e}",
        },
    )
    print_error_report(
        "Taylor-Green Vortex (v-component)",
        l2=v_l2,
        linf=v_linf,
    )
    return u_l2


if __name__ == "__main__":
    validate()
