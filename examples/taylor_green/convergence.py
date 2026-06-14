"""Grid convergence study for Taylor-Green vortex.

Runs the Taylor-Green vortex simulation on grids from 16x16 to 128x128
and measures L2 error against the exact analytical solution.

Usage:
    python -m examples.taylor_green.convergence
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, FreeSlipWall, PeriodicWall
from cfd_solver.validation import (
    compute_l2_error,
    run_grid_convergence,
    compute_convergence_rate,
    print_convergence_table,
    save_convergence_plot,
)
from examples.taylor_green.run import taylor_green_ic


def run_tg(nx, ny):
    """Run Taylor-Green vortex on grid (nx, ny)."""
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
        grid_size=(nx, ny),
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
    s.solve(simulation_time=sim_time, verbose=False)
    return s


def compute_error(solver):
    """Compute L2 error of u against exact analytical solution."""
    Lx = solver.mesh.Lx
    Ly = solver.mesh.Ly
    nu = solver.nu
    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    d = nu * (kx**2 + ky**2)

    Xf, Yf = solver.mesh.u_face_grid()
    u_exact = -np.sin(kx * Xf) * np.cos(ky * Yf) * np.exp(-d * solver.time)

    return compute_l2_error(solver.u[:, 1:-1], u_exact)


def main():
    grids = [(16, 16), (32, 32), (64, 64), (128, 128)]

    print("Taylor-Green Vortex — Grid Convergence Study")
    print("=" * 50)

    errors = run_grid_convergence(run_tg, grids, compute_error)
    rates = compute_convergence_rate(errors, grids)

    print_convergence_table(grids, errors, rates, name="Taylor-Green Convergence")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "..", "..", "images")
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, "taylor_green_convergence.png")

    save_convergence_plot(grids, errors, rates, plot_path,
                          name="Taylor-Green Vortex Grid Convergence")


if __name__ == "__main__":
    main()
