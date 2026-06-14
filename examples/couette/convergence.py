"""Grid convergence study for Couette flow.

Runs Couette flow simulation on grids from 16x32 to 128x256
and measures L2 error against the analytical transient Fourier series solution.

Usage:
    python -m examples.couette.convergence
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, NoSlipWall, PeriodicWall
from cfd_solver.validation import (
    extract_profile,
    compute_l2_error,
    run_grid_convergence,
    compute_convergence_rate,
    print_convergence_table,
    save_convergence_plot,
)
from examples.couette.run import couette_analytical_transient


def run_couette(nx, ny):
    """Run Couette flow on grid (nx, ny)."""
    nu = 0.01
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    # Use 80% of explicit diffusion stability limit for 2D
    dt = 0.8 * min(dx**2, dy**2) / (4.0 * nu)
    sim_time = 5.0

    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0),
        bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(),
        right=PeriodicWall(),
    )
    s = Solver(
        grid_size=(nx, ny),
        nu=nu,
        dt=dt,
        Lx=1.0,
        Ly=1.0,
        boundary_config=bc,
        lid_speed=0.0,
        smooth_lid=False,
        force=True,
    )
    s.solve(simulation_time=sim_time, verbose=False)
    return s


def compute_error(solver):
    """Compute L2 error of u-profile against analytical transient solution."""
    y, u_num = extract_profile(solver, direction="u", axis="y")
    u_exact = couette_analytical_transient(y, solver.time, solver.nu)
    return compute_l2_error(u_num, u_exact)


def main():
    grids = [(16, 32), (32, 64), (64, 128), (128, 256)]

    print("Couette Flow — Grid Convergence Study")
    print("=" * 50)

    errors = run_grid_convergence(run_couette, grids, compute_error)
    rates = compute_convergence_rate(errors, grids)

    print_convergence_table(grids, errors, rates, name="Couette Flow Convergence")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "..", "..", "images")
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, "couette_convergence.png")

    save_convergence_plot(grids, errors, rates, plot_path,
                          name="Couette Flow Grid Convergence")


if __name__ == "__main__":
    main()
