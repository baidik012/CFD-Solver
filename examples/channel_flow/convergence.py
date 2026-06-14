"""Grid convergence study for channel (Poiseuille) flow.

Runs channel flow simulation on grids from 32x16 to 256x64
and measures L2 error against the analytical parabolic profile.

Usage:
    python -m examples.channel_flow.convergence
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import (
    BoundaryConditions, NoSlipWall, InletWall, OutletWall,
)
from cfd_solver.validation import (
    extract_profile,
    compute_l2_error,
    run_grid_convergence,
    compute_convergence_rate,
    print_convergence_table,
    save_convergence_plot,
)


def parabolic_profile(y, H, U_max=1.0):
    """Analytical Poiseuille profile."""
    return 4.0 * U_max * y * (H - y) / (H ** 2)


def run_channel(nx, ny):
    """Run channel flow on grid (nx, ny)."""
    nu = 0.01
    dt = 0.0005
    sim_time = 10.0

    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(
        grid_size=(nx, ny),
        nu=nu,
        dt=dt,
        Lx=10.0,
        Ly=1.0,
        boundary_config=bc,
        force=True,
    )
    s.solve(simulation_time=sim_time, verbose=False)
    return s


def compute_error(solver):
    """Compute L2 error of u-profile against analytical parabolic profile."""
    y, u_num = extract_profile(solver, direction="u", axis="y", position=solver.Lx / 2)
    u_exact = parabolic_profile(y, solver.Ly)
    return compute_l2_error(u_num, u_exact)


def main():
    grids = [(32, 16), (64, 32), (128, 32), (256, 64)]

    print("Channel Flow (Poiseuille) — Grid Convergence Study")
    print("=" * 50)

    errors = run_grid_convergence(run_channel, grids, compute_error)
    rates = compute_convergence_rate(errors, grids)

    print_convergence_table(grids, errors, rates, name="Channel Flow Convergence")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "..", "..", "images")
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, "channel_convergence.png")

    save_convergence_plot(grids, errors, rates, plot_path,
                          name="Channel Flow Grid Convergence")


if __name__ == "__main__":
    main()
