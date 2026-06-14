"""Grid convergence study for lid-driven cavity (Ghia validation).

Runs cavity simulation on grids from 32x32 to 128x128 at Re=100
and measures L2 error against Ghia et al. (1982) benchmark data.

Usage:
    python -m examples.cavity.convergence
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.validation import (
    compute_l2_error,
    run_grid_convergence,
    compute_convergence_rate,
    print_convergence_table,
    save_convergence_plot,
)

# Import Ghia reference data and extraction utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ghia_validation import (
    GHIA_DATA, GHIA_Y,
    SOLVER_CONFIG, extract_u_profile,
)


def run_cavity(nx, ny):
    """Run cavity flow at Re=100 on grid (nx, ny)."""
    cfg = SOLVER_CONFIG[100]
    # Scale steps inversely with grid size (rough estimate for convergence)
    # Use enough steps to reach near-steady state
    steps = max(5000, int(cfg["steps"] * (nx / cfg["grid"][0]) ** 2))

    solver = Solver(
        grid_size=(nx, ny),
        nu=cfg["nu"],
        dt=cfg["dt"],
        lid_speed=1.0,
        smooth_lid=False,
        advection_scheme=cfg["advection"],
    )
    solver.solve(steps, verbose=False)
    return solver


def compute_error(solver):
    """Compute L2 error of u-profile against Ghia data (Re=100)."""
    ghia_u = GHIA_DATA[100]["u"]

    y_solver, u_solver = extract_u_profile(solver, x_probe=0.5)
    u_at_ghia = np.interp(GHIA_Y, y_solver, u_solver)

    return compute_l2_error(u_at_ghia, ghia_u)


def main():
    grids = [(32, 32), (64, 64), (128, 128)]

    print("Lid-Driven Cavity — Grid Convergence Study (Re=100)")
    print("=" * 50)

    errors = run_grid_convergence(run_cavity, grids, compute_error)
    rates = compute_convergence_rate(errors, grids)

    print_convergence_table(grids, errors, rates, name="Cavity Convergence (Re=100)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "..", "..", "images")
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, "cavity_convergence.png")

    save_convergence_plot(grids, errors, rates, plot_path,
                          name="Lid-Driven Cavity Grid Convergence (Re=100)")


if __name__ == "__main__":
    main()
