"""Grid convergence study for Couette flow.

Runs Couette flow simulation on grids from 16x32 to 128x256 and measures
L2 error against the analytical transient Fourier-series solution.

The study deliberately scales the explicit-diffusion time step with dy^2.
Therefore the observed rate is a combined spatial/temporal refinement rate;
it must not be described as a pure spatial order unless dt is independently
refined to remove temporal error.
"""

import os
import sys

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
    """Run the transient periodic-x Couette problem on one grid."""
    nu = 0.01
    Lx, Ly = 1.0, 1.0
    dy = Ly / ny
    dt = 0.8 * dy**2 / (4.0 * nu)
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
        Lx=Lx,
        Ly=Ly,
        boundary_config=bc,
        lid_speed=0.0,
        smooth_lid=False,
        diffusion_scheme="explicit",
        force=True,
    )
    s.solve(simulation_time=sim_time, verbose=False)
    return s


def compute_error(solver):
    """Compute L2 error against the exact transient solution."""
    y, u_num = extract_profile(solver, direction="u", axis="y")
    u_exact = couette_analytical_transient(
        y, solver.time, solver.nu, H=solver.Ly
    )
    return compute_l2_error(u_num, u_exact)


def main():
    grids = [(16, 32), (32, 64), (64, 128), (128, 256)]

    print("Couette Flow — Transient Grid Convergence Study")
    print("=" * 55)

    errors = run_grid_convergence(run_couette, grids, compute_error)
    rates = compute_convergence_rate(errors, grids)

    print_convergence_table(
        grids, errors, rates, name="Couette Flow Transient Convergence"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "..", "..", "images")
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, "couette_convergence.png")

    save_convergence_plot(
        grids,
        errors,
        rates,
        plot_path,
        name="Couette Flow Transient Grid Convergence",
    )


if __name__ == "__main__":
    main()
