"""Validate Couette flow against analytical Fourier series solution.

Usage:
    python -m examples.couette.validate
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, NoSlipWall
from cfd_solver.validation import (
    extract_profile, compute_l2_error, compute_linf_error, print_error_report,
)
from examples.couette.run import couette_analytical_transient


def validate():
    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0), bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(u=0.0), right=NoSlipWall(u=0.0),
    )
    s = Solver(grid_size=(32, 64), nu=0.01, dt=0.001, Lx=1.0, Ly=1.0,
               boundary_config=bc)
    s.solve(simulation_time=5.0, verbose=True)

    y, u_num = extract_profile(s, direction="u", axis="y")
    u_exact = couette_analytical_transient(y, s.time, s.nu)

    l2 = compute_l2_error(u_num, u_exact)
    linf = compute_linf_error(u_num, u_exact)

    print_error_report(
        "Couette Flow Validation",
        l2=l2, linf=linf,
        divergence=s.max_divergence(),
        grid=f"{s.Nx}x{s.Ny}",
        extra={"Time": f"{s.time:.2f}s"},
    )
    return l2


if __name__ == "__main__":
    validate()
