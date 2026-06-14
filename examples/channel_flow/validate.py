"""Validate channel (Poiseuille) flow against analytical solution.

Compares numerical profiles at multiple x-locations to verify fully-developed flow.

Usage:
    python -m examples.channel_flow.validate
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
    extract_profile, compute_l2_error, compute_linf_error, print_error_report,
)


def parabolic_profile(y, H, U_max=1.0):
    return 4.0 * U_max * y * (H - y) / (H ** 2)


def validate():
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(
        grid_size=(128, 32), nu=0.01, dt=0.0005,
        Lx=10.0, Ly=1.0, boundary_config=bc, force=True,
    )
    s.solve(simulation_time=10.0, verbose=True)

    H = s.Ly
    y, u_mid = extract_profile(s, direction="u", axis="y", position=s.Lx / 2)
    u_exact = parabolic_profile(y, H)

    l2 = compute_l2_error(u_mid, u_exact)
    linf = compute_linf_error(u_mid, u_exact)

    y_left, u_left = extract_profile(s, direction="u", axis="y", position=2.0)
    y_right, u_right = extract_profile(s, direction="u", axis="y", position=8.0)
    collapse = compute_linf_error(u_left, u_right)

    print_error_report(
        "Channel Flow Validation",
        l2=l2, linf=linf,
        divergence=s.max_divergence(),
        grid=f"{s.Nx}x{s.Ny}",
        extra={
            "Profile collapse (x=2 vs x=8)": f"{collapse:.6e}",
            "Centerline u": f"{u_mid[s.Ny // 2]:.4f} (exact: {u_exact[s.Ny // 2]:.4f})",
        },
    )
    return l2


if __name__ == "__main__":
    validate()
