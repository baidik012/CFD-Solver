"""Validate lid-driven cavity against Ghia et al. (1982) benchmark.

Uses the shared cfd_solver.validation module for error computation.
The standalone run_ghia_validation.py script is preserved for backward compatibility.

Usage:
    python -m examples.cavity.validate [Re]

    Re can be 100 (default), 400, or 1000.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.validation import compute_l2_error, compute_linf_error, print_error_report

# Import Ghia reference data and extraction utilities from the existing script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ghia_validation import (
    GHIA_DATA, GHIA_Y, GHIA_X, SOLVER_CONFIG,
    extract_u_profile, extract_v_profile,
)


def validate(Re=100):
    """Run Ghia validation for a given Reynolds number.

    Parameters
    ----------
    Re : int
        Reynolds number (100, 400, or 1000).

    Returns
    -------
    u_l2 : float
        L2 error of u-profile against Ghia data.
    """
    if Re not in GHIA_DATA:
        print(f"Re={Re} not available. Choose from: {list(GHIA_DATA.keys())}")
        sys.exit(1)

    cfg = SOLVER_CONFIG[Re]
    grid_size = cfg["grid"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    steps = cfg["steps"]
    advection = cfg["advection"]

    print(f"\nCavity validation: Re={Re}, grid={grid_size}, nu={nu}, dt={dt}, "
          f"steps={steps}, advection={advection}")

    solver = Solver(
        grid_size=grid_size,
        nu=nu,
        dt=dt,
        lid_speed=1.0,
        smooth_lid=False,
        advection_scheme=advection,
    )
    solver.solve(steps, verbose=True)

    # Extract profiles using existing functions
    y_solver, u_solver = extract_u_profile(solver, x_probe=0.5)
    x_solver, v_solver = extract_v_profile(solver, y_probe=0.5)

    # Interpolate onto Ghia stations
    u_at_ghia = np.interp(GHIA_Y, y_solver, u_solver)
    v_at_ghia = np.interp(GHIA_X, x_solver, v_solver)

    ghia_u = GHIA_DATA[Re]["u"]
    ghia_v = GHIA_DATA[Re]["v"]

    u_l2 = compute_l2_error(u_at_ghia, ghia_u)
    u_linf = compute_linf_error(u_at_ghia, ghia_u)
    v_l2 = compute_l2_error(v_at_ghia, ghia_v)
    v_linf = compute_linf_error(v_at_ghia, ghia_v)

    print_error_report(
        f"Cavity (Re={Re}) — u-profile along x=0.5",
        l2=u_l2,
        linf=u_linf,
        divergence=solver.max_divergence(),
        grid=f"{grid_size[0]}x{grid_size[1]}",
        extra={"Re": str(Re), "Advection": advection},
    )
    print_error_report(
        f"Cavity (Re={Re}) — v-profile along y=0.5",
        l2=v_l2,
        linf=v_linf,
    )
    return u_l2


if __name__ == "__main__":
    Re = 100
    if len(sys.argv) > 1:
        try:
            Re = int(sys.argv[1])
        except ValueError:
            print("Usage: python -m examples.cavity.validate [Re]")
            print("  Re can be 100, 400, or 1000")
            sys.exit(1)
    validate(Re)
