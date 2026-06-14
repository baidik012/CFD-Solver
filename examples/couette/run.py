"""Couette flow example — flow between parallel plates.

Top wall moves at U=1.0, bottom wall at rest.
Analytical steady-state: u(y) = U * y / H (linear profile)
Analytical transient: Fourier series from rest.

Usage:
    python -m examples.couette.run
    python -m examples.couette.run --output /path/to/result.png
"""

import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, NoSlipWall, PeriodicWall
from cfd_solver.solver.viz import save_contour
from cfd_solver.validation import (
    extract_profile, compute_l2_error, compute_linf_error, print_error_report,
)


def couette_analytical_steady(y, U=1.0, H=1.0):
    """Steady-state: linear profile u(y) = U * y / H."""
    return U * y / H


def couette_analytical_transient(y, t, nu, U=1.0, H=1.0, n_terms=50):
    """Transient solution from rest via Fourier series.

    u(y, t) = U*y/H + sum_{n=1}^inf [2U/(n*pi)] * (-1)^n
              * sin(n*pi*y/H) * exp(-n^2 * pi^2 * nu * t / H^2)
    """
    result = U * y / H
    for n in range(1, n_terms + 1):
        lam = n * np.pi / H
        result += (2.0 * U / (n * np.pi)) * ((-1) ** n) * np.sin(lam * y) * np.exp(-lam**2 * nu * t)
    return result


def run_couette(cfg, output_path):
    """Run Couette flow simulation."""
    geo = cfg["geometry"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    sim_time = cfg.get("simulation_time", 5.0)

    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0),
        bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(),
        right=PeriodicWall(),
    )
    solver = Solver(
        grid_size=(geo["Nx"], geo["Ny"]), nu=nu, dt=dt,
        Lx=geo["Lx"], Ly=geo["Ly"],
        boundary_config=bc,
    )
    solver.solve(simulation_time=sim_time, verbose=True)

    y, u_num = extract_profile(solver, direction="u", axis="y")
    H = solver.Ly
    u_steady = couette_analytical_steady(y)
    u_transient = couette_analytical_transient(y, solver.time, nu)

    l2_steady = compute_l2_error(u_num, u_steady)
    l2_transient = compute_l2_error(u_num, u_transient)
    linf = compute_linf_error(u_num, u_transient)

    save_contour(solver.mesh, solver.u, solver.v, solver.p, output_path)

    print_error_report(
        "Couette Flow",
        l2=l2_transient, linf=linf,
        divergence=solver.max_divergence(),
        grid=f"{solver.Nx}x{solver.Ny}",
        extra={
            "Time": f"{solver.time:.2f}s",
            "L2 vs steady": f"{l2_steady:.6e}",
            "Centerline u": f"{u_num[solver.Ny // 2]:.4f} (steady: {u_steady[solver.Ny // 2]:.4f})",
            "Output": output_path,
        },
    )
    return solver


def main():
    parser = argparse.ArgumentParser(description="Couette flow")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out = args.output or os.path.join(script_dir, "..", "..", "output", "couette", "result.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run_couette(cfg, out)


if __name__ == "__main__":
    main()
