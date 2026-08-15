"""Couette flow example — flow between parallel plates.

Top wall moves at U=1.0, bottom wall at rest.
Analytical steady-state: u(y) = U * y / H (linear profile)
Analytical transient: Fourier series from rest.

For the periodic-x Couette setup the exact pressure field is spatially
constant (pressure gradient is zero). A non-zero pressure gradient in an
output image is therefore a validation failure, not a feature of Couette
flow.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, NoSlipWall, PeriodicWall
from cfd_solver.solver.viz import save_contour
from cfd_solver.validation import (
    extract_profile, compute_l2_error, compute_linf_error, print_error_report,
)
from cfd_solver.config_loader import load_config


def couette_analytical_steady(y, U=1.0, H=1.0):
    """Steady-state: linear profile u(y) = U * y / H."""
    return U * y / H


def couette_analytical_transient(y, t, nu, U=1.0, H=1.0, n_terms=50):
    """Transient solution from rest via Fourier series."""
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
    H = geo["Ly"]

    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0),
        bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(),
        right=PeriodicWall(),
    )
    solver = Solver(
        grid_size=(geo["Nx"], geo["Ny"]), nu=nu, dt=dt,
        Lx=geo["Lx"], Ly=geo["Ly"],
        lid_speed=0.0, smooth_lid=False,
        boundary_config=bc,
        # Periodic x currently uses explicit diffusion. State this explicitly
        # rather than relying on Solver's Crank-Nicolson fallback.
        diffusion_scheme="explicit",
    )
    solver.solve(simulation_time=sim_time, verbose=True)

    y, u_num = extract_profile(solver, direction="u", axis="y")
    u_steady = couette_analytical_steady(y, H=H)
    u_transient = couette_analytical_transient(y, solver.time, nu, H=H)

    l2_steady = compute_l2_error(u_num, u_steady)
    l2_transient = compute_l2_error(u_num, u_transient)
    linf = compute_linf_error(u_num, u_transient)

    # Pure Couette flow has no imposed pressure gradient. Pressure is defined
    # only up to an additive constant, so its spatial variation is the useful
    # diagnostic rather than its absolute value.
    p_phys = solver.p[1:-1, 1:-1]
    pressure_range = float(np.ptp(p_phys)) if p_phys.size else 0.0

    save_contour(solver.mesh, solver.u, solver.v, solver.p, output_path)

    print_error_report(
        "Couette Flow",
        l2=l2_transient, linf=linf,
        divergence=solver.max_divergence(),
        grid=f"{solver.Nx}x{solver.Ny}",
        extra={
            "Time": f"{solver.time:.2f}s",
            "L2 vs steady": f"{l2_steady:.6e}",
            "Pressure range": f"{pressure_range:.6e}",
            "Centerline u": f"{u_num[solver.Ny // 2]:.4f} (steady: {u_steady[solver.Ny // 2]:.4f})",
            "Output": output_path,
        },
    )
    return solver


def main():
    parser = argparse.ArgumentParser(description="Couette flow")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config YAML (default: examples/couette/config.yaml)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(script_dir, "config.yaml")
    cfg = load_config(config_path)

    out = args.output or os.path.join(script_dir, "..", "..", "output", "couette", "result.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run_couette(cfg, out)


if __name__ == "__main__":
    main()
