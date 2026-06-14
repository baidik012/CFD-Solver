"""Channel (Poiseuille) flow example — inlet/outlet variant.

Usage:
    python -m examples.channel_flow.run
    python -m examples.channel_flow.run --variant inlet
    python -m examples.channel_flow.run --variant body-force
    python -m examples.channel_flow.run --output /path/to/result.png
"""

import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import (
    BoundaryConditions, NoSlipWall, InletWall, OutletWall,
)
from cfd_solver.solver.viz import save_contour
from cfd_solver.validation import (
    extract_profile, compute_l2_error, compute_linf_error, print_error_report,
)


def parabolic_profile(y, H, U_max=1.0):
    """Analytical Poiseuille profile."""
    return 4.0 * U_max * y * (H - y) / (H ** 2)


def run_inlet_outlet(cfg, output_path):
    """Run channel flow with inlet/outlet BCs."""
    geo = cfg["geometry"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    sim_time = cfg.get("simulation_time", 10.0)
    U_max = 1.0

    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=U_max),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    solver = Solver(
        grid_size=(geo["Nx"], geo["Ny"]), nu=nu, dt=dt,
        Lx=geo["Lx"], Ly=geo["Ly"],
        boundary_config=bc, force=True,
    )
    solver.solve(simulation_time=sim_time, verbose=True)

    y, u_num = extract_profile(solver, direction="u", axis="y")
    H = solver.Ly
    u_exact = parabolic_profile(y, H, U_max)

    l2 = compute_l2_error(u_num, u_exact)
    linf = compute_linf_error(u_num, u_exact)

    save_contour(solver.mesh, solver.u, solver.v, solver.p, output_path)

    print_error_report(
        "Channel Flow (Inlet/Outlet)",
        l2=l2, linf=linf,
        divergence=solver.max_divergence(),
        grid=f"{solver.Nx}x{solver.Ny}",
        extra={"Time": f"{solver.time:.1f}s", "Output": output_path},
    )
    return solver


def run_body_force(cfg, output_path):
    """Run channel flow with body force (all no-slip walls, closed box)."""
    geo = cfg["geometry"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    sim_time = cfg.get("simulation_time", 5.0)
    f_val = cfg.get("body_force", {}).get("u", 8.0)

    bc = BoundaryConditions(
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(u=0.0),
        right=NoSlipWall(u=0.0),
    )
    solver = Solver(
        grid_size=(geo["Nx"], geo["Ny"]), nu=nu, dt=dt,
        Lx=geo["Lx"], Ly=geo["Ly"],
        lid_speed=0.0, boundary_config=bc,
        body_force=lambda u, v, t: (np.full_like(u, f_val), np.zeros_like(v)),
    )
    solver.solve(simulation_time=sim_time, verbose=True)

    dp_dx = np.mean((solver.p[2:-1, 1:-1] - solver.p[1:-2, 1:-1]) / solver.dx)

    save_contour(solver.mesh, solver.u, solver.v, solver.p, output_path)

    print_error_report(
        "Channel Flow (Body Force, Closed Box)",
        l2=0.0, linf=0.0,
        divergence=solver.max_divergence(),
        grid=f"{solver.Nx}x{solver.Ny}",
        extra={
            "dp/dx": f"{dp_dx:.6f} (target: {-f_val:.1f})",
            "Mean u": f"{np.mean(solver.u[1:-1, 1:-1]):.6f} (target: ~0)",
            "Time": f"{solver.time:.1f}s",
            "Output": output_path,
        },
    )
    return solver


def main():
    parser = argparse.ArgumentParser(description="Channel flow")
    parser.add_argument("--variant", choices=["inlet", "body-force"], default="inlet")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config YAML (overrides --variant)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.config:
        config_path = args.config
    elif args.variant == "inlet":
        config_path = os.path.join(script_dir, "config_inlet.yaml")
    else:
        config_path = os.path.join(script_dir, "config_body_force.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out = args.output or os.path.join(script_dir, "..", "..", "output", "channel_flow", f"result_{args.variant}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    if args.variant == "inlet":
        run_inlet_outlet(cfg, out)
    else:
        run_body_force(cfg, out)


if __name__ == "__main__":
    main()
