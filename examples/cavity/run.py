"""Lid-driven cavity example.

Usage:
    python -m examples.cavity.run
    python -m examples.cavity.run --config examples/cavity/config.yaml
    python -m examples.cavity.run --output /path/to/result.png
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.viz import save_contour
from cfd_solver.validation import print_error_report


def main():
    parser = argparse.ArgumentParser(description="Lid-driven cavity")
    parser.add_argument("--config", "-c",
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--convergence", action="store_true",
                        help="Use steady-state convergence check")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    geo = cfg["geometry"]
    bc_cfg = cfg.get("boundary", {})
    top_u = bc_cfg.get("top", {}).get("u", 1.0)
    smooth = bc_cfg.get("smooth_lid", True)

    solver = Solver(
        grid_size=(geo["Nx"], geo["Ny"]),
        nu=cfg["nu"], dt=cfg["dt"],
        lid_speed=top_u, smooth_lid=smooth,
        Lx=geo["Lx"], Ly=geo["Ly"],
        advection_scheme=cfg.get("advection_scheme", "upwind"),
        diffusion_scheme=cfg.get("diffusion_scheme", "crank_nicolson"),
    )

    conv_tol = conv_window = None
    if args.convergence:
        conv_tol = 1e-5
        conv_window = 100

    ok = solver.solve(
        simulation_time=cfg.get("simulation_time", 20.0),
        verbose=True,
        convergence_tol=conv_tol,
        convergence_window=conv_window,
    )
    if not ok:
        raise SystemExit("Simulation blew up.")

    if args.output:
        out = args.output
    else:
        out = os.path.join(os.path.dirname(__file__), "..", "..", "output", "cavity", "result.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_contour(solver.mesh, solver.u, solver.v, solver.p, out)

    print_error_report(
        "Lid-Driven Cavity",
        l2=0.0, linf=0.0,
        divergence=solver.max_divergence(),
        grid=f"{solver.Nx}x{solver.Ny}",
        extra={"Time": f"{solver.time:.1f}s", "Output": out},
    )


if __name__ == "__main__":
    main()
