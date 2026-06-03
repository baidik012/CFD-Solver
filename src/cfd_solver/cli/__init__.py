"""Command-line interface."""

import argparse
import os
import yaml
from cfd_solver.solver import Solver
from cfd_solver.solver.viz import save_contour, save_quiver


def run(args):
    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        raise SystemExit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    geo = cfg["geometry"]
    Lx, Ly = geo["Lx"], geo["Ly"]
    Nx, Ny = geo["Nx"], geo["Ny"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    steps = cfg["steps"]

    bc_cfg = cfg.get("boundary", {})
    top = bc_cfg.get("top", {})
    top_u = top.get("u", 1.0)
    smooth = bc_cfg.get("smooth_lid", True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    solver = Solver(
        grid_size=(Nx, Ny), nu=nu, dt=dt, lid_speed=top_u,
        smooth_lid=smooth, Lx=Lx, Ly=Ly,
    )
    solver.solve(steps, verbose=True)
    save_contour(solver.mesh, solver.u, solver.v, solver.p, args.output)

    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="CFD Solver")
    sub = parser.add_subparsers(required=True)

    run_parser = sub.add_parser("run", help="Run a simulation")
    run_parser.add_argument("config", help="YAML config file")
    run_parser.add_argument(
        "--output", "-o", default="output/result.png",
        help="Output plot path (default: output/result.png)",
    )
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
