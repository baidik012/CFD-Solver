"""Command-line interface."""

import argparse
import os
import yaml
from cfd_solver.solver import Grid, Solver, StaggeredSolver, BoundaryConditions
from cfd_solver.solver.viz import save_velocity_plot, save_velocity_contour


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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.solver == "staggered":
        solver = StaggeredSolver(
            Lx, Ly, Nx, Ny, nu, dt,
            u_bc={"top": top_u, "bottom": 0.0, "left": 0.0, "right": 0.0},
        )
        solver.solve(steps, verbose=True)
        save_velocity_contour(solver, args.output)
    else:
        grid = Grid(Lx, Ly, Nx, Ny)
        bc = BoundaryConditions(top_u=top_u)
        solver = Solver(grid, nu, dt, bc)
        solver.solve(steps, verbose=True)
        save_velocity_plot(grid, solver.u, solver.v, args.output)

    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="CFD Solver")
    sub = parser.add_subparsers(required=True)

    run_parser = sub.add_parser("run", help="Run a simulation")
    run_parser.add_argument("config", help="YAML config file")
    run_parser.add_argument(
        "--solver", choices=["staggered", "original"], default="staggered",
        help='Solver backend (default: staggered)',
    )
    run_parser.add_argument(
        "--output", "-o", default="output/result.png",
        help="Output plot path (default: output/result.png)",
    )
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
