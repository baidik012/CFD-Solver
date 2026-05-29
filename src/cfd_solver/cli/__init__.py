"""Command-line interface."""

import argparse
import yaml
from cfd_solver.solver import Grid, Solver, BoundaryConditions
from cfd_solver.solver.viz import save_velocity_plot


def run(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    geo = cfg["geometry"]
    grid = Grid(geo["Lx"], geo["Ly"], geo["Nx"], geo["Ny"])

    bc_cfg = cfg.get("boundary", {})
    bc = BoundaryConditions(
        top_u=bc_cfg.get("top", {}).get("u", 1.0),
    )

    solver = Solver(grid, cfg["nu"], cfg["dt"], bc)
    solver.solve(cfg["steps"], verbose=True)

    save_velocity_plot(grid, solver.u, solver.v, "output/result.png")
    print(f"Saved to output/result.png")


def main():
    parser = argparse.ArgumentParser(description="CFD Solver")
    sub = parser.add_subparsers(required=True)

    run_parser = sub.add_parser("run", help="Run a simulation")
    run_parser.add_argument("config", help="YAML config file")
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
