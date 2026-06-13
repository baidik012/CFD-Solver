"""
Command-line interface for the CFD Solver.

This subpackage implements the 'cfd-solver' command-line tool, allowing users
to run simulations using configuration files, resume from checkpoints,
and save visualizations.
"""

import argparse
import os
import sys
import yaml
from cfd_solver.solver import Solver
from cfd_solver.solver.validate import validate_config
from cfd_solver.solver.viz import save_contour, save_quiver
from cfd_solver.utils import handle_error


def run(args):
    """
    Execute the simulation based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments.
    """
    if args.resume:
        # Resume from an existing checkpoint
        if not os.path.exists(args.resume):
            print(f"Error: checkpoint not found: {args.resume}")
            raise SystemExit(1)
        solver = Solver.from_checkpoint(args.resume)
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        # Start a new simulation from a config file
        if not os.path.exists(args.config):
            print(f"Error: config file not found: {args.config}")
            raise SystemExit(1)

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # Validate the configuration schema
        errors = validate_config(cfg)
        if errors:
            print("Config validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            raise SystemExit(1)

        # Extract parameters from config
        geo = cfg["geometry"]
        Lx, Ly = geo["Lx"], geo["Ly"]
        Nx, Ny = geo["Nx"], geo["Ny"]
        nu = cfg["nu"]
        dt = cfg["dt"]

        bc_cfg = cfg.get("boundary", {})
        top = bc_cfg.get("top", {})
        top_u = top.get("u", 1.0)
        smooth = bc_cfg.get("smooth_lid", True)
        advection_scheme = cfg.get("advection_scheme", "upwind")
        diffusion_scheme = cfg.get("diffusion_scheme", "crank_nicolson")

        # Warn about validated-but-unsupported fields rather than silently
        # ignoring them, so configs are never silently misinterpreted.
        ignored = []
        if top.get("v") not in (None, 0, 0.0):
            ignored.append("boundary.top.v")
        other = bc_cfg.get("other", {})
        if any(other.get(k) not in (None, 0, 0.0) for k in ("u", "v")):
            ignored.append("boundary.other")
        if ignored:
            print(
                f"  [warning] Config fields not supported by the solver and "
                f"ignored: {', '.join(ignored)}",
                file=sys.stderr,
            )

        # Initialize the solver
        solver = Solver(
            grid_size=(Nx, Ny), nu=nu, dt=dt, lid_speed=top_u,
            smooth_lid=smooth, Lx=Lx, Ly=Ly,
            advection_scheme=advection_scheme,
            diffusion_scheme=diffusion_scheme,
        )

    # Simulation loop — prefer simulation_time over steps
    sim_time = cfg.get("simulation_time")
    steps = cfg.get("steps")
    if sim_time is None and steps is None:
        # Auto-compute from flow parameters: max(10, 0.1*Re) convective time units
        nu_val = cfg.get("nu", 0.01)
        top_u = cfg.get("boundary", {}).get("top", {}).get("u", 1.0)
        Re = abs(top_u) / max(nu_val, 1e-10)
        t_conv = 1.0 / max(abs(top_u), 1e-10)
        sim_time = t_conv * min(max(10.0, 0.1 * Re), 200.0)
        print(f"  [info] No simulation_time or steps set. Auto-selecting {sim_time:.1f}s "
              f"(Re={Re:.0f})", file=sys.stderr)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if sim_time is not None:
        ok = solver.solve(simulation_time=sim_time, verbose=True)
    else:
        ok = solver.solve(steps, verbose=True)

    if not ok:
        print(
            "\nSimulation aborted due to blowup. Skipping plot and checkpoint "
            "to avoid writing NaN/Inf data.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Save results and checkpoint
    save_contour(solver.mesh, solver.u, solver.v, solver.p, args.output)
    solver.checkpoint(args.output.rsplit(".", 1)[0] + ".npz")

    print(f"Saved to {args.output}")


def main():
    """
    Main entry point for the CLI.
    
    Parses arguments, checks for updates, and dispatches to the appropriate command.
    """
    from cfd_solver import __version__
    from cfd_solver.version_check import check_for_updates
    check_for_updates()

    parser = argparse.ArgumentParser(description=f"CFD Solver v{__version__}")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(required=True)

    # 'run' subcommand
    run_parser = sub.add_parser("run", help="Run a simulation")
    run_parser.add_argument("config", help="YAML config file")
    run_parser.add_argument(
        "--output", "-o", default="output/result.png",
        help="Output plot path (default: output/result.png)",
    )
    run_parser.add_argument(
        "--resume", "-r", default=None, metavar="CHECKPOINT",
        help="Resume from a .npz checkpoint file instead of starting fresh",
    )
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        handle_error(exc)


if __name__ == "__main__":
    main()
