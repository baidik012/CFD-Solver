"""Command-line interface."""

import argparse
import os
import sys
import yaml
from cfd_solver.solver import Solver
from cfd_solver.solver.validate import validate_config
from cfd_solver.solver.viz import save_contour, save_quiver


_FRIENDLY_ERRORS = {
    ModuleNotFoundError: (
        "Missing required package. Run:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install -e ."
    ),
    FileNotFoundError: "File not found. Check that the path is correct.",
    PermissionError: "Permission denied. Check file/folder permissions.",
    yaml.YAMLError: "Config file is invalid YAML. Check the syntax.",
    KeyError: "Config file is missing a required field. Check the YAML keys.",
    ValueError: "Invalid parameter value in config file.",
    MemoryError: "Not enough memory. Try a smaller grid size.",
}


def _handle_error(exc):
    """Print a user-friendly message and exit."""
    print("\n" + "=" * 50, file=sys.stderr)
    print("  ERROR", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    msg = _FRIENDLY_ERRORS.get(type(exc))
    if msg:
        print(f"  {msg}", file=sys.stderr)
    else:
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)

    print("=" * 50 + "\n", file=sys.stderr)
    raise SystemExit(1)


def run(args):
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: checkpoint not found: {args.resume}")
            raise SystemExit(1)
        solver = Solver.from_checkpoint(args.resume)
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        if not os.path.exists(args.config):
            print(f"Error: config file not found: {args.config}")
            raise SystemExit(1)

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        errors = validate_config(cfg)
        if errors:
            print("Config validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            raise SystemExit(1)

        geo = cfg["geometry"]
        Lx, Ly = geo["Lx"], geo["Ly"]
        Nx, Ny = geo["Nx"], geo["Ny"]
        nu = cfg["nu"]
        dt = cfg["dt"]

        bc_cfg = cfg.get("boundary", {})
        top = bc_cfg.get("top", {})
        top_u = top.get("u", 1.0)
        smooth = bc_cfg.get("smooth_lid", True)

        solver = Solver(
            grid_size=(Nx, Ny), nu=nu, dt=dt, lid_speed=top_u,
            smooth_lid=smooth, Lx=Lx, Ly=Ly,
        )

    steps = cfg.get("steps", 200)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    solver.solve(steps, verbose=True)
    save_contour(solver.mesh, solver.u, solver.v, solver.p, args.output)
    solver.checkpoint(args.output.rsplit(".", 1)[0] + ".npz")

    print(f"Saved to {args.output}")


def main():
    from cfd_solver import __version__
    from cfd_solver.version_check import check_for_updates
    check_for_updates()

    parser = argparse.ArgumentParser(description=f"CFD Solver v{__version__}")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(required=True)

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
        _handle_error(exc)


if __name__ == "__main__":
    main()
