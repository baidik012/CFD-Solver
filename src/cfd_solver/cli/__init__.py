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
import numpy as np
from cfd_solver.solver import Solver
from cfd_solver.solver.bc import (
    BoundaryConditions, NoSlipWall, FreeSlipWall,
    InletWall, OutletWall, PeriodicWall, WALL_TYPE_REGISTRY,
)
from cfd_solver.solver.validate import validate_config, WALL_TYPE_VALUES
from cfd_solver.solver.viz import save_contour, save_quiver
from cfd_solver.utils import handle_error


# Wall-type lookup derived from the single source of truth in bc.py.
# (Audit finding P0-1 / P0-3 — previously this was a hand-maintained
# duplicate of the same mapping.)
_WALL_TYPE_MAP = WALL_TYPE_REGISTRY


def _parse_boundary_config(bc_cfg):
    """Parse a boundary config dict into a :class:`BoundaryConditions`.

    Supports both the legacy ``top`` / ``other`` format and the new
    per-wall format with ``type`` keys.

    Parameters
    ----------
    bc_cfg : dict
        The ``boundary`` section of the YAML config.

    Returns
    -------
    BoundaryConditions
    """
    if bc_cfg is None:
        bc_cfg = {}

    smooth_lid = bc_cfg.get("smooth_lid", True)

    # Detect new per-wall format: any of left/right/bottom has a 'type' key
    has_new_format = any(
        isinstance(bc_cfg.get(w), dict) and "type" in bc_cfg.get(w, {})
        for w in ("left", "right", "bottom")
    )

    if has_new_format:
        walls = {}
        for wall_name in ("left", "right", "top", "bottom"):
            wall_cfg = bc_cfg.get(wall_name)
            if not isinstance(wall_cfg, dict) or "type" not in wall_cfg:
                walls[wall_name] = NoSlipWall(u=0.0, v=0.0)
                continue
            wtype = wall_cfg["type"]
            cls = _WALL_TYPE_MAP.get(wtype, NoSlipWall)
            if wtype == "wall":
                walls[wall_name] = NoSlipWall(
                    u=wall_cfg.get("u", 0.0),
                    v=wall_cfg.get("v", 0.0),
                )
            elif wtype == "free_slip":
                walls[wall_name] = FreeSlipWall(
                    u=wall_cfg.get("u", 0.0),
                    v=wall_cfg.get("v", 0.0),
                )
            elif wtype == "inlet":
                walls[wall_name] = InletWall(
                    profile=wall_cfg.get("profile", "uniform"),
                    U_max=wall_cfg.get("U_max", 1.0),
                )
            elif wtype == "outlet":
                walls[wall_name] = OutletWall(
                    method=wall_cfg.get("method", "zero_gradient"),
                )
            elif wtype == "periodic":
                walls[wall_name] = PeriodicWall()
            else:
                walls[wall_name] = NoSlipWall(u=0.0, v=0.0)

        return BoundaryConditions(
            top=walls['top'],
            bottom=walls['bottom'],
            left=walls['left'],
            right=walls['right'],
            smooth_lid=smooth_lid,
        )

    # Legacy format: top.u + smooth_lid
    top = bc_cfg.get("top", {})
    top_u = top.get("u", 1.0)
    return BoundaryConditions(top=top_u, smooth_lid=smooth_lid)


def _parse_body_force(bf_cfg):
    if bf_cfg is None:
        return None
    fu_val = float(bf_cfg.get('u', 0.0))
    fv_val = float(bf_cfg.get('v', 0.0))
    if fu_val == 0.0 and fv_val == 0.0:
        return None
    def bf(u, v, t):
        return (np.full_like(u, fu_val), np.full_like(v, fv_val))
    return bf


def _parse_convergence(conv_cfg):
    if conv_cfg is None:
        return None, None
    tol = conv_cfg.get('tol')
    window = conv_cfg.get('window', 100)
    return tol, window


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
        smooth = bc_cfg.get("smooth_lid", True)
        advection_scheme = cfg.get("advection_scheme", "upwind")
        diffusion_scheme = cfg.get("diffusion_scheme", "crank_nicolson")

        # Parse body force
        body_force = _parse_body_force(cfg.get("body_force"))

        # Parse boundary conditions (supports both legacy and new per-wall format)
        bc = _parse_boundary_config(bc_cfg)

        # Warn about validated-but-unsupported fields rather than silently
        # ignoring them, so configs are never silently misinterpreted.
        ignored = []
        top_cfg = bc_cfg.get("top", {})
        if isinstance(top_cfg, dict) and top_cfg.get("v") not in (None, 0, 0.0):
            ignored.append("boundary.top.v")
        other = bc_cfg.get("other", {})
        if isinstance(other, dict) and any(
            other.get(k) not in (None, 0, 0.0) for k in ("u", "v")
        ):
            ignored.append("boundary.other")
        if ignored:
            print(
                f"  [warning] Config fields not supported by the solver and "
                f"ignored: {', '.join(ignored)}",
                file=sys.stderr,
            )

        # Initialize the solver
        solver = Solver(
            grid_size=(Nx, Ny), nu=nu, dt=dt,
            Lx=Lx, Ly=Ly,
            advection_scheme=advection_scheme,
            diffusion_scheme=diffusion_scheme,
            boundary_config=bc,
            body_force=body_force,
        )

    # Simulation loop — prefer simulation_time over steps
    sim_time = cfg.get("simulation_time")
    steps = cfg.get("steps")
    conv_tol, conv_window = _parse_convergence(cfg.get("convergence"))
    if sim_time is None and steps is None:
        # Auto-compute from flow parameters: max(10, 0.1*Re) convective time units
        nu_val = cfg.get("nu", 0.01)
        bc_cfg_auto = cfg.get("boundary", {})
        top_cfg = bc_cfg_auto.get("top", {})
        top_u = top_cfg.get("u", 1.0) if isinstance(top_cfg, dict) else 1.0
        Re = abs(top_u) / max(nu_val, 1e-10)
        t_conv = 1.0 / max(abs(top_u), 1e-10)
        sim_time = t_conv * min(max(10.0, 0.1 * Re), 200.0)
        print(f"  [info] No simulation_time or steps set. Auto-selecting {sim_time:.1f}s "
              f"(Re={Re:.0f})", file=sys.stderr)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if sim_time is not None:
        ok = solver.solve(simulation_time=sim_time, verbose=True,
                          convergence_tol=conv_tol, convergence_window=conv_window)
    else:
        ok = solver.solve(steps, verbose=True,
                          convergence_tol=conv_tol, convergence_window=conv_window)

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


def run_example(args):
    """Run a bundled example by name."""
    import importlib
    name = args.name
    script_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", name)
    script_dir = os.path.normpath(script_dir)

    if not os.path.isdir(script_dir):
        print(f"Error: example '{name}' not found at {script_dir}", file=sys.stderr)
        raise SystemExit(1)

    run_py = os.path.join(script_dir, "run.py")
    if not os.path.exists(run_py):
        print(f"Error: no run.py in example '{name}'", file=sys.stderr)
        raise SystemExit(1)

    cmd_args = []
    if args.output:
        cmd_args.extend(["--output", args.output])
    if args.variant:
        cmd_args.extend(["--variant", args.variant])

    sys.argv = [run_py] + cmd_args
    import runpy
    runpy.run_path(run_py, run_name="__main__")


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

    # 'run-example' subcommand
    example_parser = sub.add_parser("run-example", help="Run a bundled example")
    example_parser.add_argument(
        "name", help="Example name (cavity, channel_flow, couette, taylor_green)",
    )
    example_parser.add_argument(
        "--output", "-o", default=None,
        help="Output plot path (default: output/<name>/result.png)",
    )
    example_parser.add_argument(
        "--variant", default=None,
        help="Example variant (e.g. 'inlet' or 'body-force' for channel_flow)",
    )
    example_parser.set_defaults(func=run_example)

    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        handle_error(exc)


if __name__ == "__main__":
    main()
