"""Lid-driven cavity example (YAML-configurable).

Run: python3 examples/lid_cavity.py --config examples/lid_cavity.yaml
     python3 examples/lid_cavity.py --output output/custom.png
"""

import os
import argparse
import yaml

from cfd_solver.solver import Solver
from cfd_solver.solver.viz import save_contour


def main():
    parser = argparse.ArgumentParser(description="Run a lid-driven cavity simulation")
    parser.add_argument("--config", "-c", default="examples/lid_cavity.yaml",
                        help="YAML config file (default: examples/lid_cavity.yaml)")
    parser.add_argument("--output", "-o", default="output/lid_cavity.png",
                        help="Output plot path (default: output/lid_cavity.png)")
    args = parser.parse_args()

    defaults = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "Nx": 64, "Ny": 64},
        "nu": 0.01, "dt": 0.001, "simulation_time": 20.0,
        "boundary": {"top": {"u": 1.0, "v": 0.0}, "other": {"u": 0.0, "v": 0.0}},
    }

    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v

    geo = cfg["geometry"]
    top_u = cfg.get("boundary", {}).get("top", {}).get("u", 1.0)
    smooth = cfg.get("boundary", {}).get("smooth_lid", True)

    solver = Solver(
        grid_size=(geo["Nx"], geo["Ny"]),
        nu=cfg["nu"], dt=cfg["dt"],
        lid_speed=top_u, smooth_lid=smooth,
        Lx=geo["Lx"], Ly=geo["Ly"],
        advection_scheme=cfg.get("advection_scheme", "upwind"),
        diffusion_scheme=cfg.get("diffusion_scheme", "crank_nicolson"),
    )
    ok = solver.solve(simulation_time=cfg.get("simulation_time", 20.0), verbose=True)
    if not ok:
        raise SystemExit("Simulation blew up; not saving NaN output.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_contour(solver.mesh, solver.u, solver.v, solver.p, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
