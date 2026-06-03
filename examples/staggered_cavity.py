"""Lid-driven cavity example (YAML-configurable).

Run: python examples/staggered_cavity.py --config examples/staggered_cavity.yaml
"""

import os
import argparse
import yaml

from cfd_solver.solver import Solver
from cfd_solver.solver.viz import save_contour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="examples/staggered_cavity.yaml")
    args = parser.parse_args()

    defaults = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "Nx": 64, "Ny": 64},
        "nu": 0.01, "dt": 0.001, "steps": 200,
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
    )
    solver.solve(cfg["steps"], verbose=True)

    os.makedirs("output", exist_ok=True)
    save_contour(solver.mesh, solver.u, solver.v, solver.p, "output/result.png")
    print("Saved output/result.png")


if __name__ == "__main__":
    main()
