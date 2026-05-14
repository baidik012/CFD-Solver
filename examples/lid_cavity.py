"""Lid-driven cavity example (YAML-configurable).

Run: python3 examples/lid_cavity.py --config examples/lid_cavity.yaml
"""

import sys
import os
import argparse
import yaml
sys.path.insert(0, "src")

from cfd_solver.solver import StaggeredSolver
from cfd_solver.solver.viz import save_velocity_contour


def load_cfg(path, defaults):
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    # merge defaults
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='examples/lid_cavity.yaml')
    args = parser.parse_args()

    defaults = {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'Nx': 64, 'Ny': 64},
        'nu': 0.01, 'dt': 0.001, 'steps': 200,
        'boundary': {'top': {'u': 1.0, 'v': 0.0}, 'other': {'u': 0.0, 'v': 0.0}}
    }

    cfg = load_cfg(args.config, defaults)
    geo = cfg['geometry']
    Lx, Ly = geo['Lx'], geo['Ly']
    Nx, Ny = geo['Nx'], geo['Ny']
    nu, dt = cfg.get('nu', defaults['nu']), cfg.get('dt', defaults['dt'])
    steps = cfg.get('steps', defaults['steps'])

    # Boundary
    bc_cfg = cfg.get('boundary', {})
    top = bc_cfg.get('top', {})
    top_u = top.get('u', 1.0)

    solver = StaggeredSolver(Lx, Ly, Nx, Ny, nu, dt, u_bc={"top": top_u, "bottom": 0.0, "left": 0.0, "right": 0.0})
    solver.solve(steps, verbose=True)

    os.makedirs('output', exist_ok=True)
    plot_cfg = cfg.get('plot', {})
    skip = plot_cfg.get('skip', None)
    scale = plot_cfg.get('scale', None)
    save_velocity_contour(solver, 'output/lid_cavity.png', skip=skip, scale=scale)
    print('Saved output/lid_cavity.png')


if __name__ == '__main__':
    main()
