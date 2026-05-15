"""T-junction pipe flow example (YAML-configurable).

Run: python3 examples/t_pipe.py --config examples/t_pipe.yaml
"""

import sys
import os
import argparse
import yaml
sys.path.insert(0, "src")

import numpy as np
from cfd_solver.solver.staggered_solver import StaggeredSolver
from cfd_solver.solver.viz import save_velocity_contour


def load_cfg(path, defaults):
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='examples/t_pipe.yaml')
    args = parser.parse_args()

    defaults = {
        'geometry': {'Lx': 2.0, 'Ly': 2.0, 'Nx': 96, 'Ny': 96},
        'nu': 0.001, 'dt': 1e-4, 'steps': 800,
        'inlet_u': 0.5
    }

    cfg = load_cfg(args.config, defaults)
    geo = cfg['geometry']
    Lx, Ly = geo['Lx'], geo['Ly']
    Nx, Ny = geo['Nx'], geo['Ny']
    nu, dt = cfg.get('nu', defaults['nu']), cfg.get('dt', defaults['dt'])
    steps = cfg.get('steps', defaults['steps'])
    inlet_u = cfg.get('inlet_u', defaults['inlet_u'])

    solver = StaggeredSolver(Lx, Ly, Nx, Ny, nu, dt)

    # Build a simple T-mask
    cell_mask = np.zeros((Nx, Ny), dtype=bool)
    center_y = Ny // 2
    half_h = max(2, Ny // 16)
    cell_mask[:Nx//2, center_y-half_h:center_y+half_h] = True  # horizontal inlet (left)
    branch_width = max(2, Nx // 32)
    # FIX: branch must extend down to connect to full pipe height
    cell_mask[Nx//2 - branch_width:Nx//2 + branch_width, center_y-half_h:Ny] = True  # vertical branch (up)
    cell_mask[Nx//2:Nx, center_y-half_h:center_y+half_h] = True  # horizontal outlet (right)

    # Face masks
    u_mask = np.zeros((Nx+1, Ny), dtype=bool)
    for i in range(Nx+1):
        for j in range(Ny):
            left = cell_mask[i-1, j] if i-1 >= 0 else False
            right = cell_mask[i, j] if i < Nx else False
            u_mask[i, j] = left or right

    v_mask = np.zeros((Nx, Ny+1), dtype=bool)
    for i in range(Nx):
        for j in range(Ny+1):
            bottom = cell_mask[i, j-1] if j-1 >= 0 else False
            top = cell_mask[i, j] if j < Ny else False
            v_mask[i, j] = bottom or top

    inlet_j = slice(center_y-half_h, center_y+half_h)

    print("T-pipe: running simulation")
    for n in range(steps):
        solver.step()
        solver.u[~u_mask] = 0.0
        solver.v[~v_mask] = 0.0
        solver.p[~cell_mask] = 0.0
        solver.u[0, inlet_j] = inlet_u
        if n % 100 == 0:
            print(f"Step {n}: max |∇·u| = {solver.max_divergence():.3e}")

    os.makedirs('output', exist_ok=True)
    # pass cell_mask so the plot omits solids
    save_velocity_contour(solver, 'output/t_pipe.png', cell_mask=cell_mask)
    print('Saved output/t_pipe.png')


if __name__ == '__main__':
    main()
