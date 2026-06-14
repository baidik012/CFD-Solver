#!/usr/bin/env python
"""Run the Taylor-Green vortex decay example.

Analytical solution (decaying):
    u(x, y, t) = -U0 * sin(kx * x) * cos(ky * y) * exp(-d * t)
    v(x, y, t) =  U0 * cos(kx * x) * sin(ky * y) * exp(-d * t)

where kx = 2*pi/Lx, ky = 2*pi/Ly, d = nu * (kx^2 + ky^2).
"""

import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, FreeSlipWall, PeriodicWall


def taylor_green_ic(mesh):
    """Taylor-Green initial condition."""
    Lx, Ly = mesh.Lx, mesh.Ly
    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    U0 = 1.0

    Xc, Yc = mesh.cell_center_grid()

    u = np.zeros(mesh.shape_u)
    v = np.zeros(mesh.shape_v)
    p = np.zeros(mesh.shape_p)

    # u-velocity at x-faces
    Xf, Yf = mesh.u_face_grid()
    u[:, 1:-1] = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf)

    # v-velocity at v-faces
    Xv, Yv = mesh.v_face_grid()
    v[1:-1, :] = U0 * np.cos(kx * Xv) * np.sin(ky * Yv)

    # Pressure at cell centers
    p[1:-1, 1:-1] = -0.25 * U0**2 * (np.cos(2.0 * kx * Xc) + np.cos(2.0 * ky * Yc))

    return u, v, p


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geo = cfg["geometry"]
    Nx, Ny = geo["Nx"], geo["Ny"]
    Lx, Ly = geo["Lx"], geo["Ly"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    sim_time = cfg["simulation_time"]

    bc = BoundaryConditions(
        top=FreeSlipWall(u=0.0),
        bottom=FreeSlipWall(u=0.0),
        left=PeriodicWall(),
        right=PeriodicWall(),
    )

    s = Solver(
        grid_size=(Nx, Ny), nu=nu, dt=dt, Lx=Lx, Ly=Ly,
        lid_speed=0.0, smooth_lid=False,
        boundary_config=bc,
        initial_condition=taylor_green_ic,
        force=True,
    )

    s.solve(simulation_time=sim_time, verbose=True)

    # Compute analytical decay
    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    U0 = 1.0
    d = nu * (kx**2 + ky**2)
    Xf, Yf = s.mesh.u_face_grid()
    u_exact = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf) * np.exp(-d * s.time)

    # L2 error
    u_num = s.u[:, 1:-1]
    l2 = np.sqrt(np.mean((u_num - u_exact)**2))
    print(f"\n  L2 error in u at t={s.time:.3f}: {l2:.6e}")

    out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(out_dir, exist_ok=True)
    s.save(os.path.join(out_dir, "taylor_green_result.png"))
    s.save_streamlines(os.path.join(out_dir, "taylor_green_streamlines.png"))
    s.checkpoint(os.path.join(out_dir, "taylor_green.npz"))
    print(f"  Saved to {out_dir}/taylor_green_result.png")


if __name__ == "__main__":
    main()
