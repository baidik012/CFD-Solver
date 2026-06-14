"""Validate channel (Poiseuille) flow against analytical solution.

Analytical steady-state solution for flow between parallel plates:
    u(y) = (1/2*nu) * G * y * (H - y)
where G = -dp/dx is the pressure gradient.

For the inlet/outlet BC approach, the analytical profile at the outlet
should match the inlet profile (fully-developed flow is preserved).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import (
    BoundaryConditions, NoSlipWall, InletWall, OutletWall,
)


def run_channel_flow(Nx=64, Ny=16, nu=0.01, U_max=1.0, Lx=5.0, Ly=1.0, dt=0.001):
    """Run channel flow simulation and return results."""
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=U_max),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(
        grid_size=(Nx, Ny), nu=nu, dt=dt,
        Lx=Lx, Ly=Ly,
        boundary_config=bc,
        force=True,
    )
    s.solve(simulation_time=3.0, verbose=False)
    return s


def parabolic_profile(y, H, U_max):
    """Analytical parabolic velocity profile."""
    return 4.0 * U_max * y * (H - y) / (H ** 2)


def validate():
    s = run_channel_flow()

    # Extract u-velocity profile at the channel midpoint (x = Lx/2)
    Nx, Ny = s.Nx, s.Ny
    dy = s.dy
    H = s.Ly

    # u is stored at vertical faces: u[i, j+1] for interior j=0..Ny-1
    # Face y-positions: yc = (j + 0.5) * dy
    mid_i = Nx // 2
    u_numerical = s.u[mid_i, 1:-1]  # interior faces
    y = (np.arange(Ny) + 0.5) * dy

    u_analytical = parabolic_profile(y, H, U_max=1.0)

    # L2 error
    l2_error = np.sqrt(np.mean((u_numerical - u_analytical) ** 2))
    max_error = np.max(np.abs(u_numerical - u_analytical))

    print(f"Grid: {Nx}x{Ny}")
    print(f"L2 error:  {l2_error:.6f}")
    print(f"Max error: {max_error:.6f}")
    print(f"Max divergence: {s.max_divergence():.2e}")

    # Save result
    s.save("output/channel_flow.png")
    print("Saved to output/channel_flow.png")

    return l2_error


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    validate()
