"""Run lid-driven cavity example."""

import sys
sys.path.insert(0, "src")

from cfd_solver.solver import Grid, Solver, BoundaryConditions
from cfd_solver.solver.viz import save_velocity_plot

# Parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64
nu = 0.01
dt = 0.001
steps = 200

# Setup
grid = Grid(Lx, Ly, Nx, Ny)
bc = BoundaryConditions(top_u=1.0)
solver = Solver(grid, nu, dt, bc)

# Run
print("Running lid-driven cavity simulation...")
solver.solve(steps, verbose=True)

# Check divergence (should be near zero)
print(f"\nFinal max divergence: {solver.divergence():.6e}")

# Save result
save_velocity_plot(grid, solver.u, solver.v, "output/lid_cavity.png")