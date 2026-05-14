"""Run lid-driven cavity example using the production solver."""

import sys
sys.path.insert(0, "src")

from cfd_solver.solver import StaggeredSolver
from cfd_solver.solver.viz import save_velocity_contour

# Parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 64, 64
nu = 0.01
dt = 0.001
steps = 500

print(f"Lid-driven cavity: {Nx}x{Ny} grid, {steps} steps")
print(f"Memory: ~{3 * (Nx+1) * Ny * 8 / 1e6:.1f} MB")
print()

# Setup
solver = StaggeredSolver(Lx, Ly, Nx, Ny, nu, dt, u_bc={"top": 1.0, "bottom": 0.0, "left": 0.0, "right": 0.0})

# Run
print("Running simulation...")
solver.solve(steps, verbose=True)

# Check divergence (should be ~0)
print(f"\nFinal divergence (L2): {solver.divergence_norm():.2e}")
print(f"Final divergence (max): {solver.max_divergence():.2e}")
print(f"CFL: {solver.cfl():.3f}")

# Save result
save_velocity_contour(solver, "output/lid_cavity.png")