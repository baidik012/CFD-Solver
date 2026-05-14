"""Run staggered grid lid-driven cavity."""

import sys
sys.path.insert(0, "src")

from cfd_solver.solver import StaggeredSolver
from cfd_solver.solver.viz import save_velocity_contour
import time

# Parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 128, 128  # good balance of accuracy and speed
nu = 0.01
dt = 0.001
steps = 1000

print(f"Lid-driven cavity: {Nx}x{Ny} grid, {steps} steps")
print(f"Memory: ~{3 * (Nx+1) * Ny * 8 / 1e6:.1f} MB")
print()

# Setup
t0 = time.time()
solver = StaggeredSolver(Lx, Ly, Nx, Ny, nu, dt, u_bc={"top": 1.0})
print(f"Setup time: {time.time() - t0:.2f}s")

# Run
t0 = time.time()
solver.solve(steps, verbose=True)
print(f"\nSolve time: {time.time() - t0:.2f}s")

# Results
print(f"\nFinal divergence (L2): {solver.divergence_norm():.2e}")
print(f"Final divergence (max): {solver.max_divergence():.2e}")
print(f"CFL: {solver.cfl():.3f}")

# Save
save_velocity_contour(solver, "output/staggered_result.png")