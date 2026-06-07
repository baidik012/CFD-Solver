import numpy as np
from cfd_solver.solver import Solver

# Solver parameters for Re=100 (U=1, L=1, nu=UL/Re = 1*1/100 = 0.01)
grid_size = (128, 128)
nu = 0.01
dt = 0.001
lid_speed = 1.0
steps = 10000 # Sufficiently large number of steps to reach steady state

print(f"Running Ghia validation for Re={int(lid_speed * 1.0 / nu)} with grid_size={grid_size}, nu={nu}, dt={dt}, lid_speed={lid_speed}, steps={steps}")

solver = Solver(
    grid_size=grid_size,
    nu=nu,
    dt=dt,
    lid_speed=lid_speed,
    smooth_lid=False,
    advection_scheme="central",
)

solver.solve(steps, verbose=True)

# --- Extract centerline velocities ---
# u along vertical centerline: x-index = Nx//2 (face at x=0.5 exactly)
u_x_idx = solver.Nx // 2
u_profile = solver.u[u_x_idx, 1:-1]  # shape (Ny,)
u_centerline_y = solver.mesh.yc      # shape (Ny,)

# v along horizontal centerline: y-index = Ny//2
v_y_idx = solver.Ny // 2
v_profile = solver.v[1:-1, v_y_idx]  # shape (Nx,)
v_centerline_x = solver.mesh.xc      # shape (Nx,)

# --- Print profiles ---
print("")
print("--- U-velocity along vertical centerline (x=0.5) ---")
print("| y      | u            |")
print("|--------|--------------|")
for y, u_val in zip(u_centerline_y, u_profile):
    print(f"| {y:<6.4f} | {u_val:<12.5f} |")

print("")
print("--- V-velocity along horizontal centerline (y=0.5) ---")
print("| x      | v            |")
print("|--------|--------------|")
for x, v_val in zip(v_centerline_x, v_profile):
    print(f"| {x:<6.4f} | {v_val:<12.5f} |")
