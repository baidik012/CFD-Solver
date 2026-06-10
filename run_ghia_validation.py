import os
import sys
import numpy as np

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cfd_solver.solver import Solver

# Solver parameters for Re=100 (U=1, L=1, nu=UL/Re = 1*1/100 = 0.01)
grid_size = (128, 128)
nu = 0.01
dt = 0.001
lid_speed = 1.0
steps = 10000  # sufficiently large number of steps to reach steady state

print(
    f"Running Ghia validation for Re={int(lid_speed * 1.0 / nu)} "
    f"with grid_size={grid_size}, nu={nu}, dt={dt}, lid_speed={lid_speed}, steps={steps}"
)

solver = Solver(
    grid_size=grid_size,
    nu=nu,
    dt=dt,
    lid_speed=lid_speed,
    smooth_lid=False,  # set True if you want smoother corner behavior
    advection_scheme="central",
)

solver.solve(steps, verbose=True)

# --- Extract centerline velocities (correct staggered probe extraction) ---
# u is stored at vertical faces x = xf[i], shape (Nx+1, Ny+2) with ghost in y.
# v is stored at horizontal faces y = yv[j], shape (Nx+2, Ny+1) with ghost in x.
#
# Ghia-style comparisons typically want values along the geometric lines:
#   x = 0.5 (vertical centerline) and y = 0.5 (horizontal centerline),
# but because u/v are face-based, we interpolate onto the probe lines.

x_probe = 0.5
y_probe = 0.5

# u-profile at x = x_probe, as a function of y (use cell-center y coordinates mesh.yc)
# Exclude ghost cells in y: u[:, 1:-1] -> shape (Nx+1, Ny)
u_interior_y = solver.u[:, 1:-1]
u_face_x = solver.mesh.xf  # shape (Nx+1,)

if not (u_face_x[0] <= x_probe <= u_face_x[-1]):
    raise ValueError(
        f"x_probe={x_probe} outside u face range [{u_face_x[0]}, {u_face_x[-1]}]"
    )

i = int(np.searchsorted(u_face_x, x_probe) - 1)
i = max(0, min(i, len(u_face_x) - 2))

x0, x1 = u_face_x[i], u_face_x[i + 1]
t = (x_probe - x0) / (x1 - x0) if x1 != x0 else 0.0

u_profile = (1.0 - t) * u_interior_y[i, :] + t * u_interior_y[i + 1, :]
u_centerline_y = solver.mesh.yc  # shape (Ny,)

# v-profile at y = y_probe, as a function of x (use cell-center x coordinates mesh.xc)
# Exclude ghost cells in x: v[1:-1, :] -> shape (Nx, Ny+1)
v_interior_x = solver.v[1:-1, :]
v_face_y = solver.mesh.yv  # shape (Ny+1,)

if not (v_face_y[0] <= y_probe <= v_face_y[-1]):
    raise ValueError(
        f"y_probe={y_probe} outside v face range [{v_face_y[0]}, {v_face_y[-1]}]"
    )

j = int(np.searchsorted(v_face_y, y_probe) - 1)
j = max(0, min(j, len(v_face_y) - 2))

y0, y1 = v_face_y[j], v_face_y[j + 1]
s = (y_probe - y0) / (y1 - y0) if y1 != y0 else 0.0

v_profile = (1.0 - s) * v_interior_x[:, j] + s * v_interior_x[:, j + 1]
v_centerline_x = solver.mesh.xc  # shape (Nx,)

assert len(u_profile) == len(u_centerline_y), "u_profile and u_centerline_y length mismatch"
assert len(v_profile) == len(v_centerline_x), "v_profile and v_centerline_x length mismatch"

# --- Print profiles ---
print("\n--- U-velocity along vertical centerline (x=0.5) [interpolated] ---")
print("| y      | u            |")
print("|--------|--------------|")
for y, u_val in zip(u_centerline_y, u_profile):
    print(f"| {y:<6.4f} | {u_val:<12.5f} |")

print("\n--- V-velocity along horizontal centerline (y=0.5) [interpolated] ---")
print("| x      | v            |")
print("|--------|--------------|")
for x, v_val in zip(v_centerline_x, v_profile):
    print(f"| {x:<6.4f} | {v_val:<12.5f} |")
