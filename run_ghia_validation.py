"""
Ghia validation script for the Lid-Driven Cavity flow.

Compares the solver output against the benchmark data from:
    Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
    High-Re solutions for incompressible flow using the Navier-Stokes
    equations and a multigrid method. Journal of Computational Physics, 48(3), 387-411.

Run directly:
    python run_ghia_validation.py

Compares u-velocity along x=0.5 and v-velocity along y=0.5 against the
tabulated Ghia data for Re=100.
"""

import os
import sys
import numpy as np

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cfd_solver.solver import Solver

# --- Ghia et al. (1982) Reference Data for Re=100 ---
# Tabulated from Table II in the paper.

# u-velocity along the vertical centerline (x = 0.5)
# Columns: y, u
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000,
])
GHIA_U = np.array([
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
    -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
    0.68717, 0.73722, 0.78871, 0.84123, 1.00000,
])

# v-velocity along the horizontal centerline (y = 0.5)
# Columns: x, v
GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000,
])
GHIA_V = np.array([
    0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
    0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
    -0.10313, -0.08864, -0.07391, -0.05906, 0.00000,
])


def extract_u_profile(solver, x_probe=0.5):
    """Extract u-velocity along a vertical line at x = x_probe.

    Interpolates between the two nearest u-faces to get values on the
    cell-center y-grid.

    Returns
    -------
    y : ndarray
        y-coordinates (cell centers).
    u : ndarray
        Interpolated u-velocity at each y.
    """
    mesh = solver.mesh
    u_interior = solver.u[:, 1:-1]  # (Nx+1, Ny)
    xf = mesh.xf

    i = int(np.searchsorted(xf, x_probe) - 1)
    i = max(0, min(i, len(xf) - 2))

    x0, x1 = xf[i], xf[i + 1]
    t = (x_probe - x0) / (x1 - x0) if x1 != x0 else 0.0
    u_profile = (1.0 - t) * u_interior[i, :] + t * u_interior[i + 1, :]

    return mesh.yc.copy(), u_profile


def extract_v_profile(solver, y_probe=0.5):
    """Extract v-velocity along a horizontal line at y = y_probe.

    Interpolates between the two nearest v-faces to get values on the
    cell-center x-grid.

    Returns
    -------
    x : ndarray
        x-coordinates (cell centers).
    v : ndarray
        Interpolated v-velocity at each x.
    """
    mesh = solver.mesh
    v_interior = solver.v[1:-1, :]  # (Nx, Ny+1)
    yv = mesh.yv

    j = int(np.searchsorted(yv, y_probe) - 1)
    j = max(0, min(j, len(yv) - 2))

    y0, y1 = yv[j], yv[j + 1]
    s = (y_probe - y0) / (y1 - y0) if y1 != y0 else 0.0
    v_profile = (1.0 - s) * v_interior[:, j] + s * v_interior[:, j + 1]

    return mesh.xc.copy(), v_profile


def interpolate_to_ghia(xy_solver, uv_solver, xy_ghia):
    """Interpolate solver profile onto Ghia stations using linear interpolation.

    Parameters
    ----------
    xy_solver : ndarray
        Solver coordinate array (y for u-profile, x for v-profile).
    uv_solver : ndarray
        Solver velocity values at those coordinates.
    xy_ghia : ndarray
        Ghia tabulated coordinate stations.

    Returns
    -------
    uv_interp : ndarray
        Solver values interpolated to Ghia stations.
    """
    return np.interp(xy_ghia, xy_solver, uv_solver)


def compute_errors(sol, ref):
    """Compute L2 and max absolute error.

    Parameters
    ----------
    sol : ndarray
        Solver values at reference stations.
    ref : ndarray
        Reference (Ghia) values.

    Returns
    -------
    l2 : float
        L2 norm of the error (not normalized).
    max_err : float
        Maximum absolute error.
    """
    err = sol - ref
    l2 = np.sqrt(np.mean(err ** 2))
    max_err = np.max(np.abs(err))
    return l2, max_err


def print_comparison_table(stations, ref, sol, label):
    """Print a comparison table."""
    print(f"\n--- {label} ---")
    print(f"| {'Station':>8} | {'Ghia':>12} | {'Solver':>12} | {'Abs Error':>12} |")
    print(f"|{'-'*10}|{'-'*14}|{'-'*14}|{'-'*14}|")
    for st, r, s in zip(stations, ref, sol):
        print(f"| {st:>8.4f} | {r:>12.5f} | {s:>12.5f} | {abs(s - r):>12.5f} |")


def main():
    # --- Solver Configuration ---
    grid_size = (128, 128)
    nu = 0.01
    dt = 0.001
    lid_speed = 1.0
    steps = 10000

    Re = int(lid_speed / nu)
    print(
        f"Ghia validation: Re={Re}, grid={grid_size}, "
        f"nu={nu}, dt={dt}, steps={steps}"
    )

    solver = Solver(
        grid_size=grid_size,
        nu=nu,
        dt=dt,
        lid_speed=lid_speed,
        smooth_lid=False,
        advection_scheme="central",
    )
    solver.solve(steps, verbose=True)

    # --- Extract profiles ---
    y_solver, u_solver = extract_u_profile(solver, x_probe=0.5)
    x_solver, v_solver = extract_v_profile(solver, y_probe=0.5)

    # --- Interpolate onto Ghia stations ---
    u_at_ghia = interpolate_to_ghia(y_solver, u_solver, GHIA_Y)
    v_at_ghia = interpolate_to_ghia(x_solver, v_solver, GHIA_X)

    # --- Error metrics ---
    u_l2, u_max = compute_errors(u_at_ghia, GHIA_U)
    v_l2, v_max = compute_errors(v_at_ghia, GHIA_V)

    print(f"\n{'='*60}")
    print(f"  ERROR METRICS")
    print(f"{'='*60}")
    print(f"  u-profile (x=0.5):  L2 = {u_l2:.6f},  Max = {u_max:.6f}")
    print(f"  v-profile (y=0.5):  L2 = {v_l2:.6f},  Max = {v_max:.6f}")
    print(f"{'='*60}")

    # --- Comparison tables ---
    print_comparison_table(GHIA_Y, GHIA_U, u_at_ghia, "U-velocity along x=0.5")
    print_comparison_table(GHIA_X, GHIA_V, v_at_ghia, "V-velocity along y=0.5")

    # --- Plot ---
    matplotlib_available = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        matplotlib_available = False
        print("\n[skip] matplotlib not available — skipping plot generation.")

    if matplotlib_available:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: u-velocity
        ax1.plot(u_at_ghia, GHIA_Y, "b-", linewidth=1.5, label="Solver (128x128)")
        ax1.plot(GHIA_U, GHIA_Y, "ks", markersize=5, label="Ghia et al. (1982)")
        ax1.set_xlabel("u-velocity")
        ax1.set_ylabel("y")
        ax1.set_title("U along x = 0.5")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: v-velocity
        ax2.plot(GHIA_X, v_at_ghia, "b-", linewidth=1.5, label="Solver (128x128)")
        ax2.plot(GHIA_X, GHIA_V, "ks", markersize=5, label="Ghia et al. (1982)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("v-velocity")
        ax2.set_title("V along y = 0.5")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Ghia Validation — Re=100, 128x128, {steps} steps", fontsize=13)
        fig.tight_layout()

        out_path = os.path.join(os.path.dirname(__file__), "ghia_comparison.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
