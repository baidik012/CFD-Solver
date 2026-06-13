"""
Ghia validation script for the Lid-Driven Cavity flow.

Compares the solver output against the benchmark data from:
    Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
    High-Re solutions for incompressible flow using the Navier-Stokes
    equations and a multigrid method. Journal of Computational Physics, 48(3), 387-411.

Run directly:
    python run_ghia_validation.py [Re]

    Re can be 100 (default), 400, or 1000.
"""

import os
import sys
import numpy as np

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cfd_solver.solver import Solver

# --- Ghia et al. (1982) Reference Data ---
# Tabulated from Tables II, III, IV in the paper.
# Same 17 stations for all Re.

GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000,
])

GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000,
])

# u-profile along x=0.5, v-profile along y=0.5
GHIA_DATA = {
    100: {
        "u": np.array([
            0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
            -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
            0.68717, 0.73722, 0.78871, 0.84123, 1.00000,
        ]),
        "v": np.array([
            0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
            0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
            -0.10313, -0.08864, -0.07391, -0.05906, 0.00000,
        ]),
    },
    400: {
        "u": np.array([
            0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
            -0.32726, -0.17119, -0.11477, 0.02138, 0.16256, 0.29093,
            0.55892, 0.61756, 0.68439, 0.75837, 1.00000,
        ]),
        "v": np.array([
            0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
            0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
            -0.22847, -0.19254, -0.15663, -0.12146, 0.00000,
        ]),
    },
    1000: {
        "u": np.array([
            0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
            -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
            0.46604, 0.51117, 0.57492, 0.65928, 1.00000,
        ]),
        "v": np.array([
            0.00000, 0.27485, 0.29032, 0.30353, 0.32627, 0.37095,
            0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
            -0.39188, -0.33714, -0.27669, -0.21388, 0.00000,
        ]),
    },
}

# Solver settings per Re
SOLVER_CONFIG = {
    100:  {"nu": 0.01,   "dt": 0.001, "steps": 10000, "advection": "central",  "grid": (128, 128)},
    400:  {"nu": 0.0025, "dt": 0.001, "steps": 20000, "advection": "central",  "grid": (128, 128)},
    1000: {"nu": 0.001,  "dt": 0.0005, "steps": 40000, "advection": "upwind",  "grid": (256, 256)},
}


def extract_u_profile(solver, x_probe=0.5):
    """Extract u-velocity along a vertical line at x = x_probe."""
    mesh = solver.mesh
    u_interior = solver.u[:, 1:-1]
    xf = mesh.xf

    i = int(np.searchsorted(xf, x_probe) - 1)
    i = max(0, min(i, len(xf) - 2))

    x0, x1 = xf[i], xf[i + 1]
    t = (x_probe - x0) / (x1 - x0) if x1 != x0 else 0.0
    u_profile = (1.0 - t) * u_interior[i, :] + t * u_interior[i + 1, :]

    return mesh.yc.copy(), u_profile


def extract_v_profile(solver, y_probe=0.5):
    """Extract v-velocity along a horizontal line at y = y_probe."""
    mesh = solver.mesh
    v_interior = solver.v[1:-1, :]
    yv = mesh.yv

    j = int(np.searchsorted(yv, y_probe) - 1)
    j = max(0, min(j, len(yv) - 2))

    y0, y1 = yv[j], yv[j + 1]
    s = (y_probe - y0) / (y1 - y0) if y1 != y0 else 0.0
    v_profile = (1.0 - s) * v_interior[:, j] + s * v_interior[:, j + 1]

    return mesh.xc.copy(), v_profile


def compute_errors(sol, ref):
    """Compute L2 and max absolute error."""
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


def run_ghia(Re=100):
    """Run the Ghia validation for a given Reynolds number."""
    if Re not in GHIA_DATA:
        print(f"Re={Re} not available. Choose from: {list(GHIA_DATA.keys())}")
        sys.exit(1)

    cfg = SOLVER_CONFIG[Re]
    grid_size = cfg["grid"]
    nu = cfg["nu"]
    dt = cfg["dt"]
    steps = cfg["steps"]
    advection = cfg["advection"]
    lid_speed = 1.0

    print(f"\nGhia validation: Re={Re}, grid={grid_size}, nu={nu}, dt={dt}, "
          f"steps={steps}, advection={advection}")

    solver = Solver(
        grid_size=grid_size,
        nu=nu,
        dt=dt,
        lid_speed=lid_speed,
        smooth_lid=False,
        advection_scheme=advection,
    )
    solver.solve(steps, verbose=True)

    # Extract profiles
    y_solver, u_solver = extract_u_profile(solver, x_probe=0.5)
    x_solver, v_solver = extract_v_profile(solver, y_probe=0.5)

    # Interpolate onto Ghia stations
    u_at_ghia = np.interp(GHIA_Y, y_solver, u_solver)
    v_at_ghia = np.interp(GHIA_X, x_solver, v_solver)

    ghia_u = GHIA_DATA[Re]["u"]
    ghia_v = GHIA_DATA[Re]["v"]

    # Error metrics
    u_l2, u_max = compute_errors(u_at_ghia, ghia_u)
    v_l2, v_max = compute_errors(v_at_ghia, ghia_v)

    print(f"\n{'='*60}")
    print(f"  ERROR METRICS  (Re={Re})")
    print(f"{'='*60}")
    print(f"  u-profile (x=0.5):  L2 = {u_l2:.6f},  Max = {u_max:.6f}")
    print(f"  v-profile (y=0.5):  L2 = {v_l2:.6f},  Max = {v_max:.6f}")
    print(f"{'='*60}")

    print_comparison_table(GHIA_Y, ghia_u, u_at_ghia, f"U-velocity along x=0.5 (Re={Re})")
    print_comparison_table(GHIA_X, ghia_v, v_at_ghia, f"V-velocity along y=0.5 (Re={Re})")

    return u_at_ghia, v_at_ghia, u_l2, u_max, v_l2, v_max


def main():
    # Parse Re from command line
    Re = 100
    if len(sys.argv) > 1:
        try:
            Re = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python run_ghia_validation.py [Re]")
            print(f"  Re can be 100, 400, or 1000")
            sys.exit(1)

    u_at_ghia, v_at_ghia, u_l2, u_max, v_l2, v_max = run_ghia(Re)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[skip] matplotlib not available — skipping plot.")
        return

    ghia_u = GHIA_DATA[Re]["u"]
    ghia_v = GHIA_DATA[Re]["v"]
    cfg = SOLVER_CONFIG[Re]
    grid = cfg["grid"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(u_at_ghia, GHIA_Y, "b-", linewidth=1.5, label=f"Solver ({grid[0]}x{grid[1]})")
    ax1.plot(ghia_u, GHIA_Y, "ks", markersize=5, label="Ghia et al. (1982)")
    ax1.set_xlabel("u-velocity")
    ax1.set_ylabel("y")
    ax1.set_title("U along x = 0.5")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(GHIA_X, v_at_ghia, "b-", linewidth=1.5, label=f"Solver ({grid[0]}x{grid[1]})")
    ax2.plot(GHIA_X, ghia_v, "ks", markersize=5, label="Ghia et al. (1982)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v-velocity")
    ax2.set_title("V along y = 0.5")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Ghia Validation — Re={Re}, {grid[0]}x{grid[1]}, {cfg['steps']} steps",
                 fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), f"ghia_re{Re}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
