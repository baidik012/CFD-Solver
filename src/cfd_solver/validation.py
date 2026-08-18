"""Reusable validation helpers for CFD examples and tests.

Provides:
  - extract_profile: extract a velocity profile at a given position
  - compute_l2_error, compute_linf_error: error norms
  - print_error_report: formatted output
  - run_grid_convergence: run simulation on multiple grids
  - compute_convergence_rate: fit convergence rate from grid study
  - print_convergence_table: formatted convergence table
  - save_convergence_plot: log-log convergence plot as PNG
"""

import numpy as np


def extract_profile(solver, direction="u", axis="y", position=None):
    """Extract a velocity profile from the solver at a given position.

    Parameters
    ----------
    solver : Solver
        A solved (or partially solved) Solver instance.
    direction : str
        Velocity component: 'u' (horizontal) or 'v' (vertical).
    axis : str
        Profile direction: 'y' (vertical slice at given x) or
        'x' (horizontal slice at given y).
    position : float or None
        Physical coordinate along the slicing axis.
        None defaults to domain center.

    Returns
    -------
    coord : ndarray
        Physical coordinates along the profile axis.
    values : ndarray
        Interpolated velocity values.
    """
    mesh = solver.mesh

    if direction == "u":
        field = solver.u[:, 1:-1]  # strip ghost cells in y
        if axis == "y":
            # Vertical profile at given x (u-face coordinates)
            if position is None:
                i = mesh.Nx // 2
            else:
                i = int(round(position / mesh.dx))
                i = max(0, min(i, mesh.Nx))
            coord = (np.arange(mesh.Ny) + 0.5) * mesh.dy
            values = field[i, :]
        else:
            # Horizontal profile at given y (u-face coordinates)
            if position is None:
                j = mesh.Ny // 2
            else:
                j = int(round(position / mesh.dy))
                j = max(0, min(j, mesh.Ny - 1))
            coord = mesh.xf  # u-face x-coordinates
            values = field[:, j]
    elif direction == "v":
        field = solver.v[1:-1, :]  # strip ghost cells in x
        if axis == "x":
            # Horizontal profile at given y (v-face coordinates)
            if position is None:
                j = mesh.Ny // 2
            else:
                j = int(round(position / mesh.dy))
                j = max(0, min(j, mesh.Ny))
            coord = (np.arange(mesh.Nx) + 0.5) * mesh.dx
            values = field[:, j]
        else:
            # Vertical profile at given x (v-face coordinates)
            if position is None:
                i = mesh.Nx // 2
            else:
                i = int(round(position / mesh.dx))
                i = max(0, min(i, mesh.Nx - 1))
            coord = mesh.yv  # v-face y-coordinates
            values = field[i, :]
    else:
        raise ValueError(f"direction must be 'u' or 'v', got {direction!r}")

    return coord, values


def compute_l2_error(numerical, analytical):
    """Root-mean-square error between two arrays."""
    return float(np.sqrt(np.mean((numerical - analytical) ** 2)))


def compute_linf_error(numerical, analytical):
    """Maximum absolute error between two arrays."""
    return float(np.max(np.abs(numerical - analytical)))


def print_error_report(name, l2, linf, divergence=None, grid=None, extra=None):
    """Print a formatted error report.

    Parameters
    ----------
    name : str
        Test case name.
    l2, linf : float
        L2 and L-infinity errors.
    divergence : float, optional
        Maximum divergence of the velocity field.
    grid : str, optional
        Grid description (e.g. '128x32').
    extra : dict, optional
        Additional key-value pairs to print.
    """
    print(f"--- {name} ---")
    if grid:
        print(f"  Grid: {grid}")
    print(f"  L2 error:    {l2:.6e}")
    print(f"  L-inf error: {linf:.6e}")
    if divergence is not None:
        print(f"  Max div:     {divergence:.2e}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")


def run_grid_convergence(run_fn, grids, compute_error_fn):
    """Run simulation on multiple grids and collect L2 errors.

    Parameters
    ----------
    run_fn : callable
        Function(nx, ny) -> solver instance (already solved).
    grids : list of (Nx, Ny) tuples
        Grid sizes to test, from coarsest to finest.
    compute_error_fn : callable
        Function(solver) -> float (L2 error).

    Returns
    -------
    errors : list of float
        L2 error for each grid.
    """
    errors = []
    for nx, ny in grids:
        solver = run_fn(nx, ny)
        err = compute_error_fn(solver)
        errors.append(err)
    return errors


def compute_convergence_rate(errors, grids):
    """Compute convergence rates from grid refinement study.

    For each adjacent pair of grids, computes:
        rate = log2(errors[i] / errors[i+1])

    This assumes uniform refinement (grid size doubles each step).

    Parameters
    ----------
    errors : list of float
        L2 errors from coarsest to finest.
    grids : list of (Nx, Ny) tuples
        Corresponding grid sizes.

    Returns
    -------
    rates : list of float
        Convergence rates (len = len(errors) - 1).
        None entries indicate undefined rates (e.g., zero error).
    """
    rates = []
    for i in range(len(errors) - 1):
        e_coarse = errors[i]
        e_fine = errors[i + 1]
        if e_fine > 0 and e_coarse > 0:
            # Grid size ratio (assuming Nx doubles)
            ratio = grids[i + 1][0] / grids[i][0]
            rate = np.log2(e_coarse / e_fine) / np.log2(ratio)
            rates.append(rate)
        else:
            rates.append(None)
    return rates


def print_convergence_table(grids, errors, rates, name=""):
    """Print a formatted convergence table.

    Parameters
    ----------
    grids : list of (Nx, Ny) tuples
    errors : list of float
    rates : list of float or None
    name : str, optional
        Table title.
    """
    if name:
        print(f"\n--- {name} ---")
    print(f"| {'Grid':>8} | {'L2 error':>12} | {'Rate':>8} |")
    print(f"|{'-' * 10}|{'-' * 14}|{'-' * 10}|")
    for i, (grid, err) in enumerate(zip(grids, errors)):
        grid_str = f"{grid[0]}x{grid[1]}"
        if i == 0 or rates is None or rates[i - 1] is None:
            rate_str = "—"
        else:
            rate_str = f"{rates[i - 1]:.2f}"
        print(f"| {grid_str:>8} | {err:>12.6e} | {rate_str:>8} |")


def save_convergence_plot(grids, errors, rates, output_path, name=""):
    """Save a log-log convergence plot as PNG.

    Parameters
    ----------
    grids : list of (Nx, Ny) tuples
    errors : list of float
    rates : list of float or None
    output_path : str
        Path to save the PNG file.
    name : str, optional
        Plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — skipping convergence plot.")
        return

    grid_sizes = [g[0] for g in grids]
    valid_mask = [e > 0 for e in errors]
    plot_sizes = [s for s, v in zip(grid_sizes, valid_mask) if v]
    plot_errors = [e for e, v in zip(errors, valid_mask) if v]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(plot_sizes, plot_errors, "bo-", linewidth=2, markersize=8,
              label="L2 error")

    # Add slope annotations
    if rates:
        for i, rate in enumerate(rates):
            if rate is not None:
                mid_x = np.sqrt(grid_sizes[i] * grid_sizes[i + 1])
                mid_y = np.sqrt(errors[i] * errors[i + 1])
                ax.annotate(f"rate={rate:.2f}", (mid_x, mid_y),
                           textcoords="offset points", xytext=(10, 10),
                           fontsize=9, color="blue")

    # Reference 2nd-order line
    if len(plot_sizes) >= 2:
        ref_line = [plot_errors[0] * (plot_sizes[0] / s) ** 2
                    for s in plot_sizes]
        ax.loglog(plot_sizes, ref_line, "k--", alpha=0.4, label="2nd order ref")

    ax.set_xlabel("Grid size (N)")
    ax.set_ylabel("L2 error")
    ax.set_title(name or "Grid Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Convergence plot saved to: {output_path}")
    plt.close(fig)