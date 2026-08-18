"""Reusable validation helpers for CFD examples and tests."""

import numpy as np


def extract_profile(solver, direction="u", axis="y", position=None):
    """Extract a velocity profile from the solver at a given position."""
    mesh = solver.mesh

    if direction == "u":
        field = solver.u[:, 1:-1]
        if axis == "y":
            i = mesh.Nx // 2 if position is None else int(round(position / mesh.dx))
            i = max(0, min(i, mesh.Nx))
            coord = (np.arange(mesh.Ny) + 0.5) * mesh.dy
            values = field[i, :]
        elif axis == "x":
            if position is None:
                j = mesh.Ny // 2
            else:
                # u is located at y=(j+1/2)dy.
                j = int(round(position / mesh.dy - 0.5))
                j = max(0, min(j, mesh.Ny - 1))
            coord = mesh.xf
            values = field[:, j]
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    elif direction == "v":
        field = solver.v[1:-1, :]
        if axis == "x":
            if position is None:
                j = mesh.Ny // 2
            else:
                # v is located at x=(i+1/2)dx; for a horizontal profile,
                # position is y and therefore selects an integer y-face.
                j = int(round(position / mesh.dy))
                j = max(0, min(j, mesh.Ny))
            coord = (np.arange(mesh.Nx) + 0.5) * mesh.dx
            values = field[:, j]
        elif axis == "y":
            if position is None:
                i = mesh.Nx // 2
            else:
                # v is located at x=(i+1/2)dx.
                i = int(round(position / mesh.dx - 0.5))
                i = max(0, min(i, mesh.Nx - 1))
            coord = mesh.yv
            values = field[i, :]
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
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
    """Print a formatted error report."""
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
    """Run simulations on multiple grids and collect errors."""
    errors = []
    for nx, ny in grids:
        solver = run_fn(nx, ny)
        errors.append(compute_error_fn(solver))
    return errors


def compute_convergence_rate(errors, grids):
    """Compute observed rates from successive grid refinements."""
    if len(errors) != len(grids):
        raise ValueError("errors and grids must have the same length")

    rates = []
    for i in range(len(errors) - 1):
        e_coarse = errors[i]
        e_fine = errors[i + 1]
        nx_ratio = grids[i + 1][0] / grids[i][0]
        ny_ratio = grids[i + 1][1] / grids[i][1]
        ratios = [r for r in (nx_ratio, ny_ratio) if r > 1.0]
        if e_fine <= 0 or e_coarse <= 0 or not ratios:
            rates.append(None)
            continue
        if len(ratios) == 2 and not np.isclose(ratios[0], ratios[1]):
            rates.append(None)
            continue
        refinement = ratios[0]
        rates.append(np.log(e_coarse / e_fine) / np.log(refinement))
    return rates


def print_convergence_table(grids, errors, rates, name=""):
    """Print a formatted convergence table."""
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
    """Save a log-log convergence plot as PNG."""
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

    if rates:
        for i, rate in enumerate(rates):
            if rate is not None:
                mid_x = np.sqrt(grid_sizes[i] * grid_sizes[i + 1])
                mid_y = np.sqrt(errors[i] * errors[i + 1])
                ax.annotate(f"rate={rate:.2f}", (mid_x, mid_y),
                           textcoords="offset points", xytext=(10, 10),
                           fontsize=9, color="blue")

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
