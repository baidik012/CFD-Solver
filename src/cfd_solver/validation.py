"""Reusable validation helpers for CFD examples and tests.

Provides:
  - extract_profile: extract a velocity profile at a given position
  - compute_l2_error, compute_linf_error: error norms
  - print_error_report: formatted output
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
