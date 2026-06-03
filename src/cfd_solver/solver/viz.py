"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt


def save_velocity_plot(grid, u, v, path, skip=None, scale=None):
    """Save a quiver plot of the velocity field."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if skip is None:
        skip = max(1, min(grid.Nx, grid.Ny) // 32)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Interpolate face velocities to cell centers for plotting
    if u.shape == grid.shape_u and v.shape == grid.shape_v:
        U_full = 0.5 * (u[1:, :] + u[:-1, :])  # shape (Nx, Ny)
        V_full = 0.5 * (v[:, 1:] + v[:, :-1])
    else:
        U_full = u
        V_full = v

    # Cell-center coordinates (Nx, Ny) — transposed to match U/V shape
    X = grid.X
    Y = grid.Y

    Xp = X[::skip, ::skip]
    Yp = Y[::skip, ::skip]
    U = U_full[::skip, ::skip]
    V = V_full[::skip, ::skip]

    q = ax.quiver(Xp, Yp, U, V, color='black', alpha=0.6, scale=scale)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Velocity Field (quiver)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def _cell_center_plot_fields(solver):
    """Return cell-centered fields in Matplotlib's (y, x) array order."""
    Nx, Ny = solver.Nx, solver.Ny

    u_center = 0.5 * (solver.u[1:, :] + solver.u[:-1, :])
    v_center = 0.5 * (solver.v[:, 1:] + solver.v[:, :-1])
    speed = np.sqrt(u_center**2 + v_center**2)

    x = np.linspace(solver.dx / 2, solver.Lx - solver.dx / 2, Nx)
    y = np.linspace(solver.dy / 2, solver.Ly - solver.dy / 2, Ny)
    X, Y = np.meshgrid(x, y)

    return {
        "X": X,
        "Y": Y,
        "pressure": solver.p.T,
        "speed": speed.T,
        "u": u_center.T,
        "v": v_center.T,
    }


def save_velocity_contour(solver, path, skip=None, scale=None, cell_mask=None):
    """Save pressure contours and velocity vectors for staggered solver.

    Parameters:
    - cell_mask: optional (Nx, Ny) bool array where True=fluid, False=solid
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fields = _cell_center_plot_fields(solver)
    X = fields["X"]
    Y = fields["Y"]
    pressure = fields["pressure"]
    speed = fields["speed"]
    u_center = fields["u"]
    v_center = fields["v"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Prepare mask overlay once if needed
    if cell_mask is not None:
        solid_mask = (~cell_mask).astype(float).T
        mask_kw = dict(levels=[0.5, 1.5], colors=['lightgray'], alpha=0.5)

    # Pressure contours
    ax = axes[0]
    cf = ax.contourf(X, Y, pressure, levels=100, cmap='RdBu_r')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pressure Field")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="p")
    if cell_mask is not None:
        ax.contourf(X, Y, solid_mask, **mask_kw)

    # Velocity field
    ax = axes[1]
    cf = ax.contourf(X, Y, speed, levels=100, cmap='viridis')
    if skip is None:
        skip = max(1, min(solver.Nx, solver.Ny) // 32)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_center[::skip, ::skip], v_center[::skip, ::skip],
              color='white', alpha=0.7, scale=scale)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Velocity Magnitude")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="|u|")
    if cell_mask is not None:
        ax.contourf(X, Y, solid_mask, **mask_kw)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")
