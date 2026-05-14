"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt


def save_velocity_plot(grid, u, v, path, skip=None, scale=None):
    """Save a quiver plot of the velocity field."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if skip is None:
        try:
            skip = max(1, grid.Nx // 32)
        except Exception:
            skip = 1

    fig, ax = plt.subplots(figsize=(6, 5))

    X = getattr(grid, 'X', None)
    Y = getattr(grid, 'Y', None)

    if X is not None and Y is not None:
        Xp = X[::skip, ::skip]
        Yp = Y[::skip, ::skip]
    else:
        xi = np.arange(u.shape[0])
        yi = np.arange(u.shape[1])
        Xidx, Yidx = np.meshgrid(xi, yi)
        Xp, Yp = Xidx[::skip, ::skip], Yidx[::skip, ::skip]

    U = u[::skip, ::skip]
    V = v[::skip, ::skip]

    q = ax.quiver(Xp, Yp, U, V, color='black', alpha=0.6, scale=scale)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Velocity Field (quiver)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def save_velocity_contour(solver, path, skip=None, scale=None, cell_mask=None):
    """Save pressure contours and velocity vectors for staggered solver.

    Parameters:
    - cell_mask: optional (Nx, Ny) bool array where True=fluid, False=solid
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    Nx, Ny = solver.Nx, solver.Ny
    Lx, Ly = solver.Lx, solver.Ly

    u_center = 0.5 * (solver.u[1:, :] + solver.u[:-1, :])
    v_center = 0.5 * (solver.v[:, 1:] + solver.v[:, :-1])

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    speed = np.sqrt(u_center**2 + v_center**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pressure contours
    ax = axes[0]
    cf = ax.contourf(X, Y, solver.p, levels=20, cmap='RdBu_r')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pressure Field")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="p")
    
    # If mask provided, overlay solid regions with gray hatch
    if cell_mask is not None:
        x_cell = np.linspace(0, Lx, Nx)
        y_cell = np.linspace(0, Ly, Ny)
        X_cell, Y_cell = np.meshgrid(x_cell, y_cell)
        solid_mask = (~cell_mask).astype(float)
        ax.contourf(X_cell, Y_cell, solid_mask, levels=[0.5, 1.5], colors=['lightgray'], alpha=0.5)

    # Velocity field
    ax = axes[1]
    cf = ax.contourf(X, Y, speed, levels=20, cmap='viridis')
    if skip is None:
        skip = max(1, Nx // 32)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_center[::skip, ::skip], v_center[::skip, ::skip],
              color='white', alpha=0.7, scale=scale)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Velocity Magnitude")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="|u|")
    
    # If mask provided, overlay solid regions with gray
    if cell_mask is not None:
        solid_mask = (~cell_mask).astype(float)
        ax.contourf(X_cell, Y_cell, solid_mask, levels=[0.5, 1.5], colors=['lightgray'], alpha=0.5)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")
