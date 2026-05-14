"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt


def save_velocity_plot(grid, u, v, path):
    """Save a quiver plot of the velocity field."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    skip = max(1, grid.Nx // 32)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Grid stores cell-center coordinates as Xc, Yc (shape: Ny x Nx). Transpose
    # them to match velocity array shapes (Nx x Ny) used elsewhere.
    X = getattr(grid, 'Xc', None)
    Y = getattr(grid, 'Yc', None)
    if X is None or Y is None:
        # Fall back to attributes X, Y if present
        X = getattr(grid, 'X', None)
        Y = getattr(grid, 'Y', None)
    if X is not None and Y is not None:
        ax.quiver(X.T[::skip, ::skip], Y.T[::skip, ::skip],
                  u[::skip, ::skip], v[::skip, ::skip], color='black', alpha=0.6)
    else:
        # Last resort: plot with array indices
        xi = np.arange(u.shape[0])
        yi = np.arange(u.shape[1])
        Xidx, Yidx = np.meshgrid(xi, yi)
        ax.quiver(Xidx[::skip, ::skip], Yidx[::skip, ::skip],
                  u[::skip, ::skip], v[::skip, ::skip], color='black', alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Field")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def save_velocity_contour(solver, path):
    """Save pressure contours and velocity vectors for staggered solver."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    Nx, Ny = solver.Nx, solver.Ny
    Lx, Ly = solver.Lx, solver.Ly

    # Interpolate u and v to cell centers for plotting
    u_center = 0.5 * (solver.u[1:, :] + solver.u[:-1, :])
    v_center = 0.5 * (solver.v[:, 1:] + solver.v[:, :-1])

    # Cell center coordinates
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Velocity magnitude
    speed = np.sqrt(u_center**2 + v_center**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pressure contours
    ax = axes[0]
    cf = ax.contourf(X, Y, solver.p, levels=20, cmap='RdBu_r')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Pressure Field")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="p")

    # Velocity field
    ax = axes[1]
    cf = ax.contourf(X, Y, speed, levels=20, cmap='viridis')
    skip = max(1, Nx // 32)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_center[::skip, ::skip], v_center[::skip, ::skip],
              color='white', alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Magnitude")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="|u|")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")