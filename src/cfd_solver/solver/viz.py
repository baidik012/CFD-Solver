"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt


def save_velocity_plot(grid, u, v, path, skip=None, scale=None):
    """Save a quiver plot of the velocity field.

    Parameters:
    - grid: Grid object with X/Y (cell-center) coordinates
    - u, v: velocity arrays (same shape)
    - path: output filepath
    - skip: integer stride for plotting (if None, chosen from grid size)
    - scale: quiver scale passed to matplotlib.quiver (optional)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if skip is None:
        try:
            skip = max(1, grid.Nx // 32)
        except Exception:
            skip = 1

    fig, ax = plt.subplots(figsize=(6, 5))

    # Prefer cell-center coordinates; Grid provides X, Y properties
    X = getattr(grid, 'X', None)
    Y = getattr(grid, 'Y', None)

    if X is not None and Y is not None:
        Xp = X[::skip, ::skip]
        Yp = Y[::skip, ::skip]
    else:
        # Fall back to array indices scaled by grid spacing if available
        xi = np.arange(u.shape[0])
        yi = np.arange(u.shape[1])
        Xidx, Yidx = np.meshgrid(xi, yi)
        Xp, Yp = Xidx[::skip, ::skip], Yidx[::skip, ::skip]

    U = u[::skip, ::skip]
    V = v[::skip, ::skip]

    q = ax.quiver(Xp, Yp, U, V, color='black', alpha=0.6, scale=scale)

    # Label axes with physical units when grid provided
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Velocity Field (quiver)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def save_velocity_contour(solver, path, skip=None, scale=None):
    """Save pressure contours and velocity vectors for staggered solver.

    Parameters:
    - solver: StaggeredSolver instance
    - path: output filepath
    - skip: integer stride for plotting (optional)
    - scale: quiver scale passed to matplotlib.quiver (optional)
    """
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
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pressure Field")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="p")

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

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")