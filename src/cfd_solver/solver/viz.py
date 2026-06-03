"""Visualization utilities for staggered incompressible flow.

Provides two main functions:
  - save_quiver: velocity vector plot
  - save_contour: pressure + velocity magnitude side-by-side

Both accept a Mesh and the velocity/pressure arrays directly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _default_skip(Nx, Ny):
    return max(1, min(Nx, Ny) // 32)


def _interpolate_to_centers(u, v):
    """Interpolate face velocities to cell centers."""
    u_c = 0.5 * (u[1:, :] + u[:-1, :])
    v_c = 0.5 * (v[:, 1:] + v[:, :-1])
    return u_c, v_c


def save_quiver(mesh, u, v, path, skip=None, scale=None):
    """Save a quiver plot of the velocity field.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (Nx+1, Ny)
    v : ndarray, shape (Nx, Ny+1)
    path : str
        Output file path.
    skip : int, optional
        Subsampling factor for quiver arrows.
    scale : float, optional
        Matplotlib quiver scale parameter.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if skip is None:
        skip = _default_skip(mesh.Nx, mesh.Ny)

    X, Y = mesh.cell_center_grid()
    u_c, v_c = _interpolate_to_centers(u, v)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_c[::skip, ::skip], v_c[::skip, ::skip],
              color="black", alpha=0.6, scale=scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Field")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def save_contour(mesh, u, v, p, path, skip=None, scale=None):
    """Save pressure contours and velocity magnitude side-by-side.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (Nx+1, Ny)
    v : ndarray, shape (Nx, Ny+1)
    p : ndarray, shape (Nx, Ny)
    path : str
        Output file path.
    skip : int, optional
        Subsampling factor for quiver arrows.
    scale : float, optional
        Matplotlib quiver scale parameter.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if skip is None:
        skip = _default_skip(mesh.Nx, mesh.Ny)

    X, Y = mesh.cell_center_grid()
    u_c, v_c = _interpolate_to_centers(u, v)
    speed = np.sqrt(u_c**2 + v_c**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pressure
    cf = axes[0].contourf(X, Y, p, levels=100, cmap="RdBu_r")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Pressure")
    axes[0].set_aspect("equal")
    plt.colorbar(cf, ax=axes[0], label="p")

    # Velocity magnitude
    cf = axes[1].contourf(X, Y, speed, levels=100, cmap="viridis")
    axes[1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                   u_c[::skip, ::skip], v_c[::skip, ::skip],
                   color="white", alpha=0.7, scale=scale)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Velocity Magnitude")
    axes[1].set_aspect("equal")
    plt.colorbar(cf, ax=axes[1], label="|u|")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")
