"""Visualization utilities for staggered incompressible flow.

Provides three main functions:
  - save_quiver: velocity vector plot
  - save_contour: pressure + velocity magnitude side-by-side
  - save_streamlines: pressure + velocity streamlines side-by-side
"""

import os
import numpy as np
import matplotlib
# Set Agg backend only if no backend has been chosen yet (force=False).
# This avoids breaking interactive plotting in Jupyter when importing
# from cfd_solver.solver (which re-exports these functions).
matplotlib.use('Agg', force=False)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _default_skip(Nx, Ny):
    return max(1, min(Nx, Ny) // 32)


def _interpolate_to_centers(u, v):
    """Interpolate face velocities to cell centers."""
    u_phys = u[:, 1:-1]
    v_phys = v[1:-1, :]
    u_c = 0.5 * (u_phys[1:, :] + u_phys[:-1, :])
    v_c = 0.5 * (v_phys[:, 1:] + v_phys[:, :-1])
    return u_c, v_c


def save_quiver(mesh, u, v, path, skip=None, scale=None):
    """Save a quiver plot of the velocity field.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (Nx+1, Ny+2)
    v : ndarray, shape (Nx+2, Ny+1)
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
    u : ndarray, shape (Nx+1, Ny+2)
    v : ndarray, shape (Nx+2, Ny+1)
    p : ndarray, shape (Nx+2, Ny+2)
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
    cf = axes[0].contourf(X, Y, p[1:-1, 1:-1], levels=100, cmap="RdBu_r")
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


def save_streamlines(mesh, u, v, p, path, density=2.0, vmin=None, vmax=None):
    """Save pressure contours and velocity streamlines side-by-side.

    Parameters
    ----------
    mesh : Mesh
    u : ndarray, shape (Nx+1, Ny+2)
    v : ndarray, shape (Nx+2, Ny+1)
    p : ndarray, shape (Nx+2, Ny+2)
    path : str
        Output file path.
    density : float, optional
        Matplotlib streamplot density parameter.
    vmin : float, optional
        Lower bound for the log10(speed) colour scale. If None (default),
        derived from the data: max(-6, floor(min(log_speed))). This
        prevents low-speed flows from being clipped at a hardcoded floor.
    vmax : float, optional
        Upper bound for the log10(speed) colour scale. If None (default),
        derived from the data: max(0, ceil(max(log_speed))).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    X, Y = mesh.cell_center_grid()
    u_c, v_c = _interpolate_to_centers(u, v)
    speed = np.sqrt(u_c**2 + v_c**2)
    log_speed = np.log10(speed + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pressure
    cf = axes[0].contourf(X, Y, p[1:-1, 1:-1], levels=100, cmap="RdBu_r")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Pressure")
    axes[0].set_aspect("equal")
    plt.colorbar(cf, ax=axes[0], label="p")

    # Velocity streamlines — derive colour-scale bounds from the data
    # unless the caller supplied explicit values. Previously vmin was
    # hardcoded to -6, which clipped any flow slower than 1e-6 m/s
    # (e.g. Stokes flow, natural-convection benchmarks) to the bottom
    # of the scale and hid the actual flow structure.
    finite_log = log_speed[np.isfinite(log_speed)]
    if vmin is None:
        vmin = max(-6, int(np.floor(np.min(finite_log)))) if finite_log.size else -6
    if vmax is None:
        vmax = max(0, int(np.ceil(np.max(finite_log)))) if finite_log.size else 0

    st = axes[1].streamplot(
        mesh.xc, mesh.yc, u_c.T, v_c.T,
        color=log_speed.T, cmap="viridis",
        density=density, norm=mcolors.Normalize(vmin, vmax)
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Velocity Streamlines")
    axes[1].set_aspect("equal")

    # Custom colorbar for log-velocity with physical labels
    ticks = np.arange(vmin, vmax + 1)
    labels = [f"1e{t}" if t != 0 else "1.0" for t in ticks]

    cbar = fig.colorbar(st.lines, ax=axes[1], ticks=ticks, label="|u|")
    cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")