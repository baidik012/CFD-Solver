"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt


def save_velocity_plot(grid, u, v, path):
    """Save a quiver plot of the velocity field."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Subsample for cleaner arrows
    skip = max(1, grid.Nx // 32)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(grid.X[::skip, ::skip], grid.Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip], color='black', alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Field")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")