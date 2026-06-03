"""Flow diagnostics for staggered incompressible solvers."""

import numpy as np


def divergence(u, v, dx, dy, interior_only=False):
    """Pointwise divergence du/dx + dv/dy at cell centers.

    Parameters
    ----------
    u : ndarray, shape (Nx+1, Ny)
    v : ndarray, shape (Nx, Ny+1)
    dx, dy : float
    interior_only : bool
        If True, exclude boundary cells from the result.
    """
    div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy
    if interior_only and div.shape[0] > 2 and div.shape[1] > 2:
        div = div[1:-1, 1:-1]
    return div


def divergence_norm(u, v, dx, dy):
    """RMS divergence."""
    return np.sqrt(np.mean(divergence(u, v, dx, dy) ** 2))


def max_divergence(u, v, dx, dy, interior_only=False):
    """Max absolute divergence."""
    return np.max(np.abs(divergence(u, v, dx, dy, interior_only)))


def cfl(u, v, dx, dy, dt):
    """Maximum CFL number."""
    return np.max(np.abs(u)) * dt / dx + np.max(np.abs(v)) * dt / dy


def is_blowup(u, v):
    """Return True if velocity contains NaN or Inf."""
    return not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
