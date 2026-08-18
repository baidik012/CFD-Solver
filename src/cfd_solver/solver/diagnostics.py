"""Flow diagnostics for staggered incompressible solvers.

Provides mass conservation (divergence) and stability (CFL) monitoring.
"""

import numpy as np


def divergence(u, v, dx, dy, interior_only=False):
    """Pointwise divergence ∇·u = ∂u/∂x + ∂v/∂y at cell centers.

    Parameters
    ----------
    u : ndarray, shape (Nx+1, Ny+2)
        u-velocity array.
    v : ndarray, shape (Nx+2, Ny+1)
        v-velocity array.
    dx, dy : float
        Grid spacing.
    interior_only : bool, optional
        If True, exclude boundary cells from the result.

    Returns
    -------
    div : ndarray
        Divergence at cell centers.
    """
    u_phys = u[:, 1:-1]
    v_phys = v[1:-1, :]
    div = (u_phys[1:, :] - u_phys[:-1, :]) / dx + (v_phys[:, 1:] - v_phys[:, :-1]) / dy
    if interior_only and div.shape[0] > 2 and div.shape[1] > 2:
        div = div[1:-1, 1:-1]
    return div


def divergence_norm(u, v, dx, dy):
    """RMS divergence: global measure of incompressibility error."""
    return np.sqrt(np.mean(divergence(u, v, dx, dy) ** 2))


def max_divergence(u, v, dx, dy, interior_only=False):
    """Maximum absolute divergence: detects local incompressibility violations."""
    return np.max(np.abs(divergence(u, v, dx, dy, interior_only)))


def cfl(u, v, dx, dy, dt):
    """Maximum CFL number: max(|u|*dt/dx + |v|*dt/dy).

    Returns np.inf if velocity field contains NaN/Inf (blowup).
    """
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return np.inf
    u_max = np.max(np.abs(u[:, 1:-1]))
    v_max = np.max(np.abs(v[1:-1, :]))
    return u_max * dt / dx + v_max * dt / dy


def is_blowup(u, v):
    """True if velocity field contains NaN or Inf."""
    return not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)))