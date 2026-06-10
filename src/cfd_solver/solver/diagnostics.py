"""Flow diagnostics for staggered incompressible solvers.

This module provides tools to monitor the physical and numerical health of the
simulation, including mass conservation (divergence) and stability (CFL).
"""

import numpy as np


def divergence(u, v, dx, dy, interior_only=False):
    """Pointwise divergence ∇·u = ∂u/∂x + ∂v/∂y at cell centers.

    In an incompressible flow, the divergence should be zero everywhere.
    Measuring the divergence is the primary way to verify the success of the
    pressure projection step.

    Parameters
    ----------
    u : ndarray, shape (Nx+1, Ny+2)
        u-velocity array.
    v : ndarray, shape (Nx+2, Ny+1)
        v-velocity array.
    dx, dy : float
        Grid spacing.
    interior_only : bool, optional
        If True, exclude boundary cells from the result (default False).

    Returns
    -------
    div : ndarray, shape (Nx, Ny) or smaller
        Divergence calculated at cell centers.
    """
    u_phys = u[:, 1:-1]
    v_phys = v[1:-1, :]
    div = (u_phys[1:, :] - u_phys[:-1, :]) / dx + (v_phys[:, 1:] - v_phys[:, :-1]) / dy
    if interior_only and div.shape[0] > 2 and div.shape[1] > 2:
        div = div[1:-1, 1:-1]
    return div


def divergence_norm(u, v, dx, dy):
    """Calculate the Root Mean Square (RMS) divergence.

    Provides a global measure of how well incompressibility is satisfied.

    Returns
    -------
    float
        The L2 norm of the divergence field.
    """
    return np.sqrt(np.mean(divergence(u, v, dx, dy) ** 2))


def max_divergence(u, v, dx, dy, interior_only=False):
    """Calculate the maximum absolute divergence in the field.

    Useful for detecting local violations of incompressibility,
    often near boundaries.

    Returns
    -------
    float
        The maximum absolute divergence value.
    """
    return np.max(np.abs(divergence(u, v, dx, dy, interior_only)))


def cfl(u, v, dx, dy, dt):
    """Calculate the maximum Courant-Friedrichs-Lewy (CFL) number.

    The CFL number is a measure of how much information travels across a
    grid cell in a single time step:
        CFL = (|u|*dt/dx) + (|v|*dt/dy)
    For numerical stability, especially in explicit schemes, the CFL number
    should typically be less than 1.0.

    Returns
    -------
    float
        The maximum CFL number in the domain.
    """
    return np.max(np.abs(u[:, 1:-1])) * dt / dx + np.max(np.abs(v[1:-1, :])) * dt / dy


def is_blowup(u, v):
    """Check if the simulation has become numerically unstable.

    Instability (blowup) is characterized by velocity values becoming
    extremely large (Inf) or undefined (NaN).

    Returns
    -------
    bool
        True if the velocity field contains NaN or Inf values.
    """
    return not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
