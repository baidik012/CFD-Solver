"""Advection schemes for staggered incompressible flow with ghost cells.

This module provides functions to calculate the advective term (u·∇)u in the
Navier-Stokes equations. Due to the staggered Arakawa C-grid, velocity
components (u, v) are not co-located, requiring interpolation to evaluate
the advection at the correct face locations.

Advection Term:
---------------
For the u-momentum equation, the advection term is: u * ∂u/∂x + v * ∂u/∂y
For the v-momentum equation, the advection term is: u * ∂v/∂x + v * ∂v/∂y

Schemes:
--------
1. Upwind: First-order accurate. It uses the direction of the flow to select the
   derivative approximation, providing high numerical stability (it avoids
   oscillations) at the cost of being "diffusive" (it tends to smooth out
   sharp gradients).
2. Central: Second-order accurate. It uses a symmetric average for derivatives,
   providing higher accuracy but can be unstable and prone to oscillations
   (wiggles) at high Reynolds numbers (high cell Peclet numbers).
"""

import numpy as np


def upwind(u, v, dx, dy):
    """First-order upwind advection on a staggered grid with ghost cells.

    The upwind scheme determines the derivative of a quantity based on the
    direction of the velocity (the "wind").

    Parameters
    ----------
    u : ndarray, shape (Nx+1, Ny+2)
        u-velocity array including ghost cells in y.
    v : ndarray, shape (Nx+2, Ny+1)
        v-velocity array including ghost cells in x.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    adv_u : ndarray, shape (Nx+1, Ny+2)
        The advective contribution for the u-momentum equation.
        Non-zero only at interior physical faces.
    adv_v : ndarray, shape (Nx+2, Ny+1)
        The advective contribution for the v-momentum equation.
        Non-zero only at interior physical faces.
    """
    Nx = u.shape[0] - 1
    Ny = u.shape[1] - 2  # u has shape (Nx+1, Ny+2)

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    # --- u-advection: u*du/dx + v*du/dy ---
    # Evaluated at active physical interior faces (i=1..Nx-1, j=1..Ny)
    ui = slice(1, Nx)
    uj = slice(1, Ny + 1)
    u_ij = u[ui, uj]

    # Upwind du/dx: uses u[i-1] if u > 0, else u[i+1]
    du_dx = np.where(
        u_ij > 0,
        (u_ij - u[ui.start - 1:ui.stop - 1, uj]) / dx,
        (u[ui.start + 1:ui.stop + 1, uj] - u_ij) / dx,
    )

    # Interpolate v to u-face location: average of 4 surrounding v-values
    v_at_u = 0.25 * (
        v[ui.start:ui.stop, uj.start - 1:uj.stop - 1]
        + v[ui.start:ui.stop, uj.start:uj.stop]
        + v[ui.start + 1:ui.stop + 1, uj.start - 1:uj.stop - 1]
        + v[ui.start + 1:ui.stop + 1, uj.start:uj.stop]
    )

    # Upwind du/dy: uses u[j-1] if v_at_u > 0, else u[j+1]
    du_dy = np.where(
        v_at_u > 0,
        (u_ij - u[ui, uj.start - 1:uj.stop - 1]) / dy,
        (u[ui, uj.start + 1:uj.stop + 1] - u_ij) / dy,
    )

    adv_u[ui, uj] = u_ij * du_dx + v_at_u * du_dy

    # --- v-advection: u*dv/dx + v*dv/dy ---
    # Evaluated at active physical interior faces (i=1..Nx, j=1..Ny-1)
    vi = slice(1, Nx + 1)
    vj = slice(1, Ny)
    v_ij = v[vi, vj]

    # Interpolate u to v-face location: average of 4 surrounding u-values
    u_at_v = 0.25 * (
        u[vi.start - 1:vi.stop - 1, vj.start:vj.stop]
        + u[vi.start:vi.stop, vj.start:vj.stop]
        + u[vi.start - 1:vi.stop - 1, vj.start + 1:vj.stop + 1]
        + u[vi.start:vi.stop, vj.start + 1:vj.stop + 1]
    )

    # Upwind dv/dx: uses v[i-1] if u_at_v > 0, else v[i+1]
    dv_dx = np.where(
        u_at_v > 0,
        (v_ij - v[vi.start - 1:vi.stop - 1, vj]) / dx,
        (v[vi.start + 1:vi.stop + 1, vj] - v_ij) / dx,
    )

    # Upwind dv/dy: uses v[j-1] if v_ij > 0, else v[j+1]
    dv_dy = np.where(
        v_ij > 0,
        (v_ij - v[vi, vj.start - 1:vj.stop - 1]) / dy,
        (v[vi, vj.start + 1:vj.stop + 1] - v_ij) / dy,
    )

    adv_v[vi, vj] = u_at_v * dv_dx + v_ij * dv_dy

    return adv_u, adv_v


def central(u, v, dx, dy):
    """Second-order central difference advection on a staggered grid.

    The central scheme uses a symmetric average to compute derivatives.

    Parameters
    ----------
    u : ndarray, shape (Nx+1, Ny+2)
    v : ndarray, shape (Nx+2, Ny+1)
    dx : float
    dy : float

    Returns
    -------
    adv_u : ndarray, shape (Nx+1, Ny+2)
    adv_v : ndarray, shape (Nx+2, Ny+1)
    """
    Nx = u.shape[0] - 1
    Ny = u.shape[1] - 2

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    # --- u-advection ---
    ui = slice(1, Nx)
    uj = slice(1, Ny + 1)
    u_ij = u[ui, uj]

    # Central du/dx: (u[i+1] - u[i-1]) / (2*dx)
    du_dx = (u[ui.start + 1:ui.stop + 1, uj] - u[ui.start - 1:ui.stop - 1, uj]) / (2 * dx)

    # Interpolate v to u-face
    v_at_u = 0.25 * (
        v[ui.start:ui.stop, uj.start - 1:uj.stop - 1]
        + v[ui.start:ui.stop, uj.start:uj.stop]
        + v[ui.start + 1:ui.stop + 1, uj.start - 1:uj.stop - 1]
        + v[ui.start + 1:ui.stop + 1, uj.start:uj.stop]
    )

    # Central du/dy: (u[j+1] - u[j-1]) / (2*dy)
    du_dy = (u[ui, uj.start + 1:uj.stop + 1] - u[ui, uj.start - 1:uj.stop - 1]) / (2 * dy)

    adv_u[ui, uj] = u_ij * du_dx + v_at_u * du_dy

    # --- v-advection ---
    vi = slice(1, Nx + 1)
    vj = slice(1, Ny)
    v_ij = v[vi, vj]

    # Interpolate u to v-face
    u_at_v = 0.25 * (
        u[vi.start - 1:vi.stop - 1, vj.start:vj.stop]
        + u[vi.start:vi.stop, vj.start:vj.stop]
        + u[vi.start - 1:vi.stop - 1, vj.start + 1:vj.stop + 1]
        + u[vi.start:vi.stop, vj.start + 1:vj.stop + 1]
    )

    # Central dv/dx: (v[i+1] - v[i-1]) / (2*dx)
    dv_dx = (v[vi.start + 1:vi.stop + 1, vj] - v[vi.start - 1:vi.stop - 1, vj]) / (2 * dx)

    # Central dv/dy: (v[j+1] - v[j-1]) / (2*dy)
    dv_dy = (v[vi, vj.start + 1:vj.stop + 1] - v[vi, vj.start - 1:vj.stop - 1]) / (2 * dy)

    adv_v[vi, vj] = u_at_v * dv_dx + v_ij * dv_dy

    return adv_u, adv_v
