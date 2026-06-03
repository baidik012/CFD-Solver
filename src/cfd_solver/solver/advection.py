"""Advection schemes for staggered incompressible flow.

All functions return adv_u (Nx+1, Ny) and adv_v (Nx, Ny+1) —
non-zero only at interior faces.

Available schemes:
  - upwind: 1st-order, diffusive but stable
  - central: 2nd-order, oscillatory without sufficient viscosity
"""

import numpy as np


def upwind(u, v, dx, dy):
    """First-order upwind advection on a staggered grid.

    Returns adv_u and adv_v with zeros on boundary faces.
    """
    Nx, Ny = u.shape[0] - 1, u.shape[1]

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    # --- u-advection at interior u-faces (i=1..Nx-1, j=1..Ny-2) ---
    ui = slice(1, Nx)
    uj = slice(1, Ny - 1)
    u_ij = u[ui, uj]

    du_dx = np.where(
        u_ij > 0,
        (u_ij - u[ui.start - 1:ui.stop - 1, uj]) / dx,
        (u[ui.start + 1:ui.stop + 1, uj] - u_ij) / dx,
    )

    v_at_u = 0.25 * (
        v[ui.start - 1:ui.stop - 1, uj]
        + v[ui.start - 1:ui.stop - 1, uj.start + 1:uj.stop + 1]
        + v[ui.start:ui.stop, uj]
        + v[ui.start:ui.stop, uj.start + 1:uj.stop + 1]
    )

    du_dy = np.where(
        v_at_u > 0,
        (u_ij - u[ui, uj.start - 1:uj.stop - 1]) / dy,
        (u[ui, uj.start + 1:uj.stop + 1] - u_ij) / dy,
    )

    adv_u[ui, uj] = u_ij * du_dx + v_at_u * du_dy

    # --- v-advection at interior v-faces (i=1..Nx-2, j=1..Ny-1) ---
    vi = slice(1, Nx - 1)
    vj = slice(1, Ny)
    v_ij = v[vi, vj]

    u_at_v = 0.25 * (
        u[vi.start:vi.stop, vj.start - 1:vj.stop - 1]
        + u[vi.start + 1:vi.stop + 1, vj.start - 1:vj.stop - 1]
        + u[vi.start:vi.stop, vj.start:vj.stop]
        + u[vi.start + 1:vi.stop + 1, vj.start:vj.stop]
    )

    dv_dx = np.where(
        u_at_v > 0,
        (v_ij - v[vi.start - 1:vi.stop - 1, vj]) / dx,
        (v[vi.start + 1:vi.stop + 1, vj] - v_ij) / dx,
    )

    dv_dy = np.where(
        v_ij > 0,
        (v_ij - v[vi, vj.start - 1:vj.stop - 1]) / dy,
        (v[vi, vj.start + 1:vj.stop + 1] - v_ij) / dy,
    )

    adv_v[vi, vj] = u_at_v * dv_dx + v_ij * dv_dy

    return adv_u, adv_v


def central(u, v, dx, dy):
    """Second-order central difference advection on a staggered grid.

    Returns adv_u and adv_v with zeros on boundary faces.
    """
    Nx, Ny = u.shape[0] - 1, u.shape[1]

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    # --- u-advection at interior u-faces (i=1..Nx-1, j=1..Ny-2) ---
    ui = slice(1, Nx)
    uj = slice(1, Ny - 1)
    u_ij = u[ui, uj]

    du_dx = (u[ui.start + 1:ui.stop + 1, uj] - u[ui.start - 1:ui.stop - 1, uj]) / (2 * dx)

    v_at_u = 0.25 * (
        v[ui.start - 1:ui.stop - 1, uj]
        + v[ui.start - 1:ui.stop - 1, uj.start + 1:uj.stop + 1]
        + v[ui.start:ui.stop, uj]
        + v[ui.start:ui.stop, uj.start + 1:uj.stop + 1]
    )

    du_dy = (u[ui, uj.start + 1:uj.stop + 1] - u[ui, uj.start - 1:uj.stop - 1]) / (2 * dy)

    adv_u[ui, uj] = u_ij * du_dx + v_at_u * du_dy

    # --- v-advection at interior v-faces (i=1..Nx-2, j=1..Ny-1) ---
    vi = slice(1, Nx - 1)
    vj = slice(1, Ny)
    v_ij = v[vi, vj]

    u_at_v = 0.25 * (
        u[vi.start:vi.stop, vj.start - 1:vj.stop - 1]
        + u[vi.start + 1:vi.stop + 1, vj.start - 1:vj.stop - 1]
        + u[vi.start:vi.stop, vj.start:vj.stop]
        + u[vi.start + 1:vi.stop + 1, vj.start:vj.stop]
    )

    dv_dx = (v[vi.start + 1:vi.stop + 1, vj] - v[vi.start - 1:vi.stop - 1, vj]) / (2 * dx)

    dv_dy = (v[vi, vj.start + 1:vj.stop + 1] - v[vi, vj.start - 1:vj.stop - 1]) / (2 * dy)

    adv_v[vi, vj] = u_at_v * dv_dx + v_ij * dv_dy

    return adv_u, adv_v
