"""Chorin projection method for incompressible Navier-Stokes.

Orchestrates one time step: advection → diffusion → pressure solve → corrector.
"""

import numpy as np


def step(u, v, p, mesh, dt, advection_fn, diffusion_solver, pressure_solver, bc):
    """Advance one time step using Chorin's projection method.

    Parameters
    ----------
    u : ndarray, shape (Nx+1, Ny)
    v : ndarray, shape (Nx, Ny+1)
    p : ndarray, shape (Nx, Ny)
    mesh : Mesh
    dt : float
    advection_fn : callable(u, v, dx, dy) -> (adv_u, adv_v)
    diffusion_solver : explicit() or CrankNicolson
    pressure_solver : PressureSolver
    bc : BoundaryConditions

    Returns
    -------
    u, v, p : updated arrays (modified in-place)
    """
    Nx, Ny = mesh.Nx, mesh.Ny
    dx, dy = mesh.dx, mesh.dy

    # 1. Apply boundary conditions
    bc.apply(u, v, Nx, Ny)

    # 2. Compute advection
    adv_u, adv_v = advection_fn(u, v, dx, dy)

    # 3. Diffusion predictor
    if callable(diffusion_solver):
        u_star, v_star = diffusion_solver(u, v, adv_u, adv_v, dx, dy, dt,
                                          diffusion_solver.nu, bc, Nx, Ny)
    else:
        u_star, v_star = diffusion_solver.solve(u, v, adv_u, adv_v)

    # 4. Pressure Poisson
    p[:] = pressure_solver.solve(u_star, v_star, dt)

    # 5. Corrector: apply pressure gradient
    grad_p_x = (p[1:, :] - p[:-1, :]) / dx
    grad_p_y = (p[:, 1:] - p[:, :-1]) / dy

    u[1:-1, :] = u_star[1:-1, :] - dt * grad_p_x
    v[:, 1:-1] = v_star[:, 1:-1] - dt * grad_p_y

    # 6. Re-apply boundary conditions
    bc.apply(u, v, Nx, Ny)
