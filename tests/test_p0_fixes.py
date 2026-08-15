"""Regression tests for the v0.3.2 P0 numerical fixes."""

import numpy as np
import pytest

from cfd_solver.solver import BoundaryConditions, Mesh, Solver, create_pressure_solver
from cfd_solver.solver.bc import InletWall, NoSlipWall, OutletWall, PeriodicWall
from cfd_solver.solver.p0_fixes import GeneralPeriodicPressureSolver


def _periodic_bc(x=False, y=False):
    return BoundaryConditions(
        left=PeriodicWall() if x else NoSlipWall(),
        right=PeriodicWall() if x else NoSlipWall(),
        bottom=PeriodicWall() if y else NoSlipWall(),
        top=PeriodicWall() if y else NoSlipWall(),
    )


def test_periodic_pressure_factory_supports_each_topology():
    mesh = Mesh(1.0, 1.0, 16, 12)
    for x, y in ((True, False), (False, True), (True, True)):
        solver = create_pressure_solver(mesh, _periodic_bc(x, y))
        assert isinstance(solver, GeneralPeriodicPressureSolver)
        assert solver.periodic_x is x
        assert solver.periodic_y is y


def test_asymmetric_periodic_boundary_is_rejected():
    mesh = Mesh(1.0, 1.0, 8, 8)
    bc = BoundaryConditions(
        left=PeriodicWall(), right=NoSlipWall(),
        bottom=NoSlipWall(), top=NoSlipWall(),
    )
    with pytest.raises(ValueError, match="Periodic x boundary"):
        create_pressure_solver(mesh, bc)


def test_cn_inlet_outlet_remains_supported_by_wall_ghost_coefficients():
    """CN remains available for inlet/outlet BCs with the audited wall hooks."""
    bc = BoundaryConditions(
        left=InletWall(), right=OutletWall(),
        bottom=NoSlipWall(), top=NoSlipWall(),
    )
    solver = Solver(
        grid_size=(16, 8), nu=0.01, dt=1.0e-4,
        boundary_config=bc, diffusion_scheme="crank_nicolson", force=True,
    )
    assert solver._diffusion is not None


def test_periodic_face_advection_is_evolved_on_both_axes():
    bc = _periodic_bc(x=True, y=True)
    solver = Solver(grid_size=(12, 10), nu=0.01, dt=1.0e-4,
                    boundary_config=bc, diffusion_scheme="explicit")
    rng = np.random.default_rng(7)
    solver.u[:] = rng.normal(size=solver.u.shape)
    solver.v[:] = rng.normal(size=solver.v.shape)
    solver.bc.apply(solver.u, solver.v, solver.Nx, solver.Ny)

    adv_u, adv_v = solver._advection_fn(solver.u, solver.v, solver.dx, solver.dy)
    assert np.any(np.abs(adv_u[0, 1:-1]) > 0.0)
    assert np.any(np.abs(adv_v[1:-1, 0]) > 0.0)
    assert np.allclose(adv_u[0, 1:-1], adv_u[-1, 1:-1])
    assert np.allclose(adv_v[1:-1, 0], adv_v[1:-1, -1])


def test_periodic_pressure_projection_has_small_discrete_residual():
    mesh = Mesh(1.0, 1.0, 12, 10)
    bc = _periodic_bc(x=True, y=True)
    pressure = create_pressure_solver(mesh, bc)
    rng = np.random.default_rng(11)
    u = rng.normal(size=mesh.shape_u)
    v = rng.normal(size=mesh.shape_v)
    bc.apply(u, v, mesh.Nx, mesh.Ny)
    dt = 1.0e-3
    p = pressure.solve(u, v, dt)

    lap = (
        (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / mesh.dx**2
        + (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / mesh.dy**2
    )
    div = (
        (u[1:, 1:-1] - u[:-1, 1:-1]) / mesh.dx
        + (v[1:-1, 1:] - v[1:-1, :-1]) / mesh.dy
    )
    assert np.max(np.abs(lap - div / dt)) < 1.0e-9
