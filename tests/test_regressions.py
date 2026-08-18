"""Regression tests for recently fixed edge cases."""

import numpy as np
import pytest

from cfd_solver.solver import Solver
from cfd_solver.solver.bc import BoundaryConditions, InletWall, NoSlipWall, OutletWall
from cfd_solver.solver.mesh import Mesh
from cfd_solver.solver.pressure import PressureSolver
from cfd_solver.validation import compute_convergence_rate


def test_p1_rejects_cn_with_inlet_outlet():
    bc = BoundaryConditions(
        left=InletWall(),
        right=OutletWall(),
        top=NoSlipWall(),
        bottom=NoSlipWall(),
    )
    with pytest.raises(ValueError, match="Crank-Nicolson"):
        Solver(
            grid_size=(16, 8),
            nu=0.01,
            dt=1e-4,
            diffusion_scheme="crank_nicolson",
            boundary_config=bc,
        )


def test_top_and_bottom_parabolic_inlets_use_x_direction():
    bc = BoundaryConditions(
        top=InletWall(profile="parabolic", U_max=2.0),
        bottom=NoSlipWall(),
        left=NoSlipWall(),
        right=NoSlipWall(),
    )
    bc._domain_width = 4.0
    bc._domain_height = 1.0
    u = np.zeros((17, 10))
    v = np.zeros((18, 9))
    bc.apply(u, v, Nx=16, Ny=8)

    assert v[1:-1, 8].shape == (16,)
    assert np.all(v[1:-1, 8] > 0.0)
    assert np.isclose(v[1:-1, 8].max(), 2.0, rtol=0.02)


def test_top_outlet_pressure_reference_is_pinned():
    mesh = Mesh(2.0, 1.0, 8, 4)
    bc = BoundaryConditions(
        top=OutletWall(),
        bottom=NoSlipWall(),
        left=NoSlipWall(),
        right=NoSlipWall(),
    )
    solver = PressureSolver(mesh, bc=bc)
    expected = set(range((mesh.Ny - 1) * mesh.Nx, mesh.Ny * mesh.Nx))
    assert expected.issubset(solver._outlet_cols)


def test_convergence_rate_supports_y_refinement():
    rates = compute_convergence_rate(
        [1.0, 0.25, 0.0625],
        [(16, 16), (16, 32), (16, 64)],
    )
    assert np.allclose(rates, [2.0, 2.0])


def test_convergence_rate_rejects_nonuniform_2d_refinement():
    rates = compute_convergence_rate(
        [1.0, 0.25],
        [(16, 16), (32, 64)],
    )
    assert rates == [None]
