"""Regression tests for the v0.3.2 P1 lifecycle/restart fixes."""

import numpy as np

from cfd_solver.solver import BoundaryConditions, NoSlipWall, Solver


def _make_solver(initial_condition=None):
    bc = BoundaryConditions(
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    return Solver(
        grid_size=(8, 8),
        nu=0.01,
        dt=1.0e-4,
        boundary_config=bc,
        diffusion_scheme="explicit",
        smooth_lid=False,
        lid_speed=0.0,
        initial_condition=initial_condition,
    )


def test_initial_condition_is_applied_only_once_across_solve_calls():
    calls = []

    def initial_condition(mesh):
        calls.append(1)
        u = np.zeros(mesh.shape_u)
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        u[1:-1, 1:-1] = 0.25
        return u, v, p

    solver = _make_solver(initial_condition=initial_condition)
    solver.solve(steps=1, verbose=False)
    assert len(calls) == 1

    solver.solve(steps=1, verbose=False)
    assert len(calls) == 1


def test_checkpoint_restores_physical_time(tmp_path):
    solver = _make_solver()
    solver.time = 3.75
    solver.u[1:-1, 1:-1] = 0.2

    path = tmp_path / "state.npz"
    solver.checkpoint(str(path))

    restored = Solver.from_checkpoint(
        str(path),
        force=True,
        boundary_config=solver.bc,
    )
    assert restored.time == 3.75
    assert np.allclose(restored.u, solver.u)


def test_checkpoint_restore_does_not_reapply_initial_condition(tmp_path):
    calls = []

    def initial_condition(mesh):
        calls.append(1)
        u = np.zeros(mesh.shape_u)
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        u[1:-1, 1:-1] = 0.5
        return u, v, p

    solver = _make_solver(initial_condition=initial_condition)
    solver.solve(steps=1, verbose=False)
    assert len(calls) == 1
    solver.time = 2.0

    path = tmp_path / "state.npz"
    solver.checkpoint(str(path))

    restored = Solver.from_checkpoint(
        str(path),
        force=True,
        boundary_config=solver.bc,
        initial_condition=initial_condition,
    )
    restored.solve(steps=1, verbose=False)

    assert len(calls) == 1
    assert restored.time > 2.0
