
import sys
import os
import numpy as np

# Ensure local src is importable
sys.path.insert(0, os.path.abspath('src'))

from cfd_solver.solver.grid import Grid
from cfd_solver.solver.boundaries import BoundaryConditions
from cfd_solver.solver.solver import Solver
from cfd_solver.solver.staggered_solver import StaggeredSolver
from cfd_solver.solver.viz import _cell_center_plot_fields, save_velocity_plot


def test_grid_shapes():
    g = Grid(1.0, 1.0, 8, 6)
    assert g.shape_u == (9, 6)
    assert g.shape_v == (8, 7)
    assert g.shape_p == (8, 6)
    X = g.X
    Y = g.Y
    assert X.shape == (8, 6)
    assert Y.shape == (8, 6)
    assert g.Xv.shape == (7, 8)
    assert g.Yv.shape == (7, 8)
    assert g.Xv_T.shape == (8, 7)
    assert g.Yv_T.shape == (8, 7)


def test_pressure_zero_mean_after_step():
    s = StaggeredSolver(1.0, 1.0, 8, 6, nu=0.01, dt=1e-4,
                        u_bc={"top": 1.0, "bottom": 0.0, "left": 0.0, "right": 0.0})
    # single step should run and leave pressure with near-zero mean
    s.step()
    assert abs(np.mean(s.p)) < 1e-6


def test_divergence_nonnegative_and_finite():
    s = StaggeredSolver(1.0, 1.0, 8, 6, nu=0.01, dt=1e-4)
    s._apply_bc()
    dn = s.divergence_norm()
    md = s.max_divergence()
    assert dn >= 0 and np.isfinite(dn)
    assert md >= 0 and np.isfinite(md)


def test_cell_center_plot_fields_match_matplotlib_axis_order():
    s = StaggeredSolver(2.0, 1.0, 4, 3, nu=0.01, dt=1e-4,
                        u_bc={"top": 1.0, "bottom": 0.0, "left": 0.0, "right": 0.0})
    s._apply_bc()

    fields = _cell_center_plot_fields(s)

    assert fields["pressure"].shape == (s.Ny, s.Nx)
    assert fields["speed"].shape == (s.Ny, s.Nx)
    assert fields["u"].shape == (s.Ny, s.Nx)
    assert fields["v"].shape == (s.Ny, s.Nx)
    assert np.allclose(fields["u"][-1, :], 1.0)
    assert np.allclose(fields["u"][:, -1], [0.0, 0.0, 1.0])


def test_boundary_conditions_apply_to_staggered_sides():
    g = Grid(1.0, 1.0, 5, 4)
    u = np.zeros(g.shape_u)
    v = np.zeros(g.shape_v)
    bc = BoundaryConditions(
        top_u=1.0, top_v=2.0,
        bottom_u=3.0, bottom_v=4.0,
        left_u=5.0, left_v=6.0,
        right_u=7.0, right_v=8.0,
    )

    bc.apply(u, v)

    assert np.allclose(u[1:-1, -1], bc.top_u)
    assert np.allclose(u[1:-1, 0], bc.bottom_u)
    assert np.allclose(u[0, 1:-1], bc.left_u)
    assert np.allclose(u[-1, 1:-1], bc.right_u)
    assert np.allclose(v[1:-1, -1], bc.top_v)
    assert np.allclose(v[1:-1, 0], bc.bottom_v)
    # v is NOT defined on left/right walls (cell-center x positions)


def test_original_solver_lid_cavity_remains_finite():
    g = Grid(1.0, 1.0, 10, 10)
    s = Solver(g, nu=0.01, dt=0.001, bc=BoundaryConditions(top_u=1.0))

    for _ in range(5):
        s.step()

    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert np.isfinite(s.p).all()
    assert abs(np.mean(s.p)) < 1e-10
    assert s.divergence() < 20.0


def test_original_solver_reports_small_interior_divergence():
    g = Grid(1.0, 1.0, 12, 12)
    s = Solver(g, nu=0.01, dt=0.0005, bc=BoundaryConditions(top_u=1.0))

    for _ in range(20):
        s.step()

    assert s.divergence(interior_only=True) < 1e-2


def test_cli_module_imports():
    import cfd_solver.cli as cli

    assert callable(cli.run)


def test_save_velocity_plot_handles_staggered_velocity_shapes(tmp_path):
    g = Grid(1.0, 1.0, 8, 6)
    u = np.ones(g.shape_u)
    v = np.zeros(g.shape_v)
    path = tmp_path / "velocity.png"

    save_velocity_plot(g, u, v, str(path))

    assert path.exists()
