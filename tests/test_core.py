"""Tests for the refactored CFD solver with ghost cells."""

import numpy as np
import pytest

from cfd_solver.solver.mesh import Mesh
from cfd_solver.solver.bc import BoundaryConditions
from cfd_solver.solver.solver import Solver
from cfd_solver.solver import advection
from cfd_solver.solver.diffusion import CrankNicolson, explicit
from cfd_solver.solver.pressure import PressureSolver
from cfd_solver.solver import diagnostics
from cfd_solver.solver.viz import save_quiver, save_contour


# ── Mesh ─────────────────────────────────────────────────────────────

def test_mesh_shapes():
    m = Mesh(1.0, 1.0, 8, 6)
    assert m.shape_u == (9, 8)  # (Nx+1, Ny+2)
    assert m.shape_v == (10, 7)  # (Nx+2, Ny+1)
    assert m.shape_p == (10, 8)  # (Nx+2, Ny+2)


def test_mesh_spacing():
    m = Mesh(2.0, 1.0, 4, 5)
    assert m.dx == 0.5
    assert m.dy == 0.2


def test_mesh_grids():
    m = Mesh(1.0, 1.0, 4, 3)
    Xc, Yc = m.cell_center_grid()
    assert Xc.shape == (4, 3)
    assert Yc.shape == (4, 3)

    Xf, Yf = m.u_face_grid()
    assert Xf.shape == (5, 3)

    Xv, Yv = m.v_face_grid()
    assert Xv.shape == (4, 4)


# ── Boundary Conditions ─────────────────────────────────────────────

def test_bc_apply_constant_lid():
    m = Mesh(1.0, 1.0, 5, 4)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(top=1.0, bottom=0.0, left=0.0, right=0.0)
    bc.apply(u, v, m.Nx, m.Ny)

    # Top ghost cell values should be 2.0 to reflect 1.0 boundary velocity
    assert np.allclose(u[:, -1], 2.0)
    assert np.allclose(u[:, 0], 0.0)
    # Normal velocity at walls must be zero
    assert np.allclose(u[0, 1:-1], 0.0)
    assert np.allclose(u[m.Nx, 1:-1], 0.0)
    assert np.allclose(v[1:-1, 0], 0.0)
    assert np.allclose(v[1:-1, m.Ny], 0.0)


def test_bc_apply_smooth_lid():
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(top=1.0, smooth_lid=True)
    bc.apply(u, v, m.Nx, m.Ny)

    # Smooth lid: zero at corners, peak in middle
    assert u[0, -1] == pytest.approx(0.0)
    assert u[-1, -1] == pytest.approx(0.0)
    assert u[m.Nx // 2, -1] == pytest.approx(2.0, abs=0.02)
    # Bottom wall still zero
    assert np.allclose(u[:, 0], 0.0)


def test_bc_lid_values():
    bc = BoundaryConditions(top=2.0, smooth_lid=True)
    lid = bc.lid_values(8)
    assert len(lid) == 9
    assert lid[0] == pytest.approx(0.0)
    assert lid[4] == pytest.approx(2.0, abs=0.01)


# ── Advection ────────────────────────────────────────────────────────

def test_upwind_returns_zeros_on_boundary():
    u = np.random.randn(9, 8)
    v = np.random.randn(10, 7)
    adv_u, adv_v = advection.upwind(u, v, 0.1, 0.1)
    # Boundary faces should be zero
    assert np.allclose(adv_u[0, :], 0.0)
    assert np.allclose(adv_u[-1, :], 0.0)
    assert np.allclose(adv_u[:, 0], 0.0)
    assert np.allclose(adv_u[:, -1], 0.0)


def test_central_returns_zeros_on_boundary():
    u = np.random.randn(9, 8)
    v = np.random.randn(10, 7)
    adv_u, adv_v = advection.central(u, v, 0.1, 0.1)
    assert np.allclose(adv_u[0, :], 0.0)
    assert np.allclose(adv_u[-1, :], 0.0)


def test_upwind_constant_field_gives_zero():
    u = np.ones((9, 8)) * 3.0
    v = np.ones((10, 7)) * 2.0
    adv_u, adv_v = advection.upwind(u, v, 0.1, 0.1)
    assert np.allclose(adv_u, 0.0, atol=1e-14)
    assert np.allclose(adv_v, 0.0, atol=1e-14)


# ── Diffusion ────────────────────────────────────────────────────────

def test_explicit_diffusion_preserves_bc():
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(top=1.0)
    bc.apply(u, v, m.Nx, m.Ny)

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    u_s, v_s = explicit(u, v, adv_u, adv_v, m.dx, m.dy, 1e-4, 0.01, bc, m.Nx, m.Ny)
    assert np.allclose(u_s[:, -1], 2.0, atol=1e-3)
    assert np.allclose(u_s[:, 0], 0.0, atol=1e-3)


def test_crank_nicolson_builds_matrices():
    m = Mesh(1.0, 1.0, 8, 6)
    bc = BoundaryConditions(top=1.0)
    cn = CrankNicolson(m, nu=0.01, dt=1e-4, bc=bc)
    assert cn.A_u.shape == ((8 - 1) * 6, (8 - 1) * 6)  # 42 x 42
    assert cn.A_v.shape == (8 * (6 - 1), 8 * (6 - 1))  # 40 x 40


# ── Pressure ─────────────────────────────────────────────────────────

def test_pressure_poisson_zero_divergence():
    m = Mesh(1.0, 1.0, 8, 6)
    ps = PressureSolver(m)

    # Uniform u_star, v_star → zero divergence → zero pressure
    u_s = np.zeros(m.shape_u)
    u_s[:, 1:-1] = 1.0
    v_s = np.zeros(m.shape_v)
    p = ps.solve(u_s, v_s, dt=0.001)
    assert np.allclose(p[1:-1, 1:-1], 0.0, atol=1e-10)


def test_pressure_zero_mean():
    m = Mesh(1.0, 1.0, 8, 6)
    ps = PressureSolver(m)

    u_s = np.random.randn(*m.shape_u) * 0.01
    v_s = np.random.randn(*m.shape_v) * 0.01
    p = ps.solve(u_s, v_s, dt=0.001)
    assert abs(np.mean(p[1:-1, 1:-1])) < 1e-10


# ── Diagnostics ──────────────────────────────────────────────────────

def test_divergence_uniform_flow():
    u = np.ones((9, 8))
    v = np.zeros((10, 7))
    div = diagnostics.divergence(u, v, 0.1, 0.1)
    assert np.allclose(div, 0.0)


def test_cfl_computation():
    u = np.ones((9, 8)) * 2.0
    v = np.ones((10, 7)) * 3.0
    c = diagnostics.cfl(u, v, 0.1, 0.1, 0.01)
    assert c == pytest.approx(0.5)


def test_is_blowup():
    u = np.ones((9, 8))
    v = np.ones((10, 7))
    assert not diagnostics.is_blowup(u, v)

    u[3, 2] = np.nan
    assert diagnostics.is_blowup(u, v)


# ── Solver (integration) ────────────────────────────────────────────

def test_solver_step_runs():
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert np.isfinite(s.p).all()


def test_solver_pressure_zero_mean():
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    assert abs(np.mean(s.p[1:-1, 1:-1])) < 1e-6


def test_solver_remains_finite():
    s = Solver(grid_size=(10, 10), nu=0.01, dt=0.001, lid_speed=1.0)
    for _ in range(5):
        s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()


def test_solver_smooth_lid():
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0, smooth_lid=True)
    s.step()
    assert np.isfinite(s.u).all()


def test_solver_divergence_decreases():
    s = Solver(grid_size=(12, 12), nu=0.01, dt=0.0005, lid_speed=1.0)
    d0 = s.max_divergence(interior_only=True)
    for _ in range(20):
        s.step()
    d1 = s.max_divergence(interior_only=True)
    assert d1 < d0 + 0.1  # divergence should not grow


def test_solver_upwind_and_central():
    for scheme in ["upwind", "central"]:
        s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4,
                   advection_scheme=scheme, lid_speed=1.0)
        s.step()
        assert np.isfinite(s.u).all()


def test_full_grid_divergence_matches_interior():
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.0005, lid_speed=1.0)
    for _ in range(50):
        s.step()
    interior = s.max_divergence(interior_only=True)
    full = s.max_divergence(interior_only=False)
    assert full < 1e-4  # should now match interior
    assert abs(full - interior) < 1e-4


# ── Viz ──────────────────────────────────────────────────────────────

def test_save_quiver(tmp_path):
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    path = tmp_path / "quiver.png"
    save_quiver(s.mesh, s.u, s.v, str(path))
    assert path.exists()


def test_save_contour(tmp_path):
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    path = tmp_path / "contour.png"
    save_contour(s.mesh, s.u, s.v, s.p, str(path))
    assert path.exists()


# ── CLI imports ──────────────────────────────────────────────────────

def test_cli_module_imports():
    import cfd_solver.cli as cli
    assert callable(cli.run)
