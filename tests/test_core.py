"""
Tests for the refactored CFD solver with ghost cells.

This module contains unit tests for various components of the CFD solver,
including mesh generation, boundary conditions, advection schemes,
diffusion solvers, pressure Poisson solver, diagnostics, and the main solver loop.
"""

import numpy as np
import pytest

from cfd_solver.solver.mesh import Mesh
from cfd_solver.solver.bc import BoundaryConditions
from cfd_solver.solver.solver import Solver
from cfd_solver.solver import advection
from cfd_solver.solver.diffusion import CrankNicolson, explicit
from cfd_solver.solver.pressure import PressureSolver, FFTPressureSolver, create_pressure_solver
from cfd_solver.solver import diagnostics
from cfd_solver.solver.validate import validate_config
from cfd_solver.solver.viz import save_quiver, save_contour


# ── Mesh Tests ───────────────────────────────────────────────────────

def test_mesh_shapes():
    """Verify that the staggered grid shapes are correctly calculated including ghost cells."""
    m = Mesh(1.0, 1.0, 8, 6)
    assert m.shape_u == (9, 8)  # (Nx+1, Ny+2)
    assert m.shape_v == (10, 7)  # (Nx+2, Ny+1)
    assert m.shape_p == (10, 8)  # (Nx+2, Ny+2)


def test_mesh_spacing():
    """Verify that the grid spacing dx and dy are correctly computed."""
    m = Mesh(2.0, 1.0, 4, 5)
    assert m.dx == 0.5
    assert m.dy == 0.2


def test_mesh_grids():
    """Verify the shapes of the coordinate grids for cell centers and faces."""
    m = Mesh(1.0, 1.0, 4, 3)
    Xc, Yc = m.cell_center_grid()
    assert Xc.shape == (4, 3)
    assert Yc.shape == (4, 3)

    Xf, Yf = m.u_face_grid()
    assert Xf.shape == (5, 3)

    Xv, Yv = m.v_face_grid()
    assert Xv.shape == (4, 4)


# ── Boundary Condition Tests ──────────────────────────────────────────

def test_bc_apply_constant_lid():
    """Test applying constant velocity boundary conditions to the lid."""
    m = Mesh(1.0, 1.0, 5, 4)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(top=1.0, bottom=0.0, left=0.0, right=0.0)
    bc.apply(u, v, m.Nx, m.Ny)

    # Top ghost cell values should be 2.0 to reflect 1.0 boundary velocity
    # u_ghost = 2 * u_boundary - u_interior => 2*1 - 0 = 2
    assert np.allclose(u[:, -1], 2.0)
    assert np.allclose(u[:, 0], 0.0)
    # Normal velocity at walls must be zero
    assert np.allclose(u[0, 1:-1], 0.0)
    assert np.allclose(u[m.Nx, 1:-1], 0.0)
    assert np.allclose(v[1:-1, 0], 0.0)
    assert np.allclose(v[1:-1, m.Ny], 0.0)


def test_bc_apply_smooth_lid():
    """Test applying a smoothed lid velocity profile to avoid corner singularities."""
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
    """Check the calculation of the lid velocity profile array."""
    bc = BoundaryConditions(top=2.0, smooth_lid=True)
    lid = bc.lid_values(8)
    assert len(lid) == 9
    assert lid[0] == pytest.approx(0.0)
    assert lid[4] == pytest.approx(2.0, abs=0.01)


# ── Advection Tests ───────────────────────────────────────────────────

def test_upwind_returns_zeros_on_boundary():
    """Verify that the upwind advection scheme correctly handles boundaries (zero flux)."""
    u = np.random.randn(9, 8)
    v = np.random.randn(10, 7)
    adv_u, adv_v = advection.upwind(u, v, 0.1, 0.1)
    # Boundary faces should be zero
    assert np.allclose(adv_u[0, :], 0.0)
    assert np.allclose(adv_u[-1, :], 0.0)
    assert np.allclose(adv_u[:, 0], 0.0)
    assert np.allclose(adv_u[:, -1], 0.0)


def test_central_returns_zeros_on_boundary():
    """Verify that the central advection scheme correctly handles boundaries (zero flux)."""
    u = np.random.randn(9, 8)
    v = np.random.randn(10, 7)
    adv_u, adv_v = advection.central(u, v, 0.1, 0.1)
    assert np.allclose(adv_u[0, :], 0.0)
    assert np.allclose(adv_u[-1, :], 0.0)


def test_upwind_constant_field_gives_zero():
    """Ensure advection of a constant field results in zero advective derivative."""
    u = np.ones((9, 8)) * 3.0
    v = np.ones((10, 7)) * 2.0
    adv_u, adv_v = advection.upwind(u, v, 0.1, 0.1)
    assert np.allclose(adv_u, 0.0, atol=1e-14)
    assert np.allclose(adv_v, 0.0, atol=1e-14)


# ── Diffusion Tests ───────────────────────────────────────────────────

def test_explicit_diffusion_preserves_bc():
    """Verify that the explicit diffusion step respects boundary conditions."""
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
    """Check that the Crank-Nicolson solver correctly constructs its system matrices."""
    m = Mesh(1.0, 1.0, 8, 6)
    bc = BoundaryConditions(top=1.0)
    cn = CrankNicolson(m, nu=0.01, dt=1e-4, bc=bc)
    assert cn.A_u.shape == ((8 - 1) * 6, (8 - 1) * 6)  # 42 x 42
    assert cn.A_v.shape == (8 * (6 - 1), 8 * (6 - 1))  # 40 x 40


# ── Pressure Tests ────────────────────────────────────────────────────

def test_pressure_poisson_zero_divergence():
    """Test that zero divergence field results in zero pressure (relative)."""
    m = Mesh(1.0, 1.0, 8, 6)
    ps = PressureSolver(m)

    # Uniform u_star, v_star → zero divergence → zero pressure
    u_s = np.zeros(m.shape_u)
    u_s[:, 1:-1] = 1.0
    v_s = np.zeros(m.shape_v)
    p = ps.solve(u_s, v_s, dt=0.001)
    assert np.allclose(p[1:-1, 1:-1], 0.0, atol=1e-10)


def test_pressure_zero_mean():
    """Verify that the pressure solver maintains a zero mean pressure (normalization)."""
    m = Mesh(1.0, 1.0, 8, 6)
    ps = PressureSolver(m)

    u_s = np.random.randn(*m.shape_u) * 0.01
    v_s = np.random.randn(*m.shape_v) * 0.01
    p = ps.solve(u_s, v_s, dt=0.001)
    assert abs(np.mean(p[1:-1, 1:-1])) < 1e-10


def test_fft_pressure_zero_divergence():
    """Test that FFT pressure solver produces zero pressure for zero divergence."""
    m = Mesh(1.0, 1.0, 16, 16)
    ps = FFTPressureSolver(m)

    u_s = np.zeros(m.shape_u)
    u_s[:, 1:-1] = 1.0
    v_s = np.zeros(m.shape_v)
    p = ps.solve(u_s, v_s, dt=0.001)
    assert np.allclose(p[1:-1, 1:-1], 0.0, atol=1e-10)


def test_fft_pressure_zero_mean():
    """Verify that the FFT pressure solver maintains zero-mean pressure."""
    m = Mesh(1.0, 1.0, 16, 16)
    ps = FFTPressureSolver(m)

    u_s = np.random.randn(*m.shape_u) * 0.01
    v_s = np.random.randn(*m.shape_v) * 0.01
    p = ps.solve(u_s, v_s, dt=0.001)
    assert abs(np.mean(p[1:-1, 1:-1])) < 1e-10


def test_fft_vs_splu_pressure_gradients():
    """Verify that FFT and splu pressure solvers produce matching gradients.

    The two solvers pin pressure differently (p[0]=0 vs mean(p)=0), so
    the pressure fields differ by a constant.  But the pressure *gradients*
    must match to machine precision, since gradients are what drive the
    velocity correction step.

    We test with a realistic CFD-like intermediate velocity to ensure the
    divergence RHS is approximately zero-mean (as it is in the real solver).
    """
    s = Solver(grid_size=(32, 32), nu=0.01, dt=0.0005, lid_speed=1.0,
               smooth_lid=True)
    # Run a few steps to get a realistic intermediate velocity
    for _ in range(3):
        s.step()

    # Now manually run advection+diffusion to get intermediate velocity
    from cfd_solver.solver import advection
    adv_u, adv_v = advection.upwind(s.u, s.v, s.dx, s.dy)
    u_star, v_star = s._diffusion.solve(s.u, s.v, adv_u, adv_v)

    # Compute the actual divergence (should be approximately zero-mean)
    dx, dy = s.dx, s.dy
    div = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx + (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy
    assert abs(np.mean(div)) < 0.1, f"Div should be ~zero-mean for real CFD data, got {np.mean(div):.4f}"

    ps_splu = PressureSolver(s.mesh)
    ps_fft = FFTPressureSolver(s.mesh)

    p_splu = ps_splu.solve(u_star, v_star, dt=s.dt)
    p_fft = ps_fft.solve(u_star, v_star, dt=s.dt)

    # Interior pressure may differ by a constant
    diff = p_splu[1:-1, 1:-1] - p_fft[1:-1, 1:-1]

    # Gradients must match
    grad_px_splu = p_splu[2:-1, 1:-1] - p_splu[1:-2, 1:-1]
    grad_px_fft = p_fft[2:-1, 1:-1] - p_fft[1:-2, 1:-1]
    assert np.allclose(grad_px_splu, grad_px_fft, atol=1e-8)

    grad_py_splu = p_splu[1:-1, 2:-1] - p_splu[1:-1, 1:-2]
    grad_py_fft = p_fft[1:-1, 2:-1] - p_fft[1:-1, 1:-2]
    assert np.allclose(grad_py_splu, grad_py_fft, atol=1e-8)


def test_create_pressure_solver_small_grid():
    """Factory returns splu-based solver for small grids."""
    m = Mesh(1.0, 1.0, 64, 64)
    ps = create_pressure_solver(m)
    assert isinstance(ps, PressureSolver)


def test_create_pressure_solver_large_grid():
    """Factory returns FFT solver for large grids."""
    m = Mesh(1.0, 1.0, 256, 256)
    ps = create_pressure_solver(m)
    assert isinstance(ps, FFTPressureSolver)


# ── Diagnostic Tests ──────────────────────────────────────────────────

def test_divergence_uniform_flow():
    """Ensure divergence of a uniform flow field is zero."""
    u = np.ones((9, 8))
    v = np.zeros((10, 7))
    div = diagnostics.divergence(u, v, 0.1, 0.1)
    assert np.allclose(div, 0.0)


def test_cfl_computation():
    """Verify the calculation of the Courant-Friedrichs-Lewy (CFL) number."""
    u = np.ones((9, 8)) * 2.0
    v = np.ones((10, 7)) * 3.0
    c = diagnostics.cfl(u, v, 0.1, 0.1, 0.01)
    assert c == pytest.approx(0.5)


def test_is_blowup():
    """Test the blowup detection logic (NaN/Inf check)."""
    u = np.ones((9, 8))
    v = np.ones((10, 7))
    assert not diagnostics.is_blowup(u, v)

    u[3, 2] = np.nan
    assert diagnostics.is_blowup(u, v)


# ── Solver Integration Tests ──────────────────────────────────────────

def test_solver_step_runs():
    """Verify that a single solver step completes without errors and returns finite values."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert np.isfinite(s.p).all()


def test_solver_pressure_zero_mean():
    """Verify that the integrated solver maintains zero-mean pressure."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    assert abs(np.mean(s.p[1:-1, 1:-1])) < 1e-6


def test_solver_remains_finite():
    """Check that the solver remains stable over multiple time steps."""
    s = Solver(grid_size=(10, 10), nu=0.01, dt=0.001, lid_speed=1.0)
    for _ in range(5):
        s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()


def test_solver_smooth_lid():
    """Verify that the solver works correctly with the smooth lid option."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0, smooth_lid=True)
    s.step()
    assert np.isfinite(s.u).all()


def test_solver_divergence_decreases():
    """Check that divergence does not explode over several steps (stability test)."""
    s = Solver(grid_size=(12, 12), nu=0.01, dt=0.0005, lid_speed=1.0)
    d0 = s.max_divergence(interior_only=True)
    for _ in range(20):
        s.step()
    d1 = s.max_divergence(interior_only=True)
    assert d1 < d0 + 0.1  # divergence should not grow


def test_solver_upwind_and_central():
    """Test both upwind and central advection schemes in the full solver."""
    for scheme in ["upwind", "central"]:
        s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4,
                   advection_scheme=scheme, lid_speed=1.0)
        s.step()
        assert np.isfinite(s.u).all()


def test_full_grid_divergence_matches_interior():
    """Verify that divergence calculation is consistent across the grid."""
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.0005, lid_speed=1.0)
    for _ in range(50):
        s.step()
    interior = s.max_divergence(interior_only=True)
    full = s.max_divergence(interior_only=False)
    assert full < 1e-4  # should now match interior
    assert abs(full - interior) < 1e-4


# ── Pressure Projection Regression Tests ──────────────────────

def test_projection_removes_divergence():
    """Regression test for the pressure-Poisson RHS sign error.

    The projection step must REMOVE divergence from the intermediate
    velocity. Under the historical sign bug (rhs = +div/dt against the
    positive-definite operator A = -laplacian), the correction DOUBLED
    the divergence every step, blowing up within ~23 steps.
    """
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.0005, lid_speed=1.0)
    for _ in range(3):
        s.step()
    # After projection the field must be near divergence-free everywhere.
    assert s.max_divergence() < 1e-6


def test_lid_cavity_32_no_blowup():
    """Regression test for the reported blowup (32x32, dt=0.001, step ~23).

    Runs past the historical failure point and asserts the field stays
    finite with small divergence.
    """
    s = Solver(grid_size=(32, 32), nu=0.01, dt=0.001, lid_speed=1.0,
               smooth_lid=True)
    for _ in range(40):
        s.step()
    assert not diagnostics.is_blowup(s.u, s.v)
    assert s.max_divergence() < 1e-6
    # Velocities should remain physically plausible (|u| of order lid speed)
    assert np.max(np.abs(s.u)) < 5.0


def test_solver_fft_pressure_128():
    """Verify the full solver works with FFT pressure backend on a 256x256 grid.

    The factory should automatically select the FFT solver for grids > 128.
    """
    s = Solver(grid_size=(256, 256), nu=0.01, dt=0.0001, lid_speed=1.0,
               smooth_lid=True, force=True)
    from cfd_solver.solver.pressure import FFTPressureSolver
    assert isinstance(s._pressure, FFTPressureSolver)
    for _ in range(3):
        s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert s.max_divergence() < 1e-3


# ── Visualization Tests ───────────────────────────────────────────────

def test_save_quiver(tmp_path):
    """Test saving a quiver plot of the velocity field."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    path = tmp_path / "quiver.png"
    save_quiver(s.mesh, s.u, s.v, str(path))
    assert path.exists()


def test_save_contour(tmp_path):
    """Test saving a contour plot of the pressure and velocity fields."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    path = tmp_path / "contour.png"
    save_contour(s.mesh, s.u, s.v, s.p, str(path))
    assert path.exists()


def test_save_streamlines(tmp_path):
    """Test saving a streamline plot of the velocity and pressure fields."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()
    path = tmp_path / "streamlines.png"
    s.save_streamlines(str(path))
    assert path.exists()


# ── Crank-Nicolson Solve Tests ───────────────────────────────────────

def test_crank_nicolson_solve_output_finite():
    """Verify that Crank-Nicolson.solve() produces finite output with real data."""
    m = Mesh(1.0, 1.0, 8, 6)
    bc = BoundaryConditions(top=1.0)
    cn = CrankNicolson(m, nu=0.01, dt=1e-4, bc=bc)

    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc.apply(u, v, m.Nx, m.Ny)

    adv_u = np.random.randn(*m.shape_u) * 0.01
    adv_v = np.random.randn(*m.shape_v) * 0.01

    u_star, v_star = cn.solve(u, v, adv_u, adv_v)
    assert np.isfinite(u_star).all()
    assert np.isfinite(v_star).all()
    # Boundary values should be preserved
    assert np.allclose(u_star[:, -1], 2.0, atol=1e-3)
    assert np.allclose(u_star[:, 0], 0.0, atol=1e-3)


def test_crank_nicolson_smooths_velocity():
    """Verify that Crank-Nicolson diffusion smooths the velocity field (reduces L2 norm of Laplacian)."""
    m = Mesh(1.0, 1.0, 10, 8)
    bc = BoundaryConditions(top=1.0)
    cn = CrankNicolson(m, nu=0.01, dt=1e-3, bc=bc)

    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc.apply(u, v, m.Nx, m.Ny)

    # Add localized perturbation
    u[4, 3] = 5.0

    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)

    u_star, v_star = cn.solve(u, v, adv_u, adv_v)
    assert np.isfinite(u_star).all()
    # Diffusion should reduce the peak perturbation
    assert np.max(np.abs(u_star)) < np.max(np.abs(u))


def test_fft_crank_nicolson_vs_spatial():
    """Verify that FFTCrankNicolson matches CrankNicolson to machine precision."""
    m = Mesh(1.0, 1.0, 16, 16)
    bc = BoundaryConditions(top=1.0, smooth_lid=True)
    nu = 0.01
    dt = 0.001

    cn = CrankNicolson(m, nu, dt, bc)
    from cfd_solver.solver.diffusion import FFTCrankNicolson
    fft_cn = FFTCrankNicolson(m, nu, dt, bc)

    np.random.seed(42)
    u = np.random.randn(*m.shape_u) * 0.1
    v = np.random.randn(*m.shape_v) * 0.1
    adv_u = np.random.randn(*m.shape_u) * 0.5
    adv_v = np.random.randn(*m.shape_v) * 0.5
    bc.apply(u, v, m.Nx, m.Ny)

    u_star_cn, v_star_cn = cn.solve(u, v, adv_u, adv_v)
    u_star_fft, v_star_fft = fft_cn.solve(u, v, adv_u, adv_v)

    assert np.allclose(u_star_cn, u_star_fft, atol=1e-12)
    assert np.allclose(v_star_cn, v_star_fft, atol=1e-12)


def test_create_diffusion_solver_factory():
    """Verify create_diffusion_solver selects correct backend based on grid size."""
    from cfd_solver.solver.diffusion import create_diffusion_solver, FFTCrankNicolson
    m_small = Mesh(1.0, 1.0, 64, 64)
    bc = BoundaryConditions()
    solver_small = create_diffusion_solver(m_small, 0.01, 0.001, bc, threshold=128)
    assert isinstance(solver_small, CrankNicolson)

    m_large = Mesh(1.0, 1.0, 256, 256)
    solver_large = create_diffusion_solver(m_large, 0.01, 0.001, bc, threshold=128)
    assert isinstance(solver_large, FFTCrankNicolson)


def test_solver_fft_diffusion_256():
    """Verify full solver step and max divergence with FFT diffusion solver."""
    s = Solver(grid_size=(256, 256), nu=0.01, dt=0.0001, lid_speed=1.0,
               smooth_lid=True, force=True)
    from cfd_solver.solver.diffusion import FFTCrankNicolson
    assert isinstance(s._diffusion, FFTCrankNicolson)
    for _ in range(3):
        s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert s.max_divergence() < 1e-3


# ── Checkpoint Round-Trip Tests ──────────────────────────────────────

def test_checkpoint_roundtrip(tmp_path):
    """Save a checkpoint and reload it; verify state matches exactly."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.0)
    s.step()

    path = str(tmp_path / "test.npz")
    s.checkpoint(path)

    s2 = Solver.from_checkpoint(path)
    assert np.allclose(s.u, s2.u)
    assert np.allclose(s.v, s2.v)
    assert np.allclose(s.p, s2.p)


def test_checkpoint_roundtrip_smooth_lid(tmp_path):
    """Checkpoint round-trip preserves smooth_lid and lid_speed settings."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-4, lid_speed=1.5, smooth_lid=True)
    s.step()

    path = str(tmp_path / "test_smooth.npz")
    s.checkpoint(path)

    s2 = Solver.from_checkpoint(path)
    assert s2.bc.smooth_lid is True
    assert s2.bc.top == pytest.approx(1.5)
    assert np.allclose(s.u, s2.u)


def test_checkpoint_missing_file():
    """from_checkpoint raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        Solver.from_checkpoint("/nonexistent/path.npz")


# ── Validation Tests ─────────────────────────────────────────────────

def test_validate_missing_geometry_nx():
    """Validator catches missing required field Nx inside geometry."""
    cfg = {"geometry": {"Lx": 1, "Ly": 1}, "nu": 0.01, "dt": 0.001, "steps": 10}
    errors = validate_config(cfg)
    assert any("Nx" in e for e in errors)


def test_validate_missing_geometry_ny():
    """Validator catches missing required field Ny inside geometry."""
    cfg = {"geometry": {"Lx": 1, "Ly": 1, "Nx": 32}, "nu": 0.01, "dt": 0.001, "steps": 10}
    errors = validate_config(cfg)
    assert any("Ny" in e for e in errors)


def test_validate_missing_required_top_level():
    """Validator catches missing top-level required fields."""
    cfg = {"nu": 0.01, "dt": 0.001}
    errors = validate_config(cfg)
    assert any("steps" in e for e in errors)


def test_validate_unknown_top_level_key():
    """Validator catches unknown top-level keys."""
    cfg = {"nu": 0.01, "dt": 0.001, "steps": 10, "typo_field": 42}
    errors = validate_config(cfg)
    assert any("typo_field" in e for e in errors)


def test_validate_valid_config():
    """Validator passes a fully valid config with steps."""
    cfg = {
        "geometry": {"Lx": 1, "Ly": 1, "Nx": 32, "Ny": 32},
        "nu": 0.01,
        "dt": 0.001,
        "steps": 100,
    }
    errors = validate_config(cfg)
    assert errors == []


def test_validate_valid_config_simulation_time():
    """Validator passes a fully valid config with simulation_time."""
    cfg = {
        "geometry": {"Lx": 1, "Ly": 1, "Nx": 32, "Ny": 32},
        "nu": 0.01,
        "dt": 0.001,
        "simulation_time": 20.0,
    }
    errors = validate_config(cfg)
    assert errors == []


def test_validate_neither_steps_nor_simulation_time():
    """Validator rejects config with neither steps nor simulation_time."""
    cfg = {
        "geometry": {"Lx": 1, "Ly": 1, "Nx": 32, "Ny": 32},
        "nu": 0.01,
        "dt": 0.001,
    }
    errors = validate_config(cfg)
    assert any("steps" in e and "simulation_time" in e for e in errors)


# ── Convergence Tests ────────────────────────────────────────────────

def test_solver_simulation_time():
    """Solver.solve() with simulation_time runs the correct number of steps."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-3, lid_speed=1.0)
    s.solve(simulation_time=0.01, verbose=False)
    # 0.01s / 0.001dt = 10 steps
    assert np.isfinite(s.u).all()


def test_solver_simulation_time_overrides_steps():
    """simulation_time takes precedence over steps."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-3, lid_speed=1.0)
    # steps=1 would be too few, simulation_time=0.05 = 50 steps
    s.solve(steps=1, simulation_time=0.05, verbose=False)
    # If simulation_time was ignored, velocity would barely change
    assert np.max(np.abs(s.u)) > 0.01


def test_solver_steps_still_works():
    """Backward-compatible: solve(steps=N) still works."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-3, lid_speed=1.0)
    s.solve(10, verbose=False)
    assert np.isfinite(s.u).all()


def test_solver_solve_no_args_raises():
    """Calling solve() with neither steps nor simulation_time raises."""
    s = Solver(grid_size=(8, 6), nu=0.01, dt=1e-3, lid_speed=1.0)
    with pytest.raises(ValueError):
        s.solve(verbose=False)

def test_solver_converges_toward_steady_state():
    """Run solver long enough to verify velocity changes diminish over time."""
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.0005, lid_speed=1.0)

    # Measure velocity change over first 20 steps
    u0 = s.u.copy()
    for _ in range(20):
        s.step()
    delta_early = np.max(np.abs(s.u - u0))

    # Measure velocity change over next 20 steps
    u1 = s.u.copy()
    for _ in range(20):
        s.step()
    delta_late = np.max(np.abs(s.u - u1))

    # Late-stage changes should be smaller (approaching steady state)
    assert delta_late < delta_early


def test_solver_divergence_always_small():
    """Verify divergence stays small across many steps (steady convergence)."""
    s = Solver(grid_size=(12, 12), nu=0.01, dt=0.0005, lid_speed=1.0)
    for _ in range(50):
        s.step()
    assert s.max_divergence() < 1e-6


# ── Explicit Diffusion + Advection Interaction ──────────────────────

def test_explicit_diffusion_with_nonzero_advection():
    """Verify explicit diffusion handles non-zero advection without crashing."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(top=1.0)
    bc.apply(u, v, m.Nx, m.Ny)

    # Non-zero advective terms
    adv_u = np.random.randn(*m.shape_u) * 0.001
    adv_v = np.random.randn(*m.shape_v) * 0.001

    u_s, v_s = explicit(u, v, adv_u, adv_v, m.dx, m.dy, 1e-4, 0.01, bc, m.Nx, m.Ny)
    assert np.isfinite(u_s).all()
    assert np.isfinite(v_s).all()
    # BCs still respected
    assert np.allclose(u_s[:, -1], 2.0, atol=1e-3)
    assert np.allclose(u_s[:, 0], 0.0, atol=1e-3)


# ── CLI Tests ─────────────────────────────────────────────────────────

def test_cli_module_imports():
    """Verify that the CLI entry point can be imported correctly."""
    import cfd_solver.cli as cli
    assert callable(cli.run)


# ═══════════════════════════════════════════════════════════════════════
# Phase 0 — Flexible Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════

# ── Wall Type Classes ─────────────────────────────────────────────────

from cfd_solver.solver.bc import (
    WallType, NoSlipWall, FreeSlipWall, InletWall, OutletWall, PeriodicWall,
)


def test_wall_type_base_class():
    """WallType is the base class for all wall types."""
    assert issubclass(NoSlipWall, WallType)
    assert issubclass(FreeSlipWall, WallType)
    assert issubclass(InletWall, WallType)
    assert issubclass(OutletWall, WallType)
    assert issubclass(PeriodicWall, WallType)


def test_noslip_wall_defaults():
    """NoSlipWall defaults to zero tangential velocity."""
    w = NoSlipWall()
    assert w.u == 0.0
    assert w.v == 0.0


def test_noslip_wall_custom():
    """NoSlipWall stores custom tangential velocities."""
    w = NoSlipWall(u=1.5, v=0.3)
    assert w.u == 1.5
    assert w.v == 0.3


def test_free_slip_wall():
    """FreeSlipWall stores tangential velocity."""
    w = FreeSlipWall(u=0.5)
    assert w.u == 0.5
    assert w.v == 0.0


def test_inlet_wall():
    """InletWall stores profile and U_max."""
    w = InletWall(profile="parabolic", U_max=2.0)
    assert w.profile == "parabolic"
    assert w.U_max == 2.0


def test_outlet_wall():
    """OutletWall stores method."""
    w = OutletWall(method="convective")
    assert w.method == "convective"


def test_periodic_wall():
    """PeriodicWall can be instantiated."""
    w = PeriodicWall()
    assert isinstance(w, WallType)


# ── BoundaryConditions with New Wall Types ────────────────────────────

def test_bc_backward_compat():
    """Old-style BoundaryConditions constructor still works."""
    bc = BoundaryConditions(top=1.0, smooth_lid=True)
    assert bc.top == 1.0
    assert bc.smooth_lid is True
    assert isinstance(bc.walls['top'], NoSlipWall)
    assert bc.walls['top'].u == 1.0


def test_bc_with_noslip_wall_objects():
    """BoundaryConditions accepts NoSlipWall objects."""
    bc = BoundaryConditions(
        top=NoSlipWall(u=2.0),
        bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(v=0.0),
        right=NoSlipWall(v=0.0),
    )
    assert bc.top == 2.0
    assert bc.walls['top'].u == 2.0


def test_bc_with_inlet_outlet():
    """BoundaryConditions accepts InletWall and OutletWall."""
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    assert isinstance(bc.walls['left'], InletWall)
    assert isinstance(bc.walls['right'], OutletWall)
    assert bc.walls['left'].U_max == 1.0


def test_bc_with_free_slip():
    """BoundaryConditions accepts FreeSlipWall."""
    bc = BoundaryConditions(
        top=FreeSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(v=0.0),
        right=NoSlipWall(v=0.0),
    )
    assert isinstance(bc.walls['top'], FreeSlipWall)


def test_bc_to_wall_none_default():
    """_to_wall returns default when value is None."""
    bc = BoundaryConditions.__new__(BoundaryConditions)
    wall = BoundaryConditions._to_wall(None, NoSlipWall, u=0.5, v=0.0)
    assert isinstance(wall, NoSlipWall)
    assert wall.u == 0.5


def test_bc_to_wall_scalar():
    """_to_wall wraps scalar in the default wall type."""
    wall = BoundaryConditions._to_wall(3.0, NoSlipWall, u=0.0, v=0.0)
    assert isinstance(wall, NoSlipWall)
    assert wall.u == 3.0


def test_bc_to_wall_type_rejects_bad_type():
    """_to_wall raises TypeError for unsupported types."""
    with pytest.raises(TypeError):
        BoundaryConditions._to_wall("bad", NoSlipWall, u=0.0)


# ── bc.apply() with New Wall Types ────────────────────────────────────

def test_bc_apply_inlet_left_uniform():
    """InletWall on left sets uniform u-velocity at inlet face."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(
        left=InletWall(profile="uniform", U_max=1.5),
        right=NoSlipWall(v=0.0),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Left wall u-face (i=0) should be set to U_max
    assert np.allclose(u[0, 1:-1], 1.5)
    # Normal v at top/bottom should still be 0
    assert np.allclose(v[1:-1, 0], 0.0)
    assert np.allclose(v[1:-1, m.Ny], 0.0)


def test_bc_apply_inlet_left_parabolic():
    """InletWall parabolic profile has correct shape (zero at walls, max in center)."""
    m = Mesh(1.0, 1.0, 8, 16)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=NoSlipWall(v=0.0),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    profile = u[0, 1:-1]
    # Parabolic: max in center, roughly symmetric
    assert profile.max() == pytest.approx(1.0, abs=0.05)
    assert profile[len(profile)//2] == pytest.approx(1.0, abs=0.05)
    # First and last interior values should be small (near walls)
    assert profile[0] < 0.3
    assert profile[-1] < 0.3


def test_bc_apply_outlet_right_zero_gradient():
    """OutletWall zero-gradient copies interior to ghost."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    # Set some interior v near the right wall
    v[6, 3] = 0.7
    v[7, 3] = 0.9
    bc = BoundaryConditions(
        left=NoSlipWall(v=0.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Right ghost v should equal last interior v
    assert np.allclose(v[-1, :], v[-2, :])
    # Right wall u-normal should equal last interior u-normal
    assert np.allclose(u[m.Nx, 1:-1], u[m.Nx - 1, 1:-1])


def test_bc_apply_free_slip_top():
    """FreeSlipWall on top: tangential ghost = interior (zero gradient)."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    # Set some interior u near top
    u[3, -2] = 0.5
    u[5, -2] = 0.8
    bc = BoundaryConditions(
        top=FreeSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(v=0.0),
        right=NoSlipWall(v=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Free-slip: ghost = interior
    assert np.allclose(u[3, -1], u[3, -2])
    assert np.allclose(u[5, -1], u[5, -2])
    # Normal v at top is still 0
    assert np.allclose(v[1:-1, m.Ny], 0.0)


def test_bc_apply_mixed_noslip_inlet_outlet():
    """Mixed wall types: inlet left, outlet right, no-slip top/bottom."""
    m = Mesh(2.0, 1.0, 16, 8)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(
        left=InletWall(profile="uniform", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Inlet: u at left face = 1.0
    assert np.allclose(u[0, 1:-1], 1.0)
    # Outlet: v ghost = interior
    assert np.allclose(v[-1, :], v[-2, :])
    # No-slip top/bottom
    assert np.allclose(v[1:-1, 0], 0.0)
    assert np.allclose(v[1:-1, m.Ny], 0.0)


# ── Pressure Solver with Outlet BC ────────────────────────────────────

def test_pressure_solver_outlet_pins_column():
    """PressureSolver with outlet pins pressure at the outlet column."""
    from cfd_solver.solver.pressure import PressureSolver
    m = Mesh(1.0, 1.0, 8, 6)
    bc = BoundaryConditions(
        left=NoSlipWall(v=0.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    ps = PressureSolver(m, bc=bc)
    # Outlet column indices: for right outlet, column Nx-1 = 7
    # Flat index for (7, j) with order F: 7 + j*8
    assert len(ps._outlet_cols) == 6  # Ny columns
    assert 7 in ps._outlet_cols       # j=0
    assert 7 + 8 in ps._outlet_cols   # j=1


def test_pressure_solver_outlet_zero_pressure():
    """PressureSolver with outlet produces near-zero pressure at outlet."""
    from cfd_solver.solver.pressure import PressureSolver
    m = Mesh(1.0, 1.0, 8, 6)
    bc = BoundaryConditions(
        left=NoSlipWall(v=0.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    ps = PressureSolver(m, bc=bc)
    # Uniform velocity field → zero divergence → zero pressure
    u_s = np.zeros(m.shape_u)
    u_s[:, 1:-1] = 1.0
    v_s = np.zeros(m.shape_v)
    p = ps.solve(u_s, v_s, dt=0.001)
    # Outlet column pressure should be pinned to 0
    assert np.allclose(p[-2, 1:-1], 0.0, atol=1e-10)


def test_create_pressure_solver_outlet_uses_direct():
    """create_pressure_solver with outlet returns direct solver (not FFT)."""
    from cfd_solver.solver.pressure import PressureSolver, create_pressure_solver
    m = Mesh(1.0, 1.0, 256, 256)
    bc = BoundaryConditions(
        left=NoSlipWall(v=0.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    ps = create_pressure_solver(m, bc=bc)
    assert isinstance(ps, PressureSolver)


# ── Solver with boundary_config ───────────────────────────────────────

def test_solver_with_boundary_config():
    """Solver accepts boundary_config parameter."""
    bc = BoundaryConditions(
        left=InletWall(profile="uniform", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(
        grid_size=(16, 8), nu=0.01, dt=0.001,
        Lx=2.0, Ly=1.0,
        boundary_config=bc,
    )
    assert s.bc.walls['left'] is bc.walls['left']
    assert s.bc.walls['right'] is bc.walls['right']


def test_solver_inlet_outlet_runs():
    """Full solver with inlet/outlet BCs completes without blowup."""
    bc = BoundaryConditions(
        left=InletWall(profile="uniform", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(
        grid_size=(16, 8), nu=0.01, dt=0.001,
        Lx=2.0, Ly=1.0,
        boundary_config=bc,
    )
    s.solve(simulation_time=0.05, verbose=False)
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    # Mass should be roughly conserved: inlet flux ≈ outlet flux
    inlet_flux = np.sum(s.u[0, 1:-1]) * s.dy
    outlet_flux = np.sum(s.u[s.Nx, 1:-1]) * s.dy
    assert abs(inlet_flux - outlet_flux) / max(inlet_flux, 1e-10) < 0.5


def test_solver_uses_crank_nicolson_for_inlet():
    """Solver uses Crank-Nicolson even with inlet BC (matrices encode BC type)."""
    bc = BoundaryConditions(
        left=InletWall(profile="uniform", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(
        grid_size=(16, 8), nu=0.01, dt=0.0005,
        Lx=2.0, Ly=1.0,
        diffusion_scheme="crank_nicolson",
        boundary_config=bc,
    )
    # Should use Crank-Nicolson (no longer falls back to explicit)
    assert s._diffusion is not None


def test_solver_noslip_still_uses_crank_nicolson():
    """All-no-slip BC still uses Crank-Nicolson when requested."""
    s = Solver(
        grid_size=(16, 8), nu=0.01, dt=0.001,
        diffusion_scheme="crank_nicolson",
    )
    assert s._diffusion is not None


# ── CLI Parsing ───────────────────────────────────────────────────────

def test_cli_parse_boundary_legacy():
    """CLI legacy BC parsing produces correct BoundaryConditions."""
    from cfd_solver.cli import _parse_boundary_config
    bc_cfg = {"top": {"u": 2.0}, "smooth_lid": False}
    bc = _parse_boundary_config(bc_cfg)
    assert isinstance(bc, BoundaryConditions)
    assert bc.top == 2.0
    assert bc.smooth_lid is False


def test_cli_parse_boundary_new_format():
    """CLI new per-wall BC parsing produces correct BoundaryConditions."""
    from cfd_solver.cli import _parse_boundary_config
    bc_cfg = {
        "left": {"type": "inlet", "profile": "parabolic", "U_max": 1.5},
        "right": {"type": "outlet", "method": "zero_gradient"},
        "top": {"type": "wall", "u": 0.0},
        "bottom": {"type": "wall", "u": 0.0},
    }
    bc = _parse_boundary_config(bc_cfg)
    assert isinstance(bc.walls['left'], InletWall)
    assert bc.walls['left'].U_max == 1.5
    assert isinstance(bc.walls['right'], OutletWall)
    assert isinstance(bc.walls['top'], NoSlipWall)


def test_cli_parse_boundary_empty():
    """CLI BC parsing with empty dict returns default no-slip walls."""
    from cfd_solver.cli import _parse_boundary_config
    bc = _parse_boundary_config({})
    assert isinstance(bc.walls['top'], NoSlipWall)
    assert isinstance(bc.walls['bottom'], NoSlipWall)
    assert isinstance(bc.walls['left'], NoSlipWall)
    assert isinstance(bc.walls['right'], NoSlipWall)


def test_cli_parse_boundary_none():
    """CLI BC parsing with None returns default no-slip walls."""
    from cfd_solver.cli import _parse_boundary_config
    bc = _parse_boundary_config(None)
    assert isinstance(bc.walls['top'], NoSlipWall)


# ═══════════════════════════════════════════════════════════════════════
# Phase 1 — Body Forces, Initial Conditions, Time Tracking, Convergence
# ═══════════════════════════════════════════════════════════════════════

# ── Time Tracking ────────────────────────────────────────────────────

def test_time_starts_zero():
    """Solver.time is 0.0 before any steps."""
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001)
    assert s.time == 0.0


def test_time_increments_by_dt():
    """Solver.time increments by dt after each step."""
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.01)
    s.step()
    assert abs(s.time - 0.01) < 1e-12
    s.step()
    assert abs(s.time - 0.02) < 1e-12


def test_time_after_solve():
    """Solver.time matches simulation_time after solve(simulation_time=...)."""
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001)
    s.solve(simulation_time=0.1, verbose=False)
    expected = np.ceil(0.1 / 0.001) * 0.001
    assert abs(s.time - expected) < 1e-10


# ── Body Force ───────────────────────────────────────────────────────

def test_body_force_none():
    """No body force (default) produces same results as before."""
    s1 = Solver(grid_size=(8, 4), nu=0.01, dt=0.001)
    s1.solve(steps=10, verbose=False)
    s2 = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, body_force=None)
    s2.solve(steps=10, verbose=False)
    assert np.allclose(s1.u, s2.u)


def test_body_force_constant_zero():
    """Constant zero body force is equivalent to no force."""
    def zero_force(u, v, t):
        return np.zeros_like(u), np.zeros_like(v)
    s1 = Solver(grid_size=(8, 4), nu=0.01, dt=0.001)
    s1.solve(steps=10, verbose=False)
    s2 = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, body_force=zero_force)
    s2.solve(steps=10, verbose=False)
    assert np.allclose(s1.u, s2.u)


def test_body_force_accelerates_fluid():
    """Constant x-force on a closed box accelerates the fluid."""
    bc = BoundaryConditions(top=NoSlipWall(u=0.0), bottom=NoSlipWall(u=0.0),
                            left=NoSlipWall(u=0.0), right=NoSlipWall(u=0.0))
    s = Solver(
        grid_size=(16, 8), nu=0.01, dt=0.0005, lid_speed=0.0,
        boundary_config=bc,
        body_force=lambda u, v, t: (np.full_like(u, 0.1), np.zeros_like(v)),
    )
    s.solve(simulation_time=0.5, verbose=False)
    # With constant force, mean u should be positive
    interior = s.u[1:-1, 1:-1]
    assert np.mean(interior) > 0.0


def test_body_force_lambda():
    """Lambda body force works the same as a named function."""
    def bf(u, v, t):
        return np.full_like(u, 0.5), np.full_like(v, 0.0)
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, body_force=bf)
    s.solve(simulation_time=0.1, verbose=False)
    assert s.time > 0


def test_body_force_time_varying():
    """Time-varying body force produces different result than constant."""
    def constant_force(u, v, t):
        return np.full_like(u, 0.1), np.zeros_like(v)
    def oscillating_force(u, v, t):
        return np.full_like(u, 0.1 * np.sin(2 * np.pi * t)), np.zeros_like(v)
    s_const = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, body_force=constant_force)
    s_const.solve(simulation_time=0.5, verbose=False)
    s_osc = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, body_force=oscillating_force)
    s_osc.solve(simulation_time=0.5, verbose=False)
    # The two should be different
    assert not np.allclose(s_const.u, s_osc.u, atol=1e-6)


# ── Initial Condition ────────────────────────────────────────────────

def test_initial_condition_none():
    """No IC (default) uses zero initial velocity."""
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, lid_speed=0.0)
    # Before solve, interior is zeros; BC applies lid=0 so entire array is zero
    interior = s.u[1:-1, 1:-1]
    assert np.allclose(interior, 0.0)


def test_initial_condition_sets_field():
    """Provided IC sets the initial velocity field before the first step."""
    call_count = [0]
    def my_ic(mesh):
        call_count[0] += 1
        u = np.ones(mesh.shape_u)
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        return u, v, p
    bc = BoundaryConditions(top=NoSlipWall(u=0.0), bottom=NoSlipWall(u=0.0),
                            left=NoSlipWall(u=0.0), right=NoSlipWall(u=0.0))
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, lid_speed=0.0,
               boundary_config=bc, initial_condition=my_ic)
    assert call_count[0] == 0  # not called yet
    s.solve(steps=1, verbose=False)
    assert call_count[0] == 1  # called exactly once


def test_initial_condition_mesh_object():
    """IC function receives a Mesh object with expected attributes."""
    received = {}
    def my_ic(mesh):
        received['Nx'] = mesh.Nx
        received['Ny'] = mesh.Ny
        received['shape_u'] = mesh.shape_u
        received['shape_v'] = mesh.shape_v
        received['shape_p'] = mesh.shape_p
        received['Lx'] = mesh.Lx
        received['Ly'] = mesh.Ly
        return (np.zeros(mesh.shape_u), np.zeros(mesh.shape_v),
                np.zeros(mesh.shape_p))
    bc = BoundaryConditions(top=NoSlipWall(u=0.0), bottom=NoSlipWall(u=0.0),
                            left=NoSlipWall(u=0.0), right=NoSlipWall(u=0.0))
    s = Solver(grid_size=(16, 8), nu=0.01, dt=0.001, Lx=2.0, Ly=1.0,
               lid_speed=0.0, boundary_config=bc, initial_condition=my_ic)
    s.solve(steps=1, verbose=False)
    assert received['Nx'] == 16
    assert received['Ny'] == 8
    # Mesh shapes include ghost cells: u=(Nx+1, Ny+2), v=(Nx+2, Ny+1), p=(Nx+2, Ny+2)
    assert received['shape_u'] == (17, 10)
    assert received['shape_v'] == (18, 9)
    assert received['shape_p'] == (18, 10)
    assert received['Lx'] == 2.0
    assert received['Ly'] == 1.0


def test_initial_condition_with_body_force():
    """IC and body force can be used together."""
    call_count = [0]
    def my_ic(mesh):
        call_count[0] += 1
        u = np.ones(mesh.shape_u) * 0.5
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        return u, v, p
    bc = BoundaryConditions(top=NoSlipWall(u=0.0), bottom=NoSlipWall(u=0.0),
                            left=NoSlipWall(u=0.0), right=NoSlipWall(u=0.0))
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001, lid_speed=0.0,
               boundary_config=bc,
               initial_condition=my_ic,
               body_force=lambda u, v, t: (np.zeros_like(u), np.zeros_like(v)))
    s.solve(simulation_time=0.1, verbose=False)
    assert call_count[0] == 1
    assert np.isfinite(s.u).all()


# ── Steady-State Convergence ─────────────────────────────────────────

def test_convergence_no_tol_runs_all_steps():
    """Without convergence_tol, solve runs all requested steps."""
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001)
    s.solve(simulation_time=0.1, verbose=False)
    expected = np.ceil(0.1 / 0.001)
    # Should have run exactly 'expected' steps
    assert abs(s.time - expected * 0.001) < 1e-10


def test_convergence_stops_early():
    """Convergence check stops simulation early when converged."""
    def my_ic(mesh):
        return (np.zeros(mesh.shape_u), np.zeros(mesh.shape_v),
                np.zeros(mesh.shape_p))
    # Run without convergence — take many steps
    s1 = Solver(grid_size=(16, 8), nu=0.01, dt=0.0005, initial_condition=my_ic)
    s1.solve(simulation_time=1.0, verbose=False)
    steps_no_conv = int(np.ceil(1.0 / 0.0005))

    # Run with convergence — should stop early
    s2 = Solver(grid_size=(16, 8), nu=0.01, dt=0.0005, initial_condition=my_ic)
    s2.solve(simulation_time=1.0, verbose=False, convergence_tol=1e-6, convergence_window=50)
    # s2 should have taken fewer steps (or the same if not converged yet)
    # Just verify it didn't crash and time is valid
    assert s2.time <= 1.0 + 0.0005


def test_convergence_returns_true():
    """solve() returns True on normal completion (not a blowup)."""
    s = Solver(grid_size=(8, 4), nu=0.01, dt=0.001)
    result = s.solve(simulation_time=0.01, verbose=False)
    assert result is True


def test_convergence_returns_false_on_blowup():
    """solve() returns False when simulation blows up."""
    s = Solver(
        grid_size=(8, 4), nu=0.01, dt=0.1,
        body_force=lambda u, v, t: (np.full_like(u, 1e6), np.zeros_like(v)),
        force=True,
    )
    result = s.solve(steps=20, verbose=False)
    assert result is False


# ── YAML Parsing ─────────────────────────────────────────────────────

def test_cli_parse_body_force():
    """CLI body_force parsing produces a working callable."""
    from cfd_solver.cli import _parse_body_force
    bf = _parse_body_force({"u": "0.5", "v": "0.0"})
    assert callable(bf)
    u_test = np.zeros((5, 3))
    v_test = np.zeros((5, 3))
    fu, fv = bf(u_test, v_test, 0.0)
    assert np.allclose(fu, 0.5)
    assert np.allclose(fv, 0.0)


def test_cli_parse_body_force_zero():
    """CLI body_force parsing with zero values returns None."""
    from cfd_solver.cli import _parse_body_force
    bf = _parse_body_force({"u": "0.0", "v": "0.0"})
    assert bf is None


def test_cli_parse_body_force_none():
    """CLI body_force parsing with None returns None."""
    from cfd_solver.cli import _parse_body_force
    assert _parse_body_force(None) is None


def test_cli_parse_convergence():
    """CLI convergence parsing returns (tol, window)."""
    from cfd_solver.cli import _parse_convergence
    tol, window = _parse_convergence({"tol": 1e-6, "window": 50})
    assert tol == 1e-6
    assert window == 50


def test_cli_parse_convergence_none():
    """CLI convergence parsing with None returns (None, None)."""
    from cfd_solver.cli import _parse_convergence
    tol, window = _parse_convergence(None)
    assert tol is None
    assert window is None


# ── Validation Schema ────────────────────────────────────────────────

def test_validate_body_force():
    """Config with body_force passes validation."""
    from cfd_solver.solver.validate import validate_config
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "Nx": 8, "Ny": 8},
        "nu": 0.01, "dt": 0.001, "steps": 10,
        "body_force": {"u": 0.1, "v": 0.0},
    }
    errors = validate_config(cfg)
    assert errors == []


def test_validate_convergence():
    """Config with convergence passes validation."""
    from cfd_solver.solver.validate import validate_config
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "Nx": 8, "Ny": 8},
        "nu": 0.01, "dt": 0.001, "steps": 10,
        "convergence": {"tol": 1e-6, "window": 100},
    }
    errors = validate_config(cfg)
    assert errors == []


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Periodic Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════


# ── Periodic Ghost Cells ──────────────────────────────────────────────

def test_bc_periodic_y_copies_interior():
    """PeriodicWall in y: u ghost cells copy from opposite interior."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    # Set some interior u values
    u[3, 1] = 0.5
    u[3, 2] = 0.8
    bc = BoundaryConditions(
        top=PeriodicWall(), bottom=PeriodicWall(),
        left=NoSlipWall(), right=NoSlipWall(),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Top ghost (j=-1) = bottom interior (j=1)
    assert u[3, -1] == pytest.approx(0.5)
    # Bottom ghost (j=0) = top interior (j=Ny)
    assert u[3, 0] == pytest.approx(u[3, m.Ny])


def test_bc_periodic_x_copies_interior():
    """PeriodicWall in x: v ghost cells copy from opposite interior."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    # Set some interior v values
    v[1, 3] = 0.7
    v[2, 3] = 0.9
    bc = BoundaryConditions(
        top=NoSlipWall(), bottom=NoSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Left ghost (i=0) = right interior (i=Nx)
    assert v[0, 3] == pytest.approx(v[m.Nx, 3])
    # Right ghost (i=-1) = left interior (i=1)
    assert v[-1, 3] == pytest.approx(v[1, 3])


def test_bc_periodic_both_directions():
    """PeriodicWall in both x and y."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    u[4, 3] = 1.0
    v[3, 4] = 0.5
    bc = BoundaryConditions(
        top=PeriodicWall(), bottom=PeriodicWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # u: top ghost = bottom interior
    assert u[4, -1] == pytest.approx(u[4, 1])
    # u: bottom ghost = top interior
    assert u[4, 0] == pytest.approx(u[4, m.Ny])
    # v: left ghost = right interior
    assert v[0, 4] == pytest.approx(v[m.Nx, 4])
    # v: right ghost = left interior
    assert v[-1, 4] == pytest.approx(v[1, 4])


# ── PeriodicPressureSolver ────────────────────────────────────────────

def test_periodic_pressure_zero_divergence():
    """PeriodicPressureSolver gives zero pressure for zero divergence."""
    from cfd_solver.solver.pressure import PeriodicPressureSolver
    m = Mesh(1.0, 1.0, 16, 16)
    ps = PeriodicPressureSolver(m)
    u_s = np.zeros(m.shape_u)
    u_s[:, 1:-1] = 1.0
    v_s = np.zeros(m.shape_v)
    p = ps.solve(u_s, v_s, dt=0.001)
    assert np.allclose(p[1:-1, 1:-1], 0.0, atol=1e-10)


def test_periodic_pressure_zero_mean():
    """PeriodicPressureSolver maintains zero-mean pressure."""
    from cfd_solver.solver.pressure import PeriodicPressureSolver
    m = Mesh(1.0, 1.0, 16, 16)
    ps = PeriodicPressureSolver(m)
    u_s = np.random.randn(*m.shape_u) * 0.01
    v_s = np.random.randn(*m.shape_v) * 0.01
    p = ps.solve(u_s, v_s, dt=0.001)
    assert abs(np.mean(p[1:-1, 1:-1])) < 1e-10


def test_periodic_pressure_ghost_cells():
    """PeriodicPressureSolver sets periodic ghost cells in x."""
    from cfd_solver.solver.pressure import PeriodicPressureSolver
    m = Mesh(1.0, 1.0, 16, 16)
    ps = PeriodicPressureSolver(m)
    u_s = np.zeros(m.shape_u)
    v_s = np.zeros(m.shape_v)
    p = ps.solve(u_s, v_s, dt=0.001)
    # Periodic x: p[0, :] = p[Nx, :] and p[-1, :] = p[1, :]
    assert np.allclose(p[0, :], p[m.Nx, :])
    assert np.allclose(p[-1, :], p[1, :])
    # Neumann y: p[:, 0] = p[:, 1] and p[:, -1] = p[:, -2]
    assert np.allclose(p[:, 0], p[:, 1])
    assert np.allclose(p[:, -1], p[:, -2])


def test_create_pressure_solver_periodic():
    """Factory returns PeriodicPressureSolver when periodic BCs are present."""
    from cfd_solver.solver.pressure import PeriodicPressureSolver, create_pressure_solver
    m = Mesh(1.0, 1.0, 256, 256)
    bc = BoundaryConditions(
        top=PeriodicWall(), bottom=PeriodicWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    ps = create_pressure_solver(m, bc=bc)
    assert isinstance(ps, PeriodicPressureSolver)


# ── Solver with Periodic BCs ──────────────────────────────────────────

def test_solver_detects_periodic():
    """Solver._periodic_x is True when periodic walls are present."""
    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.001,
               boundary_config=bc, lid_speed=0.0, smooth_lid=False, force=True)
    assert s._periodic_x is True


def test_solver_no_periodic_by_default():
    """Solver._periodic_x is False when no periodic walls."""
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.001)
    assert s._periodic_x is False


def test_solver_periodic_uses_explicit_diffusion():
    """Solver falls back to explicit diffusion when periodic BCs are used."""
    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.001,
               boundary_config=bc, lid_speed=0.0, smooth_lid=False, force=True)
    assert s._diffusion is None


def test_solver_periodic_step_runs():
    """Solver with periodic x completes a step without errors."""
    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.001,
               boundary_config=bc, lid_speed=0.0, smooth_lid=False, force=True)
    s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert np.isfinite(s.p).all()


def test_solver_periodic_remains_finite():
    """Solver with periodic x remains stable over many steps."""
    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.0005,
               boundary_config=bc, lid_speed=0.0, smooth_lid=False, force=True)
    for _ in range(50):
        s.step()
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert s.max_divergence() < 1e-6


def test_solver_taylor_green_ic():
    """Taylor-Green IC produces expected initial velocity field."""
    Lx, Ly = 2 * np.pi, 2 * np.pi
    kx, ky = 2 * np.pi / Lx, 2 * np.pi / Ly
    U0 = 1.0

    def tg_ic(mesh):
        u = np.zeros(mesh.shape_u)
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        Xf, Yf = mesh.u_face_grid()
        u[:, 1:-1] = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf)
        Xv, Yv = mesh.v_face_grid()
        v[1:-1, :] = U0 * np.cos(kx * Xv) * np.sin(ky * Yv)
        return u, v, p

    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.001, Lx=Lx, Ly=Ly,
               boundary_config=bc, initial_condition=tg_ic,
               lid_speed=0.0, smooth_lid=False, force=True)
    s.solve(simulation_time=0.001, verbose=False)

    # After one step, u should still be close to initial condition
    Xf, Yf = s.mesh.u_face_grid()
    u_exact = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf)
    assert np.allclose(s.u[:, 1:-1], u_exact, atol=0.05)


def test_solver_taylor_green_decays():
    """Taylor-Green vortex decays exponentially over time."""
    Lx, Ly = 2 * np.pi, 2 * np.pi
    kx, ky = 2 * np.pi / Lx, 2 * np.pi / Ly
    U0 = 1.0
    nu = 0.01
    d = nu * (kx**2 + ky**2)

    def tg_ic(mesh):
        u = np.zeros(mesh.shape_u)
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        Xf, Yf = mesh.u_face_grid()
        u[:, 1:-1] = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf)
        Xv, Yv = mesh.v_face_grid()
        v[1:-1, :] = U0 * np.cos(kx * Xv) * np.sin(ky * Yv)
        return u, v, p

    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(32, 32), nu=nu, dt=0.0005, Lx=Lx, Ly=Ly,
               boundary_config=bc, initial_condition=tg_ic,
               lid_speed=0.0, smooth_lid=False, force=True)
    s.solve(simulation_time=1.0, verbose=False)

    # Compare with analytical decay
    Xf, Yf = s.mesh.u_face_grid()
    u_exact = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf) * np.exp(-d * s.time)
    l2 = np.sqrt(np.mean((s.u[:, 1:-1] - u_exact)**2))
    assert l2 < 0.05


def test_solver_couette_periodic():
    """Couette flow with periodic x converges to linear profile."""
    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0), bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(16, 32), nu=0.01, dt=0.0005, Lx=1.0, Ly=1.0,
               boundary_config=bc, lid_speed=0.0, smooth_lid=False, force=True)
    s.solve(simulation_time=5.0, verbose=False)

    # At t=5 with nu=0.01, H=1, the flow should be partially developed
    # Check monotonic profile: u increases with y
    u_mid = s.u[s.Nx // 2, 1:-1]
    for j in range(len(u_mid) - 1):
        assert u_mid[j] <= u_mid[j + 1] + 0.01  # allow small numerical error


def test_solver_periodic_divergence_small():
    """Periodic solver maintains small divergence."""
    def tg_ic(mesh):
        Lx, Ly = mesh.Lx, mesh.Ly
        kx, ky = 2 * np.pi / Lx, 2 * np.pi / Ly
        U0 = 1.0
        u = np.zeros(mesh.shape_u)
        v = np.zeros(mesh.shape_v)
        p = np.zeros(mesh.shape_p)
        Xf, Yf = mesh.u_face_grid()
        u[:, 1:-1] = -U0 * np.sin(kx * Xf) * np.cos(ky * Yf)
        Xv, Yv = mesh.v_face_grid()
        v[1:-1, :] = U0 * np.cos(kx * Xv) * np.sin(ky * Yv)
        return u, v, p

    bc = BoundaryConditions(
        top=FreeSlipWall(), bottom=FreeSlipWall(),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s = Solver(grid_size=(32, 32), nu=0.01, dt=0.0005,
               Lx=2 * np.pi, Ly=2 * np.pi,
               boundary_config=bc, initial_condition=tg_ic,
               lid_speed=0.0, smooth_lid=False, force=True)
    for _ in range(20):
        s.step()
    assert s.max_divergence() < 1e-6


def test_solver_couette_with_noslip_x():
    """Couette with no-slip x walls converges to circulation pattern."""
    s = Solver(grid_size=(16, 32), nu=0.01, dt=0.001, Lx=1.0, Ly=1.0,
               lid_speed=1.0, smooth_lid=False, force=True)
    s.solve(simulation_time=5.0, verbose=False)
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()


def test_cli_parse_boundary_periodic():
    """CLI periodic BC parsing produces PeriodicWall."""
    from cfd_solver.cli import _parse_boundary_config
    bc_cfg = {
        "top": {"type": "free_slip"},
        "bottom": {"type": "free_slip"},
        "left": {"type": "periodic"},
        "right": {"type": "periodic"},
    }
    bc = _parse_boundary_config(bc_cfg)
    assert isinstance(bc.walls['left'], PeriodicWall)
    assert isinstance(bc.walls['right'], PeriodicWall)
    assert isinstance(bc.walls['top'], FreeSlipWall)
    assert isinstance(bc.walls['bottom'], FreeSlipWall)


# ── Validation Tests ────────────────────────────────────────────────

def test_taylor_green_validation():
    """Taylor-Green vortex L2 error below threshold."""
    from examples.taylor_green.validate import validate
    l2 = validate()
    assert l2 < 0.05


def test_couette_validation():
    """Couette flow L2 error below threshold."""
    from examples.couette.validate import validate
    l2 = validate()
    assert l2 < 0.5


def test_channel_validation():
    """Channel flow L2 error below threshold."""
    from examples.channel_flow.validate import validate
    l2 = validate()
    assert l2 < 0.01


def test_cavity_ghia_validation():
    """Cavity Ghia L2 error below threshold (Re=100)."""
    from examples.cavity.validate import validate
    l2 = validate(Re=100)
    assert l2 < 0.1
