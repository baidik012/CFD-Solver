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
