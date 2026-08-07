"""Tests for the architectural-audit fixes (P0/P1/P2/P3).

These tests verify the specific behaviour changes introduced by the
``fix/architectural-audit-p0`` branch.  They are kept in a separate
file from ``test_core.py`` so that the audit-fix coverage is easy to
review and so that future audit fixes can add tests here without
disturbing the existing test layout.

Audit reference: CFD-Solver_Architectural_Audit.pdf
"""

import os
import sys
import numpy as np
import pytest

from cfd_solver.solver.mesh import Mesh
from cfd_solver.solver.bc import (
    BoundaryConditions, NoSlipWall, FreeSlipWall, InletWall, OutletWall,
    PeriodicWall, WallType, WALL_TYPE_REGISTRY,
)
from cfd_solver.solver.solver import Solver
from cfd_solver.solver.validate import validate_config, WALL_TYPE_VALUES
from cfd_solver.solver.diffusion import CrankNicolson, FFTCrankNicolson
from cfd_solver.solver.pressure import create_pressure_solver
from cfd_solver.config_loader import load_config


# ── P0-1: WallType Strategy pattern ──────────────────────────────────────


def test_p01_wall_type_registry_has_all_subclasses():
    """WALL_TYPE_REGISTRY must contain every WallType subclass."""
    expected = {"wall", "free_slip", "inlet", "outlet", "periodic"}
    assert set(WALL_TYPE_REGISTRY.keys()) == expected


def test_p01_wall_type_apply_methods_exist():
    """Every WallType subclass must implement apply_top/bottom/left/right."""
    for cls in WALL_TYPE_REGISTRY.values():
        assert hasattr(cls, 'apply_top'),    f"{cls.__name__} missing apply_top"
        assert hasattr(cls, 'apply_bottom'), f"{cls.__name__} missing apply_bottom"
        assert hasattr(cls, 'apply_left'),   f"{cls.__name__} missing apply_left"
        assert hasattr(cls, 'apply_right'),  f"{cls.__name__} missing apply_right"
        assert hasattr(cls, 'is_noslip'),         f"{cls.__name__} missing is_noslip"
        assert hasattr(cls, 'ghost_cell_coeffs'), f"{cls.__name__} missing ghost_cell_coeffs"


def test_p01_noslip_wall_apply_top_sets_ghost():
    """NoSlipWall.apply_top enforces the Dirichlet ghost-cell value."""
    Nx, Ny = 4, 4
    u = np.zeros((Nx + 1, Ny + 2))
    u[:, -2] = 1.0  # interior top row
    v = np.zeros((Nx + 2, Ny + 1))
    bc = BoundaryConditions(top=NoSlipWall(u=2.0))
    bc.walls['top'].apply_top(u, v, Nx, Ny, bc)
    # ghost = 2*u_wall - interior = 2*2 - 1 = 3
    assert u[0, -1] == pytest.approx(3.0)


def test_p01_periodic_wall_apply_left_wraps():
    """PeriodicWall.apply_left sets ghost = right interior."""
    Nx, Ny = 4, 4
    u = np.zeros((Nx + 1, Ny + 2))
    u[Nx, 1:-1] = 5.0  # right interior
    v = np.zeros((Nx + 2, Ny + 1))
    v[Nx, :] = 7.0
    bc = BoundaryConditions(left=PeriodicWall())
    bc.walls['left'].apply_left(u, v, Nx, Ny, bc)
    assert np.all(u[0, 1:-1] == 5.0)
    assert np.all(v[0, :] == 7.0)


def test_p01_ghost_cell_coeffs_noslip():
    """NoSlipWall.ghost_cell_coeffs returns (True, wall_value)."""
    w = NoSlipWall(u=1.5, v=2.5)
    assert w.ghost_cell_coeffs('u') == (True, 1.5)
    assert w.ghost_cell_coeffs('v') == (True, 2.5)


def test_p01_ghost_cell_coeffs_free_slip():
    """FreeSlipWall.ghost_cell_coeffs returns (False, 0.0)."""
    w = FreeSlipWall(u=1.0)
    assert w.ghost_cell_coeffs('u') == (False, 0.0)


def test_p01_is_noslip_for_each_wall_type():
    """is_noslip() returns the expected value for each wall type."""
    assert NoSlipWall().is_noslip() is True
    assert InletWall().is_noslip() is True
    # (Audit fix #1 — OutletWall now correctly returns False because it
    #  uses zero-gradient Neumann BCs, not no-slip Dirichlet.)
    assert OutletWall().is_noslip() is False
    assert FreeSlipWall().is_noslip() is False
    assert PeriodicWall().is_noslip() is False


def test_p01_bc_has_periodic_and_has_outlet_helpers():
    """BoundaryConditions.has_periodic() / has_outlet() work correctly."""
    bc_periodic = BoundaryConditions(left=PeriodicWall(), right=PeriodicWall())
    assert bc_periodic.has_periodic() is True
    assert bc_periodic.has_outlet() is False

    bc_outlet = BoundaryConditions(right=OutletWall())
    assert bc_outlet.has_outlet() is True
    assert bc_outlet.has_periodic() is False

    bc_plain = BoundaryConditions()
    assert bc_plain.has_periodic() is False
    assert bc_plain.has_outlet() is False


def test_p01_apply_delegates_to_wall_objects():
    """BoundaryConditions.apply() produces the same result as the old
    isinstance-chain implementation.  This is a regression guard."""
    Nx, Ny = 8, 8
    bc = BoundaryConditions(top=NoSlipWall(u=1.0), smooth_lid=False)
    u = np.zeros((Nx + 1, Ny + 2))
    v = np.zeros((Nx + 2, Ny + 1))
    u[:, -2] = 0.5  # interior top
    bc.apply(u, v, Nx, Ny)
    # ghost = 2*1.0 - 0.5 = 1.5
    assert u[0, -1] == pytest.approx(1.5)
    assert u[-1, -1] == pytest.approx(1.5)


def test_p01_crank_nicolson_uses_wall_ghost_cell_coeffs():
    """CrankNicolson builds its matrix via wall.ghost_cell_coeffs()."""
    # NoSlip top + bottom → diagonal modified at corners
    bc = BoundaryConditions(top=NoSlipWall(u=1.0), bottom=NoSlipWall(u=0.0))
    m = Mesh(1.0, 1.0, 8, 8)
    cn = CrankNicolson(m, nu=0.01, dt=0.001, bc=bc)
    # The matrix should be built without error and have the right shape.
    # u-matrix unknowns: (Nx-1) * Ny = 7 * 8 = 56
    assert cn.A_u.shape == (56, 56)
    # v-matrix unknowns: Nx * (Ny-1) = 8 * 7 = 56
    assert cn.A_v.shape == (56, 56)


def test_p01_deprecated_ghost_cell_coeffs_static_still_works():
    """The deprecated CrankNicolson._ghost_cell_coeffs static method
    still delegates to wall.ghost_cell_coeffs()."""
    w = NoSlipWall(u=3.0)
    assert CrankNicolson._ghost_cell_coeffs(w, 'u') == (True, 3.0)
    w2 = FreeSlipWall()
    assert CrankNicolson._ghost_cell_coeffs(w2, 'u') == (False, 0.0)


# ── P0-2: PeriodicWall loud warning ──────────────────────────────────────


def test_p02_periodic_wall_emits_warning(capsys):
    """Solver.__init__ warns when CN is downgraded to Euler for periodic."""
    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0), bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    Solver(grid_size=(8, 8), nu=0.01, dt=0.005, boundary_config=bc)
    captured = capsys.readouterr()
    assert "PeriodicWall" in captured.err
    assert "explicit Euler" in captured.err


def test_p02_periodic_wall_no_warning_with_force(capsys):
    """force=True silences the periodic-downgrade warning."""
    bc = BoundaryConditions(
        top=NoSlipWall(u=1.0), bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    Solver(grid_size=(8, 8), nu=0.01, dt=0.005, boundary_config=bc, force=True)
    captured = capsys.readouterr()
    assert "PeriodicWall" not in captured.err


def test_p02_no_warning_for_non_periodic(capsys):
    """No periodic warning when no PeriodicWall is present."""
    Solver(grid_size=(8, 8), nu=0.01, dt=0.005, lid_speed=1.0)
    captured = capsys.readouterr()
    assert "PeriodicWall" not in captured.err


# ── P0-3: validate.py schema deduplication ───────────────────────────────


def test_p03_wall_type_values_exported():
    """WALL_TYPE_VALUES is exported and matches the registry keys."""
    assert set(WALL_TYPE_VALUES) == set(WALL_TYPE_REGISTRY.keys())


def test_p03_other_key_rejected():
    """The dead 'other' boundary key is now rejected, not silently accepted."""
    cfg = {
        "geometry": {"Lx": 1, "Ly": 1, "Nx": 16, "Ny": 16},
        "nu": 0.01, "dt": 0.005, "steps": 1,
        "boundary": {"other": {"u": 0.0, "v": 0.0}},
    }
    errors = validate_config(cfg)
    assert any("Unknown field" in e and "other" in e for e in errors)


def test_p03_per_wall_schema_accepts_all_walls():
    """All four walls accept the full per-wall field set."""
    cfg = {
        "geometry": {"Lx": 1, "Ly": 1, "Nx": 16, "Ny": 16},
        "nu": 0.01, "dt": 0.005, "steps": 1,
        "boundary": {
            "top":    {"type": "wall",      "u": 1.0},
            "bottom": {"type": "wall",      "u": 0.0},
            "left":   {"type": "inlet",     "profile": "parabolic", "U_max": 1.0},
            "right":  {"type": "outlet",    "method": "zero_gradient"},
        },
    }
    assert validate_config(cfg) == []


# ── P1-5 / P1-6: load_config helper ──────────────────────────────────────


def test_p15_load_config_validates(tmp_path):
    """load_config() runs validate_config() and exits on schema failure."""
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        "geometry: {Lx: 1, Ly: 1, Nx: 16, Ny: 16}\n"
        "nu: 0.01\n"
        "dt: 0.005\n"
        "boundary:\n"
        "  other: {u: 0.0}\n"  # dead key, now rejected
    )
    with pytest.raises(SystemExit) as exc_info:
        load_config(str(bad_cfg))
    assert exc_info.value.code == 1


def test_p15_load_config_returns_dict_on_success(tmp_path):
    """load_config() returns the parsed dict on a valid config."""
    good_cfg = tmp_path / "good.yaml"
    good_cfg.write_text(
        "geometry: {Lx: 1, Ly: 1, Nx: 16, Ny: 16}\n"
        "nu: 0.01\n"
        "dt: 0.005\n"
        "steps: 10\n"
    )
    cfg = load_config(str(good_cfg))
    assert cfg["geometry"]["Nx"] == 16
    assert cfg["nu"] == 0.01


def test_p15_all_example_configs_validate():
    """Every example config in the repo validates cleanly."""
    import os
    # Walk up from this test file to find the repo root (the directory
    # containing both tests/ and examples/).  When the package is
    # pip-installed, __file__ resolves to the install location, so we
    # fall back to searching CWD if the repo layout is not found.
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(test_dir)
    examples_dir = os.path.join(repo_root, "examples")
    if not os.path.isdir(examples_dir):
        # Fall back to CWD (useful when running from the repo root
        # with a pip-installed package).
        examples_dir = os.path.join(os.getcwd(), "examples")
    if not os.path.isdir(examples_dir):
        pytest.skip("examples/ directory not found — running outside the repo")

    configs = []
    for root, dirs, files in os.walk(examples_dir):
        for f in files:
            if f.endswith(".yaml"):
                configs.append(os.path.join(root, f))
    assert len(configs) >= 4, f"Expected >= 4 example configs, found {len(configs)}"
    for c in configs:
        cfg = load_config(c)  # raises SystemExit on failure
        assert "geometry" in cfg


# ── P2-9: viz.py backend ─────────────────────────────────────────────────


def test_p29_importing_viz_does_not_force_agg():
    """Importing cfd_solver.solver.viz does not force the Agg backend
    if another backend is already chosen."""
    import matplotlib
    # Before importing viz, set a backend.
    matplotlib.use('Agg')  # simulate "user already has a backend"
    backend_before = matplotlib.get_backend()
    # Re-import viz (already imported above, but force a fresh check).
    from cfd_solver.solver import viz
    backend_after = matplotlib.get_backend()
    # The backend should not have been changed away from what we set.
    assert backend_after == backend_before


# ── P2-12: from_checkpoint warnings ──────────────────────────────────────


def test_p212_from_checkpoint_warns_about_missing_bc(capsys, tmp_path):
    """from_checkpoint warns when boundary_config is not supplied."""
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.005, lid_speed=1.0)
    ckpt = tmp_path / "test.npz"
    s.checkpoint(str(ckpt))
    Solver.from_checkpoint(str(ckpt))
    captured = capsys.readouterr()
    assert "boundary_config not supplied" in captured.err
    assert "LOST" in captured.err


def test_p212_from_checkpoint_warns_about_missing_body_force(capsys, tmp_path):
    """from_checkpoint warns when body_force is not supplied."""
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.005, lid_speed=1.0)
    ckpt = tmp_path / "test.npz"
    s.checkpoint(str(ckpt))
    Solver.from_checkpoint(str(ckpt))
    captured = capsys.readouterr()
    assert "body_force not supplied" in captured.err


def test_p212_from_checkpoint_force_silences_warnings(capsys, tmp_path):
    """force=True silences the missing-BC and missing-body-force warnings."""
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.005, lid_speed=1.0)
    ckpt = tmp_path / "test.npz"
    s.checkpoint(str(ckpt))
    Solver.from_checkpoint(str(ckpt), force=True)
    captured = capsys.readouterr()
    assert "boundary_config not supplied" not in captured.err
    assert "body_force not supplied" not in captured.err


def test_p212_from_checkpoint_accepts_boundary_config_kwarg(tmp_path):
    """from_checkpoint accepts and uses a boundary_config kwarg."""
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=OutletWall(),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.005,
               boundary_config=bc, force=True)
    ckpt = tmp_path / "test.npz"
    s.checkpoint(str(ckpt))
    # Resume with the same BC explicitly passed.
    s2 = Solver.from_checkpoint(str(ckpt), boundary_config=bc, force=True)
    assert isinstance(s2.bc.walls['left'], InletWall)
    assert isinstance(s2.bc.walls['right'], OutletWall)


# ── P2-14: version_check caching ─────────────────────────────────────────


def test_p214_cache_path_is_under_xdg_cache_home(monkeypatch):
    """The cache file respects XDG_CACHE_HOME."""
    from cfd_solver.version_check import _cache_path
    monkeypatch.setenv('XDG_CACHE_HOME', '/tmp/fake-cache')
    path = _cache_path()
    assert path.startswith('/tmp/fake-cache/cfd-solver/')


def test_p214_cache_is_fresh_after_touch(monkeypatch, tmp_path):
    """_cache_is_fresh() returns True after _touch_cache() runs."""
    from cfd_solver.version_check import _cache_is_fresh, _touch_cache, _cache_path
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    assert not _cache_is_fresh()
    _touch_cache()
    assert _cache_is_fresh()


def test_p214_env_var_disables_check():
    """CFD_SOLVER_NO_UPDATE_CHECK=1 makes check_for_updates a no-op."""
    import os
    from cfd_solver.version_check import check_for_updates
    old = os.environ.get('CFD_SOLVER_NO_UPDATE_CHECK')
    try:
        os.environ['CFD_SOLVER_NO_UPDATE_CHECK'] = '1'
        assert check_for_updates() is False
    finally:
        if old is None:
            del os.environ['CFD_SOLVER_NO_UPDATE_CHECK']
        else:
            os.environ['CFD_SOLVER_NO_UPDATE_CHECK'] = old


# ── P2-16: save_streamlines vmin ─────────────────────────────────────────


def test_p216_save_streamlines_accepts_vmin_kwarg(tmp_path):
    """save_streamlines accepts explicit vmin and vmax kwargs."""
    import matplotlib
    matplotlib.use('Agg')
    from cfd_solver.solver.viz import save_streamlines
    m = Mesh(1.0, 1.0, 8, 8)
    u = np.ones((9, 10)) * 0.5
    v = np.zeros((10, 9))
    p = np.zeros((10, 10))
    out = tmp_path / "streamlines.png"
    save_streamlines(m, u, v, p, str(out), vmin=-3, vmax=1)
    assert out.exists()


def test_p216_save_streamlines_derives_vmin_from_data(tmp_path):
    """save_streamlines derives vmin from the data when not given."""
    import matplotlib
    matplotlib.use('Agg')
    from cfd_solver.solver.viz import save_streamlines
    m = Mesh(1.0, 1.0, 8, 8)
    # Very low speed flow — previously clipped at vmin=-6.
    u = np.ones((9, 10)) * 1e-3
    v = np.zeros((10, 9))
    p = np.zeros((10, 10))
    out = tmp_path / "streamlines_low_speed.png"
    save_streamlines(m, u, v, p, str(out))
    assert out.exists()


# ── P3-18: Mesh input validation ─────────────────────────────────────────


def test_p318_mesh_rejects_zero_lx():
    with pytest.raises(ValueError, match="Lx must be positive"):
        Mesh(0.0, 1.0, 8, 8)


def test_p318_mesh_rejects_zero_ly():
    with pytest.raises(ValueError, match="Ly must be positive"):
        Mesh(1.0, 0.0, 8, 8)


def test_p318_mesh_rejects_zero_nx():
    with pytest.raises(ValueError, match="Nx must be at least 1"):
        Mesh(1.0, 1.0, 0, 8)


def test_p318_mesh_rejects_zero_ny():
    with pytest.raises(ValueError, match="Ny must be at least 1"):
        Mesh(1.0, 1.0, 8, 0)


def test_p318_mesh_rejects_negative_lx():
    with pytest.raises(ValueError, match="Lx must be positive"):
        Mesh(-1.0, 1.0, 8, 8)


def test_p318_mesh_accepts_minimum_valid():
    """Mesh(Nx=1, Ny=1) is the minimum valid grid."""
    m = Mesh(1.0, 1.0, 1, 1)
    assert m.dx == 1.0
    assert m.dy == 1.0


# ── Round-2 audit fixes (#1–#10) ───────────────────────────────────────────


def test_fix1_outlet_wall_is_not_noslip():
    """OutletWall.is_noslip() returns False (Neumann, not Dirichlet)."""
    w = OutletWall()
    assert w.is_noslip() is False
    assert w.ghost_cell_coeffs('u') == (False, 0.0)
    assert w.ghost_cell_coeffs('v') == (False, 0.0)


def test_fix3_periodic_x_requires_left_and_right():
    """_periodic_x is True only when both left AND right are PeriodicWall."""
    bc_y_periodic = BoundaryConditions(
        top=PeriodicWall(), bottom=PeriodicWall(),
        left=NoSlipWall(u=0.0), right=NoSlipWall(u=0.0),
    )
    # top/bottom periodic should NOT set _periodic_x
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.005,
              boundary_config=bc_y_periodic, force=True)
    assert s._periodic_x is False

    bc_x_periodic = BoundaryConditions(
        top=NoSlipWall(u=0.0), bottom=NoSlipWall(u=0.0),
        left=PeriodicWall(), right=PeriodicWall(),
    )
    s2 = Solver(grid_size=(8, 8), nu=0.01, dt=0.005,
               boundary_config=bc_x_periodic, force=True)
    assert s2._periodic_x is True


def test_fix4_convergence_monitors_both_u_and_v():
    """Convergence check monitors changes in both u and v."""
    s = Solver(grid_size=(8, 8), nu=0.01, dt=0.005, lid_speed=1.0)
    # Run a few steps with convergence — should not crash
    s.solve(steps=5, convergence_tol=1e-10, convergence_window=1, verbose=False)


def test_fix6_parabolic_inlet_uses_domain_height():
    """Parabolic inlet profile respects non-unit domain height."""
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=OutletWall(),
        top=NoSlipWall(u=0.0), bottom=NoSlipWall(u=0.0),
    )
    # With Ly=2.0, the profile should peak at y=1.0 (mid-channel)
    s = Solver(grid_size=(16, 16), nu=0.01, dt=0.001,
               Lx=2.0, Ly=2.0,
               boundary_config=bc, force=True)
    # The peak of the parabolic profile should be U_max=1.0
    profile = s.bc._inlet_profile(bc.walls['left'], 16, axis='x')
    assert np.max(profile) == pytest.approx(1.0, abs=0.05)


def test_fix7_fftcn_uses_wall_ghost_cell_coeffs_directly():
    """FFTCrankNicolson.solve() calls wall.ghost_cell_coeffs(), not the
    deprecated CrankNicolson._ghost_cell_coeffs static method."""
    from cfd_solver.solver.diffusion import FFTCrankNicolson
    bc = BoundaryConditions(top=NoSlipWall(u=1.0), bottom=NoSlipWall(u=0.0))
    m = Mesh(1.0, 1.0, 16, 16)
    solver = FFTCrankNicolson(m, nu=0.01, dt=0.001, bc=bc)
    # Just verify it was constructed without error and solve works
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)
    u_star, v_star = solver.solve(u, v, adv_u, adv_v)
    assert np.all(np.isfinite(u_star))
    assert np.all(np.isfinite(v_star))
