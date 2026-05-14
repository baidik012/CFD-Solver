import sys
import os
import numpy as np

# Ensure local src is importable
sys.path.insert(0, os.path.abspath('src'))

from cfd_solver.solver.grid import Grid
from cfd_solver.solver.staggered_solver import StaggeredSolver


def test_grid_shapes():
    g = Grid(1.0, 1.0, 8, 6)
    assert g.shape_u == (9, 6)
    assert g.shape_v == (8, 7)
    assert g.shape_p == (8, 6)
    X = g.X
    Y = g.Y
    assert X.shape == (8, 6)
    assert Y.shape == (8, 6)


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
