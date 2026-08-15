"""Regression tests for the Couette validation case."""

import numpy as np

from examples.couette.run import (
    couette_analytical_steady,
    couette_analytical_transient,
)


def test_couette_steady_solution_uses_domain_height():
    y = np.array([0.25, 0.5, 0.75])
    np.testing.assert_allclose(
        couette_analytical_steady(y, U=2.0, H=2.0),
        [0.25, 0.5, 0.75],
    )


def test_couette_transient_solution_uses_domain_height():
    y = np.array([0.25, 0.5, 0.75])
    a = couette_analytical_transient(y, t=0.5, nu=0.01, H=1.0)
    b = couette_analytical_transient(y * 2.0, t=0.5, nu=0.01, H=2.0)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_couette_transient_starts_from_rest_away_from_wall_gibbs_region():
    # The finite 50-term Fourier truncation has a small Gibbs/truncation
    # residual immediately at t=0 near the discontinuous wall startup.
    y = np.array([0.25, 0.5, 0.75])
    u0 = couette_analytical_transient(y, t=0.0, nu=0.01, H=1.0)
    assert np.max(np.abs(u0)) < 1.0e-2
