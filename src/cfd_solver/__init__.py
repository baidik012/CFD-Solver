"""
CFD Solver — Incompressible fluid flow simulation package.

This package provides tools for simulating 2D incompressible fluid flow
using the Navier-Stokes equations. It includes a staggered grid solver,
various advection and diffusion schemes, and visualization utilities.
"""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("cfd-solver")
except Exception:
    # Fallback if the package is not installed (e.g., during development)
    __version__ = "0.3.3"