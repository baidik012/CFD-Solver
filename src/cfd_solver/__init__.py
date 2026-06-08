"""CFD Solver — incompressible fluid flow simulation."""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("cfd-solver")
except Exception:
    __version__ = "0.0.0+unknown"