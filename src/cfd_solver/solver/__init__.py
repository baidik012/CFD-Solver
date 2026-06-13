"""Solver subpackage — modular incompressible Navier-Stokes components."""

from .mesh import Mesh
from .bc import BoundaryConditions
from .solver import Solver
from . import advection
from .diffusion import CrankNicolson, explicit
from .pressure import PressureSolver
from .validate import validate_config
from . import diagnostics
from .viz import save_quiver, save_contour, save_streamlines

__all__ = [
    "Mesh",
    "BoundaryConditions",
    "Solver",
    "advection",
    "CrankNicolson",
    "explicit",
    "PressureSolver",
    "validate_config",
    "diagnostics",
    "save_quiver",
    "save_contour",
    "save_streamlines",
]
