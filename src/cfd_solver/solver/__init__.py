"""Solver subpackage — modular incompressible Navier-Stokes components."""

from .mesh import Mesh
from .bc import BoundaryConditions
from .solver import Solver
from . import advection
from .diffusion import CrankNicolson, FFTCrankNicolson, explicit, create_diffusion_solver
from .pressure import PressureSolver, FFTPressureSolver, create_pressure_solver
from .validate import validate_config
from . import diagnostics
from .viz import save_quiver, save_contour, save_streamlines

__all__ = [
    "Mesh",
    "BoundaryConditions",
    "Solver",
    "advection",
    "CrankNicolson",
    "FFTCrankNicolson",
    "explicit",
    "create_diffusion_solver",
    "PressureSolver",
    "FFTPressureSolver",
    "create_pressure_solver",
    "validate_config",
    "diagnostics",
    "save_quiver",
    "save_contour",
    "save_streamlines",
]
