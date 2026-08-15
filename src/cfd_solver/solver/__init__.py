"""Solver subpackage — modular incompressible Navier-Stokes components."""

from .mesh import Mesh
from .bc import BoundaryConditions
from .solver import Solver as _BaseSolver
from . import advection
from .diffusion import CrankNicolson, FFTCrankNicolson, explicit, create_diffusion_solver
from .pressure import PressureSolver, FFTPressureSolver, create_pressure_solver as _base_pressure_factory
from .validate import validate_config
from . import diagnostics
from .viz import save_quiver, save_contour, save_streamlines

# Apply the isolated v0.3.2 P0 numerical-correctness layer.  It patches the
# submodule symbol too, so direct and package-level Solver imports receive the
# same fixed implementation.
from .p0_fixes import (  # noqa: E402
    P0Solver,
    GeneralPeriodicPressureSolver,
    create_pressure_solver_p0,
)
Solver = P0Solver
create_pressure_solver = create_pressure_solver_p0

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
    "GeneralPeriodicPressureSolver",
    "create_pressure_solver",
    "validate_config",
    "diagnostics",
    "save_quiver",
    "save_contour",
    "save_streamlines",
]
