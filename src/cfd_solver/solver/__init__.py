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

# Apply the isolated v0.3.2 P0 numerical-correctness layer.
from .p0_fixes import (  # noqa: E402
    P0Solver,
    GeneralPeriodicPressureSolver,
    create_pressure_solver_p0,
)

# Restore compatibility accidentally regressed by the first P0 implementation.
from .p0_compat import (  # noqa: E402
    CorrectedP0Solver,
    CompatiblePeriodicPressureSolver,
    create_pressure_solver_compat,
)

Solver = CorrectedP0Solver
create_pressure_solver = create_pressure_solver_compat

__all__ = [
    "Mesh",
    "BoundaryConditions",
    "Solver",
    "P0Solver",
    "CorrectedP0Solver",
    "advection",
    "CrankNicolson",
    "FFTCrankNicolson",
    "explicit",
    "create_diffusion_solver",
    "PressureSolver",
    "FFTPressureSolver",
    "GeneralPeriodicPressureSolver",
    "CompatiblePeriodicPressureSolver",
    "create_pressure_solver",
    "validate_config",
    "diagnostics",
    "save_quiver",
    "save_contour",
    "save_streamlines",
]
