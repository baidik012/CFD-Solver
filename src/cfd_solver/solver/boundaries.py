"""Boundary conditions."""

from dataclasses import dataclass


@dataclass
class BoundaryConditions:
    """Boundary velocities for the lid-driven cavity."""

    top_u: float = 1.0   # lid speed (m/s)
    top_v: float = 0.0
    bottom_u: float = 0.0
    bottom_v: float = 0.0
    left_u: float = 0.0
    left_v: float = 0.0
    right_u: float = 0.0
    right_v: float = 0.0

    def apply(self, u, v):
        """Set velocity at domain boundaries on staggered grid.
        
        u shape: (Nx+1, Ny) at x-faces  — u lives on vertical faces
        v shape: (Nx, Ny+1) at y-faces  — v lives on horizontal faces

        u is defined on left/right walls (i=0, i=Nx) and on top/bottom
        walls (j=0, j=Ny-1).  v is defined on top/bottom walls (j=0, j=Ny)
        but NOT on left/right walls (those are at cell-center x positions).
        """
        # u on all walls
        u[0, :] = self.left_u
        u[-1, :] = self.right_u
        u[:, 0] = self.bottom_u
        u[:, -1] = self.top_u          # set last so lid corners = top_u

        # v on top and bottom walls only
        v[:, 0] = self.bottom_v
        v[:, -1] = self.top_v
