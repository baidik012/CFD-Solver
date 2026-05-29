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
        
        u shape: (Nx+1, Ny) at x-faces
        v shape: (Nx, Ny+1) at y-faces
        """
        # Left/right are x-boundaries for u.
        u[0, :] = self.left_u
        u[-1, :] = self.right_u

        # Top/bottom are y-boundaries for u. Applied last so lid corners
        # follow the moving lid convention used by the cavity examples.
        u[:, 0] = self.bottom_u
        u[:, -1] = self.top_u

        # Top/bottom are y-boundaries for v.
        v[:, 0] = self.bottom_v
        v[:, -1] = self.top_v

        # Left/right are x-boundaries for v.
        v[0, :] = self.left_v
        v[-1, :] = self.right_v
