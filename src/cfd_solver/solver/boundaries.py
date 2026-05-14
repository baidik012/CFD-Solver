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
        """Set velocity at domain boundaries."""
        # Top (y = Ly)
        u[-1, :] = self.top_u
        v[-1, :] = self.top_v

        # Bottom (y = 0)
        u[0, :] = self.bottom_u
        v[0, :] = self.bottom_v

        # Left (x = 0)
        u[:, 0] = self.left_u
        v[:, 0] = self.left_v

        # Right (x = Lx)
        u[:, -1] = self.right_u
        v[:, -1] = self.right_v