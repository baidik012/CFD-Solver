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
        # Top (y = Ly): v at y-faces, so v[-1, :] is at top boundary
        u[:, -1] = self.top_u   # u at top (y-index -1)
        v[-1, :] = self.top_v   # v at top (y-index -1)

        # Bottom (y = 0): v at y-faces, so v[0, :] is at bottom boundary
        u[:, 0] = self.bottom_u   # u at bottom (y-index 0)
        v[0, :] = self.bottom_v    # v at bottom (y-index 0)

        # Left (x = 0): u at x-faces, so u[:, 0] is at left boundary
        u[0, :] = self.left_u    # u at left (x-index 0)
        v[:, 0] = self.left_v    # v at left (x-index 0)

        # Right (x = Lx): u at x-faces, so u[-1, :] is at right boundary
        u[-1, :] = self.right_u  # u at right (x-index -1)
        v[:, -1] = self.right_v  # v at right (x-index -1)