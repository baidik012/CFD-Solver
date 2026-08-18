"""Staggered Cartesian mesh for incompressible flow.

Arakawa C-grid layout:
- Pressure (p): cell centers
- u-velocity: vertical face centers (x-faces)
- v-velocity: horizontal face centers (y-faces)

For Nx x Ny cells:
- p:   (Nx, Ny)
- u:   (Nx+1, Ny)
- v:   (Nx, Ny+1)

Ghost cells extend these by 2 in relevant directions for boundary conditions.
"""

import numpy as np


class Mesh:
    """Staggered grid with precomputed coordinates and spacing."""

    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int):
        if Lx <= 0:
            raise ValueError(f"Lx must be positive, got {Lx}")
        if Ly <= 0:
            raise ValueError(f"Ly must be positive, got {Ly}")
        if Nx < 1:
            raise ValueError(f"Nx must be at least 1, got {Nx}")
        if Ny < 1:
            raise ValueError(f"Ny must be at least 1, got {Ny}")

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Cell-center coordinates (for pressure)
        self.xc = np.linspace(self.dx / 2, Lx - self.dx / 2, Nx)
        self.yc = np.linspace(self.dy / 2, Ly - self.dy / 2, Ny)

        # u-face x-coordinates (vertical faces)
        self.xf = np.linspace(0, Lx, Nx + 1)

        # v-face y-coordinates (horizontal faces)
        self.yv = np.linspace(0, Ly, Ny + 1)

    @property
    def shape_u(self):
        """Shape of u-velocity array including ghost layers: (Nx+1, Ny+2)."""
        return (self.Nx + 1, self.Ny + 2)

    @property
    def shape_v(self):
        """Shape of v-velocity array including ghost layers: (Nx+2, Ny+1)."""
        return (self.Nx + 2, self.Ny + 1)

    @property
    def shape_p(self):
        """Shape of pressure array including ghost layers: (Nx+2, Ny+2)."""
        return (self.Nx + 2, self.Ny + 2)

    def cell_center_grid(self):
        """Return (X, Y) meshgrid at cell centers (for pressure)."""
        return np.meshgrid(self.xc, self.yc, indexing="ij")

    def u_face_grid(self):
        """Return (X, Y) meshgrid at u-velocity faces."""
        Y = np.tile(self.yc, (self.Nx + 1, 1))
        X = np.repeat(self.xf[:, np.newaxis], self.Ny, axis=1)
        return X, Y

    def v_face_grid(self):
        """Return (X, Y) meshgrid at v-velocity faces."""
        X = np.tile(self.xc, (self.Ny + 1, 1)).T
        Y = np.repeat(self.yv[np.newaxis, :], self.Nx, axis=0)
        return X, Y