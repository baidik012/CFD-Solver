"""Staggered Cartesian mesh for incompressible flow.

Uses the Arakawa C-grid layout:
  u-velocity: (Nx+1, Ny) — vertical faces at x = i*dx
  v-velocity: (Nx, Ny+1) — horizontal faces at y = j*dy
  pressure:   (Nx, Ny)   — cell centers
"""

import numpy as np


class Mesh:
    """Staggered grid with precomputed coordinates and spacing.

    Parameters
    ----------
    Lx, Ly : float
        Domain size in x and y.
    Nx, Ny : int
        Number of cells in x and y.
    """

    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int):
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
        """Shape of u-velocity array: (Nx+1, Ny)."""
        return (self.Nx + 1, self.Ny)

    @property
    def shape_v(self):
        """Shape of v-velocity array: (Nx, Ny+1)."""
        return (self.Nx, self.Ny + 1)

    @property
    def shape_p(self):
        """Shape of pressure array: (Nx, Ny)."""
        return (self.Nx, self.Ny)

    def cell_center_grid(self):
        """Return (X, Y) meshgrid at cell centers, shape (Nx, Ny)."""
        return np.meshgrid(self.xc, self.yc, indexing="ij")

    def u_face_grid(self):
        """Return (X, Y) meshgrid at u-faces, shape (Nx+1, Ny)."""
        Y = np.tile(self.yc, (self.Nx + 1, 1))
        X = np.repeat(self.xf[:, np.newaxis], self.Ny, axis=1)
        return X, Y

    def v_face_grid(self):
        """Return (X, Y) meshgrid at v-faces, shape (Nx, Ny+1)."""
        X = np.tile(self.xc, (self.Ny + 1, 1)).T
        Y = np.repeat(self.yv[np.newaxis, :], self.Nx, axis=0)
        return X, Y
