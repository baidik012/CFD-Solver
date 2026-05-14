"""Structured Cartesian grid."""

import numpy as np


class Grid:
    """Staggered grid for incompressible flow.

    Uses Arakawa C-grid:
    - u-velocity at cell faces (i+1/2, j)
    - v-velocity at cell faces (i, j+1/2)
    - pressure at cell centers (i, j)

    Grid dimensions:
        u: (Nx+1) × Ny
        v: Nx × (Ny+1)
        p: Nx × Ny
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Cell centers (for pressure)
        self.xc = np.linspace(self.dx/2, Lx - self.dx/2, Nx)
        self.yc = np.linspace(self.dy/2, Ly - self.dy/2, Ny)
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc)

        # Cell faces (for u-velocity)
        self.xf = np.linspace(0, Lx, Nx + 1)
        self.yf = np.linspace(self.dy/2, Ly - self.dy/2, Ny)
        self.Xf, self.Yf = np.meshgrid(self.xf, self.yf)

    @property
    def shape_u(self):
        """u-velocity shape (Nx+1, Ny)."""
        return (self.Nx + 1, self.Ny)

    @property
    def shape_v(self):
        """v-velocity shape (Nx, Ny+1)."""
        return (self.Nx, self.Ny + 1)

    @property
    def shape_p(self):
        """Pressure shape (Nx, Ny)."""
        return (self.Nx, self.Ny)
    @property
    def X(self):
        """Cell-center X coordinates transposed to match array shapes (Nx x Ny)."""
        return self.Xc.T

    @property
    def Y(self):
        """Cell-center Y coordinates transposed to match array shapes (Nx x Ny)."""
        return self.Yc.T

    @property
    def Xf_T(self):
        """Face X coordinates (transposed)."""
        return self.Xf.T

    @property
    def Yf_T(self):
        """Face Y coordinates (transposed)."""
        return self.Yf.T
