"""Structured Cartesian grid."""

import numpy as np


class Grid:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def shape(self):
        """Interior shape (excluding boundaries)."""
        return (self.Nx, self.Ny)