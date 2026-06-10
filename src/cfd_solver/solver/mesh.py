"""Staggered Cartesian mesh for incompressible flow.

This module implements the Arakawa C-grid, a staggered grid layout where different
physical variables are stored at different locations within a computational cell.
This arrangement is particularly effective for incompressible flows because it
couples the velocity and pressure fields, preventing the "checkerboard"
instability that can occur on collocated grids.

Arakawa C-grid layout:
----------------------
- Pressure (p):   Stored at cell centers.
- u-velocity:     Stored at the centers of the vertical faces (x-faces).
- v-velocity:     Stored at the centers of the horizontal faces (y-faces).

Dimensions:
-----------
For a grid of Nx x Ny cells:
- p:   (Nx, Ny)
- u:   (Nx+1, Ny)
- v:   (Nx, Ny+1)

Ghost cells are added by the solver to handle boundary conditions, typically
extending these dimensions by 2 in relevant directions.
"""

import numpy as np


class Mesh:
    """Staggered grid with precomputed coordinates and spacing.

    The Mesh class provides the geometric framework for the simulation,
    defining cell sizes and the locations of velocity components and
    pressure values according to the Arakawa C-grid specification.

    Parameters
    ----------
    Lx : float
        Domain length in the x-direction.
    Ly : float
        Domain length in the y-direction.
    Nx : int
        Number of computational cells in the x-direction.
    Ny : int
        Number of computational cells in the y-direction.

    Attributes
    ----------
    dx : float
        Grid spacing in the x-direction (Lx / Nx).
    dy : float
        Grid spacing in the y-direction (Ly / Ny).
    xc : ndarray
        x-coordinates of cell centers (for pressure), length Nx.
    yc : ndarray
        y-coordinates of cell centers (for pressure), length Ny.
    xf : ndarray
        x-coordinates of u-velocity faces, length Nx+1.
    yv : ndarray
        y-coordinates of v-velocity faces, length Ny+1.
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
        """Shape of u-velocity array including ghost layers.

        Returns
        -------
        tuple
            (Nx+1, Ny+2). The additional +2 in y represents ghost cells
            used to enforce tangential boundary conditions.
        """
        return (self.Nx + 1, self.Ny + 2)

    @property
    def shape_v(self):
        """Shape of v-velocity array including ghost layers.

        Returns
        -------
        tuple
            (Nx+2, Ny+1). The additional +2 in x represents ghost cells
            used to enforce tangential boundary conditions.
        """
        return (self.Nx + 2, self.Ny + 1)

    @property
    def shape_p(self):
        """Shape of pressure array including ghost layers.

        Returns
        -------
        tuple
            (Nx+2, Ny+2). Ghost cells on all sides are used to enforce
            Neumann (zero-gradient) boundary conditions.
        """
        return (self.Nx + 2, self.Ny + 2)

    def cell_center_grid(self):
        """Return (X, Y) meshgrid at cell centers.

        Useful for plotting pressure or other cell-centered quantities.

        Returns
        -------
        X : ndarray, shape (Nx, Ny)
            X-coordinates at cell centers.
        Y : ndarray, shape (Nx, Ny)
            Y-coordinates at cell centers.
        """
        return np.meshgrid(self.xc, self.yc, indexing="ij")

    def u_face_grid(self):
        """Return (X, Y) meshgrid at u-velocity faces.

        Returns
        -------
        X : ndarray, shape (Nx+1, Ny)
            X-coordinates at vertical faces.
        Y : ndarray, shape (Nx+1, Ny)
            Y-coordinates at vertical faces.
        """
        Y = np.tile(self.yc, (self.Nx + 1, 1))
        X = np.repeat(self.xf[:, np.newaxis], self.Ny, axis=1)
        return X, Y

    def v_face_grid(self):
        """Return (X, Y) meshgrid at v-velocity faces.

        Returns
        -------
        X : ndarray, shape (Nx, Ny+1)
            X-coordinates at horizontal faces.
        Y : ndarray, shape (Nx, Ny+1)
            Y-coordinates at horizontal faces.
        """
        X = np.tile(self.xc, (self.Ny + 1, 1)).T
        Y = np.repeat(self.yv[np.newaxis, :], self.Nx, axis=0)
        return X, Y
