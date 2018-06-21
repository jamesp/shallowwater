# -*- coding: utf-8 -*-
"""The Arakawa-C Grid"""

import numpy as np

class Arakawa1D(object):
    def __init__(self, nx, Lx):
        super(Arakawa1D, self).__init__()
        self.nx = nx
        self.Lx = Lx

        # Arakawa-C grid
        # +-------+    * (nx)   phi points at grid centres
        # u  phi  u    * (nx+1) u points on vertical edges  (u[0] and u[nx] are boundary values)
        # +-------+
        self._u = np.zeros((nx+3), dtype=np.float)
        self._phi = np.zeros((nx+2), dtype=np.float)

        self.dx = dx = float(Lx) / nx

        # positions of the nodes
        self.ux = (-Lx/2 + np.arange(nx+1)*dx)
        self.phix = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)

        self.shape = self.phi.shape
        self._shape = self._phi.shape
        self.true_slice = [slice(1,-1)]*len(self.shape)  # slice of state arrays w/out BCs

    # define u, v and h properties to return state without the boundaries
    @property
    def u(self):
        return self._u[1:-1]

    @property
    def phi(self):
        return self._phi[1:-1]

    @property
    def state(self):
        return np.array([self.u, self.phi])

    @state.setter
    def state(self, value):
        u, phi = value
        self.u[:] = u
        self.phi[:] = phi

    # Define finite-difference methods on the grid
    def diffx(self, psi):
        """Calculate ∂/∂x[psi] over a single grid square.

        i.e. d/dx(psi)[i,j] = (psi[i+1/2, j] - psi[i-1/2, j]) / dx

        The derivative is returned at x points at the midpoint between
        x points of the input array."""
        return (psi[1:] - psi[:-1]) / self.dx

    def diff2x(self, psi):
        """Calculate ∂2/∂x2[psi] over a single grid square.

        i.e. d2/dx2(psi)[i,j] = (psi[i+1, j] - psi[i, j] + psi[i-1, j]) / dx^2

        The derivative is returned at the same x points as the
        x points of the input array, with dimension (nx-2)."""
        return (psi[:-2] - 2*psi[1:-1] + psi[2:]) / self.dx**2

    del2 = diff2x

    def x_average(self, psi):
        """Average adjacent values in the x dimension.
        If psi has shape (nx), returns an array of shape (nx-1)."""
        return 0.5*(psi[:-1] + psi[1:])

    def advect(self, field):
        """Calculates the conservation of the advected tracer by the fluid flow.

        ∂[q]/∂t + ∇ . (uq) = 0

        Returns the divergence term i.e. ∇.(uq)
        """
        q_at_u = self.x_average(field)  # (nx+1)

        return self.diffx(q_at_u * self.u)  # (nx)

    def _apply_boundary_conditions(self):
        # left and right-hand boundary values the same for u
        # u[0] = u[nx]
        # copy u[dx] to u[nx+dx]
        # and u[nx-dx] to u[-dx]
        # to simulate periodic continuity
        self._u[0] = self._u[-3]
        self._u[1] = self._u[-2]
        self._u[-1] = self._u[2]

        # phi points are not on boundary
        # so just simulate periodic continuity
        self._phi[0] = self._phi[-2]
        self._phi[-1] = self._phi[1]

    def apply_boundary_conditions_to(self, field):
        # periodic boundary in the x-direction
        field[0] = field[-2]
        field[-1] = field[1]


class ArakawaCGrid(object):
    def __init__(self, nx, ny, Lx, Ly):
        super(ArakawaCGrid, self).__init__()
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.true_slice = [slice(1,-1)]*2  # slice of state arrays w/out BCs

        # Arakawa-C grid
        # +-- v --+
        # |       |    * (nx, ny)   phi points at grid centres
        # u  phi  u    * (nx+1, ny) u points on vertical edges  (u[0] and u[nx] are boundary values)
        # |       |    * (nx, ny+1) v points on horizontal edges
        # +-- v --+
        self._u = np.zeros((nx+3, ny+2), dtype=np.float)
        self._v = np.zeros((nx+2, ny+3), dtype=np.float)
        self._phi = np.zeros((nx+2, ny+2), dtype=np.float)

        self.dx = dx = float(Lx) / nx
        self.dy = dy = float(Ly) / ny

        # positions of the nodes
        self.ux = (-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]
        self.vx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]

        self.vy = (-Ly/2 + np.arange(ny+1)*dy)[np.newaxis, :]
        self.uy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

        self.phix = self.vx
        self.phiy = self.uy

        self.shape = self.phi.shape
        self._shape = self._phi.shape

    # define u, v and h properties to return state without the boundaries
    @property
    def u(self):
        return self._u[1:-1, 1:-1]

    @property
    def v(self):
        return self._v[1:-1, 1:-1]

    @property
    def phi(self):
        return self._phi[1:-1, 1:-1]

    @property
    def state(self):
        return np.array([self.u, self.v, self.phi])

    @state.setter
    def state(self, value):
        u, v, phi = value
        self.u[:] = u
        self.v[:] = v
        self.phi[:] = phi

    # Define finite-difference methods on the grid
    def diffx(self, psi):
        """Calculate ∂/∂x[psi] over a single grid square.

        i.e. d/dx(psi)[i,j] = (psi[i+1/2, j] - psi[i-1/2, j]) / dx

        The derivative is returned at x points at the midpoint between
        x points of the input array."""
        return (psi[1:,:] - psi[:-1,:]) / self.dx

    def diffy(self, psi):
        """Calculate ∂/∂y[psi] over a single grid square.

        i.e. d/dy(psi)[i,j] = (psi[i, j+1/2] - psi[i, j-1/2]) / dy

        The derivative is returned at y points at the midpoint between
        y points of the input array."""
        return (psi[:, 1:] - psi[:,:-1]) / self.dy

    def del2(self, psi):
        """Returns the Laplacian of psi."""
        return self.diff2x(psi)[:, 1:-1] + self.diff2y(psi)[1:-1, :]

    def diff2x(self, psi):
        """Calculate ∂2/∂x2[psi] over a single grid square.

        i.e. d2/dx2(psi)[i,j] = (psi[i+1, j] - psi[i, j] + psi[i-1, j]) / dx^2

        The derivative is returned at the same x points as the
        x points of the input array, with dimension (nx-2, ny)."""
        return (psi[:-2, :] - 2*psi[1:-1, :] + psi[2:, :]) / self.dx**2

    def diff2y(self, psi):
        """Calculate ∂2/∂y2[psi] over a single grid square.

        i.e. d2/dy2(psi)[i,j] = (psi[i, j+1] - psi[i, j] + psi[i, j-1]) / dy^2

        The derivative is returned at the same y points as the
        y points of the input array, with dimension (nx, ny-2)."""
        return (psi[:, :-2] - 2*psi[:, 1:-1] + psi[:, 2:]) / self.dy**2

    def centre_average(self, psi):
        """Returns the four-point average at the centres between grid points.
        If psi has shape (nx, ny), returns an array of shape (nx-1, ny-1)."""
        return 0.25*(psi[:-1,:-1] + psi[:-1,1:] + psi[1:, :-1] + psi[1:,1:])

    def y_average(self, psi):
        """Average adjacent values in the y dimension.
        If psi has shape (nx, ny), returns an array of shape (nx, ny-1)."""
        return 0.5*(psi[:,:-1] + psi[:,1:])

    def x_average(self, psi):
        """Average adjacent values in the x dimension.
        If psi has shape (nx, ny), returns an array of shape (nx-1, ny)."""
        return 0.5*(psi[:-1,:] + psi[1:,:])

    def divergence(self):
        """Returns the horizontal divergence at h points."""
        return self.diffx(self.u) + self.diffy(self.v)

    def vorticity(self):
        """Returns the vorticity at grid corners."""
        return self.diffy(self.u)[1:-1, :] - self.diffx(self.v)[:, 1:-1]

    def uvath(self):
        """Calculate the value of u at h points (cell centres)."""
        ubar = self.x_average(self.u)  # (nx, ny)
        vbar = self.y_average(self.v)  # (nx, ny)
        return ubar, vbar

    def uvatuv(self):
        """Calculate the value of u at v and v at u."""
        ubar = self.centre_average(self._u)[1:-1, :]  # (nx, ny+1)
        vbar = self.centre_average(self._v)[:, 1:-1]  # (nx+1, ny)
        return ubar, vbar

    def _fix_boundary_corners(self, field):
        # fix corners to be average of neighbours
        field[0, 0] =  0.5*(field[1, 0] + field[0, 1])
        field[-1, 0] = 0.5*(field[-2, 0] + field[-1, 1])
        field[0, -1] = 0.5*(field[1, -1] + field[0, -2])
        field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])

    def advect(self, field):
        """Calculates the conservation of the advected tracer by the fluid flow.

        ∂[q]/∂t + ∇ . (uq) = 0

        Returns the divergence term i.e. ∇.(uq)
        """
        q_at_u = self.x_average(field)[:, 1:-1]  # (nx+1, ny)
        q_at_v = self.y_average(field)[1:-1, :]  # (nx, ny+1)

        return self.diffx(q_at_u * self.u) + self.diffy(q_at_v * self.v)  # (nx, ny)


    # def apply_boundary_conditions(self):
    #     """Set the boundary values of the u v and phi fields.
    #     This should be implemented by a subclass."""
    #     raise NotImplemented

    # def apply_boundary_conditions_to(self, field):
    #     """Set the boundary values of a given field.
    #     This should be implemented by a subclass."""
    #     raise NotImplemented



class PeriodicBoundaries:
    """Peridoic domain in the x-direction.
    This is a mixin class for the ArakawaCGrid to produce a grid with
    periodic boundaries in the x-direction.
    """
    def apply_boundary_conditions(self):
        # left and right-hand boundary values the same for u
        # u[0] = u[nx]
        # copy u[dx] to u[nx+dx]
        # and u[nx-dx] to u[-dx]
        # to simulate periodic continuity
        self._u[0, :] = self._u[-3, :]
        self._u[1, :] = self._u[-2, :]
        self._u[-1, :] = self._u[2, :]

        # other fields are not on boundary
        # so just simulate periodic continuity
        self._v[0, :] = self._v[-2, :]
        self._v[-1, :] = self._v[1, :]
        self._phi[0, :] = self._phi[-2, :]
        self._phi[-1, :] = self._phi[1, :]

        # top and bottom boundaries: zero derivative
        fields = self._u, self._v, self._phi
        for field in fields:
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
            self._fix_boundary_corners(field)

    def apply_boundary_conditions_to(self, field):
        # periodic boundary in the x-direction
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

        # top and bottom boundaries: zero derivative
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        self._fix_boundary_corners(field)


class WallBoundaries:
    """Add walls to the domain at x=0 and x=Lx.
    This is a mixin class for the ArakawaCGrid to produce a grid with
    walled boundaries in the x-direction.
    """
    def apply_boundary_conditions(self):
        # No flow through the boundary at x=0
        self._u[0, :] = 0
        self._u[1, :] = 0
        self._u[-1, :] = 0
        self._u[-2, :] = 0

        # free-slip of other variables: zero-derivative
        self._v[0, :] = self._v[1, :]
        self._v[-1, :] = self._v[-2, :]
        self._phi[0, :] = self._phi[1, :]
        self._phi[-1, :] = self._phi[-2, :]

        fields = self._u, self._v, self._phi
        # top and bottom boundaries: zero deriv
        for field in fields:
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
            self._fix_boundary_corners(field)

    def apply_boundary_conditions_to(self, field):
        # free slip on left and right boundares: zero derivative
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]

        # top and bottom boundaries: zero deriv and damping
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        self._fix_boundary_corners(field)
