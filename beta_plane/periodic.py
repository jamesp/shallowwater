#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shallow Water Model

- Two dimensional shallow water in a rotating frame
- Staggered Arakawa-C lat:lon grid
- periodic in the x-dimension
- fixed boundary conditions in the y-dimension

∂/∂t[u] - fv = - g ∂/∂x[φ]
∂/∂t[v] + fu = - g ∂/∂y[φ]
∂/∂t[φ] + H(∂/∂x[u] + ∂/∂y[v]) = 0

f = f0 + βy
φ = gh
"""

import numpy as np

def adamsbashforthgen(rhs_fn, dt):
    dx, pdx, ppdx = 0, 0, 0
    dt1, dt2, dt3 = 0, 0, 0

    # first step Euler
    dt1 = dt
    dx = rhs_fn()
    val = dt1*dx
    pdx = dx
    yield val

    # AB2 at step 2
    dt1 = 1.5*dt
    dt2 = -0.5*dt
    dx = rhs_fn()
    val = dt1*dx + dt2*pdx
    ppdx, pdx = pdx, dx
    yield val

    while True:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt
        dx = rhs_fn()
        val = dt1*dx + dt2*pdx + dt3*ppdx
        ppdx, pdx = pdx, dx
        yield val

class PeriodicShallowWater(object):
    def __init__(self, nx, ny, Lx=1.0e7, Ly=1.0e7, f0=0, beta=2.0e-11, nu=1.0e-5, r=1.0e-5, dt=1000.0):
        super(PeriodicShallowWater, self).__init__()
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.f0 = f0
        self.beta = beta


        # Arakawa-C grid
        # +-- v --+
        # |       |    * (nx, ny)   h points at grid centres
        # u   h   u    * (nx+1, ny) u points on vertical edges  (u[0] and u[nx] are boundary values)
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

        # dissipation and friction
        self.nu = nu
        self.r = r
        self.sponge_ny = ny//7
        self.sponge = np.exp(-np.linspace(0, 5, self.sponge_ny))

        # timestepping
        self.dt = dt
        self.tc = 0  # number of timesteps taken
        self.t = 0.0

        self._stepper = adamsbashforthgen(self.rhs, self.dt)

        self._forcings = []

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


    def add_forcing(self, fn):
        """Add a forcing term to the model.  Typically used as a decorator:

            sw = PeriodicShallowWater(nx, ny)

            @sw.add_forcing
            def dissipate(swmodel):
                dstate = np.zeros_like(swmodel.state)
                dstate[:] = -swmodel.state*0.001
                return dstate

        Forcing functions should take a single argument for the model object itself,
        and return a state delta the same shape as state.
        """
        self._forcings.append(fn)
        return fn

    def damping(self, var):
        # sponges are active at the top and bottom of the domain by applying Rayleigh friction
        # with exponential decay towards the centre of the domain
        var_sponge = np.zeros_like(var)
        var_sponge[:, :self.sponge_ny] = self.sponge[np.newaxis, :]
        var_sponge[:, -self.sponge_ny:] = self.sponge[::-1][np.newaxis, :]
        return self.r*var_sponge*var


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

    def _apply_boundary_conditions(self):

        # left and right-hand boundaries are the same for u
        self._u[0, :] = self._u[-3, :]
        self._u[1, :] = self._u[-2, :]
        self._u[-1, :] = self._u[2, :]

        self._v[0, :] = self._v[-2, :]
        self._v[-1, :] = self._v[1, :]
        self._phi[0, :] = self._phi[-2, :]
        self._phi[-1, :] = self._phi[1, :]

        fields = self._u, self._v, self._phi
        # top and bottom boundaries: zero deriv and damping
        for field in fields:
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
            self._fix_boundary_corners(field)

    def _apply_boundary_conditions_to(self, field):
        # periodic boundary in the x-direction
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

        # top and bottom boundaries: zero deriv and damping
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

        self._fix_boundary_corners(field)


    def _fix_boundary_corners(self, field):
        # fix corners to be average of neighbours
        field[0, 0] =  0.5*(field[1, 0] + field[0, 1])
        field[-1, 0] = 0.5*(field[-2, 0] + field[-1, 1])
        field[0, -1] = 0.5*(field[1, -1] + field[0, -2])
        field[-1, -1] = 0.5*(field[-1, -2] + field[-2, -1])

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

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""

        self._apply_boundary_conditions()
        u_at_v, v_at_u = self.uvatuv()   # (nx, ny+1), (nx+1, ny)
        ubarx = self.x_average(self._u)[:, 1:-1]    # u averaged to v lons
        ubary = self.y_average(self._u)[1:-1, :]    # u averaged to v lats

        vbary = self.y_average(self._v)[1:-1, :]
        vbarx = self.x_average(self._v)[:, 1:-1]


        # the height equation
        phi_at_u = self.x_average(self._phi)[:, 1:-1]  # (nx+1, ny)
        phi_at_v = self.y_average(self._phi)[1:-1, :]  # (nx, ny+1)

        phi_rhs  = - self.diffx(phi_at_u * self.u) - self.diffy(phi_at_v * self.v)  # (nx, ny)
        phi_rhs += self.nu*self.del2(self._phi)  # diffusion
        #phi_rhs -= self.damping(self.phi)        # damping at top and bottom boundaries


        # the u equation
        dhdx = self.diffx(self._phi)[:, 1:-1]  # (nx+2, ny)
        ududx = 0.5*self.diffx(ubarx**2)            # u*du/dx at u points
        vdudy = v_at_u*self.diffy(ubary)            # v*du/dy at u points

        u_rhs  = -dhdx + (self.f0 + self.beta*self.uy)*v_at_u
        u_rhs += self.nu*self.del2(self._u)
        u_rhs += - ududx - vdudy               # nonlin u advection terms
        #u_rhs -= self.damping(self.u)


        # the v equation
        dhdy  = self.diffy(self._phi)[1:-1, :]
        udvdx = u_at_v*self.diffx(vbarx)
        vdvdy = 0.5*self.diffy(vbary**2)            # v*dv/dy at v points

        v_rhs  = -dhdy -(self.f0 + self.beta*self.vy)*u_at_v
        v_rhs += self.nu*self.del2(self._v)
        v_rhs += - udvdx - vdvdy
        #v_rhs -= self.damping(self.v)

        dstate = np.array([u_rhs, v_rhs, phi_rhs])

        for fn in self._forcings:
            dstate += fn(self)

        return dstate

    def step(self):
        dt, tc = self.dt, self.tc

        newstate = self.state + next(self._stepper)
        self.state = newstate


        self.t  += dt
        self.tc += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.signal

    def background(spectra, fsteps=10, ksteps=10):
        """Uses a 1-2-1 filter to generate 'red noise' background field for a spectra (as per WK1998)
            `fsteps` is the number of times to apply the filter in the frequency direction
            `ksteps` is the number of times to apply the filter in the wavenumber direction

        Returns a background field of same dimensions as `spectra`.
        """
        # create a 1D 1-2-1 averaging footprint
        bgf = spectra
        for i in range(fsteps):
            # repeated application of the 1-2-1 blur filter to the spectra
            footprint = np.array([[0,1,0], [0,2,0], [0,1,0]]) / 4.0
            bgf = scipy.signal.convolve2d(bgf, footprint, mode='same', boundary='wrap')
        for i in range(ksteps):
            # repeated application of the 1-2-1 blur filter to the spectra
            footprint = np.array([[0,0,0], [1,2,1], [0,0,0]]) / 4.0
            bgf = scipy.signal.convolve2d(bgf, footprint, mode='same', boundary='wrap')

        return bgf

    nx = 128
    ny = 129
    beta=2.0e-11
    Lx = 1.0e7
    Ly = 1.0e7

    phi0 = 10.0

    ocean = PeriodicShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=3000, nu=1.0e3)

    d = 25
    hump = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]


    ocean.phi[:] += phi0
    ocean.phi[70-d:70+d, ny//2-d:ny//2+d] += hump*0.1

    initial_phi = ocean.phi.copy()

    plt.ion()

    num_levels = 24
    colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

    en = []

    eq_reg = []
    ts = []

    plt.show()
    for i in range(100000):
        ocean.step()

        if i % 10 == 0:
            eq = ocean.u[:, ny//2-5:ny//2+5]
            eq_reg.append(np.sum(eq, axis=1))
            ts.append(ocean.t)

        if i % 50 == 0:

            plt.figure(1, figsize=(12,12))
            plt.clf()

            plt.subplot(221)
            plt.plot(ocean.phi[:, ny//2])
            plt.plot(ocean.phi[:, ny//2+8])
            plt.ylim(phi0*.99, phi0*1.01)
            plt.title('Equatorial Height')

            en.append(np.sum(ocean.phi - initial_phi))
            plt.subplot(222)
            plt.plot(en)
            plt.title('Geopotential Loss')

            plt.subplot(223)
            plt.contourf(ocean.phi.T, cmap=plt.cm.RdBu, levels=phi0+colorlevels*phi0*0.01)
            plt.title('Geopotential')

            spec = np.fft.fft2(eq_reg)
            spec = spec - background(spec, 10, 0)
            nw, nk = spec.shape
            plt.subplot(224)
            plt.pcolormesh(np.fft.fftshift(np.log(np.abs(spec)))[nw//4:nw//2, nk//4:3*nk//4][::-1], cmap=plt.cm.bone)

            plt.pause(0.01)
            plt.draw()

