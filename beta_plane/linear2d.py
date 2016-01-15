#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
"""Shallow Water Model
  
- Two dimensional shallow water in a rotating frame
- Staggered Arakawa-C lat:lon grid
- periodic in the x-dimension
- fixed boundary conditions in the y-dimension
 
h = H + η
 
∂/∂t[u] - fv = - g ∂/∂x[η]
∂/∂t[v] + fu = - g ∂/∂y[η]
∂/∂t[h] + H(∂/∂x[u] + ∂/∂y[v]) = 0
 
f = f0 + βy
"""

import numpy as np

class LinearShallowWater(object):
    def __init__(self, nx, ny, Lx=1.0e7, Ly=1.0e7, f0=0, beta=2.0e-11, H=100.0, g=0.05, nu=1.0e-5, r=1.0e-5, dt=1000.0, bcond='periodicx'):
        super(LinearShallowWater, self).__init__()
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.f0 = f0
        self.beta = beta
        self.H = H
        self.g = g

        # Arakawa-C grid
        # +-- v --+   
        # |       |    * (nx, ny)   h points at grid centres
        # u   h   u    * (nx+1, ny) u points on vertical edges  (u[0] and u[nx] are boundary values)
        # |       |    * (nx, ny+1) v points on horizontal edges
        # +-- v --+
        self._u = np.zeros((nx+3, ny+2), dtype=np.float)
        self._v = np.zeros((nx+2, ny+3), dtype=np.float)
        self._h = np.zeros((nx+2, ny+2), dtype=np.float)

        self.dx = dx = float(Lx) / nx
        self.dy = dy = float(Ly) / ny

        # positions of the nodes
        self.ux = (-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]
        self.vx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]

        self.vy = (-Ly/2 + np.arange(ny+1)*dy)[np.newaxis, :]
        self.uy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

        self.hx = self.vx
        self.hy = self.uy

        self.bcond = bcond

        # dissipation and friction
        self.nu = nu
        self.r = r
        self.sponge_ny = ny//7
        self.sponge = np.exp(-np.linspace(0, 5, self.sponge_ny))
        #self.sponge = np.linspace(1.0, 0.0, self.sponge_ny)
        #print(self.sponge)

        # timestepping
        self._pdstate, self._ppdstate = 0.0, 0.0
        self.dt = dt
        self.tc = 0  # number of timesteps taken
        self.t = 0.0

        self._forcings = []

    # define u, v and h properties to return state without the boundaries
    @property
    def u(self):
        return self._u[1:-1, 1:-1]

    @property
    def v(self):
        return self._v[1:-1, 1:-1]

    @property
    def h(self):
        return self._h[1:-1, 1:-1]

    @property
    def state(self):
        return np.array([self.u, self.v, self.h])
    
    @state.setter
    def state(self, value):
        u, v, h = value
        self.u[:] = u
        self.v[:] = v
        self.h[:] = h


    def add_forcing(self, fn):
        """Add a forcing term to the model.  Typically used as a decorator:

            sw = LinearShallowWater(nx, ny)

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
    def diffx(self, phi):
        """Calculate ∂/∂x[phi] over a single grid square.
         
        i.e. d/dx(phi)[i,j] = (phi[i+1/2, j] - phi[i-1/2, j]) / dx
         
        The derivative is returned at x points at the midpoint between
        x points of the input array."""
        return (phi[1:,:] - phi[:-1,:]) / self.dx

    def diffy(self, phi):
        """Calculate ∂/∂y[phi] over a single grid square.
         
        i.e. d/dy(phi)[i,j] = (phi[i, j+1/2] - phi[i, j-1/2]) / dy
         
        The derivative is returned at y points at the midpoint between
        y points of the input array."""
        return (phi[:, 1:] - phi[:,:-1]) / self.dy

    def del2(self, phi):
        """Returns the Laplacian of phi."""
        return self.diff2x(phi)[:, 1:-1] + self.diff2y(phi)[1:-1, :]

    def diff2x(self, phi):
        """Calculate ∂2/∂x2[phi] over a single grid square.
         
        i.e. d2/dx2(phi)[i,j] = (phi[i+1, j] - phi[i, j] + phi[i-1, j]) / dx^2
         
        The derivative is returned at the same x points as the
        x points of the input array, with dimension (nx-2, ny)."""
        return (phi[:-2, :] - 2*phi[1:-1, :] + phi[2:, :]) / self.dx**2

    def diff2y(self, phi):
        """Calculate ∂2/∂y2[phi] over a single grid square.
         
        i.e. d2/dy2(phi)[i,j] = (phi[i, j+1] - phi[i, j] + phi[i, j-1]) / dy^2
         
        The derivative is returned at the same y points as the
        y points of the input array, with dimension (nx, ny-2)."""
        return (phi[:, :-2] - 2*phi[:, 1:-1] + phi[:, 2:]) / self.dy**2

    def centre_average(self, phi):
        """Returns the four-point average at the centres between grid points.
        If phi has shape (nx, ny), returns an array of shape (nx-1, ny-1)."""
        return 0.25*(phi[:-1,:-1] + phi[:-1,1:] + phi[1:, :-1] + phi[1:,1:])
     
    def y_average(self, phi):
        """Average adjacent values in the y dimension.
        If phi has shape (nx, ny), returns an array of shape (nx, ny-1)."""
        return 0.5*(phi[:,:-1] + phi[:,1:])
     
    def x_average(self, phi):
        """Average adjacent values in the x dimension.
        If phi has shape (nx, ny), returns an array of shape (nx-1, ny)."""
        return 0.5*(phi[:-1,:] + phi[1:,:])

    def divergence(self):
        """Returns the horizontal divergence at h points."""
        return self.diffx(self.u) + self.diffy(self.v)

    def vorticity(self):
        """Returns the vorticity at grid corners."""
        return self.diffy(self.u)[1:-1, :] - self.diffx(self.v)[:, 1:-1]

    def _apply_boundary_conditions(self):
        fields = self._u, self._v, self._h

        if 'periodicx' in self.bcond:
            for field in fields:
                # copy the left rows over to the right
                field[:2, :] = field[-3:-1, :]
                field[-1, :] = field[2, :]

        # add solid walls on left and right: no zonal flow through them
        if 'wallsx' in self.bcond:
            self._u[:2, :]  = 0
            self._u[-2:, :] = 0
            for field in (self._v, self._h):
                # zero derivative for v and h fields
                field[0,:] = field[1, :]
                field[-1,:] = field[-2, :]

        # top and bottom boundaries: zero deriv and damping
        for field in fields:
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]

        for field in fields:
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
        f0, beta, g, H, nu = self.f0, self.beta, self.g, self.H, self.nu
        
        self._apply_boundary_conditions()
        uu, vv = self.uvatuv()

        # the height equation
        h_rhs = -H*self.divergence() + nu*self.del2(self._h) - self.damping(self.h)

        # the u equation
        dhdx = self.diffx(self._h)[:, 1:-1]
        u_rhs = (f0 + beta*self.uy)*vv - g*dhdx + nu*self.del2(self._u) - self.damping(self.u)
        
        # the v equation
        dhdy  = self.diffy(self._h)[1:-1, :]
        v_rhs = -(f0 + beta*self.vy)*uu - g*dhdy + nu*self.del2(self._v) - self.damping(self.v)

        dstate = np.array([u_rhs, v_rhs, h_rhs])
        
        for fn in self._forcings:
            dstate += fn(self)

        return dstate

    def step(self):
        dt, tc = self.dt, self.tc

        self._apply_boundary_conditions()
        
        dstate = self.rhs()

        # take adams-bashforth step in time
        if tc==0:
            # forward euler
            dt1 = dt
            dt2 = 0.0
            dt3 = 0.0
        elif tc==1:
            # AB2 at step 2
            dt1 = 1.5*dt
            dt2 = -0.5*dt
            dt3 = 0.0
        else:
            # AB3 from step 3 on
            dt1 = 23./12.*dt
            dt2 = -16./12.*dt
            dt3 = 5./12.*dt
        
        newstate = self.state + dt1*dstate + dt2*self._pdstate + dt3*self._ppdstate
        self.state = newstate
        self._ppdstate = self._pdstate
        self._pdstate = dstate

        self.t  += dt
        self.tc += 1


if __name__ == '__main__':
        
    nx = 128
    ny = 129
    beta=2.0e-11
    Lx = 1.0e7
    Ly = 1.0e7

    ocean = LinearShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, g=0.1, H=100.0, dt=3000, nu=1000.0, bcond='wallsx')
    #ocean.h[10:20, 60:80] = 1.0
    #ocean.h[-20:-10] = 1.0
    d = 25
    ocean.h[10:10+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    #ocean.h[100:100+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    import matplotlib.pyplot as plt


    atmos = LinearShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, g=3.0, H=10.0, dt=1000, bcond='periodicx')
    plt.ion()

    num_levels = 24
    colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

    ts = []
    es = []
    plt.show()
    for i in range(10000):
        ocean.step()
        if i % 10 == 0:
            plt.figure(1)
            plt.clf()
            #plt.plot(ocean.h[:,0])
            #plt.plot(ocean.h[:,64])
            #plt.ylim(-1,1)
            plt.contourf(ocean.h.T, cmap=plt.cm.RdBu, levels=colorlevels)

            plt.figure(2)
            plt.clf()
            plt.plot(ocean.h[:,0])
            plt.plot(ocean.h[:,48])
            plt.plot(ocean.h[:,64])
            plt.ylim(-1,1)

            plt.figure(3)
            plt.clf()
            energy = np.sum(ocean.g*ocean.h) + np.sum(ocean.u**2) + np.sum(ocean.v**2)
            ts.append(ocean.t)
            es.append(energy)
            plt.plot(ts, es)

            plt.pause(0.01)
            plt.draw()
