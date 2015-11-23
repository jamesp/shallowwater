#!/usr/bin/env bash
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
from numpy import mod, floor, ceil

def centre_average(phi):
    """Returns the four-point average at the centres between grid points."""
    return 0.25*(phi[:-1,:-1] + phi[:-1,1:] + phi[1:, :-1] + phi[1:,1:])

def y_average(phi):
    """Average adjacent values in the y dimension.
    If phi has shape (nx, ny), returns an array of shape (nx, ny - 1)."""
    return 0.5*(phi[:,:-1] + phi[:,1:])

def x_average(phi):
    """Average adjacent values in the x dimension.
    If phi has shape (nx, ny), returns an array of shape (nx - 1, ny)."""
    return 0.5*(phi[:-1,:] + phi[1:,:])


class LinearShallowWater(object):
    """A model of the two-dimensional linearised shallow water equations"""
    def __init__(self, nx, ny, f0=1.0, beta=0.0, dt=0.1, maxt=1.0, domain=(10000.0, 10000.0), H=1000.0, g=9.8):
        super(LinearShallowWater, self).__init__()
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.f0  = f0
        self.beta = beta  # TODO: Implement beta plane
        self.maxt = maxt
        self.domain = domain
        self.Lx = domain[0]
        self.Ly = domain[1]
        self.dx = dx = self.Lx / nx
        self.dy = dy = self.Ly / ny
        self.H  = H
        self.u = np.zeros((nx+1, ny))    # u points are on the vertical edges
        self.v = np.zeros((nx, ny+1))    # v points are on horizontal edges so there is an extra point from the last vertex
        self.eta = np.zeros((nx, ny))    # h points are at cell centres
        self.g = g

        # the positions of u, v and h nodes on the grid
        self.ux = np.linspace(0, domain[0], nx+1)[:, np.newaxis]
        self.uy = dy/2.0 + np.linspace(0, domain[0], ny)[np.newaxis, :]
        self.vx = dx/2.0 + np.linspace(0, domain[1], nx)[:, np.newaxis]
        self.vy = dy/2.0 + np.linspace(0, domain[1], ny+1)[np.newaxis, :]
        self.hx = self.vx
        self.hy = self.uy

        self._forcings = []
        self._diagnostics = {
            'divergence': self.divergence,
            'vorticity': self.vorticity
            }
        self._outputs = []

        self._dstates = []
        self._dstate = np.zeros_like(self.state)
        
        self._pdstate = self._dstate
        self._ppdstate = self._dstate  # initialise the previous two delta states (will be overwritten once before used in Adams Bashforth)

        self.t = 0.0
        self.tc = 0    # count of steps

    def apply_force(self, dstate):
        """Apply a forcing to the state at this timestep."""
        self._dstates.append(dstate)

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

    def add_output(self, every_n_steps=1, fn=None):
        def _diagnostic(fn):
            self._outputs.append((every_n_steps, fn))
            return fn

        if fn is None:
            return _diagnostic           # used as a decorator
        else:
            return _diagnostic(fn)       # used as a normal function

    def _process_outputs(self):
        for t, fn in self._outputs:
            if self.tc % t == 1:
                fn(self)      # perform some side effects

    def add_diagnostic(self, name, fn=None):
        """Add a diagnostic calculation to the model.  Typically used as a decorator:

            sw = LinearShallowWater(nx, ny)

            @sw.add_diagnostic('q')
            def pot_vort(swmodel):
                q = np.zeros_like(swmodel.eta)
                q[:] = calc_vorticity(swmodel) + f0
                return q

        Diagnostic functions take the model as a single argument and return the diagnostic
        value at the current timestep.
        """
        def _diagnostic(fn):
            self._diagnostics[name] = fn
            return fn

        if fn is None:
            return _diagnostic           # used as a decorator
        else:
            return _diagnostic(fn)       # used as a normal function

    def get_diagnostic(self, name):
        if name in self._diagnostics:
            val = self._diagnostics[name](self)
            return val
        else:
            raise Error('Diagnostic %r not defined.' % name)

    # shortcut to get diagnostic values
    d = get_diagnostic

    def get_all_diagnostics(self):
        return {k: fn(self) for k, fn in self._diagnostics.items()}

    def step(self):
        #ur, vr, hr = self.rhs()
        rhs = self.rhs()

        # calculate the influence of all the applied forcings
        dforce = sum([f(self) for f in self._forcings])
        dstate = self._dstate = rhs + dforce + np.sum(self._dstates, axis=0)
        self._dstates = []

        # take adams-bashforth step in time
        if self.tc==0:
            # forward euler
            dt1 = self.dt
            dt2 = 0.0
            dt3 = 0.0
        elif self.tc==1:
            # AB2 at step 2
            dt1 = 1.5*self.dt
            dt2 = -0.5*self.dt
            dt3 = 0.0
        else:
            # AB3 from step 3 on
            dt1 = 23./12.*self.dt
            dt2 = -16./12.*self.dt
            dt3 = 5./12.*self.dt

        newstate = self.state + dt1*dstate + dt2*self._pdstate + dt3*self._ppdstate
        self.state = newstate
        self._ppdstate = self._pdstate
        self._pdstate = dstate

        self.t  += self.dt
        self.tc += 1
        self._process_outputs()

    @property    
    def _state_with_bcs(self):
        """Returns the u, v and h fields with boundaries applied.
        Returns fields 2 larger in both dimensions."""
        ub = self._add_all_bcs(self.u, self._ubc())
        vb = self._add_all_bcs(self.v, self._vbc())
        hb = self._add_all_bcs(self.h, self._hbc())
        return np.array([ub, vb, hb])

    def _ubc(self):
        """Returns the u velocity boundaries.
        Returns tuple (left, right, top, bottom)."""
        # # periodic in x, zero derivative on y
        # return (self.u[-1,:], self.u[0,:], self.u[:,-1], self.u[:,0])
        # no derivative at the boundaries
        return (self.u[0,:], self.u[-1,:], self.u[:,-1], self.u[:,0])

    def _hbc(self):
        """Returns the h boundaries.\
        Returns tuple (left, right, top, bottom)."""
        # # periodic in x, zero derivative on y
        # return (self.eta[-1,:], self.eta[0,:], self.eta[:,-1], self.eta[:,0])
        # no derivative at the boundaries
        return (self.eta[0,:], self.eta[-1,:], self.eta[:,-1], self.eta[:,0])

    def _vbc(self):
        """Returns the v boundary values."""
        # no derivative at the boundaries
        return (self.v[0,:], self.v[-1,:], self.v[:,-1], self.v[:,0])  


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

    @property
    def h(self):
        return self.H + self.eta
    
    def divergence(self):
        """Returns the horizontal divergence at h points."""
        return self.diffx(self.u) + self.diffy(self.v)

    def vorticity(self):
        """Returns the vorticity calculated at grid corners."""
        ul, ur, ut, ub = self._ubc()
        vl, vr, vt, vb = self._vbc()

        uu = self._add_tb_bcs(ub, self.u, ut)
        vv = self._add_lr_bcs(vl, self.v, vr)
        return self.diffx(vv) - self.diffy(uu)

    def _add_lr_bcs(self, lbc, phi, rbc):
        """Add the left and right boundary conditions to a field."""
        return np.vstack([lbc, phi, rbc])

    def _add_tb_bcs(self, bbc, phi, tbc):
        """Add the bottom and top boundary conditions to a field."""
        return np.hstack([bbc[:, np.newaxis], phi, tbc[:, np.newaxis]])

    def _add_all_bcs(self, phi, bcs):
        l,r,t,b = bcs
        # corners are average of nearest boundary
        tl = 0.5*(l[-1] + t[0])
        bl = 0.5*(l[0] + b[0])
        tr = 0.5*(r[-1] + t[0])
        br = 0.5*(r[0] + b[0])
        with_lr = self._add_lr_bcs(l, phi, r)
        with_all = self._add_tb_bcs(np.hstack([bl, b, br]), with_lr, np.hstack([tl, t, tr]))
        return with_all

    def uvatuv(self):
        """Calculate the value of u at v and v at u."""
        uu, vv, hh = self._state_with_bcs

        # ul, ur, ut, ub = self._ubc()       # need the boundary conditions to average u at v points
        # uu = self._add_tb_bcs(ub, self.u, ut)
        ubar = centre_average(uu[1:-1, :])
        
        # vl, vr, vt, vb = self._vbc() 
        # vv = self._add_lr_bcs(vl, self.v, vr)
        vbar = centre_average(vv[:, 1:-1])
        return ubar, vbar

    def uvath(self):
        ubar = x_average(self.u)
        vbar = y_average(self.v)
        return ubar, vbar

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""
        # the height equation
        h_rhs = -self.H*(self.diffx(self.u) + self.diffy(self.v))
        
        uu, vv = self.uvatuv()
        
        # the u equation
        hl, hr, ht, hb = self._hbc()
        hx = self._add_lr_bcs(hl, self.eta, hr)
        dhdx = self.diffx(hx)
        u_rhs = self.f0*vv - self.beta*self.uy*vv - self.g*dhdx
        
        # the v equation
        hy = self._add_tb_bcs(hb, self.eta, ht)
        dhdy = self.diffy(hy)
        v_rhs = -self.f0*uu -self.beta*self.vy*uu - self.g*dhdy

        return np.array([u_rhs, v_rhs, h_rhs])

    @property
    def state(self):
        return np.array([self.u, self.v, self.eta])

    @state.setter
    def state(self, value):
        self.u, self.v, self.eta = value

    def run(self):
        while self.t < self.maxt:
            self.step()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    plt.ion()

    def plot_hv(m):
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.plot(m.H + m.eta[:, ny/2].T)
        plt.xlabel('x')
        plt.ylabel('height (m)')
        plt.ylim((m.H*0.98, m.H*1.02))
        plt.subplot(212)
        plt.plot(m.v[:, ny/2].T)
        plt.xlabel('x')
        plt.ylabel('v velocity (m.s^-2)')
        plt.ylim((-2, 2))
        plt.pause(0.01)
        plt.draw()

    ts = []
    qs = []
    def plot_vorticity(m):
        plt.figure(2)
        plt.clf()
        
        plt.subplot(211)
        q = m.d('q')
        plt.imshow((q - q0).T)
        plt.colorbar()

        ts.append(m.t)
        qs.append(np.sum(q))
        plt.subplot(212)
        plt.plot(ts, qs)

        plt.pause(0.01)
        plt.draw()


    # Geostrophic Adjustment example
    # There is no variation in the y-dimension

    nx=320
    ny=5
    sw = LinearShallowWater(nx, ny, f0=1.0, maxt=10000.0)

    @sw.add_diagnostic('q')
    def potential_vorticity(m):
        # vorticity is calculated on grid cell corners, move to grid centres to add to 
        zeta = centre_average(m.vorticity())  
        return zeta - m.f0*m.eta/m.H


    # set an initial condition of height discontinuity along x = Lx/2
    IC =  np.zeros_like(sw.eta)
    # IC[:sw.nx/2, :] = sw.H * 0.01
    # IC[sw.nx/2:, :] = -sw.H * 0.01
    IC[:] = -np.tanh((sw.hx - sw.Lx/2)/(sw.Lx/50))*sw.H*0.01
    sw.eta[:] = IC

    # calculate the initial potential vorticity
    q0 = sw.d('q')

    plot_hv(sw)
    plot_vorticity(sw)
    plt.pause(3)


    @sw.add_forcing
    def dissipate(m):
        # add dissipation terms - basic rayleigh friction on all fields
        dstate = -m.state*0.001
        return dstate

    # relax/damp faster at the left and right boundaries
    rel_profile = np.zeros(sw.nx)
    rel_profile[:10] = np.exp(-np.arange(0, 10, 1)*0.3)
    rel_profile[-10:] = np.exp(-np.arange(10, 0, -1)*0.3)
    rel_profile = rel_profile[:, np.newaxis]

    @sw.add_forcing
    def relax(m):
        # relax back towards the initial condition
        dstate = np.zeros_like(m.state)
        dstate[2] = (IC - m.eta)*rel_profile
        return dstate

    @sw.add_output(5)
    def print_status(m):
        print('time %f: Max vel. %.3g' % (m.t, np.max(m.u)))
        print('total potential vorticity: %.3g' % np.sum(m.d('q')))
    
    sw.add_output(10, plot_hv)
    sw.add_output(100, plot_vorticity)

    sw.run()
