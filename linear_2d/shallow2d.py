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

from collections import defaultdict
 
import numpy as np
from numpy import mod, floor, ceil

def centre_average(phi):
    """Returns the four-point average at the centres between grid points."""
    return 0.25*(phi[:-1,:-1] + phi[:-1,1:] + phi[1:, :-1] + phi[1:,1:])



class EventEmitter(object):
    """A very simple event driven object to make it easier
    to tie custom functionality into the model."""
    def __init__(self):
        self._events = defaultdict(list)

    def on(self, event, callback=None):
        def _on(callback):
            self._events[event].append(callback)
            return self

        if callback is None:
            return _on           # used as a decorator
        else:
            return _on(callback) # used as a normal function

    def emit(self, event, *args, **kwargs):
        for callback in self._events[event]:
            callback(*args, **kwargs)


class LinearShallowWater(EventEmitter):
    """A model of the two-dimensional linearised shallow water equations"""
    def __init__(self, nx, ny, f=1.0, r=0.01, dt=0.1, maxt=1.0, domain=(10000.0, 10000.0), H=1000.0, g=9.8):
        super(LinearShallowWater, self).__init__()
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.f  = f
        self.maxt = maxt
        self.domain = domain
        self.dx = domain[0] / nx
        self.dy = domain[1] / ny
        self.H  = H
        self.u = np.zeros((nx+1, ny))
        self.v = np.zeros((nx, ny+1))    # v points are on horizontal edges so there exists an extra point from the last vertex
        self.eta = np.zeros((nx, ny))
        self.r = r
        self.g = g

        self._forcings = []
        self._diagnostics = {}

        self._dstates = []
        self._dstate = np.zeros_like(self.state)
        
        self._pdstate = self._dstate
        self._ppdstate = self._dstate  # initialise the previous two delta states (will be overwritten once before used in Adams Bashforth)

        self.t = 0.0
        self.tc = 0  # count of steps

    # def add_force(self, force_fn):
    #   """Add a forcing function to the equations."""
    #   self._forcings.append(force_fn)

    def apply_forcing(self, dstate):
        """Apply a change to the state at this timestep."""
        self._dstates.append(dstate)

    def forcing(self, fn):
        """Add a forcing term to the model.  Typically used as a decorator:

            sw = LinearShallowWater(nx, ny)

            @sw.forcing
            def dissipate(swmodel):
                dstate = np.zeros_like(swmodel.state)
                dstate[:] = -swmodel.state*0.001
                return dstate
        Forcing functions should take a single argument for the model object itself,
        and return a state delta the same shape as state.
        """
        self._forcings.append(fn)
        return fn

    def diagnostic(self, name, fn=None):
        """Add a diagnostic calculation to the model.  Typically used as a decorator:

            sw = LinearShallowWater(nx, ny)

            @sw.diagnostic('q')
            def pot_vort(swmodel):
                q = np.zeros_like(swmodel.eta)
                q[:] = calc_vorticity(swmodel) + f0
                return q           
        """
        def _diagnostic(fn):
            self._diagnostics[name] = fn
            return fn

        if fn is None:
            return _diagnostic           # used as a decorator
        else:
            return _diagnostic(fn)       # used as a normal function


    def calc_diagnostic(self, name):
        if name in self._diagnostics:
            val = self._diagnostics[name](self)
            return val
        else:
            if name is 'vorticity':
                return self.vorticity
            if name is 'divergence':
                return self.divergence
            raise Error('Diagnostic %r not defined.' % name)


    def step(self):
        #ur, vr, hr = self.rhs()
        self.emit('step:start', self)
        rhs = self.rhs()

        self.emit('step:force', self)
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
        self.emit('step:end', self)


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
    
    @property
    def divergence(self):
        """Returns the horizontal divergence at h points."""
        return self.diffx(self.u) + self.diffy(self.v)

    @property
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

    def uvatuv(self):
        """Calculate the value of u at v and v at u."""
        ul, ur, ut, ub = self._ubc()       # need the boundary conditions to average u at v points
        uu = self._add_tb_bcs(ub, self.u, ut)
        ubar = centre_average(uu)
        
        vl, vr, vt, vb = self._vbc() 
        vv = self._add_lr_bcs(vl, self.v, vr)
        vbar = centre_average(vv)
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
        u_rhs = self.f*vv - self.g*dhdx
        
        # the v equation
        hy = np.hstack([hb[:, np.newaxis], self.eta, ht[:, np.newaxis]])
        dhdy = self.diffy(hy)
        v_rhs = -self.f*uu - self.g*dhdy

        return np.array([u_rhs, v_rhs, h_rhs])

    @property
    def state(self):
        return np.array([self.u, self.v, self.eta])

    @state.setter
    def state(self, value):
        self.u, self.v, self.eta = value

    def run(self):
        self.emit('initialise', self)
        while self.t < self.maxt:
            self.step()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    plt.ion()

    # Geostrophic Adjustment example
    # There is no variation in the y-dimension

    nx=320
    ny=80
    sw = LinearShallowWater(nx, ny, f=0.1, maxt=10000.0)

    # set an initial condition of height discontinuity along x = Lx/2
    IC =  np.zeros_like(sw.eta)
    IC[:sw.nx/2, :] = sw.H * 0.01
    IC[sw.nx/2:, :] = -sw.H * 0.01
    sw.eta[:] = IC


    @sw.on('initialise')
    def init_model(m):
        plot_hv(m)
        plot_vorticity(m)
        plt.pause(3)


    @sw.forcing
    def dissipate(m):
        # add dissipation terms - basic rayleigh friction
        dstate = -m.state*0.001
        #m.apply_forcing(dstate)
        return dstate

    rel_profile = np.concatenate([np.linspace(1, 0, nx/2), np.linspace(0,1, nx/2)])[:, np.newaxis]
    @sw.forcing
    def relax(m):
        # relax back towards the initial condition
        # relax faster at the boundaries
        dstate = np.zeros_like(m.state)
        dstate[2] = (IC - m.eta)*rel_profile*0.1
        #m.apply_forcing(dstate)
        return dstate

    @sw.diagnostic('q')
    def potential_vorticity(m):
        q = centre_average(m.vorticity) - m.f*m.eta/m.H
        return q

    q0 = sw.calc_diagnostic('q')

    def plot_hv(m):
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.plot(m.H + m.eta[:, ny/2].T)
        plt.xlabel('x')
        plt.ylabel('height (m)')
        plt.ylim((m.H-20, m.H+20))
        plt.subplot(212)
        plt.plot(m.v[:, ny/2].T)
        plt.xlabel('x')
        plt.ylabel('v velocity (m.s^-2)')
        plt.ylim((-1, 1))
        plt.pause(0.01)
        plt.draw()

    ts = []
    qs = []
    def plot_vorticity(m):
        plt.figure(2)
        plt.clf()
        
        plt.subplot(211)
        q = m.calc_diagnostic('q')
        plt.imshow((q - q0).T)
        plt.colorbar()

        ts.append(m.t)
        qs.append(np.sum(m.vorticity))
        plt.subplot(212)
        plt.plot(ts, qs)

        plt.pause(0.01)
        plt.draw()

    @sw.on('step:end')
    def print_status(m):
        print('time %f: Max vel. %.3g' % (m.t, np.max(m.u)))
        print('max h: %.3g' % np.max(m.eta))
        print('vorticity: %.3g' % np.sum(m.vorticity))
        if m.tc % 10 == 1:
            plot_hv(m)
        if m.tc % 100 == 2:
            plot_vorticity(m)

    sw.run()
