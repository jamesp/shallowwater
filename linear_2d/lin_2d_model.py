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


class EventEmitter(object):
    """A very simple event driven object to make it easier
    to tie custom functionality into the model."""
    def __init__(self):
        self._events = defaultdict(list)

    def on(self, event, callback=None):
        def _on(callback):
            self._events[event].append(callback)

        if callback is None:
            return _on
        else:
            return _on(callback)
        return self

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
        self.h = np.zeros((nx, ny))
        self.r = r
        self.g = g

        #self._forcings = []
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

    def step(self):
        #ur, vr, hr = self.rhs()
        self.emit('step:start', self)
        rhs = self.rhs()

        self.emit('step:force', self)
        dstate = self._dstate = rhs + np.sum(self._dstates, axis=0)
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
        # return (self.h[-1,:], self.h[0,:], self.h[:,-1], self.h[:,0])
        # no derivative at the boundaries
        return (self.h[0,:], self.h[-1,:], self.h[:,-1], self.h[:,0])

    def _vbc(self):
        """Returns the v boundary values."""
        # no derivative at the boundaries
        return (self.v[0,:], self.v[-1,:], self.v[:,-1], self.v[:,0])  

    def __update_x_periodic(self):
        self.u[-1, :] = self.u[0, :]

    def _periodicl(self, psi):
        """Copy the last column to in front of the field to make zonally periodic."""
        return np.vstack((psi[-1, :], psi))

    def _periodicr(self, psi):
        """Copy the first column to after the array to make zonally periodic."""
        return np.vstack((psi, psi[0, :]))

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

    def hdiv(self):
        """Returns the horizontal divergence at h points."""
        return self.diffx(self.u) + self.diffy(self.v)

    def uvatuv(self):
        """Calculate the value of u at v and v at u."""
        ul, ur, ut, ub = self._ubc()       # need the boundary conditions to average u at v points
        uu = np.hstack([ub[:, np.newaxis], self.u, ut[:, np.newaxis]])
        ubar = 0.25*(uu[:-1,:-1] + uu[:-1,1:] + uu[1:, :-1] + uu[1:,1:])
        
        vl, vr, vt, vb = self._vbc() 
        vv = np.vstack([vl, self.v, vr])
        vbar = 0.25*(vv[:-1,:-1] + vv[:-1,1:] + vv[1:,:-1] + vv[1:,1:])
        return ubar, vbar

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""
        # the height equation
        h_rhs = -self.H*(self.diffx(self.u) + self.diffy(self.v))
        
        uu, vv = self.uvatuv()
        
        # the u equation
        hl, hr, ht, hb = self._hbc()
        hx = np.vstack([hl, self.h, hr])
        dhdx = self.diffx(hx)
        u_rhs = self.f*vv - self.g*dhdx
        
        # the v equation
        hy = np.hstack([hb[:, np.newaxis], self.h, ht[:, np.newaxis]])
        dhdy = self.diffy(hy)
        v_rhs = -self.f*uu - self.g*dhdy

        return np.array([u_rhs, v_rhs, h_rhs])

    @property
    def state(self):
        return np.array([self.u, self.v, self.h])

    @state.setter
    def state(self, value):
        self.u, self.v, self.h = value

    def run(self):
        self.emit('initialise', self)
        while self.t < self.maxt:
            self.step()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    plt.ion()

    # Geostrophic Adjustment example
    # There are no 

    nx=320
    ny=80
    sw = LinearShallowWater(nx, ny, f=0.1, maxt=1000.0)

    # set an initial condition of height discontinuity along x = Lx/2
    IC =  np.zeros_like(sw.h)
    IC[:sw.nx/2, :] = sw.H * 0.01
    IC[sw.nx/2:, :] = -sw.H * 0.01

    def plot_hv(m):
        plt.clf()
        plt.subplot(211)
        plt.plot(m.H + m.h[:, ny/2].T)
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

    @sw.on('initialise')
    def init_model(m):
        m.h[:] = IC
        plot_hv(m)
        plt.pause(1)


    @sw.on('step:force')
    def dissipate(m):
        # add dissipation terms - basic rayleigh friction
        dstate = -m.state*0.01
        m.apply_forcing(dstate)

    rel_profile = np.concatenate([np.linspace(1, 0, nx/2), np.linspace(0,1, nx/2)])[:, np.newaxis]
    @sw.on('step:force')
    def relax(m):
        # relax back towards the initial condition
        # relax faster at the boundaries
        dstate = np.zeros_like(m.state)
        dstate[2] = (IC - m.h)*rel_profile*0.1
        m.apply_forcing(dstate)


    @sw.on('step:end')
    def print_status(m):
        print('time %f: Max vel. %.3g' % (m.t, np.max(m.u)))
        print('max h: %.3g' % np.max(m.h))
        if m.tc % 10 == 1:
            plot_hv(m)

    sw.run()
