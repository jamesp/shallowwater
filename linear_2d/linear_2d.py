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


SHOW_CHART = True

class LinearShallowWater(object):
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

        self.prev_state = self.state

        self.t = 0.0

        
    def _ubc(self):
        """Returns the u velocity boundaries.
        Returns tuple (left, right, top, bottom)."""
        # periodic in x, zero derivative on y
        return (self.u[-1,:], self.u[0,:], self.u[:,-1], self.u[:,0])

    def _hbc(self):
        """Returns the h boundaries.
        Returns tuple (left, right, top, bottom)."""
        # periodic in x, zero derivative on y
        return (self.h[-1,:], self.h[0,:], self.h[:,-1], self.h[:,0])

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
        ur = np.hstack([ub[:, np.newaxis], self.u, ut[:, np.newaxis]])
       
        ubar = np.zeros_like(self.v)
        ubar[:,:] = 0.25*(ur[:-1,:-1] + ur[:-1,1:] + ur[1:, :-1] + ur[1:,1:])
        
        vl = self._periodicl(self.v)
        vbar = 0.25*(vl[:-1,:-1] + vl[:-1,1:] + vl[1:,:-1] + vl[1:,1:])
        return ubar, vbar

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""
        # the height equation
        h_rhs = -self.H*(self.diffx(self.u) + self.diffy(self.v))
        
        # the u and v equations
        uu, vv = self.uvatuv()
        
        hx = self._periodicl(self.h)
        dhdx = self.diffx(hx)
        u_rhs = self.f*vv - self.g*dhdx
        u_rhs = self._periodicr(u_rhs) #+ self.r*self.u
        
        hl, hr, ht, hb = self._hbc()
        hy = np.hstack([hb[:, np.newaxis], self.h, ht[:, np.newaxis]])
        dhdy = self.diffy(hy)
        v_rhs = -self.f*uu - self.g*dhdy #+ self.r*self.v
        return np.array([u_rhs, v_rhs, h_rhs])


    def raw_filter(self, prev, curr, new, nu=0.01):
        """Robert-Asselin Filter."""
        return curr + nu*(new - 2.0*curr + prev)

    @property
    def state(self):
        return np.array([self.u, self.v, self.h])

    @state.setter
    def state(self, value):
        self.u, self.v, self.h = value

    def step(self):
        #ur, vr, hr = self.rhs()
        # take leapfrog step forward in time
        next_state = self.prev_state + 2.0*self.dt*self.rhs()
        self.prev_state = self.raw_filter(self.prev_state, self.state, next_state)
        self.state = next_state
        self.t = self.t + 2.0*self.dt



if __name__ == '__main__':
    nx=320
    ny=320
    sw = LinearShallowWater(nx, ny, f=0.1, maxt=1000.0)


    i, j = np.indices(sw.h.shape)
    sw.h[:] = np.exp(-(np.sqrt((j - ny/2.0)**2 + (i - nx/2.0)**2)/20)**2)*sw.H*0.01
    #sw.h[:] = np.random.random(sw.h.shape)*sw.H*0.01
    #sw.h[:] = np.sin(np.pi*10*j/ny)*np.sin(np.pi*11*i/nx)*10
    steps = 0

    import time
    if SHOW_CHART:
        import matplotlib
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(211)
        im1 = ax.imshow(sw.h.T, cmap=plt.cm.seismic)
        im1.set_clim((-10, 10))
        plt.colorbar(im1)

        ax = fig.add_subplot(212)
        im2 = ax.imshow(sw.u.T, cmap=plt.cm.seismic)

        plt.colorbar(im2)
        fig.show()
        time.sleep(1)
        while sw.t < sw.maxt:
            sw.step()
            print('step %f: Max vel. %.3g' % (sw.t, np.max(sw.u)))
            print('max h: %.3g' % np.max(sw.h))
            if steps % 10 == 0:
                im1.set_data(sw.h.T)
                im2.set_data(sw.u.T)
                #im1.set_clim((np.min(sw.h), np.max(sw.h)))
                im2.set_clim((np.min(sw.u), np.max(sw.u)))
                im1.axes.figure.canvas.draw()
                im2.axes.figure.canvas.draw()
                plt.pause(0.0001)
            steps = steps + 1