#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shallow Water Model

- Two dimensional shallow water in a rotating frame
- Staggered Arakawa-C lat:lon grid
- periodic in the x-dimension
- fixed boundary conditions in the y-dimension

η = H + h

∂/∂t[u] - fv = - g ∂/∂x[h]
∂/∂t[v] + fu = - g ∂/∂y[h]
∂/∂t[h] + H(∂/∂x[u] + ∂/∂y[v]) = 0

f = f0 + βy
"""

import numpy as np

from arakawac import ArakawaCGrid, PeriodicBoundaries, WallBoundaries
from timesteppers import AdamsBashforth3

class Tracer(AdamsBashforth3):
    kappa = 0.0

    def __init__(self, grid):
        self.grid = grid
        self.dt = grid.dt
        self._value = np.zeros_like(grid._phi)
        self.forcings = []

    @property
    def value(self):
        return self._value[1:-1, 1:-1]

    @value.setter
    def value(self, newvalue):
        self.value[:] = newvalue

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

    def __iter__(self):
        return self.value.__iter__()

    def add_forcing(self, fn):
        """Add a forcing term to the model.  Typically used as a decorator:

            sw = PeriodicShallowWater(nx, ny)
            q = sw.add_tracer('q')

            @q.add_forcing
            def dissipate(q):
                dstate = np.zeros_like(q.value)
                dstate[:] = -q.value*0.001
                return dstate

        Forcing functions should take a single argument for the tracer object itself,
        and return a value delta the same shape as value.
        """
        self.forcings.append(fn)
        return fn

    def apply_boundary_conditions(self):
        self.grid.apply_boundary_conditions_to(self._value)

    def rhs(self):
        """Calculates the right-hand side of the timestepping equation."""
        dvalue = np.zeros_like(self.value)
        dvalue -= self._dynamics_terms()
        dvalue += self._diffusive_term()
        for f in self.forcings:
            dvalue += f(self)
        return dvalue

    def _dynamics_terms(self):
        """Calculates the conservation of an advected tracer.

        ∂[q]/∂t + ∇ . (uq) = 0

        Returns the divergence term i.e. ∇.(uq)
        """
        q = self._value
        grid = self.grid

        # the height equation
        q_at_u = grid.x_average(q)[:, 1:-1]  # (nx+1, ny)
        q_at_v = grid.y_average(q)[1:-1, :]  # (nx, ny+1)

        return grid.diffx(q_at_u * grid.u) + grid.diffy(q_at_v * grid.v)  # (nx, ny)

    def _diffusive_term(self):
        return self.kappa*self.grid.del2(self._value)

    def step(self):
        newval = self._step()
        self.value = self.value + newval

class LinearShallowWater(ArakawaCGrid, AdamsBashforth3):
    def __init__(self, nx, ny, Lx=1.0e7, Ly=1.0e7, f0=0.0, beta=0.0, g=9.8, H=10.0, nu=1.0e3, nu_h=None, r=1.0e-5, dt=1000.0):
        super(LinearShallowWater, self).__init__(nx, ny, Lx, Ly)

        # Coriolis terms
        self.f0 = f0
        self.beta = beta

        self.g = g
        self.H = H

        # dissipation and friction
        self.nu = nu                                    # u, v dissipation
        self.nu_h = nu if nu_h is None else nu_h        # h dissipation
        self.r = r
        self.sponge_ny = ny//7
        self.sponge = np.exp(-np.linspace(0, 5, self.sponge_ny))

        # timestepping
        self.dt = dt

        self.forcings = []
        self.tracers  = []

        self.hx = self.phix
        self.hy = self.phiy

    # make h an proxy for phi
    @property
    def h(self):
        return self.phi

    @property
    def _h(self):
        return self._phi

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
        self.forcings.append(fn)
        return fn

    def damping(self, var):
        # sponges are active at the top and bottom of the domain by applying Rayleigh friction
        # with exponential decay towards the centre of the domain
        var_sponge = np.zeros_like(var)
        var_sponge[:, :self.sponge_ny] = self.sponge[np.newaxis, :]
        var_sponge[:, -self.sponge_ny:] = self.sponge[::-1][np.newaxis, :]
        return self.r*var_sponge*var

    def _dynamics_terms(self):
        """Calculate the dynamics of the u, v and h equations."""
        f0, beta, g, H, nu = self.f0, self.beta, self.g, self.H, self.nu

        uu, vv = self.uvatuv()

        # the height equation
        dh = -H*self.divergence() - self.damping(self.h)

        # the u equation
        dhdx = self.diffx(self._h)[:, 1:-1]
        du = (f0 + beta*self.uy)*vv - g*dhdx  - self.damping(self.u)

        # the v equation
        dhdy  = self.diffy(self._h)[1:-1, :]
        dv = -(f0 + beta*self.vy)*uu - g*dhdy - self.damping(self.v)

        dstate = np.array([du, dv, dh])
        return dstate

    def _dissipation_terms(self):
        du = self.nu*self.del2(self._u)
        dv = self.nu*self.del2(self._v)
        dh = self.nu_h*self.del2(self._h)
        return np.array([du, dv, dh])

    def rhs(self):
        """Calculates the right-hand side of the timestepping equation."""
        dstate = np.zeros_like(self.state)
        dstate += self._dynamics_terms()
        dstate += self._dissipation_terms()
        for f in self.forcings:
            dstate += f(self)
        return dstate

    def add_tracer(self, name):
        """Add a tracer to the shallow water model.

        Dq/Dt + q(∇ . u) = k∆q + F

        Tracers are advected by the flow. Forcings can be added to a tracer
        in the same way they can to the basic shallow water flow.

        Once a tracer has been added to the model it's value can be accessed
        by the self.<name>.value.
        """
        t = Tracer(self)
        setattr(self, name, t)
        self.tracers.append(t)
        return t

    def step(self):
        # update boundaries and calculate step forward
        self.apply_boundary_conditions()
        newfields = []
        for t in self.tracers:
            t.apply_boundary_conditions()
            newfield = t.value + t._step()
            newfields.append(newfield)
        newstate = self.state + self._step()

        # set the tracer and state values to new vals
        for t, newfield in zip(self.tracers, newfields):
            t.value = newfield
        self.state = newstate

# examples of class composition
class PeriodicLinearShallowWater(PeriodicBoundaries, LinearShallowWater): pass
class WalledLinearShallowWater(WallBoundaries, LinearShallowWater): pass

if __name__ == '__main__':
    nx = 128
    ny = 129
    beta=2.0e-11
    Lx = 1.0e7
    Ly = 1.0e7

    ocean = PeriodicLinearShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, g=0.1, H=100.0, dt=3000, nu=1000.0)

    d = 25
    #ocean.h[10:10+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    #ocean.h[100:100+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    import matplotlib.pyplot as plt

    @ocean.add_forcing
    def heating(ocean):
        dstate = np.zeros_like(ocean.state)
        Q = np.zeros_like(ocean.h)
        Q[10:10+2*d, ny//2-d:ny//2+d+1] = (np.sin(np.linspace(0, np.pi, 2*d+1))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
        dstate[2] += Q*1e-6
        dstate[2] -= ocean.h * 1e-6
        return dstate
    plt.ion()

    num_levels = 24
    colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

    q = ocean.add_tracer('q')
    q[:] = 4.0

    ts = []
    es = []
    plt.show()
    for i in range(10000):
        ocean.step()
        if i % 10 == 0:
            print('[t={:7.2f} h range [{:.2f}, {:.2f}]'.format(ocean.t/86400, ocean.h.min(), ocean.h.max()))
            plt.clf()
            plt.subplot(221)
            plt.contourf(ocean.h.T, cmap=plt.cm.RdBu, levels=colorlevels)

            plt.subplot(222)
            plt.plot(ocean.h[:,0])
            plt.plot(ocean.h[:,48])
            plt.plot(ocean.h[:,64])
            plt.ylim(-1,1)

            plt.subplot(223)
            energy = np.sum(ocean.g*ocean.h) + np.sum(ocean.u**2) + np.sum(ocean.v**2)
            ts.append(ocean.t)
            es.append(energy)
            plt.plot(ts, es)

            plt.subplot(224)
            plt.imshow(ocean.q.value.T - 4.0, cmap=plt.cm.RdBu)
            plt.clim(-.1, .1)
            plt.colorbar()

            plt.pause(0.01)
            plt.draw()
