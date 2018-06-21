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
from timesteppers import AdamsBashforth3, sync_step

class Dynamic(AdamsBashforth3):
    """Common base class for all shallow water models and tracers."""
    def __init__(self):
        super(Dynamic, self).__init__()
        self.forcings = []

    def add_forcing(self, fn):
        """Add a forcing term to the model.  Typically used as a decorator:

            @sw.add_forcing
            def dissipate(swmodel):
                dstate = np.zeros_like(swmodel.state)
                dstate[:] = -swmodel.state*0.001
                return dstate

        Forcing functions should take a single argument for the model/tracer itself,
        and return a state delta the same shape as state.
        """
        self.forcings.append(fn)
        return fn

    def _dstate(self):
        dstate = np.zeros_like(self.state)
        if self.forcings:
            for f in self.forcings:
                dstate += f(self)
        return self._dynamics() + dstate

    def _dynamics(self):
        # should be implemented by the model
        raise NotImplemented()


class Model(Dynamic):
    def __init__(self):
        super(Model, self).__init__()
        self.tracers  = {}

    def add_tracer(self, name, initial_state=0.0, kappa=0.0):
        """Add a tracer to the shallow water model.

        Dq/Dt + q(∇ . u) = k∆q

        Tracers are advected by the flow.

        `kappa` is a coefficient of dissipation.

        Once a tracer has been added to the model it's value can be accessed
        by the from the model.tracers dict, or from model.tracer_name.
        """
        t = Tracer(name, grid=self, kappa=kappa,
                            initial_state=initial_state)
        self.tracers[name] = t
        if not hasattr(self, name):
            self.__dict__[name] = t
        return t

    def tracer(self, name):
        return self.tracers[name]

    def step(self):  # override the basic timestepping `step` to support tracers
        self.apply_boundary_conditions()
        for tracer in self.tracers.values():
            tracer.apply_boundary_conditions()

        sync_step(self, *self.tracers.values())

class ShallowWater(ArakawaCGrid, Model):
    """The Shallow Water Equations on the Arakawa-C grid."""
    def __init__(self, nx, ny, Lx=1.0e7, Ly=1.0e7, f0=0.0,
                    beta=0.0, nu=1.0e3, nu_phi=None,
                    r=1.0e-5, dt=1000.0):
        super(ShallowWater, self).__init__(nx, ny, Lx, Ly)

        # Coriolis terms
        self.f0 = f0
        self.beta = beta

        # dissipation and friction
        self.nu = nu                                    # u, v dissipation
        self.nu_phi = nu if nu_phi is None else nu_phi  # phi dissipation
        self.r = r      # rayleigh damping at edges
        self.sponge_ny = ny//7
        self.sponge = np.exp(-np.linspace(0, 5, self.sponge_ny))

        # timestepping
        self.dt = dt

    def damping(self, var):
        # sponges are active at the top and bottom of the domain by applying Rayleigh friction
        # with exponential decay towards the centre of the domain
        var_sponge = np.zeros_like(var)
        var_sponge[:, :self.sponge_ny] = self.sponge[np.newaxis, :]
        var_sponge[:, -self.sponge_ny:] = self.sponge[::-1][np.newaxis, :]
        return self.r*var_sponge*var

    def _dynamics(self):
        """Calculate the dynamics for the u, v and phi equations."""
        # ~~~ Nonlinear Dynamics ~~~
        u_at_v, v_at_u = self.uvatuv()              # (nx, ny+1), (nx+1, ny)
        ubarx = self.x_average(self._u)[:, 1:-1]    # u averaged to v lons
        ubary = self.y_average(self._u)[1:-1, :]    # u averaged to v lats

        vbary = self.y_average(self._v)[1:-1, :]
        vbarx = self.x_average(self._v)[:, 1:-1]

        # the height equation
        phi_at_u = self.x_average(self._phi)[:, 1:-1]  # (nx+1, ny)
        phi_at_v = self.y_average(self._phi)[1:-1, :]  # (nx, ny+1)

        phi_rhs  = - self.diffx(phi_at_u * self.u) - self.diffy(phi_at_v * self.v)  # (nx, ny)
        phi_rhs += self.nu_phi*self.del2(self._phi)       # diffusion
        #phi_rhs -= self.damping(self.phi)               # damping at top and bottom boundaries

        # the u equation
        dhdx = self.diffx(self._phi)[:, 1:-1]       # (nx+2, ny)
        ududx = 0.5*self.diffx(ubarx**2)            # u*du/dx at u points
        vdudy = v_at_u*self.diffy(ubary)            # v*du/dy at u points

        u_rhs  = -dhdx + (self.f0 + self.beta*self.uy)*v_at_u
        u_rhs += self.nu*self.del2(self._u)
        u_rhs += - ududx - vdudy               # nonlin u advection terms
        u_rhs -= self.damping(self.u)

        # the v equation
        dhdy  = self.diffy(self._phi)[1:-1, :]
        udvdx = u_at_v*self.diffx(vbarx)
        vdvdy = 0.5*self.diffy(vbary**2)            # v*dv/dy at v points

        v_rhs  = -dhdy -(self.f0 + self.beta*self.vy)*u_at_v
        v_rhs += self.nu*self.del2(self._v)
        v_rhs += - udvdx - vdvdy
        v_rhs -= self.damping(self.v)

        dstate = np.array([u_rhs, v_rhs, phi_rhs])

        return dstate



class LinearShallowWater(ShallowWater):
    def __init__(self, nx, ny, Lx=1.0e7, Ly=1.0e7, f0=0.0, beta=0.0, g=9.8, H=10.0, nu=1.0e3, nu_phi=None, r=1.0e-5, dt=1000.0):
        super(LinearShallowWater, self).__init__(nx, ny, Lx, Ly, f0, beta, nu, nu_phi, r, dt)

        self.g = g
        self.H = H

        self.hx = self.phix
        self.hy = self.phiy

    # make h an proxy for phi
    @property
    def h(self):
        return self.phi

    @property
    def _h(self):
        return self._phi

    def _dynamics(self):
        """Calculate the dynamics of the u, v and h equations."""
        # ~~~ Linear dynamics ~~~
        f0, beta, g, H, nu = self.f0, self.beta, self.g, self.H, self.nu

        uu, vv = self.uvatuv()

        # the height equation
        h_rhs = -H*self.divergence() + self.nu_phi*self.del2(self._h) - self.damping(self.h)

        # the u equation
        dhdx = self.diffx(self._h)[:, 1:-1]
        u_rhs = (f0 + beta*self.uy)*vv - g*dhdx + nu*self.del2(self._u) - self.damping(self.u)

        # the v equation
        dhdy  = self.diffy(self._h)[1:-1, :]
        v_rhs = -(f0 + beta*self.vy)*uu - g*dhdy + nu*self.del2(self._v) - self.damping(self.v)

        dstate = np.array([u_rhs, v_rhs, h_rhs])

        return dstate


class Tracer(Dynamic):
    def __init__(self, name, grid, kappa=0.0, initial_state=0.0):
        super(Tracer, self).__init__()
        self.name = name
        self.grid = grid

        self._state = np.zeros(grid._shape)  # store tracer on cell centres
        self.kappa = kappa # diffusion

        self.state = initial_state

        self.dt = grid.dt

    @property
    def state(self):
        # view without boundary conditions
        return self._state[self.grid.true_slice]

    @state.setter
    def state(self, value):
        self._state[self.grid.true_slice] = value

    def _diffusion(self):
        return self.kappa*self.grid.del2(self._state)

    def _dynamics(self):
        return self._diffusion() - self.grid.advect(self._state)

    def rhs(self):
        """Set a right-hand side term for the equation.
        Default is 0.0, override this method when subclassing."""
        return 0.0

    def step(self):
        self.apply_boundary_conditions()
        self.state = self.state + self.dstate()
        self._incr_timestep()

    def apply_boundary_conditions(self):
        self.grid.apply_boundary_conditions_to(self._state)

    def __getattr__(self, attr):
        return getattr(self.state, attr)

    def __getitem__(self, slice):
        return self.state[slice]

    def __setitem__(self, slice, value):
        self.state[slice] = value

class PeriodicShallowWater(PeriodicBoundaries, ShallowWater): pass
class WalledShallowWater(WallBoundaries, ShallowWater): pass
class PeriodicLinearShallowWater(PeriodicBoundaries, LinearShallowWater): pass
class WalledLinearShallowWater(WallBoundaries, LinearShallowWater): pass


if __name__ == '__main__':
    nx = 128
    ny = 129
    beta=2.0e-11
    Lx = 1.0e7
    Ly = 1.0e7

    ocean = PeriodicLinearShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, g=0.1, H=100.0, dt=3000, nu=1000.0)
    #ocean.h[10:20, 60:80] = 1.0
    #ocean.h[-20:-10] = 1.0
    d = 25
    ocean.h[10:10+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    #ocean.h[100:100+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    import matplotlib.pyplot as plt

    ocean.add_tracer('q', initial_state=1.0)

    @ocean.add_forcing
    def heating(model):
        dstate = np.zeros_like(model.state)
        dstate[2] = np.zeros_like(model.h)
        dstate[2][10:10+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis] * 1e-6
        dstate[2] -= model.h / 1e7
        return dstate

    plt.ion()

    num_levels = 24
    colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

    print(ocean.q.state)

    ts = []
    es = []
    plt.show()
    for i in range(10000):
        ocean.step()
        if i % 10 == 0:
            print('[t={:7.2f} h range [{:.2f}, {:.2f}]'.format(ocean.t/86400, ocean.h.min(), ocean.h.max()))
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

            plt.figure(4)
            plt.clf()
            #plt.plot(ocean.h[:,0])
            #plt.plot(ocean.h[:,64])
            #plt.ylim(-1,1)
            plt.imshow(ocean.q.T, cmap=plt.cm.RdBu)
            plt.colorbar()

            plt.pause(0.01)
            plt.draw()
