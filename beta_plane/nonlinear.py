#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shallow Water Model

- Two dimensional shallow water in a rotating frame
- Staggered Arakawa-C grid
- periodic or bounded in the x-dimension
- fixed boundary conditions in the y-dimension (free slip)

D/Dt[u] - fv = - ∂/∂x[φ]
D/Dt[v] + fu = - ∂/∂y[φ]
D/Dt[φ] + φ (∂/∂x[u] + ∂/∂y[v]) = 0

f = f0 + βy
φ = gh
"""

import numpy as np

from arakawac import ArakawaCGrid, PeriodicBoundaries, WallBoundaries
from timesteppers import adamsbashforthgen


class NonLinShallowWater(ArakawaCGrid):
    def __init__(self, nx, ny, Lx=1.0e7, Ly=1.0e7, f0=0, beta=2.0e-11, nu=1.0e-5, r=1.0e-5, dt=1000.0):
        super(NonLinShallowWater, self).__init__(nx, ny, Lx, Ly)

        # Coriolis terms
        self.f0 = f0
        self.beta = beta

        # dissipation and friction
        self.nu = nu
        self.r = r
        self.sponge_ny = ny//7
        self.sponge = np.exp(-np.linspace(0, 5, self.sponge_ny))

        # timestepping
        self.dt = dt
        self.tc = 0  # number of timesteps taken
        self.t = 0.0

        self._stepper = adamsbashforthgen(self._dynamics, self.dt)
        self._tracers = {}

    def damping(self, var):
        # sponges are active at the top and bottom of the domain by applying Rayleigh friction
        # with exponential decay towards the centre of the domain
        var_sponge = np.zeros_like(var)
        var_sponge[:, :self.sponge_ny] = self.sponge[np.newaxis, :]
        var_sponge[:, -self.sponge_ny:] = self.sponge[::-1][np.newaxis, :]
        return self.r*var_sponge*var


    def dynamics(self):
        """Calculate the dynamics for the u, v and phi equations."""

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


    def rhs(self):
        """Apply a right hand side term to the u, v and h equations.
        By default this is zero, can be overridden in subclasses."""
        return 0.0

    def _dynamics(self):
        return self.dynamics() +  self.rhs()


    def add_tracer(self, name, initial_state, rhs=0, kappa=0.0, apply_damping=True):
        """Add a tracer to the shallow water model.

        Dq/Dt + q(∇ . u) = k∆q + rhs

        Tracers are advected by the flow.  `rhs` can be a constant
        or a function that takes the shallow water object as a single argument.

        `kappa` is a coefficient of dissipation.

        Once a tracer has been added to the model it's value can be accessed
        by the `tracer(name)` method.
        """

        state = np.zeros_like(self._phi)  # tracer values held at cell centres
        state[1:-1, 1:-1] = initial_state

        def _rhs():
            orhs = -self._tracer_dynamics_terms(name)
            if kappa:
                orhs += kappa*self.del2(state)
            if apply_damping:
                orhs += -self.damping(state[1:-1, 1:-1])
            if callable(rhs):
                orhs += rhs(self)
            else:
                orhs += rhs
            return orhs

        stepper = adamsbashforthgen(_rhs, self.dt)
        self._tracers[name] = (state, stepper)

    def tracer(self, name):
        return self._tracers[name][0][1:-1, 1:-1]

    def _tracer_dynamics_terms(self, name):
        """Calculates the conservation of an advected tracer.

        ∂[q]/∂t + ∇ . (uq) = 0

        Returns the divergence term i.e. ∇.(uq)
        """
        q = self._tracers[name][0]

        # the height equation
        q_at_u = self.x_average(q)[:, 1:-1]  # (nx+1, ny)
        q_at_v = self.y_average(q)[1:-1, :]  # (nx, ny+1)

        return self.diffx(q_at_u * self.u) - self.diffy(q_at_v * self.v)  # (nx, ny)

    def step(self):
        dt, tc = self.dt, self.tc

        self._apply_boundary_conditions()
        for (field, stepper) in self._tracers.values():
            self._apply_boundary_conditions_to(field)
            field[1:-1, 1:-1] = field[1:-1, 1:-1] + next(stepper)

        newstate = self.state + next(self._stepper)

        self.state = newstate

        self.t  += dt
        self.tc += 1

class PeriodicShallowWater(PeriodicBoundaries, NonLinShallowWater): pass
class WalledShallowWater(WallBoundaries, NonLinShallowWater): pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from spectral_analysis import background, kiladis_spectra

    nx = 128
    ny = 129
    beta=2.0e-11
    Lx = 1.0e7
    Ly = 1.0e7

    dt = 3000.0
    phi0 = 10.0

    class ShallowWater(PeriodicShallowWater):
        def rhs(self):
            dstate = np.zeros_like(self.state)
            q = self.tracer('q')
            gamma = 1e-6
            #dstate[2] = gamma*q
            return dstate


    ocean = ShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=1.0e3)

    d = 25
    hump = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]

    ocean.phi[:] += phi0
    ocean.phi[70-d:70+d, ny//2-d:ny//2+d] += hump*0.1
    #ocean.phi[:] -= hump.sum()/(ocean.nx*ocean.ny)

    initial_phi = ocean.phi.copy()

    q = np.zeros_like(ocean.phi)
    q[nx//2-d:nx//2+d, ny//2-d:ny//2+d] += hump
    q0 = q.sum()

    def q_rhs(model):
        q = model.tracer('q').copy()
        minq = np.zeros_like(q)
        minq[q < 0] = -q[q<0]
        return - (model.phi - phi0)*1e-6# + minq*0.1

    ocean.add_tracer('q', q, q_rhs)

    def force_geopot(model):
        dstate = np.zeros_like(model.state)
        q = model.tracer('q')
        gamma = 1e-6
        #dstate[2] = gamma*q
        return dstate

    plt.ion()

    num_levels = 24
    colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

    en = []
    qn = []

    eq_reg = []
    ts = []

    plt.show()
    for i in range(100000):
        ocean.step()

        if i % 10 == 0:
            eq = ocean.u.copy()[:, ny//2-5:ny//2+5]
            eq_reg.append(eq)
            ts.append(ocean.t)

            eq_reg = eq_reg[-1000:]
            ts = ts[-1000:]

        if i % 100 == 0:

            plt.figure(1, figsize=(16,12))
            plt.clf()

            plt.subplot(231)
            x, y = np.meshgrid(ocean.phix/ocean.Lx, ocean.phiy/ocean.Ly)
            plt.contourf(x, y, ocean.phi.T, cmap=plt.cm.RdBu, levels=phi0+colorlevels*phi0*0.01)
            plt.xlim(-0.5, 0.5)
            plt.title('Geopotential')

            plt.subplot(232)
            en.append(np.sum(ocean.phi - initial_phi))
            qn.append(ocean.tracer('q').sum() - q0)
            plt.plot(en)
            #plt.plot(qn)
            plt.title('Geopotential Loss')

            plt.subplot(233)
            if len(ts) > 50:
                specs = kiladis_spectra(eq_reg)
                spec = np.sum(specs, axis=0)
                nw, nk = spec.shape
                fspec = np.fft.fftshift(spec)
                fspec -= background(fspec, 10, 0)
                om = np.fft.fftshift(np.fft.fftfreq(nw, ts[1]-ts[0]))
                k = np.fft.fftshift(np.fft.fftfreq(nk, 1.0/nk))
                plt.pcolormesh(k, om, np.log(1+np.abs(fspec)), cmap=plt.cm.bone)
                plt.xlim(-15, 15)
                plt.ylim(-0.00002, 0.00002)

            plt.subplot(234)
            plt.plot(ocean.phix/ocean.Lx, ocean.phi[:, ny//2])
            plt.plot(ocean.phix/ocean.Lx, ocean.phi[:, ny//2+8])
            plt.xlim(-0.5, 0.5)
            plt.ylim(phi0*.99, phi0*1.01)
            plt.title('Equatorial Height')


            plt.subplot(235)
            plt.contourf(x, y, ocean.tracer('q').T, cmap=plt.cm.RdBu, levels=colorlevels)
            c = plt.Circle((0,0), float(d)/nx/2, fill=False)
            plt.gca().add_artist(c)
            plt.xlim(-.5, .5)
            plt.ylim(-.5, .5)




            plt.pause(0.01)
            plt.draw()
