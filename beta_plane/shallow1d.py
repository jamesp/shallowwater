
import numpy as np

from arakawac import Arakawa1D
from timesteppers import AdamsBashforth3


class ShallowWater1D(Arakawa1D, AdamsBashforth3, Forceable):
    """The Shallow Water Equations on the Arakawa-C grid."""
    def __init__(self, nx, Lx=1.0e7, nu=1.0e3, nu_phi=None, dt=1000.0):
        super(ShallowWater1D, self).__init__(nx, Lx)

        # dissipation and friction
        self.nu = nu                                    # u, v dissipation
        self.nu_phi = nu if nu_phi is None else nu_phi  # phi dissipation

        # timestepping
        self.dt = dt
        self.tc = 0  # number of timesteps taken
        self.t = 0.0

        #self._stepper = adamsbashforthgen(self._rhs, self.dt)

        self._tracers  = {}



    def rhs(self):
        """Set a right-hand side term for the equation.
        Default is [0,0], override this method when subclassing."""
        zeros = np.zeros_like(self.state)
        return zeros

    def _dynamics_terms(self):
        """Calculate the dynamics for the u, v and phi equations."""
        # ~~~ Nonlinear Dynamics ~~~
        ubarx = self.x_average(self._u)

        # the height equation
        phi_at_u = self.x_average(self._phi)  # (nx+1)
        phi_rhs  = - self.diffx(phi_at_u * self.u)       # (nx) nonlinear
        #phi_rhs  = - self.diffx(np.mean(phi_at_u) * self.u)       # (nx) linear
        phi_rhs += self.nu_phi*self.diff2x(self._phi)    # (nx) diffusion

        # the u equation
        dhdx = self.diffx(self._phi)         # (nx+2)
        ududx = 0.5*self.diffx(ubarx**2)     # u*du/dx at u points

        u_rhs  = -dhdx
        u_rhs += self.nu*self.diff2x(self._u)
        #u_rhs += - ududx                     # nonlin u advection terms

        dstate = np.array([u_rhs, phi_rhs])
        return dstate

    def step(self):
        dt, tc = self.dt, self.tc

        # apply boundary conditions to all fields
        self._apply_boundary_conditions()
        for tracer in self.tracers.values():
            tracer.apply_boundary_conditions()

        newstate = self.state + self.dstate()
        # calculate all tracer dstates before updating any of them
        dstates = [t.dstate() for t in self.tracers.values()]
        # now update them all
        for tracer, dstate in zip(self.tracers.values(), dstates):
            tracer.state = tracer.state + dstate
            tracer._incr_timestep()

        self.state = newstate
        self._incr_timestep()

    def add_tracer(self, name, initial_state=0.0, rhs=0, kappa=0.0, damping=1.0):
        """Add a tracer to the shallow water model.

        Dq/Dt + q(∇ . u) = k∆q + rhs

        Tracers are advected by the flow.  `rhs` can be a constant
        or a function that takes the shallow water object as a single argument.

        `kappa` is a coefficient of dissipation.

        Once a tracer has been added to the model it's value can be accessed
        by the `tracer(name)` method.
        """
        t = ShallowWaterTracer(name, grid=self, kappa=kappa,
                            initial_state=initial_state, damping=damping)
        self.tracers[name] = t
        return t

    def tracer(self, name):
        return self.tracers[name]

    # allow tracers to be called as properties of the object
    def __getattr__(self, name):
        if name in self.tracers:
            return self.tracer(name)

class LinearShallowWater1D(ShallowWater1D):
    def __init__(self, nx, Lx=1.0e7, H=100., nu=1.0e3, nu_phi=None, dt=1000.0):
        super(LinearShallowWater1D, self).__init__(nx, Lx=Lx, nu=nu, nu_phi=nu_phi, dt=dt)
        self.H = H

    def _dynamics_terms(self):
        """Calculate the dynamics for the u, v and phi equations."""
        # ~~~ Linear Dynamics ~~~
        ubarx = self.x_average(self._u)

        # the height equation
        phi_rhs  = - self.H*self.diffx(self.u)       # (nx) linear
        phi_rhs += self.nu_phi*self.diff2x(self._phi)    # (nx) diffusion

        # the u equation
        dhdx = self.diffx(self._phi)         # (nx+2)
        u_rhs  = -dhdx
        u_rhs += self.nu*self.diff2x(self._u)

        dstate = np.array([u_rhs, phi_rhs])
        return dstate


class ShallowWaterTracer1D(AdamsBashforth3, Forceable):
    def __init__(self, name, grid, kappa=0.0, initial_state=0.0, damping=0.0):
        self.name = name
        self.grid = grid

        self._state = np.zeros_like(grid._phi)  # store tracer on cell centres
        self.state = initial_state

        self.kappa = kappa # diffusion
        self.damping = damping

        self.dt = grid.dt

        self.forcings = []

    @property
    def state(self):
        # view without boundary conditions
        return self._state[1:-1]

    @state.setter
    def state(self, value):
        self._state[1:-1] = value

    def _advection(self):
        """Calculates the conservation of the advected tracer by the fluid flow.

        ∂[q]/∂t + ∇ . (uq) = 0

        Returns the divergence term i.e. ∇.(uq)
        """
        grid = self.grid
        q = self._state

        q_at_u = grid.x_average(q)[1:-1]  # (nx+1, ny)

        return grid.diffx(q_at_u * grid.u) # (nx, ny)

    def _diffusion(self):
        return self.kappa*self.grid.del2(self._state) + self.damping*self.grid.damping(self.state)

    def _rhs(self):
        forcings = np.zeros_like(self.state)
        for f in self.forcings:
            forcings += f(self)
        return self._diffusion() - self._advection() + self.rhs() + forcings

    def rhs(self):
        """Set a right-hand side term for the equation.
        Default is 0.0, override this method when subclassing."""
        return 0.0

    def add_forcing(self, fn):
        """Add a forcing term to the tracer.  Typically used as a decorator,
        see the ShallowWater class for an example.

        Forcing functions should take a single argument for the tracer object itself,
        and return a state delta the same shape as state.
        """
        self.forcings.append(fn)
        return fn

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    H = 2.
    s = 1.
    nx = 128
    Lx = 2*np.pi
    xi_ref_frame = False

    sw = LinearShallowWater1D(nx, Lx, H=H, nu=3e-3, dt=.01)

    # #### forcings ####
    def exoplanet_diurnal_cycle(t):
        phi_eq = np.exp(1j*(sw.phix - s*t))
        phi_eq[phi_eq < 0] = 0.0
        return phi_eq*0.1*H

    def diurnal_forcing(sw):
        t_rad = 10.0
        t_fric = 10.0

        # rayleigh friction
        du = np.zeros_like(sw.u)
        du = -sw.u/t_fric

        # newtonian cooling
        dphi = np.zeros_like(sw.phi)
        ss = exoplanet_diurnal_cycle(sw.t)
        dphi[:] = (ss - sw.phi) / t_rad
        return np.array([du, dphi])


    # #### initial state ####
    # sw.phi[:] = 0  # this is the default
    sw.phi[:] += np.exp(-((1.0-sw.phix)/.3)**2)  # gaussian blob centred at 1.0
    #sw.add_forcing(diurnal_forcing)

    plt.ion()
    plt.show()
    for i in range(10000):
        sw.step()
        if i % 20 == 0:
            print('[t={:7.2f} h range [{:.2f}, {:.2f}]'.format(sw.t/86400, sw.phi.min(), sw.phi.max()))
            plt.figure(1)
            plt.clf()

            peq = exoplanet_diurnal_cycle(sw.t)

            if xi_ref_frame:
                rollx = np.argmax(peq)
                plt.plot(sw.phix, np.roll(sw.phi, -rollx+nx//2))
                if diurnal_forcing in sw.forcings:
                    plt.plot(sw.phix, np.roll(peq, -rollx+nx//2))
            else:
                # plot in the x reference frame
                plt.plot(sw.phix, sw.phi)
                if diurnal_forcing in sw.forcings:
                    plt.plot(sw.phix, peq)

            plt.ylim(-1, 1)
            plt.xlim(-Lx/2, Lx/2)
            plt.pause(0.01)
            plt.draw()
