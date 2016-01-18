import numpy as np
from linear2d import LinearShallowWater, adamsbashforthgen

class NonLinearShallowWater(LinearShallowWater):
    def __init__(self, *args, **kwargs):
        super(NonLinearShallowWater, self).__init__(*args, **kwargs)
        self._tracers = {}

    def nonlin_rhs(self):
        u_at_v, v_at_u = self.uvatuv()

        ubarx = self.x_average(self._u)[:, 1:-1]    # u averaged to v lons
        ubary = self.y_average(self._u)[1:-1, :]    # u averaged to v lats

        vbary = self.y_average(self._v)[1:-1, :]
        vbarx = self.x_average(self._v)[:, 1:-1]

        ududx = 0.5*self.diffx(ubarx**2)            # u*du/dx at u points
        vdudy = v_at_u*self.diffy(ubary)            # v*du/dy at u points
        nonlin_u = - ududx - vdudy                  # nonlin u terms at u points

        udvdx = u_at_v*self.diffx(vbarx)
        vdvdy = 0.5*self.diffy(vbary**2)            # v*dv/dy at v points
        nonlin_v = - udvdx - vdvdy

        udhdx = self.x_average(self.u*self.diffx(self._h[:, 1:-1]))
        vdhdy = self.y_average(self.v*self.diffy(self._h[1:-1, :]))

        nonlin_h =  - udhdx - vdhdy - self.h * self.divergence()
        nonlinear_rhs = np.array([nonlin_u, nonlin_v, nonlin_h])
        return nonlinear_rhs

    def tracer(self, name):
        return self._tracers[name][0][1:-1, 1:-1]

    def tracer_conservation(self, name):
        q = self._tracers[name][0]

        self._apply_boundary_conditions_to(q)

        # advection
        # u . grad q
        dqdx = self.diffx(q)[:, 1:-1]
        udqdx = self.x_average(self.u * dqdx)

        dqdy = self.diffy(q)[1:-1, :]
        vdqdy = self.y_average(self.v * dqdy)

        # conservation
        # q * div u
        qc = q[1:-1, 1:-1] * self.divergence()

        return udqdx + vdqdy + qc


    def add_tracer(self, name, initial_state, rhs=0):
        """Add a tracer to the shallow water model.

        Dq/Dt + q(div u) = rhs

        Tracers are advected by the flow.  `rhs` can be a constant
        or a function that takes the shallow water object as a single argument.

        Once a tracer has been added to the model it's value can be accessed
        by the `tracer(name)` method.
        """

        state = np.zeros_like(self._h)  # tracer values held at cell centres
        state[1:-1, 1:-1] = initial_state

        if not callable(rhs):
            def _rhs():
                return -self.tracer_conservation(name)
        else:
            def _rhs():
                return rhs(self) - self.tracer_conservation(name)

        stepper = adamsbashforthgen(_rhs, self.dt)
        self._tracers[name] = (state, stepper)

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""
        # extend the linear dynamics to include nonlinear terms of the advection equation
        linear_rhs = super(NonLinearShallowWater, self).rhs()

        nonlinear_rhs = self.nonlin_rhs()
        return linear_rhs #+ nonlinear_rhs

    def step(self):
        dt, tc = self.dt, self.tc

        newstate = self.state + next(self._stepper)

        for name, (tstate, stepper) in self._tracers.items():
            tstate[1:-1, 1:-1] = tstate[1:-1, 1:-1] + next(stepper)
            print(name, np.max(tstate))

        self.state = newstate

        self.t  += dt
        self.tc += 1
