def adamsbashforthgen(rhs_fn, dt):
    """Returns an adams-bashforth three-step timestepping generator."""
    dx, pdx, ppdx = 0, 0, 0
    dt1, dt2, dt3 = 0, 0, 0

    # first step Euler
    dt1 = dt
    dx = rhs_fn()
    val = dt1*dx
    pdx = dx
    yield val

    # AB2 at step 2
    dt1 = 1.5*dt
    dt2 = -0.5*dt
    dx = rhs_fn()
    val = dt1*dx + dt2*pdx
    ppdx, pdx = pdx, dx
    yield val

    while True:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt
        dx = rhs_fn()
        val = dt1*dx + dt2*pdx + dt3*ppdx
        ppdx, pdx = pdx, dx
        yield val


class Timestepper(object):
    """Base class for timestepping.
    Numerically progress functions of the form
        dy/dt = f(y,t).
    """

    def __init__(self, f, dt):
        self.f = f
        self.dt = dt
        self.t  = 0.0   # current time
        self.tc = 0     # number of time steps taken

    def calculate_tendency(self, state):
        raise NotImplemented

    def step(self, state):
        """Returns tendency over a timestep."""
        dstate = self.calculate_tendency(state)
        self.t = self.t + self.dt
        self.tc += 1
        return dstate

class Euler(Timestepper):
    def calculate_tendency(self, state):
        dstate =  self.dt*self.f(state, self.t)
        return dstate

class AdamsBashforth3(Timestepper):
    _pfstate, _ppfstate = 0.0, 0.0

    def calculate_tendency(self, state):
        dt = self.dt
        fstate = self.f(state, self.t)

        if self.tc == 0:
            # first step Euler
            dt1 = dt
            dstate = dt1*fstate

        elif self.tc == 1:
            dt1 = 1.5*dt
            dt2 = -0.5*dt
            dstate = dt1*fstate + dt2*self._pfstate

        else:
            dt1 = 23./12.*dt
            dt2 = -16./12.*dt
            dt3 = 5./12.*dt
            dstate = dt1*fstate + dt2*self._pfstate + dt3*self._ppfstate

        # update the cached previous fstate values
        self._ppfstate, self._pfstate = self._pfstate, fstate

        return dstate