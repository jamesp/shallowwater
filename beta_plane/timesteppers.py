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

class AdamsBashforth3:
    __prev_state = (0.0, 0.0)
    t = 0.0
    tc = 0

    def __init__(self, rhs, dt):
        self.dt = dt
        self.rhs = rhs

    def _step(self):
        dt = self.dt
        dx = self.rhs()
        pdx, ppdx = self.__prev_state

        if self.tc == 0:
            # Euler at step 1
            dt1 = dt
            val = dt1*dx

        elif self.tc == 1:
            # AB2 at step 2
            dt1 = 1.5*dt
            dt2 = -0.5*dt
            val = dt1*dx + dt2*pdx

        else:
            # AB3 from step 3 on
            dt1 = 23./12.*dt
            dt2 = -16./12.*dt
            dt3 = 5./12.*dt
            val = dt1*dx + dt2*pdx + dt3*ppdx

        # update the stored values of previous timesteps
        self.__prev_state = (dx, pdx)
        self.tc = self.tc + 1
        self.t  = self.t + dt

        return val

    def step(self):
        # raise NotImplemented
        pass