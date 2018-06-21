import itertools

class Timestepper(object):
    """Calculate the time-tendencies and timestepping of the equation
        dstate/dt = _dstate()
    """
    t = 0.0
    tc = 0

    def step(self):
        self.state[:] = self.state + self.dstate()
        self._incr_timestep()

    def _incr_timestep(self):
        self.t = self.t + self.dt
        self.tc = self.tc + 1

    def dstate(self):
        raise NotImplemented()

class Euler(Timestepper):
    def dstate(self):
        dstate =  self.dt*self._dstate()
        return dstate


class AdamsBashforth3(Timestepper):
    _pfstate, _ppfstate = 0.0, 0.0

    def dstate(self):
        dt = self.dt
        fstate = self._dstate()

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


def sync_step(*timesteppers):
    """Synchronize the stepping of several timesteppers.
    This is important if values of each at a given timestep depend on each
    other, e.g. if a tracer has a feedback onto other tracers or state variables.
    """
    dstates = [obj.dstate() for obj in timesteppers]
    for obj, dstate in zip(timesteppers, dstates):
        obj.state = obj.state + dstate
        obj._incr_timestep()

