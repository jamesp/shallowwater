
def adamsbashforthgen(rhs_fn, dt):
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

class AdamsBashforthTimestepping:
    def integrate(self):
        dt = self.dt
        du = self._dynamics()

        if self.tc == 1:
            # first step Euler
            dt1 = dt
            val = dt1*du
            self.__pdu = du

        elif self.tc == 2:
            # AB2 at step 2
            dt1 = 1.5*dt
            dt2 = -0.5*dt
            val = dt1*du + dt2*self.__pdu

        else:
            # AB3 from step 3 on
            dt1 = 23./12.*dt
            dt2 = -16./12.*dt
            dt3 = 5./12.*dt
            val = dt1*du + dt2*self.__pdu + dt3*self.__ppdu

        self.__ppdu, self._pdu = self.__pdu, du
        return val