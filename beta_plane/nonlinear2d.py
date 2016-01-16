import numpy as np
from linear2d import LinearShallowWater

class NonLinearShallowWater(LinearShallowWater):
    def __init__(self, *args, **kwargs):
        super(NonLinearShallowWater, self).__init__(*args, **kwargs)

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

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""
        # extend the linear dynamics to include nonlinear terms of the advection equation
        linear_rhs = super(NonLinearShallowWater, self).rhs()
        nonlinear_rhs = self.nonlin_rhs()
        return linear_rhs + nonlinear_rhs
