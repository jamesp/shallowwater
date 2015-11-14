import numpy as np
from linear2d import LinearShallowWater, centre_average

def y_average(phi):
    """Average adjacent values in the y dimension.
    If phi has shape (nx, ny), returns an array of shape (nx, ny - 1)."""
    return 0.5*(phi[:,:-1] + phi[:,1:])

def x_average(phi):
    """Average adjacent values in the x dimension.
    If phi has shape (nx, ny), returns an array of shape (nx - 1, ny)."""
    return 0.5*(phi[:-1,:] + phi[1:,:])

class NonlinearShallowWater(LinearShallowWater):
    def __init__(self, *args, **kwargs):
        super(NonlinearShallowWater, self).__init__(*args, **kwargs)

    def rhs(self):
        """Calculate the right hand side of the u, v and h equations."""
        # extend the linear dynamics to include nonlinear terms of the advection equation
        linear_rhs = super(NonlinearShallowWater, self).rhs()
        
        u_with_bcs = self._add_all_bcs(self.u, self._ubc())
        v_with_bcs = self._add_all_bcs(self.v, self._vbc())
        h_with_bcs = self._add_all_bcs(self.h, self._hbc())
        
        uu, vv = self.uvatuv()
        ubar = centre_average(u_with_bcs)  # u averaged to v points with bcs
        vbar = centre_average(v_with_bcs)  # v averaged to u points with bcs

        ududx = y_average(0.5*self.diffx(ubar**2))         # u*du/dx at u points
        vdudy = vv*y_average(self.diffy(u_with_bcs[1:-1, :]))
        nonlin_u = - ududx - vdudy # nonlin u terms at u points

        udvdx = uu*x_average(self.diffx(v_with_bcs[:, 1:-1]))
        vdvdy = x_average(0.5*self.diffy(vbar**2))         # v*dv/dy at v points
        nonlin_v = - udvdx - vdvdy 

        udhdx = x_average(self.u*self.diffx(h_with_bcs[:, 1:-1]))
        vdhdy = y_average(self.v*self.diffy(h_with_bcs[1:-1, :]))
        nonlin_h =  - udhdx - vdhdy - self.eta * self.divergence
        nonlinear_rhs = np.array([nonlin_u, nonlin_v, nonlin_h])
        return linear_rhs + nonlinear_rhs

    def _ubc(self):
        """Returns the u velocity boundaries.
        Returns tuple (left, right, top, bottom)."""
        # # periodic in x, zero derivative on y
        # return (self.u[-1,:], self.u[0,:], self.u[:,-1], self.u[:,0])
        # no derivative at the boundaries
        return (self.u[-1,:], self.u[0,:], self.u[:,-1], self.u[:,0])


    def _hbc(self):
        """Returns the h boundaries.\
        Returns tuple (left, right, top, bottom)."""
        # # periodic in x, zero derivative on y
        # return (self.eta[-1,:], self.eta[0,:], self.eta[:,-1], self.eta[:,0])
        # no derivative at the boundaries
        return (self.eta[-1,:], self.eta[0,:], self.eta[:,-1], self.eta[:,0])


    def _vbc(self):
        """Returns the v boundary values."""
        # no derivative at the boundaries
        return (self.v[-1,:], self.v[0,:], self.v[:,-1], self.v[:,0])  



import matplotlib.pyplot as plt
plt.ion()

sw = NonlinearShallowWater(128, 128, dt=0.01, maxt=1000, f=1e-5, domain=(1000, 1000), H=100.0)

sw.eta[:] = np.random.random(sw.h.shape)*sw.H*0.1
#sw.eta[40:80, 40:80] = np.exp(-(np.arange(-20, 20)[np.newaxis, :]**2 + np.arange(-20, 20)[:, np.newaxis]**2)/100)*3
#sw.eta[:, :] = np.sin(2*np.pi*np.arange(128)/128)[:, np.newaxis]  * np.sin(2*np.pi*np.arange(128)/128)[np.newaxis, :] * 0.1
#sw.eta[30:60, 30:60] = 1.0
#sw.u[:] = 0.1


@sw.forcing
def dissipate(m):
    return -m.state*0.0

ts = []
us = []
usl = []
@sw.on('step:end')
def plot_surface(m):
    ts.append(m.t)
    us.append(np.abs(m.u.max()))
    usl.append(m.u.mean(axis=1))
    if m.tc % 100 == 0:
        print(m.t)
    if m.tc % 20 == 1:
        plt.clf()
        plt.subplot(311)
        plt.imshow(m.eta.T)
        #plt.clim(-1, 1)
        plt.colorbar()

        plt.subplot(313)
        plt.imshow(np.array(usl).T)

        plt.subplot(312)
        plt.plot(ts, us)
        #plt.ylim(-5, 5)
        plt.pause(0.01)
        plt.draw()

@sw.diagnostic('spec_u')
def specta(m):
    m.tc
    spectra = np.fft.fftn(m.u, axis=(0,1))

plot_surface(sw)
sw.run()