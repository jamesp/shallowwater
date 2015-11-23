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
        h_with_bcs = self._add_all_bcs(self.eta, self._hbc())
        
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
        return (self.u[-1,:], self.u[0,:], self.u[:,-1], self.u[:,0])


    def _hbc(self):
        """Returns the h boundaries.\
        Returns tuple (left, right, top, bottom)."""
        # # periodic in x, zero derivative on y
        return (self.eta[-1,:], self.eta[0,:], self.eta[:,-1], self.eta[:,0])


    def _vbc(self):
        """Returns the v boundary values."""
        return (self.v[-1,:], self.v[0,:], self.v[:,-1], self.v[:,0])  



import matplotlib.pyplot as plt
import scipy.signal

def background(spectra, fsteps=10, ksteps=10):
    """Uses a 1-2-1 filter to generate 'red noise' background field for a spectra (as per WK1998)
        `fsteps` is the number of times to apply the filter in the frequency direction
        `ksteps` is the number of times to apply the filter in the wavenumber direction
    
    Returns a background field of same dimensions as `spectra`.
    """
    # create a 1D 1-2-1 averaging footprint
    bgf = spectra
    for i in range(fsteps):
        # repeated application of the 1-2-1 blur filter to the spectra
        footprint = np.array([[0,1,0], [0,2,0], [0,1,0]]) / 4.0
        bgf = scipy.signal.convolve2d(bgf, footprint, mode='same', boundary='wrap')
    for i in range(ksteps):
        # repeated application of the 1-2-1 blur filter to the spectra
        footprint = np.array([[0,0,0], [1,2,1], [0,0,0]]) / 4.0
        bgf = scipy.signal.convolve2d(bgf, footprint, mode='same', boundary='wrap')
    
    return bgf

def remove_background(spectra):
    """A simple background removal to eliminate frequency noise."""
    bg = background(spectra, fsteps=10, ksteps=0)
    return spectra - bg

plt.ion()

sw = NonlinearShallowWater(128, 128, dt=0.01, maxt=1000, f0=1.7, beta=0.0, domain=(1000, 1000), H=100.0)

sw.eta[:] = np.random.randn(*sw.h.shape)*sw.H*0.03
#sw.eta[40:80, 40:80] = np.exp(-(np.arange(-20, 20)[np.newaxis, :]**2 + np.arange(-20, 20)[:, np.newaxis]**2)/100)*3
#sw.eta[40:100, 40:100] = np.sin(np.pi*np.arange(60)/60)[:, np.newaxis]  * np.sin(np.pi*np.arange(60)/60)[np.newaxis, :] * sw.H*0.1
#sw.eta[30:60, 30:60] = 10.0
#sw.eta[20:30, 20:30] = 10.0
#sw.u[:] = 10  # add a background velocity


# @sw.forcing
# def dissipate(m, r=0.01):
#     dstate = np.zeros_like(m.state)
#     dstate[0] = -m.u*r
#     dstate[1] = -m.v*r
#     return dstate

# damping_factor = np.zeros(sw.ny)
# damping_factor[:10] = np.exp(-np.linspace(0, 5, 10))
# damping_factor[-10:] = np.exp(-np.linspace(5, 0, 10))
# damping_factor = damping_factor[np.newaxis, :]
# @sw.forcing
# def relax(m):
#     # damp surface perturbations at the top and bottom of the domain (i.e. prevent wave reflection off boundary)
#     dstate = np.zeros_like(m.state)
#     deta = (m.h - m.H)*damping_factor*0.1
#     dstate[2] = deta
#     return dstate

@sw.diagnostic('PE')
def pot_energy(m):
    return m.g*m.h

@sw.diagnostic('KE')
def kinetic_energy(m):
    uu, vv = m.uvath()
    return 0.5*(uu**2 + vv**2) 

ts = []
us = []
es = []
usl = []
uspec = []
@sw.on('step:end')
def plot_surface(m):
    ts.append(m.t)
    es.append(np.sum(m.calc_diagnostic('KE') + m.calc_diagnostic('PE')))
    us.append(m.h.mean())
    usl.append(m.u.mean(axis=1))
    
    if m.tc % 100 == 0:
        print(m.t)
    if m.tc % 20 == 1:
        uspec.append(m.u.mean(axis=1))    
        uspectra = np.fft.fftshift(np.fft.fft2(np.array(uspec)))
        
        
        plt.figure(1, figsize=(8,8))
        plt.clf()
        plt.imshow(m.eta.T, cmap=plt.cm.seismic)
        plt.clim(-3, 3)
        plt.colorbar()

        plt.figure(2, figsize=(8,8))
        plt.clf()
        #plt.subplot(221)
        spec_mid=uspectra.shape[0]/2
        spec = remove_background(uspectra)
        plt.imshow(np.log(np.abs(spec)**2)[spec_mid:spec_mid+128, :][::-1])
        plt.colorbar()

        plt.figure(3)
        plt.clf()
        plt.plot(m.h.mean(axis=(1)))


        plt.figure(4)
        plt.clf()
        plt.plot(ts, es)
        plt.xlabel('time')
        plt.ylabel('energy')


        plt.figure(5)
        plt.clf()
        plt.plot(ts, us)
        # plt.subplot(313)
        # plt.imshow(np.array(usl).T)

        # plt.subplot(312)
        # plt.plot(ts, us)
        # #plt.ylim(-5, 5)
        plt.pause(0.01)
        plt.draw()


@sw.diagnostic('spec_u')
def u_specta(m):
    spectra = np.fft.fft(m.u.mean(axis=1))
    return spectra

plot_surface(sw)
sw.run()