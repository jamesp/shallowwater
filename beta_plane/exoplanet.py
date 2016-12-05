from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from shallowwater import PeriodicShallowWater
from plotting import plot_wind_arrows

nx = 257
ny = 129

DAY = 86400
RADIUS = 6371e3

# # Radius of deformation: Rd = sqrt(2 c / beta)
Rd = 2000.0e3  # Fix Rd at 1000km

Lx = 2*np.pi*RADIUS
Ly = Lx//2

beta=2.28e-12
#c = Rd**2 * beta  # Kelvin/gravity wave speed: c = sqrt(phi0)
c = np.sqrt(300)

print('c', c)
phi0 = c**2       # Set phi baseline from deformation radius
delta_phi = phi0*0.1

# cfl = 0.4         # For numerical stability CFL = |u| dt / dx < 1.0
# dx  = Lx / nx
# dt = np.floor(cfl * dx / (c*4))
# print('dt', dt)

dt = 1200

gamma = 2.0e-4
tau_rad  = 5*DAY
tau_fric = 5*DAY

class MatsunoGill(PeriodicShallowWater):
    def __init__(self, nx, ny, Lx, Ly, alpha, beta, phi0,
        tau_fric, tau_rad, dt=dt, nu=5.0e4):
        super(MatsunoGill, self).__init__(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=5.0e4)
        self.alpha = alpha
        self.phi0 = phi0
        self.phi[:] += phi0

    def to_dataset(self):
        dataset = super(MatsunoGill, self).to_dataset()
        dataset['phi_eq'] = xr.DataArray(self.phi_eq().T.copy(), coords=(dataset.y, dataset.x))
        dataset['phi_eq_xi'] = xr.DataArray(self.centre_substellar(self.phi_eq()).T.copy(), coords=(dataset.y, dataset.x))
        dataset['phi_xi'] = xr.DataArray(self.centre_substellar(self.phi).T.copy(), coords=(dataset.y, dataset.x))
        return dataset


    def substellarx(self, t=None):
        if t is None:
            t = self.t
        return np.fmod(t*self.alpha*self.c, self.Lx)

    @property
    def c(self):
        return np.sqrt(self.phi0)

    @property
    def phixi(self):
        subx = self.substellarx()
        sx = self.phix - subx
        sx[sx < -self.Lx/2] = sx[sx < -self.Lx/2] + self.Lx
        sx[sx > self.Lx/2] = sx[sx > self.Lx/2] - self.Lx
        return sx

    def centre_substellar(self, psi):
        subi = np.argmin(self.phixi**2)
        return np.roll(psi, self.nx//2 - subi, axis=0)


    def phi_eq(self):
        return phi0 + delta_phi*np.exp(-((self.phixi)**2 + self.phiy**2) / (Rd**2))

    def rhs(self):
        u, v, phi = self.state

        # phi rhs
        dphi = np.zeros_like(phi)
        du, dv = np.zeros_like(self.u), np.zeros_like(self.v)

        #  Newtonian cooling / Rayleigh Friction
        dphi += (self.phi_eq() - phi)/tau_rad
        du -= u / tau_fric
        dv -= v / tau_fric

        return np.array([du, dv, dphi])


offsets = []
alphas = [-1., -.5, 0., .5, 1.]
for a in alphas:
    atmos = MatsunoGill(nx, ny, Lx, Ly, beta=beta, alpha=a,
        phi0=phi0, tau_fric=tau_fric, tau_rad=tau_rad,
        dt=dt, nu=5.0e4)

    snapshots = []
    while atmos.t < 20*DAY:

        if atmos.tc % 50 == 0:
            print('%.1f\t%.1f' % (atmos.t/DAY, np.max(atmos.u**2)))
            dset = atmos.to_dataset()
            dset.coords['time'] = atmos.t
            snapshots.append(dset)
        atmos.step()

    adata = xr.concat(snapshots, dim='time')

    rphi = atmos.centre_substellar(atmos.phi.sum(axis=1))
    rphieq = atmos.centre_substellar(atmos.phi_eq().sum(axis=1))
    offsets.append(atmos.phix[np.argmax(rphi)]/atmos.Lx)



    # plt.plot(atmos.phix/Lx, rphi, label='equator')
    # plt.plot(atmos.phix/Lx, rphieq, label='phi_eq')
    # plt.show()

for a, o in zip(alphas, offsets):
    print(a, o)


# plt.ion()

# num_levels = 24
# colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

# def forceAspect(ax,aspect=1):
#     im = ax.get_images()
#     extent =  im[0].get_extent()
#     ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

# plt.show()
# fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 6))
# for i in range(2000):
#     atmos.step()

#     if i % 10 == 0:
#         fig.clf()

#         fig.suptitle('State at T=%.2f days' % (atmos.t / 86400.0))

#         plt.subplot(211)
#         x, y = np.meshgrid(atmos.phix/RADIUS, atmos.phiy/RADIUS)
#         rng = np.abs(atmos.phi - phi0).max()
#         if rng > 0:
#             plt.contourf(x, y, atmos.phi.T - phi0, cmap=plt.cm.RdBu, levels=colorlevels*rng)
#             plt.colorbar()
#             plot_wind_arrows(atmos, (x,y), narrows=(25,25), hide_below=0.01)


#         plt.title('Geopotential')

#         plt.subplot(212)
#         plt.plot(atmos.phix/Lx, atmos.phi.sum(axis=1), label='equator')
#         plt.plot(atmos.phix/Lx, atmos.phi_eq().sum(axis=1), label='phi_eq')
#         #plt.plot(atmos.phix/Rd, atmos.phi[:, ny//2+(Ly//Rd//2)], label='tropics')
#         #plt.ylim(phi0*.99, phi0*1.01)
#         #plt.legend(loc='lower right')
#         plt.title('Longitudinal Geopotential')
#         plt.xlabel('x (multiples of Rd)')
#         plt.ylabel('Geopotential')
#         plt.xlim(-.5, .5)
#         plt.pause(0.01)
#         plt.draw()