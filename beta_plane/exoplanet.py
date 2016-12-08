import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

from shallowwater import PeriodicShallowWater
from plotting import plot_wind_arrows

nx = 257
ny = 129
nd = 20     # number of days to run

DAY = 86400
RADIUS = 6371e3

# # Radius of deformation: Rd = sqrt(2 c / beta)
Rd = 2000.0e3  # Fix Rd at 1000km

Lx = 2*np.pi*RADIUS
Ly = Lx//2

beta0=3e-14
# Kelvin/gravity wave speed: c = sqrt(phi0)
phi0 = float(sys.argv[1])
c = np.sqrt(phi0)
delta_phi = phi0*0.1

print('c', c)

# cfl = 0.4         # For numerical stability CFL = |u| dt / dx < 1.0
# dx  = Lx / nx
# dt = np.floor(cfl * dx / (c*4))
# print('dt', dt)

dt = 400

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



alphas = [-2., -1., -.75,  -.5, -.25, -.1,  0., .1,  .25,  .5, .75, 1., 2.]
betas = [1, 3, 10, 30, 100, 300]

odata = []
for b in tqdm(betas):
    beta = b*beta0
    bdata = []
    for a in tqdm(alphas):
        atmos = MatsunoGill(nx, ny, Lx, Ly, beta=beta, alpha=a,
            phi0=phi0, tau_fric=tau_fric, tau_rad=tau_rad,
            dt=dt, nu=5.0e4)

        snapshots = []
        #print('alpha: %.2f' % a)
        for i in tqdm(range(int(nd*DAY/dt))):
            if atmos.t % 86400 == 0:
                #print('%.1f\t%.2f' % (atmos.t/DAY, np.max(atmos.u**2)))
                dset = atmos.to_dataset()
                dset.coords['time'] = atmos.t
                snapshots.append(dset)
            atmos.step()

        adata = xr.concat(snapshots, dim='time')
        adata.coords['alpha'] = a
        bdata.append(adata)

    data = xr.concat(bdata, dim='alpha')
    data.coords['beta'] = b
    odata.append(data)

data = xr.concat(odata, dim='beta')
data.to_netcdf('beta_data_h%.0f.nc' % (phi0))