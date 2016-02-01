import numpy as np
import matplotlib.pyplot as plt

from linear import PeriodicLinearShallowWater
from plotting import colourlevels, plot_wind_arrows

nx = 128
ny = 129

Rd = 1000.0e3  # Fix Rd at 1000km

Lx = 20*Rd
Ly = 20*Rd


nx = 128
ny = 129


Lx = 10000e3        # 10,000 km
Ly = 10000e3        # 10,000 km

f0 = 2.0e-5
beta=0.0#2.28e-11       # f = f0 + beta y
c = 30.0    # Kelvin/gravity wave speed: c = sqrt(g H)
g = 1.0             # Set gravity to 1.0.  It's only the quantity (gH) that matters.
H = c**2             # Average fluid height

cfl = 0.7         # For numerical stability CFL = |u| dt / dx < 1.0
dx  = Ly / nx
#dt = np.floor(cfl * dx / (c*4))  # TODO check this calculation for c-grid
dt = 300.0



class FixedHeightSW(PeriodicLinearShallowWater):
    def rhs(self):
        r = super(PeriodicLinearShallowWater, self).rhs()
        r[2][:] = 0  # Fix the h field to not change in time
        return r

atmos = FixedHeightSW(nx, ny, Lx=Lx, Ly=Ly, g=g, H=H, f0=f0, beta=beta, dt=dt, r=0.0, nu=1e5)

# Create a Gaussian of radius Rd
d = 2* (Ly // Rd)
hump = (np.sin(np.arange(0, np.pi, np.pi/(2*d)))**2)[np.newaxis, :] * (np.sin(np.arange(0, np.pi, np.pi/(2*d)))**2)[:, np.newaxis]

atmos.h[nx//2-2*d:nx//2, ny//2:ny//2+2*d] += hump*H*0.01
atmos.h[nx//2:nx//2+2*d, ny//2-2*d:ny//2] -= hump*H*0.01


plt.ion()

plt.show()
for i in range(100000):
    atmos.step()

    if i % 40 == 0:

        plt.figure(1, figsize=(8, 8))
        plt.clf()

        plt.suptitle('State at T=%.2f days' % (atmos.t / 86400.0))
        x, y = np.meshgrid(atmos.phix/Rd, atmos.phiy/Rd)
        plt.contourf(x, y, atmos.phi.T, cmap=plt.cm.RdBu_r, levels=colourlevels(24)*H*0.01)
        plot_wind_arrows(atmos, (x,y), narrows=(25,25), hide_below=0.01)
        #plt.contourf(atmos.u.T)

        plt.pause(0.001)
        plt.draw()
