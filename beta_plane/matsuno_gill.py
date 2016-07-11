from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from shallowwater import PeriodicShallowWater
from plotting import plot_wind_arrows

nx = 128
ny = 129


# Radius of deformation: Rd = sqrt(2 c / beta)
Rd = 1000.0e3  # Fix Rd at 1000km

Lx = 20*Rd
Ly = 20*Rd

beta=2.28e-11
c = Rd**2 * beta  # Kelvin/gravity wave speed: c = sqrt(phi0)

print('c', c)
phi0 = c**2       # Set phi baseline from deformation radius

cfl = 0.7         # For numerical stability CFL = |u| dt / dx < 1.0
dx  = Ly / nx
dt = np.floor(cfl * dx / (c*4))  # TODO check this calculation for c-grid
print('dt', dt)

gamma = 2.0e-4
tau = dt*15.0


class MatsunoGill(PeriodicShallowWater):
    def rhs(self):
        phi = self.phi

        # phi rhs
        dphi = np.zeros_like(phi)

        #  Fixed heating on equator
        dphi[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = -hump*gamma
        #  Newtonian relaxation
        dphi -= (phi - phi0)/tau

        return np.array([[0], [0], dphi])

# Add a lump of fluid with scale 2 Rd
d = (Ly // Rd)
hump = (np.sin(np.arange(0, np.pi, np.pi/(2*d)))**2)[np.newaxis, :] * (np.sin(np.arange(0, np.pi, np.pi/(2*d)))**2)[:, np.newaxis]

atmos = MatsunoGill(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=5.0e4)
atmos.phi[:] += phi0

plt.ion()

num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

plt.show()
for i in range(2000):


    atmos.step()

    if i % 10 == 0:

        plt.figure(1, figsize=(8, 12))
        plt.clf()

        plt.suptitle('State at T=%.2f days' % (atmos.t / 86400.0))
        plt.subplot(211)
        x, y = np.meshgrid(atmos.phix/Rd, atmos.phiy/Rd)
        rng = np.abs(atmos.phi - phi0).max()
        plt.contourf(x, y, atmos.phi.T - phi0, cmap=plt.cm.RdBu, levels=colorlevels*rng)
        plot_wind_arrows(atmos, (x,y), narrows=(25,25), hide_below=0.01)



        #plt.xlim(-0.5, 0.5)
        # # Kelvin wavespeed tracer
        # kx = ((atmos.t*np.sqrt(phi0)/Lx % 1) - .5)
        # plt.scatter([kx], [0.4], label='sqrt(phi) tracer')
        # Heating souce location
        c = plt.Circle((0,0), 0.5, fill=False)
        plt.gca().add_artist(c)
        plt.text(0, 0.7, 'Heating')
        plt.xlabel('x (multiples of Rd)')
        plt.ylabel('y (multiples of Rd)')
        plt.xlim(-Lx/Rd/2, Lx/Rd/2)
        plt.ylim(-Ly/Rd/2, Ly/Rd/2)
        plt.title('Geopotential')

        plt.subplot(212)
        plt.plot(atmos.phix/Rd, atmos.phi[:, ny//2], label='equator')
        plt.plot(atmos.phix/Rd, atmos.phi[:, ny//2+(Ly//Rd//2)], label='tropics')
        plt.ylim(phi0*.99, phi0*1.01)
        plt.legend(loc='lower right')
        plt.title('Longitudinal Geopotential')
        plt.xlabel('x (multiples of Rd)')
        plt.ylabel('Geopotential')
        plt.xlim(-Lx/Rd/2, Lx/Rd/2)
        plt.pause(0.01)
        plt.draw()

# plt.figure(figsize=(12, 12))
# plt.title('Geopotential disturbance at T=%.2f days' % (atmos.t / 86400.0))
# x, y = np.meshgrid(atmos.phix/Rd, atmos.phiy/Rd)
# rng = np.abs(atmos.phi - phi0).max()
# plt.contourf(x, y, atmos.phi.T - phi0, cmap=plt.cm.RdBu, levels=colorlevels*rng)
# plot_wind_arrows(atmos, (x,y), narrows=(25,25), hide_below=0.01)
# c = plt.Circle((0,0), 0.5, fill=False)
# plt.gca().add_artist(c)
# plt.text(0, 0.7, 'Heating')
# plt.xlabel('x (multiples of Rd)')
# plt.ylabel('y (multiples of Rd)')
# plt.xlim(-Lx/Rd/2, Lx/Rd/2)
# plt.ylim(-Ly/Rd/2, Ly/Rd/2)
# plt.savefig('gill_pattern.pdf')