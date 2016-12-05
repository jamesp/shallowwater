from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from shallowwater import PeriodicShallowWater
from plotting import plot_wind_arrows

nx = 256
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
delta_h = phi0*0.1

cfl = 0.4         # For numerical stability CFL = |u| dt / dx < 1.0
dx  = Lx / nx
dt = np.floor(cfl * dx / (c*4))
print('dt', dt)

gamma = 2.0e-4
tau_rad  = 5*DAY
tau_fric = 5*DAY

class MatsunoGill(PeriodicShallowWater):
    alpha = 20

    def h_eq(self):
        subx = np.fmod(self.t*self.alpha, self.Lx)
        sx = self.phix - subx
        sx[sx < -self.Lx/2] = sx[sx < -self.Lx/2] + self.Lx
        sx[sx > self.Lx/2] = sx[sx > self.Lx/2] - self.Lx
        #print(sx)
        return phi0 + delta_h*np.exp(-((sx)**2 + self.phiy**2) / (Rd**2))

    def rhs(self):
        u, v, phi = self.state

        # phi rhs
        dphi = np.zeros_like(phi)
        du, dv = np.zeros_like(self.u), np.zeros_like(self.v)

        #  Newtonian cooling / Rayleigh Friction
        dphi += (self.h_eq() - phi)/tau_rad
        du -= u / tau_fric
        dv -= v / tau_fric

        return np.array([du, dv, dphi])


atmos = MatsunoGill(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=5.0e4)
atmos.phi[:] += phi0

plt.ion()

num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

plt.show()
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 6))
for i in range(2000):
    atmos.step()

    if i % 10 == 0:
        fig.clf()

        fig.suptitle('State at T=%.2f days' % (atmos.t / 86400.0))

        plt.subplot(211)
        x, y = np.meshgrid(atmos.phix/RADIUS, atmos.phiy/RADIUS)
        rng = np.abs(atmos.phi - phi0).max()
        if rng > 0:
            plt.contourf(x, y, atmos.phi.T - phi0, cmap=plt.cm.RdBu, levels=colorlevels*rng)
            plt.colorbar()
            plot_wind_arrows(atmos, (x,y), narrows=(25,25), hide_below=0.01)

        #plt.xlim(-0.5, 0.5)
        # # Kelvin wavespeed tracer
        # kx = ((atmos.t*np.sqrt(phi0)/Lx % 1) - .5)
        # plt.scatter([kx], [0.4], label='sqrt(phi) tracer')
        # Heating souce location
        # c = plt.Circle((0,0), 0.5, fill=False)
        # ax1.add_artist(c)
        # ax1.text(0, 0.7, 'Heating')
        # ax1.set_xlabel('x (multiples of Rd)')
        # ax1.set_ylabel('y (multiples of Rd)')
        # ax1.set_xlim(-.5, .5)
        # ax1.set_ylim(-.5, .5)

        plt.title('Geopotential')

        plt.subplot(212)
        plt.plot(atmos.phix/Lx, atmos.phi.sum(axis=1), label='equator')
        plt.plot(atmos.phix/Lx, atmos.h_eq().sum(axis=1), label='h_eq')
        #plt.plot(atmos.phix/Rd, atmos.phi[:, ny//2+(Ly//Rd//2)], label='tropics')
        #plt.ylim(phi0*.99, phi0*1.01)
        #plt.legend(loc='lower right')
        plt.title('Longitudinal Geopotential')
        plt.xlabel('x (multiples of Rd)')
        plt.ylabel('Geopotential')
        plt.xlim(-.5, .5)
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