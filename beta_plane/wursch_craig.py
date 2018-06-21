#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The one-dimensional shallow water Equations with convection simulation."""

import numpy as np
import matplotlib.pyplot as plt

from shallow1d import ShallowWater1D

### Configuration
SHOW_ANIMATION = True
LOG_VARS = False
N_STEPS_LOG_OUTPUT = 60
N_STEPS_CHART_REFRESH = 60
RANDOM_SEED = 1234


### Model Parameters
X = 500000.0
T = 100000
dx = 500.0
dt = 1.0

H0 = 90.0
Hc = 90.02
Hr = 90.4

g = 10.0
phi_c = 89.977 * g

beta = 1/150.0
alpha = 2.5e-4
c2 = H0*g  # c^2

ubar = 0.005
l = 2000.0

F_rate = 1.6e-6

Ku = 30000.0 / 4.0
Kh = 30000.0 / 4.0
Kr = 200.0 / 4.0

def seconds_to_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def log(msg):
    print(msg)


### Global Variables
rdm = np.random.RandomState(RANDOM_SEED)
x = np.arange(0, X, dx, dtype=np.float32)
N = len(x)


### Würsch-Craig Equations
def phi(H, h, Hc=Hc):
    """Modified geopotential.

    When H + h > Hc the geopotential is set to an
    artificially reduced value."""
    z = H + h
    return np.where(z > Hc, phi_c + g*H, g*z)

def Fn(x, xn, ubar=ubar, l=l):
    """Perturbation function.
    Added to the velocity field at random locations."""
    #p = dfdx(np.exp(-(x-x[xn])**2 / l**2))
    sig = l/dx
    p = S(1.0/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-(x[xn]-dx/2))/l)**2), -1, 0, 1) / (2.0*dx)
    p_norm = p / np.max(p)  # normalise the disturbance ~ 1.0
    return ubar*p_norm

def perturb(x, rate, dt, X=X, ubar=ubar, l=l):
    """Create perturbations at given rate over the domain x."""
    p = np.zeros_like(x)
    for _ in range(rdm.poisson(rate*X*dt, 1)):
        p = p + Fn(x, rdm.randint(0, N), ubar, l)
    return p



sw = LinearShallowWater1D(nx, Lx, H=H, nu=3e-3, dt=.01)

plt.ion()
fig, ax = plt.subplots(3)
plt.show()

for i,q in enumerate(['fluid velocity [$m.s^{-1}$]', 'fluid height [m]', 'rain (mass content) [$10^2$]']):
    ax[i].set_xlabel('x')
    ax[i].set_ylabel(q)

uline, = ax[0].plot(sw.ux, sw.u)
hline, = ax[1].plot(sw.phix, sw.h)
rline, = ax[2].plot(sw.ux, sw.r)



for i in range(10000):
    sw.step()
    if i % 20 == 0:
        print('[t={:7.2f} h range [{:.2f}, {:.2f}]'.format(sw.t/86400, sw.phi.min(), sw.phi.max()))
        plt.figure(1)
        plt.clf()

        peq = exoplanet_diurnal_cycle(sw.t)

        if xi_ref_frame:
            rollx = np.argmax(peq)
            plt.plot(sw.phix, np.roll(sw.phi, -rollx+nx//2))
            if diurnal_forcing in sw.forcings:
                plt.plot(sw.phix, np.roll(peq, -rollx+nx//2))
        else:
            # plot in the x reference frame
            plt.plot(sw.phix, sw.phi)
            if diurnal_forcing in sw.forcings:
                plt.plot(sw.phix, peq)

        plt.ylim(-1, 1)
        plt.xlim(-Lx/2, Lx/2)
        plt.pause(0.01)
        plt.draw()
