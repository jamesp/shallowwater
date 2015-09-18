#!/usr/bin/env python
"""beta plane barotropic vorticity model"""

import numpy as np
from numpy import pi, cos, sin
import pyfftw
from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn

pyfftw.interfaces.cache.enable()

def raw_filter(prev, curr, new, nu=0.01):
    """Robert-Asselin Filter."""
    return curr + nu*(new - 2.0*curr + prev)

def courant_number(psix, psiy, dx, dt):
    """Calculates the Courant Number given the velocity field and step size."""
    maxu = np.max(np.abs(psiy))
    maxv = np.max(np.abs(psix))
    maxvel = maxu + maxv;
    return maxvel*dt/dx;

def ft(phi):
    """Go from physical space to spectral space."""
    return fftshift(fftn(phi, axes=(0,1)))

def ift(psi):
    """Go from spectral space to physical space."""
    return ifftn(fftshift(psi), axes=(0,1))

def enstrophy(zt):
    return 0.5*zt*np.conj(zt)











nx = ny = n = 128

# initialize the vorticity field
# using FFTW array for speed
z = pyfftw.n_byte_align_empty((nx, nx), 16, 'complex128')
z[:] = 0

domain = 1.0


## Physical Domain
dx = domain / nx
dy = domain / ny
dt = 0.4 * 16.0 / nx

x = np.arange(0, domain, dx)
y = np.arange(0, domain, dy)
i, j = np.indices((nx, ny))

## Spectral Domain
dk = 2.0*pi/domain;
k = np.arange(-n/2, n/2)*dk
l = np.arange(-n/2, n/2)*dk

si, sj = np.indices((len(k), len(l)))
kk, ll = np.meshgrid(k, l)
ksq = kk**2 + ll**2
ksq[64, 64] = 1.0   # avoid divide by zero
rksq = 1.0 / ksq

dbdx = 1j*kk;
dbdy = 1j*ll;



### CONSTANTS
# physical
f0 = 1.0e-4     # dimensional
ubar = 0.0      # non-dim
beta = 1.7      # non-dim

# dissipation
tau = 0.5
nu = (((domain)/(np.floor(n/3)*2.0*pi))**4)/tau
del4 = 1.0 / (1.0 + nu*ksq**2*dt)

# Range of indices for anti-aliasing
n1 = np.ceil(n/6)+1;
n2 = n+2 - n1;





### INITIAL CONDITIONS

# single spot of max val 2.0 in lower half of plane
d = n / 4.0
ppxy = np.abs(d - i)**2 + np.abs(d*2 - j)**2
dist = np.sqrt(ppxy)
z[dist < d] = (2.0*cos(0.5 * pi * (d - dist) / d + 0.5*pi)**2)[dist < d]


# transform to spectral space
zt = ft(z)
_zt = zt_ = zt   # set initial previous value

# Poisson equation for streamfunction and vorticity
# $\zeta = \nabla^2 \psi$
# Using the Fourier form
# $\psi = a_k(t) \exp(\imag (k x + l y))$
# => $\zeta = -(k^2 + l^2) \psi
psit = -rksq * zt    # F[\psi] = - F[\zeta] / (k^2 + l^2)
psixt = dbdx * psit  # d/dx F[\psi] = ik F[\psi]
psiyt = dbdy * psit  # d/dy F[\psi] = ik F[\psi]

psix = ift(psixt)    # v = - \psi_x
psiy = ift(psiyt)    # u = \psi_y







# initial enstrophy spectrum
zz0 = zz = enstrophy(zt)
ee0 = zz*rksq




### SETUP


def step(zt, _zt, dt):
    """Take current and previous values of F[zeta] and integrate one timestep."""
    psit = -rksq*zt
    psixt = dbdx*psit
    psiyt = dbdy*psit
    zxt = dbdx*zt
    zyt = dbdy*zt

    # transform back to physical space
    z =    ift(zt)
    psix = ift(psixt)
    psiy = ift(psiyt)
    zx =   ift(zxt)
    zy =   ift(zyt)



## INTEGRATE
def integrate():
    global _zt, zt, zt_, z, dt
    # spec soln of Poisson equation and calc derivatives
    psit = -rksq*zt
    psixt = dbdx*psit
    psiyt = dbdy*psit
    zxt = dbdx*zt
    zyt = dbdy*zt

    # transform back to physical space
    z =    ift(zt)
    psix = ift(psixt)
    psiy = ift(psiyt)
    zx =   ift(zxt)
    zy =   ift(zyt)

    # zz = 0.5*zt*np.conj(zt)
    # ee = zz*rksq

    # check Courant number is within bounds
    c = courant_number(psix, psiy, dx, dt)
    if c > 0.8:
        print('DEBUG: Courant No > 0.8, reducing timestep')
        dt = 0.9*dt
    elif c < 0.6:
        print('DEBUG: Courant No < 0.6, increasing timestep')
        dt = 1.1*dt

    # calculate the Jacobian
    jac = psix * zy - psiy * zx + ubar * zx

    # transform jacobian to spectral space
    jact = ft(jac)

    # avoid aliasing by eliminating short wavelengths
    jact[:n1, :] = 0.0
    jact[n2:, :] = 0.0
    jact[:, :n1] = 0.0
    jact[:, n2:] = 0.0

    # take a timestep
    zt_ = _zt - 2.0*dt*jact - 2.0*dt*beta*psixt
    zt_ = zt_ * del4  # dissipation

    # RAW filter in time
    _zt = raw_filter(_zt, zt, zt_)
    zt = zt_
    return c


## PLOT
import time
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.real(z))
fig.show()
time.sleep(0.2)
t = 0
timeit = time.time()
while True:
    t = t + 1
    if (t % 100) == 0:
        print 'Step: %d' % t
        sps = 100 / (time.time() - timeit)
        print 'Steps per second: %.2f' % sps
        timeit = time.time()
    c = integrate()
    ax.set_title('[Courant No: %3.2f] dt=%4.3f' % (c, dt))
    im.set_data(np.real(z))
    im.axes.figure.canvas.draw()
