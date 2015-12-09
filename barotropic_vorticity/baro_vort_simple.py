#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""beta plane barotropic vorticity model.

This script uses a pseudospectral method to solve the barotropic vorticity
equation in two dimensions

    D/Dt[ω] = 0                                                             (1)

where ω = ξ + f.  ξ is local vorticity ∇ × u and f is global rotation.

Assuming an incompressible two-dimensional flow u = (u, v),
the streamfunction ψ = ∇ × (ψ êz) can be used to give (u,v)

    u = ∂/∂y[ψ]         v = -∂/∂x[ψ]                                        (2)

and therefore local vorticity can be given as a Poisson equation

    ξ = ∆ψ                                                                  (3)

where ∆ is the laplacian operator.  Since ∂/∂t[f] = 0 equation (1) can be
written in terms of the local vorticity

        D/Dt[ξ] + u·∇f = 0
    =>  D/Dt[ξ] = -vβ                                                       (4)

using the beta-plane approximation f = f0 + βy.  This can be written entirely
in terms of the streamfunction and this is the form that will be solved
numerically.

    D/Dt[∆ψ] = -β ∂/∂x[ψ]                                                   (5)

The spectral method defines ψ as a Fourier sum

    ψ = Σ A(t) exp(i (kx + ly))

and as such spatial derivatives can be calculated analytically

    ∂/∂x[ψ] = ikψ       ∂/∂y[ψ] = ilψ

The pseudospectral method will use the analytic derivatives to calculate
values for (u, v) which will then be used to evaluate nonlinear terms.


References:
* This code was developed based on a MATLAB script bvebb.m
  (Original source Dr. James Kent & Prof. John Thuburn)
* And the GFDL documentation for spectral barotropic models
  found here 
  [http://www.gfdl.noaa.gov/cms-filesystem-action/user_files/pjp/barotropic.pdf]
* McWilliams Initial Condition inspired by pyqg [https://github.com/pyqg/pyqg]
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy import pi, cos, sin
from numpy.fft import fftshift, fftfreq
from numpy.fft.fftpack import rfft2, irfft2


### Configuration
nx = 128
ny = 128                        # numerical resolution
Lx = 1.0
Ly = 1.0                        # domain size [m]
ubar = 0.00                     # background zonal velocity  [m/s]
beta = 5.0                      # beta-plane f = f0 + βy     [1/s 1/m]
tau = 0.25                       # coefficient of dissipation
                                # smaller = more dissipation
r_rayleigh = 1./200000.
forcing_amp_factor=0.0001

# r_rayleigh = 1./2000.
# forcing_amp_factor=0.00001

t = 0.0
tmax = 10000
step = 0

ALLOW_SPEEDUP = True         # if True, allow the simulation to take a larger
SPEEDUP_AT_C  = 0.6          # timestep when the Courant number drops below
                             # value of parameter SPEEDUP_AT_C
SLOWDN_AT_C = 0.8            # reduce the timestep when Courant number
                             # is bigger than SLOWDN_AT_C
PLOT_EVERY_S = 10


### Function Definitions
def ft(phi):
    """Go from physical space to spectral space."""
    return rfft2(phi, axes=(-2, -1))

def ift(psi):
    """Go from spectral space to physical space."""
    return irfft2(psi, axes=(-2,-1))

def courant_number(psix, psiy, dx, dt):
    """Calculate the Courant Number given the velocity field and step size."""
    maxu = np.max(np.abs(psiy))
    maxv = np.max(np.abs(psix))
    maxvel = maxu + maxv
    return maxvel*dt/dx

def grad(phit):
    """Returns the spatial derivatives of a Fourier transformed variable.
    Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
    global ik, il
    phixt = ik*phit        # d/dx F[φ] = ik F[φ]
    phiyt = il*phit        # d/dy F[φ] = il F[φ]
    return (phixt, phiyt)

def velocity(psit):
    """Returns the velocity field (u, v) from F[ψ]."""
    psixt, psiyt = grad(psit)
    psix = ift(psixt)    # v = - ∂/∂x[ψ]
    psiy = ift(psiyt)    # u =   ∂/∂y[ψ]
    return (psiy, -psix)

def spectral_variance(phit):
    global nx, ny
    var_density = 2.0 * np.abs(phit)**2 / (nx*ny)
    var_density[:,0] /= 2
    var_density[:,-1] /= 2
    return var_density.sum()

_prhs, _pprhs  = 0.0, 0.0  # previous two right hand sides
def adams_bashforth(zt, rhs, dt):
    """Take a single step forward in time using Adams-Bashforth 3."""
    global step, t, _prhs, _pprhs
    if step is 0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif step is 1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt

    newzt = zt + dt1*rhs + dt2*_prhs + dt3*_pprhs  
    _pprhs = _prhs
    _prhs  = rhs
    return newzt

## SETUP

### Physical Domain
nl = ny
nk = nx/2 + 1
dx = Lx / nx
dy = Ly / ny
dt = 0.4 * 16.0 / nx          # choose an initial dt. This will change
                              # as the simulation progresses to maintain
                              # numerical stability

### Spectral Domain
dk = 2.0*pi/Lx
dl = 2.0*pi/Ly
# calculate the wavenumbers [1/m]
# The real FT has half the number of wavenumbers in one direction:
# FT_x[real] -> complex : 1/2 as many complex numbers needed as real signal
# FT_y[complex] -> complex : After the first transform has been done the signal
# is complex, therefore the transformed domain in second dimension is same size
# as it is in euclidean space.
# Therefore FT[(nx, ny)] -> (nx/2, ny)
# The 2D Inverse transform returns a real-only domain (nx, ny)
k = dk*np.arange(0, nk, dtype=np.float64)[np.newaxis, :]
l = dl*fftfreq(nl, d=1.0/nl)[:, np.newaxis]

ksq = k**2 + l**2
ksq[ksq == 0] = 1.0             # avoid divide by zero - set ksq = 1 at zero wavenum
rksq = 1.0 / ksq                # reciprocal 1/(k^2 + l^2)

ik = 1j*k                       # wavenumber mul. imaginary unit is useful
il = 1j*l                       # for calculating derivatives

## Dissipation & Spectral Filters
# Use ∆^4 hyperviscosity to diffuse at small scales
# use the x-dimension for reference scale values
nu = ((Lx/(np.floor(nx/3)*2.0*pi))**4)/tau

# Spectral Filter as per [Arbic and Flierl, 2003]
wvx = np.sqrt((k*dx)**2 + (l*dy)**2)
spectral_filter = np.exp(-23.6*(wvx-0.65*pi)**4)
spectral_filter[wvx <= 0.65*pi] = 1.0


z = np.zeros((ny, nx), dtype=np.float64)
zt = np.zeros((nl, nk), dtype=np.complex128)

### Initial Condition
# The McWilliams Initial Condition from [McWilliams - J. Fluid Mech. (1984)]
ck = np.zeros_like(ksq)
ck = np.sqrt(ksq + (1.0 + (ksq/36.0)**2))**-1
pit = np.random.randn(*ksq.shape)*ck + 1j*np.random.randn(*ksq.shape)*ck

pi = ift(pit)
pi = pi - pi.mean()
pit = ft(pi)
KE = spectral_variance(pit*np.sqrt(ksq)*spectral_filter)

qit = -ksq * pit / np.sqrt(KE)
qi = ift(qit)
z[:] = qi

# initialise the transformed ζ
zt[:] = ft(z)
amp = forcing_amp_factor* np.max(np.abs(zt))        # calc a reasonable forcing amplitude


## RUN THE SIMULATION
plt.ion()                       # plot in realtime
plt.figure(figsize=(12, 6))
tplot = t + PLOT_EVERY_S
while t < tmax:
    # calculate derivatives in spectral space
    psit = -rksq * zt           # F[ψ] = - F[ζ] / (k^2 + l^2)
    psixt, psiyt = grad(psit)
    zxt, zyt = grad(zt)

    # transform back to physical space for pseudospectral part
    z[:] = ift(zt)
    psix = ift(psixt)
    psiy = ift(psiyt)
    zx =   ift(zxt)
    zy =   ift(zyt)

    # Non-linear: calculate the Jacobian in real space
    # and then transform back to spectral space
    jac = psix * zy - psiy * zx + ubar * zx
    jact = ft(jac)

    # apply forcing in spectral space by exciting certain wavenumbers
    # (could also apply some real-space forcing and then convert
    #   into spectral space before adding to rhs) 
    forcet = np.zeros_like(ksq)
    idx = (40 < np.sqrt(ksq)/dk) & (np.sqrt(ksq)/dk < 45)
    forcet[idx] = 0.5*amp*(np.random.random(ksq.shape)[idx] - 0.5)#*np.sin(0.5*t)

    # calculate the size of timestep that can be taken
    # (assumes a domain where dx and dy are of the same order)
    c = courant_number(psix, psiy, dx, dt)
    if c >= SLOWDN_AT_C:
        print('DEBUG: Courant No > 0.8, reducing timestep')
        dt = 0.9*dt
    elif c < SPEEDUP_AT_C and ALLOW_SPEEDUP:
        dt = 1.1*dt

# Possible use of time-varying dissipation, as in Maltrud & Vallis 1991, eq 2.6.
# 	nu = 1.0 * np.sqrt(np.mean(z**2.))/(np.max(k)**2.)





    # take a timestep and diffuse
    rhs = -jact - beta*psixt + forcet - zt*r_rayleigh
    zt[:] = adams_bashforth(zt, rhs, dt)
    del4 = 1.0 / (1.0 + nu*ksq**2*dt)
    zt[:] = zt * del4
    


    if t > tplot:
        
        # diagnostics
        urms=np.sqrt(np.mean(psix**2 + psiy**2))
        rhines_scale = np.sqrt(urms/beta)
    
        print('[{:5d}] {:.2f} Max z: {:2.2f} c={:.2f} dt={:.2f} rh_s={:.3f}'.format(
            step, t, np.max(z), c, dt, rhines_scale))
        plt.clf()
        plt.subplot(131)
        plt.imshow(z, extent=[0, Lx, 0, Ly], cmap=plt.cm.YlGnBu)
        plt.xlabel('x')
        plt.ylabel('y')
        zmax = np.max(np.abs(z))
        plt.clim(-zmax,zmax)
        plt.colorbar(orientation='horizontal')
        plt.title('Vorticity at {:.2f}s dt={:.2f}'.format(t, dt))

        plt.subplot(132)
        power = np.fft.fftshift(np.abs(zt)**2, axes=(0,))
        power_norm = np.log(power)
#         plt.imshow(power_norm,
#                     extent=[np.min(k/dk), np.max(k/dk), np.min(l/dl), np.max(l/dl)])
        plt.imshow(power_norm[32:95,0:31],
                    extent=[k[0,0]/dk, k[0,31]/dk, l[32,0]/dl, l[95,0]/dl])                    

        plt.xlabel('k/dk')
        plt.ylabel('l/dl')
        plt.colorbar(orientation='horizontal')
        plt.title('Power Spectra')
        
    	ax1=plt.subplot(133)
        ax1.plot(-np.mean(psiy,axis=1),np.linspace(0, Ly, num=ny))
        
        ax2=ax1.twiny()
        ax2.plot(-np.mean(z,axis=1)+np.linspace(0, beta*Ly, num=ny),np.linspace(0, Ly, num=ny),'g')
        
        
        plt.pause(0.01)
        tplot = t + PLOT_EVERY_S

    t = t + dt
    step = step + 1