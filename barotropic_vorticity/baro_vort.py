#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""beta plane barotropic vorticity model.
    
    This script uses a pseudospectral method to solve the barotropic vorticity
    equation in two dimensions
    
    D/Dt[ω] = 0                                                             (1)
    
    where ω = ξ + f.  ξ is local vorticity ∇ × u and f is global rotation.
    
    Assuming an incompressible two-dimensional flow u = (u, v),
    the streamfunction ψ = ∇ × (ψ êz) can be used to give (u,v)
    
    u = ∂/∂y[ψ]         v = -∂/∂x[ψ]                                            (2)
    
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
    This code was developed based on a MATLAB script bvebb.m
    and the GFDL documentation for spectral barotropic models
    found here [http://www.gfdl.noaa.gov/cms-filesystem-action/user_files/pjp/barotropic.pdf]
    McWilliams Initial Condition inspired by pyqg [https://github.com/pyqg/pyqg]
    """

import numpy as np
from numpy import pi, cos, sin

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
    pyfftw.interfaces.cache.enable()
    PYFFTW = True
except:
    print("WARNING: pyfftw not available.  Falling back to numpy")
    from numpy.fft import fftshift, fftn, ifftn
    PYFFTW = False

### PARAMETERS
N = 128         # numerical resolution
IC = 'spot'
AA_FAC = N / 6  # anti-alias factor.  AA_FAC = N : no anti-aliasing
#                     AA_FAC = 0 : no non-lin waves retained

ubar = 0.00     # background zonal velocity
beta = 1.7      # beta-plane f = f0 + βy
tau = 0.1       # coefficient of dissipation
num_timesteps = 100

ALLOW_SPEEDUP = False        # if True, allow the simulation to take a larger
SPEEDUP_AT_C  = 0.6          # timestep when the Courant number drops below
# this parameter SPEEDUP_AT_C
SLOWDN_AT_C = 0.8            # take smaller timesteps if Courant number
# is bigger than SLOWDN_AT_C
SHOW_CHART = True

STORE_DATA = True
data_interval = 10. #After how many timesteps should the data be saved 

def raw_filter(prev, curr, new, nu=0.01):
    """Robert-Asselin Filter."""
    return curr + nu*(new - 2.0*curr + prev)

def courant_number(psix, psiy, dx, dt):
    """Calculate the Courant Number given the velocity field and step size."""
    maxu = np.max(np.abs(psiy))
    maxv = np.max(np.abs(psix))
    maxvel = maxu + maxv;
    return maxvel*dt/dx;

def leapfrog(phi, f, dt):
    """Leapfrog time integration."""
    return phi + 2.0*dt*f

def ft(phi):
    """Go from physical space to spectral space."""
    return fftn(phi, axes=(0,1))

def ift(psi):
    """Go from spectral space to physical space."""
    return ifftn(psi, axes=(0,1))

def enstrophy(zt):
    """Calculate the enstrophy from transformed vorticity field."""
    return np.real(0.5*zt*np.conj(zt))

ICS = {}
def initial(name):
    """Decorate a function as an initial condition"""
    ic_name = name
    def register_ic(fn):
        ICS[name] = fn
        return fn
    return register_ic


# Some initial condition functions
@initial('random')
def random_ic(z):
    z[:] = 2*np.random.random(z.shape) - 1

@initial('spot')
def spot_ic(z):
    # single spot of max val 2.0 in lower half of plane
    d = int(z.shape[0] / 4.0)
    i, j = np.indices(z.shape)
    ppxy = np.abs(d - i)**2 + np.abs(d*2 - j)**2
    dist = np.sqrt(ppxy)
    z[dist < d] = (2.0*cos(0.5 * pi * (d - dist) / d + 0.5*pi)**2)[dist < d]

# TODO: This code is incomplete
@initial('mcwilliams')
def mcwilliams_ic(z):
    # initial condition taken from [McWilliams 1984]
    # > Gaussian random realisation for each Fourier component of ψ
    # > where at each vector wavenumber the ensemble variance is
    # > proportional to
    # > k^-1 (1 + (k/k0)^4)^-1  for k > 0
    K0 = 6.0
    zt = ft(z)
    nk, nl = zt.shape
    K = np.sqrt(ksq)
    kappa = K**-1 * (1.0 + (K / K0)**4)**-1
    Pi_hat = np.random.randn((nk, nl))*kappa + 1j*np.random.randn((nk, nl))*kappa
    Pi = ift(Pi_hat)
    Pi = Pi - Pi.mean()

def grad(phit):
    """Returns the spatial derivatives of a Fourier transformed variable.
        Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
    phixt = ik*phit        # d/dx F[φ] = ik F[φ]
    phiyt = il*phit        # d/dy F[φ] = il F[φ]
    return (phixt, phiyt)

def velocity(psit):
    """Returns the velocity field (u, v) from F[ψ]."""
    psixt, psiyt = grad(psit)
    psix = ift(psixt)    # v = - ∂/∂x[ψ]
    psiy = ift(psiyt)    # u =   ∂/∂y[ψ]
    return (psiy, -psix)

def tot_vort_calc(vort_rel,y_arr,beta):
    """Returns the Total vorticity from the relative vorticity, z."""
    tot_vort=vort_rel+beta*y_arr
    return tot_vort

def anti_alias(phit, k_max):
    """Set the coefficients of wavenumbers > k_max to be zero."""
    phit[(np.abs(kk) >= k_max) | (np.abs(ll) >= k_max)] = 0.0

def integrate():
    global _zt, zt, zt_, z, dt
    # Poisson equation for streamfunction and vorticity
    #    ζ = ∆ψ
    # Using the Fourier form
    #    ψ = a_k(t) exp(i (k x + l y))
    # => ζ = -(k^2 + l^2) ψ
    
    psit = -rksq * zt    # F[ψ] = - F[ζ] / (k^2 + l^2)
    psixt, psiyt = grad(psit)
    zxt, zyt = grad(zt)
    
    # transform back to physical space for pseudospectral part
    z =    ift(zt)
    psix = ift(psixt)
    psiy = ift(psiyt)
    zx =   ift(zxt)
    zy =   ift(zyt)
    
    # check Courant number is within bounds and adjust if neccesary
    c = courant_number(psix, psiy, dx, dt)
    if c > SLOWDN_AT_C:
        print('DEBUG: Courant No > 0.8, reducing timestep')
        dt = 0.9*dt
    elif c < SPEEDUP_AT_C and ALLOW_SPEEDUP:
        print('DEBUG: Courant No < 0.6, increasing timestep')
        dt = 1.1*dt
    
    # calculate the Jacobian in real space
    jac = psix * zy - psiy * zx + ubar * zx

    # transform jacobian to spectral space
    jact = ft(jac)
    
    # avoid aliasing by eliminating short wavelengths
    anti_alias(jact, k_max)
    
    
    
    # take a timestep
    rhs = -jact - beta*psixt
    zt_ = leapfrog(_zt, rhs, dt)
    zt_ = zt_ * del4               # dissipation
    
    # RAW filter in time
    _zt = raw_filter(_zt, zt, zt_)
    zt = zt_
    return c



### SETUP

## Physical Domain
domain = 1.0
nx = ny = n = N               # for simplicity use a square domain
dx = domain / nx
dy = domain / ny

y_vals=np.linspace(1,ny,num=ny,endpoint=True)*domain/ny
x_vals=np.linspace(1,nx,num=nx,endpoint=True)*domain/nx
x_arr,y_arr=np.meshgrid(x_vals,y_vals)

dt = 0.4 * 16.0 / nx          # choose an initial dt. This will change
# as the simulation progresses to maintain
# numerical stability

## Spectral Domain
dk = 2.0*pi/domain;
k = np.arange(-n/2, n/2)*dk
l = np.arange(-n/2, n/2)*dk

kk, ll = [fftshift(q) for q in np.meshgrid(k, l)]  # put in FFT order
ksq = kk**2 + ll**2
ksq[ksq == 0] = 1.0   # avoid divide by zero - set ksq = 1 at zero wavenum
rksq = 1.0 / ksq      # reciprocal 1/(k^2 + l^2)

ik = 1j*kk
il = 1j*ll


## Dissipation & Anti-Aliasing
nu = (((domain)/(np.floor(n/3)*2.0*pi))**4)/tau
del4 = 1.0 / (1.0 + nu*ksq**2*dt)      # dissipate at small scales

k_max = AA_FAC*2*dk    # anti-aliasing removes wavenumbers > k_max
# from the non-linear term


# initialize the vorticity field
# using FFTW array for speed if it is available
if PYFFTW:
    z = pyfftw.n_byte_align_empty((nx, nx), 16, 'complex128')
else:
    z = np.zeros((nx, ny), dtype=np.complex128)
z[:] = 0

# apply the initial condition to z
ic_fn = ICS.get(IC)
if ic_fn is None:
    raise Error('Unknown initial condition "%r"' % IC)
ic_fn(z)        # set the vorticity using an initial condition



# calculate initial transform
zt = ft(z)
_zt = zt_ = zt   # set initial previous value (_zt)
# and initial next value (zt_) to
# be the same as current value

e0 = np.sum(enstrophy(zt))  # intial enstrophy
print("Initial Enstrophy: %.3g" % e0)

tot_vort = tot_vort_calc(np.real(z),y_arr,beta)

### data_store_initialise

if STORE_DATA:
   z_store=np.zeros((np.ceil(num_timesteps/data_interval)+1,nx,ny))


### PLOT
import time
if SHOW_CHART:
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.real(tot_vort), cmap=plt.cm.seismic,origin='lower')
    fig.show()
    plt.pause(0.001)
#    time.sleep(0.2)
t = 0
timeit = time.time()
while t < num_timesteps:
    t = t + 1
    if (t % 100) == 0:
        print('Step: %d' % t)
        sps = 100 / (time.time() - timeit)
        print('Steps per second: %.2f' % sps)
        timeit = time.time()
    c = integrate()
    tot_vort = tot_vort_calc(np.real(z),y_arr,beta)
    if (t % 10) == 0 and SHOW_CHART:
        ax.set_title('Total vorticity [Courant No: %3.2f] dt=%4.3f' % (c, dt))
        im.set_data(np.real(tot_vort))
        im.axes.figure.canvas.draw()
        plt.pause(0.001)
    if STORE_DATA and (t % data_interval) == 0:
       z_store[np.floor(t/data_interval),:,:]=np.real(z)