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

import numpy as np

from numpy import pi, cos, sin
from numpy.fft import fftshift, fftfreq
from numpy.fft.fftpack import rfft2, irfft2

# # if available, we will use pyFFTW for performing Fourier Transforms
# try:
#     import pyfftw
#     from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
#     pyfftw.interfaces.cache.enable()
#     PYFFTW = True
# except:
#     print("WARNING: pyfftw not available.  Falling back to numpy")
#     from numpy.fft import fftshift, fftfreq
#     from numpy.fft.fftpack import rfft2, irfft2
#     PYFFTW = False

def ft(phi):
    """Go from physical space to spectral space."""
    return rfft2(phi, axes=(-2, -1))

def ift(psi):
    """Go from spectral space to physical space."""
    return irfft2(psi, axes=(-2,-1))


class BarotropicVorticity(object):
    """A square domain barotropic vorticity model."""
    _prhs = 0.0
    _pprhs = 0.0

    def __init__(self,
        n,              # numerical resolution
        L=1.0,          # domain size [m]
        ubar=0.0,       # background velocity [m/s]
        beta=0.0,       # beta plane value: f = f0 + βy  [m^-1.s^-1]
        tau=0.1,        # coeff of dissipation. smaller = more diss.
        n_diss = 2.0    # Small-scale dissipation of the form ∆^2n_diss,
                        # such that n_diss = 2 would be a ∆^4 hyperviscosity.
        ):

        self.ubar = ubar
        self.beta = beta

        # Physical Domain (real):
        # for simplicity, use a square domain
        self.Lx = self.Ly = L
        self.nx = self.ny = n
        self.dx = dx = L / n
        self.dy = dy = L / n
        self.y = np.linspace(0, L, n)
        self.x = np.linspace(0, L, n)
        self.dt = 0.4 * 16.0 / n    # choose an initial dt. This will change
                                    # as the simulation progresses to maintain
                                    # numerical stability

        # Spectral Domain (complex):
        self.nl = n
        self.nk = n//2 + 1
        self.dk = 2.0*pi/L
        self.dl = 2.0*pi/L
        # calculate the wavenumbers [1/m]
        # The real FT has half the number of wavenumbers in one direction:
        # FT_x[real] -> complex : 1/2 as many complex numbers needed as real signal
        # FT_y[complex] -> complex : After the first transform has been done the signal
        # is complex, therefore the transformed domain in second dimension is same size
        # as it is in euclidean space.
        # Therefore FT[(nx, ny)] -> (nx/2, ny)
        # The 2D Inverse transform returns a real-only domain (nx, ny)
        self.k = k = self.dk*np.arange(0, self.nk, dtype=np.float64)[np.newaxis, :]
        self.l = l = self.dl*fftfreq(self.nl, d=1.0/self.nl)[:, np.newaxis]


        self.ksq = ksq = k**2 + l**2
        self.ksq[ksq == 0] = 1.0             # avoid divide by zero - set ksq = 1 at zero wavenum
        self.rksq = 1.0 / ksq                # reciprocal 1/(k^2 + l^2)

        self.ik = 1j*k                       # wavenumber mul. imaginary unit is useful
        self.il = 1j*l                       # for calculating derivatives


        # Dissipation & Spectral Filters:
        # Use ∆^2n_diss hyperviscosity to diffuse at small scales (i.e. n_diss = 2 would be ∆^4)
        # use the x-dimension for reference scale values
        self.nu = ((L/(np.floor(n/3)*2.0*pi))**(2*n_diss))/tau
        self.n_diss = n_diss

        self.t = 0.0                # time
        self.tc = 0                 # step count

        # physical and transformed variables
        self._z = np.zeros((self.nx,self.ny), dtype=np.float64)
        self._zt = np.zeros((self.nl,self.nk), dtype=np.complex128)

        self.psi = np.zeros_like(self._z)
        self.psit = np.zeros_like(self._zt)


    def courant_number(self):
        """Calculate the Courant Number given the velocity field and step size."""
        u,v = self.velocity()
        maxu = np.max(np.abs(u))
        maxv = np.max(np.abs(v))
        maxvel = maxu + maxv
        return maxvel*self.dt/self.dx

    def grad(self, phit):
        """Returns the spatial derivatives of a Fourier transformed variable.
        Returns (∂/∂x[F[φ]], ∂/∂y[F[φ]]) i.e. (ik F[φ], il F[φ])"""
        phixt = self.ik*phit        # d/dx F[φ] = ik F[φ]
        phiyt = self.il*phit        # d/dy F[φ] = il F[φ]
        return (phixt, phiyt)

    def velocity(self):
        """Returns the velocity field (u, v) from F[ψ]."""
        psixt, psiyt = self.grad(self.psit)
        psix = ift(psixt)    # v =   ∂/∂x[ψ]
        psiy = ift(psiyt)    # u = - ∂/∂y[ψ]
        return (-psiy, psix)

    def anti_alias(self, phit):
        """Set the coefficients of wavenumbers > k_mask to be zero."""
        k_mask = (8./9.)*(self.nk+1)**2.
        phit[(np.abs(self.ksq/(self.dk*self.dk)) >= k_mask)] = 0.0

    def high_wn_filter(self, phit):
        """Applies the high wavenumber filter of smith et al 2002"""
        filter_exp = 8.0
        kcut = 30.0
        filter_dec = -np.log(1.+2.*pi/self.nk)/((self.nk-kcut)**filter_exp)
        filter_idx = np.abs(self.ksq/(self.dk*self.dk)) >= kcut**2.
        phit[filter_idx] *= np.exp(filter_dec*(np.sqrt(self.ksq[filter_idx]/(self.dk*self.dk))-kcut)**filter_exp)

    @property
    def zt(self):
        return self._zt

    @zt.setter
    def zt(self, value):
        # set the transformed value of zeta
        self._zt[:] = value
        # update physical zeta and other dependents
        self._z[:] = ift(value)
        self._update_psi()

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z[:] = value
        self._zt[:] = ft(value)
        self._update_psi()

    def _update_psi(self):
        """After z or zt have changed, update the streamfunction."""
        self.psit[:] = -self.rksq * self._zt     # F[ψ] = - F[ζ] / (k^2 + l^2)
        self.psi[:] = ift(self.psit)


    def step(self):
        """Take a single step forward in time using Adams-Bashforth 3."""
        dt = self.dt

        # calculate the size of timestep that can be taken
        c = self.courant_number()
        if c >= 0.8:
            print('DEBUG: Courant No > 0.8, reducing timestep')
            dt = 0.9*dt
        elif c < 0.4:
            dt = 1.1*dt

        if self.tc is 0:
            # forward euler
            dt1 = dt
            dt2 = 0.0
            dt3 = 0.0
        elif self.tc is 1:
            # AB2 at step 2
            dt1 = 1.5*dt
            dt2 = -0.5*dt
            dt3 = 0.0
        else:
            # AB3 from step 3 on
            dt1 = 23./12.*dt
            dt2 = -16./12.*dt
            dt3 = 5./12.*dt

        rhs = self.rhs()
        newzt = self.zt + dt1*rhs + dt2*self._prhs + dt3*self._pprhs
        self._pprhs = self._prhs
        self._prhs  = rhs

        # apply hyperviscosity
        #deln = 1.0 / (1.0 + self.nu*self.ksq**self.n_diss*dt)
        #newzt = newzt*deln
        self.high_wn_filter(newzt)
        self.anti_alias(newzt)

        # update the state
        self.zt = newzt
        self.tc = self.tc + 1
        self.t = self.t + dt

    def forcing(self):
        """Apply a forcing in physical space."""
        return None

    def forcingt(self):
        """Apply a forcing in spectral space."""
        amp = 0.01
        forcet = np.zeros_like(self.zt)
        K = np.sqrt(self.ksq)/self.dk
        idx = (14 < K) & (K < 20)
        forcet[idx] = amp*0.5*(np.random.random(forcet.shape)[idx] - 0.5)*np.exp(1j*2.*pi*np.random.random(forcet.shape)[idx])
        return 0.0

    def rhs(self):
        # calculate derivatives in spectral space
        psixt, psiyt = self.grad(self.psit)
        zxt, zyt = self.grad(self.zt)

        # transform back to physical space for pseudospectral part
        psix = ift(psixt)
        psiy = ift(psiyt)
        zx =   ift(zxt)
        zy =   ift(zyt)

        # Non-linear: calculate the Jacobian in real space
        # and then transform back to spectral space
        jac = psix * zy - psiy * zx + self.ubar * zx
        jact = ft(jac)

        force = self.forcing()
        forcet = self.forcingt()
        if forcet is None:
            forcet = 0.0

        if force is not None:
            forcet = forcet + ft(force)

        rhs = -jact - self.beta*psixt + forcet
        return rhs

if __name__ == '__main__':
    bv = BarotropicVorticity(n=256, ubar=0.00, beta=8.0)
    # The McWilliams Initial Condition from [McWilliams - J. Fluid Mech. (1984)]
    ksq = bv.ksq
    ck = np.zeros_like(ksq)
    ck = np.sqrt(ksq + (1.0 + (ksq/36.0)**2))**-1
    piit = np.random.randn(*ksq.shape)*ck + 1j*np.random.randn(*ksq.shape)*ck

    pii = ift(piit)
    pii = pii - pii.mean()
    piit = ft(pii)
    #KE = spectral_variance(piit*np.sqrt(ksq)*spectral_filter)
    KE = 0.3
    qit = -ksq * piit / np.sqrt(KE)
    bv.zt = qit


    # single spot of max val 2.0 in lower half of plane
    # d = int(bv.z.shape[0] / 4.0)
    # i, j = np.indices(bv.z.shape)
    # ppxy = np.abs(d - i)**2 + np.abs(d*2 - j)**2
    # dist = np.sqrt(ppxy)
    # bv.z = np.exp(-(dist)/40)

    # PLOT
    import time
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(bv.z, cmap=plt.cm.seismic,origin='lower')
    #im = ax.imshow(tot_vort, cmap=plt.cm.seismic,origin='lower')
    fig.show()
    plt.pause(0.001)

    for i in range(1000):
        bv.step()
        im.set_data(bv.z)
        im.set_clim(bv.z.min(), bv.z.max())
        im.axes.figure.canvas.draw()
        plt.pause(0.001)
        print(bv.tc)