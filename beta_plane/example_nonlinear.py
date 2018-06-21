#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from shallowwater import PeriodicShallowWater
from spectral_analysis import background, kiladis_spectra



if __name__ == '__main__':
    nx = 128
    ny = 129
    beta=2.0e-11
    Lx = 1.0e7
    Ly = 1.0e7

    dt = 500.0
    phi0 = 100.0

    ocean = PeriodicShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=1.0e3)

    @ocean.add_forcing
    def rhs(self):
        dstate = np.zeros_like(self.state)
        return dstate

    d = 25
    hump = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]

    ocean.phi[:] += phi0
    #ocean.phi[70-d:70+d, ny//2-d:ny//2+d] += hump*0.1
    for i in range(30):
        ocean.phi[:] += 0.1*np.real(np.exp(1j*(i+1)*2*np.pi*np.random.random((nx, ny))))
        # phase = np.random.random()
        # amp = np.random.random()*0.1
        # wv = np.random.randint(30)
        # ocean.phi[:] += (amp
        #     * np.cos(2*np.pi*wv*(ocean.phix / ocean.Lx - np.random.random()))
        #     * np.cos(2*np.pi*wv*(ocean.phiy / ocean.Ly - np.random.random())))
    #ocean.phi[:] -= hump.sum()/(ocean.nx*ocean.ny)

    initial_phi = ocean.phi.copy()

    q = np.zeros_like(ocean.phi)
    q[nx//2-d:nx//2+d, ny//2-d:ny//2+d] += hump
    q0 = q.sum()

    ocean.add_tracer('q', q, kappa=0.0)

    plt.ion()

    num_levels = 24
    colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

    en = []
    qn = []

    eq_reg = []
    ts = []

    plt.show()
    for i in range(100000):
        ocean.step()

        if i % 10 == 0:
            eq = ocean.u.copy()[:, ny//2-5:ny//2+5]
            eq_reg.append(eq)
            ts.append(ocean.t)

            eq_reg = eq_reg[-1000:]
            ts = ts[-1000:]

        if i % 100 == 0:

            plt.figure(1, figsize=(16,12))
            plt.clf()

            plt.subplot(231)
            x, y = np.meshgrid(ocean.phix/ocean.Lx, ocean.phiy/ocean.Ly)
            plt.contourf(x, y, ocean.phi.T, cmap=plt.cm.RdBu, levels=phi0+colorlevels*phi0*0.01)
            plt.xlim(-0.5, 0.5)
            plt.title('Geopotential')

            plt.subplot(232)
            en.append(np.sum(ocean.phi - initial_phi))
            qn.append(ocean.tracer('q').sum() - q0)
            plt.plot(en)
            #plt.plot(qn)
            plt.title('Geopotential Loss')

            plt.subplot(233)
            if len(ts) > 50:
                specs = kiladis_spectra(eq_reg)
                spec = np.sum(specs, axis=0)
                nw, nk = spec.shape
                fspec = np.fft.fftshift(spec)
                fspec -= background(fspec, 10, 0)
                om = np.fft.fftshift(np.fft.fftfreq(nw, ts[1]-ts[0]))
                k = np.fft.fftshift(np.fft.fftfreq(nk, 1.0/nk))
                plt.pcolormesh(k, om, np.log(1+np.abs(fspec)), cmap=plt.cm.bone)
                plt.xlim(-15, 15)
                plt.ylim(-0.00002, 0.00002)
            plt.title('Power Spectra')

            plt.subplot(234)
            plt.plot(ocean.phix/ocean.Lx, ocean.phi[:, ny//2])
            plt.plot(ocean.phix/ocean.Lx, ocean.phi[:, ny//2+8])
            plt.xlim(-0.5, 0.5)
            plt.ylim(phi0*.99, phi0*1.01)
            plt.title('Equatorial Height')


            plt.subplot(235)
            plt.contourf(x, y, ocean.tracer('q').T, cmap=plt.cm.RdBu, levels=colorlevels)
            c = plt.Circle((0,0), float(d)/nx/2, fill=False)
            plt.gca().add_artist(c)
            plt.xlim(-.5, .5)
            plt.ylim(-.5, .5)
            plt.title('Tracer')

            plt.pause(0.01)
            plt.draw()
