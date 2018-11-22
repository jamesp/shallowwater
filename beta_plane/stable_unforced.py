import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from shallowwater import PeriodicShallowWater
from spectral_analysis import kiladis_spectra, background

nx = 128
ny = 129
beta=2.0e-11
Lx = 1.5e7
Ly = 1.5e7

dt = 900.0
phi0 = 100.0

ocean = PeriodicShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=1.0e2)
ocean.phi[:] += phi0
#ocean.phi[:] += (np.random.random((nx, ny)) - 0.5)*phi0*0.05

d = 25
hump = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]

ocean.phi[nx//2-d:nx//2+d, ny//2-d:ny//2+d] += hump*3.0
#ocean.phi[nx//4-d:nx//4+d, ny//2-d:ny//2+d] += -hump*5.0
initial_phi = ocean.phi.copy()



num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])


en = []
qn = []

eq_reg = []
ts = []

hx, hy = np.meshgrid(ocean.phix, ocean.phiy)
arrow_spacing = slice(ny // 16, None, ny // 9), slice(nx // 12, None, nx // 12)

plt.ion()
plt.show()
for i in range(100000):
    ocean.step()

    #qbar[:] = ocean.tracer('q').mean()

    if i % 10 == 0:
        eq_reg.append(ocean.u.copy()[:, ny//2-5:ny//2+5])
        ts.append(ocean.t)

    if i % 40 == 0:

        plt.figure(1, figsize=(16,12))
        plt.clf()

        plt.subplot(221)
        u, v  = ocean.uvath()
        vel = np.sqrt(u**2 + v**2)
        velmax = vel.max()
        velnorm = vel / velmax
        print(velmax)
        x, y = np.meshgrid(ocean.phix/ocean.Lx, ocean.phiy/ocean.Ly)
        plt.contourf(x, y, ocean.phi.T, cmap=plt.cm.RdBu, levels=phi0+colorlevels*phi0*0.03)
        plt.quiver(x[arrow_spacing], y[arrow_spacing],
                np.ma.masked_where(velnorm.T < 0, u.T / velmax)[arrow_spacing],
                np.ma.masked_where(velnorm.T < 0, v.T / velmax)[arrow_spacing], pivot='mid', scale=15, width=0.005)

        plt.xlim(-0.5, 0.5)
        plt.title('Geopotential')

        plt.subplot(222)
        #en.append(np.sum(ocean.phi - initial_phi))
        en.append(np.mean(ocean.phi))
        plt.plot(en)
        plt.ylim(phi0*0.99, phi0*1.01)
        plt.title('Geopotential Loss')

        plt.subplot(223)

        if len(ts) > 50:
            specs = kiladis_spectra(eq_reg)
            spec = np.sum(specs, axis=0)
            nw, nk = spec.shape
            fspec = np.fft.fftshift(spec)
            fspec -= background(fspec, 10, 0)
            om = np.fft.fftshift(np.fft.fftfreq(nw, ts[1]-ts[0]))
            k = np.fft.fftshift(np.fft.fftfreq(nk, 1.0/nk))
            #plt.pcolormesh(k, om, np.log(1 + np.abs(spec)**2))
            plt.pcolormesh(k, om, np.log(1+np.abs(fspec)), cmap=plt.cm.bone)
            plt.xlim(-15, 15)
            plt.ylim(-0.00002, 0.00002)
        # spec = np.fft.fft2(eq_reg)
        # #spec = spec - background(spec, 10, 0)
        # nw, nk = spec.shape
        # om = np.fft.fftshift(np.fft.fftfreq(nw, 10.0/dt))
        # k = np.fft.fftshift(np.fft.fftfreq(nk, 1.0/nk))
        # #plt.pcolormesh(np.fft.fftshift(np.log(np.abs(spec)))[4*nw//10:nw//2, nk//4:3*nk//4][::-1], cmap=plt.cm.bone)
        # log_spec=np.log(np.abs(spec)**2)
        # plt.pcolormesh(k, om, np.fft.fftshift(log_spec)[::-1], cmap=plt.cm.bone)
        # plt.xlim(-40, 40)
        # plt.ylim(0, 10)
        # plt.clim(log_spec.min()*0.3, log_spec.max()*0.7)


        plt.subplot(224)
        plt.plot(ocean.phix/ocean.Lx, ocean.phi[:, ny//2])
        plt.plot(ocean.phix/ocean.Lx, ocean.phi[:, ny//2+8])
        plt.xlim(-0.5, 0.5)
        plt.ylim(phi0*.95, phi0*1.05)
        plt.title('Equatorial Height')

        plt.pause(0.01)
        plt.draw()

# for i in range(10000):
#     ocean.step()
#     if not i % 100:
#         print i

#     #qbar[:] = ocean.tracer('q').mean()
#     if not i % 40:
#         eq = ocean.u[:, ny//2-8:ny//2+8]
#         eq_reg.append(np.sum(eq, axis=1))
#         #eq_reg.append(ocean.u[:, ny//2])
#         ts.append(ocean.t)


# plt.figure(1, figsize=(12,12))
# eq_reg = np.array(eq_reg)[10:, :]
# print eq_reg.shape
# spec = np.fft.fft2(np.array(eq_reg))
# spec = spec - background(spec, 10, 0)
# nw, nk = spec.shape
# om = np.fft.fftshift(np.fft.fftfreq(nw, 10.0/dt))
# k = np.fft.fftshift(np.fft.fftfreq(nk, 1.0/nk))
# #plt.pcolormesh(np.fft.fftshift(np.log(np.abs(spec)))[4*nw//10:nw//2, nk//4:3*nk//4][::-1], cmap=plt.cm.bone)
# log_spec=np.log(1+np.abs(spec)**2)
# plt.pcolormesh(k, om, np.fft.fftshift(log_spec)[::-1], cmap=plt.cm.bone)
# #plt.xlim(-20, 20)
# #plt.ylim(0, 10)
# #plt.clim(log_spec.min()*0.3, log_spec.max()*0.7)


# plt.show()