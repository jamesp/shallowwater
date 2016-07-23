"""A shallow water model of the ENSO

The model comprises of two coupled shallow water systems representing the
atmosphere and Pacific ocean.

- The ocean component is the mixed-layer: down to the thermocline.
- The atmosphere is the lower part of the troposphere.

The two layers interact:  The ocean heats the atmosphere inducing a flow, which,
in turn, forces the ocean through a stress forcing.

The linearised shallow water equations are used, linearised about height H
determined by the Kelvin wave speed in the two layers.

c^2 = gH

* In the atmosphere: g = 10.0 m/s^2, c = 24 m/s =>  H ~= 58m
* In the ocean: g' = 0.1m/s^2, c = 4 m/s => H = 160m
"""

import numpy as np
import matplotlib.pyplot as plt

from shallowwater import PeriodicLinearShallowWater, WalledLinearShallowWater

np.set_printoptions(precision=2, suppress=True)  # 2 dp and hide floating point error

nx = 128
ny = 129

Lx = 1.5e7
Ly = 1.0e7

# Equatorial Beta-Plane
f0 = 0.0        # /s
beta = 2.0e-11  # /m.s

# Kelvin Wave speed in the atmosphere and ocean
c_atmos = 24.0 # m/s
c_ocean = 4.0  # m/s

g_atmos = 10.0  # m/s^2
H_atmos = c_atmos**2 / g_atmos
print('H atmosphere: %.2f' % H_atmos)

g_ocean = 0.1  # m/s^2
H_ocean = c_ocean**2 / g_ocean
print('H ocean: %.2f' % H_ocean)

alpha = 1e-4  # ocean -> atmos heating coefficient
gamma = 1e-7  # wind -> ocean wind stress coefficient

# Dissipation coefficients
nu_ocean = 1.0e4
nu_atmos = 1.0e4

# due to the order of magnitude difference in wave speeds in the two fluids
# the atmosphere is integrated over a smaller timestep and more often than
# the ocean.
ocean_dt = 2500.0
atmos_dt = ocean_dt / 10

# `atmos` represents the first baroclinic mode of the atmosphere.
# Localised heating below results in convection: convergence
# at the bottom of the troposphere and divergence at the top.
# We want to simulate the wind in the lower part of the
# troposphere => heating is represented as a *thinning* of the atmosphere layer.
atmos = PeriodicLinearShallowWater(nx, ny, Lx, Ly,
            beta=beta, f0=f0,
            g=g_atmos, H=H_atmos,
            dt=atmos_dt, nu=nu_atmos, r=1e-6)

# steady trade winds in the tropics
atmos.u[:] = -1.0*np.cos(np.pi*atmos.uy/Ly)

# `ocean` represents the mixed-layer of the ocean; height `h` is the depth
# of the thermocline.
# Where the thermocline is deeper = warmer water.  When the layer thins, the
# mixed-layer is cooler due to upwelling from the abyssal ocean.
ocean = WalledLinearShallowWater(nx, ny, Lx, Ly,
            beta=beta, f0=f0,
            g=g_ocean, H=H_ocean,
            dt=ocean_dt, nu=nu_ocean, r=1e-6)


@atmos.add_forcing
def heating(a):
    global alpha, ocean, atmos
    dstate = np.zeros_like(atmos.state)
    dstate[2] = -alpha*ocean.h  # thicker ocean layer = hotter.  hotter atmos = thinner atmos
    return dstate

@ocean.add_forcing
def wind_stress(o):
    global gamma, ocean, atmos
    dstate = np.zeros_like(ocean.state)
    dstate[0] = gamma*atmos.u
    dstate[1] = gamma*atmos.v
    return dstate

# @atmos.add_forcing
# def trade_winds(a):
#     global gust, ocean, atmos
#     dstate = np.zeros_like(atmos.state)
#     gust = np.zeros_like(atmos.u)
#     if a.tc % 10000 == 0:
#         print('gust!')
#         gust[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = -H_atmos * 0.01 / atmos_dt * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#     dstate[0] = gust
#     return dstate

# Initial Condition
d = 25
#ocean.h[10:20, ny//2-10:ny//2+10] = 0.1
#ocean.h[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = H_ocean * 0.01 * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#atmos.h[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = H_atmos * 0.01 * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#atmos.u[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = H_atmos * 0.01 * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#ocean.h[:] = np.random.random((nx, ny)) -0.5

plt.figure(figsize=(18, 6))
plt.ion()

num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

cmap = plt.cm.get_cmap('RdBu_r', 13)

def absmax(x):
    return np.max(np.abs(x))

def velmag(sw):
    """Velocity magnitude."""
    u, v  = sw.uvath()
    return np.sqrt(u**2 + v**2)


hx, hy = np.meshgrid(atmos.hx, atmos.hy)
arrow_spacing = slice(ny // 16, None, ny // 9), slice(nx // 12, None, nx // 12)
avg_thermocline = ocean.h.copy()
ema_multiplier  = 2.0 / (20 + 1)
equator_zonal_winds = []

plt.show()
for i in range(1000000):
    ocean.step()
    while atmos.t <= ocean.t:
        atmos.step()

    if i % 20 == 0:
        print('Time: %.3f days' % (ocean.t / 86400.0))

    if (ocean.t / 86400.0) > 0:
        if i % 10 == 0:
            avg_thermocline = avg_thermocline + (ocean.h - avg_thermocline)*ema_multiplier
            equator_zonal_winds.append((avg_thermocline - ocean.h)[:, ny//2].copy())

        if i % 10 == 0:
            print('Time: %.3f days' % (ocean.t / 86400.0))

            u, v  = atmos.uvath()/absmax(atmos.u)
            vel = np.sqrt(u**2 + v**2)

            print('Ocean Velocity: %.3f' % absmax(velmag(ocean)))
            print('Atmos Velocity: %.3f\n' % absmax(velmag(atmos)))

            plt.clf()
            plt.subplot(131)
            scaled_h = ocean.h.T * 1e6
            plt.contourf(hx, hy, scaled_h, cmap=plt.cm.RdBu_r, levels=colorlevels*absmax(scaled_h))
            plt.title('Thermocline perturbation')
            #plt.imshow(ocean.h.T, cmap=cmap)
            plt.colorbar()

            plt.subplot(132)
            plt.contourf(hx, hy, atmos.h.T, cmap=plt.cm.RdBu, levels=colorlevels*H_atmos*0.1)#absmax(atmos.h))
            plt.colorbar()
            plt.title('Atmosphere')
            plt.quiver(hx[arrow_spacing], hy[arrow_spacing],
                np.ma.masked_where(vel.T < 0.1, u.T)[arrow_spacing],
                np.ma.masked_where(vel.T < 0.1, v.T)[arrow_spacing], pivot='mid', scale=15, width=0.005)

            # plt.subplot(223)
            # plt.contourf(hx, hy, avg_thermocline.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(avg_thermocline))

            # plt.subplot(223)
            # delta = ocean.h - avg_thermocline
            # plt.contourf(hx, hy, delta.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(delta))
            # plt.colorbar()

            plt.subplot(133)
            plt.plot(-ocean.h[:, ny//2], label='thermocline')
            plt.plot(-avg_thermocline[:, ny//2], label='moving avg.')
            plt.title('Equatorial Thermocline')
            plt.legend(loc='lower right')
            # # if len(equator_zonal_winds) % 2 == 1:
            # #     power = np.log(np.abs(np.fft.fft2(np.array(equator_zonal_winds))**2))
            # # else:
            # #     power = np.log(np.abs(np.fft.fft2(np.array(equator_zonal_winds[:-1]))**2))
            # power = np.log(np.abs(np.fft.fft2(np.array(equator_zonal_winds))**2))
            # # khat = np.fft.fftshift(np.fft.fftfreq(power.shape[1], 1.0/nx))
            # # k = khat / Ly

            # # omega = np.fft.fftshift(np.fft.fftfreq(power.shape[0], np.diff(timestamps)[-1]))
            # # w = omega / np.sqrt(beta*c)

            # plt.pcolormesh(np.fft.fftshift(power)[::-1], cmap=plt.cm.gray)
            # plt.pcolormesh(np.array(equator_zonal_winds))
            plt.pause(0.01)
            plt.draw()