import numpy as np
import matplotlib.pyplot as plt

from shallowwater import PeriodicLinearShallowWater, WalledLinearShallowWater

np.set_printoptions(precision=2, suppress=True)  # 2 dp and hide floating point error

nx = 128
ny = 129

Lx = 1.5e7
Ly = 1.0e7

f0 = 0.0
beta = 2.0e-11

# Kelvin Wave speed in the atmosphere and ocean
c_atmos = 24.0 # m/s
c_ocean = 4.0  # m/s

g_atmos = 10.0  # m/s^2
H_atmos = c_atmos**2 / g_atmos
print('H atmosphere: %.2f' % H_atmos)

g_ocean = 0.1  # m/s^2
H_ocean = c_ocean**2 / g_ocean
print('H ocean: %.2f' % H_ocean)


alpha = 1e-12
gamma = 1e-11

# Dissipation coefficients
nu_ocean = 1.0e5
nu_atmos = 1.0e5

ocean_dt = 2500.0
atmos_dt = ocean_dt / 10


atmos = PeriodicLinearShallowWater(nx, ny, Lx, Ly,
            beta=beta, f0=f0,
            g=g_atmos, H=H_atmos,
            dt=atmos_dt, nu=nu_atmos, r=1e-5)
ocean = WalledLinearShallowWater(nx, ny, Lx, Ly,
            beta=beta, f0=f0,
            g=g_ocean, H=H_ocean,
            dt=ocean_dt, nu=nu_ocean, r=1e-6)

@atmos.add_forcing
def heating(a):
    global alpha, ocean
    dstate = np.zeros_like(a.state)
    dstate[2] = alpha*ocean.h
    return dstate

@ocean.add_forcing
def wind_stress(o):
    global gamma, atmos
    dstate = np.zeros_like(o.state)
    dstate[0] = -gamma*atmos.u
    return dstate

@atmos.add_forcing
def trade_winds(a):
    dstate = np.zeros_like(a.state)
    gust = np.zeros_like(a.u)
    if a.tc % 10000 == 0:
        print('gust!')
        gust[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = -H_atmos * 0.01 / atmos_dt * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
    dstate[0] = gust
    return dstate

# Initial Condition
d = 25
#ocean.h[10:20, ny//2-10:ny//2+10] = 0.1
#ocean.h[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = H_ocean * 0.01 * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#atmos.h[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = H_atmos * 0.01 * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#atmos.u[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = H_atmos * 0.01 * (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#ocean.h[:] = np.random.random((nx, ny)) -0.5

plt.figure(figsize=(12, 12))
plt.ion()

num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

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

    if (ocean.t / 86400.0) > 50:
        if i % 10 == 0:
            avg_thermocline = avg_thermocline + (ocean.h - avg_thermocline)*ema_multiplier
            equator_zonal_winds.append((avg_thermocline - ocean.h)[:, ny//2].copy())

        if i % 200 == 0:
            print('Time: %.3f days' % (ocean.t / 86400.0))

            u, v  = atmos.uvath()/absmax(atmos.u)
            vel = np.sqrt(u**2 + v**2)

            print('Ocean Velocity: %.3f' % absmax(velmag(ocean)))
            print('Atmos Velocity: %.3f\n' % absmax(velmag(atmos)))

            plt.clf()
            plt.subplot(221)
            plt.contourf(hx, hy, ocean.h.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(ocean.h))
            plt.colorbar()

            plt.subplot(222)
            plt.contourf(hx, hy, atmos.h.T, cmap=plt.cm.RdBu, levels=colorlevels*H_atmos*0.05)#absmax(atmos.h))
            plt.colorbar()
            plt.quiver(hx[arrow_spacing], hy[arrow_spacing],
                np.ma.masked_where(vel.T < 0.1, u.T)[arrow_spacing],
                np.ma.masked_where(vel.T < 0.1, v.T)[arrow_spacing], pivot='mid', scale=15, width=0.005)

            # plt.subplot(223)
            # plt.contourf(hx, hy, avg_thermocline.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(avg_thermocline))

            plt.subplot(223)
            delta = ocean.h - avg_thermocline
            plt.contourf(hx, hy, delta.T, cmap=plt.cm.RdBu, levels=colorlevels*absmax(delta))
            plt.colorbar()

            plt.subplot(224)

            # if len(equator_zonal_winds) % 2 == 1:
            #     power = np.log(np.abs(np.fft.fft2(np.array(equator_zonal_winds))**2))
            # else:
            #     power = np.log(np.abs(np.fft.fft2(np.array(equator_zonal_winds[:-1]))**2))
            power = np.log(np.abs(np.fft.fft2(np.array(equator_zonal_winds))**2))
            # khat = np.fft.fftshift(np.fft.fftfreq(power.shape[1], 1.0/nx))
            # k = khat / Ly

            # omega = np.fft.fftshift(np.fft.fftfreq(power.shape[0], np.diff(timestamps)[-1]))
            # w = omega / np.sqrt(beta*c)

            plt.pcolormesh(np.fft.fftshift(power)[::-1], cmap=plt.cm.gray)
            #plt.pcolormesh(np.array(equator_zonal_winds))
            plt.pause(0.01)
            plt.draw()