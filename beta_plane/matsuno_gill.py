import matplotlib.pyplot as plt
import numpy as np

from nonlinear import PeriodicShallowWater


nx = 128
ny = 129

Lx = 1.5e7
Ly = 1.5e7



phi0 = 100.0
phic = phi0*1.00


# Radius of deformation: Rd = sqrt(2 c / β)
Rd = 1000.0e3  # Fix Rd at 1000km
beta=2.28e-11
c = Rd**2 * beta  # Kelvin/gravity wave speed: c = sqrt(phi0)

print('c', c)
phi0 = c**2       # Set phi baseline from deformation radius

cfl = 0.7         # For numerical stability CFL = |u| dt / dx < 1.0
dx  = Ly / nx
dt = np.floor(cfl * dx / (c*4))  # TODO check this calculation for c-grid
print('dt', dt)

gamma = 4e-4
tau = dt*10.0

ocean = PeriodicShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, dt=dt, nu=5.0e4)
ocean.phi[:] += phi0


# Add a lump of fluid with scale 2 Rd
d = (Ly // Rd)
hump = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]

@ocean.add_forcing
def rhs(model):
    phi = model.phi

    # phi rhs
    dphi = np.zeros_like(phi)

    #  Fixed heating on equator
    dphi[nx//2-d:nx//2+d, ny//2-d:ny//2+d] = hump*gamma
    #  Newtonian relaxation
    dphi -= (phi - phi0)/tau

    return np.array([[0], [0], dphi])


plt.ion()

num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

plt.show()
for i in range(100000):
    ocean.step()

    if i % 10 == 0:

        plt.figure(1, figsize=(8, 12))
        plt.clf()

        plt.suptitle('State at T=%.2f days' % (ocean.t / 86400.0))
        plt.subplot(211)
        x, y = np.meshgrid(ocean.phix/Rd, ocean.phiy/Rd)
        plt.contourf(x, y, ocean.phi.T, cmap=plt.cm.RdBu, levels=phi0+colorlevels*phi0*0.01)
        #plt.xlim(-0.5, 0.5)
        # # Kelvin wavespeed tracer
        # kx = ((ocean.t*np.sqrt(phi0)/Lx % 1) - .5)
        # plt.scatter([kx], [0.4], label='sqrt(phi) tracer')
        # Heating souce location
        c = plt.Circle((0,0), 0.5, fill=False)
        plt.gca().add_artist(c)
        plt.text(0, 0.7, 'Heating')
        plt.xlabel('x (multiples of Rd)')
        plt.ylabel('y (multiples of Rd)')
        plt.xlim(-Lx/Rd/2, Lx/Rd/2)
        plt.ylim(-Ly/Rd/2, Ly/Rd/2)
        plt.title('Geopotential')

        plt.subplot(212)
        plt.plot(ocean.phix/Rd, ocean.phi[:, ny//2], label='equator')
        plt.plot(ocean.phix/Rd, ocean.phi[:, ny//2+(Ly//Rd//2)], label='tropics')
        plt.ylim(phi0*.99, phi0*1.01)
        plt.legend(loc='lower right')
        plt.title('Longitudinal Geopotential')
        plt.xlabel('x (multiples of Rd)')
        plt.ylabel('Geopotential')
        plt.xlim(-Lx/Rd/2, Lx/Rd/2)
        plt.pause(0.01)
        plt.draw()

