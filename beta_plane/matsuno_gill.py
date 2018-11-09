import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from shallowwater import PeriodicLinearShallowWater
from plotting import plot_wind_arrows

nx = 256
ny = nx//2 + 1


# Radius of deformation: Rd = sqrt(2 c / beta)
Rd = 1000.0e3  # Fix Rd at 1000km

Lx = 10*Rd
Ly = 5*Rd

beta=2.28e-11
c = Rd**2 * beta  # Kelvin wave speed: c = sqrt(gH)
g = 1.0

H = c**2/g       # Set phi baseline from deformation radius

cfl = 0.7         # For numerical stability CFL = |u| dt / dx < 1.0
dx  = Ly / nx
dt = np.floor(cfl * dx / (c*4))
print('dt', dt)

tau = 500000
nu = 1000

atmos = PeriodicLinearShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, g=g, H=H, dt=dt, nu=nu)

x, y = np.meshgrid(atmos.phix/Rd, atmos.phiy/Rd)
k = np.pi/2
Q0 = H * 0.01
Q = (Q0*np.exp(-(1/2)*y**2)*np.cos(k*x))
Q[np.abs(x) > 1] = 0
Q = Q.T

@atmos.add_forcing
def matsuno_gill(model):    
    u, v, h = model.state
    du, dv, dh = np.zeros_like(model.state)

    # forcing terms for the linear matsuno gill problem
    du = - u/tau
    dv = - v/tau
    dh = (Q - h)/tau

    return np.array([du, dv, dh])


N = int(tau/dt*3)
for i in tqdm(range(N)):
    atmos.step()


fig, ax = plt.subplots(figsize=(6, 4))

#plt.suptitle('State at T=%.2f days' % (atmos.t / 86400.0))

plt.contourf(x, y, atmos.h.T, 21, cmap=plt.cm.YlGnBu_r)
plot_wind_arrows(atmos, (x,y), narrows=(25,25), hide_below=0.01, color='white')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
#plt.colorbar()
plt.savefig('matsuno_gill.svg')
