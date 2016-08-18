#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""beta plane barotropic vorticity model.

Solve the barotropic vorticity equation in two dimensions

    D/Dt[ω] = 0                                                             (1)

where ω = ζ + f is absolute vorticity.  ζ is local vorticity ∇ × u and
f is global rotation.

Assuming an incompressible two-dimensional flow u = (u, v),
the streamfunction ψ = ∇ × (ψ êz) can be used to give (u,v)

    u = -∂/∂y[ψ]         v = ∂/∂x[ψ]                                        (2)

and therefore local vorticity is given by the Poisson equation

    ζ = ∆ψ                                                                  (3)

Since ∂/∂t[f] = 0 equation (1) can be written in terms of the local vorticity

        D/Dt[ζ] + u·∇f = 0
    =>  D/Dt[ζ] = -vβ                                                       (4)

using the beta-plane approximation f = f0 + βy.  This can be written entirely
in terms of the streamfunction and this is the form that will be solved
numerically.

    D/Dt[∆ψ] = -β ∂/∂x[ψ]                                                   (5)

"""
import logging

import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

N = 96
Lx, Ly = (1., 1.)
nx, ny = (N, N)
beta = 10.0
U = 0.0

# setup the domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['psi', 'zeta', 'u', 'v'])

problem.parameters['beta'] = beta
problem.parameters['U'] = U

# solve the problem from the equations
# ζ = Δψ
# ∂/∂t[∆ψ] + β ∂/∂x[ψ] = -J(ζ, ψ)
problem.add_equation("zeta - dx(dx(psi)) - dy(dy(psi)) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("psi = U*y", condition="(nx == 0) and (ny == 0)")

problem.add_equation("dt(zeta) + beta*dx(psi) = - (dy(psi) - U)*dx(zeta) + dx(psi)*dy(zeta)")
problem.add_equation("u + dy(psi) = 0")
problem.add_equation("v - dx(psi) = 0")

solver = problem.build_solver(de.timesteppers.CNAB2)
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = 1000

x = domain.grid(0)
y = domain.grid(1)
zeta = solver.state['zeta']
psi = solver.state['psi']
u = solver.state['u']
v = solver.state['v']

#zeta['g'] = np.exp(-(x/0.1)**2) * np.exp(-(y/0.1)**2)
#zeta['g'] = psi.differentiate('x').differentiate('x') + psi.differentiate('y').differentiate('y')

zeta['g'] = np.random.random((nx, ny))

initial_dt = dt = 0.1 #Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

plt.ion()

x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(10,5))
p = axis.imshow(zeta['g'].T, cmap=plt.cm.YlGnBu)
plt.pause(1)

logger.info('Starting loop')
while solver.ok:
    dt = cfl.compute_dt()
    print(dt)
    # if dt > initial_dt:
    #   dt = initial_dt
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_data(zeta['g'].T)
        p.set_clim(np.min(zeta['g']), np.max(zeta['g']))
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
        plt.pause(0.001)

# Print statistics
logger.info('Iterations: %i' %solver.iteration)


