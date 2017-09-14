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

from dedalus import public as de
from dedalus.extras import flow_tools

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

PLOTTING = False
q = 1
N = 96*2**q
Lx, Ly = (1., 1.)
nx, ny = (N, N)
beta = 0.0
U = 0.0
D = 1e-16
dt = 1e-6


# setup the domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['psi'])


# solve the problem from the equations
# ζ = Δψ
# ∂/∂t[∆ψ] + β ∂/∂x[ψ] = -J(ζ, ψ)

# Everytime you ask for one of the expression on the left, you will get the expression on the right.
problem.substitutions['L(a)']         = "  (d(a,x=2) + d(a,y=2)) "
problem.substitutions['J(a,b)'] = "  (dx(a)*dy(b) - dy(a)*dx(b)) "

problem.substitutions['u']    = " -dy(psi) "
problem.substitutions['v']    = "  dx(psi) "
problem.substitutions['zeta'] = " L(psi) "

# You can combine things if you want
problem.substitutions['HD(a, n)']         = "  -D*(d(a, x=n) + d(a, y=n)) "

problem.parameters['beta'] = beta
problem.parameters['U']    = U
problem.parameters['D']   = D # hyperdiffusion coefficient

problem.add_equation("dt(zeta) + beta*v - HD(zeta, 8) = J(zeta, psi) ", condition="(nx != 0) or  (ny != 0)")
problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")
#problem.add_equation("zeta = L(psi)", condition="(nx != 0) or  (ny != 0)")


solver = problem.build_solver(de.timesteppers.SBDF3)
solver.stop_sim_time  = .1
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# vorticity & velocity are no longer states of the system. They are true diagnostic variables.
# But you still might want to set initial condiitons based on vorticity (for example).
# To do this you'll have to solve for the streamfunction.

# This will solve for an inital psi, given a vorticity field.
init = de.LBVP(domain, variables=['init_psi'])

gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

init_vorticity = domain.new_field()
init_vorticity.set_scales(1)


k = domain.bases[0].wavenumbers[:, np.newaxis]
l = domain.bases[1].wavenumbers[np.newaxis, :]
ksq = k**2 + l**2

ck = np.zeros_like(ksq)
ck = np.sqrt(ksq + (1.0 + (ksq/36.0)**2))**-1
piit = np.random.randn(*ksq.shape)*ck + 1j*np.random.randn(*ksq.shape)*ck
pii = np.fft.irfft2(piit.T)
pii = pii - pii.mean()
piit = np.fft.rfft2(pii).T
cslices = domain.dist.coeff_layout.slices(scales=1)
init_vorticity['c'] = (-ksq*piit)[cslices]


# x,y = domain.grids(scales=1)

#init_vorticity['g'] =  (0.5)*noise #+  3*np.exp( - 80* ( (x-Lx/2)**2 + (y-Ly/4)**2 ) )

init.parameters['init_vorticity'] = init_vorticity

init.add_equation(" d(init_psi,x=2) + d(init_psi,y=2) = init_vorticity", condition="(nx != 0) or  (ny != 0)")
init.add_equation(" init_psi = 0",                                        condition="(nx == 0) and (ny == 0)")

init_solver = init.build_solver()
init_solver.solve()

psi = solver.state['psi']
psi['g'] = init_solver.state['init_psi']['g']

# plt.imshow(init_vorticity['g'])
# plt.show()


# plt.imshow(psi['g'])
# plt.show()

# Now you are ready to go.
# Anytime you ask for zeta, u, or v they will be non-zero because psy is non-zero.

cfl = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=.5,
                     max_change=1.5, min_change=0.5)
cfl.add_velocities(('u','v'))


dout = solver.evaluator.add_dictionary_handler(iter=10)
fout = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=1e-3)

outputs = [fout]
if PLOTTING:
    outputs.append(dout)

for output in outputs:
    output.add_system(solver.state)
    output.add_task('zeta', scales=1, name='zeta')
    output.add_task('u', scales=1, name='u')
    output.add_task('v', scales=1, name='v')


if PLOTTING:
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axis = plt.subplots(figsize=(10,5))

    im = axis.imshow(init_vorticity['g'].T, cmap=plt.cm.YlGnBu)
    plt.pause(1)

logger.info('Starting loop')
while solver.ok:
    dt = cfl.compute_dt()
    # print(dt)
    solver.step(dt)
    if solver.iteration % 10 == 0:
        if PLOTTING:
            zeta = dout.fields['zeta']['g']
            psi = solver.state['psi']['g']
            im.set_data(zeta.T)
            im.set_clim(np.min(zeta), np.max(zeta))
            #maxzeta = np.max(np.abs(zeta))
            #im.set_clim(-maxzeta, maxzeta)
            plt.pause(0.01)
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

# Print statistics
#logger.info('Iterations: %i' %solver.iteration)
logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

