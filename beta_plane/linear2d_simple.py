#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
"""Shallow Water Model
  
- Two dimensional shallow water in a rotating frame
- Staggered Arakawa-C lat:lon grid
- periodic in the x-dimension
- fixed boundary conditions in the y-dimension
 
h = H + η
 
∂/∂t[u] - fv = - g ∂/∂x[η]
∂/∂t[v] + fu = - g ∂/∂y[η]
∂/∂t[h] + H(∂/∂x[u] + ∂/∂y[v]) = 0
 
f = f0 + βy
"""

from __future__ import print_function
import numpy as np

nx = 128
ny = 129

Lx = 1.5e7
Ly = 1.5e7

u = np.zeros((nx+3, ny+2))
v = np.zeros((nx+2, ny+3))
h = np.zeros((nx+2, ny+2))

dx = Lx / nx
dy = Ly / ny

# positions of the nodes
ux = (-Lx/2 + np.arange(nx+1)*dx)[:, np.newaxis]
vx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]

vy = (-Ly/2 + np.arange(ny+1)*dy)[np.newaxis, :]
uy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

hx = vx
hy = uy


f0 = 0.0
beta = 2.3e-11
nu = 5.0e-7             # diffusion
r = 1.0e-5              # Rayleigh damping at top and bottom of domain
g = 0.05
H = 100.0

dt = 3000.0
t = 0.0
tc = 0




sponge_ny = ny//5
sponge = np.exp(-np.linspace(0, 3, sponge_ny))
def damping(var, r=r):
    # sponges are active at the top and bottom of the domain by applying Rayleigh friction
    # with exponential decay towards the centre of the domain
    global sponge, sponge_ny
    var_sponge = np.zeros_like(var)
    var_sponge[:, :sponge_ny] = sponge[np.newaxis, :]
    var_sponge[:, -sponge_ny:] = sponge[np.newaxis, ::-1]
    return r*var_sponge*var


def _update_boundaries(*vars):
    u, v, h = vars[0:3]

    # solid walls left and right
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]
    h[0, :] = h[1, :]
    h[-1, :] = h[-2, :]

    for var in vars:
        # zero deriv y condition
        var[:, 0] = var[:, 1]
        var[:, -1] = var[:, -2]

    	# fix corners to be average of neighbours
        var[0, 0] =  0.5*(var[1, 0] + var[0, 1])
        var[-1, 0] = 0.5*(var[-2, 0] + var[-1, 1])
        var[0, -1] = 0.5*(var[1, -1] + var[0, -2])
        var[-1, -1] = 0.5*(var[-1, -2] + var[-2, -1])


def diffx(psi):
    """Calculate ∂/∂x[psi] over a single grid square.
     
    i.e. d/dx(psi)[i,j] = (psi[i+1/2, j] - psi[i-1/2, j]) / dx
     
    The derivative is returned at x points at the midpoint between
    x points of the input array."""
    global dx
    return (psi[1:,:] - psi[:-1,:]) / dx

def diff2x(psi):
    """Calculate ∂2/∂x2[psi] over a single grid square.
     
    i.e. d2/dx2(psi)[i,j] = (psi[i+1, j] - psi[i, j] + psi[i-1, j]) / dx^2
     
    The derivative is returned at the same x points as the
    x points of the input array, with dimension (nx-2, ny)."""
    global dx
    return (psi[:-2, :] - psi[1:-1, :] + psi[2:, :]) / dx**2

def diff2y(psi):
    """Calculate ∂2/∂y2[psi] over a single grid square.
     
    i.e. d2/dy2(psi)[i,j] = (psi[i, j+1] - psi[i, j] + psi[i, j-1]) / dy^2
     
    The derivative is returned at the same y points as the
    y points of the input array, with dimension (nx, ny-2)."""
    global dy
    return (psi[:, :-2] - psi[:, 1:-1] + psi[:, 2:]) / dy**2

def diffy(psi):
    """Calculate ∂/∂y[psi] over a single grid square.
     
    i.e. d/dy(psi)[i,j] = (psi[i, j+1/2] - psi[i, j-1/2]) / dy
     
    The derivative is returned at y points at the midpoint between
    y points of the input array."""
    global dy
    return (psi[:, 1:] - psi[:,:-1]) / dy

def centre_average(phi):
    """Returns the four-point average at the centres between grid points."""
    return 0.25*(phi[:-1,:-1] + phi[:-1,1:] + phi[1:, :-1] + phi[1:,1:])
 
def y_average(phi):
    """Average adjacent values in the y dimension.
    If phi has shape (nx, ny), returns an array of shape (nx, ny - 1)."""
    return 0.5*(phi[:,:-1] + phi[:,1:])
 
def x_average(phi):
    """Average adjacent values in the x dimension.
    If phi has shape (nx, ny), returns an array of shape (nx - 1, ny)."""
    return 0.5*(phi[:-1,:] + phi[1:,:])

def divergence(u, v):
    """Returns the horizontal divergence at h points."""
    return diffx(u) + diffy(v)

def del2(phi):
    """Returns the Laplacian of phi."""
    return diff2x(phi)[:, 1:-1] + diff2y(phi)[1:-1, :]

def uvatuv(u, v):
    """Calculate the value of u at v and v at u."""
    ubar = centre_average(u)[1:-1, :]
    vbar = centre_average(v)[:, 1:-1]
    return ubar, vbar

def uvath(u, v):
    ubar = x_average(u)
    vbar = y_average(v)
    return ubar, vbar


def rhs(state):
    """Calculate the right hand side of the u, v and h equations."""
    global f0, g, H
    u, v, h = state[0:3]
    uu, vv = uvatuv(u, v)
    
    u_rhs = np.zeros_like(u)
    v_rhs = np.zeros_like(v)
    h_rhs = np.zeros_like(h)

    # the height equation
    h_rhs[1:-1, 1:-1] = -H*(divergence(u, v))[1:-1, 1:-1] + nu*del2(h) - damping(h)[1:-1, 1:-1]

    # the u equation
    dhdx = diffx(h)[:, 1:-1]
    u_rhs[1:-1, 1:-1] = f0*vv + beta*uy*vv - g*dhdx + nu*del2(u) - damping(u)[1:-1, 1:-1]
     
    # the v equation
    dhdy = diffy(h)[1:-1, :]
    v_rhs[1:-1, 1:-1] = -f0*uu - beta*vy*uu - g*dhdy + nu*del2(v) - damping(v)[1:-1, 1:-1]

    return np.array([u_rhs, v_rhs, h_rhs])


_ppdstate, _pdstate = 0,0
def step(state):
    global dt, t, tc, _ppdstate, _pdstate

    _update_boundaries(*state)
    
    dstate = rhs(state)

    # take adams-bashforth step in time
    if tc==0:
        # forward euler
        dt1 = dt
        dt2 = 0.0
        dt3 = 0.0
    elif tc==1:
        # AB2 at step 2
        dt1 = 1.5*dt
        dt2 = -0.5*dt
        dt3 = 0.0
    else:
        # AB3 from step 3 on
        dt1 = 23./12.*dt
        dt2 = -16./12.*dt
        dt3 = 5./12.*dt
    
    newstate = state + dt1*dstate + dt2*_pdstate + dt3*_ppdstate
    state = newstate
    _ppdstate = _pdstate
    _pdstate = dstate

    t  += dt
    tc += 1
    return state

import matplotlib.pyplot as plt

timestamps = []
u_snapshot = []

plt.figure(figsize=(12, 12))
def plot_all(u,v,h):
    global timestamps, u_snapshot
    timestamps.append(t)
    u_snapshot.append(state[0][:, ny//2])  # add equatorial zonal velocity to u_snapshot
    if len(timestamps) % 2 == 0:
        plt.clf()
        plt.subplot(221)
        plt.imshow(u[1:-1, 1:-1].T, cmap=plt.cm.YlGnBu,
                extent=np.array([ux.min(), ux.max(), uy.min(), uy.max()])/1000)
        plt.clim(-np.abs(u).max(), np.abs(u).max())
        plt.title('u')

        plt.subplot(222)
        plt.imshow(v[1:-1, 1:-1].T, cmap=plt.cm.YlGnBu,
                extent=np.array([vx.min(), vx.max(), vy.min(), vy.max()])/1000)
        plt.clim(-np.abs(v).max(), np.abs(v).max())
        plt.title('v')

        plt.subplot(223)
        plt.imshow(h[1:-1, 1:-1].T, cmap=plt.cm.RdBu,
                 extent=np.array([hx.min(), hx.max(), hy.min(), hy.max()])/1000)
        #plt.pcolormesh(h.T, cmap=plt.cm.RdBu)
        #plt.xlim(-10, nx+10)
        #plt.ylim(-10, ny+10)
        plt.clim(-np.abs(h).max(), np.abs(h).max())
        plt.title('h')

        
        plt.subplot(224)
        c = np.sqrt(g*H)
        power = np.log(np.abs(np.fft.fft2(np.array(u_snapshot))**2))

        khat = np.fft.fftshift(np.fft.fftfreq(power.shape[1], 1.0/nx))
        k = khat / Ly

        omega = np.fft.fftshift(np.fft.fftfreq(power.shape[0], dt))
        w = omega / np.sqrt(beta*c)

        plt.pcolormesh(khat, w, np.fft.fftshift(power)[::-1], cmap=plt.cm.RdBu)
        plt.ylim(0, 20)
        plt.xlim(-nx//2, nx//2)
        plt.title('dispersion')
        plt.xlabel('k')
        plt.ylabel('omega')
        plt.pause(0.01)
    


# create a single disturbance in the middle of the domain
# with amplitude 0.01*H
h[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = np.exp(-(((np.indices((10,10)) - 5)/2.0)**2).sum(axis=0))*H*0.01

state = np.array([u, v, h])



for i in range(100000):
    state = step(state)
    if i % 50 == 0:
        
        plot_all(*state)
        print('[t={:7.2f} h range [{:.2f}, {:.2f}]'.format(t/86400, state[2].min(), state[2].max()))

