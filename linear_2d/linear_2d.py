#!/usr/bin/env bash
# -*- coding: utf-8 -*-
 
"""Shallow Water Model
 
- Two dimensional shallow water in a rotating frame
- Staggered Arakawa-C lat:lon grid
- periodic in the x-dimension
- fixed boundary conditions in the y-dimension
 
"""
 
import numpy as np
from numpy import mod, floor, ceil

# Constants
# =========
 
pi = np.pi
twopi = 2.0 * pi
piby2 = pi / 2.0
 
rearth = 6371220.0          # radius of earth 
twoomega = 1.4584e-4        # rotation rate
gravity = 9.80616           # gravitational force
 
# Model Configuration
# ===================
 
p = 6                       # grid scale
nx = 5*2**p
ny = 5*2**(p-1)
 
dt = 900.0                  # timestep in seconds
numdays = 600               # total length of simulation
 
phiref = 100.0              # refrence phi value for solver
                            # (should be similar to mean phi)
 
 
# Grid Setup
# ==========
 
dx = 2*np.pi/nx
dy = np.pi/ny
hny = ny / 2.0
hnx = nx / 2.0
 
i = np.arange(nx)           # index for lons
j = np.arange(ny)           # index for lats
 
xp = (i+0.5)*dx             # lon of phi points
xv = (i+0.5)*dx             # lon of v points
xu = i*dx                   # lon of u points
 
yp = (j+0.5-hny)*dy         # lat of phi points
yu = (j+0.5-hny)*dy         # lon of u points    
yv = (j-hny)*dy             # lon of v points

def _periodicl(psi):
    """Copy the last column to in front of the array to make zonally periodic."""
    return np.concatenate((psi[-1, np.newaxis], psi))

def _periodicr(psi):
    """Copy the first column to after the array to make zonally periodic."""
    return np.concatenate((psi, psi[0, np.newaxis]))

def _withbcs(psi, lower_bc, upper_bc):
	"""Add the upper and lower boundary conditions to the edge of the field."""
	return np.vstack()
 

def halfr(psi):
    """Take a half-step right.
    i.e. psi[i+1/2, j] = 0.5*(psi[i,j] + psi[i+1, j])
    wraps around the x-domain."""
    pr = _periodicr(psi)  # make periodic
    return 0.5*(pr[:-1,:] + pr[1:,:])
 
def halfl(psi):
    """Take a half-step left.
    i.e. psi[i, j] = 0.5*(psi[i-1/2,j] + psi[i+1/2, j])
    wraps around the x-domain."""
    pl = _periodicl(psi)  # make periodic
    return 0.5*(pl[:-1,:] + pl[1:,:])
 
def halfup(psi):
    """Take a half-step up.
    i.e. psi[i, j+1/2] = 0.5*(psi[i, j] + psi[i, j+1/2]).
    As the meridional direction does not wrap, taking a half step reduces the meridional size by 1.
    Returns a grid one latitudinal level less than psi."""
    return 0.5*(psi[:,:-1] + psi[:,1:])
 
 
### Differentiation Functions
def xdiffl(psi, dx):
    """Central difference in x over one grid square. 
    i.e. d/dy(psi)[i,j] = (psi[i-1/2, j] - psi[i+1/2, j]) / dx
 
    `dx` should be the same dimension as latitude.
    """
    pl = _periodicl(psi)
    return (pl[1:,:] - pl[:-1,:]) / dx[np.newaxis, :]
 
def ydiff(psi, dy):
    """Central difference in y over one grid square. 
    i.e. d/dy(psi)[i,j] = (psi[i, j+1/2] - psi[i, j-1/2]) / dy
    Returns a grid one latitudinal level less than psi.
    """
    return (psi[:, 1:] - psi[:,:-1]) / dy



def coriolis(u, v, phi):
    """Calculates the Coriolis terms of the momentum equation.
     
    Returns (fu, fv) : ((nx x ny+1), (nx, ny))"""
    f = twoomega * sinp[np.newaxis, :]  # value of Coriolis param f at phi points
 
    # == Calculating fv at u points ==
    tempv = np.zeros_like(v)
    tempv[:, 1:-1] = halfup(phi)*v[:, 1:-1]*cosv[np.newaxis, 1:-1]
    # average to phi points and multiply by f / phi
    tempp = halfup(tempv) * f / phi
    fv = halfl(tempp) / cosp[np.newaxis, :]
 
    # == Calculating fu at v points ==
    tempu = halfl(phi)*u
    tempp = halfr(tempu) * f / phi
    fu = np.zeros_like(v)
    fu[:, 1:-1] = halfup(tempp)
     
    return (fu, fv)

def uvatuv(u, v):
    """Calculate the value of u at v and v at u."""
    unp, vnp, usp, vsp = polar(u, v)
    v[:, 0] = vsp
    v[:, -1] = vnp
 
    ur = _periodicr(u)
    ubar = np.zeros_like(v)
    ubar[:, 1:-1] = 0.25*(ur[:-1,:-1] + ur[:-1,1:] + ur[1:, :-1] + ur[1:,1:])
    ubar[:, 0] = usp
    ubar[:, -1] = unp
 
    vl = _periodicl(v)
    vbar = 0.25*(vl[:-1,:-1] + vl[:-1,1:] + vl[1:,:-1] + vl[1:,1:])
    return ubar, vbar
 
def uvatphi(u, v):
    """Calculate the value of u and v at phi points."""
    pu = halfr(u)
    pv = halfup(v)
    return pu, pv

