#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The one-dimensional shallow water Equations with convection simulation."""

import numpy as np

from numerics import RAW_filter, stencil


### Configuration
SHOW_ANIMATION = True
LOG_VARS = False
N_STEPS_LOG_OUTPUT = 60
N_STEPS_CHART_REFRESH = 60
RANDOM_SEED = 1234



### Model Parameters
X = 500000.0
T = 100000
dx = 500.0
dt = 1.0

H0 = 90.0
Hc = 90.02
Hr = 90.4

g = 10.0
phi_c = 89.977 * g

beta = 1/150.0
alpha = 2.5e-4
c2 = H0*g  # c^2

ubar = 0.005
l = 2000.0

F_rate = 1.6e-6

Ku = 30000.0 / 4.0
Kh = 30000.0 / 4.0
Kr = 200.0 / 4.0



### Global Variables
rdm = np.random.RandomState(RANDOM_SEED)
x = np.arange(0, X, dx, dtype=np.float32)
N = len(x)

def seconds_to_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def log(msg):
    print msg



### Discrete derivative functions
# these comprise of:
# - a NxN stencil matrix
# - a dot product of the matrix with a N element vector
#   of the of the discretised value of f(x)
# The stencil implements periodic boundaries.

def stencils(N):
    """Returns a function that make stencils of a given size."""
    saved_stencils = {}  # memoize previously used stencils
    def get_stencil(_p, p, p_):
        key = (N, _p, p, p_)
        if key in saved_stencils:
            return saved_stencils[key]
        else:
            s = stencil(N, _p, p, p_)
            saved_stencils[key] = s
            return s
    return get_stencil

_S = stencils(N)
def S(phi, _s, s, s_):
    """Apply a stencil (_s, s, s_) to variable phi.
    (_s, s, s_) are the coefficients of phi_{j-1, j, j+1} respectively.
    e.g. Diffusion (2nd central-difference) can be achieved by:
    >>> S(phi, 1, -2, 1) / dx
    """
    return np.dot(_S(_s, s, s_), phi)







### Craig-Würsch Equations

def phi(H, h, Hc=Hc):
    """Modified geopotential.

    When H + h > Hc the geopotential is set to an
    artificially reduced value."""
    z = H + h
    return np.where(z > Hc, phi_c + g*H, g*z)

def Fn(x, xn, ubar=ubar, l=l):
    """Perturbation function.
    Added to the velocity field at random locations."""
    #p = dfdx(np.exp(-(x-x[xn])**2 / l**2))
    sig = l/dx
    p = S(1.0/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-(x[xn]-dx/2))/l)**2), -1, 0, 1) / (2.0*dx)
    p_norm = p / np.max(p)  # normalise the disturbance ~ 1.0
    return ubar*p_norm

def perturb(x, rate, dt, X=X, ubar=ubar, l=l):
    """Create perturbations at given rate over the domain x."""
    p = np.zeros_like(x)
    for _ in range(rdm.poisson(rate*X*dt, 1)):
        p = p + Fn(x, rdm.randint(0, N), ubar, l)
    return p

### Set initial conditions
t  = 0.0
u = _u = u0 = np.zeros_like(x)
h = _h = h0 = np.ones_like(x) * H0
r = _r = r0 = np.zeros_like(x)
H  = np.zeros_like(x)  # ground topology


### Setup Plotting
if SHOW_ANIMATION:
    import matplotlib.pyplot as plt
    plt.ion()

    fig, ax = plt.subplots(3)

    for i,q in enumerate(['fluid velocity [$m.s^{-1}$]', 'fluid height [m]', 'rain (mass content) [$10^2$]']):
        ax[i].set_xlabel('x')
        ax[i].set_ylabel(q)

    uline, = ax[0].plot(x, u0)
    hline, = ax[1].plot(x, H+h0)
    rline, = ax[2].plot(x, r0)

    ax[0].set_ylim((-ubar*8, ubar*8))
    ax[0].plot(x, np.zeros_like(x), '--k')
    ax[1].plot(x, np.ones_like(x)*Hc, '--')
    ax[1].plot(x, np.ones_like(x)*Hr, '--')
    ax[1].set_ylim((H0-0.1, Hr+0.2))
    ax[2].set_ylim(0, 0.2)
    t_label = ax[0].text(0, ubar*9, seconds_to_time(t))
    #
    # fig = plt.figure(1, figsize=(14, 6))
    # ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    # ax2 = plt.subplot2grid((3,1), (2,0))
    #
    # uline, = ax2.plot(x, u0, color='0.5')
    # hline, = ax1.plot(x, H+h0, color='0.0')
    # rline, = ax2.plot(x, r0*c2-ubar*8)
    #
    # ax2.set_ylim((-ubar*8, ubar*8))
    # ax2.plot(x, np.zeros_like(x), '--k')
    # ax1.plot(x, np.ones_like(x)*Hc, '--')
    # ax1.plot(x, np.ones_like(x)*Hr, '--')
    # ax1.set_ylim((H0-0.1, Hr+0.2))
    #
    # t_label = ax1.text(0, Hr+0.5, seconds_to_time(t))



for t in np.arange(0+dt, T, dt):
    ### Integrate forward with Leapfrog method
    u = u + perturb(x, F_rate, dt)

    u_ = _u - (dt/(2.0*dx))*S(u**2, -1, 0, 1) - (2.0*dt/dx)*S(phi(H, h) + c2*r, -1, 1, 0) + (Ku*dt/(dx*dx))*S(_u, 1, -2, 1)

    momentum = u*S(h, 1, 1, 0)
    h_ = _h - (dt/dx)*S(momentum, 0, -1, 1) + (Kh*dt/(dx*dx))*S(_h, 1, -2, 1)

    z = H + h
    ux = S(u, 0, -1, 1)
    beta_a = np.where((z > Hr) & (ux < 0), beta, 0)
    r_ = _r - alpha*dt*2.0*r - 2*beta_a*(dt/dx)*ux - (dt/(2.0*dx))*S(u, 0, 1, 1)*S(r, -1, 0, 1) + (Kr*dt/(dx*dx))*S(_r, 1, -2, 1)

    # Use the RAW filter to update the timestep variables
    _, _u, u = RAW_filter(_u, u, u_)
    _, _h, h = RAW_filter(_h, h, h_)
    _, _r, r = RAW_filter(_r, r, r_)

    if LOG_VARS:

        if (t % (dt*N_STEPS_LOG_OUTPUT)) == 0:
            log('------------------')
            log('Time step: %s' % seconds_to_time(t))
            print 'u', np.mean(u), np.min(u), np.max(u)

    if SHOW_ANIMATION:
        if (t % (dt*N_STEPS_CHART_REFRESH)) == 0:
            #plot_lines(x, u, h, H, r)
            # update the plots
            uline.set_ydata(u)
            hline.set_ydata(H+h)
            rline.set_ydata(r*c2-ubar*8)
            t_label.set_text(seconds_to_time(t))
            plt.pause(0.001)
