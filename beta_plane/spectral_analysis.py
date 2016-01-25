# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal


def best_fit(xs, ys):
    """Using the method of least squares, return the gradient
    and y-intercept of the line of best fit through xs and ys."""
    A = np.array([xs, np.ones(len(xs))])
    return np.linalg.lstsq(A.T,ys)[0]


# Wheeler-Kiladis plot
# For symmetric responses about the equator plot the odd modes m = 1,3,5,7...
# For antisymmetric responses about the equator and
# the mixed-Rossby gravity waves plot the even modes m = 0,2,4,6...


def best_fit(xs, ys):
    """Using the method of least squares, return the gradient
    and y-intercept of the line of best fit through xs and ys."""
    A = np.array([xs, np.ones(len(xs))])
    return np.linalg.lstsq(A.T,ys)[0]


def kiladis_spectra(u, dt=1.0, dx=1.0):
    """Perform Wheeler-Kiladis Spectral Analysis on variable u.

        spinup: discard the first `spinup` days as initialisation

    Returns frequency-wavenumber spectra for each latitude.
    """
    v = np.asarray(u)
    nt, nx, ny = v.shape

    ts = np.arange(nt)*dt
    xs = np.arange(nx)*dx

    fts = []
    for j in range(ny):
        data = v[:,:,j]                    # u in time and longitude at given latitude
        lng_avg = data.mean(axis=1)        # average u at each timestep
        m, c = best_fit(ts, lng_avg)       # trend in time

        # remove the trend of u over time to leave perturbations centred on zero
        perturbations = data - (m*ts + c)[:, np.newaxis]

        # window tapering - make the ends of the time window approach zero
        #                 - use a cos^2 profile over a small number of samples at each end
        taper = 30
        perturbations[:taper,:] = perturbations[:taper,:] * (np.cos(np.linspace(-np.pi/2, 0, taper))**2)[:, np.newaxis]
        perturbations[-taper:,:] = perturbations[-taper:,:] * (np.cos(np.linspace(0, np.pi/2, taper))**2)[:, np.newaxis]

        lft = np.fft.fft(perturbations, axis=1)     # FFT in space
        tft = np.fft.fft(lft, axis=0)               # FFT in time
        fts.append(tft)
    fts = np.array(fts)
    # fourier transform in numpy is defined by exp(-2pi i (kx + wt))
    # but we want exp(kx - wt) so need to negate the x-domain
    fts = fts[:, :, ::-1]
    return fts


def background(spectra, fsteps=10, ksteps=10):
    """Uses a 1-2-1 filter to generate 'red noise' background field for a spectra (as per WK1998)
        `fsteps` is the number of times to apply the filter in the frequency direction
        `ksteps` is the number of times to apply the filter in the wavenumber direction

    Returns a background field of same dimensions as `spectra`.
    """
    # create a 1D 1-2-1 averaging footprint
    bgf = spectra
    for i in range(fsteps):
        # repeated application of the 1-2-1 blur filter to the spectra
        footprint = np.array([[0,1,0], [0,2,0], [0,1,0]]) / 4.0
        bgf = scipy.signal.convolve2d(bgf, footprint, mode='same', boundary='wrap')
    for i in range(ksteps):
        # repeated application of the 1-2-1 blur filter to the spectra
        footprint = np.array([[0,0,0], [1,2,1], [0,0,0]]) / 4.0
        bgf = scipy.signal.convolve2d(bgf, footprint, mode='same', boundary='wrap')

    return bgf
def remove_background(spectra):
    """A simple background removal to eliminate frequency noise."""
    bg = background(spectra, fsteps=10, ksteps=0)
    return spectra - bg


# this can be plotted over a spectra
def plot_wavelines(c, m=(1,3,5,7), kelvin=True, color=None, linestyle='--'):
    """Plot the analytic solutions to Kelvin, Rossby and gravity waves
    on a wavenumber-frequency spectrum.
        `c` is wavespeed in [m.s^-1]
        `m` are the eigenvalues of the Rossby and gravity wave solutions to be plotted
    Plots lines in SI units"""
    T = 5.0*oneday                                    # number of seconds over which time window is observed

    nx = 1000                                         # number of sample points in space
    nt = 1000                                         # temporal sample points

    khat = np.linspace(-nx/2, nx/2, 2000000)          # nondimensional wavenumber
    what = np.linspace(-nt/2, nt/2, 2000000)          # nondimensional frequency

    k = khat / rearth                                 # in [wavelengths.m^-1]
    w = what * pi2 / T                                # in [wavelengths.s^-1]

    # c = np.sqrt(phiref)     # wavespeed in [m.s^-1]
    beta = twoomega/rearth  # in units [s^-1.m^-1]

    if color is None:
        kcolor, gcolor, rcolor, ycolor = ('blue', 'black', 'green', 'red')
    else:
        try:
            kcolor, gcolor, rcolor, ycolor = color
        except:  # not a list of 4 colors, set all 4 to the same value
            kcolor = gcolor = rcolor = ycolor = color
    ls = linestyle

    # plot analytic solutions to equatorial waves
    # a. kelvin waves
    #   w = ck
    if kelvin:
        kline, = plt.plot(k, c*k, color=kcolor, linestyle=ls)
        kline.set_label('Kelvin Wave')

    # b. rossby waves
    # c. gravity waves
    #   w^2 - c^2 k^2 - \beta c^2 k / w = (2m+1) \beta c
    # The dispersion relation is quadratic in k and cubic in w
    # so solve for k and plot that way
    # plot several modes of m
    for mi in m:
        gkp = -(beta / (2*w)) + 0.5*np.sqrt((beta/w - 2*w/c)**2 - 8*mi*beta/c)
        gkm = -(beta / (2*w)) - 0.5*np.sqrt((beta/w - 2*w/c)**2 - 8*mi*beta/c)
        if mi != 0:
            # Gravity waves: high frequency
            gline, = plt.plot(gkp[w > 0.00001], w[w > 0.00001], color=gcolor, linestyle=ls)
            plt.plot(gkm[w > 0.00001], w[w > 0.00001], color=gcolor, linestyle=ls)
            # Rossby waves: low frequency
            rline, = plt.plot(gkp[w < 0.00001], w[w < 0.00001], color=rcolor, linestyle=ls)
            plt.plot(gkm[w < 0.00001], w[w < 0.00001], color=rcolor, linestyle=ls)
        else:
            # d. Yanai Wave: when m = 0 only one solution is physically relevant
            # w = kc/2 ± 1/2 sqrt(k^2 c^2 + 4 \beta c)
            yline, = plt.plot(k, k*c/2 + 0.5*np.sqrt(k**2*c**2 + 4*beta*c), color=ycolor, linestyle=ls)
            yline.set_label('Yanai Wave')

    gline.set_label('Gravity Waves')
    rline.set_label('Rossby Waves')


def axis_cycles_per_day():
    non_dim_xticks = np.linspace(-100, 100, 21, dtype=np.int32)
    dim_xticks = non_dim_xticks / rearth
    non_dim_yticks = np.linspace(0, 1, 11)
    dim_yticks = non_dim_yticks/oneday*pi2
    plt.xticks(dim_xticks, non_dim_xticks)
    plt.yticks(dim_yticks, non_dim_yticks)
    plt.xlabel('Zonal Wavenumber $k$')
    plt.ylabel('Frequency $\omega$ (cycles per day)')


def axis_non_dim(c=10.0):
    beta = twoomega/rearth  # in units [s^-1.m^-1]
    non_dim_xticks = np.linspace(-10, 10, 11, dtype=np.int32)
    dim_xticks = non_dim_xticks / np.sqrt(c/beta)
    non_dim_yticks = np.linspace(0, 4, 5)
    dim_yticks = non_dim_yticks * np.sqrt(beta*c)
    plt.xticks(dim_xticks, non_dim_xticks)
    plt.yticks(dim_yticks, non_dim_yticks)
    plt.xlabel(r'Zonal Wavenumber $k \sqrt{c/\beta}$')
    plt.ylabel(r'Frequency $\omega / \sqrt{\beta c}$')