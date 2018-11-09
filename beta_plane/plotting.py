import numpy as np
import matplotlib.pyplot as plt


def colourlevels(num_levels=24):
    """Return a list of levels from (-1, 1) that is centred around a slightly wider band at 0.
    Useful for `contourf` plots centred on zero."""
    return np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

def plot_wind_arrows(model, meshpoints, normalise=True, narrows=(12, 9), hide_below=0.1, **kwargs):
    plotargs = dict(pivot='mid', scale=20, width=0.002)
    plotargs.update(kwargs)
    u, v = model.uvath()
    vel = np.sqrt(u**2 + v**2)
    if normalise:
        velmax = vel.max()
        velnorm = vel / velmax
    else:
        velnorm = vel
        velmax = vel
    xspacing = model.nx // (narrows[0])
    yspacing = model.ny // (narrows[1])
    arrow_spacing = slice(yspacing//2, None, yspacing), slice(xspacing//2, None, xspacing)
    if meshpoints is None:
        x, y = np.meshgrid(model.phix/model.Lx, model.phiy/model.Ly)
    else:
        x, y = meshpoints
    plt.quiver(x[arrow_spacing], y[arrow_spacing],
            np.ma.masked_where(velnorm.T < hide_below, (u / np.sqrt(velmax**2 - v**2)).T)[arrow_spacing],
            np.ma.masked_where(velnorm.T < hide_below, (v / np.sqrt(velmax**2 - u**2)).T)[arrow_spacing], 
            **plotargs)