import numpy as np
import matplotlib.pyplot as plt

def plot_wind_arrows(model, meshpoints, normalise=True, narrows=(12, 9), hide_below=0.1):
    u, v = model.uvath()
    vel = np.sqrt(u**2 + v**2)
    if normalise:
        velmax = vel.max()
        velnorm = vel / velmax
    else:
        velnorm = vel
    xspacing = model.nx // (narrows[0])
    yspacing = model.ny // (narrows[1])
    arrow_spacing = slice(yspacing//2, None, yspacing), slice(xspacing//2, None, xspacing)
    if meshpoints is None:
        x, y = np.meshgrid(model.phix/model.Lx, model.phiy/model.Ly)
    else:
        x, y = meshpoints
    plt.quiver(x[arrow_spacing], y[arrow_spacing],
            np.ma.masked_where(velnorm.T < hide_below, u.T / velmax)[arrow_spacing],
            np.ma.masked_where(velnorm.T < hide_below, v.T / velmax)[arrow_spacing], pivot='mid', scale=20, width=0.002)