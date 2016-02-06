import numpy as np
import matplotlib.pyplot as plt

from linear import LinearShallowWater, WalledBoundaries

class Shallow(WalledBoundaries, LinearShallowWater): pass

nx = 128
ny = 129
Lx = 1.5e7
Ly = 1.5e7
beta = 2.28e-11
phi0 = 100.0
g = 10.0
c = 30.0            # Kelvin wave speed c = sqrt(gH)
H = c**2 / g        # Set the height to get right speed waves
nu = 1.0e3

dx = float(Lx) / nx
dt = 0.8 * dx / (c*4)


import contextlib

plt.ion()

@contextlib.contextmanager
def iplot(fignum=1):
    plt.figure(fignum)
    plt.clf()
    yield
    plt.pause(0.001)
    plt.draw()



class ShallowWater(LinearShallowWater, PeriodicBoundaries, AdvectiveTracers): pass

sw = ShallowWater(nx, ny, Lx, Ly, beta=beta, g=g, H=H, nu=nu, dt=dt)

plt.figure(1, figsize=(8,8))
for i in range(10000):
    sw.step()
    if i % 10 == 0:
        with iplot(1):
            plt.subplot(221)
            plt.imshow(sw.h.T, cmap=plt.cm.RdBu_r)
            plt.clim(-.2, .2)
            plt.title('h')

            plt.subplot(222)
            plt.imshow(sw.tracer('q').T, cmap=plt.cm.RdBu_r)
            plt.title('q')

            plt.subplot(223)
            div = sw.divergence()
            maxdiv = np.max(np.abs(div))
            plt.imshow(div.T, cmap=plt.cm.RdBu_r)
            plt.clim(-maxdiv, maxdiv)
            plt.title('divergence')

            plt.subplot(224)
            plt.plot(sw.H+sw.h[:, ny//2])