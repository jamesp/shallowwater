import matplotlib.pyplot as plt
import numpy as np

from linear2d import LinearShallowWater
from nonlinear import NonLinearShallowWater




nx = 128
ny = 129
beta=2.0e-11
Lx = 1.0e7
Ly = 1.0e7

ocean = NonLinearShallowWater(nx, ny, Lx, Ly, beta=beta, f0=0.0, g=0.1, H=100.0, dt=5000, nu=1000.0, bcond='periodicx')
#ocean.h[10:20, 60:80] = 1.0
#ocean.h[-20:-10] = 1.0
d = 25
#ocean.h[10:10+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
#ocean.h[100:100+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]
import matplotlib.pyplot as plt


q = np.zeros((nx, ny))
q[40:40+2*d, ny//2-d:ny//2+d] = (np.sin(np.linspace(0, np.pi, 2*d))**2)[np.newaxis, :] * (np.sin(np.linspace(0, np.pi, 2*d))**2)[:, np.newaxis]*0.1

ocean.add_tracer('q', q)

@ocean.add_forcing
def q_feedback(ocean):
    dstate = np.zeros_like(ocean.state)

    q = ocean.tracer('q')
    dstate[2] = q * 1e-6
    return dstate


plt.ion()

num_levels = 24
colorlevels = np.concatenate([np.linspace(-1, -.05, num_levels//2), np.linspace(.05, 1, num_levels//2)])

ts = []
es = []
plt.show()
for i in range(10000):
    ocean.step()
    if i % 50 == 0:
        plt.figure(1)
        plt.clf()
        #plt.plot(ocean.h[:,0])
        #plt.plot(ocean.h[:,64])
        #plt.ylim(-1,1)
        plt.contourf(ocean.h.T, cmap=plt.cm.RdBu, levels=colorlevels)

        plt.figure(2)
        plt.clf()
        plt.plot(ocean.h[:,0])
        plt.plot(ocean.h[:,48])
        plt.plot(ocean.h[:,64])
        plt.ylim(-1,1)

        plt.figure(3)
        plt.clf()
        energy = np.sum(ocean.g*ocean.h) + np.sum(ocean.u**2) + np.sum(ocean.v**2)
        ts.append(ocean.t)
        es.append(energy)
        plt.plot(ts, es)

        plt.figure(4)
        plt.clf()
        plt.contourf(ocean.tracer('q').T, cmap=plt.cm.RdBu, levels=colorlevels*0.1)



        plt.pause(0.01)
        plt.draw()

        rhs = ocean.rhs()
        nlrhs = ocean.nonlin_rhs()
        lrhs = rhs - nlrhs
        le = np.sum(lrhs[0]**2) + np.sum(lrhs[1]**2)
        nle = np.sum(nlrhs[0]**2) + np.sum(nlrhs[1]**2)

        print(le, nle)