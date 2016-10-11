import matplotlib.pyplot as plt
import numpy as np

N = 128
Lx, Ly = (1., 1.)
nx, ny = (N, N)

dx = dy = Lx/nx

k = np.fft.fftfreq(N, 1/N)[:, np.newaxis]
l = np.fft.rfftfreq(N, 1/N)[np.newaxis, :]
ksq = k**2 + l**2
# initial_spectra = np.random.random(ksq.shape) + 1j*np.random.random(ksq.shape)
# initial_spectra[ksq > 12**2] = 0
# initial_spectra[ksq < 10**2] = 0

# initial_grid = np.fft.irfft2(initial_spectra)

# initial_grid = initial_grid - initial_grid.mean()

def spectral_variance(phit, n=N):
    var_density = 2.0 * np.abs(phit)**2 / (n*n)
    var_density[:,0] /= 2
    var_density[:,-1] /= 2
    return var_density.sum()

wvx = np.sqrt((k*dx)**2 + (l*dy)**2)
spectral_filter = np.exp(-23.6*(wvx-0.65*np.pi)**4)
spectral_filter[wvx <= 0.65*np.pi] = 1.0

ck = np.zeros_like(ksq)
ck = np.sqrt(ksq + (1.0 + (ksq/36.0)**2))**-1
piit = np.random.randn(*ksq.shape)*ck + 1j*np.random.randn(*ksq.shape)*ck

pii = np.fft.irfft2(piit)
pii = pii - pii.mean()
piit = np.fft.rfft2(pii)
KE = spectral_variance(piit*np.sqrt(ksq)*spectral_filter)

qit = -ksq * piit / np.sqrt(KE)
qi = np.fft.irfft2(qit)

plt.imshow(np.real(qi))
plt.show()

plt.imshow(np.real(initial_grid))
plt.show()

