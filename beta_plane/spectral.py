import numpy as np

class NotifyArray(np.ndarray):
    """An array that notifies it's callback when values are changed."""
    def __new__(cls, input_array, callback=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.callback = callback
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.callback = getattr(obj, 'callback', None)

    def __setitem__(self, *args, **kwargs):
        try:
            suppress = kwargs.pop('suppress')
        except:
            suppress = False
        super(NotifyArray, self).__setitem__(*args, **kwargs)
        if not suppress:
            self.callback()

class SpectralField(object):
    def __init__(self, nx, ny):
        self.nk = nk = nx // 2 + 1
        self.nl = nl = ny
        self._real = NotifyArray(np.zeros((ny, nx), dtype=np.float64), callback=self._real_changed)
        self._spectral = NotifyArray(np.zeros((nl, nk), dtype=np.complex128), callback=self._spec_changed)
        self.k = np.fft.rfftfreq(nx)
        self.l = np.fft.fftfreq(ny)
        self.K2 = self.k[np.newaxis, :]**2 + self.l[:, np.newaxis]**2

    def _real_changed(self):
        print('real changed, updating spectral')
        spec = np.fft.rfft2(self._real)
        self._spectral.__setitem__(slice(None, None), spec, suppress=True)

    def _spec_changed(self):
        print('spec changed, updating real')
        real = np.fft.irfft2(self._spectral)
        self._real.__setitem__(slice(None,None), real, suppress=True)

    @classmethod
    def from_array(cls, spec=None, real=None):
        if real is None:
            real = np.fft.irfft2(spec)
        if spec is None:
            spec = np.fft.rfft2(real)
        ny, nx = real.shape
        obj = cls(nx, ny)
        obj._spectral.__setitem__(slice(None, None), spec, suppress=True)
        obj._real.__setitem__(slice(None,None), real, suppress=True)
        return obj

    @property
    def t(self):
        return self._spectral[:, :]

    def __getitem__(self, *args, **kwargs):
        return self._real.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self._real.__setitem__(*args, **kwargs)

    def grad(self):
        """Return the gradient of the field."""
        diffx = 1j*self.k[np.newaxis, :]*self.t
        diffy = 1j*self.l[:, np.newaxis]*self.t
        print(np.fft.irfft2(diffx).shape)
        return (SpectralField.from_array(spec=diffx),
            SpectralField.from_array(spec=diffy))

if __name__ == '__main__':
    nx = 128
    ny = 256

    b = np.random.random((ny, nx))

    f = SpectralField(nx, ny)
    f[:] = b

    bt = np.fft.rfft2(b)

    print(f.t - bt)
    print(f.K2.shape)
    print(f.t.shape)

    print(f.grad())

