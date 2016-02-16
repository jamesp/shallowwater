from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Shallow Water',
  ext_modules = cythonize("arakawac.pyx"),
)
