#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numerics for one-dimensional PDEs."""

import numpy as np


def tridiag(n, d, u, l):
    """Create an nxn tridiagonal matrix with `d` along diagonal, `u` on
    first upper diag and `l` on first lower"""
    M = np.zeros((n,n))
    i,j = np.indices((n,n))
    M[i == j] = d
    M[i == j-1] = u
    M[i == j+1] = l
    return M

def stencil(n, _i, i, i_):
    """Create a tridiagonal stencil matrix of size n.
    Creates a matrix to dot with a vector for performing discrete spatial computations.
    i, i_ and _i are multipliers of the ith, i+1 and i-1 values of the vector respectively.
    e.g. to calculate an average at position i based on neighbouring values:
    >>> s = stencil(N, 0, 0.5, 0.5)
    >>> avg_v = np.dot(s, v)
    The stencil has periodic boundaries.

    Returns an nxn matrix.
    """
    m = tridiag(n, i, i_, _i)
    m[-1,0] = i_
    m[0,-1] = _i
    return m

def RAW_filter(_phi, phi, phi_, nu=0.1, alpha=0.53):
    """The RAW time filter, an improvement on RA filter.

    phi: A tuple of phi at time levels (n-1), n, (n+1)

    nu: Equivalent to 2*ϵ; the RA filter weighting

    alpha: Scaling factor for n and (n+1) timesteps.
           With α=1, RAW —> RA.

    For more information, see [Williams 2009].
    """
    d = nu*0.5*(_phi - 2.0*phi + phi_)
    return (_phi, phi+alpha*d, phi_ + (alpha-1)*d)
