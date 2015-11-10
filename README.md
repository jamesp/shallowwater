# shallowwater
Shallow Water Equations in Python

### Barotropic Vorticity
A spectral code for solving the Barotropic vorticity equation

D/Dt[ω] = 0

where ω = ξ + f.  ξ is local vorticity ∇ × u and f = f0 + βy is global rotation using the beta-plane approximation.

### Würsch & Craig
A one-dimensional shallow water model with simple convection parameterisation.
This is based on the data assimilation model published by Würsch & Craig in [[wursch2014]]


### Linear 2D
A finite difference solver of the linearised shallow water equations in a rotating frame.

1. ∂/∂t[u] - fv = -g ∂/∂x[h]
2. ∂/∂t[v] + fu = -g ∂/∂y[h]
3. ∂/∂t[h] + H(∂/∂x[u] + ∂/∂y[v]) = 0

Currently implemented with periodic boundary condition in the x-direction, and ∂/∂y = 0 on top and bottom boundaries.



[wursch2014]: http://www.schweizerbart.de/papers/metz/detail/23/82286/A_simple_dynamical_model_of_cumulus_convection_for_data_assimilation_research
