# Models

```{eval-rst}
.. automodule:: sumfem.models
```

## Accumulation, diffusion and sources

```{eval-rst}
.. autoclass:: Accumulation
   :members:

.. autoclass:: Diffusion
   :members:

.. autoclass:: Source
   :members:

.. autoclass:: Poisson
   :members:
```

## Boundaries

```{eval-rst}
.. autoclass:: BoundaryFlux
   :members:

.. autoclass:: Outflow
   :members:

.. autoclass:: DirichletMultiplier
   :members:
```

## Reactions

```{eval-rst}
.. autoclass:: Reaction
   :members:

.. autoclass:: RateReaction
   :members:

.. autofunction:: mass_action

.. autofunction:: arrhenius

.. autodata:: BOLTZMANN
```

## Operators

```{eval-rst}
.. autofunction:: apply_coefficient

.. autofunction:: normal_projection

.. autofunction:: tangential_projection
```
