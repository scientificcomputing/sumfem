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

.. autoclass:: Conduction
   :members:
```

## Boundaries

```{eval-rst}
.. autoclass:: BoundaryFlux
   :members:

.. autoclass:: Outflow
   :members:

.. autoclass:: BoundaryConstraint
   :members:

.. autoclass:: DirichletMultiplier
   :members:
```

## Interfaces

```{eval-rst}
.. autoclass:: InterfaceLagrangeMultiplier
   :members:

.. autoclass:: InterfacePenalty
   :members:

.. autoclass:: InterfacePenaltyEnergy
   :members:
```

## Flow

```{eval-rst}
.. autoclass:: Darcy
   :members:

.. autoclass:: Stokes
   :members:

.. autoclass:: Convection
   :members:

.. autoclass:: DarcyDrag
   :members:

.. autoclass:: ForchheimerDrag
   :members:

.. autoclass:: GradDiv
   :members:

.. autoclass:: NavierStokes
   :members:

.. autoclass:: Brinkman
   :members:

.. autoclass:: BrinkmanForchheimer
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

.. autofunction:: butler_volmer
```

## Porous media

```{eval-rst}
.. autofunction:: bruggeman

.. autofunction:: kozeny_carman

.. autofunction:: kozeny_carman_permeability

.. autofunction:: ergun_coefficient
```

## Operators

```{eval-rst}
.. autofunction:: apply_coefficient

.. autofunction:: normal_projection

.. autofunction:: tangential_projection
```
