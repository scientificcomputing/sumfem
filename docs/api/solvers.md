# The solvers layer

```{eval-rst}
.. automodule:: sumfem.solvers
```

## Newton's method

```{eval-rst}
.. autoclass:: Solver
   :members:
```

## Time stepping

```{eval-rst}
.. autoclass:: TimeScheme
   :members:

.. autoclass:: BDF
   :members:

.. autoclass:: Theta
   :members:

.. autoclass:: CrankNicolson
   :members:

.. autoclass:: ForwardEuler
   :members:
```

## Simulations

```{eval-rst}
.. autoclass:: Simulation
   :members:

.. autoclass:: AdaptiveSimulation
   :members:

.. autoclass:: AdaptiveTimeStep
   :members:

.. autoclass:: Output
   :members:

.. currentmodule:: sumfem.solvers.simulation

.. autofunction:: owned

.. autofunction:: versions

.. currentmodule:: sumfem.solvers
```

## PETSc options

```{eval-rst}
.. currentmodule:: sumfem.solvers.solver

.. autodata:: DEFAULT_PETSC_OPTIONS

.. autodata:: FACTORIZED_PETSC_OPTIONS

.. autodata:: FACTORIZED_ABSOLUTE_TOLERANCES

.. autodata:: NEWTON_TOLERANCES

.. autofunction:: direct_solver_options
```
