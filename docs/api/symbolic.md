# The symbolic layer

```{eval-rst}
.. automodule:: sumfem.symbolic
```

## Domains and coordinate systems

```{eval-rst}
.. autoclass:: Domain
   :members:

.. autoclass:: Subdomain
   :members:

.. autoclass:: CoordinateSystem
   :members:

.. autoclass:: Cartesian
   :members:

.. autoclass:: Cylindrical
   :members:

.. autoclass:: Spherical
   :members:

.. autoclass:: Weighted
   :members:

.. autofunction:: apply_coordinates
```

## Fields, parameters and known coefficients

```{eval-rst}
.. autoclass:: Field
   :members:

.. autoclass:: Parameter
   :members:

.. autoclass:: KnownCoefficient
   :members:

.. autofunction:: as_expr
```

## Components

```{eval-rst}
.. autoclass:: Component
   :members:

.. autoclass:: Residual
   :members:

.. autoclass:: ResidualSum
   :members:

.. autoclass:: Jacobian
   :members:

.. autoclass:: JacobianSum
   :members:

.. autoclass:: Functional
   :members:

.. autoclass:: FunctionalSum
   :members:
```

## Models from plain UFL

```{eval-rst}

.. autoclass:: UFLResidual
   :members:
```

## Systems and sequences

```{eval-rst}
.. autoclass:: System
   :members:

.. autofunction:: build_system

.. autoclass:: Sequential
   :members:
```

## Time

```{eval-rst}
.. autoclass:: Clock
   :members:

.. autofunction:: Dt
```

## Boundary conditions

```{eval-rst}
.. autoclass:: DirichletBC
   :members:
```

## Transfers

```{eval-rst}
.. autoclass:: Transfer
   :members:

.. autoclass:: Assignment
   :members:

.. autoclass:: Interpolation
   :members:

.. autodata:: Transferable
```

## Units and dimensional analysis

```{eval-rst}
.. autofunction:: check_units

.. autofunction:: equation_dimensions

.. autofunction:: equation_scales

.. autofunction:: pi_groups

.. autoclass:: PiGroup
   :members:

.. autoexception:: UnitError

.. autodata:: BOLTZMANN
```

## Errors

```{eval-rst}
.. autoexception:: ComponentError

.. autoexception:: SystemBuildError
```
