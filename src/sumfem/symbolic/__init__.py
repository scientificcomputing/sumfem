"""The symbolic description of variational problems, without mesh data.

- :class:`Domain`: a symbolic description of a type of mesh, with named
  markers.
- :class:`Cartesian`, :class:`Cylindrical`, :class:`Spherical` and
  :class:`Weighted`: coordinate systems of domains, which weight their
  integrals, e.g. for axisymmetric or radially symmetric bodies.
- :class:`Field`, :class:`Parameter`, :class:`KnownCoefficient`: the named
  objects residuals and functionals are written with, with units.
- :class:`Residual`: a reusable contribution to a residual, summed with
  ``+`` into a :class:`ResidualSum`.
- :class:`Functional`: a reusable contribution to a scalar functional,
  summed into a :class:`FunctionalSum`, whose variation is a residual.
- :func:`build_system`: turns a residual into blocked residual and
  Jacobian forms, optionally in dimensionless form.
- :func:`Dt` and :class:`Clock`: semi-discrete transient terms.
- :class:`DirichletBC`: boundary conditions by marker name.
- :func:`check_units`, :func:`equation_scales` and :func:`pi_groups`:
  dimensional analysis with ``ufl_units``.
"""

from ..error_handling import ComponentError, SystemBuildError
from .bcs import DirichletBC
from .components import (
    Component,
    Functional,
    FunctionalSum,
    Jacobian,
    JacobianSum,
    Residual,
    ResidualSum,
)
from .coordinates import (
    Cartesian,
    CoordinateSystem,
    Cylindrical,
    Spherical,
    Weighted,
    apply_coordinates,
)
from .coupling import Sequential
from .domains import Domain, Subdomain
from .expressions import as_expr
from .system import System, build_system
from .terminals import Field, KnownCoefficient, Parameter
from .time import Clock, Dt
from .transfer import Assignment, Interpolation, Transfer, Transferable
from .units import (
    PiGroup,
    UnitError,
    check_units,
    equation_dimensions,
    equation_scales,
    pi_groups,
)

__all__ = [
    "Cartesian",
    "CoordinateSystem",
    "Cylindrical",
    "Spherical",
    "Weighted",
    "apply_coordinates",
    "ComponentError",
    "SystemBuildError",
    "Dt",
    "Clock",
    "Component",
    "Functional",
    "FunctionalSum",
    "Jacobian",
    "JacobianSum",
    "ResidualSum",
    "DirichletBC",
    "Subdomain",
    "Sequential",
    "Transfer",
    "Assignment",
    "Interpolation",
    "Transferable",
    "Domain",
    "Field",
    "KnownCoefficient",
    "Residual",
    "Parameter",
    "PiGroup",
    "System",
    "UnitError",
    "as_expr",
    "build_system",
    "check_units",
    "equation_dimensions",
    "equation_scales",
    "pi_groups",
]
