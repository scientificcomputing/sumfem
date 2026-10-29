"""Named symbolic terminals that models are written with.

Three kinds of named coefficients appear in a model's residual:

- :class:`Field`: an unknown. It is the solution coefficient itself, and
  owns its test function ``v`` and its trial function ``du``, plus
  placeholders for its time derivative and its history.
- :class:`Parameter`: a named scalar, vector or tensor value that is
  constant in space. It lives on a Real space, so a residual can be
  differentiated with respect to it.
- :class:`KnownCoefficient`: named, spatially varying data, such as a mesh size
  or an imposed velocity field.

Every field uses its UFL count, which is unique among all coefficients, as
the ``part`` number of its test and trial functions. This makes forms
written by different models directly addable with ``+``. The final block
layout is chosen when a system is built, see
:func:`~sumfem.symbolic.system.build_system`.
"""

from __future__ import annotations

import enum
import functools
import numbers
import typing
import weakref

import basix.ufl
import numpy as np
import numpy.typing as npt
import sympy
import ufl
import ufl.algorithms
import ufl.pullback
from ufl.finiteelement import AbstractFiniteElement
from ufl_units import QuantityMixin

from ..error_handling import ComponentError, check_name, qualified_name
from .domains import Domain, domain_of
from .elements import ElementSpec, as_element
from .si import Unit

_PIOLA = (
    ufl.pullback.contravariant_piola,
    ufl.pullback.covariant_piola,
    ufl.pullback.double_contravariant_piola,
    ufl.pullback.double_covariant_piola,
    ufl.pullback.covariant_contravariant_piola,
)
"""The pullbacks of elements whose values are vectors or tensors in space."""

__all__ = [
    "Field",
    "FieldRole",
    "FieldTerminal",
    "KnownCoefficient",
    "Parameter",
    "TimeDerivative",
    "field_of",
]

# Map from the part number of a field's test and trial functions to the
# field. It is needed because test and trial functions are plain
# ufl.Argument objects and cannot point to their field. It holds fields
# weakly, so it never keeps a field alive.
_fields_by_part: weakref.WeakValueDictionary[int, Field] = weakref.WeakValueDictionary()


class Field(ufl.Coefficient):
    """An unknown of a variational problem.

    A field is the coefficient holding the solution, so models use it
    directly in expressions, e.g. ``ufl.grad(c)``. It also owns:

    - ``v``: the test function of its equation,
    - ``du``: the trial function, for hand-written Jacobians,
    - ``dot``: a placeholder for its time derivative,
    - ``previous(k)``: its value ``k`` time steps back.

    Models that write terms with the same ``field.v`` contribute to the same
    equation when they are summed. A model that uses the field without
    writing an equation for it is coupled to whichever model does.

    The name must be passed positionally: ``ufl.Coefficient.__new__`` reads
    the first positional argument.

    Args:
        name: Unique name of the field.
        domain: The domain the field lives on.
        element: The finite element, see
            :data:`~sumfem.symbolic.elements.ElementSpec`.
        unit: Unit of the field, a sympy unit expression. ``None`` means
            dimensionless. Values given for the field, such as initial and
            boundary values, are in this unit. The discretization stores them
            in SI base units.
        scale: Typical magnitude of the field in its unit, used as reference
            value in dimensional analysis.
        spatial: Whether the components of a vector-valued field are those
            of a vector in physical space, which change with the basis of
            the coordinates, e.g. a velocity or a displacement, or
            independent scalars, which do not, e.g. the concentrations of
            several species. The two differ only on domains whose coordinate
            system has a curved basis, e.g.
            :class:`~sumfem.symbolic.coordinates.Cylindrical`: there a
            vector in space gains components in the angular directions, and
            its gradient the terms of the Christoffel symbols. In Cartesian
            coordinates, and with a weight alone, they are the same.
            Elements mapped by a Piola transformation, e.g. Raviart-Thomas
            or Nedelec, are always vectors in space.

    Raises:
        ComponentError: For ``spatial=False`` with an element mapped by a
            Piola transformation.
    """

    _name: str
    """Unique name of the field."""

    _spatial: bool
    """Whether the components are those of a vector in physical space."""

    _unit: sympy.Expr
    """Unit of the field, ``1`` if dimensionless."""

    _scale: float
    """Typical magnitude of the field in its unit, for dimensional analysis."""

    _previous: dict[int, FieldTerminal]
    """Values ``k`` time steps back, by ``k``, created on first access."""

    def __init__(
        self,
        name: str,
        /,
        domain: Domain,
        element: ElementSpec,
        unit: Unit | None = None,
        scale: float = 1.0,
        spatial: bool = True,
    ):
        check_name(name, type(self).__name__)
        if not isinstance(domain, Domain):
            raise TypeError(
                f"{type(self).__name__} {name!r} needs a {qualified_name(Domain)}, "
                f"got {domain!r}."
            )
        # The space is labeled with the field name, which keeps form
        # signatures, and hence the JIT cache, independent of creation order.
        # Names are unique in a model, so the spaces of its fields differ.
        element_ = as_element(element, domain)
        if not spatial and element_.pullback in _PIOLA:
            raise ComponentError(
                f"{type(self).__name__} {name!r} has the element {element_}, mapped "
                "by a Piola transformation, so its components are those of a "
                "vector in space and it cannot have spatial=False."
            )
        space = ufl.FunctionSpace(domain, element_, label=name)
        ufl.Coefficient.__init__(self, space)
        self._name = name
        self._unit = sympy.sympify(1) if unit is None else unit
        self._scale = _positive(name, scale)
        self._spatial = spatial
        self._previous = {}
        _fields_by_part[self.count()] = self

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._name

    @property
    def spatial(self) -> bool:
        """Whether the components are those of a vector in physical space."""
        return self._spatial

    @property
    def domain(self) -> Domain:
        """Domain the field lives on."""
        return domain_of(self.ufl_function_space())

    @property
    def element(self) -> AbstractFiniteElement:
        """Finite element of the field."""
        return self.ufl_function_space().ufl_element()

    @property
    def unit(self) -> sympy.Expr:
        """Unit of the field, ``1`` if dimensionless."""
        return self._unit

    @property
    def scale(self) -> float:
        """Typical magnitude of the field in its unit, for dimensional analysis."""
        return self._scale

    # The symbolic objects below are created on first access and then kept,
    # so every model sees the same objects for a field.

    @functools.cached_property
    def v(self) -> ufl.Argument:
        """The test function of this field's equation.

        Its part number is the field's count, unique among all fields.
        """
        return ufl.Argument(self.ufl_function_space(), 0, part=self.count())

    @functools.cached_property
    def du(self) -> ufl.Argument:
        """The trial function, for hand-written Jacobians."""
        return ufl.Argument(self.ufl_function_space(), 1, part=self.count())

    @functools.cached_property
    def dot(self) -> FieldTerminal:
        """Placeholder for the time derivative of the field.

        A time-stepping scheme replaces it, for instance by a backward
        difference of the field and its :meth:`previous` values.
        """
        return FieldTerminal(self, FieldRole.DOT)

    def previous(self, k: int = 1) -> FieldTerminal:
        """The value of the field ``k`` time steps back."""
        if not isinstance(k, numbers.Integral) or k < 1:
            raise ValueError(f"History index must be a positive integer, got {k!r}.")
        k = int(k)
        if k not in self._previous:
            self._previous[k] = FieldTerminal(self, FieldRole.PREVIOUS, k)
        return self._previous[k]


class FieldRole(enum.Enum):
    """What a :class:`FieldTerminal` stands for, relative to its field."""

    DOT = enum.auto()
    """Placeholder for the time derivative of the field."""

    PREVIOUS = enum.auto()
    """The field a number of time steps back."""


class FieldTerminal(ufl.Coefficient):
    """A coefficient derived from a :class:`Field`, with a :class:`FieldRole`."""

    _name: str
    """Name, e.g. ``"c.dot"`` or ``"c.previous[1]"``."""

    _field: Field
    """The field this coefficient belongs to.

    This back-reference is the only way to trace the coefficient to its
    field, since it lives on the field's space but is a different
    coefficient. It is also a strong reference: the registry of fields holds
    fields weakly, so a form that uses only ``c.dot`` still keeps ``c``
    alive. The resulting cycle, a field caching its terminals and each
    terminal pointing back, is freed by Python's cycle collector."""

    _role: FieldRole
    """What this coefficient stands for, relative to its field."""

    _k: int
    """Number of steps back in time for :attr:`FieldRole.PREVIOUS`, else 0."""

    def __init__(self, field: Field, role: FieldRole, k: int = 0):
        match role:
            case FieldRole.DOT:
                name = f"{field.name}.dot"
            case FieldRole.PREVIOUS:
                name = f"{field.name}.previous[{k}]"
            case _:
                raise TypeError(
                    f"Role must be a {qualified_name(FieldRole)}, got {role!r}."
                )
        ufl.Coefficient.__init__(self, field.ufl_function_space())
        self._name = name
        self._field = field
        self._role = role
        self._k = k

    @property
    def name(self) -> str:
        """Name, e.g. ``"c.dot"`` or ``"c.previous[1]"``."""
        return self._name

    @property
    def field(self) -> Field:
        """The field this coefficient belongs to."""
        return self._field

    @property
    def spatial(self) -> bool:
        """Whether the field's vectors are vectors in space."""
        return self._field.spatial

    @property
    def role(self) -> FieldRole:
        """What this coefficient stands for, relative to its field."""
        return self._role

    @property
    def k(self) -> int:
        """Number of steps back in time for :attr:`FieldRole.PREVIOUS`, else 0."""
        return self._k


class TimeDerivative(ufl.Coefficient):
    """Placeholder for the time derivative of an expression of fields.

    Made by :func:`~sumfem.symbolic.time.Dt`. A time-stepping scheme
    replaces it by a combination of the expression at the current and
    previous steps, see :meth:`history`, so that the time derivative of a
    conserved quantity such as ``H * u`` is discretized as a whole.
    """

    _expression: ufl.core.expr.Expr
    """The differentiated expression."""

    _fields: tuple[Field, ...]
    """The fields of the expression, in creation order."""

    def __init__(self, expression: ufl.core.expr.Expr):
        coefficients = ufl.algorithms.extract_coefficients(expression)
        fields = sorted(
            {id(c): c for c in coefficients if isinstance(c, Field)}.values(),
            key=lambda f: f.count(),
        )
        if not fields:
            raise ComponentError(
                f"The time derivative of {expression!r} needs a "
                f"{qualified_name(Field)} in the expression."
            )
        domain = fields[0].domain
        if any(f.domain is not domain for f in fields):
            raise ComponentError(
                "The fields of a time derivative must share one domain, got "
                f"{', '.join(repr(f.name) for f in fields)}."
            )
        element = as_element(("DG", 0, expression.ufl_shape), domain)
        ufl.Coefficient.__init__(self, ufl.FunctionSpace(domain, element))
        self._expression = expression
        self._fields = tuple(fields)

    @property
    def name(self) -> str:
        """Name, ``"Dt(<fields>)"``."""
        return f"Dt({', '.join(f.name for f in self._fields)})"

    @property
    def expression(self) -> ufl.core.expr.Expr:
        """The differentiated expression."""
        return self._expression

    @property
    def fields(self) -> tuple[Field, ...]:
        """The fields of the expression, in creation order."""
        return self._fields

    def history(self, k: int) -> ufl.core.expr.Expr:
        """The expression ``k`` steps back, of every field's ``previous(k)``."""
        return ufl.replace(self._expression, {f: f.previous(k) for f in self._fields})


def field_of(obj: ufl.core.expr.Expr) -> Field | None:
    """Return the field an expression belongs to, if any.

    That is the field itself, the field of a :class:`FieldTerminal`, or the
    field whose test function ``v`` or trial function ``du`` the expression
    is. Any other argument, such as a plain ``ufl.TestFunction``, gives
    ``None``.
    """
    if isinstance(obj, Field):
        return obj
    if isinstance(obj, FieldTerminal):
        return obj.field
    if isinstance(obj, ufl.Argument):
        part = obj.part()
        field = None if part is None else _fields_by_part.get(part)
        if field is not None and (obj == field.v or obj == field.du):
            return field
    return None


def _positive(name: str, scale: float) -> float:
    """Check that a reference scale is a positive number."""
    value = float(scale)
    if not value > 0:
        raise ComponentError(f"The scale of {name!r} must be positive, got {scale!r}.")
    return value


def _validate_value(
    name: str, value: npt.ArrayLike, shape: tuple[int, ...]
) -> npt.NDArray[typing.Any]:
    """Check a value given for a parameter and return it as an array.

    A scalar given for a vector or tensor parameter is broadcast to every
    component, so ``default=0.0`` works for any shape.

    Args:
        name: Name of the parameter, used in error messages.
        value: The value, a number or a nested sequence of numbers.
        shape: The value shape of the parameter.

    Returns:
        The value as an array of the given shape.

    Raises:
        ComponentError: If the value, after broadcasting a scalar, does not have
            the given shape, or if it is not numeric.
    """
    array = np.asarray(value)
    if array.ndim == 0 and shape != ():
        # A scalar sets every component
        array = np.full(shape, array)
    if array.shape != shape:
        raise ComponentError(
            f"Value for {name!r} has shape {array.shape}, expected {shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ComponentError(
            f"Value for {name!r} must be numeric, got dtype {array.dtype}."
        )
    return array


class Parameter(ufl.Coefficient, QuantityMixin):
    """A named value that is constant in space, with a unit.

    The value lives on a Real function space, a single global degree of
    freedom per component. This means a residual can be differentiated with
    respect to a parameter, as needed for sensitivities and optimal
    control. The discretization owns the degree of freedom on one process
    and keeps all processes in sync.

    A parameter is also a ``ufl_units`` quantity, so dimensional analysis
    finds it in forms. Its scale for that analysis is the largest magnitude
    of its default, or 1 without a non-zero default.

    Args:
        name: Unique name. Parameters created through
            :meth:`~sumfem.symbolic.components.Component.parameter` are
            named ``"<component>.<name>"``.
        domain: The domain the parameter is defined on.
        shape: Value shape, ``()`` for a scalar.
        default: Default value in ``unit``, used when no other value is set.
            A scalar sets every component of a vector or tensor parameter.
        unit: Unit of the value, a sympy unit expression. ``None`` means
            dimensionless. The discretization stores values in SI base units.
        doc: Description, shown in documentation and archived metadata.

    The name must be passed positionally: ``ufl.Coefficient.__new__`` reads
    the first positional argument.
    """

    _name: str
    """Unique name."""

    _default: npt.NDArray[typing.Any] | None
    """Validated default value, or ``None`` if it must be set explicitly."""

    _doc: str
    """Description of the parameter."""

    def __init__(
        self,
        name: str,
        /,
        domain: Domain,
        shape: tuple[int, ...] = (),
        default: npt.ArrayLike | None = None,
        unit: Unit | None = None,
        doc: str = "",
    ):
        check_name(name, type(self).__name__)
        if not isinstance(domain, Domain):
            raise TypeError(
                f"{type(self).__name__} {name!r} needs a {qualified_name(Domain)}, "
                f"got {domain!r}."
            )
        shape = tuple(int(s) for s in shape)
        element = basix.ufl.real_element(
            domain.ufl_coordinate_element().cell_type, shape
        )
        space = ufl.FunctionSpace(domain, element, label=f"{name}#parameter")
        ufl.Coefficient.__init__(self, space)
        self._name = name
        self._default = (
            None if default is None else _validate_value(name, default, shape)
        )
        self._doc = doc
        magnitude = (
            0.0 if self._default is None else float(np.max(np.abs(self._default)))
        )
        self._init_units(magnitude if magnitude > 0 else 1.0, unit, name)

    @property
    def name(self) -> str:
        """Unique name."""
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        """Value shape."""
        return self.ufl_shape

    @property
    def default(self) -> npt.NDArray[typing.Any] | None:
        """Default value, or ``None`` if the value must be set explicitly."""
        return None if self._default is None else self._default.copy()

    @property
    def doc(self) -> str:
        """Description of the parameter."""
        return self._doc


class KnownCoefficient(ufl.Coefficient, QuantityMixin):
    """Named, spatially varying data that a model reads but does not solve for.

    It is also a ``ufl_units`` quantity, so dimensional analysis finds it in
    forms, with its typical magnitude as scale.

    Args:
        name: Unique name. Known coefficients created through
            :meth:`~sumfem.symbolic.components.Component.known_coefficient`
            are named ``"<component>.<name>"``.
        domain: The domain the data is defined on.
        element: The finite element, see
            :data:`~sumfem.symbolic.elements.ElementSpec`.
        unit: Unit of the data, a sympy unit expression. ``None`` means
            dimensionless. The discretization stores values in SI base units.
        scale: Typical magnitude of the data in its unit, for dimensional
            analysis.
        doc: Description of the data.

    The name must be passed positionally: ``ufl.Coefficient.__new__`` reads
    the first positional argument.
    """

    _name: str
    """Unique name."""

    _doc: str
    """Description of the data."""

    def __init__(
        self,
        name: str,
        /,
        domain: Domain,
        element: ElementSpec,
        unit: Unit | None = None,
        scale: float = 1.0,
        doc: str = "",
    ):
        check_name(name, type(self).__name__)
        if not isinstance(domain, Domain):
            raise TypeError(
                f"{type(self).__name__} {name!r} needs a {qualified_name(Domain)}, "
                f"got {domain!r}."
            )
        space = ufl.FunctionSpace(
            domain, as_element(element, domain), label=f"{name}#coefficient"
        )
        ufl.Coefficient.__init__(self, space)
        self._name = name
        self._doc = doc
        self._init_units(_positive(name, scale), unit, name)

    @property
    def name(self) -> str:
        """Unique name."""
        return self._name

    @property
    def element(self) -> AbstractFiniteElement:
        """Finite element of the data."""
        return self.ufl_function_space().ufl_element()

    @property
    def doc(self) -> str:
        """Description of the data."""
        return self._doc
