"""Symbolic building blocks for transient problems.

Models stay semi-discrete: a time derivative is written with the
placeholder ``field.dot``. A time-stepping scheme, such as
:class:`~sumfem.discrete.time_stepping.BDF`, replaces the
placeholder when the problem is discretized in time. A model becomes
transient through terms with ``field.dot``, or with :func:`Dt` of an
expression such as the momentum ``H * u``::

    def residual(self):
        return ufl.inner(Dt(H * u), w) * dx + ...
"""

from __future__ import annotations

import typing

import sympy.physics.units as syu
import ufl
from ufl.algorithms import extract_coefficients

from .domains import Domain
from .expressions import FormOrZero, as_expr
from .terminals import Field, FieldRole, FieldTerminal, Parameter, TimeDerivative

if typing.TYPE_CHECKING:
    from .components import Residual

__all__ = [
    "Clock",
    "Dt",
    "coefficients_through_time_derivatives",
    "time_derivative_fields",
    "time_derivative_expressions",
]


def Dt(expression: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """The time derivative of an expression of fields.

    For a field this is ``field.dot``. For any other expression, such as the
    momentum ``H * u`` of the shallow water equations, it is a
    :class:`~sumfem.symbolic.terminals.TimeDerivative`, which a time-stepping
    scheme replaces by a difference of the whole expression over time steps.
    That keeps conserved quantities conserved, which the chain rule
    ``u * Dt(H) + H * Dt(u)`` would not.
    """
    if isinstance(expression, Field):
        return expression.dot
    return TimeDerivative(as_expr(expression))


class Clock:
    """The current time ``t`` and the time step ``dt`` as parameters.

    Both are :class:`~sumfem.symbolic.terminals.Parameter` objects, so models
    can use them in expressions, for instance a time-dependent source
    ``ufl.sin(clock.t)``.

    Args:
        domain: The domain the parameters are defined on.
        t0: Initial time.
        dt: Default time step size.
        name: Prefix of the parameter names, ``"<name>.t"`` and ``"<name>.dt"``.
    """

    _t: Parameter
    """Current time."""

    _dt: Parameter
    """Time step size."""

    def __init__(
        self,
        domain: Domain,
        t0: float = 0.0,
        dt: float | None = None,
        name: str = "clock",
    ):
        self._t = Parameter(
            f"{name}.t", domain, default=t0, unit=syu.second, doc="Current time"
        )
        self._dt = Parameter(
            f"{name}.dt", domain, default=dt, unit=syu.second, doc="Time step size"
        )

    @property
    def t(self) -> Parameter:
        """Current time."""
        return self._t

    @property
    def dt(self) -> Parameter:
        """Time step size."""
        return self._dt


def time_derivative_fields(model: Residual) -> tuple[Field, ...]:
    """The fields whose ``field.dot`` a model uses, in creation order."""
    fields: dict[int, Field] = {}
    for m in model.components:
        for form in _forms(m):
            for c in form.coefficients():
                if isinstance(c, FieldTerminal) and c.role is FieldRole.DOT:
                    fields[id(c.field)] = c.field
    return tuple(sorted(fields.values(), key=lambda f: f.count()))


def time_derivative_expressions(model: Residual) -> tuple[TimeDerivative, ...]:
    """The time-derivative placeholders of expressions that a model uses."""
    dots: dict[int, TimeDerivative] = {}
    for m in model.components:
        for form in _forms(m):
            for c in form.coefficients():
                if isinstance(c, TimeDerivative):
                    dots[c.count()] = c
    return tuple(dots[k] for k in sorted(dots))


def coefficients_through_time_derivatives(
    form: FormOrZero,
) -> list[ufl.Coefficient]:
    """The coefficients of a form, with those inside time-derivative placeholders."""
    result: list[ufl.Coefficient] = []
    for c in form.coefficients():
        if isinstance(c, TimeDerivative):
            result.extend(extract_coefficients(c.expression))
        else:
            result.append(c)
    return result


def _forms(component: Residual) -> list[FormOrZero]:
    """A summand's residual, and its hand-written Jacobian and preconditioner."""
    terms = (component.jacobian_term, component.preconditioner_term)
    return [component.form, *(t.form for t in terms if t is not None)]
