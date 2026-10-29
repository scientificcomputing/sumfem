"""Helpers for UFL expressions and forms."""

from __future__ import annotations

import numbers
import typing
from functools import singledispatchmethod

import numpy as np
import numpy.typing as npt
import ufl
from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import Expr, Terminal
from ufl.corealg.dag_traverser import DAGTraverser

from ..error_handling import ComponentError

__all__ = ["FormOrZero", "as_expr", "check_rank", "evaluate_constant", "replace"]

FormOrZero: typing.TypeAlias = ufl.Form | ufl.ZeroBaseForm
"""A form, or ``ufl.ZeroBaseForm`` where it is zero, which keeps its arguments."""


def check_rank(form: FormOrZero, rank: int, what: str) -> None:
    """Check that a form has exactly the given rank.

    A form of rank ``r`` has arguments numbered ``0, ..., r - 1``: a residual
    has rank 1 (test functions only), a Jacobian or preconditioner rank 2
    (test and trial functions). For a residual this rejects trial functions,
    and a form with no test function at all.

    Args:
        form: The form to check.
        rank: The expected rank.
        what: Description of the form, used in the error message.

    Raises:
        ComponentError: If the argument numbers of the form are not
            ``0, ..., rank - 1``.
    """
    numbers = {a.number() for a in form.arguments()}
    expected = set(range(rank))
    if numbers != expected:
        raise ComponentError(
            f"{what} must have rank {rank}, "
            f"but its arguments have numbers {sorted(numbers)}."
        )


def as_expr(value: ufl.core.expr.Expr | npt.ArrayLike) -> ufl.core.expr.Expr:
    """Convert a component input to a UFL expression.

    UFL expressions, including a :class:`~sumfem.symbolic.terminals.Field`,
    :class:`~sumfem.symbolic.terminals.Parameter` and
    :class:`~sumfem.symbolic.terminals.KnownCoefficient`, are returned
    unchanged. Numbers and nested sequences of numbers become UFL literals.
    """
    if isinstance(value, ufl.core.expr.Expr):
        return value
    if isinstance(value, numbers.Number):
        return ufl.as_ufl(value)
    if isinstance(value, np.ndarray | list | tuple):
        array = np.asarray(value)
        if array.ndim == 0:
            # A zero-dimensional array is a number
            return ufl.as_ufl(array.item())
        return ufl.as_tensor(array.tolist())
    raise TypeError(f"Cannot use {value!r} of type {type(value).__name__} as an input.")


@typing.overload
def replace(form: ufl.Form, mapping: typing.Mapping[typing.Any, Expr]) -> ufl.Form: ...


@typing.overload
def replace(
    form: ufl.ZeroBaseForm, mapping: typing.Mapping[typing.Any, Expr]
) -> ufl.ZeroBaseForm: ...


@typing.overload
def replace(
    form: FormOrZero, mapping: typing.Mapping[typing.Any, Expr]
) -> FormOrZero: ...


@typing.overload
def replace(form: Expr, mapping: typing.Mapping[typing.Any, Expr]) -> Expr: ...


def replace(
    form: FormOrZero | Expr, mapping: typing.Mapping[typing.Any, Expr]
) -> FormOrZero | Expr:
    """Replace terminals of a form or expression, leaving derivatives unevaluated.

    Unlike :func:`ufl.replace`, which evaluates derivatives first, a
    ``ufl.derivative`` stays a derivative of the replaced form. Replacing a
    time derivative ``c.dot`` by a difference in ``c`` inside a derivative
    with respect to ``c`` then differentiates the difference too.

    Args:
        form: The form or expression.
        mapping: The replacement of each terminal, e.g. a field, its test
            function or its time derivative.
    """
    if not mapping:
        return form
    if isinstance(form, ufl.ZeroBaseForm):
        # Arguments mapped to arguments, e.g. renumbered; zero stays zero
        arguments = tuple(
            target if isinstance(target := mapping.get(a, a), ufl.Argument) else a
            for a in form.arguments()
        )
        return ufl.ZeroBaseForm(arguments)
    replacer = _Replacer(mapping)
    if isinstance(form, ufl.Form):
        return typing.cast(ufl.Form, map_integrands(replacer, form))
    return replacer(form)


class _Replacer(DAGTraverser):
    """Replace terminals, also inside unevaluated derivatives."""

    def __init__(self, mapping: typing.Mapping[typing.Any, Expr]):
        super().__init__()
        self._mapping = {k: ufl.as_ufl(v) for k, v in mapping.items()}

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        return self.reuse_if_untouched(o)

    @process.register(Terminal)
    def _(self, o: Terminal) -> Expr:
        return self._mapping.get(o, o)


def evaluate_constant(expr: ufl.core.expr.Expr) -> float | npt.NDArray[np.float64]:
    """The value of a UFL expression without a domain, e.g. ``2 * sqrt(3)``."""
    shape = expr.ufl_shape
    if shape == ():
        return float(expr)
    return np.array([float(expr[index]) for index in np.ndindex(shape)]).reshape(shape)
