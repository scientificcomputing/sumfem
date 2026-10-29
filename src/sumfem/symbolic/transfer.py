"""Transfers of data between fields, e.g. between non-matching meshes."""

from __future__ import annotations

import typing

import ufl
from ufl_units import get_factor

from ..error_handling import ComponentError, check_name
from .domains import Domain, domain_of
from .expressions import as_expr
from .terminals import Field, KnownCoefficient

__all__ = ["Assignment", "Interpolation", "Transfer", "Transferable"]

Transferable = Field | KnownCoefficient
"""The terminals a transfer reads from and writes to."""


class Transfer:
    """A stage that writes a field or known coefficient from another one.

    Transfers couple by data, not by summing forms: for instance between
    non-matching meshes, where no form can hold both, see
    :class:`Interpolation`. A transfer is a stage of a
    :class:`~sumfem.discrete.coupling.Sequential` sequence, or is
    applied by hand: ``disc[transfer].apply()``. It only describes what is
    transferred. How is up to its discrete counterpart: register one for a
    new kind of transfer, e.g. a projection, with
    :func:`@counterpart.register <sumfem.discrete.counterpart.counterpart>`.

    Args:
        name: Name of the transfer, unique among the stages it runs with.
        source: The field or known coefficient to read.
        target: The known coefficient or field to write, usually on another
            domain. A stage may solve for it, starting from the transferred
            values, e.g. a correction or an initial guess.

    Raises:
        ComponentError: If the source and the target differ in shape or in the
            dimension of their units, or are the same.
    """

    _name: str
    """Name of the transfer."""

    _source: Transferable
    """The field or known coefficient to read."""

    _target: Transferable
    """The known coefficient or field to write."""

    def __init__(self, name: str, source: Transferable, target: Transferable):
        check_name(name, type(self).__name__)
        for role, terminal in (("source", source), ("target", target)):
            if not isinstance(terminal, Field | KnownCoefficient):
                raise TypeError(
                    f"The {role} of transfer {name!r} must be a field or a known "
                    f"coefficient, got {terminal!r}."
                )
        if source is target:
            raise ComponentError(f"Transfer {name!r} maps {source.name!r} to itself.")
        if source.ufl_shape != target.ufl_shape:
            raise ComponentError(
                f"Transfer {name!r} maps shape {source.ufl_shape} to "
                f"{target.ufl_shape}."
            )
        # Values are stored in SI units, so equal dimensions suffice
        if get_factor(1.0, source.unit or 1)[2] != get_factor(1.0, target.unit or 1)[2]:
            raise ComponentError(
                f"Transfer {name!r} maps {source.name!r} in {source.unit} to "
                f"{target.name!r} in {target.unit}, of another dimension."
            )
        self._name = name
        self._source = source
        self._target = target

    @property
    def name(self) -> str:
        """Name of the transfer."""
        return self._name

    @property
    def source(self) -> Transferable:
        """The field or known coefficient to read."""
        return self._source

    @property
    def target(self) -> Transferable:
        """The known coefficient or field to write."""
        return self._target

    @property
    def target_domain(self) -> Domain:
        """The domain of the target."""
        return domain_of(self._target.ufl_function_space())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._name!r}, {self._source.name!r} -> "
            f"{self._target.name!r})"
        )


class Interpolation(Transfer):
    """Interpolate into the target, from a source on any mesh.

    The meshes need not match or be nested. The discrete layer locates the
    target's interpolation points in the source mesh once, and evaluates the
    source there at every transfer, see
    :class:`~sumfem.discrete.transfer.Interpolation`.

    Args:
        name: See :class:`Transfer`.
        source: See :class:`Transfer`.
        target: See :class:`Transfer`.
        markers: Cell markers of the target's domain to interpolate on.
            Empty for all its cells.
    """

    _markers: tuple[str, ...]
    """Cell markers of the target's domain to interpolate on."""

    def __init__(
        self,
        name: str,
        source: Transferable,
        target: Transferable,
        markers: typing.Sequence[str] = (),
    ):
        super().__init__(name, source, target)
        self._markers = tuple(markers)
        for marker in self._markers:
            self.target_domain.marker_ids(marker, "cell")

    @property
    def markers(self) -> tuple[str, ...]:
        """Cell markers of the target's domain to interpolate on."""
        return self._markers


class Assignment(Transfer):
    """Write an expression of fields into the target, point by point.

    The expression is evaluated at the target's interpolation points, so it
    may only use terminals on the target's domain. It may use the target
    itself, e.g. the update ``g + lambda (T - T_m)`` of a liquid fraction
    ``g``, clipped to ``[0, 1]``, between solves for the temperature ``T``.

    Args:
        name: See :class:`Transfer`.
        target: See :class:`Transfer`.
        expression: The expression, of the target's shape. Values are in SI
            base units, as every symbolic expression's.

    Raises:
        ComponentError: If the expression and the target differ in shape.
    """

    _expression: ufl.core.expr.Expr
    """The expression written into the target."""

    def __init__(
        self,
        name: str,
        target: Transferable,
        expression: ufl.core.expr.Expr,
    ):
        check_name(name, type(self).__name__)
        if not isinstance(target, Field | KnownCoefficient):
            raise TypeError(
                f"The target of {type(self).__name__} {name!r} must be a field or a "
                f"known coefficient, got {target!r}."
            )
        expression = as_expr(expression)
        if expression.ufl_shape != target.ufl_shape:
            raise ComponentError(
                f"{type(self).__name__} {name!r} writes shape {expression.ufl_shape} "
                f"into {target.name!r} of shape {target.ufl_shape}."
            )
        self._name = name
        self._expression = expression
        self._target = target

    @property
    def source(self) -> ufl.core.expr.Expr:  # type: ignore[override]
        """The expression written into the target."""
        return self._expression

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r} -> {self._target.name!r})"
