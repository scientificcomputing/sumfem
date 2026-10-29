"""Symbolic expressions discretized in terms of discrete functions."""

from __future__ import annotations

import typing
from functools import singledispatchmethod

import dolfinx.fem
from ufl.classes import Argument, Coefficient, Expr, GeometricQuantity, Terminal
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_unique_domain

from ..symbolic.domains import Domain
from .counterpart import counterpart

if typing.TYPE_CHECKING:
    from .discretization import Discretization

__all__: list[str] = []


@counterpart.register
def _(expr: Expr, discretization: Discretization, **options: typing.Any) -> Expr:
    """A symbolic expression, in terms of discrete functions.

    Fields, parameters and known coefficients become their functions, and
    geometric quantities of a domain, such as its spatial coordinate, become
    those of its mesh. The result can be evaluated or integrated with
    DOLFINx, e.g. in post-processing. It evaluates in SI base units.

    Raises:
        ValueError: If the expression contains test or trial functions.
    """
    return _Discretizer(discretization)(expr)


class _Discretizer(DAGTraverser):
    """Replace the symbolic terminals of an expression by discrete ones."""

    def __init__(self, discretization: Discretization):
        super().__init__()
        self._discretization = discretization

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        return self.reuse_if_untouched(o)

    @process.register(Terminal)
    def _(self, o: Terminal) -> Expr:
        return o

    @process.register(Coefficient)
    def _(self, o: Coefficient) -> Expr:
        if isinstance(o, dolfinx.fem.Function):
            return o
        return self._discretization.function(o)

    @process.register(GeometricQuantity)
    def _(self, o: GeometricQuantity) -> Expr:
        domain = extract_unique_domain(o)
        if not isinstance(domain, Domain):
            return o
        # Geometric quantities are constructed from their domain alone
        quantity_type = typing.cast(typing.Any, type(o))
        return quantity_type(self._discretization.maps.mesh(domain).ufl_domain())

    @process.register(Argument)
    def _(self, o: Argument) -> Expr:
        raise ValueError(f"An expression to evaluate cannot contain the argument {o}.")
