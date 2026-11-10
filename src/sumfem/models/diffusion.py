"""Accumulation, diffusion, sources and Poisson's equation."""

from __future__ import annotations

import typing

import numpy.typing as npt
import ufl

from sumfem.error_handling import qualified_name
from sumfem.symbolic import Field, Residual, ResidualSum, as_expr
from sumfem.symbolic.si import Unit

from .operators import apply_coefficient

__all__ = ["Accumulation", "Diffusion", "Poisson", "Source"]


class Accumulation(Residual):
    """The term ``(w * du/dt, v)`` for a field ``u`` with test function ``v``.

    Args:
        name: Name of the model.
        field: The field whose time derivative is added to its equation.
        weight: Weight ``w``, such as a porosity or a heat capacity, one by
            default. Any model input is accepted, see
            :func:`~sumfem.symbolic.expressions.as_expr`.
        markers: Optional cell markers restricting the term to subdomains.
    """

    _field: Field
    """The field whose time derivative is added to its equation."""

    _weight: ufl.core.expr.Expr
    """Weight of the time derivative."""

    _markers: tuple[str, ...]
    """Cell markers restricting the term, empty for the whole domain."""

    def __init__(
        self,
        name: str,
        field: Field,
        weight: ufl.core.expr.Expr | npt.ArrayLike = 1,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        if not isinstance(field, Field):
            raise TypeError(
                f"{type(self).__name__} needs a {qualified_name(Field)}, "
                f"got {type(field).__name__}."
            )
        self._field = field
        self._weight = as_expr(weight)
        self._markers = tuple(markers)

    @property
    def field(self) -> Field:
        """The field whose time derivative is added."""
        return self._field

    def residual(self) -> ufl.Form:
        # UFL folds a weight of one away: 1 * du/dt is du/dt
        rate = self._weight * self._field.dot
        return ufl.inner(rate, self._field.v) * self._field.domain.dx(*self._markers)


class Diffusion(Residual):
    """Diffusion ``(k grad(u), grad(v))``, the weak form of ``-div(k grad(u))``.

    Subclasses name the coefficient and give it a unit, e.g.
    :class:`~sumfem.models.charge.Conduction`.

    Args:
        name: Name of the model.
        field: The diffusing field ``u``.
        coefficient: The coefficient ``k``, a scalar or a matrix. Defaults to
            a scalar parameter ``"<name>.<coefficient name>"``.
        markers: Cell markers of the region. Empty for the whole domain.
    """

    coefficient_name: typing.ClassVar[str] = "coefficient"
    """Name of the parameter created for the coefficient."""

    coefficient_unit: typing.ClassVar[Unit | None] = None
    """Unit of the parameter created for the coefficient."""

    coefficient_doc: typing.ClassVar[str] = "Diffusion coefficient"
    """Description of the parameter created for the coefficient."""

    _field: Field
    """The diffusing field."""

    _coefficient: ufl.core.expr.Expr
    """The diffusion coefficient."""

    _markers: tuple[str, ...]
    """Cell markers of the region."""

    def __init__(
        self,
        name: str,
        field: Field,
        coefficient: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        self._field = field
        self._coefficient = self.input(
            self.coefficient_name,
            coefficient,
            domain=field.domain,
            unit=self.coefficient_unit,
            doc=self.coefficient_doc,
        )
        self._markers = tuple(markers)

    @property
    def field(self) -> Field:
        """The diffusing field."""
        return self._field

    @property
    def coefficient(self) -> ufl.core.expr.Expr:
        """The diffusion coefficient."""
        return self._coefficient

    @property
    def markers(self) -> tuple[str, ...]:
        """Cell markers of the region."""
        return self._markers

    def viscous_flux(
        self, u: ufl.core.expr.Expr, grad_u: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """The viscous flux ``F(u, grad_u) = k grad_u`` of ``-div(F) = f``.

        The field and its gradient are separate arguments, the form of a
        general flux ``F(u, grad(u))``, so that other terms can evaluate it at
        other arguments, e.g. a boundary value in place of ``u``. This flux
        does not depend on ``u``.
        """
        return apply_coefficient(self._coefficient, grad_u)

    def flux(self) -> ufl.core.expr.Expr:
        """The flux ``-k grad(u)``."""
        return -self.viscous_flux(self._field, ufl.grad(self._field))

    def residual(self) -> ufl.Form:
        u = self._field
        dx = u.domain.dx(*self._markers)
        return ufl.inner(-self.flux(), ufl.grad(u.v)) * dx


class Source(Residual):
    """A source ``f`` in the equation of a field, ``-(f, v)``.

    Args:
        name: Name of the model.
        field: The field whose equation has the source.
        source: The source, of the field's shape. Defaults to the parameter
            ``"<name>.source"``.
        markers: Cell markers of the region. Empty for the whole domain.
    """

    _field: Field
    """The field whose equation has the source."""

    _source: ufl.core.expr.Expr
    """The source."""

    _markers: tuple[str, ...]
    """Cell markers of the region."""

    def __init__(
        self,
        name: str,
        field: Field,
        source: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        self._field = field
        self._source = self.input(
            "source", source, domain=field.domain, shape=field.ufl_shape, doc="Source"
        )
        self._markers = tuple(markers)

    @property
    def source(self) -> ufl.core.expr.Expr:
        """The source."""
        return self._source

    def residual(self) -> ufl.Form:
        u = self._field
        return -ufl.inner(self._source, u.v) * u.domain.dx(*self._markers)


class Poisson(ResidualSum):
    """Poisson's equation ``-div(k grad(u)) = f``.

    The sum of a :class:`Diffusion` and a :class:`Source`, both named
    ``name``, so that the source is the parameter ``"<name>.source"`` by
    default. In a sum with other models, the two are summands of their own.

    Args:
        name: Name of the model.
        field: The unknown ``u``, of any shape.
        source: The right-hand side ``f``, of the field's shape. Defaults to
            the parameter ``"<name>.source"``.
        coefficient: The coefficient ``k``, one by default.
        markers: Cell markers of the region. Empty for the whole domain.
    """

    _diffusion: Diffusion
    """The diffusion term, ``-div(k grad(u))``."""

    _source: Source
    """The source term, ``-f``."""

    def __init__(
        self,
        name: str,
        field: Field,
        source: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        coefficient: ufl.core.expr.Expr | npt.ArrayLike = 1.0,
        markers: tuple[str, ...] = (),
    ):
        self._diffusion = Diffusion(name, field, coefficient, markers)
        self._source = Source(name, field, source, markers)
        super().__init__([self._diffusion, self._source], name)

    @property
    def diffusion(self) -> Diffusion:
        """The diffusion term, e.g. for its flux."""
        return self._diffusion

    @property
    def field(self) -> Field:
        """The unknown."""
        return self._diffusion.field

    @property
    def source(self) -> ufl.core.expr.Expr:
        """The right-hand side."""
        return self._source.source
