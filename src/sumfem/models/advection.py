"""Advection of a field by a given velocity."""

from __future__ import annotations

import numpy.typing as npt
import ufl

from sumfem.symbolic import Field, Residual, as_expr

__all__ = ["Advection", "ConservativeAdvection", "ConvectiveAdvection"]


class Advection(Residual):
    r"""Advection of a field by a velocity, in a form of a subclass.

    The velocity is any expression, e.g. a field solved by a flow model or
    data transferred from another mesh. The advected quantity is the field,
    or an expression of it, e.g. an enthalpy of a temperature. The
    subclasses write the term in the convective form,
    :class:`ConvectiveAdvection`, or the conservative one,
    :class:`ConservativeAdvection`.

    For a continuous field :math:`c` and test function :math:`v`,
    integration by parts on each cell :math:`K` relates the two forms,

    .. math::
       :label: advection-forms

       \begin{aligned}
         -(c \mathbf{u}, \nabla v)
           &= (\mathbf{u} \cdot \nabla c, v)
           + \sum_K (c \nabla \cdot \mathbf{u}, v)_K
           - \sum_F ([\![\mathbf{u} \cdot \mathbf{n}]\!] c, v)_F
           - (\mathbf{u} \cdot \mathbf{n} c, v)_{\partial \Omega},
       \end{aligned}

    with the jump :math:`[\![\mathbf{u} \cdot \mathbf{n}]\!]
    = \mathbf{u}^+ \cdot \mathbf{n}^+ + \mathbf{u}^- \cdot \mathbf{n}^-` of
    the normal velocity across the interior facets :math:`F`, and the
    outward normal :math:`\mathbf{n}`.

    The term holds only the integrals over the cells. A field that is not
    continuous, e.g. with a discontinuous element or one mapped by a Piola
    map, needs a numerical flux on the interior facets as well, e.g. an
    upwind flux, added as a model of its own.

    Args:
        name: Name of the model.
        field: The field ``c`` whose equation has the term.
        velocity: The velocity ``u``.
        markers: Cell markers restricting the term. Empty for the whole
            domain.
        quantity: The advected quantity, in place of ``c``. Defaults to
            the field.
    """

    _field: Field
    """The advected field."""

    _velocity: ufl.core.expr.Expr
    """The velocity."""

    _markers: tuple[str, ...]
    """Cell markers restricting the term."""

    _quantity: ufl.core.expr.Expr
    """The advected quantity."""

    def __init__(
        self,
        name: str,
        field: Field,
        velocity: ufl.core.expr.Expr | npt.ArrayLike,
        markers: tuple[str, ...] = (),
        quantity: ufl.core.expr.Expr | None = None,
    ):
        super().__init__(name)
        self._field = field
        self._velocity = as_expr(velocity)
        self._markers = tuple(markers)
        self._quantity = field if quantity is None else as_expr(quantity)

    @property
    def field(self) -> Field:
        """The advected field."""
        return self._field

    @property
    def velocity(self) -> ufl.core.expr.Expr:
        """The velocity."""
        return self._velocity

    @property
    def quantity(self) -> ufl.core.expr.Expr:
        """The advected quantity."""
        return self._quantity

    def flux(self) -> ufl.core.expr.Expr:
        """The advective flux ``c u``, the outer product for a vector ``c``."""
        c = self._quantity
        return ufl.outer(c, self._velocity) if c.ufl_shape else c * self._velocity


class ConvectiveAdvection(Advection):
    """Advection in convective form, ``(grad(c) u, v)``.

    It has no boundary term. It misses the jumps of the normal velocity in
    :eq:`advection-forms`, so it equals :class:`ConservativeAdvection`, up
    to the boundary term, only for a velocity that is divergence free in
    each cell with a continuous normal component, e.g. one in H(div).

    Args:
        name: Name of the model.
        field: The field ``c`` whose equation has the term.
        velocity: The velocity ``u``.
        markers: Cell markers restricting the term. Empty for the whole
            domain.
        quantity: The advected quantity, in place of ``c``. Defaults to
            the field.
    """

    def residual(self) -> ufl.Form:
        c, v = self._quantity, self._field.v
        dx = self._field.domain.dx(*self._markers)
        return ufl.inner(ufl.dot(ufl.grad(c), self._velocity), v) * dx


class ConservativeAdvection(Advection):
    """Advection in conservative form, ``-(c u, grad(v))``.

    Its boundary term is the advective flux ``(u . n c, v)``:
    :class:`~sumfem.models.Outflow` on outlets, and the inflow on inlets.
    It holds the jumps of the normal velocity in :eq:`advection-forms`, so
    it needs no facet term, and conserves the advected quantity for any
    velocity: with ``v = 1`` the interior terms cancel. Use it with a
    velocity whose normal component jumps across facets, e.g. the gradient
    of a Lagrange pressure, as in :class:`~sumfem.models.Darcy` flow.

    Args:
        name: Name of the model.
        field: The field ``c`` whose equation has the term.
        velocity: The velocity ``u``.
        markers: Cell markers restricting the term. Empty for the whole
            domain.
        quantity: The advected quantity, in place of ``c``. Defaults to
            the field.
    """

    def residual(self) -> ufl.Form:
        dx = self._field.domain.dx(*self._markers)
        return -ufl.inner(self.flux(), ufl.grad(self._field.v)) * dx
