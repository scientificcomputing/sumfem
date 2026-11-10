"""Coupling of fields on two subdomains across an interface."""

from __future__ import annotations

import numpy.typing as npt
import ufl

from sumfem.symbolic import Field, Residual, as_expr

__all__ = ["InterfaceConstraint"]


class InterfaceConstraint(Residual):
    """A condition ``a = b`` across an interface, with a Lagrange multiplier.

    Two fields on the subdomains on either side of an interface are coupled
    by a condition between a quantity ``a`` of the first side and ``b`` of
    the second, e.g. the continuity of the partial pressure of hydrogen
    across a material interface, which makes the concentrations jump. The
    multiplier ``lambda`` is the flux from the first subdomain to the
    second, so that the coupling conserves the transported quantity::

        (lambda, v_0) - (lambda, v_1) + (a - b, mu)

    on the interface, with ``v_0``, ``v_1`` and ``mu`` the test functions of
    the two fields and of the multiplier. Unlike a
    :class:`~sumfem.models.boundary.BoundaryConstraint`, whose multiplier
    enters through the derivative of the constraint, the flux is the same on
    both sides for any ``a`` and ``b``, also nonlinear ones.

    Partitioned methods for interface problems use such a flux multiplier
    (`Peterson, Bochev and Kuberry, 2019
    <https://doi.org/10.1016/j.camwa.2018.09.045>`_; `Sockwell et al., 2020
    <https://doi.org/10.1016/j.rinam.2020.100110>`_), as does a Lagrange
    multiplier method for fluid-structure interaction (`de Castro, Lee and
    Wiecek, 2025 <https://doi.org/10.1016/j.camwa.2024.12.020>`_). For a
    nonlinear condition, it corresponds to the augmented Lagrangian of
    large-strain contact in which the multiplier acts on both bodies instead
    of through the derivative of the gap (`Poulios and Renard, 2015
    <https://doi.org/10.1016/j.compstruc.2015.02.027>`_; `Chouly, Hild and
    Renard, 2023 <https://doi.org/10.1007/978-3-031-31423-0>`_, Section
    11.2.2). The Jacobian is then not symmetric.

    Args:
        name: Name of the model.
        first: The field of the subdomain on the ``"+"`` side.
        second: The field of the subdomain on the ``"-"`` side.
        first_value: The quantity ``a`` of the first side, unrestricted, e.g.
            ``(c_0 / K_S)**2`` for Sieverts' law.
        second_value: The quantity ``b`` of the second side, unrestricted,
            e.g. ``c_1 / K_H`` for Henry's law.
        multiplier: The multiplier ``lambda``, a scalar field on the
            interface, e.g. on a :class:`~sumfem.symbolic.Subdomain` of its
            facets.
        measure: The interface measure, from
            :meth:`~sumfem.symbolic.domains.Domain.interface`.

    Raises:
        ValueError: If the quantities or the multiplier are not scalars.
    """

    _first: Field
    """The field on the ``"+"`` side."""

    _second: Field
    """The field on the ``"-"`` side."""

    _first_value: ufl.core.expr.Expr
    """The quantity of the first side."""

    _second_value: ufl.core.expr.Expr
    """The quantity of the second side."""

    _multiplier: Field
    """The multiplier, the flux from the first subdomain to the second."""

    _measure: ufl.Measure
    """The interface measure."""

    def __init__(
        self,
        name: str,
        first: Field,
        second: Field,
        first_value: ufl.core.expr.Expr | npt.ArrayLike,
        second_value: ufl.core.expr.Expr | npt.ArrayLike,
        multiplier: Field,
        measure: ufl.Measure,
    ):
        super().__init__(name)
        self._first, self._second = first, second
        self._first_value = as_expr(first_value)
        self._second_value = as_expr(second_value)
        self._multiplier = multiplier
        self._measure = measure
        for label, value in (
            ("quantity of the first side", self._first_value),
            ("quantity of the second side", self._second_value),
            ("multiplier", multiplier),
        ):
            if value.ufl_shape != ():
                raise ValueError(
                    f"The {label} of {name!r} must be a scalar, got the shape "
                    f"{value.ufl_shape}."
                )

    def flux(self) -> ufl.core.expr.Expr:
        """The flux from the first subdomain to the second, the multiplier."""
        return self._multiplier("+")

    def residual(self) -> ufl.Form:
        q, dS = self.flux(), self._measure
        mu = self._multiplier.v("+")
        jump = self._first_value("+") - self._second_value("-")
        # The multiplier is the flux itself, leaving the first side and entering
        # the second, not the derivative of (lambda, a - b), which would give
        # the sides the different fluxes lambda a' and lambda b'
        return (
            q * self._first.v("+") * dS - q * self._second.v("-") * dS + jump * mu * dS
        )
