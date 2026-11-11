"""Coupling of fields on two subdomains across an interface."""

from __future__ import annotations

import numpy.typing as npt
import ufl

from sumfem.symbolic import Field, Functional, Residual, as_expr

from .operators import restricted

__all__ = [
    "InterfaceLagrangeMultiplier",
    "InterfacePenalty",
    "InterfacePenaltyEnergy",
]


class InterfaceLagrangeMultiplier(Residual):
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


class InterfacePenalty(Residual):
    """A condition ``a = b`` across an interface, by a penalty flux.

    The exchange flux ``q = k (a+ - b-)`` leaves the first subdomain and
    enters the second: ``+(q, v_0)`` in the first field's equation and
    ``-(q, v_1)`` in the second's, so the exchange conserves the transported
    quantity exactly. A large conductance ``k`` drives ``a = b``, with a jump
    of order ``q / k``. With ``a = c_0 / K_0`` and ``b = c_1 / K_1`` it is
    the penalty coupling of FESTIM's interfaces, the Robin-type condition of
    `Dark et al., 2026 <https://doi.org/10.1016/j.ijhydene.2026.153987>`_,
    Section 5.2.2. Its Jacobian is not symmetric for a nonlinear condition;
    :class:`InterfacePenaltyEnergy` is the symmetric penalty of an energy.

    Args:
        name: Name of the model.
        first: The field of the subdomain on the ``"+"`` side.
        second: The field of the subdomain on the ``"-"`` side.
        first_value: The quantity ``a`` of the first side, unrestricted,
            e.g. ``c_0 / K_0``.
        second_value: The quantity ``b`` of the second side, unrestricted.
        conductance: The conductance ``k``.
        measure: The interface measure, from
            :meth:`~sumfem.symbolic.domains.Domain.interface`.
    """

    _first: Field
    """The field on the ``"+"`` side."""

    _second: Field
    """The field on the ``"-"`` side."""

    _first_value: ufl.core.expr.Expr
    """The quantity of the first side."""

    _second_value: ufl.core.expr.Expr
    """The quantity of the second side."""

    _conductance: ufl.core.expr.Expr
    """The conductance."""

    _measure: ufl.Measure
    """The interface measure."""

    def __init__(
        self,
        name: str,
        first: Field,
        second: Field,
        first_value: ufl.core.expr.Expr | npt.ArrayLike,
        second_value: ufl.core.expr.Expr | npt.ArrayLike,
        conductance: ufl.core.expr.Expr | npt.ArrayLike,
        measure: ufl.Measure,
    ):
        super().__init__(name)
        self._first, self._second = first, second
        self._first_value = as_expr(first_value)
        self._second_value = as_expr(second_value)
        self._conductance = as_expr(conductance)
        self._measure = measure

    def flux(self) -> ufl.core.expr.Expr:
        """The exchange flux from the first subdomain to the second."""
        return restricted(self._conductance, "+") * (
            self._first_value("+") - self._second_value("-")
        )

    def residual(self) -> ufl.Form:
        q, dS = self.flux(), self._measure
        return q * self._first.v("+") * dS - q * self._second.v("-") * dS


class InterfacePenaltyEnergy(Functional):
    """A condition ``a = b`` across an interface, by the penalty energy
    ``alpha / 2 int (a+ - b-)^2 ds``.

    Added to the energy of the subdomains, its variation,
    :meth:`~sumfem.symbolic.Functional.to_residual`, penalizes the condition:
    ``alpha (a - b) (da/dc_0 v_0 - db/dc_1 v_1)`` on the interface, the
    energy-based coupling of `Dark et al., 2026
    <https://doi.org/10.1016/j.ijhydene.2026.153987>`_, Section 5.2.1. The
    Jacobian is symmetric, as the second variation of an energy. The flux
    that leaves the first subdomain, ``alpha (a - b) da/dc_0``, and the one
    that enters the second, ``alpha (a - b) db/dc_1``, differ unless the two
    derivatives agree, so the coupling conserves the transported quantity
    only up to an error that vanishes with ``alpha`` growing; for exact
    conservation use :class:`InterfacePenalty`. For Sieverts' law with
    ``b = (c_1 / K)^(1/2)``, ``db/dc_1`` grows without bound where ``c_1``
    vanishes.

    Args:
        name: Name of the model.
        first: The field of the subdomain on the ``"+"`` side.
        second: The field of the subdomain on the ``"-"`` side.
        first_value: The quantity ``a`` of the first side, unrestricted,
            e.g. ``c_0``.
        second_value: The quantity ``b`` of the second side, unrestricted,
            e.g. ``K_0 (c_1 / K_1)^n``.
        penalty: The penalty ``alpha``.
        measure: The interface measure, from
            :meth:`~sumfem.symbolic.domains.Domain.interface`.
    """

    _first: Field
    """The field on the ``"+"`` side."""

    _second: Field
    """The field on the ``"-"`` side."""

    _first_value: ufl.core.expr.Expr
    """The quantity of the first side."""

    _second_value: ufl.core.expr.Expr
    """The quantity of the second side."""

    _penalty: ufl.core.expr.Expr
    """The penalty."""

    _measure: ufl.Measure
    """The interface measure."""

    def __init__(
        self,
        name: str,
        first: Field,
        second: Field,
        first_value: ufl.core.expr.Expr | npt.ArrayLike,
        second_value: ufl.core.expr.Expr | npt.ArrayLike,
        penalty: ufl.core.expr.Expr | npt.ArrayLike,
        measure: ufl.Measure,
    ):
        super().__init__(name)
        self._first, self._second = first, second
        self._first_value = as_expr(first_value)
        self._second_value = as_expr(second_value)
        self._penalty = as_expr(penalty)
        self._measure = measure

    def mismatch(self) -> ufl.core.expr.Expr:
        """The violation ``a+ - b-`` of the condition on the interface."""
        return self._first_value("+") - self._second_value("-")

    def functional(self) -> ufl.Form:
        alpha = restricted(self._penalty, "+")
        return 0.5 * alpha * self.mismatch() ** 2 * self._measure
