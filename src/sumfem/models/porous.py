"""Effective properties of porous media."""

from __future__ import annotations

import ufl

__all__ = [
    "bruggeman",
    "ergun_coefficient",
    "kozeny_carman",
    "kozeny_carman_permeability",
]


def bruggeman(
    value: ufl.core.expr.Expr | float,
    fraction: ufl.core.expr.Expr | float,
    exponent: float = 1.5,
) -> ufl.core.expr.Expr:
    """Bruggeman's effective property ``value * fraction**exponent``.

    Args:
        value: The property of the phase, e.g. a diffusivity or the
            conductivity of the solid.
        fraction: Volume fraction of the phase, e.g. the porosity for the
            electrolyte, or one minus it for the solid.
        exponent: Bruggeman exponent.
    """
    return value * fraction**exponent


def kozeny_carman(
    fiber_diameter: ufl.core.expr.Expr,
    porosity: ufl.core.expr.Expr,
    constant: ufl.core.expr.Expr | float,
) -> ufl.core.expr.Expr:
    """Kozeny-Carman permeability ``d^2 eps^3 / (K (1 - eps)^2)`` of a fibrous medium.

    Args:
        fiber_diameter: Fiber diameter ``d``.
        porosity: Porosity ``eps``.
        constant: Kozeny-Carman constant ``K``.
    """
    return fiber_diameter**2 * porosity**3 / (constant * (1.0 - porosity) ** 2)


def kozeny_carman_permeability(
    porosity: ufl.core.expr.Expr,
    reference: ufl.core.expr.Expr | float,
    floor: float = 1e-6,
) -> ufl.core.expr.Expr:
    """Kozeny-Carman permeability ``K_0 phi^3 / ((1 - phi)^2 + e)``.

    The relation ``K_0 phi^3 / (1 - phi)^2`` of equation 3 of Terschanski et
    al., CMAME 442, 117962, 2025, with a floor ``e`` that keeps it finite in
    pure fluid, ``phi = 1``. It vanishes with the porosity, so the Darcy
    drag ``phi eta / K`` of a solid needs the porosity bounded away from
    zero.

    Args:
        porosity: The porosity, or liquid fraction, ``phi``.
        reference: The reference permeability ``K_0``.
        floor: The floor ``e``.
    """
    return reference * porosity**3 / ((1.0 - porosity) ** 2 + floor)


def ergun_coefficient(
    porosity: ufl.core.expr.Expr, length: ufl.core.expr.Expr
) -> ufl.core.expr.Expr:
    """Ergun's inertial drag coefficient ``1.75 (1 - eps) / (L eps^3)``.

    Args:
        porosity: Porosity ``eps``.
        length: Characteristic pore length ``L``.
    """
    return 1.75 * (1.0 - porosity) / (length * porosity**3)
