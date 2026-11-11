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

    It corrects a transport property of a phase for the volume the other
    phase takes and for the tortuosity of the paths through it. The exponent
    ``3/2`` is that of spherical obstructions, see `Tjaden et al., 2016
    <https://doi.org/10.1016/j.coche.2016.02.006>`_, where the tortuosity
    scales with the fraction to the power ``-1/2``. It suits macroporous
    electrodes; where the tortuosity is measured, the property
    ``value * fraction / tortuosity`` is the alternative of `Barzegari et
    al., 2026 <https://doi.org/10.26434/chemrxiv.10001658/v1>`_.

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

    An estimate of the permeability of a fibrous porous electrode, such as a
    carbon felt, from the diameter of its fibres, with an empirical constant
    ``K``. The carbon fibres of commercial cloths, papers and felts have
    diameters of 7 to 9 micrometres, see `Barzegari et al., 2026
    <https://doi.org/10.26434/chemrxiv.10001658/v1>`_. The estimate is of
    limited use for complex media, whose permeability is best fitted to
    measured pressure drops.

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

    The coefficient ``beta``, in 1/m, of Forchheimer's drag ``rho beta |u| u``
    on the superficial velocity ``u``, from Ergun's relation for randomly
    packed beds of spheres of diameter ``L`` (`Ergun and Orning, 1949
    <https://doi.org/10.1021/ie50474a011>`_). For a fibrous electrode, such
    as a carbon felt, it is an estimate only: the model of `Tomadakis and
    Robertson, 2005 <https://doi.org/10.1177/0021998305046438>`_ is made for
    random fibre structures, and fits to experiments give the coefficient of
    a given electrode, e.g. of the order of Ergun's for the 3D printed
    electrodes of `van der Heijden et al., 2023
    <https://doi.org/10.1002/admt.202300611>`_.

    Args:
        porosity: Porosity ``eps``.
        length: Characteristic pore length ``L``.
    """
    return 1.75 * (1.0 - porosity) / (length * porosity**3)
