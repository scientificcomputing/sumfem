"""Units of values, and their conversion to SI base units.

Units are sympy unit expressions, e.g. ``sympy.physics.units.siemens /
sympy.physics.units.meter``. ``None`` means dimensionless. Discrete values
are always stored in SI base units, so models compute in consistent units
whatever units their inputs were given in.
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import sympy
import sympy.physics.units as syu
from sympy.physics.units.definitions.dimension_definitions import amount_of_substance
from sympy.physics.units.systems.si import SI
from ufl_units import get_factor

__all__ = ["Unit", "has_units", "molecule", "si_factor", "to_si"]

Unit: typing.TypeAlias = sympy.Expr
"""A sympy unit expression, e.g. ``syu.meter / syu.second``."""

molecule = syu.Quantity("molecule")
"""One molecule or particle, ``1 / N_A`` mol, in which SMART and FESTIM count
species, e.g. ``molecule / syu.um**2`` for a surface density."""
SI.set_quantity_dimension(molecule, amount_of_substance)
SI.set_quantity_scale_factor(molecule, syu.mol / syu.avogadro_number)


def _as_unit(unit: Unit | None) -> sympy.Expr:
    return sympy.sympify(1) if unit is None else unit


def si_factor(unit: Unit | None) -> float:
    """The factor converting a value in ``unit`` to SI base units.

    For instance 0.01 for centimetres, and 1 for ``None`` (dimensionless).
    """
    factor, _, _ = get_factor(1.0, _as_unit(unit))
    return float(factor)


def has_units(value: typing.Any) -> bool:
    """Whether a value is a sympy expression containing units."""
    return isinstance(value, sympy.Expr) and bool(value.atoms(syu.Quantity))


def to_si(value: npt.ArrayLike | sympy.Expr, unit: Unit | None) -> typing.Any:
    """Convert a value given in ``unit`` to SI base units.

    Args:
        value: A number or array in ``unit``, or a sympy quantity carrying
            its own units, e.g. ``5 * syu.centimeter``.
        unit: The unit the value is declared in.

    Returns:
        The value in SI base units, a float or an array.

    Raises:
        ValueError: If a sympy quantity's units do not have the dimension of
            ``unit``.
    """
    if has_units(value):
        value_si, _, value_dims = get_factor(1.0, typing.cast(sympy.Expr, value))
        _, _, expected_dims = get_factor(1.0, _as_unit(unit))
        if value_dims != expected_dims:
            raise ValueError(
                f"{sympy.sstr(value)} does not have the dimension of "
                f"{sympy.sstr(_as_unit(unit))}."
            )
        return float(value_si)
    factor = si_factor(unit)
    array = np.asarray(value)
    return array * factor if factor != 1.0 else array
