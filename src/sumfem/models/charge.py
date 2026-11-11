"""Ohmic conduction."""

from __future__ import annotations

import numpy.typing as npt
import sympy.physics.units as syu
import ufl

from sumfem.symbolic import Field

from .diffusion import Diffusion

__all__ = ["Conduction"]


class Conduction(Diffusion):
    """Ohmic conduction ``(sigma grad(phi), grad(v))``.

    Used for the solid phase of a porous electrode, with an effective
    conductivity such as :func:`~sumfem.models.porous.bruggeman`'s, or
    for an electrolyte of uniform composition.

    Args:
        name: Name of the model.
        potential: The potential ``phi``.
        conductivity: The conductivity ``sigma``. Defaults to the parameter
            ``"<name>.conductivity"`` in S/m.
        markers: Cell markers of the conducting region. Empty for the whole
            domain.
    """

    coefficient_name = "conductivity"
    coefficient_unit = syu.siemens / syu.m
    coefficient_doc = "Conductivity"

    def __init__(
        self,
        name: str,
        potential: Field,
        conductivity: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name, potential, conductivity, markers)

    def current_density(self) -> ufl.core.expr.Expr:
        """Current density ``-sigma grad(phi)``."""
        return self.flux()
