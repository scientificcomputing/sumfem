"""Charge conservation in the solid and in the electrolyte."""

from __future__ import annotations

import numpy.typing as npt
import sympy.physics.units as syu
import ufl

from sumfem.symbolic import Field, Residual

from .diffusion import Diffusion
from .electrolyte import Electrolyte
from .porous import Region

__all__ = ["Conduction", "ElectrolyteConduction"]


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


class ElectrolyteConduction(Residual):
    """Charge conservation in the electrolyte, ``-(i, grad(v))``.

    The current density ``i`` of the electrolyte,
    :meth:`~sumfem.models.electrolyte.Electrolyte.current_density`,
    includes migration with the concentration-dependent conductivity and
    diffusion of every ion. Diffusivities take the transport correction of
    the region.

    Args:
        name: Name of the model.
        potential: The electrolyte potential.
        electrolyte: The electrolyte.
        region: The region filled with electrolyte.
    """

    _potential: Field
    """The electrolyte potential."""

    _electrolyte: Electrolyte
    """The electrolyte."""

    _region: Region
    """The region filled with electrolyte."""

    def __init__(
        self,
        name: str,
        potential: Field,
        electrolyte: Electrolyte,
        region: Region = Region(),
    ):
        super().__init__(name)
        self._potential = potential
        self._electrolyte = electrolyte
        self._region = region

    def current_density(self) -> ufl.core.expr.Expr:
        """The current density of the electrolyte in the region."""
        return self._electrolyte.current_density(
            self._potential, self._region.transport
        )

    def residual(self) -> ufl.Form:
        phi = self._potential
        dx = phi.domain.dx(*self._region.markers)
        return ufl.inner(-self.current_density(), ufl.grad(phi.v)) * dx
