"""Charge conservation in the electrolyte."""

from __future__ import annotations

import ufl

from sumfem.symbolic import Field, Residual

from .electrolyte import Electrolyte
from .porous import Region

__all__ = ["ElectrolyteConduction"]


class ElectrolyteConduction(Residual):
    r"""Charge conservation in the electrolyte, :math:`-(\mathbf{i}_l, \nabla v)`.

    The weak form of :math:`\nabla \cdot \mathbf{i}_l = 0`, with the current
    density :math:`\mathbf{i}_l` of :eq:`electrolyte-current`, the charge the
    ions carry, computed by the electrolyte's
    :meth:`~sumfem.electrochemistry.electrolyte.Electrolyte.current_density`
    with the diffusivities corrected for the region, :math:`t` the region's
    :attr:`~sumfem.electrochemistry.porous.Region.transport`, as in
    :eq:`nernst-planck-flux`. Sources, e.g. of an
    :class:`~sumfem.electrochemistry.ElectrodeReaction`, are added to it as
    models of their own.

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
