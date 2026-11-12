"""Charge conservation in the electrolyte."""

from __future__ import annotations

import ufl

from sumfem.symbolic import Field, Residual

from .electrolyte import Electrolyte
from .porous import Region

__all__ = ["ElectrolyteConduction"]


class ElectrolyteConduction(Residual):
    r"""Charge conservation in the electrolyte, :math:`-(\mathbf{i}_l, \nabla v)`.

    The current density of the electrolyte,
    :meth:`~sumfem.electrochemistry.electrolyte.Electrolyte.current_density`,
    carries charge by migration, with the concentration-dependent
    conductivity :math:`\kappa_\text{eff}`, and by diffusion of every ion
    :math:`k`:

    .. math::
       :label: electrolyte-current

       \begin{aligned}
         \mathbf{i}_l &= -\kappa_\text{eff} \nabla \phi
           - F \sum_k z_k D_{k,\text{eff}} \nabla c_k, \\
         \kappa_\text{eff} &= \frac{F^2}{R T}
           \sum_k z_k^2 D_{k,\text{eff}} c_k,
       \end{aligned}

    with the diffusivities corrected for the region as in
    :eq:`nernst-planck-flux`.

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
