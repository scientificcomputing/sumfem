"""The regions of a cell filled with electrolyte, free or in pores."""

from __future__ import annotations

import dataclasses

import ufl

from sumfem.models import bruggeman

__all__ = ["Region"]


@dataclasses.dataclass(frozen=True)
class Region:
    """Cells filled with electrolyte: free electrolyte, or an electrode's pores.

    Attributes:
        markers: Cell markers of the region. Empty for the whole domain.
        porosity: Porosity of a porous electrode. Defaults to one, for free
            electrolyte.
        bruggeman_exponent: Exponent of the Bruggeman correction of
            transport in the pores.
    """

    markers: tuple[str, ...] = ()
    porosity: ufl.core.expr.Expr | float = 1.0
    bruggeman_exponent: float = 1.5

    @property
    def storage(self) -> ufl.core.expr.Expr | float:
        """Volume fraction of electrolyte, the porosity."""
        return self.porosity

    @property
    def transport(self) -> ufl.core.expr.Expr | float:
        r"""Correction of diffusivities, Bruggeman's :math:`\varepsilon^b`.

        It is one in free electrolyte, with porosity one.
        """
        return bruggeman(1.0, self.porosity, self.bruggeman_exponent)
