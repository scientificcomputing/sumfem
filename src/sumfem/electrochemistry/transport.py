"""Transport of ions by diffusion, migration and advection."""

from __future__ import annotations

import ufl

from sumfem.symbolic import Field, Residual, as_expr

from .electrolyte import Electrolyte, Ion
from .porous import Region

__all__ = ["NernstPlanck"]


class NernstPlanck(Residual):
    """Nernst-Planck transport of one ion, ``-(N, grad(w))``.

    The molar flux is ``N = -D grad(c) - D z F / (R T) c grad(phi) + u c``,
    with the diffusivity ``D`` corrected for the region. Add a
    :class:`~sumfem.models.Accumulation` weighted with the region's
    :attr:`~sumfem.electrochemistry.porous.Region.storage` for transient
    transport.

    Args:
        name: Name of the model.
        ion: The ion, one of the electrolyte's with a field.
        electrolyte: The electrolyte of the ion.
        potential: The electrolyte potential ``phi``.
        region: The region filled with electrolyte.
        velocity: The superficial velocity ``u``. Defaults to zero, without
            advection.

    Raises:
        ValueError: If the ion is not in the electrolyte.
        TypeError: If the ion follows from electroneutrality.
    """

    _ion: Ion
    """The transported ion."""

    _electrolyte: Electrolyte
    """The electrolyte of the ion."""

    _potential: Field
    """The electrolyte potential."""

    _region: Region
    """The region filled with electrolyte."""

    _velocity: ufl.core.expr.Expr
    """The superficial velocity, zero without advection."""

    def __init__(
        self,
        name: str,
        ion: Ion,
        electrolyte: Electrolyte,
        potential: Field,
        region: Region = Region(),
        velocity: ufl.core.expr.Expr | None = None,
    ):
        super().__init__(name)
        if electrolyte.ions.get(ion.name) is not ion:
            raise ValueError(f"Ion {ion.name!r} is not in {electrolyte.name!r}.")
        if ion.concentration is None:
            raise TypeError(
                f"Ion {ion.name!r} follows from electroneutrality, "
                "so it has no transport equation."
            )
        self._ion = ion
        self._electrolyte = electrolyte
        self._potential = potential
        self._region = region
        self._velocity = (
            ufl.zero(potential.domain.geometric_dimension)
            if velocity is None
            else as_expr(velocity)
        )

    @property
    def ion(self) -> Ion:
        """The transported ion."""
        return self._ion

    def flux(self) -> ufl.core.expr.Expr:
        """The molar flux ``N``."""
        e, s = self._electrolyte, self._ion
        c = s.field
        D = self._region.transport * e.diffusivity(s.name)
        migration = D * s.charge * e.F * c / (e.R * e.T) * ufl.grad(self._potential)
        return -D * ufl.grad(c) - migration + self._velocity * c

    def residual(self) -> ufl.Form:
        c = self._ion.field
        dx = c.domain.dx(*self._region.markers)
        return -ufl.inner(self.flux(), ufl.grad(c.v)) * dx
