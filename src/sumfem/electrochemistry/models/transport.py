"""Transport of ions by diffusion, migration and advection."""

from __future__ import annotations

import ufl

from sumfem.models import Accumulation
from sumfem.symbolic import Field, Residual, ResidualSum, as_expr

from ..materials import Electrolyte, Ion, Region

__all__ = ["NernstPlanck", "TransientNernstPlanck"]


class NernstPlanck(Residual):
    r"""Nernst-Planck transport of one ion, :math:`-(\mathbf{N}, \nabla w)`.

    The molar flux of the ion of charge :math:`z` and concentration
    :math:`c` is

    .. math::
       :label: nernst-planck-flux

       \begin{aligned}
         \mathbf{N} &= -D_\text{eff} \nabla c
           - \frac{z F}{R T} D_\text{eff} c \nabla \phi + \mathbf{u} c,
       \end{aligned}

    with its diffusivity :math:`D` corrected for the region,
    :math:`D_\text{eff} = t D` with :math:`t` the region's
    :attr:`~sumfem.electrochemistry.Region.transport`, the electrolyte
    potential :math:`\phi` and the superficial velocity :math:`\mathbf{u}`.
    For transient transport, see :class:`TransientNernstPlanck`.

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


class TransientNernstPlanck(ResidualSum):
    r"""Storage and Nernst-Planck transport of one ion.

    The conservation law of the ion in a region of porosity
    :math:`\varepsilon`,

    .. math::
       :label: nernst-planck-balance

       \begin{aligned}
         \varepsilon \frac{\partial c}{\partial t} + \nabla \cdot \mathbf{N}
           &= 0,
       \end{aligned}

    with the flux :math:`\mathbf{N}` of :eq:`nernst-planck-flux`. Sources,
    e.g. of an :class:`~sumfem.electrochemistry.ElectrodeReaction`, are
    added to it as models of their own. It is the sum of
    ``"<name>_storage"``, an :class:`~sumfem.models.Accumulation` weighted
    with the region's :attr:`~sumfem.electrochemistry.Region.storage`,
    and ``"<name>_transport"``, a :class:`NernstPlanck`.

    Args:
        name: Name of the sum, and prefix of its summands' names.
        ion: The ion, one of the electrolyte's with a field.
        electrolyte: The electrolyte of the ion.
        potential: The electrolyte potential :math:`\phi`.
        region: The region filled with electrolyte.
        velocity: The superficial velocity :math:`\mathbf{u}`. Defaults to
            zero, without advection.

    Raises:
        ValueError: If the ion is not in the electrolyte.
        TypeError: If the ion follows from electroneutrality.
    """

    _transport: NernstPlanck
    """The transport of the ion."""

    def __init__(
        self,
        name: str,
        ion: Ion,
        electrolyte: Electrolyte,
        potential: Field,
        region: Region = Region(),
        velocity: ufl.core.expr.Expr | None = None,
    ):
        transport = NernstPlanck(
            f"{name}_transport", ion, electrolyte, potential, region, velocity
        )
        storage = Accumulation(
            f"{name}_storage",
            ion.field,
            weight=region.storage,
            markers=region.markers,
        )
        super().__init__([storage, transport], name=name)
        self._transport = transport

    @property
    def ion(self) -> Ion:
        """The transported ion."""
        return self._transport.ion

    def flux(self) -> ufl.core.expr.Expr:
        r"""The molar flux :math:`\mathbf{N}`, see :meth:`NernstPlanck.flux`."""
        return self._transport.flux()
