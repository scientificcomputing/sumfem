"""Porous electrodes, and the regions of a cell filled with electrolyte.

:class:`PorousElectrode` is an electrode of carbon fibres filled with
electrolyte: its porosity, the conductivity of its fibres, its specific
area and the permeability of its fibres to the flow. A :class:`Region` is
a part of a cell filled with electrolyte, the pores of an electrode or a
free channel, with the porosity that weights storage and corrects
transport.
"""

from __future__ import annotations

import dataclasses

import sympy.physics.units as syu
import ufl

from sumfem.models import bruggeman, ergun_coefficient, kozeny_carman
from sumfem.symbolic import Domain, ExpressionOrDefault, expression_or_parameter

__all__ = ["PorousElectrode", "Region"]


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


class PorousElectrode:
    """A porous electrode of carbon fibers, filled with electrolyte.

    Each property is an expression, used as given, e.g. a field of another
    model such as the porosity of a compressed felt or a conductivity of the
    temperature, or the default of the parameter ``"<name>.<property>"``,
    such as ``"<name>.porosity"``, in SI units or a quantity with units,
    e.g. ``0.5 * syu.micrometer``, see
    :func:`~sumfem.symbolic.expression_or_parameter`. Models that share an
    electrode share these. A parameter without a default value must be set
    on the discretization before it is used.

    Args:
        name: Prefix of the parameter names.
        domain: The domain of the parameters.
        markers: Cell markers of the electrode. Empty for the whole domain.
        porosity: The porosity.
        conductivity: The conductivity of the fiber material, in S/m.
        specific_area: The specific area, in 1/m.
        fiber_diameter: The fiber diameter, in m.
        kozeny_carman_constant: The Kozeny-Carman constant.
        pore_length: The characteristic pore length of Ergun's drag, in m.
    """

    _markers: tuple[str, ...]
    """Cell markers of the electrode."""

    _porosity: ufl.core.expr.Expr
    """Porosity."""

    _conductivity: ufl.core.expr.Expr
    """Conductivity of the fiber material."""

    _specific_area: ufl.core.expr.Expr
    """Specific area."""

    _fiber_diameter: ufl.core.expr.Expr
    """Fiber diameter."""

    _kozeny_carman_constant: ufl.core.expr.Expr
    """Kozeny-Carman constant."""

    _pore_length: ufl.core.expr.Expr
    """Characteristic pore length of Ergun's drag."""

    def __init__(
        self,
        name: str,
        domain: Domain,
        markers: tuple[str, ...] = (),
        porosity: ExpressionOrDefault = None,
        conductivity: ExpressionOrDefault = None,
        specific_area: ExpressionOrDefault = None,
        fiber_diameter: ExpressionOrDefault = None,
        kozeny_carman_constant: ExpressionOrDefault = None,
        pore_length: ExpressionOrDefault = None,
    ):
        self._markers = tuple(markers)
        self._porosity = expression_or_parameter(
            f"{name}.porosity", porosity, domain, doc="Porosity"
        )
        self._conductivity = expression_or_parameter(
            f"{name}.conductivity",
            conductivity,
            domain,
            unit=syu.siemens / syu.m,
            doc="Conductivity of the fiber material",
        )
        self._specific_area = expression_or_parameter(
            f"{name}.specific_area",
            specific_area,
            domain,
            unit=1 / syu.m,
            doc="Specific area",
        )
        self._fiber_diameter = expression_or_parameter(
            f"{name}.fiber_diameter",
            fiber_diameter,
            domain,
            unit=syu.m,
            doc="Fiber diameter",
        )
        self._kozeny_carman_constant = expression_or_parameter(
            f"{name}.kozeny_carman_constant",
            kozeny_carman_constant,
            domain,
            doc="Kozeny-Carman constant",
        )
        self._pore_length = expression_or_parameter(
            f"{name}.pore_length",
            pore_length,
            domain,
            unit=syu.m,
            doc="Characteristic pore length of Ergun's drag",
        )

    @property
    def markers(self) -> tuple[str, ...]:
        """Cell markers of the electrode."""
        return self._markers

    @property
    def region(self) -> Region:
        """The electrolyte-filled pores, as a region."""
        return Region(self._markers, self._porosity)

    @property
    def porosity(self) -> ufl.core.expr.Expr:
        """Porosity."""
        return self._porosity

    @property
    def specific_area(self) -> ufl.core.expr.Expr:
        """Specific area."""
        return self._specific_area

    @property
    def fiber_diameter(self) -> ufl.core.expr.Expr:
        """Fiber diameter."""
        return self._fiber_diameter

    def effective_conductivity(self) -> ufl.core.expr.Expr:
        r"""Bruggeman's conductivity of the solid.

        That is :math:`\sigma_\text{eff} = (1 - \varepsilon)^{3/2} \sigma`, with
        the conductivity :math:`\sigma` of the fibres and the porosity
        :math:`\varepsilon`.
        """
        return bruggeman(self._conductivity, 1.0 - self._porosity)

    def permeability(self) -> ufl.core.expr.Expr:
        """Kozeny-Carman permeability of the fibers, see
        :func:`~sumfem.models.kozeny_carman`."""
        return kozeny_carman(
            self._fiber_diameter, self._porosity, self._kozeny_carman_constant
        )

    def forchheimer(self) -> ufl.core.expr.Expr:
        """Ergun's inertial drag coefficient, an estimate for fibres, see
        :func:`~sumfem.models.ergun_coefficient`."""
        return ergun_coefficient(self._porosity, self._pore_length)
