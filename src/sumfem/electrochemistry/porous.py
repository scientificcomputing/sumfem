"""Porous electrodes and the regions of a cell filled with electrolyte."""

from __future__ import annotations

import dataclasses

import sympy.physics.units as syu
import ufl

from sumfem.models import bruggeman, ergun_coefficient, kozeny_carman
from sumfem.symbolic import Domain, Parameter

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
        """Correction of diffusivities, Bruggeman's: one in free electrolyte."""
        return bruggeman(1.0, self.porosity, self.bruggeman_exponent)


class PorousElectrode:
    """A porous electrode of carbon fibers, filled with electrolyte.

    The electrode creates a parameter for each argument below from the
    porosity on, named ``"<name>.<argument>"``, such as
    ``"<name>.porosity"``, with the argument as its default value. Models
    that share an electrode share these parameters. A parameter without a
    default value must be set on the discretization before it is used.

    Args:
        name: Prefix of the parameter names.
        domain: The domain of the parameters.
        markers: Cell markers of the electrode. Empty for the whole domain.
        porosity: Default porosity, if any.
        conductivity: Default conductivity of the fiber material in S/m, if
            any.
        specific_area: Default specific area in 1/m, if any.
        fiber_diameter: Default fiber diameter in m, if any.
        kozeny_carman_constant: Default Kozeny-Carman constant, if any.
        pore_length: Default characteristic pore length of Ergun's drag in
            m, if any.
    """

    _markers: tuple[str, ...]
    """Cell markers of the electrode."""

    _porosity: Parameter
    """Porosity."""

    _conductivity: Parameter
    """Conductivity of the fiber material."""

    _specific_area: Parameter
    """Specific area."""

    _fiber_diameter: Parameter
    """Fiber diameter."""

    _kozeny_carman_constant: Parameter
    """Kozeny-Carman constant."""

    _pore_length: Parameter
    """Characteristic pore length of Ergun's drag."""

    def __init__(
        self,
        name: str,
        domain: Domain,
        markers: tuple[str, ...] = (),
        porosity: float | None = None,
        conductivity: float | None = None,
        specific_area: float | None = None,
        fiber_diameter: float | None = None,
        kozeny_carman_constant: float | None = None,
        pore_length: float | None = None,
    ):
        self._markers = tuple(markers)
        self._porosity = Parameter(
            f"{name}.porosity", domain, default=porosity, doc="Porosity"
        )
        self._conductivity = Parameter(
            f"{name}.conductivity",
            domain,
            default=conductivity,
            unit=syu.siemens / syu.m,
            doc="Conductivity of the fiber material",
        )
        self._specific_area = Parameter(
            f"{name}.specific_area",
            domain,
            default=specific_area,
            unit=1 / syu.m,
            doc="Specific area",
        )
        self._fiber_diameter = Parameter(
            f"{name}.fiber_diameter",
            domain,
            default=fiber_diameter,
            unit=syu.m,
            doc="Fiber diameter",
        )
        self._kozeny_carman_constant = Parameter(
            f"{name}.kozeny_carman_constant",
            domain,
            default=kozeny_carman_constant,
            doc="Kozeny-Carman constant",
        )
        self._pore_length = Parameter(
            f"{name}.pore_length",
            domain,
            default=pore_length,
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
    def porosity(self) -> Parameter:
        """Porosity."""
        return self._porosity

    @property
    def specific_area(self) -> Parameter:
        """Specific area."""
        return self._specific_area

    @property
    def fiber_diameter(self) -> Parameter:
        """Fiber diameter."""
        return self._fiber_diameter

    def effective_conductivity(self) -> ufl.core.expr.Expr:
        """Bruggeman's conductivity of the solid, ``(1 - eps)^1.5 sigma``."""
        return bruggeman(self._conductivity, 1.0 - self._porosity)

    def permeability(self) -> ufl.core.expr.Expr:
        """Kozeny-Carman permeability of the fibers."""
        return kozeny_carman(
            self._fiber_diameter, self._porosity, self._kozeny_carman_constant
        )

    def forchheimer(self) -> ufl.core.expr.Expr:
        """Ergun's inertial drag coefficient."""
        return ergun_coefficient(self._porosity, self._pore_length)
