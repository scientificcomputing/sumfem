"""Dilute electrolyte solutions: ions, conductivity and current density."""

from __future__ import annotations

import dataclasses
import types
import typing

import sympy
import sympy.physics.units as syu
import ufl

from sumfem.symbolic import FARADAY, GAS_CONSTANT, Domain, Field, Parameter
from sumfem.symbolic.si import to_si

__all__ = ["Electrolyte", "Ion"]

CONCENTRATION = syu.mol / syu.m**3
"""Unit of concentrations."""

DIFFUSIVITY = syu.m**2 / syu.second
"""Unit of diffusion coefficients."""


@dataclasses.dataclass(frozen=True)
class Ion:
    """An ion in an electrolyte solution.

    The ion describes itself; the :class:`Electrolyte` it is in gives its
    diffusivity as a parameter, :meth:`Electrolyte.diffusivity`, and its
    concentration, :meth:`Electrolyte.concentration`, which for the ion
    without a field follows from electroneutrality.

    Attributes:
        name: Name of the ion, e.g. ``"V2"``.
        charge: Charge number ``z``.
        diffusivity: Diffusion coefficient in free electrolyte, in m^2/s or a
            quantity with units, e.g. ``1e-9 * syu.m**2 / syu.s``. It is the
            default value of the parameter the electrolyte creates.
        concentration: The field of a transported ion. ``None`` for the one
            ion whose concentration follows from electroneutrality.
        positive: Use a smooth positive part of the concentration in the
            conductivity and in reaction kinetics. Use it for ions that can
            be depleted, such as the active species.
    """

    name: str
    charge: float
    diffusivity: float | sympy.Expr
    concentration: Field | None = None
    positive: bool = False

    @property
    def field(self) -> Field:
        """The field of a transported ion.

        Raises:
            TypeError: For the ion given by electroneutrality.
        """
        if self.concentration is None:
            raise TypeError(
                f"Ion {self.name!r} is given by electroneutrality, so it has no field."
            )
        return self.concentration


class Electrolyte:
    """A dilute electrolyte solution: its ions, temperature and constants.

    The electrolyte creates its parameters, such as the temperature
    ``"<name>.T"``, named ``"<name>.<parameter>"``: the constants, the
    temperature and the properties of the solution, see its properties, and
    the diffusivity of each ion, see :meth:`diffusivity`. Models that share
    an electrolyte share these parameters.

    At most one ion may have no field. Its concentration follows from
    electroneutrality, ``sum_j z_j c_j = 0``. Without ions, the electrolyte
    has a uniform composition, and only provides the constants, the
    temperature and the properties of the solution.

    Args:
        name: Prefix of the parameter names.
        domain: The domain of the parameters.
        ions: The ions.
        temperature: Default temperature in K.
        density: Density of the solution: an expression in kg/m^3, e.g. a
            parameter shared with a flow model, or the default value of the
            parameter the electrolyte creates. ``None`` for no default.
        viscosity: Dynamic viscosity of the solution, in Pa s, given like
            the density.
        concentration_floor: Default regularization of positive
            concentrations in mol/m^3, see :meth:`positive_part`.

    Raises:
        ValueError: If there are ions but none has a field, more than one
            ion has none, or two ions share a name.
    """

    _name: str
    """Prefix of the parameter names."""

    _ions: types.MappingProxyType[str, Ion]
    """The ions by name, in the given order."""

    _diffusivities: dict[str, Parameter]
    """The diffusivity of each ion in free electrolyte."""

    _concentrations: dict[str, ufl.core.expr.Expr]
    """The concentration of each ion: its field, or by electroneutrality."""

    _F: Parameter
    """Faraday's constant."""

    _R: Parameter
    """The gas constant."""

    _T: Parameter
    """The temperature."""

    _floor: Parameter
    """Regularization of positive concentrations."""

    _density: ufl.core.expr.Expr
    """Density of the solution."""

    _viscosity: ufl.core.expr.Expr
    """Dynamic viscosity of the solution."""

    def __init__(
        self,
        name: str,
        domain: Domain,
        ions: typing.Sequence[Ion] = (),
        temperature: float = 298.15,
        density: ufl.core.expr.Expr | float | None = None,
        viscosity: ufl.core.expr.Expr | float | None = None,
        concentration_floor: float = 1e-14,
    ):
        names = [ion.name for ion in ions]
        if len(set(names)) != len(names):
            raise ValueError(f"Ion names must be unique, got {names}.")
        balanced = [ion for ion in ions if ion.concentration is None]
        if len(balanced) > 1:
            raise ValueError(
                "At most one ion can follow from electroneutrality, got "
                f"{[ion.name for ion in balanced]}."
            )
        if balanced and len(balanced) == len(ions):
            raise ValueError(
                "Electroneutrality needs at least one transported ion, got only "
                f"{balanced[0].name!r}."
            )
        self._name = name
        self._F = Parameter(
            f"{name}.F",
            domain,
            default=to_si(FARADAY, syu.coulomb / syu.mol),
            unit=syu.coulomb / syu.mol,
            doc="Faraday's constant",
        )
        self._R = Parameter(
            f"{name}.R",
            domain,
            default=to_si(GAS_CONSTANT, syu.joule / (syu.mol * syu.kelvin)),
            unit=syu.joule / (syu.mol * syu.kelvin),
            doc="Universal gas constant",
        )
        self._T = Parameter(
            f"{name}.T", domain, default=temperature, unit=syu.kelvin, doc="Temperature"
        )
        self._floor = Parameter(
            f"{name}.floor",
            domain,
            default=concentration_floor,
            unit=CONCENTRATION,
            doc="Regularization of positive concentrations",
        )
        self._density = (
            density
            if isinstance(density, ufl.core.expr.Expr)
            else Parameter(
                f"{name}.density",
                domain,
                default=density,
                unit=syu.kg / syu.m**3,
                doc="Density",
            )
        )
        self._viscosity = (
            viscosity
            if isinstance(viscosity, ufl.core.expr.Expr)
            else Parameter(
                f"{name}.viscosity",
                domain,
                default=viscosity,
                unit=syu.pascal * syu.second,
                doc="Dynamic viscosity",
            )
        )
        self._ions = types.MappingProxyType({ion.name: ion for ion in ions})
        self._diffusivities = {
            ion.name: Parameter(
                f"{name}.D_{ion.name}",
                domain,
                default=ion.diffusivity,
                unit=DIFFUSIVITY,
                doc=f"Diffusivity of {ion.name} in free electrolyte",
            )
            for ion in ions
        }
        self._concentrations: dict[str, ufl.core.expr.Expr] = {
            ion.name: ion.concentration for ion in ions if ion.concentration is not None
        }
        charge = sum(self._ions[n].charge * c for n, c in self._concentrations.items())
        for ion in balanced:
            self._concentrations[ion.name] = -charge / ion.charge

    @property
    def name(self) -> str:
        """Prefix of the parameter names."""
        return self._name

    @property
    def ions(self) -> typing.Mapping[str, Ion]:
        """The ions by name, in the given order, as given."""
        return self._ions

    @property
    def F(self) -> Parameter:
        """Faraday's constant."""
        return self._F

    @property
    def R(self) -> Parameter:
        """The gas constant."""
        return self._R

    @property
    def T(self) -> Parameter:
        """The temperature."""
        return self._T

    @property
    def floor(self) -> Parameter:
        """Regularization of positive concentrations."""
        return self._floor

    @property
    def density(self) -> ufl.core.expr.Expr:
        """Density of the solution."""
        return self._density

    @property
    def viscosity(self) -> ufl.core.expr.Expr:
        """Dynamic viscosity of the solution."""
        return self._viscosity

    def __getitem__(self, name: str) -> Ion:
        return self._ions[name]

    def positive_part(self, c: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """The smooth positive part ``(c + sqrt(c^2 + floor^2)) / 2``."""
        return 0.5 * (c + ufl.sqrt(c * c + self._floor * self._floor))

    def concentration(
        self, name: str, positive: bool | None = None
    ) -> ufl.core.expr.Expr:
        """Concentration of an ion, as used in conductivity and kinetics.

        That is the field of a transported ion, and the expression given by
        electroneutrality for the ion without one.

        Args:
            name: Name of the ion.
            positive: Take the positive part of the concentration. Defaults
                to :attr:`Ion.positive` of the ion.
        """
        concentration = self._concentrations[name]
        if self._ions[name].positive if positive is None else positive:
            return self.positive_part(concentration)
        return concentration

    def diffusivity(self, name: str) -> Parameter:
        """Diffusivity of an ion in free electrolyte, the parameter ``D_<ion>``.

        Args:
            name: Name of the ion.
        """
        return self._diffusivities[name]

    def conductivity(
        self, correction: ufl.core.expr.Expr | float = 1.0
    ) -> ufl.core.expr.Expr:
        """Ionic conductivity ``F^2 / (R T) sum_j z_j^2 D_j c_j``.

        Args:
            correction: Factor applied to every diffusivity, such as a
                Bruggeman correction in a porous electrode.
        """
        total = sum(
            s.charge**2 * self.diffusivity(s.name) * self.concentration(s.name)
            for s in self._ions.values()
        )
        return self._F**2 / (self._R * self._T) * correction * total

    def diffusion_current(
        self, correction: ufl.core.expr.Expr | float = 1.0
    ) -> ufl.core.expr.Expr:
        """Current carried by diffusion, ``F sum_j z_j D_j grad(c_j)``.

        Args:
            correction: See :meth:`conductivity`.
        """
        total = sum(
            s.charge
            * self.diffusivity(s.name)
            * ufl.grad(self.concentration(s.name, positive=False))
            for s in self._ions.values()
        )
        return self._F * correction * total

    def current_density(
        self,
        potential: ufl.core.expr.Expr,
        correction: ufl.core.expr.Expr | float = 1.0,
    ) -> ufl.core.expr.Expr:
        """Current density ``-kappa grad(phi) - F sum_j z_j D_j grad(c_j)``.

        Args:
            potential: The electrolyte potential.
            correction: See :meth:`conductivity`.
        """
        return -self.conductivity(correction) * ufl.grad(
            potential
        ) - self.diffusion_current(correction)
