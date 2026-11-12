"""Dilute electrolyte solutions: ions, conductivity and current density."""

from __future__ import annotations

import dataclasses
import types
import typing

import sympy
import sympy.physics.units as syu
import ufl

from sumfem.symbolic import (
    FARADAY,
    GAS_CONSTANT,
    Domain,
    Field,
    Parameter,
    expression_or_parameter,
)
from sumfem.symbolic.si import CONCENTRATION, DIFFUSIVITY, to_si

__all__ = ["Electrolyte", "Ion"]


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
    """

    name: str
    charge: float
    diffusivity: float | sympy.Expr
    concentration: Field | None = None

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
    r"""A dilute electrolyte solution: its ions, temperature and constants.

    An electrolyte is a solution of ions in a solvent, e.g. vanadium ions,
    protons and sulfate in water. In a dilute solution the ions do not
    interact with each other: each ion :math:`k`, of charge number
    :math:`z_k`, concentration :math:`c_k` and diffusivity :math:`D_k`,
    moves by diffusion, by migration in the electric field and with the
    solvent, with the Nernst-Planck flux :math:`\mathbf{N}_k` of
    :eq:`nernst-planck-flux`. The solution is electroneutral:

    .. math::
       :label: electroneutrality

       \begin{aligned}
         \sum_k z_k c_k &= 0.
       \end{aligned}

    The state of the solution is the concentrations of its ions and the
    temperature :math:`T`. Its properties are the diffusivities of the ions
    and the density and viscosity of the solution.

    The electrolyte gives the quantities of dilute-solution theory that the
    models of a solution share. These are the concentration of each ion,
    :meth:`concentration`, and the charge the ions carry, the current
    density :meth:`current_density`:

    .. math::
       :label: electrolyte-current

       \begin{aligned}
         \mathbf{i}_l &= F \sum_k z_k \mathbf{N}_k, \\
         \mathbf{i}_l &= -\kappa \nabla \phi
           - F \sum_k z_k t D_k \nabla c_k, \\
         \kappa &= \frac{F^2}{R T} \sum_k z_k^2 t D_k c_k,
       \end{aligned}

    with the electrolyte potential :math:`\phi`, the ionic
    :meth:`conductivity` :math:`\kappa`, the current of diffusion,
    :meth:`diffusion_current`, and a correction :math:`t` of every
    diffusivity, e.g. Bruggeman's in a porous electrode. Advection carries
    no charge, by :eq:`electroneutrality`. The transport, conduction and
    reaction models, e.g. :class:`~sumfem.electrochemistry.NernstPlanck`,
    :class:`~sumfem.electrochemistry.ElectrolyteConduction` and
    :class:`~sumfem.electrochemistry.ButlerVolmer`, build their terms from
    one electrolyte, so they agree on these quantities.

    At most one ion may have no field. Its concentration follows from
    :eq:`electroneutrality`, see :attr:`balanced`. Nothing keeps that
    concentration positive, so choose the most abundant ion of the
    supporting electrolyte, e.g. sulfate in a vanadium electrolyte, never an
    ion of a reaction, and check its minimum, e.g. with
    :meth:`~sumfem.electrochemistry.RedoxHalfCell.minimum_balanced_concentration`.
    Without ions, the electrolyte has a uniform composition, and only
    provides the constants, the temperature and the properties of the
    solution.

    Its constants, temperature and properties of the solution, and the
    diffusivity of each ion, see :meth:`diffusivity`, are parameters named
    ``"<name>.<parameter>"``, e.g. the temperature ``"<name>.T"``. Models
    that share an electrolyte share these parameters. The temperature,
    density and viscosity may instead be given as expressions, e.g. fields
    of a heat equation or of a flow, which couples the models to them.

    Args:
        name: Prefix of the parameter names.
        domain: The domain of the parameters.
        ions: The ions.
        temperature: The temperature: an expression in K, e.g. the field of
            a heat equation, or the default value of the parameter the
            electrolyte creates.
        density: Density of the solution: an expression in kg/m^3, e.g. a
            parameter shared with a flow model, or the default value of the
            parameter the electrolyte creates. ``None`` for no default.
        viscosity: Dynamic viscosity of the solution, in Pa s, given like
            the density.
        concentration_floor: Default regularization of positive
            concentrations in mol/m^3, see :meth:`positive_part`.

    Raises:
        ValueError: If there are ions but none has a field, more than one
            ion has none, the ion without one has no charge, or two ions
            share a name.
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

    _T: ufl.core.expr.Expr
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
        temperature: ufl.core.expr.Expr | float = 298.15,
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
        if balanced and balanced[0].charge == 0:
            raise ValueError(
                f"Ion {balanced[0].name!r} has no charge, so electroneutrality "
                "cannot give its concentration. Give it a field."
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
        self._T = expression_or_parameter(
            f"{name}.T", temperature, domain, unit=syu.kelvin, doc="Temperature"
        )
        self._floor = Parameter(
            f"{name}.floor",
            domain,
            default=concentration_floor,
            unit=CONCENTRATION,
            doc="Regularization of positive concentrations",
        )
        self._density = expression_or_parameter(
            f"{name}.density", density, domain, unit=syu.kg / syu.m**3, doc="Density"
        )
        self._viscosity = expression_or_parameter(
            f"{name}.viscosity",
            viscosity,
            domain,
            unit=syu.pascal * syu.second,
            doc="Dynamic viscosity",
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
    def balanced(self) -> str | None:
        """Name of the ion that follows from electroneutrality, if any."""
        for ion in self._ions.values():
            if ion.concentration is None:
                return ion.name
        return None

    @property
    def F(self) -> Parameter:
        """Faraday's constant."""
        return self._F

    @property
    def R(self) -> Parameter:
        """The gas constant."""
        return self._R

    @property
    def T(self) -> ufl.core.expr.Expr:
        """The temperature, the parameter ``"<name>.T"`` unless given."""
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
        """The smooth positive part ``(c + sqrt(c^2 + floor^2)) / 2``.

        Reaction kinetics take it of the concentrations they react, see
        :class:`~sumfem.electrochemistry.ButlerVolmer`, so that a
        concentration that undershoots zero neither reverses a reaction nor
        enters a logarithm.

        Args:
            c: A concentration.
        """
        return 0.5 * (c + ufl.sqrt(c * c + self._floor * self._floor))

    def concentration(self, name: str) -> ufl.core.expr.Expr:
        """Concentration of an ion, with its sign.

        That is the field of a transported ion, and the expression given by
        electroneutrality for the ion without one. The conductivity uses it
        as it is, as Nernst-Planck migration does, so the current is the
        charge the ions carry, see :meth:`current_density`.

        Args:
            name: Name of the ion.
        """
        return self._concentrations[name]

    def diffusivity(self, name: str) -> Parameter:
        """Diffusivity of an ion in free electrolyte, the parameter ``D_<ion>``.

        Args:
            name: Name of the ion.
        """
        return self._diffusivities[name]

    def conductivity(
        self, correction: ufl.core.expr.Expr | float = 1.0
    ) -> ufl.core.expr.Expr:
        r"""The ionic conductivity :math:`\kappa` of :eq:`electrolyte-current`.

        Args:
            correction: The correction :math:`t` of every diffusivity, e.g.
                Bruggeman's in a porous electrode.
        """
        total = sum(
            s.charge**2 * self.diffusivity(s.name) * self.concentration(s.name)
            for s in self._ions.values()
        )
        return self._F**2 / (self._R * self._T) * correction * total

    def diffusion_current(
        self, correction: ufl.core.expr.Expr | float = 1.0
    ) -> ufl.core.expr.Expr:
        r"""The current of diffusion, :math:`F \sum_k z_k t D_k \nabla c_k`.

        It is the current of :eq:`electrolyte-current` without migration.

        Args:
            correction: See :meth:`conductivity`.
        """
        total = sum(
            s.charge * self.diffusivity(s.name) * ufl.grad(self.concentration(s.name))
            for s in self._ions.values()
        )
        return self._F * correction * total

    def current_density(
        self,
        potential: ufl.core.expr.Expr,
        correction: ufl.core.expr.Expr | float = 1.0,
    ) -> ufl.core.expr.Expr:
        r"""The current density :math:`\mathbf{i}_l` of :eq:`electrolyte-current`.

        It is the charge the ions carry, with the Nernst-Planck fluxes of
        every ion, also the one given by :eq:`electroneutrality`, as the
        conductivity and the migration use the same concentrations.

        Args:
            potential: The electrolyte potential :math:`\phi`.
            correction: See :meth:`conductivity`.
        """
        return -self.conductivity(correction) * ufl.grad(
            potential
        ) - self.diffusion_current(correction)
