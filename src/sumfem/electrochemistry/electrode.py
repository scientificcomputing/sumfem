"""Porous electrodes, and the flow of the electrolyte through them.

:class:`PorousElectrode` is an electrode of carbon fibres filled with
electrolyte: its porosity, the conductivity of its fibres, its specific
area and the permeability of its fibres to the flow. The flow of the
electrolytes through the electrodes of a
:class:`~sumfem.electrochemistry.FullCell` is a component, an
:class:`ElectrodeFlow`, which the cell creates from a :class:`FlowSetup` of
what it knows. :class:`DarcyFlow` is Darcy flow through the electrodes;
another flow, e.g. Brinkman-Forchheimer flow or transient Stokes flow, is
another subclass, given to the cell's
:class:`~sumfem.electrochemistry.FlowSpec`.
"""

from __future__ import annotations

import abc
import dataclasses
import typing

import sympy
import sympy.physics.units as syu
import ufl

from sumfem.discrete import Discretization
from sumfem.error_handling import ComponentError
from sumfem.models import (
    BoundaryFlux,
    Darcy,
    bruggeman,
    ergun_coefficient,
    kozeny_carman,
)
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    Subdomain,
    value_at_defaults,
)

from .porous import Region

__all__ = ["DarcyFlow", "ElectrodeFlow", "FlowSetup", "PorousElectrode"]


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
        porosity: Default porosity, if any. Every default with a unit is in
            SI units, or a quantity with units, e.g. ``0.5 * syu.micrometer``.
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
        conductivity: float | sympy.Expr | None = None,
        specific_area: float | sympy.Expr | None = None,
        fiber_diameter: float | sympy.Expr | None = None,
        kozeny_carman_constant: float | None = None,
        pore_length: float | sympy.Expr | None = None,
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


@dataclasses.dataclass(frozen=True)
class FlowSetup:
    """What a full cell gives the flow of its electrolytes.

    Attributes:
        name: Prefix of the flow's models and parameters, the cell's name.
        domain: The domain of the cell.
        electrolyte: The subdomain of the electrolyte, the electrodes and
            what lies between them, where the pressure lives.
        electrodes: The porous electrodes, by side, ``"negative"`` and
            ``"positive"``, with their cell markers and properties, e.g. the
            permeability of their fibres.
        subdomains: The subdomain of each electrode, by side, e.g. for a
            velocity field of the electrode.
        viscosities: The dynamic viscosity of each side's electrolyte.
        densities: The density of each side's electrolyte.
        inlets: The facet marker of each electrode's inlet, by side.
        outlets: The facet marker of each electrode's outlet, by side.
        inlet_velocity: The superficial velocity at the inlets.
        outlet_pressure: The gauge pressure at the outlets.
        degree: The degree of the Lagrange elements of the cell's unknowns.
    """

    name: str
    domain: Domain
    electrolyte: Subdomain
    electrodes: typing.Mapping[str, PorousElectrode]
    subdomains: typing.Mapping[str, Subdomain]
    viscosities: typing.Mapping[str, ufl.core.expr.Expr]
    densities: typing.Mapping[str, ufl.core.expr.Expr]
    inlets: typing.Mapping[str, str]
    outlets: typing.Mapping[str, str]
    inlet_velocity: ufl.core.expr.Expr
    outlet_pressure: ufl.core.expr.Expr
    degree: int


class ElectrodeFlow(abc.ABC):
    """The flow of the electrolytes through the electrodes of a full cell.

    A flow gives the cell the superficial velocity :math:`\\mathbf{u}` of
    each electrode, :meth:`velocity`, which carries the ions in the
    electrode, out of its outlet and to its tank, and its own models and
    conditions, which the cell adds to its own. The cell holds the ions'
    inflow at the inlets itself, :eq:`full-cell-flow`, so the flow needs to
    take in the electrolyte there at the inlet velocity of the
    :class:`FlowSetup`.

    A subclass creates its unknowns and models from the setup in its
    constructor. Its models may hold time derivatives, e.g. of the velocity
    of transient Stokes flow, which the cell's simulation steps in time
    with the rest. Its :attr:`pressure` lives on
    :attr:`FlowSetup.electrolyte`, which the cell shares with what lies
    between the electrodes, see :class:`~sumfem.electrochemistry.FullCell`.

    Example:
        Darcy flow with a storage of the pressure, a flow that changes in
        time, is :class:`DarcyFlow` with one more model, given to the cell
        by its :class:`~sumfem.electrochemistry.FlowSpec`::

            class StoredDarcyFlow(DarcyFlow):
                @property
                def models(self):
                    storage = Accumulation(
                        f"{self.setup.name}_storage",
                        self.pressure,
                        weight=storativity,
                        markers=("NegativeElectrode", "PositiveElectrode"),
                    )
                    return [*super().models, storage]

            flow = FlowSpec(inlet_velocity=v_in, model=StoredDarcyFlow)

        A Brinkman-Forchheimer or a transient Stokes flow creates a velocity
        field on each electrode's subdomain, :attr:`FlowSetup.subdomains`,
        and its pressure on the electrolyte's,
        :attr:`FlowSetup.electrolyte`, with
        :class:`~sumfem.models.BrinkmanForchheimer` or
        :class:`~sumfem.models.Stokes` and an
        :class:`~sumfem.models.Accumulation` of the velocity, returns them
        from :meth:`velocity` and :attr:`pressure`, and holds the inlet
        velocity on the velocity in :meth:`conditions`.

    Args:
        setup: What the cell gives the flow.
    """

    _setup: FlowSetup
    """What the cell gives the flow."""

    def __init__(self, setup: FlowSetup):
        self._setup = setup

    @property
    def setup(self) -> FlowSetup:
        """What the cell gives the flow."""
        return self._setup

    @abc.abstractmethod
    def velocity(self, side: str) -> ufl.core.expr.Expr:
        """The superficial velocity in a side's electrode.

        Args:
            side: ``"negative"`` or ``"positive"``.
        """

    @property
    @abc.abstractmethod
    def models(self) -> list[Residual]:
        """The models of the flow, which the cell adds to its own."""

    @property
    @abc.abstractmethod
    def pressure(self) -> Field:
        """The pressure of the flow, on :attr:`FlowSetup.electrolyte`."""

    def conditions(self) -> list[DirichletBC]:
        """The Dirichlet conditions of the flow, none by default."""
        return []

    def initialize(self, discretization: Discretization) -> None:
        """Set the initial state of the flow's unknowns, none by default.

        Args:
            discretization: The discretization of the cell.
        """


class DarcyFlow(ElectrodeFlow):
    """Darcy flow through the electrodes, with one pressure.

    The pressure :math:`p` lives on :attr:`FlowSetup.electrolyte`, see
    :attr:`ElectrodeFlow.pressure`. In each electrode it follows
    :class:`~sumfem.models.Darcy` flow, ``"<name>_<side>_darcy"``, with the
    Kozeny-Carman permeability of its fibres,
    :meth:`PorousElectrode.permeability`, and the viscosity of its
    electrolyte.

    The electrolyte enters each electrode at its inlet at the inlet
    velocity, ``"<name>_<side>_inflow"``, a
    :class:`~sumfem.models.BoundaryFlux`, and leaves at its outlet, held at
    the outlet pressure, :meth:`conditions`.

    The pressure's scale is the pressure drop of a uniform flow at the
    inlet velocity through the negative electrode over the domain's length
    scale, if the defaults give it, else one pascal.

    Args:
        setup: What the cell gives the flow.
    """

    _pressure: Field
    """The pressure, on the electrolyte's subdomain."""

    _darcy: dict[str, Darcy]
    """The flow in each electrode, by side."""

    def __init__(self, setup: FlowSetup):
        super().__init__(setup)
        negative = setup.electrodes["negative"]
        try:
            scale = float(
                value_at_defaults(
                    setup.viscosities["negative"]
                    * setup.inlet_velocity
                    * setup.domain.length_scale
                    / negative.permeability()
                )
            )
        except ComponentError:
            scale = 1.0
        self._pressure = Field(
            "p",
            setup.electrolyte,
            ("Lagrange", setup.degree),
            unit=syu.pascal,
            scale=scale,
        )
        self._darcy = {
            side: Darcy(
                f"{setup.name}_{side}_darcy",
                self._pressure,
                electrode.permeability(),
                setup.viscosities[side],
                markers=electrode.markers,
            )
            for side, electrode in setup.electrodes.items()
        }

    def velocity(self, side: str) -> ufl.core.expr.Expr:
        """The superficial velocity of the side's Darcy flow.

        Args:
            side: ``"negative"`` or ``"positive"``.
        """
        return self._darcy[side].superficial_velocity

    @property
    def models(self) -> list[Residual]:
        """The Darcy flow and the inflow of each electrode."""
        setup = self._setup
        models: list[Residual] = []
        for side, darcy in self._darcy.items():
            models.append(darcy)
            models.append(
                BoundaryFlux(
                    f"{setup.name}_{side}_inflow",
                    self._pressure,
                    setup.inlet_velocity,
                    (setup.inlets[side],),
                )
            )
        return models

    @property
    def pressure(self) -> Field:
        """The pressure, on :attr:`FlowSetup.electrolyte`."""
        return self._pressure

    def conditions(self) -> list[DirichletBC]:
        """The outlet pressure at both outlets."""
        outlets = tuple(self._setup.outlets.values())
        return [DirichletBC(self._pressure, outlets, self._setup.outlet_pressure)]

    def initialize(self, discretization: Discretization) -> None:
        """Set the pressure to the outlet pressure.

        Args:
            discretization: The discretization of the cell.
        """
        outlet = self._setup.outlet_pressure
        assert isinstance(outlet, Parameter)
        discretization.set({self._pressure: float(discretization.value(outlet))})
