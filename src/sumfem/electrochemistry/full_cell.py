"""The full cell of a redox flow battery: two half-cells and a membrane.

:class:`FullCell` builds a cell of two :class:`RedoxHalfCell` and a
:class:`ProtonExchangeMembrane` between them, with a current collector on
the outer side of each electrode, on one mesh, each unknown on the part of
the mesh where it lives. Each side is given by a :class:`HalfCellSpec`, the
membrane by a :class:`MembraneSpec`, and the flow of the electrolytes from
their tanks, if any, by a :class:`FlowSpec`.
"""

from __future__ import annotations

import dataclasses
import types
import typing

import numpy as np
import sympy
import sympy.physics.units as syu
import ufl

from sumfem.discrete import Discretization, MeshData
from sumfem.error_handling import ComponentError
from sumfem.models import BoundaryFlux, Conduction, Darcy
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    ResidualSum,
    Subdomain,
    expression_or_parameter,
    value_at_defaults,
)

from .couple import RedoxCouple
from .electrolyte import CONCENTRATION, Ion
from .half_cell import RedoxHalfCell
from .membrane import PROPERTY, ProtonExchangeMembrane
from .porous import PorousElectrode

__all__ = ["FlowSpec", "FullCell", "HalfCellSpec", "MembraneSpec"]

SIDES = ("negative", "positive")
"""The sides of the cell, in the order of the cell's layers."""

WATER = "H2O"
"""Name of the water of the electrolytes, shared by both sides."""

PROTONS = "H"
"""Name of the protons, which carry the current through the membrane."""

VISCOSITY = syu.pascal * syu.second
"""Unit of dynamic viscosities."""


@dataclasses.dataclass(frozen=True)
class HalfCellSpec:
    """One side of a full cell: its electrolyte, its couple and its electrode.

    Attributes:
        ions: The ions of the electrolyte, see :class:`RedoxHalfCell`.
            Water, ``"H2O"``, if transported, must be an ion of both sides,
            since the water moves through the membrane.
        reservoir: The initial concentration of each transported ion, see
            :class:`RedoxHalfCell`, and with flow, that of its tank. The
            protons, ``"H"``, must be one of them.
        couple: The redox couple of the side.
        porosity: The porosity of the electrode.
        conductivity: The conductivity of the electrode's fibres, in S/m or
            with units.
        specific_area: The specific area of the electrode, in 1/m or with
            units.
        fiber_distance: The mean distance between the fibres, in m or with
            units, which limits the kinetics by the piston velocities, see
            :class:`~sumfem.electrochemistry.PistonVelocityButlerVolmer`.
            ``None`` for kinetics without them, limited by the flow's mass
            transfer with flow.
        fiber_diameter: The diameter of the electrode's fibres, in m or with
            units, for the permeability with flow, and the mass transfer
            without a fibre distance.
        kozeny_carman_constant: The Kozeny-Carman constant of the
            permeability with flow, see
            :func:`~sumfem.models.kozeny_carman`.
        density: The density of the electrolyte, in kg/m^3 or with units.
        viscosity: The dynamic viscosity of the electrolyte, in Pa s or with
            units, which the flow and the mass transfer share.
    """

    ions: tuple[Ion, ...]
    reservoir: typing.Mapping[str, float | sympy.Expr]
    couple: RedoxCouple
    porosity: float
    conductivity: float | sympy.Expr
    specific_area: float | sympy.Expr
    fiber_distance: float | sympy.Expr | None = None
    fiber_diameter: float | sympy.Expr | None = None
    kozeny_carman_constant: float | None = None
    density: float | sympy.Expr = 1399.8
    viscosity: float | sympy.Expr = 6.963e-3


@dataclasses.dataclass(frozen=True)
class MembraneSpec:
    """The properties of the membrane of a full cell.

    Each is an expression, used as given, or the default of the parameter
    ``"<membrane>.<property>"``, in SI units or with units, ``None`` for a
    parameter without a default, see :class:`ProtonExchangeMembrane`. The
    last three are those of the flow of water, with a :class:`FlowSpec`.

    Attributes:
        fixed_charge: The concentration of the fixed charges, in mol/m^3.
        proton_diffusivity: The proton diffusivity in the membrane, in
            m^2/s.
        water_diffusivity: The water diffusivity in the membrane, in m^2/s.
        electrokinetic_permeability: The electrokinetic permeability, in
            m^2.
        hydraulic_permeability: The hydraulic permeability, in m^2.
        viscosity: The viscosity of the water in the membrane, in Pa s.
    """

    fixed_charge: PROPERTY = None
    proton_diffusivity: PROPERTY = None
    water_diffusivity: PROPERTY = None
    electrokinetic_permeability: PROPERTY = None
    hydraulic_permeability: PROPERTY = None
    viscosity: PROPERTY = None


@dataclasses.dataclass(frozen=True)
class FlowSpec:
    r"""The flow of the electrolytes through the electrodes, from their tanks.

    One pressure :math:`p` lives on both electrodes and the membrane:
    :class:`~sumfem.models.Darcy` flow in each electrode, with the
    Kozeny-Carman permeability of its fibres and the viscosity of its
    electrolyte, and :class:`~sumfem.electrochemistry.SchloeglFlow` in the
    membrane, so the pressure is continuous across the faces. The
    electrolyte of each side enters its electrode at the inlet with the
    superficial velocity :math:`v_\text{in}` and the concentrations
    :math:`c_k^\text{in}` of its tank, and leaves at the outlet, held at the
    pressure :math:`p_\text{out}`, with the flow,

    .. math::
       :label: full-cell-flow

       \begin{aligned}
         -\mathbf{u} \cdot \mathbf{n} &= v_\text{in}
           && \text{at the inlet}, \\
         -\mathbf{N}_k \cdot \mathbf{n} &= v_\text{in} c_k^\text{in}
           && \text{at the inlet}, \\
         p &= p_\text{out}
           && \text{at the outlet}, \\
         (\mathbf{N}_k - \mathbf{u} c_k) \cdot \mathbf{n} &= 0
           && \text{at the outlet},
       \end{aligned}

    with the superficial velocity :math:`\mathbf{u}` of the electrode and
    the Nernst-Planck flux :math:`\mathbf{N}_k` of each ion. The inlet holds
    the flux of the ions, the flow's, rather than their concentration as
    Shah et al. do, so that the cell takes in exactly what its tank gives.

    Attributes:
        inlet_velocity: The superficial velocity at the inlets of both
            electrodes: an expression, or the default of the parameter
            ``"<cell>.inlet_velocity"``, in m/s or with units.
        outlet_pressure: The gauge pressure at the outlets, given like the
            inlet velocity, in Pa.
        tank_ratio: The volume of each tank over the volume of electrolyte in
            the pores of its electrode, ``None`` for tanks so large that the
            electrolyte at the inlets does not change.
    """

    inlet_velocity: PROPERTY = None
    outlet_pressure: PROPERTY = 0.0
    tank_ratio: float | None = None


class FullCell:
    r"""The full cell of a redox flow battery.

    The cell of `Shah et al., 2008
    <https://doi.org/10.1016/j.electacta.2008.05.067>`_, corrected: from
    left to right, a current collector, the negative porous electrode, a
    proton-exchange membrane, the positive porous electrode and a current
    collector, each a layer of the mesh, see :meth:`mesh_data`. Each unknown
    lives where it is defined: the potential of each side's solid on its
    collector and electrode, the electrolyte potential, the water and, with
    flow, the pressure on both electrodes and the membrane, and the other
    ions of each side in its electrode.

    The cell is made of:

    - a :class:`RedoxHalfCell` for each side, whose collector and membrane
      are layers of the mesh;
    - the :class:`ProtonExchangeMembrane`, whose protons carry the current
      through each face, :class:`MembraneProtonFlux`, so the ion that
      follows from electroneutrality, e.g. sulfate, stays on its side. Shah
      et al. hold the protons at the fixed charge at the faces instead,
      which lets that ion carry part of the current through them;
    - Ohmic conduction in the collectors, :class:`~sumfem.models.Conduction`;
    - the current density of the cell at the positive terminal, with the
      negative terminal grounded, see :meth:`conditions`;
    - with flow, :class:`~sumfem.models.Darcy` flow in the electrodes from
      their inlets to their outlets, :eq:`full-cell-flow`, and tanks,
      :meth:`update_tanks`.

    Args:
        domain: The domain of the cell, with the markers of :meth:`mesh_data`.
        negative: The negative side.
        positive: The positive side.
        membrane: The properties of the membrane. Parameters without
            defaults by default.
        collector_conductivity: The conductivity of the collectors: an
            expression, used as given, e.g. one of the temperature, or the
            default of the parameter ``"<name>.collector_conductivity"``, in
            S/m or with units.
        flow: The flow of the electrolytes, ``None`` for none.
        degree: Degree of the Lagrange elements of the unknowns.
        name: Prefix of the cell's own parameters and models.

    Raises:
        ValueError: If a side does not transport protons, or only one side
            transports water.
    """

    _domain: Domain
    """The domain of the cell."""

    _name: str
    """Prefix of the cell's parameters and models."""

    _subdomains: types.MappingProxyType[str, Subdomain]
    """The subdomains of the cell's unknowns, by cell marker."""

    _phi_l: Field
    """Potential of the electrolyte, over the electrodes and the membrane."""

    _water: Field | None
    """Concentration of the water, over the electrodes and the membrane."""

    _flow: FlowSpec | None
    """The flow of the electrolytes, if any."""

    _pressure: Field | None
    """Pressure over the electrodes and the membrane, with flow."""

    _inlet_velocity: ufl.core.expr.Expr | None
    """Superficial velocity at the inlets, with flow."""

    _outlet_pressure: ufl.core.expr.Expr | None
    """Gauge pressure at the outlets, with flow."""

    _darcy: types.MappingProxyType[str, Darcy]
    """The flow in each electrode, by side, empty without flow."""

    _half_cells: types.MappingProxyType[str, RedoxHalfCell]
    """The half-cells, by side."""

    _membrane: ProtonExchangeMembrane
    """The membrane."""

    _collector_conductivity: ufl.core.expr.Expr
    """Conductivity of the current collectors."""

    _current: Parameter
    """Current density of the cell, positive on discharge."""

    _model: ResidualSum
    """The sum of all models of the cell."""

    _functionals: types.MappingProxyType[str, ufl.Form]
    """The integrals of the voltage and the tanks, created once."""

    def __init__(
        self,
        domain: Domain,
        negative: HalfCellSpec,
        positive: HalfCellSpec,
        membrane: MembraneSpec = MembraneSpec(),
        collector_conductivity: PROPERTY = 1000.0,
        flow: FlowSpec | None = None,
        degree: int = 1,
        name: str = "cell",
    ):
        specs = dict(zip(SIDES, (negative, positive), strict=True))
        _check_sides(specs)
        self._domain = domain
        self._name = name
        self._flow = flow
        tdim = domain.tdim
        markers = [
            *(f"{side.capitalize()}{part}" for side in SIDES for part in _PARTS),
            "Membrane",
            "Electrolyte",
        ]
        self._subdomains = types.MappingProxyType(
            {m: Subdomain(f"{name}_{_snake(m)}", domain, tdim, m) for m in markers}
        )
        element = ("Lagrange", degree)
        electrolyte = self._subdomains["Electrolyte"]
        self._phi_l = Field("phi_l", electrolyte, element, unit=syu.volt)
        self._water = (
            Field(f"c_{WATER}", electrolyte, element, unit=CONCENTRATION, scale=1e4)
            if WATER in negative.reservoir
            else None
        )
        electrodes = {
            side: self._create_electrode(side, spec) for side, spec in specs.items()
        }
        viscosities = {
            side: expression_or_parameter(
                f"{side}_electrolyte.viscosity",
                spec.viscosity,
                self._subdomains[f"{side.capitalize()}Electrode"],
                unit=VISCOSITY,
                doc="Dynamic viscosity",
            )
            for side, spec in specs.items()
        }
        self._create_flow(electrodes, viscosities, element)
        self._half_cells = types.MappingProxyType(
            {
                side: self._create_half_cell(
                    side, spec, element, electrodes[side], viscosities[side]
                )
                for side, spec in specs.items()
            }
        )
        self._membrane = ProtonExchangeMembrane(
            f"{name}_membrane",
            self._phi_l,
            ("Membrane",),
            water=self._water,
            pressure=self._pressure,
            fixed_charge=membrane.fixed_charge,
            proton_diffusivity=membrane.proton_diffusivity,
            water_diffusivity=membrane.water_diffusivity,
            electrokinetic_permeability=membrane.electrokinetic_permeability,
            hydraulic_permeability=membrane.hydraulic_permeability,
            viscosity=membrane.viscosity,
        )
        self._collector_conductivity = expression_or_parameter(
            f"{name}.collector_conductivity",
            collector_conductivity,
            domain,
            unit=syu.siemens / syu.m,
            doc="Conductivity of the current collectors",
        )
        self._current = Parameter(
            f"{name}.current",
            domain,
            default=0.0,
            unit=syu.ampere / syu.m**2,
            doc="Current density of the cell, positive on discharge",
        )
        self._model = ResidualSum(self._create_models(), name=name)
        self._functionals = self._create_functionals()

    def _create_electrode(self, side: str, spec: HalfCellSpec) -> PorousElectrode:
        """Create the porous electrode of a side, with its parameters.

        Args:
            side: ``"negative"`` or ``"positive"``.
            spec: The side.

        Returns:
            The electrode, on the cells of the side's electrode.
        """
        prefix = side.capitalize()
        return PorousElectrode(
            f"{side}_electrode",
            self._subdomains[f"{prefix}Electrode"],
            (f"{prefix}Electrode",),
            porosity=spec.porosity,
            conductivity=spec.conductivity,
            specific_area=spec.specific_area,
            fiber_diameter=spec.fiber_diameter,
            kozeny_carman_constant=spec.kozeny_carman_constant,
        )

    def _create_flow(
        self,
        electrodes: typing.Mapping[str, PorousElectrode],
        viscosities: typing.Mapping[str, ufl.core.expr.Expr],
        element: tuple[str, int],
    ) -> None:
        """Create the pressure, its parameters and the flow in the electrodes.

        The pressure's scale is the pressure drop of a uniform flow at the
        inlet velocity through the negative electrode over the domain's
        length scale, if the defaults give it, else one pascal.

        Args:
            electrodes: The electrodes, by side.
            viscosities: The viscosities of the electrolytes, by side.
            element: The element of the pressure.
        """
        flow, name, domain = self._flow, self._name, self._domain
        if flow is None:
            self._pressure = None
            self._inlet_velocity = self._outlet_pressure = None
            self._darcy = types.MappingProxyType({})
            return
        self._inlet_velocity = expression_or_parameter(
            f"{name}.inlet_velocity",
            flow.inlet_velocity,
            domain,
            unit=syu.m / syu.second,
            doc="Superficial velocity at the inlets",
        )
        self._outlet_pressure = expression_or_parameter(
            f"{name}.outlet_pressure",
            flow.outlet_pressure,
            domain,
            unit=syu.pascal,
            doc="Gauge pressure at the outlets",
        )
        negative = electrodes["negative"]
        try:
            scale = float(
                value_at_defaults(
                    viscosities["negative"]
                    * self._inlet_velocity
                    * domain.length_scale
                    / negative.permeability()
                )
            )
        except ComponentError:
            scale = 1.0
        self._pressure = Field(
            "p",
            self._subdomains["Electrolyte"],
            element,
            unit=syu.pascal,
            scale=scale,
        )
        self._darcy = types.MappingProxyType(
            {
                side: Darcy(
                    f"{name}_{side}_darcy",
                    self._pressure,
                    electrode.permeability(),
                    viscosities[side],
                    markers=electrode.markers,
                )
                for side, electrode in electrodes.items()
            }
        )

    def _create_half_cell(
        self,
        side: str,
        spec: HalfCellSpec,
        element: tuple[str, int],
        electrode: PorousElectrode,
        viscosity: ufl.core.expr.Expr,
    ) -> RedoxHalfCell:
        """Create the half-cell of a side, with its unknowns on their subdomains.

        Args:
            side: ``"negative"`` or ``"positive"``.
            spec: The side.
            element: The element of the unknowns.
            electrode: The electrode of the side.
            viscosity: The viscosity of the side's electrolyte.

        Returns:
            The half-cell, whose collector and membrane are resolved, with the
            flow of its electrode and its outlet, if the cell has flow.
        """
        prefix = side.capitalize()
        electrode_domain = self._subdomains[f"{prefix}Electrode"]
        phi_s = Field(
            f"{side}_phi_s",
            self._subdomains[f"{prefix}Solid"],
            element,
            unit=syu.volt,
        )
        fields = {
            ion: Field(
                f"{side}_c_{ion}",
                electrode_domain,
                element,
                unit=CONCENTRATION,
                scale=1e3,
            )
            for ion in spec.reservoir
            if ion != WATER
        }
        if self._water is not None:
            fields[WATER] = self._water
        darcy = self._darcy.get(side)
        return RedoxHalfCell(
            electrode_domain,
            spec.ions,
            spec.reservoir,
            spec.couple,
            electrode,
            velocity=None if darcy is None else darcy.superficial_velocity,
            fiber_distance=spec.fiber_distance,
            collector=None,
            membrane=None,
            outlet=() if darcy is None else (f"{prefix}Outlet",),
            density=spec.density,
            viscosity=viscosity,
            name=side,
            phi_s=phi_s,
            phi_l=self._phi_l,
            fields=fields,
            electrolyte_name=f"{side}_electrolyte",
        )

    def _create_models(self) -> list[Residual]:
        """Create the models of the cell, listed in the class documentation.

        Returns:
            The models, to be summed.
        """
        name, domain = self._name, self._domain
        models: list[Residual] = [
            *self._half_cells["negative"].model.components,
            *self._half_cells["positive"].model.components,
            *self._membrane.components,
        ]
        for side, cell in self._half_cells.items():
            models.append(
                Conduction(
                    f"{name}_{side}_collector",
                    cell.phi_s,
                    self._collector_conductivity,
                    markers=(f"{side.capitalize()}Collector",),
                )
            )
        # On discharge the positive couple is reduced: the electrons it takes
        # come in through the terminal, so the current of the cell leaves the
        # solid there, and enters it on charge
        positive = self._half_cells["positive"]
        inflow = (
            self._current if positive.couple.oxidation_on_discharge else -self._current
        )
        models.append(
            BoundaryFlux(
                f"{name}_terminal_current",
                positive.phi_s,
                inflow,
                ("PositiveTerminal",),
            )
        )
        membrane = self._subdomains["Membrane"]
        for side, cell in self._half_cells.items():
            prefix = side.capitalize()
            models.append(
                self._membrane.proton_flux(
                    f"{name}_{side}_protons",
                    cell.concentrations[PROTONS],
                    domain.interface(
                        self._subdomains[f"{prefix}Electrode"],
                        membrane,
                        f"{prefix}Face",
                    ),
                    charge=cell.electrolyte[PROTONS].charge,
                )
            )
        models.extend(self._create_inflow())
        return models

    def _create_inflow(self) -> list[Residual]:
        """Create the flow in the electrodes and the inflow of the inlets.

        Returns:
            The models of :eq:`full-cell-flow` but the outlets': the Darcy
            flow and the inflow of each electrode, and the inflow of each of
            its ions, empty without flow.
        """
        if self._pressure is None:
            return []
        assert self._inlet_velocity is not None
        name, velocity = self._name, self._inlet_velocity
        models: list[Residual] = []
        for side, cell in self._half_cells.items():
            inlet = (f"{side.capitalize()}Inlet",)
            models.append(self._darcy[side])
            models.append(
                BoundaryFlux(f"{name}_{side}_inflow", self._pressure, velocity, inlet)
            )
            for ion, c in cell.concentrations.items():
                models.append(
                    BoundaryFlux(
                        f"{name}_{side}_inflow_{ion}",
                        c,
                        velocity * cell.inlet[ion],
                        inlet,
                    )
                )
        return models

    def _create_functionals(self) -> types.MappingProxyType[str, ufl.Form]:
        """Create the integrals of the voltage and of the tanks.

        Returns:
            The areas and the potential integrals of the terminals, and with
            flow, for each side, the volume of electrolyte in the pores of
            its electrode, the area of its inlet and the outflow of each ion,
            by name.
        """
        functionals: dict[str, ufl.Form] = {}
        for side, cell in self._half_cells.items():
            prefix = side.capitalize()
            ds = cell.phi_s.domain.ds(f"{prefix}Terminal")
            functionals[f"{side}_area"] = 1.0 * ds
            functionals[f"{side}_potential"] = cell.phi_s * ds
            if side not in self._darcy:
                continue
            electrode = self._subdomains[f"{prefix}Electrode"]
            functionals[f"{side}_pore_volume"] = (
                cell.electrode.porosity * electrode.dx()
            )
            functionals[f"{side}_inlet_area"] = 1.0 * electrode.ds(f"{prefix}Inlet")
            u = self._darcy[side].superficial_velocity
            for ion, c in cell.concentrations.items():
                n = ufl.FacetNormal(c.domain)
                functionals[f"{side}_outflow_{ion}"] = (
                    ufl.dot(u, n) * c * c.domain.ds(f"{prefix}Outlet")
                )
        return types.MappingProxyType(functionals)

    @staticmethod
    def mesh_data(marked: MeshData) -> MeshData:
        """Mesh data with the unions of the layers the cell's unknowns live on.

        The cell's mesh has the cell markers of its layers,
        ``"NegativeCollector"``, ``"NegativeElectrode"``, ``"Membrane"``,
        ``"PositiveElectrode"`` and ``"PositiveCollector"``, and the facet
        markers ``"NegativeTerminal"`` and ``"PositiveTerminal"``, the outer
        faces of the collectors, ``"NegativeFace"`` and ``"PositiveFace"``,
        the faces of the membrane, and with flow, ``"NegativeInlet"``,
        ``"NegativeOutlet"``, ``"PositiveInlet"`` and ``"PositiveOutlet"``,
        the ends of the electrodes. This adds ``"NegativeSolid"`` and
        ``"PositiveSolid"``, a collector and its electrode, and
        ``"Electrolyte"``, the electrodes and the membrane.

        Args:
            marked: Mesh data with the layers and the facet markers.

        Returns:
            The mesh data with the cell markers ``"NegativeSolid"``,
            ``"PositiveSolid"`` and ``"Electrolyte"`` added.
        """
        tdim = marked.mesh.topology.dim
        cells = dict(marked.markers[tdim])
        for side in ("Negative", "Positive"):
            cells[f"{side}Solid"] = (
                *cells[f"{side}Collector"],
                *cells[f"{side}Electrode"],
            )
        cells["Electrolyte"] = (
            *cells["NegativeElectrode"],
            *cells["Membrane"],
            *cells["PositiveElectrode"],
        )
        return MeshData(marked.mesh, marked.tags, {**marked.markers, tdim: cells})

    @property
    def model(self) -> Residual:
        """The sum of all models of the cell.

        These are the models of the half-cells, ``"negative_<model>"`` and
        ``"positive_<model>"``, of the membrane, ``"<name>_membrane_<model>"``,
        the conduction in the collectors, ``"<name>_<side>_collector"``, the
        current at the positive terminal, ``"<name>_terminal_current"``, the
        proton fluxes at the faces, ``"<name>_<side>_protons"``, and with
        flow, the flow in each electrode, ``"<name>_<side>_darcy"``, its
        inflow, ``"<name>_<side>_inflow"``, and that of each of its ions,
        ``"<name>_<side>_inflow_<ion>"``.
        """
        return self._model

    @property
    def half_cells(self) -> typing.Mapping[str, RedoxHalfCell]:
        """The half-cells, ``"negative"`` and ``"positive"``."""
        return self._half_cells

    @property
    def membrane(self) -> ProtonExchangeMembrane:
        """The membrane."""
        return self._membrane

    @property
    def phi_l(self) -> Field:
        """Potential of the electrolyte, over the electrodes and the membrane."""
        return self._phi_l

    @property
    def water(self) -> Field | None:
        """Concentration of the water, if transported."""
        return self._water

    @property
    def pressure(self) -> Field | None:
        """Pressure over the electrodes and the membrane, with flow."""
        return self._pressure

    @property
    def darcy(self) -> typing.Mapping[str, Darcy]:
        """The flow in each electrode, by side, empty without flow."""
        return self._darcy

    @property
    def current(self) -> Parameter:
        """Current density of the cell, positive on discharge."""
        return self._current

    def subdomain(self, marker: str) -> Subdomain:
        """The subdomain of a cell marker of the cell, e.g. ``"Membrane"``."""
        return self._subdomains[marker]

    def conditions(self) -> list[DirichletBC]:
        """The Dirichlet conditions of the cell.

        These are the negative terminal at zero potential, which fixes the
        potentials, and with flow, the outlet pressure.
        """
        conditions = [
            DirichletBC(self._half_cells["negative"].phi_s, "NegativeTerminal", 0.0)
        ]
        if self._pressure is not None:
            assert self._outlet_pressure is not None
            conditions.append(
                DirichletBC(
                    self._pressure,
                    ("NegativeOutlet", "PositiveOutlet"),
                    self._outlet_pressure,
                )
            )
        return conditions

    def initialize(self, discretization: Discretization) -> None:
        """Set a state at rest, as Shah et al. start.

        The concentrations start at the reservoir values of the sides, the
        defaults of their parameters ``"<side>.c0_<ion>"``, the electrolyte
        potential at zero, the solid of each side at the equilibrium
        potential of its couple at those concentrations, by Nernst's
        equation, so that no current flows, and the pressure at that of the
        outlets.

        Raises:
            ValueError: If the sides start with different water
                concentrations, which one water field cannot hold.
        """
        values: dict[str | ufl.Coefficient, float] = {self._phi_l: 0.0}
        for cell in self._half_cells.values():
            for ion, c in cell.concentrations.items():
                values[c] = _initial(cell, ion)
            values[cell.phi_s] = self._equilibrium_potential(cell, discretization)
        if self._water is not None:
            start = {_initial(c, WATER) for c in self._half_cells.values()}
            if len(start) != 1:
                raise ValueError(
                    "Both sides must start with the same water concentration, got "
                    f"{sorted(start)}."
                )
        if self._pressure is not None:
            assert isinstance(self._outlet_pressure, Parameter)
            values[self._pressure] = float(discretization.value(self._outlet_pressure))
        discretization.set(values)

    @staticmethod
    def _equilibrium_potential(
        cell: RedoxHalfCell, discretization: Discretization
    ) -> float:
        """The equilibrium potential of a half-cell's couple at its reservoir.

        Args:
            cell: The half-cell.
            discretization: The discretization, for the parameter values.

        Returns:
            Nernst's potential ``E0 + R T / (n F) ln(c_ox / c_red)`` in V.
        """
        value = discretization.value
        couple, e = cell.couple, cell.electrolyte
        # A half-cell's electrolyte has the temperature as a parameter
        assert isinstance(e.T, Parameter)
        E0 = value(cell.formal_potential)
        thermal = value(e.R) * value(e.T) / (couple.electrons * value(e.F))
        ratio = _initial(cell, couple.oxidized) / _initial(cell, couple.reduced)
        return float(E0 + thermal * np.log(ratio))

    def voltage(self, discretization: Discretization) -> float:
        """The cell voltage in V: the mean potential of the positive terminal
        minus that of the negative terminal."""
        assemble = discretization.assemble_scalar
        f = self._functionals
        means = {
            side: assemble(f[f"{side}_potential"]) / assemble(f[f"{side}_area"])
            for side in SIDES
        }
        return means["positive"] - means["negative"]

    def tank_volume(self, discretization: Discretization, side: str) -> float:
        """The volume of a side's tank, per unit of the dimensions the mesh
        lacks, e.g. per metre of depth for a 2D cell.

        Args:
            discretization: The discretization of the cell.
            side: ``"negative"`` or ``"positive"``.

        Raises:
            ValueError: For a cell without flow, or without tanks.
        """
        if self._flow is None or self._flow.tank_ratio is None:
            raise ValueError(f"The cell {self._name!r} has no tanks.")
        pores = discretization.assemble_scalar(self._functionals[f"{side}_pore_volume"])
        return self._flow.tank_ratio * pores

    def update_tanks(self, discretization: Discretization, dt: float) -> None:
        r"""Take a step of the tanks after a time step of the cell.

        Each tank, of volume :math:`V`, mixes what the outlet of its
        electrode returns, Shah et al.'s eq. 33 with the fluxes of the
        outlet and the inlet,

        .. math::
           :label: full-cell-tank

           \begin{aligned}
             V \frac{\mathrm{d} c_k^\text{in}}{\mathrm{d} t}
               &= \int_\text{outlet} \mathbf{u} \cdot \mathbf{n}\, c_k
                 \,\mathrm{d}s
               - v_\text{in} \int_\text{inlet} c_k^\text{in} \,\mathrm{d}s,
           \end{aligned}

        see :eq:`full-cell-flow`. The step takes the outflow of the cell's
        new state and the inflow of the inlet concentrations the cell took
        its step with, so the cell and its tanks together conserve every
        ion. The concentrations of the tanks are the reservoir parameters of
        the half-cells, ``"<side>.c0_<ion>"``. A cell without tanks keeps
        them.

        Args:
            discretization: The discretization of the cell, after the step.
            dt: The size of the step, in s.
        """
        if self._flow is None or self._flow.tank_ratio is None:
            return
        assemble, value = discretization.assemble_scalar, discretization.value
        assert isinstance(self._inlet_velocity, Parameter)
        velocity = float(value(self._inlet_velocity))
        updates: dict[str | ufl.Coefficient, float] = {}
        for side, cell in self._half_cells.items():
            volume = self.tank_volume(discretization, side)
            area = assemble(self._functionals[f"{side}_inlet_area"])
            for ion in cell.concentrations:
                tank = float(value(cell.inlet[ion]))
                outflow = assemble(self._functionals[f"{side}_outflow_{ion}"])
                inflow = velocity * area * tank
                updates[cell.inlet[ion]] = tank + dt * (outflow - inflow) / volume
        discretization.set(updates)


def _initial(cell: RedoxHalfCell, ion: str) -> float:
    """The initial concentration of an ion of a half-cell, in mol/m^3.

    Args:
        cell: The half-cell.
        ion: The name of a transported ion.

    Returns:
        The default of the reservoir concentration of the ion.
    """
    default = cell.inlet[ion].default_si
    assert default is not None
    return float(default)


_PARTS = ("Collector", "Electrode", "Solid")
"""The cell markers of each side, after the side's name."""


def _snake(marker: str) -> str:
    """A cell marker in snake case, e.g. ``"negative_solid"``.

    Args:
        marker: A cell marker in camel case, e.g. ``"NegativeSolid"``.
    """
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in marker).lstrip("_")


def _check_sides(specs: typing.Mapping[str, HalfCellSpec]) -> None:
    """Check that the sides transport protons, and water on both or neither.

    Args:
        specs: The sides, by name.

    Raises:
        ValueError: If a side does not transport protons, or only one side
            transports water.
    """
    for side, spec in specs.items():
        if PROTONS not in spec.reservoir:
            raise ValueError(
                f"The {side} side must transport protons, {PROTONS!r}, which carry "
                "the current through the membrane."
            )
    if len({WATER in spec.reservoir for spec in specs.values()}) != 1:
        raise ValueError(
            f"Water, {WATER!r}, moves through the membrane, so both sides or "
            "neither must transport it."
        )
