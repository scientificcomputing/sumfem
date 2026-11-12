"""The full cell of a redox flow battery: two half-cells and a membrane.

:class:`FullCell` builds a cell of two :class:`RedoxHalfCell` and a
:class:`ProtonExchangeMembrane` between them, with a current collector on
the outer side of each electrode, on one mesh, each unknown on the part of
the mesh where it lives. Each side is given by a :class:`HalfCellSpec`, the
membrane by a :class:`MembraneSpec`, and the flow of the electrolytes from
their tanks, if any, by a :class:`FlowSpec`.
"""

from __future__ import annotations

import types
import typing

import sympy.physics.units as syu
import ufl

from sumfem.discrete import Discretization, MeshData
from sumfem.models import BoundaryFlux, Conduction
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    ExpressionOrDefault,
    Field,
    Parameter,
    Residual,
    ResidualSum,
    Subdomain,
    expression_or_parameter,
)
from sumfem.symbolic.si import CONCENTRATION, DENSITY, VISCOSITY

from ..materials import PorousElectrode
from ..models import ElectrodeFlow, FlowSetup, ProtonExchangeMembrane
from .half_cell import RedoxHalfCell
from .specs import PROTONS, WATER, FlowSpec, HalfCellSpec, MembraneSpec

__all__ = ["FullCell"]

SIDES = ("negative", "positive")
"""The sides of the cell, in the order of the cell's layers."""


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
    - with flow, the flow of the electrolytes through the electrodes from
      their inlets to their outlets, :eq:`full-cell-flow`, an
      :class:`~sumfem.electrochemistry.ElectrodeFlow` such as
      :class:`~sumfem.electrochemistry.DarcyFlow`, see :attr:`flow`, and
      tanks, :meth:`update_tanks`.

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
        ValueError: If only one side transports water.
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

    _spec: FlowSpec | None
    """The specification of the flow of the electrolytes, if any."""

    _flow: ElectrodeFlow | None
    """The flow of the electrolytes, if any."""

    _inlet_velocity: ufl.core.expr.Expr | None
    """Superficial velocity at the inlets, with flow."""

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
        collector_conductivity: ExpressionOrDefault = 1000.0,
        flow: FlowSpec | None = None,
        degree: int = 1,
        name: str = "cell",
    ):
        specs = dict(zip(SIDES, (negative, positive), strict=True))
        if (WATER in negative.reservoir) != (WATER in positive.reservoir):
            raise ValueError(
                f"Water, {WATER!r}, moves through the membrane, so both sides or "
                "neither must transport it."
            )
        self._domain = domain
        self._name = name
        self._spec = flow
        tdim = domain.tdim
        markers = [
            *(
                f"{side.capitalize()}{part}"
                for side in SIDES
                for part in ("Collector", "Electrode", "Solid")
            ),
            "Membrane",
            "Electrolyte",
        ]
        self._subdomains = types.MappingProxyType(
            {m: Subdomain(f"{name}_{m}", domain, tdim, m) for m in markers}
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
        properties = {
            side: self._create_properties(side, spec) for side, spec in specs.items()
        }
        self._create_flow(electrodes, properties, degree)
        self._half_cells = types.MappingProxyType(
            {
                side: self._create_half_cell(
                    side, spec, element, electrodes[side], *properties[side]
                )
                for side, spec in specs.items()
            }
        )
        self._membrane = ProtonExchangeMembrane(
            f"{name}_membrane",
            self._phi_l,
            ("Membrane",),
            water=self._water,
            pressure=None if self._flow is None else self._flow.pressure,
            fixed_charge=membrane.fixed_charge,
            proton_diffusivity=membrane.proton_diffusivity,
            water_diffusivity=membrane.water_diffusivity,
            electrokinetic_permeability=membrane.electrokinetic_permeability,
            hydraulic_permeability=membrane.hydraulic_permeability,
            viscosity=membrane.viscosity,
            temperature=membrane.temperature,
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

    def _create_properties(
        self, side: str, spec: HalfCellSpec
    ) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """Create the density and the viscosity of a side's electrolyte.

        They are parameters of the side's electrolyte, which its half-cell
        and the flow share.

        Args:
            side: ``"negative"`` or ``"positive"``.
            spec: The side.

        Returns:
            The density and the viscosity.
        """
        domain = self._subdomains[f"{side.capitalize()}Electrode"]
        density = expression_or_parameter(
            f"{side}_electrolyte.density",
            spec.density,
            domain,
            unit=DENSITY,
            doc="Density",
        )
        viscosity = expression_or_parameter(
            f"{side}_electrolyte.viscosity",
            spec.viscosity,
            domain,
            unit=VISCOSITY,
            doc="Dynamic viscosity",
        )
        return density, viscosity

    def _create_flow(
        self,
        electrodes: typing.Mapping[str, PorousElectrode],
        properties: typing.Mapping[str, tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]],
        degree: int,
    ) -> None:
        """Create the inlet velocity and the flow of the flow's specification.

        Args:
            electrodes: The electrodes, by side.
            properties: The density and the viscosity of each side's
                electrolyte.
            degree: The degree of the Lagrange elements of the unknowns.
        """
        spec, name, domain = self._spec, self._name, self._domain
        if spec is None:
            self._flow = self._inlet_velocity = None
            return
        self._inlet_velocity = expression_or_parameter(
            f"{name}.inlet_velocity",
            spec.inlet_velocity,
            domain,
            unit=syu.m / syu.second,
            doc="Superficial velocity at the inlets",
        )
        outlet_pressure = expression_or_parameter(
            f"{name}.outlet_pressure",
            spec.outlet_pressure,
            domain,
            unit=syu.pascal,
            doc="Gauge pressure at the outlets",
        )
        setup = FlowSetup(
            name=name,
            domain=domain,
            electrolyte=self._subdomains["Electrolyte"],
            electrodes=types.MappingProxyType(dict(electrodes)),
            subdomains=types.MappingProxyType(
                {
                    side: self._subdomains[f"{side.capitalize()}Electrode"]
                    for side in SIDES
                }
            ),
            viscosities=types.MappingProxyType(
                {side: viscosity for side, (_, viscosity) in properties.items()}
            ),
            densities=types.MappingProxyType(
                {side: density for side, (density, _) in properties.items()}
            ),
            inlets=types.MappingProxyType(
                {side: f"{side.capitalize()}Inlet" for side in SIDES}
            ),
            outlets=types.MappingProxyType(
                {side: f"{side.capitalize()}Outlet" for side in SIDES}
            ),
            inlet_velocity=self._inlet_velocity,
            outlet_pressure=outlet_pressure,
            degree=degree,
        )
        self._flow = spec.model(setup)

    def _create_half_cell(
        self,
        side: str,
        spec: HalfCellSpec,
        element: tuple[str, int],
        electrode: PorousElectrode,
        density: ufl.core.expr.Expr,
        viscosity: ufl.core.expr.Expr,
    ) -> RedoxHalfCell:
        """Create the half-cell of a side, with its unknowns on their subdomains.

        Args:
            side: ``"negative"`` or ``"positive"``.
            spec: The side.
            element: The element of the unknowns.
            electrode: The electrode of the side.
            density: The density of the side's electrolyte.
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
        flow = self._flow
        return RedoxHalfCell(
            electrode_domain,
            spec.ions,
            spec.reservoir,
            spec.couple,
            electrode,
            velocity=None if flow is None else flow.velocity(side),
            fiber_distance=spec.fiber_distance,
            collector=None,
            membrane=None,
            outlet=() if flow is None else (f"{prefix}Outlet",),
            density=density,
            viscosity=viscosity,
            temperature=spec.temperature,
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
        """Create the models of the flow and the inflow of the ions.

        Returns:
            The models of the flow, and the inflow of each ion of each
            electrode, :eq:`full-cell-flow`, empty without flow.
        """
        if self._flow is None:
            return []
        assert self._inlet_velocity is not None
        name, velocity = self._name, self._inlet_velocity
        models: list[Residual] = list(self._flow.models)
        for side, cell in self._half_cells.items():
            inlet = (f"{side.capitalize()}Inlet",)
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
        """Create the integrals of the voltage, the amounts and the tanks.

        Returns:
            The areas and the potential integrals of the terminals, the
            amount of each ion in each electrode, the volume of electrolyte
            in its pores, and with flow, the area of its inlet and the
            outflow of each ion, by name.
        """
        functionals: dict[str, ufl.Form] = {}
        for side, cell in self._half_cells.items():
            prefix = side.capitalize()
            ds = cell.phi_s.domain.ds(f"{prefix}Terminal")
            functionals[f"{side}_area"] = 1.0 * ds
            functionals[f"{side}_potential"] = cell.phi_s * ds
            electrode = self._subdomains[f"{prefix}Electrode"]
            porosity = cell.electrode.porosity
            functionals[f"{side}_pore_volume"] = porosity * electrode.dx()
            for ion, c in cell.concentrations.items():
                functionals[f"{side}_amount_{ion}"] = (
                    porosity * c * c.domain.dx(f"{prefix}Electrode")
                )
            if self._flow is None:
                continue
            functionals[f"{side}_inlet_area"] = 1.0 * electrode.ds(f"{prefix}Inlet")
            u = self._flow.velocity(side)
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
        flow, the models of the flow, e.g. those of :class:`DarcyFlow
        <sumfem.electrochemistry.DarcyFlow>`, and the inflow of each ion of
        each electrode, ``"<name>_<side>_inflow_<ion>"``.
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
    def flow(self) -> ElectrodeFlow | None:
        """The flow of the electrolytes, if any."""
        return self._flow

    @property
    def pressure(self) -> Field | None:
        """The pressure of the flow over the electrodes and the membrane,
        ``None`` without flow."""
        return None if self._flow is None else self._flow.pressure

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
        potentials, and with flow, those of the flow, e.g. the outlet
        pressure of :class:`DarcyFlow <sumfem.electrochemistry.DarcyFlow>`.
        """
        conditions = [
            DirichletBC(self._half_cells["negative"].phi_s, "NegativeTerminal", 0.0)
        ]
        if self._flow is not None:
            conditions.extend(self._flow.conditions())
        return conditions

    def initialize(self, discretization: Discretization) -> None:
        """Set a state at rest, as Shah et al. start.

        The concentrations start at the reservoir values of the sides, the
        defaults of their parameters ``"<side>.c0_<ion>"``, the electrolyte
        potential at zero, the solid of each side at the equilibrium
        potential of its couple at those concentrations, by Nernst's
        equation, so that no current flows, and the flow as it initializes
        itself, see :meth:`ElectrodeFlow.initialize
        <sumfem.electrochemistry.ElectrodeFlow.initialize>`.

        Raises:
            ValueError: If the sides start with different water
                concentrations, which one water field cannot hold.
        """
        values: dict[str | ufl.Coefficient, float] = {self._phi_l: 0.0}
        for cell in self._half_cells.values():
            for ion, c in cell.concentrations.items():
                values[c] = cell.reservoir_concentration(ion)
            values[cell.phi_s] = cell.equilibrium_potential(discretization)
        if self._water is not None:
            start = {
                c.reservoir_concentration(WATER) for c in self._half_cells.values()
            }
            if len(start) != 1:
                raise ValueError(
                    "Both sides must start with the same water concentration, got "
                    f"{sorted(start)}."
                )
        discretization.set(values)
        if self._flow is not None:
            self._flow.initialize(discretization)

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
        if self._spec is None or self._spec.tank_ratio is None:
            raise ValueError(f"The cell {self._name!r} has no tanks.")
        pores = discretization.assemble_scalar(self._functionals[f"{side}_pore_volume"])
        return self._spec.tank_ratio * pores

    def amount(self, discretization: Discretization, side: str, ion: str) -> float:
        """The amount of an ion in a side's electrode and its tank.

        The amounts are per unit of the dimensions the mesh lacks, e.g. per
        metre of depth for a 2D cell.

        Args:
            discretization: The discretization of the cell.
            side: ``"negative"`` or ``"positive"``.
            ion: The name of an ion of the side with a field. The water of
                the membrane is in neither side's amount.

        Returns:
            The amount in mol, that of the tank included with tanks.
        """
        cell = self._half_cells[side]
        amount = discretization.assemble_scalar(
            self._functionals[f"{side}_amount_{ion}"]
        )
        if self._spec is not None and self._spec.tank_ratio is not None:
            tank = float(discretization.value(cell.inlet[ion]))
            amount += self.tank_volume(discretization, side) * tank
        return amount

    def state_of_charge(self, discretization: Discretization, side: str) -> float:
        """The state of charge of a side's couple, its tank included.

        Args:
            discretization: The discretization of the cell.
            side: ``"negative"`` or ``"positive"``.

        Returns:
            The share of the ion that charging makes: the reduced ion of a
            couple oxidized on discharge, the oxidized ion of one reduced on
            discharge.
        """
        couple = self._half_cells[side].couple
        charged, other = (
            (couple.reduced, couple.oxidized)
            if couple.oxidation_on_discharge
            else (couple.oxidized, couple.reduced)
        )
        made = self.amount(discretization, side, charged)
        return made / (made + self.amount(discretization, side, other))

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
        if self._spec is None or self._spec.tank_ratio is None:
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
