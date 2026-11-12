"""The full cell of a redox flow battery: two half-cells and a membrane.

:class:`FullCell` builds a cell of two :class:`RedoxHalfCell` and a
:class:`ProtonExchangeMembrane` between them, with a current collector on
the outer side of each electrode, on one mesh, each unknown on the part of
the mesh where it lives. Each side is given by a :class:`HalfCellSpec`.
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
from sumfem.models import BoundaryFlux, Conduction
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    ResidualSum,
    Subdomain,
    expression_or_parameter,
)

from .couple import RedoxCouple
from .electrolyte import CONCENTRATION, Ion
from .half_cell import RedoxHalfCell
from .membrane import PROPERTY, ProtonExchangeMembrane
from .porous import PorousElectrode

__all__ = ["FullCell", "HalfCellSpec", "MembraneSpec"]

SIDES = ("negative", "positive")
"""The sides of the cell, in the order of the cell's layers."""

WATER = "H2O"
"""Name of the water of the electrolytes, shared by both sides."""

PROTONS = "H"
"""Name of the protons, which carry the current through the membrane."""


@dataclasses.dataclass(frozen=True)
class HalfCellSpec:
    """One side of a full cell: its electrolyte, its couple and its electrode.

    Attributes:
        ions: The ions of the electrolyte, see :class:`RedoxHalfCell`.
            Water, ``"H2O"``, if transported, must be an ion of both sides,
            since the water moves through the membrane.
        reservoir: The initial concentration of each transported ion, see
            :class:`RedoxHalfCell`. The protons, ``"H"``, must be one of
            them.
        couple: The redox couple of the side.
        porosity: The porosity of the electrode.
        conductivity: The conductivity of the electrode's fibres, in S/m or
            with units.
        specific_area: The specific area of the electrode, in 1/m or with
            units.
        fiber_distance: The mean distance between the fibres, in m or with
            units, which limits the kinetics by the piston velocities, see
            :class:`~sumfem.electrochemistry.PistonVelocityButlerVolmer`.
            ``None`` for kinetics without mass transfer.
    """

    ions: tuple[Ion, ...]
    reservoir: typing.Mapping[str, float | sympy.Expr]
    couple: RedoxCouple
    porosity: float
    conductivity: float | sympy.Expr
    specific_area: float | sympy.Expr
    fiber_distance: float | sympy.Expr | None = None


@dataclasses.dataclass(frozen=True)
class MembraneSpec:
    """The properties of the membrane of a full cell.

    Each is an expression, used as given, or the default of the parameter
    ``"<membrane>.<property>"``, in SI units or with units, ``None`` for a
    parameter without a default, see :class:`ProtonExchangeMembrane`.

    Attributes:
        fixed_charge: The concentration of the fixed charges, in mol/m^3.
        proton_diffusivity: The proton diffusivity in the membrane, in
            m^2/s.
        water_diffusivity: The water diffusivity in the membrane, in m^2/s.
    """

    fixed_charge: PROPERTY = None
    proton_diffusivity: PROPERTY = None
    water_diffusivity: PROPERTY = None


class FullCell:
    r"""The full cell of a redox flow battery, without flow.

    The cell of `Shah et al., 2008
    <https://doi.org/10.1016/j.electacta.2008.05.067>`_, corrected, without
    the flow of the electrolyte: from left to right, a current collector, the
    negative porous electrode, a proton-exchange membrane, the positive
    porous electrode and a current collector. Its mesh has these cell
    markers:

    - ``"NegativeCollector"``, ``"NegativeElectrode"``, ``"Membrane"``,
      ``"PositiveElectrode"`` and ``"PositiveCollector"``, the layers;
    - ``"NegativeSolid"`` and ``"PositiveSolid"``, a collector and its
      electrode, where the potential of that side's solid lives;
    - ``"Electrolyte"``, the electrodes and the membrane, where the
      electrolyte potential and the water live,

    and these facet markers:

    - ``"NegativeTerminal"`` and ``"PositiveTerminal"``, the outer faces of
      the collectors;
    - ``"NegativeFace"`` and ``"PositiveFace"``, the faces of the membrane
      to the electrodes.

    :meth:`mesh_data` adds the unions to mesh data with the layers.

    Each side is a :class:`RedoxHalfCell`, whose ions live in its electrode.
    The electrolyte potential :math:`\phi_l` and the water are one field
    each, over the electrodes and the membrane, so the current and the water
    flux are continuous across the faces. The membrane is a
    :class:`ProtonExchangeMembrane`, and its protons carry the current
    through the faces, :eq:`membrane-proton-flux`, so that no other ion
    crosses them. In particular the ion that follows from electroneutrality,
    e.g. sulfate, stays on its side. This corrects Shah et al., who hold
    the protons at the concentration of the fixed charges at the faces,
    which lets that ion carry part of the current through them.

    The collectors conduct with the conductivity
    :math:`\sigma_\text{coll}`, the parameter
    ``"<name>.collector_conductivity"``. The cell is galvanostatic: the
    current density :math:`j` of the cell, the parameter
    ``"<name>.current"``, positive on discharge, enters the solid of the
    positive side at its terminal on charge and leaves it on discharge, and
    the solid of the negative side is grounded at its terminal,
    :meth:`ground`.

    The model, :attr:`model`, is the sum of the models of the two
    half-cells, named ``"negative_<model>"`` and ``"positive_<model>"``,
    of the membrane, ``"<name>_membrane_<model>"``, of the conduction in
    the collectors, ``"<name>_negative_collector"`` and
    ``"<name>_positive_collector"``, of the current at the positive
    terminal, ``"<name>_terminal_current"``, and of the proton fluxes at the
    faces, ``"<name>_negative_protons"`` and ``"<name>_positive_protons"``.

    Args:
        domain: The domain of the cell, with the markers above.
        negative: The negative side.
        positive: The positive side.
        membrane: The properties of the membrane. Parameters without
            defaults by default.
        collector_conductivity: The conductivity of the collectors: an
            expression, used as given, e.g. one of the temperature, or the
            default of the parameter ``"<name>.collector_conductivity"``, in
            S/m or with units.
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
    """The integrals of the voltage, created once."""

    def __init__(
        self,
        domain: Domain,
        negative: HalfCellSpec,
        positive: HalfCellSpec,
        membrane: MembraneSpec = MembraneSpec(),
        collector_conductivity: PROPERTY = 1000.0,
        degree: int = 1,
        name: str = "cell",
    ):
        specs = dict(zip(SIDES, (negative, positive), strict=True))
        _check_sides(specs)
        self._domain = domain
        self._name = name
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
        self._half_cells = types.MappingProxyType(
            {
                side: self._create_half_cell(side, spec, element)
                for side, spec in specs.items()
            }
        )
        self._membrane = ProtonExchangeMembrane(
            f"{name}_membrane",
            self._phi_l,
            ("Membrane",),
            water=self._water,
            fixed_charge=membrane.fixed_charge,
            proton_diffusivity=membrane.proton_diffusivity,
            water_diffusivity=membrane.water_diffusivity,
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

    def _create_half_cell(
        self, side: str, spec: HalfCellSpec, element: tuple[str, int]
    ) -> RedoxHalfCell:
        """Create the half-cell of a side, with its unknowns on their subdomains.

        Args:
            side: ``"negative"`` or ``"positive"``.
            spec: The side.
            element: The element of the unknowns.

        Returns:
            The half-cell, whose collector and membrane are resolved.
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
        electrode = PorousElectrode(
            f"{side}_electrode",
            electrode_domain,
            (f"{prefix}Electrode",),
            porosity=spec.porosity,
            conductivity=spec.conductivity,
            specific_area=spec.specific_area,
        )
        return RedoxHalfCell(
            electrode_domain,
            spec.ions,
            spec.reservoir,
            spec.couple,
            electrode,
            fiber_distance=spec.fiber_distance,
            collector=None,
            membrane=None,
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
        return models

    def _create_functionals(self) -> types.MappingProxyType[str, ufl.Form]:
        """Create the integrals of the voltage.

        Returns:
            The areas and the potential integrals of the terminals, by name.
        """
        functionals: dict[str, ufl.Form] = {}
        for side, cell in self._half_cells.items():
            ds = cell.phi_s.domain.ds(f"{side.capitalize()}Terminal")
            functionals[f"{side}_area"] = 1.0 * ds
            functionals[f"{side}_potential"] = cell.phi_s * ds
        return types.MappingProxyType(functionals)

    @staticmethod
    def mesh_data(marked: MeshData) -> MeshData:
        """Mesh data with the unions of the layers the cell's unknowns live on.

        Args:
            marked: Mesh data whose cell markers are the five layers, and
                whose facet markers are the terminals and the faces, see
                :class:`FullCell`.

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
        """The sum of all models of the cell."""
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
    def current(self) -> Parameter:
        """Current density of the cell, positive on discharge."""
        return self._current

    def subdomain(self, marker: str) -> Subdomain:
        """The subdomain of a cell marker of the cell, e.g. ``"Membrane"``."""
        return self._subdomains[marker]

    def ground(self) -> list[DirichletBC]:
        """The negative terminal at zero potential, which fixes the potentials."""
        return [
            DirichletBC(self._half_cells["negative"].phi_s, "NegativeTerminal", 0.0)
        ]

    def initialize(self, discretization: Discretization) -> None:
        """Set a state at rest, as Shah et al. start.

        The concentrations start at the reservoir values of the sides, the
        defaults of their parameters ``"<side>.c0_<ion>"``, the electrolyte
        potential at zero, and the solid of each side at the equilibrium
        potential of its couple at those concentrations, by Nernst's
        equation, so that no current flows.

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
