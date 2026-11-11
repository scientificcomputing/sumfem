"""Half-cells of a redox flow battery.

:class:`RedoxHalfCell` is a half-cell of any redox couple of dissolved ions,
negative or positive: a porous electrode, optionally beside free channels,
filled with an electrolyte that flows from a reservoir, with charge
conservation in the solid and the electrolyte, transport of every ion, and
Butler-Volmer kinetics of the couple at the fibres.
"""

from __future__ import annotations

import dataclasses
import types
import typing

import sympy
import sympy.physics.units as syu
import ufl

from sumfem.discrete import Discretization
from sumfem.models import Accumulation, BoundaryFlux, Conduction, Outflow
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    ResidualSum,
)

from .charge import ElectrolyteConduction
from .couple import RedoxCouple
from .electrolyte import CONCENTRATION, Electrolyte, Ion
from .kinetics import (
    ButlerVolmer,
    MassTransferButlerVolmer,
    PistonVelocityButlerVolmer,
)
from .porous import PorousElectrode, Region
from .transport import NernstPlanck

__all__ = ["RedoxHalfCell"]


class RedoxHalfCell:
    """A half-cell of a redox flow battery, of a couple of dissolved ions.

    The macro-homogeneous porous-electrode model of RFBniCS, `Aghabarari et
    al., 2026 <https://arxiv.org/abs/2610.01840>`_, Section II, for any
    couple, electrolyte and sign of the half-cell.

    The unknowns are the potentials ``phi_s`` of the solid and ``phi_l`` of
    the electrolyte, in V, and the concentration ``c_<ion>`` of every ion
    with a reservoir concentration, in mol/m^3. The one ion without follows
    from electroneutrality. In the porous electrode, the reaction of the
    :class:`~sumfem.electrochemistry.RedoxCouple` follows Butler-Volmer
    kinetics, limited by mass transfer to the fibres when there is flow or a
    fibre distance. In free channels the electrolyte is not corrected for
    porosity, and there is no reaction or solid phase.

    The model, :attr:`model`, is the sum of:

    - ``"<name>_solid"``: Ohmic conduction in the solid,
      :class:`~sumfem.models.Conduction` with Bruggeman's effective
      conductivity of the electrode;
    - ``"<name>_reaction"``: the kinetics of the couple,
      :class:`~sumfem.electrochemistry.ButlerVolmer`,
      :class:`~sumfem.electrochemistry.MassTransferButlerVolmer` with mass
      transfer, or :class:`~sumfem.electrochemistry.PistonVelocityButlerVolmer`
      with a fibre distance, which moves the current between the phases and
      consumes and produces the ions of the reaction;
    - with collector and membrane markers, ``"<name>_collector_current"``
      and ``"<name>_membrane_current"``: the current of the cell into the
      solid and out of the electrolyte, :class:`~sumfem.models.BoundaryFlux`;
    - for the electrode's pores and, with a suffix ``"_channels"``, the free
      channels: charge conservation in the electrolyte,
      :class:`~sumfem.electrochemistry.ElectrolyteConduction`, and for every
      transported ion its storage, :class:`~sumfem.models.Accumulation`
      weighted with the porosity, and its transport,
      :class:`~sumfem.electrochemistry.NernstPlanck`, named
      ``"<name>_storage_<ion>"`` and ``"<name>_transport_<ion>"``;
    - with an outlet, ``"<name>_outflow_<ion>"``: advection out of the cell,
      :class:`~sumfem.models.Outflow`;
    - with a proton flux, ``"<name>_proton_flux"``: the membrane current
      carried by protons, a :class:`~sumfem.models.BoundaryFlux` of the ion
      ``"H"``.

    The cell is galvanostatic: the parameter ``"<name>.current"`` is the
    current density through the membrane, in A/m^2, positive on discharge.
    On discharge the couple is oxidized in the negative half-cell and
    reduced in the positive one, so the current enters the solid at the
    current collector of the first, and leaves it at that of the second.
    The current collector carries the current density ``current *
    area_ratio``, so that the total currents match when the two areas
    differ.

    Args:
        domain: The domain of the cell.
        ions: The ions of the electrolyte, without fields: the cell creates
            the field of every ion that has a reservoir concentration.
        reservoir: The concentration of each transported ion in the
            reservoir, in mol/m^3 or a quantity with units, the default of
            the parameter ``"<name>.c0_<ion>"``.
        couple: The redox couple and its reaction. Its parameters are those
            of the cell: ``"<name>.rate_constant"``,
            ``"<name>.formal_potential"``, ``"<name>.anodic_transfer"`` and
            ``"<name>.cathodic_transfer"``.
        electrode: The porous electrode.
        channels: Cell markers of free channels. Empty for none.
        velocity: Superficial velocity of the electrolyte, or ``None``
            without flow.
        mass_transfer: Limit the kinetics by mass transfer to the fibres,
            which needs a velocity. Defaults to whether there is one.
        fiber_distance: Mean distance between the fibres, in m or with
            units. Limits the kinetics by the piston velocities of
            :class:`~sumfem.electrochemistry.PistonVelocityButlerVolmer`
            instead, with or without flow.
        collector: Facet markers of the current collector, or ``None`` for
            a cell whose collector is resolved, e.g. in a full cell.
        membrane: Facet markers of the membrane, or ``None`` for a cell
            whose membrane is resolved.
        outlet: Facet markers of an outlet, through which ions leave by
            advection. Empty for none.
        proton_flux: Carry the membrane current by protons, the ion ``"H"``,
            through the membrane.
        density: Density of the electrolyte, see
            :class:`~sumfem.electrochemistry.Electrolyte`. Pass the
            parameter of a flow model to share it.
        viscosity: Dynamic viscosity of the electrolyte, given like the
            density.
        degree: Degree of the Lagrange elements of the unknowns.
        name: Prefix of the cell's own parameters, ``"<name>.<parameter>"``,
            and of the names of its models, ``"<name>_<model>"``.
        phi_s: The potential of the solid, if it exists, e.g. over the
            electrode and its current collector. Created on the domain
            otherwise.
        phi_l: The potential of the electrolyte, if it exists, e.g. over the
            electrodes and the membrane of a full cell.
        fields: Existing fields of transported ions, by name, e.g. of water
            shared with a membrane. The others are created on the domain.
        electrolyte_name: Prefix of the parameters of the electrolyte, see
            :class:`~sumfem.electrochemistry.Electrolyte`.

    Raises:
        ValueError: For mass transfer or an outlet without a velocity, a
            proton flux without membrane markers, a reservoir concentration
            of an ion that the electrolyte does not have, a reservoir that
            lacks not exactly one ion, an ion of the reaction without one, a
            reaction that does not conserve charge, or a balanced ion
            without charge.
    """

    _domain: Domain
    """The domain of the cell."""

    _name: str
    """Prefix of the cell's parameters and models."""

    _couple: RedoxCouple
    """The redox couple and its reaction."""

    _electrode: PorousElectrode
    """The porous electrode."""

    _regions: tuple[tuple[str, Region], ...]
    """The regions filled with electrolyte: the electrode's pores, and the
    free channels if any. Each comes with the suffix of its models' names."""

    _electrolyte: Electrolyte
    """The electrolyte."""

    _phi_s: Field
    """Potential of the solid."""

    _phi_l: Field
    """Potential of the electrolyte."""

    _concentrations: types.MappingProxyType[str, Field]
    """Concentration of each transported ion, by ion name."""

    _inlet: types.MappingProxyType[str, Parameter]
    """Reservoir concentration of each transported ion, by ion name."""

    _current: Parameter
    """Current density through the membrane, positive on discharge."""

    _area_ratio: Parameter
    """Ratio of the membrane area to the current collector area."""

    _formal_potential: Parameter
    """Formal potential of the couple."""

    _reaction: ButlerVolmer
    """The reaction in the electrode."""

    _model: ResidualSum
    """The sum of all models of the cell."""

    _collector: tuple[str, ...] | None
    """Facet markers of the current collector, if not resolved."""

    _membrane: tuple[str, ...] | None
    """Facet markers of the membrane, if not resolved."""

    _functionals: types.MappingProxyType[str, ufl.Form]
    """The integrals behind the cell's diagnostics, created once so that a
    discretization compiles each of them only once."""

    def __init__(
        self,
        domain: Domain,
        ions: typing.Sequence[Ion],
        reservoir: typing.Mapping[str, float | sympy.Expr],
        couple: RedoxCouple,
        electrode: PorousElectrode,
        channels: tuple[str, ...] = (),
        velocity: ufl.core.expr.Expr | None = None,
        mass_transfer: bool | None = None,
        fiber_distance: float | sympy.Expr | None = None,
        collector: tuple[str, ...] | None = ("CurrentCollector",),
        membrane: tuple[str, ...] | None = ("Membrane",),
        outlet: tuple[str, ...] = (),
        proton_flux: bool = False,
        density: ufl.core.expr.Expr | float = 1399.8,
        viscosity: ufl.core.expr.Expr | float = 6.963e-3,
        degree: int = 1,
        name: str = "cell",
        phi_s: Field | None = None,
        phi_l: Field | None = None,
        fields: typing.Mapping[str, Field] | None = None,
        electrolyte_name: str = "electrolyte",
    ):
        # Mass transfer to the fibres depends on the speed of the flow, so it
        # is on by default whenever there is one; piston velocities take the
        # place of the flow's correlation
        if mass_transfer is None:
            mass_transfer = velocity is not None
        if fiber_distance is not None:
            mass_transfer = False
        if velocity is None and (mass_transfer or outlet):
            raise ValueError("Mass transfer and an outlet need a velocity.")
        if proton_flux and membrane is None:
            raise ValueError("A proton flux through the membrane needs its markers.")
        _check_reservoir(ions, reservoir, couple)

        self._domain = domain
        self._name = name
        self._couple = couple
        self._electrode = electrode
        self._collector = None if collector is None else tuple(collector)
        self._membrane = None if membrane is None else tuple(membrane)
        # The regions filled with electrolyte, with the suffix of their models:
        # the pores of the electrode, whose porosity weights storage and
        # corrects transport, and free channels, of porosity one
        regions = [("", electrode.region)]
        if channels:
            regions.append(("_channels", Region(tuple(channels))))
        self._regions = tuple(regions)

        self._create_fields(ions, reservoir, degree, phi_s, phi_l, fields)
        self._electrolyte = self._create_electrolyte(
            ions, electrolyte_name, density, viscosity
        )
        self._create_parameters(reservoir)
        self._reaction = self._create_reaction(velocity, mass_transfer, fiber_distance)
        self._model = ResidualSum(
            self._create_models(velocity, outlet, proton_flux), name=name
        )
        self._functionals = self._create_functionals()

    def _create_fields(
        self,
        ions: typing.Sequence[Ion],
        reservoir: typing.Mapping[str, float | sympy.Expr],
        degree: int,
        phi_s: Field | None,
        phi_l: Field | None,
        fields: typing.Mapping[str, Field] | None,
    ) -> None:
        """The unknowns: the potentials, and the concentration of every ion
        with a reservoir concentration. Existing fields, e.g. shared with a
        membrane, are used as given. A concentration is of the order of a mole
        per litre, its scale for the dimensional analysis and the scaled
        equations."""
        domain = self._domain
        element = ("Lagrange", degree)
        existing = dict(fields or {})
        self._phi_s = phi_s or Field("phi_s", domain, element, unit=syu.volt)
        self._phi_l = phi_l or Field("phi_l", domain, element, unit=syu.volt)
        self._concentrations = types.MappingProxyType(
            {
                ion.name: existing.get(ion.name)
                or Field(
                    f"c_{ion.name}", domain, element, unit=CONCENTRATION, scale=1e3
                )
                for ion in ions
                if ion.name in reservoir
            }
        )

    def _create_electrolyte(
        self,
        ions: typing.Sequence[Ion],
        name: str,
        density: ufl.core.expr.Expr | float,
        viscosity: ufl.core.expr.Expr | float,
    ) -> Electrolyte:
        """The electrolyte, with the field of every transported ion. The ion
        without a reservoir concentration, and so without a field, follows
        from electroneutrality. Its density and viscosity enter the mass
        transfer, and may be shared with a model of the flow."""
        return Electrolyte(
            name,
            self._domain,
            [
                dataclasses.replace(
                    ion, concentration=self._concentrations.get(ion.name)
                )
                for ion in ions
            ],
            density=density,
            viscosity=viscosity,
        )

    def _create_parameters(
        self, reservoir: typing.Mapping[str, float | sympy.Expr]
    ) -> None:
        """The parameters of the cell: the reservoir concentrations, the
        initial state and the values of the conditions at an inlet; the
        current density through the membrane, and the ratio of the areas
        that makes the current through the collector equal, see
        :meth:`balance_currents`; and the formal potential of the couple."""
        domain, name, couple = self._domain, self._name, self._couple
        self._inlet = types.MappingProxyType(
            {
                ion: Parameter(
                    f"{name}.c0_{ion}",
                    domain,
                    default=value,
                    unit=CONCENTRATION,
                    doc=f"Reservoir concentration of {ion}",
                )
                for ion, value in reservoir.items()
            }
        )
        self._current = Parameter(
            f"{name}.current",
            domain,
            default=0.0,
            unit=syu.ampere / syu.m**2,
            doc="Current density through the membrane, positive on discharge",
        )
        self._area_ratio = Parameter(
            f"{name}.area_ratio",
            domain,
            default=1.0,
            doc="Membrane area over current collector area",
        )
        self._formal_potential = Parameter(
            f"{name}.formal_potential",
            domain,
            default=couple.formal_potential,
            unit=syu.volt,
            doc=f"Formal potential of {couple.reduced}/{couple.oxidized}",
        )

    def _create_reaction(
        self,
        velocity: ufl.core.expr.Expr | None,
        mass_transfer: bool,
        fiber_distance: float | sympy.Expr | None,
    ) -> ButlerVolmer:
        """The reaction in the electrode: Butler-Volmer at the concentrations
        of the pores, or at the surface of the fibres with mass transfer, by
        the flow's correlation or by diffusion between the fibres. Its
        parameters are those of the cell."""
        domain, name, couple = self._domain, self._name, self._couple
        pair = f"{couple.reduced}/{couple.oxidized}"
        common: dict[str, typing.Any] = dict(
            specific_area=self._electrode.specific_area,
            formal_potential=self._formal_potential,
            rate_constant=Parameter(
                f"{name}.rate_constant",
                domain,
                default=couple.rate_constant,
                unit=syu.m / syu.second,
                doc=f"Standard rate constant of {pair}",
            ),
            anodic_transfer=Parameter(
                f"{name}.anodic_transfer",
                domain,
                default=couple.anodic_transfer,
                doc=f"Anodic transfer coefficient of {pair}",
            ),
            cathodic_transfer=Parameter(
                f"{name}.cathodic_transfer",
                domain,
                default=couple.cathodic_transfer,
                doc=f"Cathodic transfer coefficient of {pair}",
            ),
            electrons=couple.electrons,
            markers=self._electrode.markers,
            reactants=couple.reactants,
            products=couple.products,
        )
        arguments = (
            f"{name}_reaction",
            self._phi_s,
            self._phi_l,
            self._electrolyte,
            couple.reduced,
            couple.oxidized,
        )
        if fiber_distance is not None:
            distance = Parameter(
                f"{name}.fiber_distance",
                domain,
                default=fiber_distance,
                unit=syu.m,
                doc="Mean distance between the fibres",
            )
            return PistonVelocityButlerVolmer(
                *arguments, fiber_distance=distance, **common
            )
        if mass_transfer:
            assert velocity is not None
            return MassTransferButlerVolmer(
                *arguments,
                velocity,
                fiber_diameter=self._electrode.fiber_diameter,
                **common,
            )
        return ButlerVolmer(*arguments, **common)

    def _create_models(
        self,
        velocity: ufl.core.expr.Expr | None,
        outlet: tuple[str, ...],
        proton_flux: bool,
    ) -> list[Residual]:
        """The models of the cell, see the class documentation."""
        name, e = self._name, self._electrolyte
        # The current into the solid at the collector, and out of the
        # electrolyte at the membrane, on discharge. Oxidation on discharge
        # releases electrons into the solid, which leave through the
        # collector, so the current of the cell enters there; reduction on
        # discharge takes them, so the current leaves the solid there
        current = self._current
        inflow = current if self._couple.oxidation_on_discharge else -current
        models: list[Residual] = [
            Conduction(
                f"{name}_solid",
                self._phi_s,
                self._electrode.effective_conductivity(),
                markers=self._electrode.markers,
            ),
            self._reaction,
        ]
        if self._collector is not None:
            collector_current = inflow * self._area_ratio
            models.append(
                BoundaryFlux(
                    f"{name}_collector_current",
                    self._phi_s,
                    collector_current,
                    self._collector,
                )
            )
        if self._membrane is not None:
            models.append(
                BoundaryFlux(
                    f"{name}_membrane_current", self._phi_l, -inflow, self._membrane
                )
            )
        # Every region filled with electrolyte conserves charge, and stores
        # and transports every ion, with the porosity of the region
        for suffix, region in self._regions:
            models.append(
                ElectrolyteConduction(
                    f"{name}_electrolyte{suffix}", self._phi_l, e, region
                )
            )
            for ion, c in self._concentrations.items():
                models.append(
                    Accumulation(
                        f"{name}_storage_{ion}{suffix}",
                        c,
                        weight=region.porosity,
                        markers=region.markers,
                    )
                )
                models.append(
                    NernstPlanck(
                        f"{name}_transport_{ion}{suffix}",
                        e[ion],
                        e,
                        self._phi_l,
                        region,
                        velocity,
                    )
                )
        # The ions that the flow carries out of the cell
        if outlet:
            assert velocity is not None
            for ion, c in self._concentrations.items():
                models.append(Outflow(f"{name}_outflow_{ion}", c, velocity, outlet))
        # The membrane passes the current as protons: their flux is the
        # current over their charge and Faraday's constant
        if proton_flux:
            assert self._membrane is not None
            models.append(
                BoundaryFlux(
                    f"{name}_proton_flux",
                    self._concentrations["H"],
                    -inflow / (e["H"].charge * e.F),
                    self._membrane,
                )
            )
        return models

    def _create_functionals(self) -> types.MappingProxyType[str, ufl.Form]:
        """The integrals of the diagnostics: the areas and mean potentials of
        the collector and the membrane, for the voltage, and the amounts of
        the couple in the electrolyte of every region, for the state of
        charge."""
        domain = self._domain
        functionals: dict[str, ufl.Form] = {}
        if self._collector is not None:
            ds_collector = domain.ds(*self._collector)
            functionals["collector_area"] = 1.0 * ds_collector
            functionals["collector_potential"] = self._phi_s * ds_collector
        if self._membrane is not None:
            ds_membrane = domain.ds(*self._membrane)
            functionals["membrane_area"] = 1.0 * ds_membrane
            functionals["membrane_potential"] = self._phi_l * ds_membrane
        for role, ion in (
            ("reduced", self._couple.reduced),
            ("oxidized", self._couple.oxidized),
        ):
            c = self._concentrations[ion]
            functionals[f"amount_{role}"] = sum(
                region.storage * c * domain.dx(*region.markers)
                for _, region in self._regions
            )
        return types.MappingProxyType(functionals)

    @property
    def couple(self) -> RedoxCouple:
        """The redox couple and its reaction."""
        return self._couple

    @property
    def model(self) -> Residual:
        """The sum of all models of the cell."""
        return self._model

    @property
    def electrode(self) -> PorousElectrode:
        """The porous electrode."""
        return self._electrode

    @property
    def electrolyte(self) -> Electrolyte:
        """The electrolyte."""
        return self._electrolyte

    @property
    def reaction(self) -> ButlerVolmer:
        """The reaction in the electrode."""
        return self._reaction

    @property
    def phi_s(self) -> Field:
        """Potential of the solid."""
        return self._phi_s

    @property
    def phi_l(self) -> Field:
        """Potential of the electrolyte."""
        return self._phi_l

    @property
    def concentrations(self) -> typing.Mapping[str, Field]:
        """Concentration of each transported ion, by ion name."""
        return self._concentrations

    @property
    def inlet(self) -> typing.Mapping[str, Parameter]:
        """Reservoir concentration of each transported ion, by ion name."""
        return self._inlet

    @property
    def current(self) -> Parameter:
        """Current density through the membrane, positive on discharge."""
        return self._current

    @property
    def area_ratio(self) -> Parameter:
        """Ratio of the membrane area to the current collector area."""
        return self._area_ratio

    def inlet_conditions(self, markers: str | tuple[str, ...]) -> list[DirichletBC]:
        """The reservoir concentration of every transported ion on some facets."""
        return [
            DirichletBC(c, markers, self._inlet[ion])
            for ion, c in self._concentrations.items()
        ]

    def initialize(self, discretization: Discretization) -> None:
        """Set a state at rest, as RFBniCS starts.

        The concentrations start at their reservoir values, the solid at zero
        and the electrolyte at minus the formal potential.
        """
        values: dict[str | ufl.Coefficient, typing.Any] = {
            c: discretization.value(self._inlet[ion])
            for ion, c in self._concentrations.items()
        }
        values[self._phi_s] = 0.0
        values[self._phi_l] = -discretization.value(self._formal_potential)
        discretization.set(values)

    def _check_boundaries(self, what: str) -> None:
        if self._collector is None or self._membrane is None:
            raise ValueError(
                f"The {what} needs the facet markers of the current collector and "
                "the membrane; this cell resolves them."
            )

    def balance_currents(self, discretization: Discretization) -> None:
        """Set the area ratio from the areas of membrane and current collector.

        Raises:
            ValueError: For a cell without collector or membrane markers.
        """
        self._check_boundaries("area ratio")
        assemble = discretization.assemble_scalar
        ratio = assemble(self._functionals["membrane_area"]) / assemble(
            self._functionals["collector_area"]
        )
        discretization.set({self._area_ratio: ratio})

    def voltage(self, discretization: Discretization) -> float:
        """The half-cell voltage in V.

        That is the mean solid potential on the current collector minus the
        mean electrolyte potential on the membrane.

        Raises:
            ValueError: For a cell without collector or membrane markers.
        """
        self._check_boundaries("voltage")
        assemble = discretization.assemble_scalar
        f = self._functionals
        return assemble(f["collector_potential"]) / assemble(
            f["collector_area"]
        ) - assemble(f["membrane_potential"]) / assemble(f["membrane_area"])

    def state_of_charge(self, discretization: Discretization) -> float:
        """The share of the reduced ion in the couple, over the cell."""
        assemble = discretization.assemble_scalar
        reduced = assemble(self._functionals["amount_reduced"])
        oxidized = assemble(self._functionals["amount_oxidized"])
        return reduced / max(reduced + oxidized, 1e-30)

    def local_state_of_charge(self) -> ufl.core.expr.Expr:
        """The share of the reduced ion in the couple, ``c_red / (c_red + c_ox)``."""
        c_red = self._concentrations[self._couple.reduced]
        c_ox = self._concentrations[self._couple.oxidized]
        return c_red / (c_red + c_ox + self._electrolyte.floor)


def _check_reservoir(
    ions: typing.Sequence[Ion],
    reservoir: typing.Mapping[str, float | sympy.Expr],
    couple: RedoxCouple,
) -> None:
    """Check the reaction and the reservoir against the ions.

    The reservoir names the transported ions: all but one, which follows
    from electroneutrality and closes the equations. Every ion of the
    reaction is transported.
    """
    couple.check(ions)
    unknown = sorted(set(reservoir) - {ion.name for ion in ions})
    if unknown:
        raise ValueError(f"The electrolyte has no ions {unknown}.")
    balanced = [ion.name for ion in ions if ion.name not in reservoir]
    if len(balanced) != 1:
        raise ValueError(
            "The reservoir must give the concentrations of all ions but one, "
            "whose concentration follows from electroneutrality; it lacks "
            f"{balanced}."
        )
    for ion in couple.species:
        if ion not in reservoir:
            raise ValueError(
                f"The ion {ion!r} of the reaction needs a reservoir concentration."
            )
