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
from .electrolyte import CONCENTRATION, Electrolyte, Ion
from .kinetics import ButlerVolmer, MassTransferButlerVolmer
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
    from electroneutrality. In the porous electrode, the couple
    ``reduced <-> oxidized + n e-`` follows Butler-Volmer kinetics, limited
    by mass transfer to the fibres when there is flow. In free channels the
    electrolyte is not corrected for porosity, and there is no reaction or
    solid phase.

    The model, :attr:`model`, is the sum of:

    - ``"solid"``: Ohmic conduction in the solid,
      :class:`~sumfem.models.Conduction` with Bruggeman's effective
      conductivity of the electrode;
    - ``"reaction"``: the kinetics of the couple,
      :class:`~sumfem.electrochemistry.ButlerVolmer`, or
      :class:`~sumfem.electrochemistry.MassTransferButlerVolmer` with mass
      transfer, which moves the current between the phases and consumes and
      produces the ions of the couple;
    - ``"collector_current"`` and ``"membrane_current"``: the current of the
      cell into the solid and out of the electrolyte,
      :class:`~sumfem.models.BoundaryFlux`;
    - for the electrode's pores and, with a suffix ``"_channels"``, the free
      channels: charge conservation in the electrolyte,
      :class:`~sumfem.electrochemistry.ElectrolyteConduction`, and for every
      transported ion its storage, :class:`~sumfem.models.Accumulation`
      weighted with the porosity, and its transport,
      :class:`~sumfem.electrochemistry.NernstPlanck`, named
      ``"storage_<ion>"`` and ``"transport_<ion>"``;
    - with an outlet, ``"outflow_<ion>"``: advection out of the cell,
      :class:`~sumfem.models.Outflow`;
    - with a proton flux, ``"proton_flux"``: the membrane current carried by
      protons, a :class:`~sumfem.models.BoundaryFlux` of the ion ``"H"``.

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
        reduced: Name of the reduced ion of the couple. With ``oxidized``
            and ``electrons`` it gives the reaction
            ``oxidized + n e- <-> reduced``, e.g. V3 + e- <-> V2.
        oxidized: Name of the oxidized ion of the couple, of ``electrons``
            more charge than the reduced one.
        electrode: The porous electrode.
        oxidation_on_discharge: Whether discharge oxidizes the couple, as in
            the negative half-cell.
        rate_constant: Standard rate constant of the couple in m/s or with
            units, the default of the parameter ``"<name>.rate_constant"``.
        formal_potential: Formal potential of the couple in V or with
            units, the default of the parameter ``"<name>.formal_potential"``.
        anodic_transfer: Anodic transfer coefficient, the default of the
            parameter ``"<name>.anodic_transfer"``.
        cathodic_transfer: Cathodic transfer coefficient, the default of the
            parameter ``"<name>.cathodic_transfer"``.
        electrons: Number of electrons the couple transfers.
        channels: Cell markers of free channels. Empty for none.
        velocity: Superficial velocity of the electrolyte, or ``None``
            without flow.
        mass_transfer: Limit the kinetics by mass transfer to the fibres,
            which needs a velocity. Defaults to whether there is one.
        collector: Facet markers of the current collector.
        membrane: Facet markers of the membrane.
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
        name: Prefix of the cell's own parameters.

    Raises:
        ValueError: For mass transfer or an outlet without a velocity, a
            reservoir concentration of an ion that the electrolyte does not
            have, a reservoir that lacks not exactly one ion, an ion of the
            couple without one, a couple that does not conserve charge, or a
            balanced ion without charge.
    """

    _domain: Domain
    """The domain of the cell."""

    _electrode: PorousElectrode
    """The porous electrode."""

    _regions: tuple[tuple[str, Region], ...]
    """The regions filled with electrolyte: the electrode's pores, and the
    free channels if any. Each comes with the suffix of its models' names."""

    _electrolyte: Electrolyte
    """The electrolyte."""

    _reduced: str
    """Name of the reduced ion of the couple."""

    _oxidized: str
    """Name of the oxidized ion of the couple."""

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

    _collector: tuple[str, ...]
    """Facet markers of the current collector."""

    _membrane: tuple[str, ...]
    """Facet markers of the membrane."""

    _functionals: types.MappingProxyType[str, ufl.Form]
    """The integrals behind the cell's diagnostics, created once so that a
    discretization compiles each of them only once."""

    def __init__(
        self,
        domain: Domain,
        ions: typing.Sequence[Ion],
        reservoir: typing.Mapping[str, float | sympy.Expr],
        reduced: str,
        oxidized: str,
        electrode: PorousElectrode,
        oxidation_on_discharge: bool,
        rate_constant: float | sympy.Expr,
        formal_potential: float | sympy.Expr,
        anodic_transfer: float = 0.5,
        cathodic_transfer: float = 0.5,
        electrons: int = 1,
        channels: tuple[str, ...] = (),
        velocity: ufl.core.expr.Expr | None = None,
        mass_transfer: bool | None = None,
        collector: tuple[str, ...] = ("CurrentCollector",),
        membrane: tuple[str, ...] = ("Membrane",),
        outlet: tuple[str, ...] = (),
        proton_flux: bool = False,
        density: ufl.core.expr.Expr | float = 1399.8,
        viscosity: ufl.core.expr.Expr | float = 6.963e-3,
        degree: int = 1,
        name: str = "cell",
    ):
        # Mass transfer to the fibres depends on the speed of the flow, so it
        # is on by default whenever there is one
        mass_transfer = velocity is not None if mass_transfer is None else mass_transfer
        if velocity is None and (mass_transfer or outlet):
            raise ValueError("Mass transfer and an outlet need a velocity.")
        unknown = sorted(set(reservoir) - {ion.name for ion in ions})
        if unknown:
            raise ValueError(f"The electrolyte has no ions {unknown}.")
        # The one ion without a reservoir concentration has no field, and
        # follows from electroneutrality, which closes the equations
        balanced = [ion.name for ion in ions if ion.name not in reservoir]
        if len(balanced) != 1:
            raise ValueError(
                "The reservoir must give the concentrations of all ions but one, "
                "whose concentration follows from electroneutrality; it lacks "
                f"{balanced}."
            )
        # The couple is the reaction oxidized + n e- <-> reduced, which must
        # conserve charge
        charges = {ion.name: ion.charge for ion in ions}
        for ion in (reduced, oxidized):
            if ion not in reservoir:
                raise ValueError(
                    f"The ion {ion!r} of the couple needs a reservoir concentration."
                )
        if charges[oxidized] - charges[reduced] != electrons:
            raise ValueError(
                f"{oxidized} + {electrons} e- <-> {reduced} does not conserve "
                f"charge: the charges are {charges[oxidized]} and "
                f"{charges[reduced]}."
            )
        self._domain = domain
        self._electrode = electrode
        self._reduced, self._oxidized = reduced, oxidized
        # The regions filled with electrolyte, with the suffix of their models:
        # the pores of the electrode, whose porosity weights storage and
        # corrects transport, and free channels, of porosity one
        regions = [("", self._electrode.region)]
        if channels:
            regions.append(("_channels", Region(tuple(channels))))
        self._regions = tuple(regions)

        # The unknowns. A concentration is of the order of a mole per litre,
        # its scale for the dimensional analysis and the scaled equations
        element = ("Lagrange", degree)
        self._phi_s = Field("phi_s", domain, element, unit=syu.volt)
        self._phi_l = Field("phi_l", domain, element, unit=syu.volt)
        self._concentrations = types.MappingProxyType(
            {
                ion.name: Field(
                    f"c_{ion.name}", domain, element, unit=CONCENTRATION, scale=1e3
                )
                for ion in ions
                if ion.name in reservoir
            }
        )
        # The electrolyte, with the field of every transported ion. The ion
        # without a reservoir concentration, and so without a field, follows
        # from electroneutrality. Its density and viscosity enter the mass
        # transfer, and may be shared with a model of the flow
        self._electrolyte = Electrolyte(
            "electrolyte",
            domain,
            [
                dataclasses.replace(
                    ion, concentration=self._concentrations.get(ion.name)
                )
                for ion in ions
            ],
            density=density,
            viscosity=viscosity,
        )
        # The concentrations of the reservoir: the initial state, and the
        # values of the conditions at an inlet, see inlet_conditions
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
        # The current density of the cell through the membrane, and the
        # ratio of the areas that makes the current through the collector
        # equal, see balance_currents
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
        # The kinetics of the couple, parameters of the cell that the reaction
        # shares, and the specific area of the electrode
        self._formal_potential = Parameter(
            f"{name}.formal_potential",
            domain,
            default=formal_potential,
            unit=syu.volt,
            doc=f"Formal potential of {reduced}/{oxidized}",
        )
        kinetics: dict[str, typing.Any] = dict(
            specific_area=self._electrode.specific_area,
            formal_potential=self._formal_potential,
            rate_constant=Parameter(
                f"{name}.rate_constant",
                domain,
                default=rate_constant,
                unit=syu.m / syu.second,
                doc=f"Standard rate constant of {reduced}/{oxidized}",
            ),
            anodic_transfer=Parameter(
                f"{name}.anodic_transfer",
                domain,
                default=anodic_transfer,
                doc=f"Anodic transfer coefficient of {reduced}/{oxidized}",
            ),
            cathodic_transfer=Parameter(
                f"{name}.cathodic_transfer",
                domain,
                default=cathodic_transfer,
                doc=f"Cathodic transfer coefficient of {reduced}/{oxidized}",
            ),
            electrons=electrons,
            markers=self._electrode.markers,
        )
        # The reaction in the electrode: Butler-Volmer at the concentrations
        # of the pores, or at the surface of the fibres with mass transfer
        e = self._electrolyte
        if mass_transfer:
            assert velocity is not None
            self._reaction = MassTransferButlerVolmer(
                "reaction",
                self._phi_s,
                self._phi_l,
                e,
                reduced,
                oxidized,
                velocity,
                fiber_diameter=self._electrode.fiber_diameter,
                **kinetics,
            )
        else:
            self._reaction = ButlerVolmer(
                "reaction", self._phi_s, self._phi_l, e, reduced, oxidized, **kinetics
            )

        # The current into the solid at the collector, and out of the
        # electrolyte at the membrane, on discharge. Oxidation on discharge
        # releases electrons into the solid, which leave through the
        # collector, so the current of the cell enters there; reduction on
        # discharge takes them, so the current leaves the solid there
        inflow = self._current if oxidation_on_discharge else -self._current
        models: list[Residual] = [
            Conduction(
                "solid",
                self._phi_s,
                self._electrode.effective_conductivity(),
                markers=self._electrode.markers,
            ),
            self._reaction,
            BoundaryFlux(
                "collector_current", self._phi_s, inflow * self._area_ratio, collector
            ),
            BoundaryFlux("membrane_current", self._phi_l, -inflow, membrane),
        ]
        # Every region filled with electrolyte conserves charge, and stores
        # and transports every ion, with the porosity of the region
        for suffix, region in self._regions:
            models.append(
                ElectrolyteConduction(f"electrolyte{suffix}", self._phi_l, e, region)
            )
            for ion, c in self._concentrations.items():
                models.append(
                    Accumulation(
                        f"storage_{ion}{suffix}",
                        c,
                        weight=region.porosity,
                        markers=region.markers,
                    )
                )
                models.append(
                    NernstPlanck(
                        f"transport_{ion}{suffix}",
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
                models.append(Outflow(f"outflow_{ion}", c, velocity, outlet))
        # The membrane passes the current as protons: their flux is the
        # current over their charge and Faraday's constant
        if proton_flux:
            models.append(
                BoundaryFlux(
                    "proton_flux",
                    self._concentrations["H"],
                    -inflow / (e["H"].charge * e.F),
                    membrane,
                )
            )
        self._model = ResidualSum(models, name=name)

        # The integrals of the diagnostics: the areas and mean potentials of
        # the collector and the membrane, for the voltage, and the amounts of
        # the couple in the electrolyte of every region, for the state of
        # charge
        self._collector = tuple(collector)
        self._membrane = tuple(membrane)
        ds_collector = domain.ds(*self._collector)
        ds_membrane = domain.ds(*self._membrane)
        functionals = {
            "collector_area": 1.0 * ds_collector,
            "membrane_area": 1.0 * ds_membrane,
            "collector_potential": self._phi_s * ds_collector,
            "membrane_potential": self._phi_l * ds_membrane,
        }
        for role, ion in (("reduced", reduced), ("oxidized", oxidized)):
            c = self._concentrations[ion]
            functionals[f"amount_{role}"] = sum(
                region.storage * c * domain.dx(*region.markers)
                for _, region in self._regions
            )
        self._functionals = types.MappingProxyType(functionals)

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

    def balance_currents(self, discretization: Discretization) -> None:
        """Set the area ratio from the areas of membrane and current collector."""
        assemble = discretization.assemble_scalar
        ratio = assemble(self._functionals["membrane_area"]) / assemble(
            self._functionals["collector_area"]
        )
        discretization.set({self._area_ratio: ratio})

    def voltage(self, discretization: Discretization) -> float:
        """The half-cell voltage in V.

        That is the mean solid potential on the current collector minus the
        mean electrolyte potential on the membrane.
        """
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
        c_red = self._concentrations[self._reduced]
        c_ox = self._concentrations[self._oxidized]
        return c_red / (c_red + c_ox + self._electrolyte.floor)
