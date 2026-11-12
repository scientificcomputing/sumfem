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

from mpi4py import MPI

import dolfinx.fem
import numpy as np
import sympy
import sympy.physics.units as syu
import ufl

from sumfem.discrete import Discretization
from sumfem.models import BoundaryFlux, Conduction, Outflow
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
from .transport import TransientNernstPlanck

__all__ = ["RedoxHalfCell"]


class RedoxHalfCell:
    r"""A half-cell of a redox flow battery, of a couple of dissolved ions.

    The macro-homogeneous porous-electrode model of RFBniCS, `Aghabarari et
    al., 2026 <https://arxiv.org/abs/2610.01840>`_, Section II, for any
    couple, electrolyte and sign of the half-cell.

    The unknowns are the potential :math:`\phi_s` of the solid and
    :math:`\phi_l` of the electrolyte, and the concentration :math:`c_k` of
    every ion with a reservoir concentration. In the porous electrode, of
    porosity :math:`\varepsilon` and specific area :math:`a`, charge is
    conserved in the solid and the electrolyte, which exchange the current
    :math:`a i` of the reaction, and each ion is stored, transported and
    produced:

    .. math::
       :label: redox-half-cell

       \begin{aligned}
         -\nabla \cdot (\sigma_\text{eff} \nabla \phi_s) + a i &= 0, \\
         \nabla \cdot \mathbf{i}_l - a i &= 0, \\
         \varepsilon \frac{\partial c_k}{\partial t}
           + \nabla \cdot \mathbf{N}_k &= \nu_k \frac{a i}{n F}, \\
         \sum_k z_k c_k &= 0.
       \end{aligned}

    The terms are those of the components of the cell:

    - :math:`\sigma_\text{eff}`, the effective conductivity of the solid,
      :meth:`PorousElectrode.effective_conductivity
      <sumfem.electrochemistry.PorousElectrode.effective_conductivity>`;
    - :math:`\mathbf{i}_l`, the current density of the electrolyte,
      :eq:`electrolyte-current`;
    - :math:`\mathbf{N}_k`, the Nernst-Planck flux of the ion of charge
      :math:`z_k`, :eq:`nernst-planck-flux`, with the superficial velocity
      :math:`\mathbf{u}` of the electrolyte;
    - :math:`i`, the interfacial current density of the couple,
      :eq:`butler-volmer-couple`, at surface concentrations limited by mass
      transfer, :eq:`mass-transfer-balance`, with flow or a fibre distance;
    - :math:`\nu_k` and :math:`n`, the coefficient of the ion in the
      reaction of the couple, negative for a reactant, positive for a
      product and zero for an ion that takes no part, and the number of
      electrons, see :class:`~sumfem.electrochemistry.RedoxCouple`.

    The one ion without a reservoir concentration follows from
    electroneutrality, the last equation. In free channels,
    :math:`\varepsilon = 1` and there is no reaction or solid.

    The cell is galvanostatic, with the current density :math:`j` of the
    cell through the membrane, positive on discharge. On discharge the
    couple is oxidized in the negative half-cell, :math:`s = 1`, and reduced
    in the positive one, :math:`s = -1`. With the outward normal
    :math:`\mathbf{n}`, the current enters the solid at the current
    collector and leaves the electrolyte at the membrane,

    .. math::
       :label: redox-half-cell-boundary

       \begin{aligned}
         \sigma_{\text{eff}} \nabla \phi_s \cdot \mathbf{n} &= s j r
           && \text{at the current collector}, \\
         \mathbf{i}_l \cdot \mathbf{n} &= s j
           && \text{at the membrane}, \\
         \mathbf{N}_\text{H} \cdot \mathbf{n} &= \frac{s j}{z_\text{H} F}
           && \text{at the membrane, with a proton flux}, \\
         (\mathbf{N}_k - \mathbf{u} c_k) \cdot \mathbf{n} &= 0
           && \text{at an outlet},
       \end{aligned}

    where the ratio :math:`r` of the membrane area to the collector area
    makes the total currents equal, see :meth:`balance_currents`. Other
    boundaries carry no flux.

    The model, :attr:`model`, is the sum of:

    - ``"<name>_solid"``: the first equation of :eq:`redox-half-cell`
      without the reaction,
      :class:`~sumfem.models.Conduction`;
    - ``"<name>_reaction"``: the kinetics :eq:`butler-volmer-couple`, with
      :eq:`mass-transfer-balance` under mass transfer,
      :class:`~sumfem.electrochemistry.ButlerVolmer`,
      :class:`~sumfem.electrochemistry.MassTransferButlerVolmer` with mass
      transfer, or :class:`~sumfem.electrochemistry.PistonVelocityButlerVolmer`
      with a fibre distance, which adds the reaction to
      :eq:`redox-half-cell`;
    - with collector and membrane markers, ``"<name>_collector_current"``
      and ``"<name>_membrane_current"``: the first two conditions of
      :eq:`redox-half-cell-boundary`, :class:`~sumfem.models.BoundaryFlux`;
    - for the electrode's pores and, with a suffix ``"_channels"``, the free
      channels: the charge in the electrolyte,
      :class:`~sumfem.electrochemistry.ElectrolyteConduction`, and the
      balance :eq:`nernst-planck-balance` of every ion,
      :class:`~sumfem.electrochemistry.TransientNernstPlanck`, named
      ``"<name>_<ion>"``, whose summands are ``"<name>_<ion>_storage"`` and
      ``"<name>_<ion>_transport"``;
    - with an outlet, ``"<name>_outflow_<ion>"``: the outlet condition,
      :class:`~sumfem.models.Outflow`;
    - with a proton flux, ``"<name>_proton_flux"``: the proton condition at
      the membrane, a :class:`~sumfem.models.BoundaryFlux` of the ion
      ``"H"``.

    Args:
        domain: The domain of the cell.
        ions: The ions of the electrolyte, with their charges :math:`z_k`
            and diffusivities :math:`D_k`, without fields: the cell creates
            the field of every ion that has a reservoir concentration.
        reservoir: The concentration of each transported ion in the
            reservoir, in mol/m^3 or a quantity with units, the default of
            the parameter ``"<name>.c0_<ion>"``.
        couple: The redox couple and its reaction: the coefficients
            :math:`\nu_k`, :math:`n` and :math:`s`, and the kinetics
            :eq:`butler-volmer-couple`, whose parameters are those of the
            cell: ``"<name>.rate_constant"`` :math:`k_0`,
            ``"<name>.formal_potential"`` :math:`E_0`,
            ``"<name>.anodic_transfer"`` :math:`\alpha_a` and
            ``"<name>.cathodic_transfer"`` :math:`\alpha_c`.
        electrode: The porous electrode: :math:`\varepsilon`, :math:`a` and
            :math:`\sigma`.
        channels: Cell markers of free channels. Empty for none.
        velocity: The superficial velocity :math:`\mathbf{u}`, or ``None``
            without flow.
        mass_transfer: Limit the kinetics by mass transfer to the fibres,
            :eq:`mass-transfer-balance`, with the coefficients of the flow's
            correlation, :eq:`mass-transfer-correlation`, which needs a
            velocity. Defaults to whether there is one.
        fiber_distance: The mean distance :math:`\delta` between the
            fibres, in m or with units. Limits the kinetics by
            :eq:`mass-transfer-balance` with the piston velocities,
            :eq:`piston-velocity`, instead, with or without flow.
        collector: Facet markers of the current collector, or ``None`` for
            a cell whose collector is resolved, e.g. in a full cell.
        membrane: Facet markers of the membrane, or ``None`` for a cell
            whose membrane is resolved.
        outlet: Facet markers of an outlet. Empty for none.
        proton_flux: Carry the membrane current by protons, the ion ``"H"``,
            through the membrane, the third condition of
            :eq:`redox-half-cell-boundary`.
        density: Density of the electrolyte, for the mass transfer, see
            :class:`~sumfem.electrochemistry.Electrolyte`. Pass the
            parameter of a flow model to share it.
        viscosity: Dynamic viscosity of the electrolyte, given like the
            density.
        degree: Degree of the Lagrange elements of the unknowns.
        name: Prefix of the cell's own parameters, ``"<name>.<parameter>"``,
            and of the names of its models, ``"<name>_<model>"``. The
            current density :math:`j` is the parameter ``"<name>.current"``,
            in A/m^2, and the area ratio :math:`r` ``"<name>.area_ratio"``.
        phi_s: The potential :math:`\phi_s` of the solid, if it exists, e.g.
            over the electrode and its current collector. Created on the
            domain otherwise.
        phi_l: The potential :math:`\phi_l` of the electrolyte, if it exists,
            e.g. over the electrodes and the membrane of a full cell.
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

    _rate_constant: Parameter
    """Standard rate constant of the couple."""

    _anodic_transfer: Parameter
    """Anodic transfer coefficient of the couple."""

    _cathodic_transfer: Parameter
    """Cathodic transfer coefficient of the couple."""

    _fiber_distance: Parameter | None
    """Mean distance between the fibres, if the piston velocities limit the
    kinetics."""

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
        self._create_parameters(reservoir, fiber_distance)
        self._reaction = self._create_reaction(velocity, mass_transfer)
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
        """Create the unknowns of the cell.

        Concentrations get the scale of a mole per litre, for the
        dimensional analysis and the scaled equations.

        Args:
            ions: The ions of the electrolyte.
            reservoir: The reservoir concentrations. Their ions are the
                transported ones, each with a field; the one ion without
                follows from electroneutrality.
            degree: Degree of the Lagrange elements of new fields.
            phi_s: The potential of the solid, or ``None`` to create it.
            phi_l: The potential of the electrolyte, or ``None`` to create it.
            fields: Existing concentration fields, by ion name, used as
                given.
        """
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
        """Create the electrolyte, with the fields of the transported ions.

        Args:
            ions: The ions of the electrolyte. The one without a field
                follows from electroneutrality.
            name: Prefix of the electrolyte's parameters.
            density: Density of the electrolyte, a value or a shared
                expression.
            viscosity: Dynamic viscosity of the electrolyte, a value or a
                shared expression.

        Returns:
            The electrolyte.
        """
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
        self,
        reservoir: typing.Mapping[str, float | sympy.Expr],
        fiber_distance: float | sympy.Expr | None,
    ) -> None:
        """Create the parameters of the cell.

        These are the reservoir concentrations, the current density, the
        area ratio, the kinetic parameters of the couple, and the fibre
        distance if one is given.

        Args:
            reservoir: The default reservoir concentration of each
                transported ion.
            fiber_distance: The default mean distance between the fibres, or
                ``None`` for kinetics without piston velocities, which then
                has no such parameter.
        """
        domain, name, couple = self._domain, self._name, self._couple
        pair = f"{couple.reduced}/{couple.oxidized}"
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
            doc=f"Formal potential of {pair}",
        )
        self._rate_constant = Parameter(
            f"{name}.rate_constant",
            domain,
            default=couple.rate_constant,
            unit=syu.m / syu.second,
            doc=f"Standard rate constant of {pair}",
        )
        self._anodic_transfer = Parameter(
            f"{name}.anodic_transfer",
            domain,
            default=couple.anodic_transfer,
            doc=f"Anodic transfer coefficient of {pair}",
        )
        self._cathodic_transfer = Parameter(
            f"{name}.cathodic_transfer",
            domain,
            default=couple.cathodic_transfer,
            doc=f"Cathodic transfer coefficient of {pair}",
        )
        self._fiber_distance = (
            None
            if fiber_distance is None
            else Parameter(
                f"{name}.fiber_distance",
                domain,
                default=fiber_distance,
                unit=syu.m,
                doc="Mean distance between the fibres",
            )
        )

    def _create_reaction(
        self, velocity: ufl.core.expr.Expr | None, mass_transfer: bool
    ) -> ButlerVolmer:
        """Create the reaction in the electrode, from the cell's parameters.

        Args:
            velocity: The superficial velocity, or ``None`` without flow.
            mass_transfer: Limit the kinetics by the flow's mass-transfer
                correlation, unless the cell has a fibre distance.

        Returns:
            The reaction: :class:`~sumfem.electrochemistry.PistonVelocityButlerVolmer`
            with a fibre distance, else
            :class:`~sumfem.electrochemistry.MassTransferButlerVolmer` with
            mass transfer, else :class:`~sumfem.electrochemistry.ButlerVolmer`.
        """
        name, couple = self._name, self._couple
        common: dict[str, typing.Any] = dict(
            specific_area=self._electrode.specific_area,
            formal_potential=self._formal_potential,
            rate_constant=self._rate_constant,
            anodic_transfer=self._anodic_transfer,
            cathodic_transfer=self._cathodic_transfer,
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
        if self._fiber_distance is not None:
            return PistonVelocityButlerVolmer(
                *arguments, fiber_distance=self._fiber_distance, **common
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
        """Create the models of the cell, listed in the class documentation.

        Args:
            velocity: The superficial velocity, or ``None`` without flow.
            outlet: Facet markers of the outlet. Empty for none.
            proton_flux: Carry the membrane current by protons.

        Returns:
            The models, to be summed.
        """
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
            for ion in self._concentrations:
                models.append(
                    TransientNernstPlanck(
                        f"{name}_{ion}{suffix}",
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
        """Create the integrals of the voltage and the state of charge.

        Returns:
            The areas and potential integrals of the collector and the
            membrane, when they have markers, and the amounts of the reduced
            and oxidized ion, by name.
        """
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
        """Check that the collector and the membrane have facet markers.

        Args:
            what: The quantity that needs them, for the error message.

        Raises:
            ValueError: If either has none.
        """
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

    def minimum_balanced_concentration(self, discretization: Discretization) -> float:
        """The smallest concentration of the ion given by electroneutrality.

        Nothing keeps that concentration positive. A negative minimum means
        the electrolyte cannot stay neutral with the computed
        concentrations, e.g. after a time step that is too long, or because
        the balanced ion is not abundant enough, see
        :class:`~sumfem.electrochemistry.Electrolyte`.

        Args:
            discretization: The discretization of the cell.

        Returns:
            The minimum over the nodes of the concentrations' space, over all
            processes.
        """
        e = self._electrolyte
        assert e.balanced is not None
        space = discretization.function_space(next(iter(self._concentrations.values())))
        balanced = dolfinx.fem.Function(space)
        expression = discretization[e.concentration(e.balanced)]
        balanced.interpolate(
            dolfinx.fem.Expression(expression, space.element.interpolation_points)
        )
        owned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        local = balanced.x.array[:owned].min(initial=np.inf)
        return float(space.mesh.comm.allreduce(local, op=MPI.MIN))

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

    Args:
        ions: The ions of the electrolyte.
        reservoir: The reservoir concentrations. They must name all ions
            but one, which follows from electroneutrality, and every ion of
            the reaction.
        couple: The redox couple.

    Raises:
        ValueError: If the reaction or the reservoir does not match the ions.
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
