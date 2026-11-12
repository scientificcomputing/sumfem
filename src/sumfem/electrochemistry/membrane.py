"""Ion-exchange membranes between the half-cells of a redox flow battery."""

from __future__ import annotations

import typing

import sympy.physics.units as syu
import ufl

from sumfem.models import Accumulation, Conduction, ConservativeAdvection, Diffusion
from sumfem.models.operators import restricted
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    ExpressionOrDefault,
    ExpressionOrValue,
    Field,
    Residual,
    ResidualSum,
    as_expr,
    expression_or_parameter,
)
from sumfem.symbolic.si import CONCENTRATION, DIFFUSIVITY, PERMEABILITY, VISCOSITY

from .electrolyte import Electrolyte

__all__ = ["MembraneProtonFlux", "ProtonExchangeMembrane", "SchloeglFlow"]


class SchloeglFlow(Residual):
    r"""Flow of water through a charged membrane, by Schlögl's equation.

    The protons that neutralize the fixed charges of the membrane drag the
    water with them, so the velocity of the water has an electro-osmotic
    part besides the pressure-driven one:

    .. math::
       :label: schloegl-velocity

       \begin{aligned}
         \mathbf{v} &= -\frac{k_\phi}{\mu} \rho \nabla \phi
           - \frac{k_p}{\mu} \nabla p, \\
         \nabla \cdot \mathbf{v} &= 0,
       \end{aligned}

    with the electrokinetic and hydraulic permeabilities :math:`k_\phi` and
    :math:`k_p`, the viscosity :math:`\mu` of water, and the charge density
    :math:`\rho = F c_\text{H}` of the mobile protons. The residual is
    :math:`-(\mathbf{v}, \nabla q)` on the pressure, as for
    :class:`~sumfem.models.Darcy` flow.

    The equation is a bulk equation on the cells of the membrane,
    ``markers``, and the velocity holds there only: :math:`\rho` is the
    charge of the membrane's mobile ions, which the fixed charges set. With
    one pressure over the membrane and the electrodes beside it, each with
    a flow model of its own, the pressure is continuous across the faces,
    and so, weakly, is the normal velocity.

    Args:
        name: Name of the model.
        pressure: The pressure :math:`p`.
        potential: The electrolyte potential :math:`\phi`.
        charge_density: The charge density :math:`\rho` of the mobile ions.
            Defaults to the parameter ``"<name>.charge_density"`` in C/m^3.
        electrokinetic_permeability: :math:`k_\phi`. Defaults to the
            parameter ``"<name>.electrokinetic_permeability"`` in m^2.
        hydraulic_permeability: :math:`k_p`. Defaults to the parameter
            ``"<name>.hydraulic_permeability"`` in m^2.
        viscosity: :math:`\mu`. Defaults to the parameter
            ``"<name>.viscosity"`` in Pa s.
        markers: Cell markers of the membrane. Empty for the whole domain.
    """

    _pressure: Field
    """The pressure."""

    _potential: ufl.core.expr.Expr
    """The electrolyte potential."""

    _charge_density: ufl.core.expr.Expr
    """Charge density of the mobile ions."""

    _electrokinetic_permeability: ufl.core.expr.Expr
    """Electrokinetic permeability."""

    _hydraulic_permeability: ufl.core.expr.Expr
    """Hydraulic permeability."""

    _viscosity: ufl.core.expr.Expr
    """Viscosity of water."""

    _markers: tuple[str, ...]
    """Cell markers of the membrane."""

    def __init__(
        self,
        name: str,
        pressure: Field,
        potential: ufl.core.expr.Expr,
        charge_density: ExpressionOrValue = None,
        electrokinetic_permeability: ExpressionOrValue = None,
        hydraulic_permeability: ExpressionOrValue = None,
        viscosity: ExpressionOrValue = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        domain = pressure.domain
        self._pressure = pressure
        self._potential = as_expr(potential)
        self._charge_density = self.input(
            "charge_density",
            charge_density,
            domain=domain,
            unit=syu.coulomb / syu.m**3,
            doc="Charge density of the mobile ions",
        )
        self._electrokinetic_permeability = self.input(
            "electrokinetic_permeability",
            electrokinetic_permeability,
            domain=domain,
            unit=PERMEABILITY,
            doc="Electrokinetic permeability",
        )
        self._hydraulic_permeability = self.input(
            "hydraulic_permeability",
            hydraulic_permeability,
            domain=domain,
            unit=PERMEABILITY,
            doc="Hydraulic permeability",
        )
        self._viscosity = self.input(
            "viscosity", viscosity, domain=domain, unit=VISCOSITY, doc="Viscosity"
        )
        self._markers = tuple(markers)

    @property
    def velocity(self) -> ufl.core.expr.Expr:
        r"""The velocity :math:`\mathbf{v}` of :eq:`schloegl-velocity`."""
        mu = self._viscosity
        return -self._electrokinetic_permeability / mu * self._charge_density * (
            ufl.grad(self._potential)
        ) - self._hydraulic_permeability / mu * ufl.grad(self._pressure)

    def residual(self) -> ufl.Form:
        p = self._pressure
        dx = p.domain.dx(*self._markers)
        return ufl.inner(-self.velocity, ufl.grad(p.v)) * dx


class ProtonExchangeMembrane(ResidualSum):
    r"""A proton-exchange membrane saturated with water.

    The membrane of `Bernardi and Verbrugge, 1991
    <https://doi.org/10.1002/aic.690370805>`_, as in the full cell of
    `Shah et al., 2008 <https://doi.org/10.1016/j.electacta.2008.05.067>`_,
    Section 3.2. Its fixed charges, of concentration :math:`c_f`, are
    neutralized by protons, the only mobile ions, so the proton
    concentration is :math:`c_\text{H} = c_f` throughout. The membrane is
    then an Ohmic conductor for the electrolyte potential :math:`\phi`, and
    the water, of concentration :math:`c_w`, moves with the velocity
    :math:`\mathbf{v}` of :eq:`schloegl-velocity`:

    .. math::
       :label: proton-exchange-membrane

       \begin{aligned}
         -\nabla \cdot (\kappa_m \nabla \phi) &= 0, \\
         \kappa_m &= \frac{F^2}{R T} D_\text{H} c_f, \\
         \frac{\partial c_w}{\partial t}
           + \nabla \cdot (-D_w \nabla c_w + \mathbf{v} c_w) &= 0,
       \end{aligned}

    with the proton and water diffusivities :math:`D_\text{H}` and
    :math:`D_w` of the membrane. The current density
    :math:`-\kappa_m \nabla \phi` uses the sign of
    :class:`~sumfem.electrochemistry.ElectrolyteConduction`, so with one
    potential across the membrane and the electrodes beside it, the current
    is continuous across its faces, and so is the flux of water with one
    water field.

    Every equation is a bulk equation on the cells of the membrane,
    ``markers``, whose thickness the mesh resolves. Its fields may extend
    beyond them: in a :class:`~sumfem.electrochemistry.FullCell` the
    potential and the water live on both electrodes and the membrane, and
    each electrode's models add the equations there. The faces then need no
    term of their own for these fields: their continuity, and that of their
    fluxes, is natural. The protons are not a field of the membrane; the
    faces pass them to the electrodes' own fields with
    :meth:`proton_flux`, one condition per face.

    The membrane is the sum of ``"<name>_conduction"``, a
    :class:`~sumfem.models.Conduction`; with a water field,
    ``"<name>_water_storage"``, an :class:`~sumfem.models.Accumulation`, and
    ``"<name>_water_diffusion"``, a :class:`~sumfem.models.Diffusion`; and
    with a pressure, ``"<name>_flow"``, a :class:`SchloeglFlow`, and
    ``"<name>_water_advection"``, a
    :class:`~sumfem.models.ConservativeAdvection` of the water. Without a
    pressure, the water does not flow, and only diffuses.

    The membrane creates its constants and properties as parameters named
    ``"<name>.<parameter>"``: Faraday's constant, the gas constant and the
    temperature, see :class:`~sumfem.electrochemistry.Electrolyte`, and one
    for each argument from ``fixed_charge`` on, with the argument as its
    default value.

    Args:
        name: Name of the membrane, and prefix of its parameters and models.
        potential: The electrolyte potential :math:`\phi`, e.g. one field
            over the membrane and the electrodes.
        markers: Cell markers of the membrane.
        water: The water concentration :math:`c_w`, or ``None`` without
            water transport.
        pressure: The pressure :math:`p` of :eq:`schloegl-velocity`, or
            ``None`` without flow.
        fixed_charge: The concentration :math:`c_f` of fixed charges, in
            mol/m^3 or with units.
        proton_diffusivity: :math:`D_\text{H}` in the membrane, in m^2/s or
            with units.
        water_diffusivity: :math:`D_w` in the membrane, in m^2/s or with
            units.
        electrokinetic_permeability: :math:`k_\phi`, in m^2 or with units.
        hydraulic_permeability: :math:`k_p`, in m^2 or with units.
        viscosity: The viscosity :math:`\mu` of water, in Pa s or with units.
        temperature: The temperature :math:`T`: an expression in K, e.g.
            the field of a heat equation, or the default value of a
            parameter.
    """

    _potential: Field
    """The electrolyte potential."""

    _water: Field | None
    """The water concentration, if transported."""

    _flow: SchloeglFlow | None
    """The flow of water, with a pressure."""

    _constants: Electrolyte
    """Faraday's constant, the gas constant and the temperature."""

    _fixed_charge: ufl.core.expr.Expr
    """Concentration of the fixed charges."""

    _markers: tuple[str, ...]
    """Cell markers of the membrane."""

    _conduction: Conduction
    """Ohmic conduction in the membrane."""

    def __init__(
        self,
        name: str,
        potential: Field,
        markers: tuple[str, ...],
        water: Field | None = None,
        pressure: Field | None = None,
        fixed_charge: ExpressionOrDefault = None,
        proton_diffusivity: ExpressionOrDefault = None,
        water_diffusivity: ExpressionOrDefault = None,
        electrokinetic_permeability: ExpressionOrDefault = None,
        hydraulic_permeability: ExpressionOrDefault = None,
        viscosity: ExpressionOrDefault = None,
        temperature: ufl.core.expr.Expr | float = 298.15,
    ):
        domain = potential.domain
        self._potential = potential
        self._water = water
        self._markers = tuple(markers)
        self._constants = Electrolyte(name, domain, temperature=temperature)
        e = self._constants
        self._fixed_charge = expression_or_parameter(
            f"{name}.fixed_charge",
            fixed_charge,
            domain,
            unit=CONCENTRATION,
            doc="Concentration of the fixed charges",
        )
        proton = expression_or_parameter(
            f"{name}.proton_diffusivity",
            proton_diffusivity,
            domain,
            unit=DIFFUSIVITY,
            doc="Proton diffusivity in the membrane",
        )
        conductivity = e.F**2 / (e.R * e.T) * proton * self._fixed_charge
        self._conduction = Conduction(
            f"{name}_conduction", potential, conductivity, self._markers
        )
        models: list[Residual] = [self._conduction]
        self._flow = None
        if pressure is not None:
            self._flow = SchloeglFlow(
                f"{name}_flow",
                pressure,
                potential,
                charge_density=e.F * self._fixed_charge,
                electrokinetic_permeability=expression_or_parameter(
                    f"{name}.electrokinetic_permeability",
                    electrokinetic_permeability,
                    domain,
                    unit=PERMEABILITY,
                    doc="Electrokinetic permeability",
                ),
                hydraulic_permeability=expression_or_parameter(
                    f"{name}.hydraulic_permeability",
                    hydraulic_permeability,
                    domain,
                    unit=PERMEABILITY,
                    doc="Hydraulic permeability",
                ),
                viscosity=expression_or_parameter(
                    f"{name}.viscosity",
                    viscosity,
                    domain,
                    unit=VISCOSITY,
                    doc="Viscosity of water",
                ),
                markers=self._markers,
            )
            models.append(self._flow)
        if water is not None:
            diffusivity = expression_or_parameter(
                f"{name}.water_diffusivity",
                water_diffusivity,
                domain,
                unit=DIFFUSIVITY,
                doc="Water diffusivity in the membrane",
            )
            models.append(
                Accumulation(f"{name}_water_storage", water, markers=self._markers)
            )
            models.append(
                Diffusion(f"{name}_water_diffusion", water, diffusivity, self._markers)
            )
            if self._flow is not None:
                models.append(
                    ConservativeAdvection(
                        f"{name}_water_advection",
                        water,
                        self.velocity,
                        markers=self._markers,
                    )
                )
        super().__init__(models, name=name)

    @property
    def potential(self) -> Field:
        """The electrolyte potential."""
        return self._potential

    @property
    def fixed_charge(self) -> ufl.core.expr.Expr:
        """The concentration of the fixed charges."""
        return self._fixed_charge

    @property
    def markers(self) -> tuple[str, ...]:
        """Cell markers of the membrane."""
        return self._markers

    @property
    def constants(self) -> Electrolyte:
        """Faraday's constant, the gas constant and the temperature."""
        return self._constants

    def current_density(self) -> ufl.core.expr.Expr:
        r"""The current density :math:`-\kappa_m \nabla \phi` of the membrane."""
        return self._conduction.flux()

    def proton_flux(
        self,
        name: str,
        protons: Field,
        measure: ufl.Measure,
        charge: float = 1.0,
    ) -> MembraneProtonFlux:
        """The membrane current carried by the protons of an electrode.

        See :class:`MembraneProtonFlux`.

        Args:
            name: Name of the model.
            protons: The proton concentration of the electrode.
            measure: The interface measure of the face, with the electrode on
                its ``"+"`` side and the membrane on its ``"-"`` side, from
                :meth:`~sumfem.symbolic.domains.Domain.interface`.
            charge: The charge number of the protons.

        Returns:
            The proton flux at the face.
        """
        return MembraneProtonFlux(name, protons, self, measure, charge)

    @property
    def velocity(self) -> ufl.core.expr.Expr:
        """The velocity of the water in the membrane, zero without a pressure.

        It holds on the membrane's cells only, see :class:`SchloeglFlow`.
        """
        if self._flow is None:
            return ufl.zero(self._potential.domain.geometric_dimension)
        return self._flow.velocity

    def proton_conditions(
        self, protons: typing.Sequence[Field], faces: str | tuple[str, ...]
    ) -> list[DirichletBC]:
        """The proton concentration of the membrane at its faces.

        Electroneutrality holds the protons at the concentration of the
        fixed charges, also where the electrodes meet the membrane.

        Args:
            protons: The proton concentrations of the electrodes beside the
                membrane.
            faces: Facet markers of the faces of the membrane.

        Returns:
            The conditions :math:`c_\\text{H} = c_f` on the faces.
        """
        return [DirichletBC(c, faces, self._fixed_charge) for c in protons]


class MembraneProtonFlux(Residual):
    r"""The protons of an electrode carry the current of a membrane beside it.

    A cation-exchange membrane passes protons only, so at the face between
    an electrode and the membrane the protons of the electrode carry the
    whole current :math:`\mathbf{i}_m` of the membrane,

    .. math::
       :label: membrane-proton-flux

       \begin{aligned}
         \mathbf{N}_\text{H} \cdot \mathbf{n}
           &= \frac{\mathbf{i}_m \cdot \mathbf{n}}{z_\text{H} F},
       \end{aligned}

    with the normal :math:`\mathbf{n}` out of the electrode, and every other
    ion has no flux. The residual is
    :math:`(\mathbf{N}_\text{H} \cdot \mathbf{n}, w)` on the face, with the
    test function :math:`w` of the protons. The current is that of the
    membrane's side of the face, :meth:`ProtonExchangeMembrane.current_density`.

    With the ion of the electrolyte that follows from electroneutrality, e.g.
    sulfate, the zero flux of that ion is no natural condition: it holds
    when the protons carry the current, by this condition. The proton
    concentration at the face is not held at the fixed charge, so the model
    has no Donnan potential across the face.

    The condition is a surface term of the electrode's proton equation, on
    one face of the membrane and from one side, the electrode's: each face
    needs its own, with the protons of the electrode on that face. It is
    not an equation of the membrane, which has no proton field.

    Args:
        name: Name of the model.
        protons: The proton concentration :math:`c_\text{H}` of the
            electrode.
        membrane: The membrane.
        measure: The interface measure of the face, with the electrode on
            its ``"+"`` side and the membrane on its ``"-"`` side, from
            :meth:`~sumfem.symbolic.domains.Domain.interface`.
        charge: The charge number :math:`z_\text{H}` of the protons.

    Raises:
        ValueError: If the measure is not that of an interface with the
            membrane on its ``"-"`` side and the protons' subdomain on its
            ``"+"`` side.
    """

    _protons: Field
    """The proton concentration of the electrode."""

    _membrane: ProtonExchangeMembrane
    """The membrane."""

    _measure: ufl.Measure
    """The interface measure of the face."""

    _charge: float
    """The charge number of the protons."""

    def __init__(
        self,
        name: str,
        protons: Field,
        membrane: ProtonExchangeMembrane,
        measure: ufl.Measure,
        charge: float = 1.0,
    ):
        super().__init__(name)
        self._protons = protons
        self._membrane = membrane
        self._measure = measure
        self._charge = charge
        self._check_sides()

    def _check_sides(self) -> None:
        """Check that the face has the protons' electrode on its ``"+"`` side
        and the membrane on its ``"-"`` side.

        Raises:
            ValueError: If the measure is not that of such faces.
        """
        domain = self._measure.ufl_domain()
        interfaces = (
            domain.interfaces_of(self._measure) if isinstance(domain, Domain) else ()
        )
        if not interfaces:
            raise ValueError(
                f"{self.name!r} needs the measure of a face of the membrane, from "
                "Domain.interface, to know its sides."
            )
        electrode = self._protons.domain
        for first, second, marker in interfaces:
            if second.marker not in self._membrane.markers:
                raise ValueError(
                    f"The face {marker!r} of {self.name!r} must have the membrane "
                    f"on its '-' side, got {second.marker!r}."
                )
            if electrode is not first:
                raise ValueError(
                    f"The face {marker!r} of {self.name!r} must have the electrode "
                    f"of {self._protons.name!r} on its '+' side, got "
                    f"{first.marker!r}."
                )

    def flux(self) -> ufl.core.expr.Expr:
        r"""The proton flux :math:`\mathbf{N}_\text{H} \cdot \mathbf{n}` out of
        the electrode, on its side of the face."""
        n = ufl.FacetNormal(self._measure.ufl_domain())
        current = restricted(self._membrane.current_density(), "-")
        F = restricted(self._membrane.constants.F, "+")
        return ufl.dot(current, n("+")) / (self._charge * F)

    def residual(self) -> ufl.Form:
        return self.flux() * self._protons.v("+") * self._measure
