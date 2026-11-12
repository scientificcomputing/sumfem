"""Ion-exchange membranes between the half-cells of a redox flow battery."""

from __future__ import annotations

import typing

import numpy.typing as npt
import sympy
import sympy.physics.units as syu
import ufl

from sumfem.models import Accumulation, Conduction, ConservativeAdvection, Diffusion
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    ResidualSum,
    as_expr,
)

from .electrolyte import CONCENTRATION, DIFFUSIVITY, Electrolyte

__all__ = ["ProtonExchangeMembrane", "SchloeglFlow"]

INPUT: typing.TypeAlias = ufl.core.expr.Expr | npt.ArrayLike | None
"""A model input: an expression, a value, or ``None`` for a parameter."""

PROPERTY: typing.TypeAlias = ufl.core.expr.Expr | float | sympy.Expr | None
"""A property of the membrane: an expression, used as given, or the default
of the parameter the membrane creates, ``None`` for none."""

PERMEABILITY = syu.m**2
"""Unit of permeabilities."""

VISCOSITY = syu.pascal * syu.second
"""Unit of dynamic viscosities."""


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
    :class:`~sumfem.models.Darcy` flow, so the normal velocity is continuous
    across the faces of the membrane with a Darcy flow beside it.

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
        charge_density: INPUT = None,
        electrokinetic_permeability: INPUT = None,
        hydraulic_permeability: INPUT = None,
        viscosity: INPUT = None,
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
        temperature: The temperature :math:`T`, in K.
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

    def __init__(
        self,
        name: str,
        potential: Field,
        markers: tuple[str, ...],
        water: Field | None = None,
        pressure: Field | None = None,
        fixed_charge: PROPERTY = None,
        proton_diffusivity: PROPERTY = None,
        water_diffusivity: PROPERTY = None,
        electrokinetic_permeability: PROPERTY = None,
        hydraulic_permeability: PROPERTY = None,
        viscosity: PROPERTY = None,
        temperature: float = 298.15,
    ):
        domain = potential.domain
        self._potential = potential
        self._water = water
        self._markers = tuple(markers)
        self._constants = Electrolyte(name, domain, temperature=temperature)
        e = self._constants
        self._fixed_charge = _property(
            f"{name}.fixed_charge",
            fixed_charge,
            domain,
            unit=CONCENTRATION,
            doc="Concentration of the fixed charges",
        )
        proton = _property(
            f"{name}.proton_diffusivity",
            proton_diffusivity,
            domain,
            unit=DIFFUSIVITY,
            doc="Proton diffusivity in the membrane",
        )
        conductivity = e.F**2 / (e.R * e.T) * proton * self._fixed_charge
        models: list[Residual] = [
            Conduction(f"{name}_conduction", potential, conductivity, self._markers)
        ]
        self._flow = None
        if pressure is not None:
            self._flow = SchloeglFlow(
                f"{name}_flow",
                pressure,
                potential,
                charge_density=e.F * self._fixed_charge,
                electrokinetic_permeability=_property(
                    f"{name}.electrokinetic_permeability",
                    electrokinetic_permeability,
                    domain,
                    unit=PERMEABILITY,
                    doc="Electrokinetic permeability",
                ),
                hydraulic_permeability=_property(
                    f"{name}.hydraulic_permeability",
                    hydraulic_permeability,
                    domain,
                    unit=PERMEABILITY,
                    doc="Hydraulic permeability",
                ),
                viscosity=_property(
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
            diffusivity = _property(
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
    def velocity(self) -> ufl.core.expr.Expr:
        """The velocity of the water, zero without a pressure."""
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


def _property(
    name: str,
    value: PROPERTY,
    domain: Domain,
    *,
    unit: syu.Quantity | sympy.Expr,
    doc: str,
) -> ufl.core.expr.Expr:
    """A property of the membrane: an expression, or a new parameter.

    Args:
        name: The full name of the parameter, ``"<membrane>.<property>"``.
        value: An expression, e.g. a shared parameter, used as given, or the
            default value of the new parameter, in SI units or with units;
            ``None`` for a parameter without a default.
        domain: The domain of the parameter.
        unit: The unit of the parameter.
        doc: A description of the parameter.

    Returns:
        The expression or the parameter.
    """
    if isinstance(value, ufl.core.expr.Expr):
        return value
    return Parameter(name, domain, default=value, unit=unit, doc=doc)
