"""What a full cell is made of: its sides, its membrane and its flow.

The specifications of a :class:`~sumfem.electrochemistry.FullCell`: a
:class:`HalfCellSpec` for each side, a :class:`MembraneSpec` and a
:class:`FlowSpec`.
"""

from __future__ import annotations

import dataclasses
import typing

import sympy

from sumfem.symbolic import ExpressionOrDefault

from ..materials import Ion, RedoxCouple
from ..models import DarcyFlow, ElectrodeFlow, FlowSetup

__all__ = ["FlowSpec", "HalfCellSpec", "MembraneSpec"]

WATER = "H2O"
"""Name of the water of the electrolytes, shared by both sides."""

PROTONS = "H"
"""Name of the protons, which carry the current through the membrane."""


@dataclasses.dataclass(frozen=True)
class HalfCellSpec:
    """One side of a full cell: its electrolyte, its couple and its electrode.

    The properties of the electrode and the electrolyte, from the porosity
    on, are each an expression, used as given, e.g. a field of another model
    such as the porosity of a compressed felt or a viscosity of the
    temperature, or the default of the parameter the cell creates, see
    :func:`~sumfem.symbolic.expression_or_parameter`. The reservoir
    concentrations are values: with flow, the tanks change them.

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
        temperature: The temperature of the electrolyte, in K, e.g. the field
            of a heat equation.

    Raises:
        ValueError: If the side does not transport protons.
    """

    ions: tuple[Ion, ...]
    reservoir: typing.Mapping[str, float | sympy.Expr]
    couple: RedoxCouple
    porosity: ExpressionOrDefault
    conductivity: ExpressionOrDefault
    specific_area: ExpressionOrDefault
    fiber_distance: ExpressionOrDefault = None
    fiber_diameter: ExpressionOrDefault = None
    kozeny_carman_constant: ExpressionOrDefault = None
    density: ExpressionOrDefault = 1399.8
    viscosity: ExpressionOrDefault = 6.963e-3
    temperature: ExpressionOrDefault = 298.15

    def __post_init__(self) -> None:
        if PROTONS not in self.reservoir:
            raise ValueError(
                f"A side of a full cell must transport protons, {PROTONS!r}, which "
                "carry the current through the membrane."
            )


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
        temperature: The temperature of the membrane, in K, e.g. the field of
            a heat equation.
    """

    fixed_charge: ExpressionOrDefault = None
    proton_diffusivity: ExpressionOrDefault = None
    water_diffusivity: ExpressionOrDefault = None
    electrokinetic_permeability: ExpressionOrDefault = None
    hydraulic_permeability: ExpressionOrDefault = None
    viscosity: ExpressionOrDefault = None
    temperature: ExpressionOrDefault = 298.15


@dataclasses.dataclass(frozen=True)
class FlowSpec:
    r"""The flow of the electrolytes through the electrodes, from their tanks.

    The flow of the electrolytes is a :class:`ElectrodeFlow`, by default
    :class:`DarcyFlow`, which gives the superficial velocity
    :math:`\mathbf{u}` of each electrode. The electrolyte of each side
    enters its electrode at the inlet with the superficial velocity
    :math:`v_\text{in}` and the concentrations :math:`c_k^\text{in}` of its
    tank, and leaves at the outlet, held at the pressure
    :math:`p_\text{out}`, with the flow,

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
        model: The flow, created from the :class:`FlowSetup` the cell gives
            it: an :class:`ElectrodeFlow` subclass, or any callable of the
            setup, e.g. a :func:`functools.partial` of one with options of
            its own.
    """

    inlet_velocity: ExpressionOrDefault = None
    outlet_pressure: ExpressionOrDefault = 0.0
    tank_ratio: float | None = None
    model: typing.Callable[[FlowSetup], ElectrodeFlow] = DarcyFlow
