"""Butler-Volmer kinetics of electrode reactions in porous electrodes.

The Butler-Volmer equation, :func:`~sumfem.models.butler_volmer`, goes back to
`Butler, 1924 <https://doi.org/10.1039/TF9241900729>`_ and
`Erdey-Grúz and Volmer, 1930 <https://doi.org/10.1515/zpch-1930-15020>`_.
"""

from __future__ import annotations

import typing

import sympy.physics.units as syu
import ufl

from sumfem.models import RateReaction, butler_volmer
from sumfem.symbolic import ExpressionOrValue, Field, as_expr

from .electrolyte import Electrolyte, Ion

__all__ = [
    "ButlerVolmer",
    "ElectrodeReaction",
    "ExchangeCurrentButlerVolmer",
    "MassTransferButlerVolmer",
    "PistonVelocityButlerVolmer",
]


class ElectrodeReaction(RateReaction):
    """An electrode reaction ``reactants -> products + n e-`` in a porous electrode.

    A :class:`~sumfem.models.RateReaction` that also moves charge between
    the phases. The volumetric current ``f = a i``, with the specific area
    ``a`` and the interfacial current density ``i`` of
    :meth:`current_density`, leaves the solid and enters the electrolyte.
    The reaction happens ``f / (n F)`` times per unit volume and time, so it
    consumes each reactant and produces each product at that rate times its
    coefficient, e.g. ``V2+ -> V3+ + e-``, or ``VO2+ + H2O -> VO2+ + 2 H+ +
    e-``, which also produces two protons and consumes one water molecule.
    Participants without a field get no source. Subclasses give the
    kinetics.

    Args:
        name: Name of the model.
        solid_potential: The potential of the solid, ``phi_s``.
        electrolyte_potential: The potential of the electrolyte, ``phi_l``.
        electrolyte: The electrolyte, for the constants and temperature.
        reactants: The coefficient of each species that oxidation consumes,
            by concentration. None for a reaction that only moves charge.
        products: The coefficient of each species that oxidation produces.
        specific_area: Specific area ``a``. Defaults to the parameter
            ``"<name>.specific_area"`` in 1/m.
        electrons: Number of electrons ``n`` transferred.
        markers: Cell markers of the electrode. Empty for the whole domain.
    """

    _derives_rate = True
    _requires_species = False

    _solid_potential: Field
    """The potential of the solid."""

    _electrolyte_potential: Field
    """The potential of the electrolyte."""

    _electrolyte: Electrolyte
    """The electrolyte."""

    _specific_area: ufl.core.expr.Expr
    """Specific area of the electrode."""

    _electrons: int
    """Number of electrons transferred."""

    def __init__(
        self,
        name: str,
        solid_potential: Field,
        electrolyte_potential: Field,
        electrolyte: Electrolyte,
        reactants: typing.Mapping[ufl.core.expr.Expr, int] | None = None,
        products: typing.Mapping[ufl.core.expr.Expr, int] | None = None,
        specific_area: ExpressionOrValue = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
    ):
        # The reaction happens in the electrode, where the solid is
        super().__init__(
            name,
            dict(reactants or {}),
            dict(products or {}),
            measure=solid_potential.domain.dx(*markers),
        )
        self._solid_potential = solid_potential
        self._electrolyte_potential = electrolyte_potential
        self._electrolyte = electrolyte
        self._specific_area = self.input(
            "specific_area",
            specific_area,
            domain=solid_potential.domain,
            unit=1 / syu.m,
            doc="Specific area of the electrode",
        )
        self._electrons = electrons

    @property
    def electrolyte(self) -> Electrolyte:
        """The electrolyte."""
        return self._electrolyte

    def _faradaic_factor(self) -> ufl.core.expr.Expr:
        """``n F / (R T)``, the inverse of the thermal voltage per electron."""
        e = self._electrolyte
        return self._electrons * e.F / (e.R * e.T)

    def current_density(self) -> ufl.core.expr.Expr:
        """The interfacial current density ``i``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement current_density()."
        )

    def source(self) -> ufl.core.expr.Expr:
        """The volumetric current ``f = a i``."""
        return self._specific_area * self.current_density()

    def rate(self) -> ufl.core.expr.Expr:
        """The rate ``f / (n F)`` of the reaction, per unit volume."""
        return self.source() / (self._electrons * self._electrolyte.F)

    def residual(self) -> ufl.Form:
        f = self.source()
        dx = self._measure
        current = (
            f * self._solid_potential.v * dx - f * self._electrolyte_potential.v * dx
        )
        if not self._fields():
            return current
        return current + super().residual()


class ExchangeCurrentButlerVolmer(ElectrodeReaction):
    """Butler-Volmer kinetics with a given exchange current density.

    The current density of :func:`~sumfem.models.butler_volmer`,
    ``i = i0 (exp(alpha_a n F eta / (R T)) - exp(-alpha_c n F eta / (R T)))``
    with ``eta = phi_s - phi_l - E_eq``. The exchange current density may
    depend on concentrations. With both transfer coefficients one half and
    one electron, ``i = 2 i0 sinh(F eta / (2 R T))``.

    Args:
        name: Name of the model.
        solid_potential: See :class:`ElectrodeReaction`.
        electrolyte_potential: See :class:`ElectrodeReaction`.
        electrolyte: See :class:`ElectrodeReaction`.
        exchange_current: Exchange current density ``i0``. Defaults to the
            parameter ``"<name>.exchange_current"`` in A/m^2.
        equilibrium_potential: Equilibrium potential ``E_eq``. Defaults to
            the parameter ``"<name>.equilibrium_potential"`` in V.
        anodic_transfer: Anodic transfer coefficient ``alpha_a``. Defaults
            to the parameter ``"<name>.anodic_transfer"``, 0.5.
        cathodic_transfer: Cathodic transfer coefficient ``alpha_c``.
            Defaults to the parameter ``"<name>.cathodic_transfer"``, 0.5.
        reduced: See :class:`ElectrodeReaction`.
        oxidized: See :class:`ElectrodeReaction`.
        specific_area: See :class:`ElectrodeReaction`.
        electrons: See :class:`ElectrodeReaction`.
        markers: See :class:`ElectrodeReaction`.
    """

    _exchange_current: ufl.core.expr.Expr
    """Exchange current density."""

    _equilibrium_potential: ufl.core.expr.Expr
    """Equilibrium potential."""

    _anodic_transfer: ufl.core.expr.Expr
    """Anodic transfer coefficient."""

    _cathodic_transfer: ufl.core.expr.Expr
    """Cathodic transfer coefficient."""

    def __init__(
        self,
        name: str,
        solid_potential: Field,
        electrolyte_potential: Field,
        electrolyte: Electrolyte,
        exchange_current: ExpressionOrValue = None,
        equilibrium_potential: ExpressionOrValue = None,
        anodic_transfer: ExpressionOrValue = None,
        cathodic_transfer: ExpressionOrValue = None,
        reduced: ufl.core.expr.Expr | None = None,
        oxidized: ufl.core.expr.Expr | None = None,
        specific_area: ExpressionOrValue = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(
            name,
            solid_potential,
            electrolyte_potential,
            electrolyte,
            reactants=None if reduced is None else {reduced: 1},
            products=None if oxidized is None else {oxidized: 1},
            specific_area=specific_area,
            electrons=electrons,
            markers=markers,
        )
        domain = solid_potential.domain
        self._exchange_current = self.input(
            "exchange_current",
            exchange_current,
            domain=domain,
            unit=syu.ampere / syu.m**2,
            doc="Exchange current density",
        )
        self._equilibrium_potential = self.input(
            "equilibrium_potential",
            equilibrium_potential,
            domain=domain,
            unit=syu.volt,
            doc="Equilibrium potential",
        )
        self._anodic_transfer = self.input(
            "anodic_transfer",
            anodic_transfer,
            domain=domain,
            default=0.5,
            doc="Anodic transfer coefficient",
        )
        self._cathodic_transfer = self.input(
            "cathodic_transfer",
            cathodic_transfer,
            domain=domain,
            default=0.5,
            doc="Cathodic transfer coefficient",
        )

    def overpotential(self) -> ufl.core.expr.Expr:
        """``phi_s - phi_l - E_eq``."""
        return (
            self._solid_potential
            - self._electrolyte_potential
            - self._equilibrium_potential
        )

    def current_density(self) -> ufl.core.expr.Expr:
        """The interfacial current density ``i``."""
        return butler_volmer(
            self._exchange_current,
            self._faradaic_factor() * self.overpotential(),
            self._anodic_transfer,
            self._cathodic_transfer,
        )


class ButlerVolmer(ElectrodeReaction):
    r"""Butler-Volmer kinetics of a redox couple of an electrolyte.

    The Butler-Volmer equation, see :func:`~sumfem.models.butler_volmer`,
    with an exchange current density from the concentrations of the couple.
    The interfacial current density is

    .. math::
       :label: butler-volmer-couple

       \begin{aligned}
         i &= n F \left( k_a c_\text{red}^s - k_c c_\text{ox}^s \right), \\
         k_a &= k_0 e^{\alpha_a n F \eta / (R T)}, \\
         k_c &= k_0 e^{-\alpha_c n F \eta / (R T)}, \\
         \eta &= \phi_s - \phi_l - E_0,
       \end{aligned}

    with the standard rate constant :math:`k_0`, the transfer coefficients
    :math:`\alpha_a` and :math:`\alpha_c`, the formal potential :math:`E_0`,
    and the concentrations :math:`c_\text{red}^s` and :math:`c_\text{ox}^s`
    of the reduced and oxidized ion at the electrode surface. These are, by
    default, the smooth positive parts of the bulk concentrations,
    :math:`c^+ = (c + \sqrt{c^2 + c_\text{floor}^2}) / 2`, see
    :meth:`~sumfem.electrochemistry.electrolyte.Electrolyte.positive_part`.
    The reduced and oxidized ions are ions in the electrolyte, given by name.

    Subclasses change the kinetics by overriding
    :meth:`surface_concentrations`, :meth:`rate_constants` or
    :meth:`current_density`.

    Args:
        name: Name of the model.
        solid_potential: See :class:`ElectrodeReaction`.
        electrolyte_potential: See :class:`ElectrodeReaction`.
        electrolyte: The electrolyte of the ions.
        reduced: Name of the reduced ion.
        oxidized: Name of the oxidized ion.
        specific_area: See :class:`ElectrodeReaction`.
        rate_constant: Standard rate constant ``k0``. Defaults to the
            parameter ``"<name>.rate_constant"`` in m/s.
        formal_potential: Formal potential ``E0``. Defaults to the parameter
            ``"<name>.formal_potential"`` in V.
        anodic_transfer: Anodic transfer coefficient ``alpha_a``. Defaults
            to the parameter ``"<name>.anodic_transfer"``, 0.5.
        cathodic_transfer: Cathodic transfer coefficient ``alpha_c``.
            Defaults to the parameter ``"<name>.cathodic_transfer"``, 0.5.
        electrons: See :class:`ElectrodeReaction`.
        markers: See :class:`ElectrodeReaction`.
        reactants: The coefficient of each ion that oxidation consumes, by
            name, see :class:`ElectrodeReaction`. Defaults to the reduced
            ion alone, e.g. ``{"VO": 1, "H2O": 1}`` for VO2+ + H2O ->
            VO2+ + 2 H+ + e-.
        products: The coefficient of each ion that oxidation produces.
            Defaults to the oxidized ion alone, e.g. ``{"VO2": 1, "H": 2}``.

    Raises:
        ValueError: If the reactants lack the reduced ion or the products the
            oxidized one, or if an ion of the reaction other than the couple
            has no field, as the ion that follows from electroneutrality.
    """

    _reduced: Ion
    """The reduced ion."""

    _oxidized: Ion
    """The oxidized ion."""

    _rate_constant: ufl.core.expr.Expr
    """Standard rate constant."""

    _formal_potential: ufl.core.expr.Expr
    """Formal potential of the couple."""

    _anodic_transfer: ufl.core.expr.Expr
    """Anodic transfer coefficient."""

    _cathodic_transfer: ufl.core.expr.Expr
    """Cathodic transfer coefficient."""

    def __init__(
        self,
        name: str,
        solid_potential: Field,
        electrolyte_potential: Field,
        electrolyte: Electrolyte,
        reduced: str,
        oxidized: str,
        specific_area: ExpressionOrValue = None,
        rate_constant: ExpressionOrValue = None,
        formal_potential: ExpressionOrValue = None,
        anodic_transfer: ExpressionOrValue = None,
        cathodic_transfer: ExpressionOrValue = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
        reactants: typing.Mapping[str, int] | None = None,
        products: typing.Mapping[str, int] | None = None,
    ):
        reactants = dict(reactants or {reduced: 1})
        products = dict(products or {oxidized: 1})
        if reduced not in reactants or oxidized not in products:
            raise ValueError(
                f"The reactants {reactants} must hold the reduced ion {reduced!r}, "
                f"and the products {products} the oxidized ion {oxidized!r}."
            )
        others = (set(reactants) | set(products)) - {reduced, oxidized}
        for ion in sorted(others):
            if electrolyte[ion].concentration is None:
                raise ValueError(
                    f"Ion {ion!r} of the reaction follows from electroneutrality, "
                    "so it has no field to produce or consume."
                )

        def concentrations(ions: dict[str, int]) -> dict[ufl.core.expr.Expr, int]:
            return {electrolyte.concentration(ion): nu for ion, nu in ions.items()}

        super().__init__(
            name,
            solid_potential,
            electrolyte_potential,
            electrolyte,
            reactants=concentrations(reactants),
            products=concentrations(products),
            specific_area=specific_area,
            electrons=electrons,
            markers=markers,
        )
        domain = solid_potential.domain
        self._reduced = electrolyte[reduced]
        self._oxidized = electrolyte[oxidized]
        self._rate_constant = self.input(
            "rate_constant",
            rate_constant,
            domain=domain,
            unit=syu.m / syu.second,
            doc="Standard rate constant",
        )
        self._formal_potential = self.input(
            "formal_potential",
            formal_potential,
            domain=domain,
            unit=syu.volt,
            doc="Formal potential",
        )
        self._anodic_transfer = self.input(
            "anodic_transfer",
            anodic_transfer,
            domain=domain,
            default=0.5,
            doc="Anodic transfer coefficient",
        )
        self._cathodic_transfer = self.input(
            "cathodic_transfer",
            cathodic_transfer,
            domain=domain,
            default=0.5,
            doc="Cathodic transfer coefficient",
        )

    @property
    def reduced(self) -> Ion:
        """The reduced ion."""
        return self._reduced

    @property
    def oxidized(self) -> Ion:
        """The oxidized ion."""
        return self._oxidized

    def _bulk(self) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """Positive parts of the bulk concentrations of the reduced and oxidized ion."""
        e = self._electrolyte
        return (
            e.positive_part(e.concentration(self._reduced.name)),
            e.positive_part(e.concentration(self._oxidized.name)),
        )

    def rate_constants(self) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """The anodic and cathodic rate constants ``k_a`` and ``k_c``."""
        eta = (
            self._solid_potential - self._electrolyte_potential - self._formal_potential
        )
        factor = self._faradaic_factor() * eta
        return (
            self._rate_constant * ufl.exp(self._anodic_transfer * factor),
            self._rate_constant * ufl.exp(-self._cathodic_transfer * factor),
        )

    def surface_concentrations(
        self,
    ) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """Concentrations of the reduced and oxidized ion at the surface.

        These are the positive parts of the bulk concentrations.
        """
        return self._bulk()

    def current_density(self) -> ufl.core.expr.Expr:
        """The interfacial current density ``i``."""
        k_a, k_c = self.rate_constants()
        c_red, c_ox = self.surface_concentrations()
        return self._electrons * self._electrolyte.F * (k_a * c_red - k_c * c_ox)

    def equilibrium_potential(self) -> ufl.core.expr.Expr:
        """Nernst's potential ``E0 + R T / (n F) ln(c_ox / c_red)``.

        The concentrations are the positive parts of the bulk ones, each
        plus the electrolyte's concentration floor.
        """
        floor = self._electrolyte.floor
        c_red, c_ox = self._bulk()
        return (
            self._formal_potential
            + ufl.ln((c_ox + floor) / (c_red + floor)) / self._faradaic_factor()
        )

    def overpotential(self) -> ufl.core.expr.Expr:
        """``phi_s - phi_l`` minus the :meth:`equilibrium_potential`."""
        return (
            self._solid_potential
            - self._electrolyte_potential
            - self.equilibrium_potential()
        )


class MassTransferButlerVolmer(ButlerVolmer):
    r"""Butler-Volmer kinetics limited by mass transfer to the fibers.

    The surface concentrations of :eq:`butler-volmer-couple` are those at
    which mass transfer from the bulk, with the coefficients
    :math:`m_\text{red}` and :math:`m_\text{ox}`, balances the reaction:

    .. math::
       :label: mass-transfer-balance

       \begin{aligned}
         m_\text{red} \left( c_\text{red}^+ - c_\text{red}^s \right)
           &= k_a c_\text{red}^s - k_c c_\text{ox}^s, \\
         m_\text{ox} \left( c_\text{ox}^s - c_\text{ox}^+ \right)
           &= k_a c_\text{red}^s - k_c c_\text{ox}^s,
       \end{aligned}

    two linear equations, solved in closed form. Each coefficient has a
    small floor added. The coefficients follow from the Sherwood number
    :math:`\mathrm{Sh} = a \, \mathrm{Re}^b` of the flow past fibers of
    diameter :math:`d`:

    .. math::
       :label: mass-transfer-correlation

       \begin{aligned}
         m_\text{red} &= \mathrm{Sh} \frac{D_\text{red}}{d}, \\
         m_\text{ox} &= \mathrm{Sh} \frac{D_\text{ox}}{d}, \\
         \mathrm{Re} &= \frac{\rho |\mathbf{u}| d}{\mu},
       \end{aligned}

    with the diffusivities :math:`D_\text{red}` and :math:`D_\text{ox}` of
    the reduced and oxidized ion in free electrolyte, and the density
    :math:`\rho` and viscosity :math:`\mu` of the electrolyte. The
    defaults :math:`a = 7` and :math:`b = 0.4` are the correlation of `Krishnamurthy
    et al., 2011 <https://doi.org/10.1016/j.jpowsour.2011.08.024>`_, and the
    surface concentrations those of equation 20 of `Aghabarari et al., 2026
    <https://arxiv.org/abs/2610.01840>`_.

    Args:
        name: Name of the model.
        solid_potential: See :class:`ButlerVolmer`.
        electrolyte_potential: See :class:`ButlerVolmer`.
        electrolyte: See :class:`ButlerVolmer`.
        reduced: See :class:`ButlerVolmer`.
        oxidized: See :class:`ButlerVolmer`.
        velocity: The superficial velocity ``u``.
        fiber_diameter: Fiber diameter ``d``. Defaults to the parameter
            ``"<name>.fiber_diameter"`` in m.
        density: Density ``rho`` of the electrolyte. Defaults to the
            :attr:`~Electrolyte.density` of the electrolyte.
        viscosity: Dynamic viscosity ``mu`` of the electrolyte. Defaults to
            the :attr:`~Electrolyte.viscosity` of the electrolyte.
        sherwood_factor: The factor ``a`` of the Sherwood number.
        reynolds_exponent: The exponent ``b`` of the Sherwood number.
        speed_floor: Regularization of ``|u|``, in m/s.
        transfer_floor: Regularization added to ``k_m``, in m/s.
        specific_area: See :class:`ButlerVolmer`.
        rate_constant: See :class:`ButlerVolmer`.
        formal_potential: See :class:`ButlerVolmer`.
        anodic_transfer: See :class:`ButlerVolmer`.
        cathodic_transfer: See :class:`ButlerVolmer`.
        electrons: See :class:`ButlerVolmer`.
        markers: See :class:`ButlerVolmer`.
    """

    _velocity: ufl.core.expr.Expr
    """The superficial velocity."""

    _fiber_diameter: ufl.core.expr.Expr
    """Fiber diameter."""

    _density: ufl.core.expr.Expr
    """Density of the electrolyte."""

    _viscosity: ufl.core.expr.Expr
    """Dynamic viscosity of the electrolyte."""

    _sherwood_factor: float
    """The factor of the Sherwood number."""

    _reynolds_exponent: float
    """The exponent of the Sherwood number."""

    _speed_floor: ufl.core.expr.Expr
    """Regularization of the speed."""

    _transfer_floor: ufl.core.expr.Expr
    """Regularization of the mass-transfer coefficients."""

    def __init__(
        self,
        name: str,
        solid_potential: Field,
        electrolyte_potential: Field,
        electrolyte: Electrolyte,
        reduced: str,
        oxidized: str,
        velocity: ufl.core.expr.Expr,
        fiber_diameter: ExpressionOrValue = None,
        density: ExpressionOrValue = None,
        viscosity: ExpressionOrValue = None,
        sherwood_factor: float = 7.0,
        reynolds_exponent: float = 0.4,
        speed_floor: float = 1e-16,
        transfer_floor: float = 1e-14,
        specific_area: ExpressionOrValue = None,
        rate_constant: ExpressionOrValue = None,
        formal_potential: ExpressionOrValue = None,
        anodic_transfer: ExpressionOrValue = None,
        cathodic_transfer: ExpressionOrValue = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
        reactants: typing.Mapping[str, int] | None = None,
        products: typing.Mapping[str, int] | None = None,
    ):
        super().__init__(
            name,
            solid_potential,
            electrolyte_potential,
            electrolyte,
            reduced,
            oxidized,
            specific_area=specific_area,
            rate_constant=rate_constant,
            formal_potential=formal_potential,
            anodic_transfer=anodic_transfer,
            cathodic_transfer=cathodic_transfer,
            electrons=electrons,
            markers=markers,
            reactants=reactants,
            products=products,
        )
        domain = solid_potential.domain
        self._velocity = as_expr(velocity)
        self._fiber_diameter = self.input(
            "fiber_diameter",
            fiber_diameter,
            domain=domain,
            unit=syu.m,
            doc="Fiber diameter",
        )
        self._density = (
            electrolyte.density
            if density is None
            else self.input(
                "density",
                density,
                domain=domain,
                unit=syu.kg / syu.m**3,
                doc="Density of the electrolyte",
            )
        )
        self._viscosity = (
            electrolyte.viscosity
            if viscosity is None
            else self.input(
                "viscosity",
                viscosity,
                domain=domain,
                unit=syu.pascal * syu.second,
                doc="Dynamic viscosity of the electrolyte",
            )
        )
        self._sherwood_factor = sherwood_factor
        self._reynolds_exponent = reynolds_exponent
        self._speed_floor = self.parameter(
            "speed_floor",
            domain,
            default=speed_floor,
            unit=syu.m / syu.second,
            doc="Regularization of the speed",
        )
        self._transfer_floor = self.parameter(
            "transfer_floor",
            domain,
            default=transfer_floor,
            unit=syu.m / syu.second,
            doc="Regularization of the mass-transfer coefficients",
        )

    def reynolds_number(self) -> ufl.core.expr.Expr:
        """The Reynolds number ``rho |u| d / mu`` of the flow past a fiber."""
        u = self._velocity
        speed = ufl.sqrt(ufl.dot(u, u) + self._speed_floor**2)
        return self._density * speed * self._fiber_diameter / self._viscosity

    def mass_transfer_coefficients(
        self,
    ) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """The mass-transfer coefficients of the reduced and oxidized ion."""
        e = self._electrolyte
        sherwood = (
            self._sherwood_factor * self.reynolds_number() ** self._reynolds_exponent
        )
        return (
            sherwood * e.diffusivity(self._reduced.name) / self._fiber_diameter,
            sherwood * e.diffusivity(self._oxidized.name) / self._fiber_diameter,
        )

    def surface_concentrations(
        self,
    ) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """Surface concentrations balancing reaction and mass transfer."""
        m_red, m_ox = (
            m + self._transfer_floor for m in self.mass_transfer_coefficients()
        )
        return _balanced_surface(*self.rate_constants(), m_red, m_ox, *self._bulk())


def _balanced_surface(
    k_a: ufl.core.expr.Expr,
    k_c: ufl.core.expr.Expr,
    m_red: ufl.core.expr.Expr,
    m_ox: ufl.core.expr.Expr,
    c_red: ufl.core.expr.Expr,
    c_ox: ufl.core.expr.Expr,
) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
    """Surface concentrations where mass transfer balances the reaction.

    Solves ``m_red (c_red - s_red) = k_a s_red - k_c s_ox`` and
    ``m_ox (c_ox - s_ox) = -(k_a s_red - k_c s_ox)``, see equation 20 of
    `Aghabarari et al., 2026 <https://arxiv.org/abs/2610.01840>`_.

    Args:
        k_a: Anodic rate constant.
        k_c: Cathodic rate constant.
        m_red: Mass-transfer coefficient of the reduced ion.
        m_ox: Mass-transfer coefficient of the oxidized ion.
        c_red: Bulk concentration of the reduced ion.
        c_ox: Bulk concentration of the oxidized ion.

    Returns:
        The surface concentrations ``s_red`` and ``s_ox``.
    """
    denominator = 1.0 + k_a / m_red + k_c / m_ox
    surface_ox = ((1.0 + k_a / m_red) * c_ox + k_a / m_ox * c_red) / denominator
    surface_red = ((1.0 + k_c / m_ox) * c_red + k_c / m_red * c_ox) / denominator
    return surface_red, surface_ox


class PistonVelocityButlerVolmer(ButlerVolmer):
    r"""Butler-Volmer kinetics limited by diffusion between the fibres.

    The surface concentrations balance the reaction with mass transfer as in
    :eq:`mass-transfer-balance`, with the piston velocities

    .. math::
       :label: piston-velocity

       \begin{aligned}
         m_\text{red} &= \frac{D_\text{red}}{\delta}, \\
         m_\text{ox} &= \frac{D_\text{ox}}{\delta},
       \end{aligned}

    the diffusivities of the reduced and oxidized ion in free electrolyte
    over the mean distance :math:`\delta` between the fibres, see equations
    23 and 24 of `Shah et al., 2008
    <https://doi.org/10.1016/j.electacta.2008.05.067>`_. Unlike
    :class:`MassTransferButlerVolmer`, the transfer does not depend on the
    flow.

    Args:
        name: Name of the model.
        solid_potential: See :class:`ButlerVolmer`.
        electrolyte_potential: See :class:`ButlerVolmer`.
        electrolyte: See :class:`ButlerVolmer`.
        reduced: See :class:`ButlerVolmer`.
        oxidized: See :class:`ButlerVolmer`.
        fiber_distance: Mean distance ``delta`` between the fibres.
            Defaults to the parameter ``"<name>.fiber_distance"`` in m.
        transfer_floor: Regularization added to ``gamma``, in m/s.
        **kwargs: The other arguments of :class:`ButlerVolmer`.
    """

    _fiber_distance: ufl.core.expr.Expr
    """Mean distance between the fibres."""

    _transfer_floor: ufl.core.expr.Expr
    """Regularization of the piston velocities."""

    def __init__(
        self,
        name: str,
        solid_potential: Field,
        electrolyte_potential: Field,
        electrolyte: Electrolyte,
        reduced: str,
        oxidized: str,
        fiber_distance: ExpressionOrValue = None,
        transfer_floor: float = 1e-14,
        **kwargs: typing.Any,
    ):
        super().__init__(
            name,
            solid_potential,
            electrolyte_potential,
            electrolyte,
            reduced,
            oxidized,
            **kwargs,
        )
        domain = solid_potential.domain
        self._fiber_distance = self.input(
            "fiber_distance",
            fiber_distance,
            domain=domain,
            unit=syu.m,
            doc="Mean distance between the fibres",
        )
        self._transfer_floor = self.parameter(
            "transfer_floor",
            domain,
            default=transfer_floor,
            unit=syu.m / syu.second,
            doc="Regularization of the piston velocities",
        )

    def mass_transfer_coefficients(
        self,
    ) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """The piston velocities ``D / delta`` of the reduced and oxidized ion."""
        e = self._electrolyte
        return (
            e.diffusivity(self._reduced.name) / self._fiber_distance,
            e.diffusivity(self._oxidized.name) / self._fiber_distance,
        )

    def surface_concentrations(
        self,
    ) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """Surface concentrations balancing reaction and mass transfer."""
        m_red, m_ox = (
            m + self._transfer_floor for m in self.mass_transfer_coefficients()
        )
        return _balanced_surface(*self.rate_constants(), m_red, m_ox, *self._bulk())
