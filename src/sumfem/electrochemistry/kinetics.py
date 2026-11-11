"""Butler-Volmer kinetics of electrode reactions in porous electrodes.

The Butler-Volmer equation, :func:`~sumfem.models.butler_volmer`, goes back to
`Butler, 1924 <https://doi.org/10.1039/TF9241900729>`_ and
`Erdey-Grúz and Volmer, 1930 <https://doi.org/10.1515/zpch-1930-15020>`_.
"""

from __future__ import annotations

import numpy.typing as npt
import sympy.physics.units as syu
import ufl

from sumfem.models import butler_volmer
from sumfem.symbolic import Field, Residual, as_expr

from .electrolyte import Electrolyte, Ion

__all__ = [
    "ButlerVolmer",
    "ElectrodeReaction",
    "ExchangeCurrentButlerVolmer",
    "MassTransferButlerVolmer",
]

INPUT = ufl.core.expr.Expr | npt.ArrayLike | None
"""A model input: an expression, a value, or ``None`` for a parameter."""


class ElectrodeReaction(Residual):
    """The reaction ``reduced -> oxidized + n e-`` in a porous electrode.

    The volumetric current ``f = a i``, with the specific area ``a`` and the
    interfacial current density ``i`` of :meth:`current_density`, leaves the
    solid and enters the electrolyte. It consumes the reduced species and
    produces the oxidized one at the rate ``f / (n F)``. Species without a
    field get no source. Subclasses give the kinetics.

    Args:
        name: Name of the model.
        solid_potential: The potential of the solid, ``phi_s``.
        electrolyte_potential: The potential of the electrolyte, ``phi_l``.
        electrolyte: The electrolyte, for the constants and temperature.
        reduced: Concentration of the reduced species, if any.
        oxidized: Concentration of the oxidized species, if any.
        specific_area: Specific area ``a``. Defaults to the parameter
            ``"<name>.specific_area"`` in 1/m.
        electrons: Number of electrons ``n`` transferred.
        markers: Cell markers of the electrode. Empty for the whole domain.
    """

    _solid_potential: Field
    """The potential of the solid."""

    _electrolyte_potential: Field
    """The potential of the electrolyte."""

    _electrolyte: Electrolyte
    """The electrolyte."""

    _reduced_concentration: ufl.core.expr.Expr | None
    """Concentration of the reduced species, if any."""

    _oxidized_concentration: ufl.core.expr.Expr | None
    """Concentration of the oxidized species, if any."""

    _specific_area: ufl.core.expr.Expr
    """Specific area of the electrode."""

    _electrons: int
    """Number of electrons transferred."""

    _markers: tuple[str, ...]
    """Cell markers of the electrode."""

    def __init__(
        self,
        name: str,
        solid_potential: Field,
        electrolyte_potential: Field,
        electrolyte: Electrolyte,
        reduced: ufl.core.expr.Expr | None = None,
        oxidized: ufl.core.expr.Expr | None = None,
        specific_area: INPUT = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        self._solid_potential = solid_potential
        self._electrolyte_potential = electrolyte_potential
        self._electrolyte = electrolyte
        self._reduced_concentration = reduced
        self._oxidized_concentration = oxidized
        self._specific_area = self.input(
            "specific_area",
            specific_area,
            domain=solid_potential.domain,
            unit=1 / syu.m,
            doc="Specific area of the electrode",
        )
        self._electrons = electrons
        self._markers = tuple(markers)

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

    def residual(self) -> ufl.Form:
        f = self.source()
        dx = self._solid_potential.domain.dx(*self._markers)
        F = f * self._solid_potential.v * dx - f * self._electrolyte_potential.v * dx
        rate = f / (self._electrons * self._electrolyte.F)
        species = (
            (self._reduced_concentration, 1.0),
            (self._oxidized_concentration, -1.0),
        )
        for c, sign in species:
            if isinstance(c, Field):
                F += sign * rate * c.v * dx
        return F


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
        exchange_current: INPUT = None,
        equilibrium_potential: INPUT = None,
        anodic_transfer: INPUT = None,
        cathodic_transfer: INPUT = None,
        reduced: ufl.core.expr.Expr | None = None,
        oxidized: ufl.core.expr.Expr | None = None,
        specific_area: INPUT = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(
            name,
            solid_potential,
            electrolyte_potential,
            electrolyte,
            reduced=reduced,
            oxidized=oxidized,
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
    """Butler-Volmer kinetics of a redox couple of an electrolyte.

    The Butler-Volmer equation, see :func:`~sumfem.models.butler_volmer`,
    with an exchange current density from the concentrations of the couple.
    The interfacial current density is

        ``i = n F (k_a c_red - k_c c_ox)``,
        ``k_a = k0 exp(alpha_a n F eta / (R T))``,
        ``k_c = k0 exp(-alpha_c n F eta / (R T))``,

    with ``eta = phi_s - phi_l - E0`` relative to the formal potential
    ``E0``, and the concentrations at the electrode surface, by default the
    positive parts of the bulk concentrations. The reduced and oxidized ions
    are ions in the electrolyte, given by name.

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
        specific_area: INPUT = None,
        rate_constant: INPUT = None,
        formal_potential: INPUT = None,
        anodic_transfer: INPUT = None,
        cathodic_transfer: INPUT = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(
            name,
            solid_potential,
            electrolyte_potential,
            electrolyte,
            reduced=electrolyte.concentration(reduced, positive=False),
            oxidized=electrolyte.concentration(oxidized, positive=False),
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
            e.concentration(self._reduced.name, positive=True),
            e.concentration(self._oxidized.name, positive=True),
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
    """Butler-Volmer kinetics limited by mass transfer to the fibers.

    The surface concentrations balance the reaction with mass transfer
    ``k_m (c - c_s)`` from the bulk. The mass-transfer coefficient of each
    ion is ``k_m = Sh D / d`` with the Sherwood number ``Sh = a Re^b`` and
    the Reynolds number ``Re = rho |u| d / mu`` of the flow past fibers of
    diameter ``d``. ``D`` is the diffusivity in free electrolyte.

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
        fiber_diameter: INPUT = None,
        density: INPUT = None,
        viscosity: INPUT = None,
        sherwood_factor: float = 7.0,
        reynolds_exponent: float = 0.4,
        speed_floor: float = 1e-16,
        transfer_floor: float = 1e-14,
        specific_area: INPUT = None,
        rate_constant: INPUT = None,
        formal_potential: INPUT = None,
        anodic_transfer: INPUT = None,
        cathodic_transfer: INPUT = None,
        electrons: int = 1,
        markers: tuple[str, ...] = (),
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
        k_a, k_c = self.rate_constants()
        m_red, m_ox = (
            m + self._transfer_floor for m in self.mass_transfer_coefficients()
        )
        c_red, c_ox = self._bulk()
        denominator = 1.0 + k_a / m_red + k_c / m_ox
        surface_ox = ((1.0 + k_a / m_red) * c_ox + k_a / m_ox * c_red) / denominator
        surface_red = ((1.0 + k_c / m_ox) * c_red + k_c / m_red * c_ox) / denominator
        return surface_red, surface_ox
