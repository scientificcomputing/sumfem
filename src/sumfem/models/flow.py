"""Flow of a fluid, free or through a porous medium.

The flows of a velocity and a pressure are sums of components:
:class:`Stokes` flow, with the viscous, pressure and continuity terms, and
terms of the momentum equation added to it, :class:`Convection`,
:class:`DarcyDrag` and :class:`ForchheimerDrag`, and grad-div
stabilization, :class:`GradDiv`. :class:`NavierStokes`, :class:`Brinkman`
and :class:`BrinkmanForchheimer` are the usual sums, under one name, whose
parameters are named after it. :class:`Darcy` flow has the pressure alone.
"""

from __future__ import annotations

import numpy.typing as npt
import sympy.physics.units as syu
import ufl

from sumfem.symbolic import Field, Residual, ResidualSum, input_or_parameter

__all__ = [
    "Brinkman",
    "BrinkmanForchheimer",
    "Convection",
    "Darcy",
    "DarcyDrag",
    "ForchheimerDrag",
    "GradDiv",
    "NavierStokes",
    "Stokes",
]

VISCOSITY = syu.pascal * syu.second
"""Unit of dynamic viscosities."""

DENSITY = syu.kg / syu.m**3
"""Unit of densities."""

INPUT = ufl.core.expr.Expr | npt.ArrayLike | None
"""A model input: an expression, a value, or ``None`` for a parameter."""


class Darcy(Residual):
    """Darcy flow through a porous medium, ``(K / mu grad(p), grad(q))``.

    The superficial velocity is ``u = -K / mu grad(p)``. Prescribe inflow
    with a :class:`~sumfem.models.BoundaryFlux` on the
    pressure, and the outlet pressure with a Dirichlet condition.

    Args:
        name: Name of the model.
        pressure: The pressure ``p``.
        permeability: Permeability ``K``. Defaults to the parameter
            ``"<name>.permeability"`` in m^2.
        viscosity: Dynamic viscosity ``mu``. Defaults to the parameter
            ``"<name>.viscosity"`` in Pa s.
        markers: Cell markers of the porous medium. Empty for the whole
            domain.
    """

    _pressure: Field
    """The pressure."""

    _permeability: ufl.core.expr.Expr
    """Permeability of the medium."""

    _viscosity: ufl.core.expr.Expr
    """Dynamic viscosity of the fluid."""

    _markers: tuple[str, ...]
    """Cell markers of the porous medium."""

    def __init__(
        self,
        name: str,
        pressure: Field,
        permeability: INPUT = None,
        viscosity: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        domain = pressure.domain
        self._pressure = pressure
        self._permeability = self.input(
            "permeability",
            permeability,
            domain=domain,
            unit=syu.m**2,
            doc="Permeability",
        )
        self._viscosity = self.input(
            "viscosity", viscosity, domain=domain, unit=VISCOSITY, doc="Viscosity"
        )
        self._markers = tuple(markers)

    @property
    def superficial_velocity(self) -> ufl.core.expr.Expr:
        """The superficial velocity ``-K / mu grad(p)``, the volume flux per unit
        area of the medium, not the velocity of the fluid in its pores."""
        return -self._permeability / self._viscosity * ufl.grad(self._pressure)

    def residual(self) -> ufl.Form:
        p = self._pressure
        dx = p.domain.dx(*self._markers)
        return ufl.inner(-self.superficial_velocity, ufl.grad(p.v)) * dx


class Stokes(Residual):
    """Steady incompressible Stokes flow.

    ``(2 mu epsilon(u), epsilon(w)) - (p, div(w)) - (div(u), q)``

    with the symmetric gradient ``epsilon(u) = (grad(u) + grad(u)^T) / 2``,
    the velocity ``u`` and its test function ``w``, and the pressure ``p``
    and its test function ``q``. The stress is ``2 mu epsilon(u) - p I``, so
    the term holds for a viscosity that varies in space, e.g. ``mu / eps``
    with a porosity field, and its natural boundary condition is a traction
    that vanishes. Add the other terms of the momentum equation as
    components, e.g. :class:`Convection`.

    Args:
        name: Name of the model.
        velocity: The velocity ``u``.
        pressure: The pressure ``p``.
        viscosity: Dynamic viscosity ``mu``. Defaults to the parameter
            ``"<name>.viscosity"`` in Pa s.
        markers: Cell markers of the flow region. Empty for the whole domain.
    """

    _velocity: Field
    """The velocity."""

    _pressure: Field
    """The pressure."""

    _viscosity: ufl.core.expr.Expr
    """Dynamic viscosity of the viscous term."""

    _markers: tuple[str, ...]
    """Cell markers of the flow region."""

    def __init__(
        self,
        name: str,
        velocity: Field,
        pressure: Field,
        viscosity: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        self._velocity = velocity
        self._pressure = pressure
        self._viscosity = self.input(
            "viscosity",
            viscosity,
            domain=velocity.domain,
            unit=VISCOSITY,
            doc="Viscosity",
        )
        self._markers = tuple(markers)

    @property
    def velocity(self) -> Field:
        """The velocity."""
        return self._velocity

    @property
    def pressure(self) -> Field:
        """The pressure."""
        return self._pressure

    @property
    def viscosity(self) -> ufl.core.expr.Expr:
        """The viscosity of the viscous term."""
        return self._viscosity

    def viscous_flux(
        self, u: ufl.core.expr.Expr, grad_u: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """The stress ``2 mu epsilon(u) - p I`` of the momentum equation.

        A function of ``u`` and ``grad_u`` separately, for Nitsche's
        method, which evaluates it at the boundary.
        """
        return 2 * self._viscosity * ufl.sym(grad_u) - self._pressure * ufl.Identity(
            len(u)
        )

    def residual(self) -> ufl.Form:
        u, p = self._velocity, self._pressure
        dx = u.domain.dx(*self._markers)
        viscous = (
            2
            * self._viscosity
            * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(u.v)))
        )
        return viscous * dx - p * ufl.div(u.v) * dx - ufl.div(u) * p.v * dx


class _Momentum(Residual):
    """A term ``(f, w)`` of the momentum equation of a velocity.

    Args:
        name: Name of the model.
        velocity: The velocity ``u``, whose test function is ``w``.
        markers: Cell markers of the term. Empty for the whole domain.
    """

    _velocity: Field
    """The velocity."""

    _markers: tuple[str, ...]
    """Cell markers of the term."""

    def __init__(self, name: str, velocity: Field, markers: tuple[str, ...] = ()):
        super().__init__(name)
        self._velocity = velocity
        self._markers = tuple(markers)

    def force(self) -> ufl.core.expr.Expr:
        """The term ``f``, per unit volume."""
        raise NotImplementedError(f"{type(self).__name__} must implement force().")

    def residual(self) -> ufl.Form:
        u = self._velocity
        return ufl.inner(self.force(), u.v) * u.domain.dx(*self._markers)


class Convection(_Momentum):
    """Convection of momentum, ``(rho grad(u) u, w)``.

    Args:
        name: Name of the model.
        velocity: The velocity ``u``.
        density: Density ``rho``. Defaults to the parameter
            ``"<name>.density"`` in kg/m^3.
        markers: Cell markers of the term. Empty for the whole domain.
    """

    _density: ufl.core.expr.Expr
    """Density of the fluid."""

    def __init__(
        self,
        name: str,
        velocity: Field,
        density: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name, velocity, markers)
        self._density = self.input(
            "density", density, domain=velocity.domain, unit=DENSITY, doc="Density"
        )

    def force(self) -> ufl.core.expr.Expr:
        u = self._velocity
        return self._density * ufl.dot(ufl.grad(u), u)


class DarcyDrag(_Momentum):
    """Darcy's drag of a porous medium, ``(mu / K u, w)``.

    Args:
        name: Name of the model.
        velocity: The superficial velocity ``u``.
        viscosity: Dynamic viscosity ``mu``. Defaults to the parameter
            ``"<name>.viscosity"`` in Pa s.
        permeability: Permeability ``K``. Defaults to the parameter
            ``"<name>.permeability"`` in m^2.
        markers: Cell markers of the porous medium. Empty for the whole
            domain.
    """

    _viscosity: ufl.core.expr.Expr
    """Dynamic viscosity of the fluid."""

    _permeability: ufl.core.expr.Expr
    """Permeability of the medium."""

    def __init__(
        self,
        name: str,
        velocity: Field,
        viscosity: INPUT = None,
        permeability: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name, velocity, markers)
        domain = velocity.domain
        self._viscosity = self.input(
            "viscosity", viscosity, domain=domain, unit=VISCOSITY, doc="Viscosity"
        )
        self._permeability = self.input(
            "permeability",
            permeability,
            domain=domain,
            unit=syu.m**2,
            doc="Permeability",
        )

    def force(self) -> ufl.core.expr.Expr:
        return self._viscosity / self._permeability * self._velocity


class ForchheimerDrag(_Momentum):
    """Forchheimer's inertial drag of a porous medium, ``(rho beta |u| u, w)``.

    The speed is regularized, ``|u| = sqrt(u . u + s^2)``, so that the term is
    differentiable where the fluid rests.

    Args:
        name: Name of the model.
        velocity: The superficial velocity ``u``.
        density: Density ``rho``. Defaults to the parameter
            ``"<name>.density"`` in kg/m^3.
        coefficient: Inertial drag coefficient ``beta``, e.g. Ergun's,
            :func:`~sumfem.models.ergun_coefficient`. Defaults to the
            parameter ``"<name>.forchheimer"`` in 1/m.
        markers: Cell markers of the porous medium. Empty for the whole
            domain.
        speed_floor: Regularization ``s`` of the speed, in m/s, or the
            default of the parameter ``"<name>.speed_floor"``.
    """

    _density: ufl.core.expr.Expr
    """Density of the fluid."""

    _coefficient: ufl.core.expr.Expr
    """Inertial drag coefficient."""

    _speed_floor: ufl.core.expr.Expr
    """Regularization of the speed."""

    def __init__(
        self,
        name: str,
        velocity: Field,
        density: INPUT = None,
        coefficient: INPUT = None,
        markers: tuple[str, ...] = (),
        speed_floor: ufl.core.expr.Expr | float = 1e-12,
    ):
        super().__init__(name, velocity, markers)
        domain = velocity.domain
        self._density = self.input(
            "density", density, domain=domain, unit=DENSITY, doc="Density"
        )
        self._coefficient = self.input(
            "forchheimer",
            coefficient,
            domain=domain,
            unit=1 / syu.m,
            doc="Inertial drag coefficient",
        )
        self._speed_floor = (
            speed_floor
            if isinstance(speed_floor, ufl.core.expr.Expr)
            else self.parameter(
                "speed_floor",
                domain,
                default=speed_floor,
                unit=syu.m / syu.second,
                doc="Regularization of the speed",
            )
        )

    def force(self) -> ufl.core.expr.Expr:
        u = self._velocity
        speed = ufl.sqrt(ufl.dot(u, u) + self._speed_floor**2)
        return self._density * self._coefficient * speed * u


class GradDiv(Residual):
    """Grad-div stabilization, ``(gamma div(u), div(w))``.

    The term vanishes for a velocity without divergence, so it leaves the
    exact solution unchanged, and penalizes the divergence of the discrete
    velocity: it improves the conservation of mass of Taylor-Hood elements,
    and the robustness of the velocity against the pressure. See `John et
    al., 2017 <https://doi.org/10.1137/15M1047696>`_, Section 4.2.

    Args:
        name: Name of the model.
        velocity: The velocity ``u``.
        weight: The weight ``gamma``, a viscosity. Defaults to the parameter
            ``"<name>.weight"`` in Pa s.
        markers: Cell markers of the term. Empty for the whole domain.
    """

    _velocity: Field
    """The velocity."""

    _weight: ufl.core.expr.Expr
    """The weight of the term."""

    _markers: tuple[str, ...]
    """Cell markers of the term."""

    def __init__(
        self,
        name: str,
        velocity: Field,
        weight: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        self._velocity = velocity
        self._weight = self.input(
            "weight",
            weight,
            domain=velocity.domain,
            unit=VISCOSITY,
            doc="Weight of the grad-div term",
        )
        self._markers = tuple(markers)

    def residual(self) -> ufl.Form:
        u = self._velocity
        dx = u.domain.dx(*self._markers)
        return self._weight * ufl.div(u) * ufl.div(u.v) * dx


class _Flow(ResidualSum):
    """A sum of flow components under one name, around a :class:`Stokes` flow.

    Args:
        name: Name of the sum.
        stokes: The Stokes flow of the sum.
        terms: The other terms of the momentum equation.
    """

    _stokes: Stokes
    """The Stokes flow of the sum."""

    def __init__(self, name: str, stokes: Stokes, terms: list[Residual]):
        super().__init__([stokes, *terms], name=name)
        self._stokes = stokes

    @property
    def velocity(self) -> Field:
        """The velocity."""
        return self._stokes.velocity

    @property
    def pressure(self) -> Field:
        """The pressure."""
        return self._stokes.pressure

    def viscous_flux(
        self, u: ufl.core.expr.Expr, grad_u: ufl.core.expr.Expr
    ) -> ufl.core.expr.Expr:
        """The stress of the Stokes flow, see :meth:`Stokes.viscous_flux`."""
        return self._stokes.viscous_flux(u, grad_u)


class NavierStokes(_Flow):
    """Steady incompressible flow with convection.

    The sum of :class:`Stokes` flow, ``"<name>_stokes"``, and
    :class:`Convection`, ``"<name>_convection"``.

    Args:
        name: Name of the model.
        velocity: The velocity ``u``.
        pressure: The pressure ``p``.
        viscosity: Dynamic viscosity ``mu``. Defaults to the parameter
            ``"<name>.viscosity"`` in Pa s.
        density: Density ``rho``. Defaults to the parameter
            ``"<name>.density"`` in kg/m^3.
        markers: Cell markers of the flow region. Empty for the whole domain.
    """

    def __init__(
        self,
        name: str,
        velocity: Field,
        pressure: Field,
        viscosity: INPUT = None,
        density: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        domain = velocity.domain
        mu = input_or_parameter(
            f"{name}.viscosity", viscosity, domain, unit=VISCOSITY, doc="Viscosity"
        )
        rho = input_or_parameter(
            f"{name}.density", density, domain, unit=DENSITY, doc="Density"
        )
        super().__init__(
            name,
            Stokes(f"{name}_stokes", velocity, pressure, mu, markers),
            [Convection(f"{name}_convection", velocity, rho, markers)],
        )


class Brinkman(_Flow):
    """Steady creeping flow through a porous medium.

    The sum of :class:`Stokes` flow of the superficial velocity ``u`` with
    the viscosity ``mu / eps`` in the pores, ``"<name>_stokes"``, and
    :class:`DarcyDrag`, ``"<name>_drag"``. The pressure is that of the fluid
    in the pores. Of the effective viscosities of the Brinkman term in the
    literature, ``mu``, ``mu / eps`` and others, this is the one of volume
    averaging.

    Args:
        name: Name of the model.
        velocity: The superficial velocity ``u``.
        pressure: The pressure ``p``.
        porosity: Porosity ``eps``. Defaults to the parameter
            ``"<name>.porosity"``.
        permeability: Permeability ``K``. Defaults to the parameter
            ``"<name>.permeability"`` in m^2.
        viscosity: Dynamic viscosity ``mu``. Defaults to the parameter
            ``"<name>.viscosity"`` in Pa s.
        markers: Cell markers of the porous medium. Empty for the whole
            domain.
    """

    def __init__(
        self,
        name: str,
        velocity: Field,
        pressure: Field,
        porosity: INPUT = None,
        permeability: INPUT = None,
        viscosity: INPUT = None,
        markers: tuple[str, ...] = (),
    ):
        domain = velocity.domain
        self._porosity = input_or_parameter(
            f"{name}.porosity", porosity, domain, unit=None, doc="Porosity"
        )
        self._permeability = input_or_parameter(
            f"{name}.permeability",
            permeability,
            domain,
            unit=syu.m**2,
            doc="Permeability",
        )
        self._viscosity = input_or_parameter(
            f"{name}.viscosity", viscosity, domain, unit=VISCOSITY, doc="Viscosity"
        )
        super().__init__(
            name,
            Stokes(
                f"{name}_stokes",
                velocity,
                pressure,
                self._viscosity / self._porosity,
                markers,
            ),
            [
                DarcyDrag(
                    f"{name}_drag",
                    velocity,
                    self._viscosity,
                    self._permeability,
                    markers,
                ),
                *self._inertia(name, velocity, markers),
            ],
        )

    def _inertia(
        self, name: str, velocity: Field, markers: tuple[str, ...]
    ) -> list[Residual]:
        """The inertial terms, none for creeping flow."""
        return []


class BrinkmanForchheimer(Brinkman):
    """Steady flow through a porous medium, with inertia.

    The sum of :class:`Brinkman` flow, :class:`Convection` in the pores with
    the density ``rho / eps^2``, ``"<name>_convection"``, and
    :class:`ForchheimerDrag`, ``"<name>_forchheimer"``. With the superficial
    velocity, this is the generalized model of flow through porous media of
    `Hsu and Cheng, 1990 <https://doi.org/10.1016/0017-9310(90)90015-M>`_
    and `Nithiarasu, Seetharamu and Sundararajan, 1997
    <https://doi.org/10.1016/S0017-9310(96)00175-1>`_.

    Args:
        name: Name of the model.
        velocity: See :class:`Brinkman`.
        pressure: See :class:`Brinkman`.
        porosity: See :class:`Brinkman`.
        permeability: See :class:`Brinkman`.
        forchheimer: Inertial drag coefficient ``beta``. Defaults to the
            parameter ``"<name>.forchheimer"`` in 1/m.
        viscosity: See :class:`Brinkman`.
        density: Density ``rho``. Defaults to the parameter
            ``"<name>.density"`` in kg/m^3.
        markers: See :class:`Brinkman`.
        speed_floor: Default of the regularization of ``|u|``, the
            parameter ``"<name>.speed_floor"`` in m/s.
    """

    def __init__(
        self,
        name: str,
        velocity: Field,
        pressure: Field,
        porosity: INPUT = None,
        permeability: INPUT = None,
        forchheimer: INPUT = None,
        viscosity: INPUT = None,
        density: INPUT = None,
        markers: tuple[str, ...] = (),
        speed_floor: float = 1e-12,
    ):
        domain = velocity.domain
        self._forchheimer = input_or_parameter(
            f"{name}.forchheimer",
            forchheimer,
            domain,
            unit=1 / syu.m,
            doc="Inertial drag coefficient",
        )
        self._density = input_or_parameter(
            f"{name}.density", density, domain, unit=DENSITY, doc="Density"
        )
        self._speed_floor = input_or_parameter(
            f"{name}.speed_floor",
            None,
            domain,
            unit=syu.m / syu.second,
            doc="Regularization of the speed",
            default=speed_floor,
        )
        super().__init__(
            name, velocity, pressure, porosity, permeability, viscosity, markers
        )

    def _inertia(
        self, name: str, velocity: Field, markers: tuple[str, ...]
    ) -> list[Residual]:
        """Convection in the pores, and inertial drag."""
        return [
            Convection(
                f"{name}_convection",
                velocity,
                self._density / self._porosity**2,
                markers,
            ),
            ForchheimerDrag(
                f"{name}_forchheimer",
                velocity,
                self._density,
                self._forchheimer,
                markers,
                self._speed_floor,
            ),
        ]
