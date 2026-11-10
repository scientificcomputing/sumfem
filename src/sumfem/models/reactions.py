"""Reactions between species, in volumes and on surfaces.

Species are fields, e.g. concentrations, on a :class:`~sumfem.symbolic.Domain`
or on a lower-dimensional part of one, a :class:`~sumfem.symbolic.Subdomain`. A
reaction adds its rate to the equation of every species it consumes or
produces, so reaction networks are sums of reactions, as in FESTIM,
https://github.com/festim-dev/festim, and SMART,
https://github.com/RangamaniLabUCSD/smart. Diffusion, accumulation and
sources of the species are the models of :mod:`sumfem.models.diffusion`,
on volumes and surfaces alike.
"""

from __future__ import annotations

import typing

import numpy.typing as npt
import ufl
from ufl.algorithms import extract_coefficients

from sumfem.symbolic import BOLTZMANN, Domain, Field, Residual, Subdomain, as_expr
from sumfem.symbolic.expressions import replace

__all__ = ["RateReaction", "Reaction", "arrhenius", "mass_action"]

Participants = typing.Mapping[ufl.core.expr.Expr, int]
"""Species and their stoichiometric coefficients."""


def mass_action(
    rate: ufl.core.expr.Expr, participants: Participants
) -> ufl.core.expr.Expr:
    """The rate of a reaction by the law of mass action, ``k prod(c_i ** nu_i)``.

    By the `law of mass action <https://en.wikipedia.org/wiki/Law_of_mass_action>`_,
    the rate of an elementary reaction is proportional to the product of the
    concentrations of its participants, each to the power of its
    stoichiometric coefficient.

    Args:
        rate: The rate constant ``k``.
        participants: The participants ``c_i`` and their stoichiometric
            coefficients ``nu_i``, e.g. the reactants of the forward reaction.

    Returns:
        The rate ``k prod(c_i ** nu_i)``.
    """
    result = rate
    for species, nu in participants.items():
        result *= species**nu
    return result


def arrhenius(
    prefactor: ufl.core.expr.Expr | float,
    activation_energy: ufl.core.expr.Expr | float,
    temperature: ufl.core.expr.Expr | float,
    boltzmann: float = BOLTZMANN,
) -> ufl.core.expr.Expr:
    """The rate constant by the Arrhenius law, ``k_0 exp(-E / (k_B T))``.

    The `Arrhenius equation <https://en.wikipedia.org/wiki/Arrhenius_equation>`_
    gives the dependence of a rate constant on the temperature, through the
    activation energy of the reaction.

    Args:
        prefactor: ``k_0``.
        activation_energy: ``E``, in the energy unit of ``boltzmann``.
        temperature: ``T``, in K.
        boltzmann: ``k_B`` in J/K, or e.g. in eV/K for ``E`` in eV. A
            parameter or input with a unit is stored in SI, so in J.

    Returns:
        The rate constant ``k_0 exp(-E / (k_B T))``.
    """
    return as_expr(prefactor) * ufl.exp(
        -as_expr(activation_energy) / (boltzmann * as_expr(temperature))
    )


class RateReaction(Residual):
    """A reaction ``sum_i nu_i A_i -> sum_j nu_j B_j`` at a rate ``R``.

    Each time the reaction happens, it consumes ``nu_i`` of each reactant
    ``A_i`` and produces ``nu_j`` of each product ``B_j``. The rate ``R`` is
    how many times it happens, per unit volume and time. So a reactant is
    consumed at ``nu_i R`` and a product produced at ``nu_j R``.

    For example, ``3 A + 4 B -> 4 C`` is ``reactants={A: 3, B: 4}`` and
    ``products={C: 4}``, here at the rate ``R(A, B, C) = k A B / (K + A)``,
    one expression of the concentrations for the whole reaction. The same
    rate, times the coefficient of each species, enters the equations of the
    three species, whose other terms, e.g. diffusion, are left as ``...``::

        Dt(A) + ... + 3 R(A, B, C) = 0
        Dt(B) + ... + 4 R(A, B, C) = 0
        Dt(C) + ... - 4 R(A, B, C) = 0

    As the rate depends on the species, it couples their equations. The
    reaction adds their weak form, ``3 (R, v_A) + 4 (R, v_B) - 4 (R, v_C)``,
    to the residual, with ``v_A``, ``v_B`` and ``v_C`` the test functions.

    The coefficients and the rate are independent here. :class:`Reaction`
    takes the rate from the law of mass action instead.

    A negative rate runs the reaction backwards. A participant that is an
    expression rather than a field, e.g. the free sites of a trap, may enter
    the rate, but has no equation of its own.

    The reaction is integrated over the domain of its species. If one of
    them lives on a :class:`~sumfem.symbolic.Subdomain` of lower dimension,
    e.g. a surface, the reaction happens on that surface, ``R`` is per unit
    area and time, and it is a flux for the species of the volume, e.g. the
    adsorption of a volume species onto a surface species.

    A reaction may also couple the species on the two sides of an interface
    between two subdomains of cells, e.g. a release from the ER into the
    cytosol across the ER membrane. Its measure is then the interface
    measure of :meth:`~sumfem.symbolic.Domain.interface`, ``(first, second,
    marker)``, and each field is taken on its own side: those of ``first``
    on ``"+"``, those of ``second`` on ``"-"``, and the others, e.g. of the
    membrane itself, on ``"+"``.

    Args:
        name: Name of the model.
        reactants: The coefficient ``nu_i`` of each reactant, e.g.
            ``{A: 3, B: 4}``.
        products: The coefficient ``nu_j`` of each product, e.g. ``{C: 4}``.
        rate: The rate ``R``. Defaults to the parameter ``"<name>.rate"``.
        measure: The measure to integrate over, e.g. ``domain.dx("Electrode")``,
            ``domain.ds("Surface")`` for a reaction of volume species at a
            surface, or an interface measure. Defaults to the whole domain
            of the participants.

    Example:
        An enzyme converts a substrate ``S`` into a product ``P`` at the
        rate ``R = v_max S / (K_m + S)`` of `Michaelis-Menten kinetics
        <https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics>`_,
        so ``S`` gets ``+R`` and ``P`` gets ``-R`` in its equation::

            S = Field("S", domain, ("Lagrange", 1))
            P = Field("P", domain, ("Lagrange", 1))
            v_max, K_m = 2.0, 0.5
            enzyme = RateReaction("enzyme", {S: 1}, {P: 1}, rate=v_max * S / (K_m + S))
    """

    _reactants: dict[ufl.core.expr.Expr, int]
    """The reactants and their coefficients."""

    _products: dict[ufl.core.expr.Expr, int]
    """The products and their coefficients."""

    _domain: Domain
    """The domain of the participants, which holds the rate's parameters."""

    _rate: ufl.core.expr.Expr | None
    """The given rate, or ``None`` for a subclass that derives it."""

    _measure: ufl.Measure
    """The measure the reaction is integrated over."""

    def __init__(
        self,
        name: str,
        reactants: Participants,
        products: Participants,
        rate: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        measure: ufl.Measure | None = None,
    ):
        super().__init__(name)
        self._reactants = {as_expr(k): int(v) for k, v in reactants.items()}
        self._products = {as_expr(k): int(v) for k, v in products.items()}
        fields = self._fields()
        if not fields:
            raise ValueError(f"Reaction {name!r} has no species that is a field.")
        for nu in (*self._reactants.values(), *self._products.values()):
            if nu < 1:
                raise ValueError(f"Coefficients must be positive, got {nu}.")
        self._measure = measure  # type: ignore[assignment]
        sides = self._sides()
        if sides is not None:
            self._domain = sides[0].parent
        else:
            self._domain = self._reaction_domain(fields)
        if measure is None:
            self._measure = self._domain.dx()
        self._rate = (
            None
            if type(self)._derives_rate
            else self.input("rate", rate, domain=self._domain, doc="Reaction rate")
        )

    _derives_rate: typing.ClassVar[bool] = False
    """Whether a subclass derives the rate, instead of taking it."""

    @property
    def domain(self) -> Domain:
        """The domain of the participants, which holds the rate's parameters."""
        return self._domain

    def _fields(self) -> list[Field]:
        return [s for s in (*self._reactants, *self._products) if isinstance(s, Field)]

    def _sides(self) -> tuple[Subdomain, Subdomain] | None:
        """The subdomains on the two sides of an interface measure, if it is one."""
        measure = self._measure
        if measure is None or measure.integral_type() != "interior_facet":
            return None
        domain = measure.ufl_domain()
        ids = measure.subdomain_id()
        if (
            not isinstance(domain, Domain)
            or not isinstance(ids, tuple)
            or len(ids) != 1
        ):
            return None
        names = {i: name for name, i in domain.markers[domain.tdim - 1].items()}
        sides = domain.interface_sides(names.get(ids[0], ""))
        return None if sides is None else (sides[0], sides[1])

    def _reaction_domain(self, fields: list[Field]) -> Domain:
        """The domain of lowest dimension among the participants'."""
        lowest = min(f.domain.tdim for f in fields)
        domains = {f.domain for f in fields if f.domain.tdim == lowest}
        if len(domains) > 1:
            raise ValueError(
                f"Reaction {self.name!r} couples species on several domains of "
                f"dimension {lowest}: {sorted(d.name for d in domains)}. Across "
                "an interface between subdomains, give its measure."
            )
        (domain,) = domains
        if lowest < max(f.domain.tdim for f in fields) and not isinstance(
            domain, Subdomain
        ):
            raise ValueError(
                f"Reaction {self.name!r} couples {domain.name!r} with a domain of "
                "higher dimension, but it is not a subdomain of it."
            )
        return domain

    def rate(self) -> ufl.core.expr.Expr:
        """The rate ``R``."""
        assert self._rate is not None
        return self._rate

    def residual(self) -> ufl.Form:
        R = self.rate()
        stoichiometry = [
            (species, nu)
            for species, nu in self._reactants.items()
            if isinstance(species, Field)
        ] + [
            (species, -nu)
            for species, nu in self._products.items()
            if isinstance(species, Field)
        ]
        sides = self._sides()
        if sides is None:
            terms = [nu * R * c.v * self._measure for c, nu in stoichiometry]
        else:
            first, second = sides

            def side(terminal: ufl.core.expr.Expr) -> str:
                domain = getattr(terminal, "domain", None)
                return "-" if domain is second else "+"

            mapping = {t: t(side(t)) for t in extract_coefficients(R)}
            R = replace(R, mapping)
            terms = [nu * R * c.v(side(c)) * self._measure for c, nu in stoichiometry]
        return sum(terms[1:], terms[0])


class Reaction(RateReaction):
    """A reaction ``sum_i nu_i A_i <-> sum_j nu_j B_j`` by mass action.

    By the `law of mass action <https://en.wikipedia.org/wiki/Law_of_mass_action>`_,
    the reaction runs forward at a rate proportional to the concentration of
    each reactant raised to its coefficient, and backward at a rate
    proportional to that of each product raised to its coefficient. The
    rate ``R`` is the difference, and is negative where the reaction runs
    backwards::

        R = k_f prod_i A_i^nu_i - k_b prod_j B_j^nu_j

    with the forward and backward rate constants ``k_f`` and ``k_b``.

    The law holds for elementary reactions; for any other rate law, use
    :class:`RateReaction`.

    For example, ``3 A + 4 B <-> 4 C``, ``reactants={A: 3, B: 4}`` and
    ``products={C: 4}``, has the rate ``R(A, B, C) = k_f A^3 B^4 - k_b C^4``.
    It enters the equations of the species as in :class:`RateReaction`, and
    as it depends on all three, it couples them::

        Dt(A) + ... + 3 R(A, B, C) = 0
        Dt(B) + ... + 4 R(A, B, C) = 0
        Dt(C) + ... - 4 R(A, B, C) = 0

    Args:
        name: Name of the model.
        reactants: The coefficient ``nu_i`` of each reactant, e.g.
            ``{A: 3, B: 4}``.
        products: The coefficient ``nu_j`` of each product, e.g. ``{C: 4}``.
        forward_rate: The rate constant ``k_f``. Defaults to the parameter
            ``"<name>.forward_rate"``.
        backward_rate: The rate constant ``k_b``, zero for an irreversible
            reaction. Defaults to zero.
        measure: See :class:`RateReaction`.

    Example:
        The reversible reaction above, with the forward rate constant a
        parameter ``"r.forward_rate"``, set when it is discretized::

            A, B, C = (Field(name, domain, ("Lagrange", 1)) for name in "ABC")
            reaction = Reaction("r", {A: 3, B: 4}, {C: 4}, backward_rate=0.5)
    """

    _derives_rate = True

    _forward_rate: ufl.core.expr.Expr
    """The forward rate constant."""

    _backward_rate: ufl.core.expr.Expr
    """The backward rate constant."""

    def __init__(
        self,
        name: str,
        reactants: Participants,
        products: Participants,
        forward_rate: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        backward_rate: ufl.core.expr.Expr | npt.ArrayLike = 0.0,
        measure: ufl.Measure | None = None,
    ):
        super().__init__(name, reactants, products, measure=measure)
        self._forward_rate = self.input(
            "forward_rate",
            forward_rate,
            domain=self.domain,
            doc="Forward rate constant",
        )
        self._backward_rate = as_expr(backward_rate)

    def rate(self) -> ufl.core.expr.Expr:
        """The rate ``R = k_f prod(c_i ** nu_i) - k_b prod(c_j ** nu_j)``."""
        forward = mass_action(self._forward_rate, self._reactants)
        backward = mass_action(self._backward_rate, self._products)
        return forward - backward
