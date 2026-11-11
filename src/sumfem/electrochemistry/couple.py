"""Redox couples and their reactions."""

from __future__ import annotations

import dataclasses
import types
import typing

import sympy

from .electrolyte import Ion

__all__ = ["RedoxCouple"]


@dataclasses.dataclass(frozen=True)
class RedoxCouple:
    """A redox couple and its reaction at an electrode.

    Oxidation turns the reduced ion into the oxidized one, as the reaction
    ``reactants -> products + n e-``. The reactants hold the reduced ion and
    the products the oxidized one, and either may hold other species of the
    reaction, e.g. VO2+ + H2O -> VO2+ + 2 H+ + e- consumes water and
    produces protons. The coefficients are those of
    :class:`~sumfem.models.RateReaction`. The kinetics are those of
    :class:`~sumfem.electrochemistry.ButlerVolmer`, in the concentrations of
    the couple.

    Attributes:
        reduced: Name of the reduced ion, e.g. ``"V2"``.
        oxidized: Name of the oxidized ion, e.g. ``"V3"``.
        rate_constant: Standard rate constant, in m/s or with units.
        formal_potential: Formal potential, in V or with units.
        oxidation_on_discharge: Whether discharge oxidizes the couple, as in
            the negative half-cell of a battery; the positive half-cell
            reduces its couple on discharge.
        electrons: Number of electrons ``n`` the reaction transfers.
        anodic_transfer: Anodic transfer coefficient.
        cathodic_transfer: Cathodic transfer coefficient.
        reactants: The coefficient of each ion that oxidation consumes, by
            name. Defaults to the reduced ion alone, e.g.
            ``{"VO": 1, "H2O": 1}``.
        products: The coefficient of each ion that oxidation produces.
            Defaults to the oxidized ion alone, e.g. ``{"VO2": 1, "H": 2}``.

    Raises:
        ValueError: If the reactants lack the reduced ion or the products the
            oxidized one.
    """

    reduced: str
    oxidized: str
    rate_constant: float | sympy.Expr
    formal_potential: float | sympy.Expr
    oxidation_on_discharge: bool
    electrons: int = 1
    anodic_transfer: float = 0.5
    cathodic_transfer: float = 0.5
    reactants: typing.Mapping[str, int] | None = None
    products: typing.Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        reactants = dict(self.reactants or {self.reduced: 1})
        products = dict(self.products or {self.oxidized: 1})
        if self.reduced not in reactants or self.oxidized not in products:
            raise ValueError(
                f"The reactants {reactants} must hold the reduced ion "
                f"{self.reduced!r}, and the products {products} the oxidized "
                f"ion {self.oxidized!r}."
            )
        object.__setattr__(self, "reactants", types.MappingProxyType(reactants))
        object.__setattr__(self, "products", types.MappingProxyType(products))

    @property
    def species(self) -> tuple[str, ...]:
        """Names of the ions of the reaction: the reactants, then the products."""
        assert self.reactants is not None and self.products is not None
        return (*self.reactants, *self.products)

    def check(self, ions: typing.Sequence[Ion]) -> None:
        """Check the reaction against the ions of an electrolyte.

        Raises:
            ValueError: If an ion of the reaction is not among the ions, or
                the reaction does not conserve charge.
        """
        assert self.reactants is not None and self.products is not None
        charges = {ion.name: ion.charge for ion in ions}
        missing = [ion for ion in self.species if ion not in charges]
        if missing:
            raise ValueError(f"The electrolyte has no ions {missing} of the reaction.")
        produced = sum(nu * charges[ion] for ion, nu in self.products.items())
        consumed = sum(nu * charges[ion] for ion, nu in self.reactants.items())
        if produced - consumed != self.electrons:
            raise ValueError(
                f"{dict(self.reactants)} -> {dict(self.products)} + "
                f"{self.electrons} e- does not conserve charge: the charge "
                f"goes from {consumed} to {produced}."
            )
