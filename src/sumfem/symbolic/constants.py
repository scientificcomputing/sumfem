"""Physical constants, with their units.

The constants that define the SI since its 2019 revision are exact, so they
are written out rather than taken from a library.
"""

from __future__ import annotations

import weakref

import sympy
import sympy.physics.units as syu

from .domains import Domain
from .si import to_si
from .terminals import Parameter

__all__ = ["BOLTZMANN", "boltzmann_constant"]

_BOLTZMANN_UNIT = syu.joule / syu.kelvin
"""The unit of the Boltzmann constant."""

BOLTZMANN: sympy.Expr = sympy.Rational(1380649, 10**29) * _BOLTZMANN_UNIT
"""The Boltzmann constant ``k_B = 1.380649e-23 J/K``, exact by the definition
of the SI, as a sympy quantity. :func:`boltzmann_constant` gives it on a
domain, for forms."""

_on_domains: weakref.WeakKeyDictionary[Domain, Parameter] = weakref.WeakKeyDictionary()
"""The Boltzmann constant of each domain it was asked for."""


def boltzmann_constant(domain: Domain) -> Parameter:
    """The Boltzmann constant on a domain, a parameter in J/K.

    The parameter is named ``"<domain>.k_B"``, so that a model over several
    domains has one of each, and has :data:`BOLTZMANN` as default. The same
    parameter is returned for every call with a domain.

    Args:
        domain: The domain of the forms that use it.
    """
    if domain not in _on_domains:
        _on_domains[domain] = Parameter(
            f"{domain.name}.k_B",
            domain,
            default=to_si(BOLTZMANN, _BOLTZMANN_UNIT),
            unit=_BOLTZMANN_UNIT,
            doc="Boltzmann constant",
        )
    return _on_domains[domain]
