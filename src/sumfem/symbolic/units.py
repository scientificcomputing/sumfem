"""Dimensional analysis of systems, with ``ufl_units``.

Every equation of a system is analyzed on its own, because different
equations, say for a potential and for a concentration, have different
dimensions. Within an equation, every term must have the same dimension.

The analysis needs a typical magnitude of everything in the equations:

- parameters and known coefficients carry their own unit and scale,
- each field contributes its unit and
  :attr:`~sumfem.symbolic.terminals.Field.scale`,
- each domain contributes its
  :attr:`~sumfem.symbolic.domains.Domain.length_scale`,
- time derivatives ``field.dot`` contribute a reference time, ``time_scale``.

Analyse the semi-discrete system, built without a time-stepping scheme, to
get Pi groups in terms of the reference time rather than of the time step.
"""

from __future__ import annotations

import dataclasses
import math

import sympy
import sympy.physics.units as syu
import ufl
import ufl_units
from ufl_units import InconsistentDimensions, Quantity, QuantityMixin

from ..error_handling import ComponentError, qualified_name
from .domains import Domain
from .system import System
from .terminals import Field, FieldRole, FieldTerminal, TimeDerivative

__all__ = [
    "PiGroup",
    "UnitError",
    "check_units",
    "equation_dimensions",
    "equation_scales",
    "pi_groups",
]


class UnitError(ComponentError):
    """Raised when an equation is dimensionally inconsistent."""


@dataclasses.dataclass(frozen=True)
class PiGroup:
    """A dimensionless group of the quantities of a system.

    Attributes:
        expression: The group as a product of powers of quantity symbols.
        value: Its value for the scales of the quantities.
    """

    expression: sympy.Expr
    value: float


def _equations(system: System) -> list[tuple[Field, ufl.Form]]:
    residual = (
        system.residual if isinstance(system.residual, tuple) else (system.residual,)
    )
    return list(zip(system.unknowns, residual, strict=True))


def _integration_domain(field: Field, form: ufl.Form) -> Domain:
    domains = {integral.ufl_domain() for integral in form.integrals()}
    if len(domains) != 1:
        raise ComponentError(
            f"The equation for {field.name!r} is integrated over several domains, "
            "which dimensional analysis does not support."
        )
    (domain,) = domains
    if not isinstance(domain, Domain):
        raise ComponentError(f"{domain!r} is not a {qualified_name(Domain)}.")
    return domain


def _mapping(form: ufl.Form, domain: Domain, time_scale: float) -> dict:
    """Reference quantities of the domain, the fields and their time derivatives."""
    mapping: dict = {
        domain: Quantity(domain, domain.length_scale, syu.meter, f"L_{domain.name}")
    }
    references: dict[int, Quantity] = {}
    time_reference = Quantity(domain, time_scale, syu.second, "t_ref")
    for c in form.coefficients():
        if isinstance(c, Field):
            field, role = c, None
        elif isinstance(c, FieldTerminal):
            field, role = c.field, c.role
        else:
            continue
        reference = references.get(field.count())
        if reference is None:
            reference = Quantity(domain, field.scale, field.unit, f"{field.name}_ref")
            references[field.count()] = reference
        if role is FieldRole.DOT:
            mapping[c] = reference / time_reference * c
        else:
            mapping[c] = reference * c
    for c in form.coefficients():
        if isinstance(c, TimeDerivative):
            # A time derivative has the dimension of its expression per time
            fields = {}
            for field in c.fields:
                reference = references.get(field.count())
                if reference is None:
                    reference = Quantity(
                        domain, field.scale, field.unit, f"{field.name}_ref"
                    )
                    references[field.count()] = reference
                fields[field] = reference * field
            mapping[c] = ufl.replace(c.expression, fields) / time_reference
    return mapping


def _quantities(
    field: Field, form: ufl.Form, time_scale: float
) -> tuple[list[QuantityMixin], dict]:
    """The quantities of an equation, and the mapping."""
    domain = _integration_domain(field, form)
    mapping = _mapping(form, domain, time_scale)
    return ufl_units.collect_quantities(form, mapping=mapping), mapping


def equation_dimensions(
    system: System, time_scale: float = 1.0
) -> dict[str, sympy.Expr]:
    """The dimension of each equation of a system, by field name.

    Args:
        system: The system.
        time_scale: Reference time in seconds, for time derivatives.

    Raises:
        UnitError: If an equation is dimensionally inconsistent.
    """
    dimensions = {}
    for field, form in _equations(system):
        quantities, mapping = _quantities(field, form, time_scale)
        try:
            dimensions[field.name] = ufl_units.get_dimension(
                form, quantities, mapping=mapping
            )
        except InconsistentDimensions as error:
            raise UnitError(
                f"The equation for {field.name!r} is dimensionally inconsistent. "
                f"{error}"
            ) from error
    return dimensions


def check_units(system: System, time_scale: float = 1.0) -> None:
    """Check that every term of every equation has the same dimension.

    Args:
        system: The system.
        time_scale: Reference time in seconds, for time derivatives.

    Raises:
        UnitError: Naming the first dimensionally inconsistent equation.
    """
    for field, form in _equations(system):
        quantities, mapping = _quantities(field, form, time_scale)
        try:
            ufl_units.factorize(form, quantities, mapping=mapping)
        except InconsistentDimensions as error:
            raise UnitError(
                f"The equation for {field.name!r} is dimensionally inconsistent. "
                f"{error}"
            ) from error


def equation_scales(system: System, time_scale: float = 1.0) -> dict[str, float]:
    """The typical size of each equation of a system, by field name.

    Each term of an equation gets a typical size from the scales of the
    quantities in it. The size of the equation is that of its largest term,
    in SI base units. Dividing the equation by it makes its dominant terms
    of order one.

    Args:
        system: The system.
        time_scale: Reference time in seconds, for time derivatives.

    Raises:
        UnitError: If an equation is dimensionally inconsistent.
    """
    # Terms of different dimension have no common size
    check_units(system, time_scale)
    scales = {}
    for field, form in _equations(system):
        quantities, mapping = _quantities(field, form, time_scale)
        sizes = []
        for integral in form.integrals():
            try:
                term = ufl_units.factorize(
                    ufl.Form([integral]), quantities, mapping=mapping
                )
            except InconsistentDimensions as error:
                raise UnitError(
                    f"A term of the equation for {field.name!r} is dimensionally "
                    f"inconsistent. {error}"
                ) from error
            if term.factor is not None:
                sizes.append(
                    abs(
                        math.prod(
                            float(q.factor) ** float(e)
                            for q, e in zip(quantities, term.factor, strict=True)
                        )
                    )
                )
        scales[field.name] = max(sizes, default=1.0) or 1.0
    return scales


def pi_groups(system: System, time_scale: float = 1.0) -> list[PiGroup]:
    """The dimensionless groups of a system, by Buckingham Pi analysis.

    Args:
        system: The system.
        time_scale: Reference time in seconds, for time derivatives.
    """
    by_symbol: dict[str, QuantityMixin] = {}
    for field, form in _equations(system):
        quantities, _ = _quantities(field, form, time_scale)
        for quantity in quantities:
            by_symbol.setdefault(str(quantity.symbol), quantity)
    quantities = list(by_symbol.values())
    _, _, groups = ufl_units.buckingham_pi_analysis(quantities)
    result = []
    for group in groups:
        exponents = [sympy.nsimplify(e) for e in group]
        expression = sympy.Mul(
            *[q.symbol**e for q, e in zip(quantities, exponents, strict=True) if e != 0]
        )
        value = math.prod(
            q.factor ** float(e) for q, e in zip(quantities, exponents, strict=True)
        )
        result.append(PiGroup(expression, float(value)))
    return result
