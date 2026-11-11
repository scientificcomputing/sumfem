"""Dimensional analysis of systems, with
`ufl-units <https://github.com/fenics-dolfiny/ufl-units>`_, after `Habera and
Zilian, 2026 <https://arxiv.org/abs/2601.06535>`_.

Every equation of a system is analyzed on its own, because different
equations, say for a potential and for a concentration, have different
dimensions, and each gets a reference size of its own (Section 3.6 of the
paper). Within an equation, every term must have the same dimension.

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
import typing

import sympy
import sympy.physics.units as syu
import ufl
import ufl_units
from ufl_units import InconsistentDimensions, Quantity, QuantityMixin

from ..error_handling import ComponentError, qualified_name
from .domains import Domain
from .system import System
from .terminals import (
    Field,
    FieldRole,
    FieldTerminal,
    KnownCoefficient,
    Parameter,
    TimeDerivative,
)

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
    of order one: the normalization by the dominant term of `Habera and
    Zilian, 2026 <https://arxiv.org/abs/2601.06535>`_, Section 3.5.

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


Repeating = Field | Domain | Parameter | KnownCoefficient | str
"""A quantity to build dimensionless groups from: the scale of a field, the
length scale of a domain, a parameter or known coefficient, or a quantity by
its symbol, e.g. ``"t_ref"``."""


def _symbol(quantity: Repeating) -> str:
    """The symbol of a quantity in dimensional analysis, see :func:`_mapping`."""
    if isinstance(quantity, Field):
        return f"{quantity.name}_ref"
    if isinstance(quantity, Domain):
        return f"L_{quantity.name}"
    if isinstance(quantity, Parameter | KnownCoefficient):
        return quantity.name
    return quantity


def pi_groups(
    system: System,
    time_scale: float = 1.0,
    repeating: typing.Sequence[Repeating] = (),
) -> list[PiGroup]:
    """The dimensionless groups of a system, by Buckingham Pi analysis.

    Any product of powers of dimensionless groups is one too, so the groups
    are not unique (`Habera and Zilian, 2026
    <https://arxiv.org/abs/2601.06535>`_, Section 2.3). By default, they are
    a basis of the null space of the dimension matrix, whose form depends on
    the order in which the quantities appear in the forms (Section 4.1 of
    the paper). With ``repeating`` quantities, they
    are those of the method of repeating variables (Çengel and Cimbala,
    *Fluid Mechanics: Fundamentals and Applications*, 2nd ed., McGraw-Hill,
    2010, Section 7.4): one group for every other quantity, that quantity
    times powers of the repeating ones.

    Args:
        system: The system.
        time_scale: Reference time in seconds, for time derivatives.
        repeating: The quantities to build every group from, given by the
            objects whose scales the analysis uses: a
            :class:`~sumfem.symbolic.Domain` for its ``length_scale``, a
            :class:`~sumfem.symbolic.Field` for its ``scale``, a
            :class:`~sumfem.symbolic.Parameter` or
            :class:`~sumfem.symbolic.KnownCoefficient` for its value, or a
            symbol, such as ``"t_ref"`` for ``time_scale``. Choose one for
            each base dimension of the problem, e.g. the length, the scale of
            the unknown and a material coefficient: they must be
            dimensionally independent, and their dimensions must make up
            those of every other quantity.

    Raises:
        ComponentError: If a repeating quantity is not one of the system,
            or the repeating quantities are not dimensionally independent,
            or do not make up the dimensions of the others.

    Example:
        In heat conduction ``-div(kappa grad(T)) = q`` on ``domain``, the
        length scale of the domain, the scale of ``T`` and the conductivity
        make up the dimensions of the source, in which mass and time only
        appear together, as kg/s^3. They give one group, of the source::

            pi_groups(system, repeating=[domain, T, kappa])
            # [PiGroup(expression=L_domain**2*q/(T_ref*kappa), value=...)]
    """
    by_symbol: dict[str, QuantityMixin] = {}
    for field, form in _equations(system):
        quantities, _ = _quantities(field, form, time_scale)
        for quantity in quantities:
            by_symbol.setdefault(str(quantity.symbol), quantity)
    if not repeating:
        quantities = list(by_symbol.values())
        _, _, groups = ufl_units.buckingham_pi_analysis(quantities)
        exponents = [[sympy.nsimplify(e) for e in group] for group in groups]
    else:
        quantities, exponents = _repeating_groups(
            by_symbol, [_symbol(q) for q in repeating]
        )
    result = []
    for group in exponents:
        expression = sympy.Mul(
            *[q.symbol**e for q, e in zip(quantities, group, strict=True) if e != 0]
        )
        value = math.prod(
            q.factor ** float(e) for q, e in zip(quantities, group, strict=True)
        )
        result.append(PiGroup(expression, float(value)))
    return result


def _repeating_groups(
    by_symbol: dict[str, QuantityMixin], symbols: list[str]
) -> tuple[list[QuantityMixin], list[list[sympy.Expr]]]:
    """The groups of each other quantity with the repeating ones.

    Each group is ``q_j prod(r_i ** a_i)``, with the exponents ``a`` solving
    ``D_r a = -d_j`` for the dimension matrix ``D_r`` of the repeating
    quantities and the dimensions ``d_j`` of ``q_j``.

    Returns:
        The quantities, the repeating ones first, and the exponents of each
        group in that order.
    """
    missing = [s for s in symbols if s not in by_symbol]
    if missing:
        raise ComponentError(
            f"The repeating quantities {missing} are not quantities of the system, "
            f"{list(by_symbol)}."
        )
    others = [s for s in by_symbol if s not in symbols]
    quantities = [by_symbol[s] for s in symbols + others]
    matrix, _ = ufl_units.dimension_matrix(quantities)
    count = len(symbols)
    repeating = matrix[:, :count]
    if repeating.rank() != count:
        raise ComponentError(
            f"The repeating quantities {symbols} are not dimensionally independent."
        )
    if repeating.rank() != matrix.rank():
        raise ComponentError(
            f"The dimensions of the repeating quantities {symbols} do not make up "
            "those of the others, which then form no group with them."
        )
    exponents = []
    for j in range(count, len(quantities)):
        solution, _ = repeating.gauss_jordan_solve(-matrix[:, j])
        group = [sympy.nsimplify(e) for e in solution] + [sympy.Integer(0)] * len(
            others
        )
        group[j] = sympy.Integer(1)
        exponents.append(group)
    return quantities, exponents
