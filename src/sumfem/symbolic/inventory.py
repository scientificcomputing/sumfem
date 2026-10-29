"""The named objects of forms: fields, equations, parameters and known coefficients."""

from __future__ import annotations

import dataclasses
import types
import typing

import ufl

from ..error_handling import ComponentError, qualified_name
from .domains import Domain
from .expressions import FormOrZero
from .terminals import Field, FieldTerminal, KnownCoefficient, Parameter, field_of
from .time import coefficients_through_time_derivatives

__all__ = ["Inventory", "analyze_forms", "collect_inventory"]


@dataclasses.dataclass(frozen=True)
class Inventory:
    """The named objects found in a set of forms.

    Attributes:
        fields: Every field with a terminal in the forms, in creation order.
        equations: Fields whose test function appears, in creation order.
        trial_fields: Fields whose trial function appears, in creation order.
        parameters: Parameters by name.
        known_coefficients: Known coefficients by name.
        domains: Domains used by the forms, in order of first appearance.
    """

    fields: tuple[Field, ...]
    equations: tuple[Field, ...]
    trial_fields: tuple[Field, ...]
    parameters: typing.Mapping[str, Parameter]
    known_coefficients: typing.Mapping[str, KnownCoefficient]
    domains: tuple[Domain, ...]


def _creation_order(fields: typing.Iterable[Field]) -> tuple[Field, ...]:
    """The distinct fields, ordered by creation.

    Fields are compared by identity, since UFL compares expressions by
    value. Their UFL count, which increases with every new coefficient,
    gives the creation order, so the order is the same however the forms
    list the fields.
    """
    unique = {id(f): f for f in fields}
    return tuple(sorted(unique.values(), key=lambda f: f.count()))


def collect_inventory(
    forms: typing.Sequence[FormOrZero], context: str = "component"
) -> Inventory:
    """Find and validate the named objects used in a set of forms.

    Raises:
        ComponentError: If a form contains an unnamed coefficient, a
            ``ufl.Constant``, an argument that does not belong to a field, a
            mesh that is not a :class:`Domain`, or two different objects with
            the same name.
    """
    fields: list[Field] = []
    equations: list[Field] = []
    trial_fields: list[Field] = []
    parameters: dict[str, Parameter] = {}
    known_coefficients: dict[str, KnownCoefficient] = {}
    domains: dict[int, Domain] = {}
    names: dict[str, object] = {}

    def register_name(name: str, obj: object) -> None:
        other = names.setdefault(name, obj)
        if other is not obj:
            raise ComponentError(
                f"The name {name!r} is used by two different objects in {context}: "
                f"{other!r} and {obj!r}. Names must be unique."
            )

    def register_domain(domain: object) -> None:
        if not isinstance(domain, Domain):
            raise ComponentError(
                f"{context} uses the mesh {domain!r}, "
                f"which is not a {qualified_name(Domain)}."
            )
        domains.setdefault(id(domain), domain)
        register_name(f"domain:{domain.name}", domain)

    for form in forms:
        if not isinstance(form, ufl.Form | ufl.ZeroBaseForm):
            raise ComponentError(
                f"{context} must return a ufl.Form, got {type(form).__name__}."
            )
        constants = form.constants() if isinstance(form, ufl.Form) else ()
        for constant in constants:
            raise ComponentError(
                f"{context} contains the ufl.Constant {constant!r}. "
                f"Use a {qualified_name(Parameter)} instead, so it is named and can be "
                "differentiated against."
            )
        for coefficient in coefficients_through_time_derivatives(form):
            if isinstance(coefficient, Field):
                fields.append(coefficient)
                register_name(coefficient.name, coefficient)
            elif isinstance(coefficient, FieldTerminal):
                fields.append(coefficient.field)
                register_name(coefficient.field.name, coefficient.field)
            elif isinstance(coefficient, Parameter):
                parameters[coefficient.name] = coefficient
                register_name(coefficient.name, coefficient)
            elif isinstance(coefficient, KnownCoefficient):
                known_coefficients[coefficient.name] = coefficient
                register_name(coefficient.name, coefficient)
            else:
                raise ComponentError(
                    f"{context} contains the unnamed coefficient {coefficient!r}. "
                    f"Use a {qualified_name(Field)}, {qualified_name(Parameter)} "
                    f"or {qualified_name(KnownCoefficient)} instead."
                )
            register_domain(coefficient.ufl_function_space().ufl_domain())
        for argument in form.arguments():
            # Only the field's own v and du are allowed. A plain
            # ufl.TestFunction on the field's space would escape the
            # renumbering when the system is built.
            field = field_of(argument)
            if field is None:
                raise ComponentError(
                    f"{context} contains the argument {argument!r}, which does not "
                    f"belong to a {qualified_name(Field)}. Use field.v as test "
                    "function and field.du "
                    "as trial function."
                )
            fields.append(field)
            register_name(field.name, field)
            if argument.number() == 0:
                equations.append(field)
            else:
                trial_fields.append(field)
        for domain in form.ufl_domains():
            register_domain(domain)

    return Inventory(
        fields=_creation_order(fields),
        equations=_creation_order(equations),
        trial_fields=_creation_order(trial_fields),
        parameters=types.MappingProxyType(dict(sorted(parameters.items()))),
        known_coefficients=types.MappingProxyType(
            dict(sorted(known_coefficients.items()))
        ),
        domains=tuple(domains.values()),
    )


def analyze_forms(
    residual_forms: typing.Sequence[FormOrZero],
    linearization_forms: typing.Sequence[FormOrZero | None],
    context: str,
) -> Inventory:
    """The named objects used by a residual's forms.

    Equations come from the residual forms only. Hand-written Jacobians and
    preconditioners only contribute the fields, parameters and known
    coefficients they use.

    Args:
        residual_forms: The residual of each summand.
        linearization_forms: Hand-written Jacobians and preconditioners,
            ``None`` where a summand has none.
        context: Description of the component, used in error messages.
    """
    residual_inventory = collect_inventory(residual_forms, context)
    extra = [f for f in linearization_forms if f is not None]
    full_inventory = collect_inventory([*residual_forms, *extra], context)
    return dataclasses.replace(full_inventory, equations=residual_inventory.equations)
