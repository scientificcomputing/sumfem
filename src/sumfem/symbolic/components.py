"""Components: reusable pieces of residuals and functionals, to extend and sum.

A :class:`Residual` contributes terms to the equations of one or more
fields. Residuals are combined with ``+``::

    cell = Diffusion("solid", phi_s) + ButlerVolmer("bv", phi_s, phi_l)

The sum is the union of the fields, parameters and known coefficients of the
summands, matched by object identity, and the sum of their residuals. Two
components writing terms with the same ``field.v`` contribute to the same
equation.

A :class:`Jacobian` is a bilinear form in ``field.v`` and ``field.du``: a
hand-written linearization of a residual, or a preconditioner. Jacobians sum
the same way, and :meth:`Residual.to_jacobian` gives the derived
linearization of a residual.

A :class:`Functional` contributes to a scalar functional, e.g. an energy,
an objective or a quantity of interest. Functionals sum the same way, and
:meth:`Functional.to_residual` gives the residual of the functional's
stationary point.
"""

from __future__ import annotations

import functools
import typing

import numpy.typing as npt
import ufl
from ufl.domain import extract_unique_domain

from ..error_handling import ComponentError, check_name
from .domains import Domain
from .elements import ElementSpec
from .expressions import FormOrZero, as_expr, check_rank, replace
from .inventory import Inventory, analyze_forms, collect_inventory
from .si import Unit
from .terminals import (
    Field,
    FieldRole,
    FieldTerminal,
    KnownCoefficient,
    Parameter,
    TimeDerivative,
)

__all__ = [
    "Component",
    "Functional",
    "FunctionalSum",
    "Jacobian",
    "JacobianSum",
    "Residual",
    "ResidualSum",
]


class Component:
    """What residuals and functionals share: a name, and named inputs.

    Args:
        name: Unique name of the component. It prefixes the names of
            parameters and known coefficients the component creates, as
            ``"<name>.<parameter>"``.
    """

    _name: str
    """Unique name of the component, prefixing the names of what it creates."""

    _owned_parameters: dict[str, Parameter]
    """Parameters created by :meth:`parameter`, by unqualified name."""

    _owned_known_coefficients: dict[str, KnownCoefficient]
    """Known coefficients created by :meth:`known_coefficient`, by name."""

    def __init__(self, name: str):
        check_name(name, type(self).__name__, allow_dots=False)
        self._name = name
        self._owned_parameters = {}
        self._owned_known_coefficients = {}

    @property
    def name(self) -> str:
        """Name of the component."""
        return self._name

    def parameter(
        self,
        name: str,
        domain: Domain,
        shape: tuple[int, ...] = (),
        default: npt.ArrayLike | None = None,
        unit: Unit | None = None,
        doc: str = "",
    ) -> Parameter:
        """Return the parameter ``"<component>.<name>"``, creating it on first call.

        Calling this again with the same name returns the same object, so it
        is safe to call while building a form.
        """
        check_name(name, Parameter.__name__, allow_dots=False)
        existing = self._owned_parameters.get(name)
        if existing is None:
            existing = Parameter(
                f"{self.name}.{name}",
                domain,
                shape=shape,
                default=default,
                unit=unit,
                doc=doc,
            )
            self._owned_parameters[name] = existing
        elif extract_unique_domain(existing) is not domain or existing.shape != tuple(
            shape
        ):
            raise ComponentError(
                f"{Parameter.__name__} {existing.name!r} was already created "
                f"on {extract_unique_domain(existing)} with shape {existing.shape}."
            )
        return existing

    def known_coefficient(
        self,
        name: str,
        domain: Domain,
        element: ElementSpec,
        unit: Unit | None = None,
        scale: float = 1.0,
        doc: str = "",
    ) -> KnownCoefficient:
        """Return the known coefficient ``"<component>.<name>"``, made on first call."""
        check_name(name, KnownCoefficient.__name__, allow_dots=False)
        existing = self._owned_known_coefficients.get(name)
        if existing is None:
            existing = KnownCoefficient(
                f"{self.name}.{name}", domain, element, unit=unit, scale=scale, doc=doc
            )
            self._owned_known_coefficients[name] = existing
        elif extract_unique_domain(existing) is not domain:
            raise ComponentError(
                f"{KnownCoefficient.__name__} {existing.name!r} was already created "
                f"on {extract_unique_domain(existing)}."
            )
        return existing

    def input(
        self,
        name: str,
        value: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        *,
        domain: Domain,
        element: ElementSpec | None = None,
        shape: tuple[int, ...] = (),
        default: npt.ArrayLike | None = None,
        unit: Unit | None = None,
        scale: float = 1.0,
        doc: str = "",
    ) -> ufl.core.expr.Expr:
        """Resolve an input given to the constructor.

        If ``value`` is given, it is converted with
        :func:`as_expr`. Passing a :class:`Field`
        couples this component to the field. Otherwise a placeholder named
        ``"<component>.<name>"`` is created: a :class:`KnownCoefficient` when
        ``element`` is given, else a :class:`Parameter` of the given
        ``shape``. ``unit`` applies to either, ``scale`` to a known
        coefficient and ``default`` to a parameter.
        """
        if value is not None:
            return as_expr(value)
        if element is not None:
            return self.known_coefficient(
                name, domain, element, unit=unit, scale=scale, doc=doc
            )
        return self.parameter(
            name, domain, shape=shape, default=default, unit=unit, doc=doc
        )

    @property
    def components(self) -> tuple[Component, ...]:
        """The summands. A single component is its own summand."""
        return (self,)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r})"


def _flatten(
    components: typing.Sequence[Component], kind: type[Component]
) -> list[Component]:
    """The summands of a sum: nested sums flattened, each component once."""
    leaves: list[Component] = []
    for component in components:
        if not isinstance(component, kind):
            raise TypeError(
                f"Can only sum {kind.__name__}s, got {type(component).__name__}."
            )
        leaves.extend(component.components)
    if len(leaves) == 0:
        raise ComponentError("Cannot create an empty sum.")
    seen: set[int] = set()
    for leaf in leaves:
        if id(leaf) in seen:
            raise ComponentError(
                f"Component {leaf.name!r} appears more than once in the sum, which "
                "would count its terms twice. Create a second instance instead."
            )
        seen.add(id(leaf))
    return leaves


_C = typing.TypeVar("_C", bound=Component)


class _Sum(Component, typing.Generic[_C]):
    """What the sums share: flattened summands, and their forms added.

    A sum inherits from this and from the kind it sums, and gives that kind's
    hook the sum of the summands' forms, :meth:`_summed_form`.

    Args:
        components: The components to sum.
        name: Name of the sum. Defaults to the summand names joined by ``+``.

    Raises:
        ComponentError: If the same component appears twice.
    """

    _kind: typing.ClassVar[type[Component]]
    """The kind of component summed."""

    _components: tuple[_C, ...]
    """The summands, flattened, in order."""

    def __init__(self, components: typing.Sequence[_C], name: str | None = None):
        leaves = typing.cast(list[_C], _flatten(components, self._kind))
        super().__init__("+".join(m.name for m in leaves) if name is None else name)
        self._components = tuple(leaves)

    @property
    def components(self) -> tuple[_C, ...]:
        """The summands, flattened, in order."""
        return self._components

    def _summed_form(self) -> typing.Any:
        """Sum of the summands' forms."""
        return functools.reduce(
            lambda a, b: a + b,
            (typing.cast(typing.Any, m).form for m in self._components),
        )


class Residual(Component):
    """Base class for a reusable contribution to a residual.

    Subclasses override :meth:`residual` and write it with the fields they
    act on and their test functions ``field.v``. Everything the residual uses is
    discovered from the form, so there is no registration of unknowns,
    parameters or known coefficients. A subclass that overrides one term only
    changes that term.

    The residual is built on first use and then cached, so a component
    should be fully configured before it is summed or turned into a system.

    Args:
        name: Unique name of the component. It prefixes the names of
            parameters and known coefficients the component creates, as
            ``"<name>.<parameter>"``.
    """

    # Hooks for subclasses

    def residual(self) -> FormOrZero:
        """Return this component's contribution to the residual.

        The form must be linear in test functions ``field.v`` and contain no
        trial functions. A contribution that is zero, e.g. for some inputs,
        is ``ufl.ZeroBaseForm((field.v,))``: UFL drops the test function
        from a form whose terms vanish.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement residual().")

    def jacobian(self) -> Jacobian | ufl.Form | None:
        """Return a hand-written linearization of :meth:`residual`, or ``None``.

        A :class:`Jacobian`, or a form with ``field.du`` as trial functions.
        Blocks for unknowns it does not mention are zero. With ``None``, the
        Jacobian is derived automatically.
        """
        return None

    def preconditioner(self) -> Jacobian | ufl.Form | None:
        """Return this component's contribution to a preconditioner, or ``None``.

        If any summand provides a preconditioner, summands without one
        contribute their Jacobian.
        """
        return None

    # Structure and analysis

    @property
    def components(self) -> tuple[Residual, ...]:
        """The summands. A single component is its own summand."""
        return (self,)

    @functools.cached_property
    def form(self) -> FormOrZero:
        """The residual, built on first access and then cached."""
        form = self.residual()
        if not isinstance(form, ufl.Form | ufl.ZeroBaseForm):
            raise ComponentError(
                f"{type(self).__name__}.residual() must return a ufl.Form, "
                f"got {type(form).__name__}."
            )
        if not form.arguments():
            raise ComponentError(
                f"Residual of {self.name!r} has no test function. If it is "
                "zero, return ufl.ZeroBaseForm((field.v,)) for the field whose "
                "equation it belongs to."
            )
        check_rank(form, 1, f"Residual of {self.name!r}")
        return form

    @functools.cached_property
    def jacobian_term(self) -> Jacobian | None:
        """The hand-written Jacobian, if any, built on first access."""
        return _as_jacobian(self.jacobian(), self.name, f"Jacobian of {self.name!r}")

    @functools.cached_property
    def preconditioner_term(self) -> Jacobian | None:
        """The preconditioner contribution, if any, built on first access."""
        return _as_jacobian(
            self.preconditioner(), self.name, f"Preconditioner of {self.name!r}"
        )

    @functools.cached_property
    def inventory(self) -> Inventory:
        """The named objects used by this component, validated."""
        return analyze_forms(
            [m.form for m in self.components],
            [
                term.form if term is not None else None
                for m in self.components
                for term in (m.jacobian_term, m.preconditioner_term)
            ],
            context=f"component {self.name!r}",
        )

    @property
    def fields(self) -> tuple[Field, ...]:
        """Every field the component uses, in creation order."""
        return self.inventory.fields

    @property
    def equations(self) -> tuple[Field, ...]:
        """Fields the component writes equation terms for, in creation order."""
        return self.inventory.equations

    @property
    def parameters(self) -> typing.Mapping[str, Parameter]:
        """Parameters used by the component, by name."""
        return self.inventory.parameters

    @property
    def known_coefficients(self) -> typing.Mapping[str, KnownCoefficient]:
        """Known coefficients used by the component, by name."""
        return self.inventory.known_coefficients

    @property
    def domains(self) -> tuple[Domain, ...]:
        """Domains used by the component."""
        return self.inventory.domains

    # Linearization

    def to_jacobian(
        self, fields: typing.Sequence[Field] | None = None, name: str | None = None
    ) -> Jacobian:
        """The derived linearization ``sum_f derivative(F, f, f.du)``.

        It is derived after the substitutions of the system it is used in,
        e.g. of a time-stepping scheme, so the terms of time derivatives are
        linearized too.

        Args:
            fields: The fields to linearize in. Defaults to the unknowns of
                the system it is used in, or, on its own, the fields this
                residual writes equations for.
            name: Name of the Jacobian. Defaults to the residual's name.
        """
        return _ResidualDerivative(
            self.name if name is None else name,
            self.form,
            None if fields is None else tuple(fields),
        )

    def with_jacobian(self, jacobian: Jacobian) -> Residual:
        """This residual, linearized by the given Jacobian."""
        return _Linearized(self, jacobian=jacobian)

    def with_preconditioner(self, preconditioner: Jacobian) -> Residual:
        """This residual, with the given preconditioner.

        It replaces the whole preconditioner of this residual. Its Jacobian
        is unchanged, including hand-written Jacobians of its summands.
        """
        return _Linearized(self, preconditioner=preconditioner)

    def _linearization(self) -> Jacobian:
        """The Jacobian of each summand: hand-written, or derived."""
        return _sum_jacobians(
            [m.jacobian_term or m.to_jacobian() for m in self.components]
        )

    # Summation

    def __add__(self, other: object) -> ResidualSum:
        if not isinstance(other, Residual):
            return NotImplemented
        return ResidualSum([self, other])

    def __radd__(self, other: object) -> Residual:
        # Support sum(components), which starts from 0
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    # Copying

    def copy(self, suffix: str) -> Residual:
        """Return a copy of this component that solves for new fields.

        Each field the component writes an equation for is replaced by a new
        field named ``"<name><suffix>"``, with the same domain, element, unit
        and scale. Its test and trial functions, time derivative and previous
        values are replaced too. The new fields have their own UFL counts, so
        the copy can be summed with the original.

        Each summand is copied as ``"<name><suffix>"``, together with the
        parameters and known coefficients it created. They keep their
        defaults, units and docs, under the new name. Everything else the
        component uses is shared with the original, such as fields it has no
        equation for and inputs given to its constructor.

        The copy has the original's forms with these replacements. It is not
        an instance of the original's class, so attributes set by the
        original's constructor are not available on it.

        Args:
            suffix: Appended to the names of the copied fields, components,
                parameters and known coefficients, e.g. ``"_pos"``.

        Raises:
            ComponentError: If the suffix is empty.

        Example:
            The positive half-cell of a flow battery has the same equations
            as the negative one::

                positive = negative.copy("_pos")
                cell = negative + positive
        """
        if not suffix:
            raise ComponentError(f"Copying {self.name!r} needs a non-empty suffix.")
        fields = {
            field: Field(
                f"{field.name}{suffix}",
                field.domain,
                field.element,
                unit=field.unit,
                scale=field.scale,
                spatial=field.spatial,
            )
            for field in self.equations
        }
        return self._copy(suffix, fields)[0]

    def _copy(
        self, suffix: str, fields: typing.Mapping[Field, Field]
    ) -> tuple[Residual, dict[ufl.core.expr.Expr, ufl.core.expr.Expr]]:
        """The copy, and what replaces each terminal of the original."""
        copies, replacements = [], {}
        for leaf in self.components:
            copy, leaf_replacements = leaf._copy_leaf(suffix, fields)
            copies.append(copy)
            replacements.update(leaf_replacements)
        if not isinstance(self, ResidualSum):
            (copy,) = copies
            return copy, replacements
        # A sum keeps a name it was given, and otherwise gets the default one
        default_name = "+".join(m.name for m in self.components)
        name = None if self.name == default_name else f"{self.name}{suffix}"
        return ResidualSum(copies, name=name), replacements

    def _copy_leaf(
        self, suffix: str, fields: typing.Mapping[Field, Field]
    ) -> tuple[Residual, dict[ufl.core.expr.Expr, ufl.core.expr.Expr]]:
        copy = _ResidualCopy(self, suffix, fields)
        return copy, copy.replacements


class ResidualSum(_Sum[Residual], Residual):
    """The sum of several residuals.

    Sums are flattened, so ``(a + b) + c`` and ``a + (b + c)`` both have the
    summands ``(a, b, c)``. Use ``a + b`` or ``sum([a, b, c])`` rather than
    constructing this class directly.

    Args:
        components: The components to sum.
        name: Name of the sum. Defaults to the summand names joined by ``+``.

    Raises:
        ComponentError: If the same component appears twice.
    """

    _kind = Residual

    def residual(self) -> FormOrZero:
        """Sum of the summands' residuals."""
        return self._summed_form()


class _ResidualCopy(Residual):
    """A summand of a residual copied by :meth:`Residual.copy`.

    Args:
        original: The summand to copy.
        suffix: Appended to the names of the copy and of what it creates.
        fields: The new field replacing each copied field.
    """

    _original: Residual
    """The copied summand."""

    _replacements: dict[ufl.core.expr.Expr, ufl.core.expr.Expr]
    """What replaces each terminal of the original's forms."""

    def __init__(
        self,
        original: Residual,
        suffix: str,
        fields: typing.Mapping[Field, Field],
    ):
        super().__init__(f"{original.name}{suffix}")
        self._original = original
        replacements: dict[ufl.core.expr.Expr, ufl.core.expr.Expr] = {}
        for name, parameter in original._owned_parameters.items():
            replacements[parameter] = self.parameter(
                name,
                typing.cast(Domain, extract_unique_domain(parameter)),
                shape=parameter.shape,
                default=parameter.default,
                unit=parameter.unit,
                doc=parameter.doc,
            )
        for name, known in original._owned_known_coefficients.items():
            replacements[known] = self.known_coefficient(
                name,
                typing.cast(Domain, extract_unique_domain(known)),
                known.element,
                unit=known.unit,
                scale=known.scale,
                doc=known.doc,
            )
        for old, new in fields.items():
            replacements[old] = new
            replacements[old.v] = new.v
            replacements[old.du] = new.du
        terms = (original.jacobian_term, original.preconditioner_term)
        forms = (original.form, *(t.form for t in terms if t is not None))
        for form in forms:
            for c in () if form is None else form.coefficients():
                if isinstance(c, FieldTerminal) and c.field in fields:
                    new = fields[c.field]
                    replacements[c] = (
                        new.dot if c.role is FieldRole.DOT else new.previous(c.k)
                    )
                elif isinstance(c, TimeDerivative) and any(
                    f in fields for f in c.fields
                ):
                    replacements[c] = TimeDerivative(
                        ufl.replace(c.expression, replacements)
                    )
        self._replacements = replacements

    @property
    def original(self) -> Residual:
        """The copied summand."""
        return self._original

    @property
    def replacements(self) -> dict[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """What replaces each terminal of the original's forms."""
        return self._replacements

    def residual(self) -> FormOrZero:
        """The original's residual, with the copied terminals replaced."""
        return replace(self._original.form, self._replacements)

    def jacobian(self) -> Jacobian | None:
        """The original's hand-written Jacobian, if any, with replacements."""
        term = self._original.jacobian_term
        return None if term is None else term.replaced(self._replacements)

    def preconditioner(self) -> Jacobian | None:
        """The original's preconditioner, if any, with replacements."""
        term = self._original.preconditioner_term
        return None if term is None else term.replaced(self._replacements)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, original={self._original!r})"


class _Linearized(Residual):
    """A residual with a given Jacobian or preconditioner.

    Args:
        original: The residual.
        jacobian: Its Jacobian, or ``None`` to keep the original's.
        preconditioner: Its preconditioner, or ``None`` to keep the
            original's.
    """

    def __init__(
        self,
        original: Residual,
        jacobian: Jacobian | None = None,
        preconditioner: Jacobian | None = None,
    ):
        super().__init__(original.name)
        self._original = original
        self._jacobian = jacobian
        self._preconditioner = preconditioner

    def residual(self) -> FormOrZero:
        """The original's residual."""
        return self._original.form

    def jacobian(self) -> Jacobian | None:
        """The given Jacobian, or the original's, if any summand has one."""
        if self._jacobian is not None:
            return self._jacobian
        summands = self._original.components
        if all(m.jacobian_term is None for m in summands):
            return None
        return self._original._linearization()

    def preconditioner(self) -> Jacobian | None:
        """The given preconditioner, or the original's, if any summand has one."""
        if self._preconditioner is not None:
            return self._preconditioner
        summands = self._original.components
        if all(m.preconditioner_term is None for m in summands):
            return None
        return _sum_jacobians(
            [
                m.preconditioner_term or m.jacobian_term or m.to_jacobian()
                for m in summands
            ]
        )

    def _copy_leaf(
        self, suffix: str, fields: typing.Mapping[Field, Field]
    ) -> tuple[Residual, dict[ufl.core.expr.Expr, ufl.core.expr.Expr]]:
        # The summands of the original are copied with their parameters,
        # and the given Jacobian and preconditioner with the same replacements
        original, replacements = self._original._copy(suffix, fields)
        copy = _Linearized(
            original,
            None if self._jacobian is None else self._jacobian.replaced(replacements),
            None
            if self._preconditioner is None
            else self._preconditioner.replaced(replacements),
        )
        return copy, replacements


class Jacobian(Component):
    """Base class for a reusable bilinear form: a linearization or preconditioner.

    Subclasses override :meth:`jacobian` with a form in the test functions
    ``field.v`` and trial functions ``field.du``. Jacobians sum with ``+``
    into a :class:`JacobianSum`, and a residual takes one as its Jacobian or
    preconditioner with :meth:`Residual.with_jacobian` or
    :meth:`Residual.with_preconditioner`, or from its hooks.

    Args:
        name: Unique name of the component. It prefixes the names of
            parameters and known coefficients the component creates.
    """

    def jacobian(self) -> FormOrZero:
        """Return this component's bilinear form."""
        raise NotImplementedError(f"{type(self).__name__} must implement jacobian().")

    @property
    def components(self) -> tuple[Jacobian, ...]:
        """The summands. A single component is its own summand."""
        return (self,)

    @functools.cached_property
    def form(self) -> FormOrZero:
        """The bilinear form, built on first access and then cached."""
        form = self.jacobian()
        if not isinstance(form, ufl.Form | ufl.ZeroBaseForm):
            raise ComponentError(
                f"{type(self).__name__}.jacobian() must return a ufl.Form, "
                f"got {type(form).__name__}."
            )
        check_rank(form, 2, f"Jacobian {self.name!r}")
        return form

    @functools.cached_property
    def inventory(self) -> Inventory:
        """The named objects used by this component, validated."""
        return collect_inventory(
            [m.form for m in self.components], f"Jacobian {self.name!r}"
        )

    @property
    def fields(self) -> tuple[Field, ...]:
        """Every field the form uses, in creation order."""
        return self.inventory.fields

    @property
    def parameters(self) -> typing.Mapping[str, Parameter]:
        """Parameters used by the form, by name."""
        return self.inventory.parameters

    @property
    def known_coefficients(self) -> typing.Mapping[str, KnownCoefficient]:
        """Known coefficients used by the form, by name."""
        return self.inventory.known_coefficients

    @property
    def domains(self) -> tuple[Domain, ...]:
        """Domains used by the form."""
        return self.inventory.domains

    def __add__(self, other: object) -> JacobianSum:
        if not isinstance(other, Jacobian):
            return NotImplemented
        return JacobianSum([self, other])

    def __radd__(self, other: object) -> Jacobian:
        # Support sum(components), which starts from 0
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    def linearized(self, unknowns: typing.Sequence[Field]) -> FormOrZero:
        """The form in a system with these unknowns.

        Derived linearizations, of :meth:`Residual.to_jacobian` without
        fields, are taken with respect to the unknowns.
        """
        return functools.reduce(
            lambda a, b: a + b, (m._linearized(unknowns) for m in self.components)
        )

    def _linearized(self, unknowns: typing.Sequence[Field]) -> FormOrZero:
        return self.form

    def replaced(
        self, mapping: typing.Mapping[typing.Any, ufl.core.expr.Expr]
    ) -> Jacobian:
        """This Jacobian with terminals replaced, derivatives left unevaluated.

        See :func:`~sumfem.symbolic.expressions.replace`.
        """
        return _sum_jacobians([m._replaced(mapping) for m in self.components])

    def _replaced(
        self, mapping: typing.Mapping[typing.Any, ufl.core.expr.Expr]
    ) -> Jacobian:
        return _FormJacobian(self.name, replace(self.form, mapping))


class JacobianSum(_Sum[Jacobian], Jacobian):
    """The sum of several Jacobians, flattened like a :class:`ResidualSum`.

    Args:
        components: The components to sum.
        name: Name of the sum. Defaults to the summand names joined by ``+``.

    Raises:
        ComponentError: If the same component appears twice.
    """

    _kind = Jacobian

    def jacobian(self) -> FormOrZero:
        """Sum of the summands' forms."""
        return self._summed_form()


class _FormJacobian(Jacobian):
    """A Jacobian given as a form, e.g. by a residual's hook."""

    def __init__(self, name: str, form: FormOrZero):
        super().__init__(name)
        self._form = form

    def jacobian(self) -> FormOrZero:
        """The given form."""
        return self._form


class _ResidualDerivative(Jacobian):
    """The derived linearization of a residual.

    Args:
        name: Name of the component.
        residual: The residual form.
        fields: The fields to linearize in, or ``None`` for the unknowns of
            the system it is used in.
    """

    def __init__(
        self, name: str, residual: FormOrZero, fields: tuple[Field, ...] | None
    ):
        super().__init__(name)
        self._residual = residual
        self._fields = fields

    def jacobian(self) -> FormOrZero:
        """The linearization in the fields the residual writes equations for."""
        fields = self._fields
        if fields is None:
            fields = collect_inventory([self._residual]).equations
        return _derivative(self._residual, fields, "du")

    def _linearized(self, unknowns: typing.Sequence[Field]) -> FormOrZero:
        return _derivative(
            self._residual, unknowns if self._fields is None else self._fields, "du"
        )

    def _replaced(
        self, mapping: typing.Mapping[typing.Any, ufl.core.expr.Expr]
    ) -> Jacobian:
        fields = self._fields
        if fields is not None:
            fields = tuple(typing.cast(Field, mapping.get(f, f)) for f in fields)
        return _ResidualDerivative(self.name, replace(self._residual, mapping), fields)


def _derivative(
    form: FormOrZero,
    fields: typing.Sequence[Field],
    argument: typing.Literal["v", "du"],
) -> FormOrZero:
    """``sum_f derivative(form, f, f.<argument>)``, left unevaluated.

    Along the test functions ``f.v`` it is the variation of a functional, and
    along the trial functions ``f.du`` the linearization of a residual.

    Raises:
        ComponentError: If there are no fields.
    """
    if len(fields) == 0:
        raise ComponentError("A derivative needs at least one field.")
    directions = [getattr(f, argument) for f in fields]
    if isinstance(form, ufl.ZeroBaseForm):
        return ufl.ZeroBaseForm((*form.arguments(), directions[0]))
    if len(fields) == 1:
        return ufl.derivative(form, fields[0], directions[0])
    return ufl.derivative(form, list(fields), directions)


def _sum_jacobians(jacobians: typing.Sequence[Jacobian]) -> Jacobian:
    return jacobians[0] if len(jacobians) == 1 else JacobianSum(jacobians)


def _as_jacobian(
    term: Jacobian | ufl.Form | None, name: str, context: str
) -> Jacobian | None:
    """A hook's Jacobian or preconditioner, as a :class:`Jacobian`."""
    if term is None or isinstance(term, Jacobian):
        return term
    if not isinstance(term, ufl.Form):
        raise ComponentError(
            f"{context} must be a Jacobian or a ufl.Form, got {type(term).__name__}."
        )
    check_rank(term, 2, context)
    return _FormJacobian(name, term)


class Functional(Component):
    """Base class for a reusable contribution to a scalar functional.

    Subclasses override :meth:`functional` with a form of rank zero, e.g. an
    energy, an objective or a quantity of interest. Functionals sum with
    ``+`` into a :class:`FunctionalSum`. The form may be assembled,
    e.g. with
    :meth:`~sumfem.discrete.discretization.Discretization.assemble_scalar`,
    and :meth:`to_residual` gives the residual of its stationary point.

    Args:
        name: Unique name of the component. It prefixes the names of
            parameters and known coefficients the component creates.
    """

    def functional(self) -> ufl.Form:
        """Return this component's contribution to the functional."""
        raise NotImplementedError(f"{type(self).__name__} must implement functional().")

    @property
    def components(self) -> tuple[Functional, ...]:
        """The summands. A single component is its own summand."""
        return (self,)

    @functools.cached_property
    def form(self) -> ufl.Form:
        """The functional, built on first access and then cached."""
        form = self.functional()
        if not isinstance(form, ufl.Form):
            raise ComponentError(
                f"{type(self).__name__}.functional() must return a ufl.Form, "
                f"got {type(form).__name__}."
            )
        check_rank(form, 0, f"Functional {self.name!r}")
        return form

    @functools.cached_property
    def inventory(self) -> Inventory:
        """The named objects used by this component, validated."""
        return collect_inventory(
            [m.form for m in self.components], f"functional {self.name!r}"
        )

    @property
    def fields(self) -> tuple[Field, ...]:
        """Every field the functional uses, in creation order."""
        return self.inventory.fields

    @property
    def parameters(self) -> typing.Mapping[str, Parameter]:
        """Parameters used by the functional, by name."""
        return self.inventory.parameters

    @property
    def known_coefficients(self) -> typing.Mapping[str, KnownCoefficient]:
        """Known coefficients used by the functional, by name."""
        return self.inventory.known_coefficients

    @property
    def domains(self) -> tuple[Domain, ...]:
        """Domains used by the functional."""
        return self.inventory.domains

    def __add__(self, other: object) -> FunctionalSum:
        if not isinstance(other, Functional):
            return NotImplemented
        return FunctionalSum([self, other])

    def __radd__(self, other: object) -> Functional:
        # Support sum(components), which starts from 0
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    def to_residual(
        self, fields: typing.Sequence[Field] | None = None, name: str | None = None
    ) -> Residual:
        """The residual of the functional's stationary point.

        Its residual is ``sum_f derivative(J, f, f.v)`` over the given
        fields, so each field gets the equation of the functional's
        variation in it.

        Args:
            fields: The fields to vary, e.g. the unknowns of one stage.
                Defaults to every field of the functional.
            name: Name of the residual. Defaults to the
                functional's name.
        """
        return _FunctionalDerivative(
            self, self.fields if fields is None else tuple(fields), name
        )


class FunctionalSum(_Sum[Functional], Functional):
    """The sum of several functionals, flattened like a :class:`ResidualSum`.

    Args:
        components: The components to sum.
        name: Name of the sum. Defaults to the summand names joined by ``+``.

    Raises:
        ComponentError: If the same component appears twice.
    """

    _kind = Functional

    def functional(self) -> ufl.Form:
        """Sum of the summands' functionals."""
        return self._summed_form()


class _FunctionalDerivative(Residual):
    """The variation of a functional in some of its fields.

    Args:
        functional: The functional.
        fields: The fields to vary.
        name: Name of the component, by default the functional's.
    """

    def __init__(
        self,
        functional: Functional,
        fields: tuple[Field, ...],
        name: str | None,
    ):
        super().__init__(functional.name if name is None else name)
        if not fields:
            raise ComponentError(f"Functional {functional.name!r} varies no field.")
        self._functional = functional
        self._fields = fields

    @property
    def functional(self) -> Functional:
        """The functional."""
        return self._functional

    def residual(self) -> ufl.Form:
        """``sum_f derivative(J, f, f.v)``."""
        return typing.cast(
            ufl.Form, _derivative(self._functional.form, self._fields, "v")
        )
