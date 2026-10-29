"""Turn a residual into a system of equations with a fixed block layout.

Fields carry test and trial functions with globally unique ``part``
numbers, so forms from different components can be added freely. The unknowns
of a system are the fields its component writes equations for. Building a
system fixes their order, and replaces each field's test and trial
function by the matching argument of a ``ufl.MixedFunctionSpace`` over the
unknowns. Residual and Jacobian blocks
are then extracted with ``ufl.extract_blocks``.
"""

import dataclasses
import functools
import types
import typing

import ufl
from ufl.algorithms import expand_derivatives

from ..error_handling import SystemBuildError
from .components import Jacobian, Residual
from .coordinates import Cartesian, apply_coordinates, supports_vectors
from .domains import Domain, place_subdomain_integrals
from .expressions import FormOrZero, replace
from .inventory import Inventory, analyze_forms
from .terminals import Field, KnownCoefficient, Parameter

__all__ = ["System", "build_system"]

Blocks1 = ufl.Form | tuple[ufl.Form, ...]
Blocks2 = ufl.Form | tuple[tuple[ufl.Form | None, ...], ...]


def _nonempty(form: ufl.Form | ufl.ZeroBaseForm | None) -> ufl.Form | None:
    """Return the form, or ``None`` if it is zero.

    UFL represents a zero block in several ways: ``None`` from
    ``ufl.extract_blocks``, a ``ufl.ZeroBaseForm``, or a form without
    integrals after differentiation. This maps all of them to ``None``, the
    marker for a zero block in a :class:`System`.
    """
    if form is None or isinstance(form, ufl.ZeroBaseForm):
        return None
    if isinstance(form, ufl.Form) and form.empty():
        return None
    return form


def _placed(form: ufl.Form | ufl.ZeroBaseForm | None) -> ufl.Form | None:
    """A non-empty block, with its boundary integrals placed for assembly.

    Blocks are placed separately, since each is assembled over its own mesh.
    """
    block = _nonempty(form)
    return None if block is None else place_subdomain_integrals(block)


@dataclasses.dataclass(frozen=True, eq=False)
class System:
    """The equations of a residual, in a fixed block order.

    A system is given by its component, the block order, substitutions and
    equation scales. Everything else is derived from these on first use.
    The unknowns are the fields the component writes equations for, its data
    are the other terminals of its forms, and its forms are blocked over a
    ``ufl.MixedFunctionSpace`` of the unknowns. :func:`build_system` creates
    systems, and also checks units and computes equation scales.

    With one unknown the residual and Jacobian are single forms. With
    several unknowns the residual is a tuple with one form per unknown, and
    the Jacobian a tuple of rows with ``None`` for zero blocks. Each integral
    is weighted by the coordinate system of its domain, see
    :mod:`~sumfem.symbolic.coordinates`.

    Attributes:
        component: The residual.
        field_order: Block order, a permutation of the unknowns given as
            fields or names. ``None`` orders them by creation.
        substitutions: Replacements in every form before the Jacobian is
            derived, such as those of a time-stepping scheme. Objects they
            introduce become part of the system.
        equation_scales: The size each equation is divided by, in SI base
            units, by the name of its unknown. Empty for dimensional
            equations.

    Raises:
        SystemBuildError: If the component has no equations, if ``field_order``
            is not a permutation of the unknowns, if a hand-written Jacobian
            or preconditioner uses the trial function of a field that is not
            an unknown, if ``equation_scales`` does not give one positive
            scale per unknown, if a vector in space is an unknown on a domain
            whose coordinate system supports scalar unknowns only, or if a
            tensor-valued unknown lies on a domain whose coordinate system is
            not Cartesian.
    """

    component: Residual
    field_order: typing.Sequence[Field | str] | None = None
    substitutions: typing.Mapping[ufl.Coefficient, ufl.core.expr.Expr] = (
        types.MappingProxyType({})
    )
    equation_scales: typing.Mapping[str, float] = types.MappingProxyType({})

    def __post_init__(self) -> None:
        # The derived attributes are computed on first use, but the inputs
        # are checked here, where the system is created
        unknowns = self.unknowns
        for field in self.inventory.trial_fields:
            if not any(field is u for u in unknowns):
                raise SystemBuildError(
                    f"A hand-written Jacobian or preconditioner uses {field.name}.du, "
                    f"but {field.name!r} is not an unknown of this system."
                )
        if self.equation_scales:
            names = sorted(u.name for u in unknowns)
            if sorted(self.equation_scales) != names:
                raise SystemBuildError(
                    f"equation_scales must give a scale for each unknown {names}, "
                    f"got {sorted(self.equation_scales)}."
                )
            invalid = {n: v for n, v in self.equation_scales.items() if not v > 0}
            if invalid:
                raise SystemBuildError(
                    f"Equation scales must be positive, got {invalid}."
                )

    @functools.cached_property
    def _summands(
        self,
    ) -> tuple[tuple[FormOrZero, Jacobian | None, Jacobian | None], ...]:
        """Residual, hand-written Jacobian and preconditioner of each summand.

        The substitutions are applied, the equation scales are not.
        Derivatives are left unevaluated, so that a derived linearization
        includes the substituted terms.
        """
        replacements = dict(self.substitutions)

        def substitute(term: Jacobian | None) -> Jacobian | None:
            if term is None or not replacements:
                return term
            return term.replaced(replacements)

        return tuple(
            (
                replace(m.form, replacements),
                substitute(m.jacobian_term),
                substitute(m.preconditioner_term),
            )
            for m in self.component.components
        )

    @functools.cached_property
    def inventory(self) -> Inventory:
        """The named objects of the component's forms, after substitution."""
        if not self.substitutions:
            return self.component.inventory
        return analyze_forms(
            [F for F, _, _ in self._summands],
            [
                term.form if term is not None else None
                for _, J, P in self._summands
                for term in (J, P)
            ],
            context=f"component {self.component.name!r}",
        )

    @functools.cached_property
    def unknowns(self) -> tuple[Field, ...]:
        """The fields the component writes equations for, in block order."""
        equations = self.inventory.equations
        if len(equations) == 0:
            raise SystemBuildError(
                f"Component {self.component.name!r} has no equations."
            )
        if self.field_order is None:
            return equations
        order = _resolve_fields(self.field_order, equations)
        if len(order) != len(equations):
            raise SystemBuildError(
                "field_order must list every unknown exactly once, got "
                f"{[f.name for f in order]} for unknowns "
                f"{[f.name for f in equations]}."
            )
        return tuple(order)

    @property
    def parameters(self) -> typing.Mapping[str, Parameter]:
        """Parameters used by the system, by name."""
        return self.inventory.parameters

    @functools.cached_property
    def known_coefficients(self) -> typing.Mapping[str, KnownCoefficient | Field]:
        """The data of the system, by name.

        These are its known coefficients, and the fields it uses but does
        not solve for, such as fields solved for in another stage.
        """
        known: dict[str, KnownCoefficient | Field] = {
            f.name: f
            for f in self.inventory.fields
            if not any(f is u for u in self.unknowns)
        }
        known.update(self.inventory.known_coefficients)
        return types.MappingProxyType(dict(sorted(known.items())))

    @property
    def domains(self) -> tuple[Domain, ...]:
        """Domains used by the system."""
        return self.inventory.domains

    @functools.cached_property
    def test_functions(self) -> tuple[ufl.Argument, ...]:
        """The test function of each block, as used in the residual."""
        return _block_arguments(self.unknowns, 0)

    @functools.cached_property
    def trial_functions(self) -> tuple[ufl.Argument, ...]:
        """The trial function of each block, as used in the Jacobian."""
        return _block_arguments(self.unknowns, 1)

    @property
    def residual(self) -> Blocks1:
        """Residual form, or one form per unknown."""
        return self._blocks[0]

    @property
    def jacobian(self) -> Blocks2:
        """Jacobian form, or blocks ``J[i][j] = dF_i/du_j``."""
        return self._blocks[1]

    @property
    def preconditioner(self) -> Blocks2 | None:
        """Preconditioner form or blocks, or ``None``."""
        return self._blocks[2]

    @functools.cached_property
    def _blocks(self) -> tuple[Blocks1, Blocks2, Blocks2 | None]:
        """Residual, Jacobian and preconditioner in the block layout.

        They are built together, since the preconditioner reuses the
        Jacobian of each summand without one.
        """
        order = self.unknowns
        for u in order:
            coordinates = u.domain.coordinates
            name = type(coordinates).__name__
            if u.ufl_shape and u.spatial and not supports_vectors(coordinates):
                raise SystemBuildError(
                    f"The unknown {u.name!r} is vector-valued, on domain "
                    f"{u.domain.name!r} with {name} coordinates, which support "
                    "scalar unknowns only."
                )
            if len(u.ufl_shape) > 1 and not isinstance(coordinates, Cartesian):
                raise SystemBuildError(
                    f"The unknown {u.name!r} is tensor-valued, on domain "
                    f"{u.domain.name!r} with {name} coordinates, where its "
                    "angular components would be unknowns too."
                )
        # Scale each equation through its test function
        scale: dict[typing.Any, ufl.core.expr.Expr] = {
            u.v: u.v / self.equation_scales[u.name]
            for u in order
            if self.equation_scales
        }
        # Then map each field's uniquely numbered arguments to the blocks
        renumber: dict[ufl.Argument, ufl.Argument] = {}
        for field, test, trial in zip(
            order, self.test_functions, self.trial_functions, strict=True
        ):
            renumber[field.v] = test
            renumber[field.du] = trial
        trials = self.trial_functions

        def renumbered(form: FormOrZero) -> FormOrZero:
            return replace(replace(apply_coordinates(form), scale), renumber)

        def linearize(form: FormOrZero) -> FormOrZero:
            if isinstance(form, ufl.ZeroBaseForm):
                return ufl.ZeroBaseForm((*form.arguments(), trials[0]))
            if len(order) == 1:
                return ufl.derivative(form, order[0], trials[0])
            return ufl.derivative(form, list(order), list(trials))

        def total(forms: typing.Iterable[FormOrZero]) -> FormOrZero:
            return functools.reduce(lambda a, b: a + b, forms)

        residual_forms = [F_m for F_m, _, _ in self._summands]
        jacobians = [J_m for _, J_m, _ in self._summands]
        preconditioners = [P_m for _, _, P_m in self._summands]
        F = expand_derivatives(renumbered(total(residual_forms)))

        # The Jacobian of each summand: hand-written if given, else derived
        jacobian_terms = [
            renumbered(J_m.linearized(order))
            if J_m is not None
            else linearize(renumbered(F_m))
            for F_m, J_m in zip(residual_forms, jacobians, strict=True)
        ]
        if all(J_m is None for J_m in jacobians):
            J = expand_derivatives(linearize(F))
        else:
            J = expand_derivatives(total(jacobian_terms))

        P = None
        if any(P_m is not None for P_m in preconditioners):
            P = expand_derivatives(
                total(
                    renumbered(P_m.linearized(order)) if P_m is not None else J_m
                    for P_m, J_m in zip(preconditioners, jacobian_terms, strict=True)
                )
            )

        F_blocks = (F,) if len(order) == 1 else ufl.extract_blocks(F)
        missing = [f.name for f, F_i in zip(order, F_blocks) if _nonempty(F_i) is None]
        if missing:
            raise SystemBuildError(f"The equations for {missing} are empty.")
        if len(order) == 1:
            placed = place_subdomain_integrals
            return placed(F), placed(J), None if P is None else placed(P)

        def as_block_matrix(form: ufl.Form) -> tuple[tuple[ufl.Form | None, ...], ...]:
            rows = ufl.extract_blocks(form)
            return tuple(tuple(_placed(block) for block in row) for row in rows)

        return (
            tuple(place_subdomain_integrals(F_i) for F_i in F_blocks),
            as_block_matrix(J),
            None if P is None else as_block_matrix(P),
        )

    @property
    def blocked(self) -> bool:
        """Whether the system has more than one unknown."""
        return len(self.unknowns) > 1

    def index(self, field: Field | str) -> int:
        """Block index of an unknown, given as a field or by name."""
        for i, unknown in enumerate(self.unknowns):
            if unknown is field or unknown.name == field:
                return i
        name = field.name if isinstance(field, Field) else field
        raise KeyError(f"{name!r} is not an unknown of this system.")

    def derivative(
        self, wrt: Parameter | KnownCoefficient | Field | str
    ) -> ufl.Form | tuple[ufl.Form | None, ...] | None:
        """Derivative of the residual with respect to a parameter or data.

        The result is linear in a trial function on the space of ``wrt``,
        giving the sensitivity forms ``dF_i/dm`` used by adjoint gradients.

        Args:
            wrt: A parameter or a known coefficient of this system, given as
                the object or by name.

        Returns:
            One form, or one form per unknown with ``None`` where the residual
            block does not depend on ``wrt``.
        """
        target = self._resolve_data(wrt)
        trial = ufl.Argument(target.ufl_function_space(), 1)

        def diff(form: ufl.Form) -> ufl.Form | None:
            return _placed(expand_derivatives(ufl.derivative(form, target, trial)))

        if self.blocked:
            assert isinstance(self.residual, tuple)
            return tuple(diff(F_i) for F_i in self.residual)
        assert isinstance(self.residual, ufl.Form)
        return diff(self.residual)

    def _resolve_data(
        self, wrt: Parameter | KnownCoefficient | Field | str
    ) -> ufl.Coefficient:
        """Find the coefficient that :meth:`derivative` differentiates against.

        Args:
            wrt: A parameter or a known coefficient of this system, given as
                the object or by name.

        Raises:
            KeyError: If ``wrt`` is an unknown of the system, whose derivative
                is a column of the Jacobian rather than a sensitivity, or if it
                is not used by the system.
        """
        for unknown in self.unknowns:
            if unknown is wrt or unknown.name == wrt:
                raise KeyError(
                    f"{unknown.name!r} is an unknown. "
                    "Its derivative is a column of the Jacobian."
                )
        name = wrt if isinstance(wrt, str) else wrt.name
        data: dict[str, ufl.Coefficient] = {
            **self.parameters,
            **self.known_coefficients,
        }
        found = data.get(name)
        if found is None:
            raise KeyError(
                f"{name!r} is not a parameter or known coefficient of this system."
            )
        if not isinstance(wrt, str) and found is not wrt:
            raise KeyError(f"{wrt!r} is not used by this system.")
        return found


def _block_arguments(
    unknowns: typing.Sequence[Field], number: int
) -> tuple[ufl.Argument, ...]:
    """The test (``number`` 0) or trial (1) function of each block.

    A single unknown has a plain argument. Several unknowns have the
    arguments of a ``ufl.MixedFunctionSpace``, with the block index as
    ``part``. Arguments compare by space, number and part, so calling this
    again gives arguments equal to those in the system's forms.

    Raises:
        ValueError: If ``number`` is not 0 or 1.
    """
    if number not in (0, 1):
        raise ValueError(
            f"Arguments are numbered 0 for test and 1 for trial functions, "
            f"got {number}."
        )
    if len(unknowns) == 1:
        return (ufl.Argument(unknowns[0].ufl_function_space(), number),)
    W = ufl.MixedFunctionSpace(*[f.ufl_function_space() for f in unknowns])
    arguments = ufl.TestFunctions(W) if number == 0 else ufl.TrialFunctions(W)
    return tuple(arguments)


def _resolve_fields(
    requested: typing.Sequence[Field | str], unknowns: tuple[Field, ...]
) -> list[Field]:
    """Turn a block order, with fields given as objects or by name, into fields.

    Args:
        requested: The fields asked for, in order.
        unknowns: The unknowns of the system.

    Returns:
        The requested fields, in the requested order.

    Raises:
        SystemBuildError: If a field is not an unknown, or is listed twice.
    """
    resolved: list[Field] = []
    for item in requested:
        match = [f for f in unknowns if f is item or f.name == item]
        if len(match) == 0:
            name = item.name if isinstance(item, Field) else item
            raise SystemBuildError(
                f"Field {name!r} in field_order is not an unknown: the component "
                "writes no equation for it. The unknowns are "
                f"{[f.name for f in unknowns]}."
            )
        if any(match[0] is r for r in resolved):
            raise SystemBuildError(
                f"Field {match[0].name!r} is listed twice in field_order."
            )
        resolved.append(match[0])
    return resolved


def build_system(
    component: Residual,
    field_order: typing.Sequence[Field | str] | None = None,
    substitutions: typing.Mapping[ufl.Coefficient, ufl.core.expr.Expr] | None = None,
    check_units: bool = False,
    factorize_equations: bool = False,
    time_scale: float = 1.0,
) -> System:
    """Build the system of equations for a residual.

    The component is not modified.

    Args:
        component: A residual, or a sum of residuals. The unknowns are
            the fields it writes an equation for, with a term in ``field.v``.
            Other fields
            used by the component are known coefficients of the system, as in
            a stage of a sequence.
        field_order: Block order, a permutation of the unknowns, as objects
            or names. Defaults to field creation order.
        substitutions: Coefficients to replace in every form before the
            Jacobian is derived, such as ``{c.dot: (c - c.previous()) / dt}``
            from a time-stepping scheme. Objects introduced by a substitution
            become part of the system.
        check_units: Check that every equation is dimensionally consistent,
            see :func:`~sumfem.symbolic.units.check_units`.
        factorize_equations: Make the equations dimensionless. Each equation
            is divided by its typical size, see
            :func:`~sumfem.symbolic.units.equation_scales`, by dividing
            the test function ``field.v`` of its unknown. The unknowns keep
            their SI values. :attr:`System.equation_scales` records the sizes.
            This checks units too, since the sizes are only meaningful for
            consistent equations.
        time_scale: Reference time in seconds for time derivatives, used by
            the unit check and the equation sizes.

    Raises:
        SystemBuildError: If the component has no equations, if a hand-written
            Jacobian uses the trial function of a field that is not an
            unknown, if ``field_order`` is not a permutation of the unknowns,
            if a vector in space is an unknown on a domain whose coordinate
            system supports scalar unknowns only, or if a tensor-valued unknown lies
            on a domain whose coordinate system is not Cartesian.
        UnitError: If units are checked, or equations factorized, and an
            equation is dimensionally inconsistent.
    """
    # Imported here, since the units module depends on systems
    from . import units

    order = None if field_order is None else tuple(field_order)
    replacements = types.MappingProxyType(dict(substitutions or {}))
    scales: dict[str, float] = {}
    if factorize_equations:
        # Sizes come from the semi-discrete system, where time derivatives are
        # measured against the reference time rather than a time step
        scales = units.equation_scales(System(component, order), time_scale)
    system = System(component, order, replacements, types.MappingProxyType(scales))
    if check_units and not factorize_equations:
        units.check_units(system, time_scale)
    # Build the forms now, so that errors in them show here
    system.jacobian
    return system
