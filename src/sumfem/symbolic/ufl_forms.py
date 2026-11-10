"""Residuals from plain UFL forms, as in the UFL files of DOLFINx C++ demos.

A residual, and optionally its Jacobian and a preconditioner, written with
plain UFL objects, a :class:`ufl.Mesh`, :class:`ufl.FunctionSpace`,
:class:`ufl.Coefficient`, :class:`ufl.Constant` and arguments, without
data, as in the UFL files of the DOLFINx C++ demos, is turned by
:class:`UFLResidual` into a residual on sumfem terminals. It is discretized
and solved like any residual.

A model written both in plain UFL and from components gives the same
discrete residual, Jacobian and solution, which verifies the components.
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Mapping, Sequence

import ufl
from ufl.classes import Argument, Coefficient, Constant, GeometricQuantity
from ufl.corealg.traversal import unique_pre_traversal
from ufl.domain import extract_unique_domain

from ..error_handling import ComponentError
from .components import Residual
from .domains import MEASURE_ENTITY_TYPE, Domain, MeasureName
from .expressions import replace
from .terminals import Field, KnownCoefficient, Parameter

__all__ = ["UFLResidual"]


class UFLResidual(Residual):
    """A residual based on plain UFL forms (with no data attached),
    as done in DOLFINx C++ API.

    For example, Poisson's equation::

        u = ufl.Coefficient(V)
        v = ufl.TestFunction(V)
        F = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
        poisson = UFLResidual("poisson", F, u, names={"kappa": kappa, "f": f})

    and a linear problem ``a(u, v) = L(v)``::

        UFLResidual("linear", ufl.action(a, u) - L, u, jacobian=a)

    Several unknowns are written on a :class:`ufl.MixedFunctionSpace`, e.g.
    Stokes flow::

        W = ufl.MixedFunctionSpace(V, Q)
        u, p = ufl.Coefficient(V), ufl.Coefficient(Q)
        v, q = ufl.TestFunctions(W)
        F = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q) * ufl.dx
        stokes = UFLResidual("stokes", F, [u, p], names={"u": u, "p": p})

    The constants and coefficients of the forms become parameters and known
    coefficients of this component, set when it is discretized, and the
    unknowns become its fields, :attr:`unknowns`.

    Args:
        name: Name of the component. Its parameters and known coefficients
            are named ``"<name>.<variable>"``, e.g. ``"poisson.kappa"``.
        residual: The residual, linear in test functions on the space of
            ``unknown``, or on the sub-spaces of a mixed space.
        unknown: The coefficient to solve for, or one per sub-space of a
            mixed space, in its order.
        jacobian: The linearization, bilinear on the same spaces. Derived
            from the residual if not given.
        preconditioner: A form to precondition the Jacobian with, bilinear
            on the same spaces.
        names: The variable name of the forms' meshes, coefficients and
            constants. Needed for each coefficient and constant not given in
            ``terminals`` or ``previous``, except the unknowns, which default
            to ``"u"``, or ``"u0"``, ``"u1"``, ... Meshes default to
            ``"mesh0"``, ``"mesh1"``, ...
        domains: The domain of a mesh, e.g. from
            :meth:`~sumfem.symbolic.domains.Domain.from_mesh`, or a
            :class:`~sumfem.symbolic.domains.Subdomain` for a submesh. Other
            meshes take the domain of their ``terminals``, or one created
            from their coordinate element.
        markers: The marker name of each integer subdomain id, by measure,
            e.g. ``{"ds": {1: "Dirichlet"}}`` for ``ds(1)``. A tuple of ids,
            ``ds((1, 2))``, needs a name for each, and each id its own name.
        terminals: What replaces a coefficient or constant of the forms
            that is defined elsewhere: the field of an unknown, or any UFL
            expression of the same shape, e.g. the field of another component,
            ``{T: heat.field}``, an expression of it, ``{b: -ufl.grad(T)}``,
            or a time step, ``{dt: clock.dt}``.
        previous: The unknown one, two, ... steps back, e.g. ``(u_n, u_nn)``,
            for a time discretization written out, or, for several unknowns,
            those of each, e.g. ``{u: (u_n,)}``. A single step may be given
            alone, ``u_n``. They become
            :meth:`~sumfem.symbolic.terminals.Field.previous` of its field.

    Raises:
        ComponentError: If a form has other arguments than described above,
            a coefficient, constant or subdomain id has no name, or a
            terminal or earlier step has another element or shape.
    """

    _fields: tuple[Field, ...]
    """The unknowns, in the order given."""

    _residual: ufl.Form
    """The residual, on sumfem terminals."""

    _jacobian: ufl.Form | None
    """The given linearization, if any, on sumfem terminals."""

    _preconditioner: ufl.Form | None
    """The given preconditioner, if any, on sumfem terminals."""

    def __init__(
        self,
        name: str,
        residual: ufl.Form,
        unknown: ufl.Coefficient | Sequence[ufl.Coefficient],
        jacobian: ufl.Form | None = None,
        preconditioner: ufl.Form | None = None,
        *,
        names: Mapping[str, typing.Any] | None = None,
        domains: Mapping[ufl.Mesh, Domain] | None = None,
        markers: Mapping[str, Mapping[int, str]] | None = None,
        terminals: Mapping[typing.Any, typing.Any] | None = None,
        previous: (
            ufl.Coefficient
            | Sequence[ufl.Coefficient]
            | Mapping[ufl.Coefficient, ufl.Coefficient | Sequence[ufl.Coefficient]]
        ) = (),
    ):
        unknowns = (
            [unknown] if isinstance(unknown, ufl.core.expr.Expr) else list(unknown)
        )
        if not unknowns or not all(isinstance(u, Coefficient) for u in unknowns):
            raise ComponentError(
                f"The unknown must be a ufl.Coefficient or a sequence of them, got "
                f"{unknown!r}."
            )
        if {a.number() for a in residual.arguments()} != {0}:
            raise ComponentError(
                f"The residual of {name!r} must be linear in a test function."
            )
        # Checking arity of jacobian and preconditioners
        bilinear = {"jacobian": jacobian, "preconditioner": preconditioner}
        for role, form in bilinear.items():
            if form is not None and {a.number() for a in form.arguments()} != {0, 1}:
                raise ComponentError(
                    f"The {role} of {name!r} must be bilinear, with a test and a "
                    "trial function."
                )
        # Checks that every test and trial function
        # of each form belongs to an unknown and is on its space.
        for form in (residual, jacobian, preconditioner):
            for argument in () if form is None else form.arguments():
                _extract_index(argument, unknowns, name)

        # Each integer subdomain id needs a marker name of its own: UFL splits
        # ds((1, 2)) into an integral per id, so ids sharing a name would each
        # integrate over the region of all of them
        for measure, marker_names in (markers or {}).items():
            names_of_ids = list(marker_names.values())
            shared = sorted({n for n in names_of_ids if names_of_ids.count(n) > 1})
            if shared:
                ids = [i for i, n in marker_names.items() if n == shared[0]]
                raise ComponentError(
                    f"The ids {ids} of {measure} in {name!r} share the marker name "
                    f"{shared[0]!r}: give each id its own name, e.g. "
                    f"{{{ids[0]}: {shared[0] + '_' + str(ids[0])!r}, ...}}."
                )

        # The variable name of each object in names, looked up by identity, as
        # names may hold values that cannot be hashed
        names_by_id = {id(value): key for key, value in (names or {}).items()}
        super().__init__(name)
        # The domain of each mesh of the forms, filled as meshes are found
        meshes: dict[ufl.Mesh, Domain] = dict(domains or {})

        # Replace the terminals of the forms given in terminals, coefficients
        # and constants, by UFL expressions, including existing coefficients and
        # constants. An unknown can only be replaced by a field. This also maps
        # the plain mesh of each terminal to the symbolic domain of its
        # replacement, so the forms' integrals are on that domain.
        terminal_replacements: dict[Coefficient | Constant, ufl.core.expr.Expr] = {}
        for terminal, replacement in (terminals or {}).items():
            # Only coefficients and constants are replaced; arguments are
            # mapped to the unknowns by their part
            if not isinstance(terminal, Coefficient | Constant):
                raise ComponentError(
                    f"{terminal!r} in terminals is not a coefficient or constant."
                )
            replacement = ufl.as_ufl(replacement)
            # An unknown becomes the field it is given, which must have its
            # element, as the system is built on the field's space. Anything
            # else may be any expression of the same shape.
            if isinstance(terminal, Coefficient) and terminal in unknowns:
                if (
                    not isinstance(replacement, Coefficient)
                    or replacement.ufl_element() != terminal.ufl_element()
                ):
                    raise ComponentError(
                        f"{_label(terminal, names_by_id)} is an unknown: it must be "
                        f"given a field with the element {terminal.ufl_element()}, "
                        f"not {replacement}."
                    )
            elif replacement.ufl_shape != terminal.ufl_shape:
                raise ComponentError(
                    f"{_label(terminal, names_by_id)} has the shape "
                    f"{terminal.ufl_shape}, but {replacement} has "
                    f"{replacement.ufl_shape}."
                )
            # The mesh of the terminal takes the domain of its replacement, if
            # it has one, unless domains gives one
            replacement_mesh = extract_unique_domain(replacement)
            if replacement_mesh is not None:
                mesh = typing.cast(ufl.Mesh, extract_unique_domain(terminal))
                meshes.setdefault(mesh, typing.cast(Domain, replacement_mesh))
            terminal_replacements[terminal] = replacement

        # The earlier steps of each unknown, which become field.previous(k).
        # They are given as the unknown is: a sequence for a single unknown,
        # and a mapping by unknown for several, as a sequence cannot say which
        # unknown each step belongs to. Unknowns without steps are left out.
        history: dict[ufl.Coefficient, tuple[ufl.Coefficient, ...]] = {}
        if isinstance(previous, Mapping):
            for u, steps in previous.items():
                if u not in unknowns:
                    raise ComponentError(
                        f"{_label(u, names_by_id)} in previous of {name!r} is not "
                        "an unknown."
                    )
                history[u] = _extract_previous_steps(steps, u, names_by_id)
        elif len(unknowns) == 1:
            history[unknowns[0]] = _extract_previous_steps(
                previous, unknowns[0], names_by_id
            )
        elif previous:
            raise ComponentError(
                f"{name!r} has several unknowns: give previous by unknown, e.g. "
                "{u: (u_n,)}."
            )

        # The field of each unknown: the one given in terminals, or a new field
        # with the unknown's element on the domain of its mesh, named after its
        # variable, or "u", or "u0", "u1", ... for several unknowns
        fields = []
        for i, u in enumerate(unknowns):
            field = terminal_replacements.get(u)
            if field is None:
                default = "u" if len(unknowns) == 1 else f"u{i}"
                space = u.ufl_function_space()
                field = Field(
                    names_by_id.get(id(u), default),
                    _get_or_create_domain(space.ufl_domain(), meshes, names_by_id),
                    space.ufl_element(),
                )
            if not isinstance(field, Field):
                raise ComponentError(
                    f"The unknown of {name!r} is given {field}, not a field."
                )
            fields.append(field)
        self._fields = tuple(fields)

        # What replaces each plain terminal of the forms: terminal_replacements,
        # each unknown by its field and each of its earlier steps by the field
        # k steps back. _update_replacement_map adds the arguments and the
        # other terminals.
        replacement_map: dict[typing.Any, typing.Any] = dict(terminal_replacements)
        for u, field in zip(unknowns, fields, strict=True):
            replacement_map[u] = field
            for k, terminal in enumerate(history.get(u, ()), start=1):
                replacement_map[terminal] = field.previous(k)

        # The residual goes first in each stage, so its domains and terminals
        # are created first
        forms = [f for f in (residual, jacobian, preconditioner) if f is not None]
        # Stage 1: the symbolic domain of every mesh of the forms
        for form in forms:
            _update_meshes(form, meshes, names_by_id)
        # Stage 2: complete the replacement map with every terminal of the
        # forms, creating the parameters and known coefficients of this
        # component
        for form in forms:
            self._update_replacement_map(
                form, replacement_map, fields, unknowns, names_by_id, meshes
            )
        # Stage 3: the forms on sumfem objects, which creates nothing more
        replace_terminals = functools.partial(
            _replace_terminals,
            replacement_map=replacement_map,
            meshes=meshes,
            markers=markers or {},
            name=name,
        )
        self._residual = replace_terminals(residual)
        self._jacobian = None if jacobian is None else replace_terminals(jacobian)
        self._preconditioner = (
            None if preconditioner is None else replace_terminals(preconditioner)
        )

    @property
    def field(self) -> Field:
        """The unknown, if there is one.

        Raises:
            ComponentError: If there are several, see :attr:`unknowns`.
        """
        if len(self._fields) != 1:
            raise ComponentError(
                f"{self.name!r} has {len(self._fields)} unknowns; use unknowns."
            )
        return self._fields[0]

    @property
    def unknowns(self) -> tuple[Field, ...]:
        """The unknowns, in the order given."""
        return self._fields

    def residual(self) -> ufl.Form:
        return self._residual

    def jacobian(self) -> ufl.Form | None:
        return self._jacobian

    def preconditioner(self) -> ufl.Form | None:
        return self._preconditioner

    def _reconstruct_terminal(
        self,
        terminal: Coefficient | Constant | GeometricQuantity,
        name: str | None,
        domain: Domain,
    ) -> Parameter | KnownCoefficient | GeometricQuantity:
        """The sumfem terminal for a plain terminal of the forms.

        A constant becomes a parameter of this component and a coefficient a
        known coefficient of it, both registered on it under ``name``, as
        ``"<component>.<name>"``. A geometric quantity, e.g. the spatial
        coordinate, is recreated on the domain.

        Args:
            terminal: A constant, coefficient or geometric quantity of the
                forms, not given in ``terminals`` or ``previous``.
            name: Its variable name, from ``names``.
            domain: The symbolic domain of the terminal's mesh.

        Raises:
            ComponentError: If a constant or coefficient has no name.
        """
        if isinstance(terminal, GeometricQuantity):
            return type(terminal)(domain)
        if name is None:
            raise ComponentError(
                f"{terminal!r} in {self.name!r} has no name. Name it with "
                "names={'<name>': ...}."
            )
        if isinstance(terminal, Constant):
            return self.parameter(name, domain, shape=terminal.ufl_shape)
        return self.known_coefficient(name, domain, terminal.ufl_element())

    def _update_replacement_map(
        self,
        form: ufl.Form,
        replacement_map: dict[typing.Any, typing.Any],
        fields: Sequence[Field],
        unknowns: Sequence[ufl.Coefficient],
        names_by_id: Mapping[int, str],
        meshes: Mapping[ufl.Mesh, Domain],
    ) -> None:
        """Add the replacements of a form's new terminals to the map.

        A terminal already in ``replacement_map`` is kept: an unknown, an
        earlier step, a terminal given in ``terminals``, or one of another form
        of this component, so a terminal the forms share becomes one sumfem
        object, e.g. one parameter. Of the others, a test or trial function
        becomes ``v`` or ``du`` of its unknown's field, and anything else its
        sumfem terminal from :meth:`_reconstruct_terminal`, which registers
        parameters and known coefficients on this component.

        Args:
            form: A form of the component.
            replacement_map: The sumfem object of each plain terminal found so
                far. Modified in place.
            fields: The field of each unknown.
            unknowns: The unknowns, in the order of ``fields``.
            names_by_id: The variable names.
            meshes: The domain of each mesh of the form, see
                :func:`_update_meshes`.
        """
        for terminal in _extract_terminals(form):
            if terminal in replacement_map:
                continue
            if isinstance(terminal, Argument):
                field = fields[_extract_index(terminal, unknowns, self.name)]
                replacement_map[terminal] = (
                    field.v if terminal.number() == 0 else field.du
                )
                continue
            mesh = typing.cast(ufl.Mesh, extract_unique_domain(terminal))
            replacement_map[terminal] = self._reconstruct_terminal(
                terminal, names_by_id.get(id(terminal)), meshes[mesh]
            )


def _label(terminal: typing.Any, names_by_id: Mapping[int, str]) -> str:
    """The variable name of a terminal, or its UFL name."""
    return repr(names_by_id.get(id(terminal), str(terminal)))


def _extract_index(
    argument: Argument, unknowns: Sequence[ufl.Coefficient], name: str
) -> int:
    """The index of the unknown that a test or trial function belongs to.

    An argument on a sub-space of a :class:`ufl.MixedFunctionSpace` has the
    index of the sub-space as its part, and belongs to that unknown. An
    argument without a part belongs to the unknown if there is only one.

    Args:
        argument: A test or trial function of the forms.
        unknowns: The unknowns, one per sub-space, in its order.
        name: Name of the component, for the error messages.

    Returns:
        The index of the argument's unknown in ``unknowns``.

    Raises:
        ComponentError: If the argument has no part and there are several
            unknowns, its sub-space has no unknown, or it is not on the space
            of its unknown.
    """
    index = argument.part()
    # Argument that does not stem from MixedFunctionSpace
    if index is None:
        if len(unknowns) > 1:
            raise ComponentError(
                f"The argument {argument} of {name!r} has no part, but there are "
                f"{len(unknowns)} unknowns: write the forms on a "
                "ufl.MixedFunctionSpace of their spaces, with ufl.TestFunctions "
                "and ufl.TrialFunctions."
            )
        index = 0
    # Argument in MixedSpace, but no unknown for that sub-space
    elif index >= len(unknowns):
        raise ComponentError(
            f"The argument {argument} of {name!r} is on sub-space {index} of a "
            "mixed space, which has no unknown: give one unknown per sub-space."
        )
    # Also for the only unknown: the argument may be on another mixed function space
    if argument.ufl_function_space() != unknowns[index].ufl_function_space():
        if len(unknowns) == 1:
            raise ComponentError(
                f"The argument {argument} of {name!r} is not on the space of the "
                "unknown."
            )
        raise ComponentError(
            f"The argument {argument} of {name!r} is not on the space of unknown "
            f"{index}: give the unknowns in the order of the sub-spaces of the "
            "mixed space."
        )
    return index


def _extract_previous_steps(
    steps: ufl.Coefficient | Sequence[ufl.Coefficient],
    unknown: ufl.Coefficient,
    names_by_id: Mapping[int, str],
) -> tuple[ufl.Coefficient, ...]:
    """The earlier steps of an unknown, checked.

    A single coefficient is not iterated: UFL would iterate over its
    components, or fail for a scalar.

    Args:
        steps: The unknown one, two, ... steps back, or a single step.
        unknown: The unknown they belong to.
        names_by_id: The variable names, for the error messages.

    Returns:
        The steps, one step back first.

    Raises:
        ComponentError: If a step is not a coefficient, or has another element
            than the unknown, as it becomes a value of the same field.
    """
    steps = (steps,) if isinstance(steps, ufl.core.expr.Expr) else tuple(steps)
    for k, step in enumerate(steps, start=1):
        if not isinstance(step, Coefficient):
            raise ComponentError(f"{step!r} in previous is not a coefficient.")
        if step.ufl_element() != unknown.ufl_element():
            raise ComponentError(
                f"{_label(step, names_by_id)}, {k} steps back, has another element "
                "than its unknown."
            )
    return steps


def _replace_terminals(
    form: ufl.Form,
    replacement_map: Mapping[typing.Any, typing.Any],
    meshes: Mapping[ufl.Mesh, Domain],
    markers: Mapping[str, Mapping[int, str]],
    name: str,
) -> ufl.Form:
    """The form on sumfem objects instead of plain UFL ones.

    Each terminal is replaced by its entry in ``replacement_map``, and each
    integral is moved from its ``ufl.Mesh`` to the symbolic
    :class:`~sumfem.symbolic.domains.Domain` of the mesh, with its integer
    subdomain ids replaced by marker ids.

    Args:
        form: A form of the component.
        replacement_map: The sumfem object of each terminal of the form, see
            :meth:`UFLResidual._update_replacement_map`.
        meshes: The domain of each mesh of the form, see
            :func:`_update_meshes`.
        markers: The marker name of each integer subdomain id, by measure.
        name: Name of the component, for the error messages.

    Raises:
        ComponentError: If a subdomain id has no marker name.
    """
    integrals = []
    for integral in form.integrals():
        domain = meshes[integral.ufl_domain()]
        integrals.append(
            integral.reconstruct(
                integrand=replace(integral.integrand(), replacement_map),
                domain=domain,
                subdomain_id=_subdomain_marker_ids(integral, domain, markers, name),
            )
        )
    return ufl.Form(integrals)


def _subdomain_marker_ids(
    integral: ufl.Integral,
    domain: Domain,
    markers: Mapping[str, Mapping[int, str]],
    name: str,
) -> str | tuple[int, ...]:
    """The subdomain id of an integral of the forms, on its symbolic domain.

    An integer subdomain id of the forms, e.g. the ``1`` of ``ds(1)``, is
    named by ``markers`` and replaced by the domain's id for that marker name.
    The domain assigns an id to a marker name it does not have yet.

    Args:
        integral: An integral of the forms.
        domain: The symbolic domain of its mesh.
        markers: The marker name of each integer subdomain id, by measure.
        name: Name of the component, for the error message.

    Returns:
        ``"everywhere"`` or ``"otherwise"`` as they are, else the domain's ids
        of the marker names.

    Raises:
        ComponentError: If a subdomain id has no marker name.
    """
    subdomain_id = integral.subdomain_id()
    if subdomain_id in ("everywhere", "otherwise"):
        return subdomain_id
    measure = typing.cast(
        MeasureName,
        ufl.measure.integral_type_to_measure_name[integral.integral_type()],
    )
    marker_names = dict(markers.get(measure, {}))
    ids = subdomain_id if isinstance(subdomain_id, tuple) else (subdomain_id,)
    missing = [i for i in ids if i not in marker_names]
    if missing:
        raise ComponentError(
            f"{measure}{tuple(missing)} in {name!r} has no marker name. "
            f"Name it with markers={{{measure!r}: {{{missing[0]}: 'name'}}}}."
        )
    return domain.marker_ids(
        [marker_names[i] for i in ids], MEASURE_ENTITY_TYPE[measure]
    )


def _extract_terminals(
    form: ufl.Form,
) -> list[Argument | Coefficient | Constant | GeometricQuantity]:
    """The arguments, coefficients, constants and geometry of a form.

    In the order they first appear, unlike :func:`ufl.algorithms.extract_type`,
    which gives a set. The order decides the order in which parameters, known
    coefficients and domains are created, so it is the same on every process.
    """
    found: dict[Argument | Coefficient | Constant | GeometricQuantity, None] = {}
    for integral in form.integrals():
        for node in unique_pre_traversal(integral.integrand()):
            if isinstance(node, Argument | Coefficient | Constant | GeometricQuantity):
                found[node] = None
    return list(found)


def _update_meshes(
    form: ufl.Form, meshes: dict[ufl.Mesh, Domain], names_by_id: Mapping[int, str]
) -> None:
    """Add the symbolic domain of each new mesh of a form to the mesh map.

    The meshes are those of the form's terminals, in the order they first
    appear, and of its integrals, as an integrand need not have a terminal on
    its mesh, e.g. ``1.0 * dx``. Afterwards every mesh of the form has a
    domain in ``meshes``.

    Args:
        form: A form of the component.
        meshes: The domain of each mesh found so far. Modified in place.
        names_by_id: The variable names, see :func:`_get_or_create_domain`.
    """
    for terminal in _extract_terminals(form):
        mesh = typing.cast(ufl.Mesh, extract_unique_domain(terminal))
        _get_or_create_domain(mesh, meshes, names_by_id)
    for integral in form.integrals():
        _get_or_create_domain(integral.ufl_domain(), meshes, names_by_id)


def _get_or_create_domain(
    mesh: ufl.Mesh, meshes: dict[ufl.Mesh, Domain], names_by_id: Mapping[int, str]
) -> Domain:
    """The symbolic domain of a mesh of the forms, created on first use.

    A mesh without a domain in ``meshes`` gets a new one, which is added to
    ``meshes``, so every later use of the mesh gets the same domain.

    Args:
        mesh: A mesh of the forms.
        meshes: The domain of each mesh found so far. Modified in place.
        names_by_id: The variable names. A new domain is named after the
            mesh's variable, or ``"mesh<i>"``, with ``i`` the number of meshes
            found before it.

    Returns:
        The domain of the mesh.
    """
    if mesh not in meshes:
        name = names_by_id.get(id(mesh), f"mesh{len(meshes)}")
        meshes[mesh] = _create_domain(name, mesh)
    return meshes[mesh]


def _create_domain(name: str, mesh: ufl.Mesh) -> Domain:
    """A domain with the coordinate element of a UFL mesh."""
    element = mesh.ufl_coordinate_element()
    return Domain(
        name,
        typing.cast(typing.Any, element.cell_type.name),
        gdim=element.reference_value_shape[0],
        degree=element.embedded_superdegree,
        lagrange_variant=getattr(element, "lagrange_variant", None),
    )
