"""The discrete state of symbolic components on the meshes of domain maps.

:func:`discretize` gives a :class:`Discretization` of components on
:class:`~sumfem.discrete.domain_maps.DomainMaps`. It owns one DOLFINx
function for every field, parameter, known coefficient and previous value of
its components, shared by all of them, so a field one system solves for is
up to date as data in the next, without copying. Indexing it gives the
discrete counterpart of a symbolic object: ``disc[residual]`` the discrete
system, ``disc[field]`` the function.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Iterable, Mapping

from mpi4py import MPI

import basix
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import sympy
import ufl
from ufl.algorithms import extract_coefficients
from ufl.domain import extract_domains
from ufl.finiteelement import AbstractFiniteElement

from ..error_handling import SystemBuildError, qualified_name
from ..symbolic.bcs import DirichletBC
from ..symbolic.components import Functional, Jacobian, Residual
from ..symbolic.coordinates import lift
from ..symbolic.coupling import Sequential
from ..symbolic.domains import (
    Domain,
    domain_of,
)
from ..symbolic.expressions import FormOrZero, evaluate_constant
from ..symbolic.si import has_units, si_factor, to_si
from ..symbolic.system import System
from ..symbolic.terminals import (
    Field,
    FieldRole,
    FieldTerminal,
    KnownCoefficient,
    Parameter,
    TimeDerivative,
    _validate_value,
    field_of,
)
from ..symbolic.time import Clock, coefficients_through_time_derivatives
from ..symbolic.transfer import Transfer
from . import expressions  # noqa: F401  (registers expressions)
from .counterpart import counterpart
from .domain_maps import DomainMaps
from .values import Value, assign

__all__ = ["Discretization", "discretize"]


class Discretization:
    """The discrete state of symbolic components on domain maps.

    Made by :func:`discretize`. It holds what its components need: the
    submesh of each subdomain they use, in the maps, and a function for each
    field, parameter, known coefficient and previous value in them. These are
    created when the components are declared, by :func:`discretize` or
    :meth:`add`. A terminal not held raises, so nothing is created behind a
    lookup.

    Indexing gives discrete counterparts, compiled on first access and kept:
    see :meth:`__getitem__`. Every system of a discretization shares its
    functions and compiler options.

    Declaring is collective: every process must declare the same components
    in the same order, since creating a submesh is.

    Args:
        maps: The meshes of the domains.
        max_quadrature_degree: Upper bound on the quadrature degree UFL
            estimates for each integral without an explicit degree, through
            FFCx's ``max_quadrature_degree`` metadata. Use it to keep
            nonlinear integrands cheap. ``None`` for no bound.
        form_compiler_options: Options for FFCx.
        jit_options: Options for the JIT compilation of the generated code.

    Raises:
        ValueError: If the bound is negative.
    """

    _maps: DomainMaps
    """The meshes of the domains."""

    _max_quadrature_degree: int | None
    """Upper bound on estimated quadrature degrees, or ``None``."""

    _form_compiler_options: dict
    """Options for FFCx."""

    _jit_options: dict
    """Options for the JIT compilation."""

    _spaces: dict[tuple[str, AbstractFiniteElement], dolfinx.fem.FunctionSpace]
    """Function spaces shared by known coefficients and parameters, by
    domain name and element."""

    _field_spaces: dict[Field, dolfinx.fem.FunctionSpace]
    """The function space of each field. Every field gets its own space, so
    that boundary conditions are assigned to the right block."""

    _functions: dict[ufl.Coefficient, dolfinx.fem.Function]
    """The function of each symbolic terminal."""

    _unset: set[Parameter]
    """Parameters without a value."""

    _counterparts: dict[int, tuple[typing.Any, typing.Any]]
    """Discrete counterparts by the id of their symbolic object, which is
    kept alive here."""

    def __init__(
        self,
        maps: DomainMaps,
        max_quadrature_degree: int | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        if max_quadrature_degree is not None and max_quadrature_degree < 0:
            raise ValueError(
                "max_quadrature_degree must be non-negative, "
                f"got {max_quadrature_degree}."
            )
        self._maps = maps
        self._max_quadrature_degree = max_quadrature_degree
        self._form_compiler_options = dict(form_compiler_options or {})
        self._jit_options = dict(jit_options or {})
        self._spaces = {}
        self._field_spaces = {}
        self._functions = {}
        self._unset = set()
        self._counterparts = {}

    def add(self, *components: Declarable) -> None:
        """Declare components, creating the submeshes and functions they need.

        A component is a residual, functional or Jacobian, a symbolic
        system, a Dirichlet condition, a transfer, a sequence, a
        clock, a form, an expression, a single terminal, or a domain, e.g. a
        subdomain whose submesh is needed. A system holds the
        terminals a time-stepping scheme introduced, such as previous values
        and weights, which its residual does not. A parameter starts at its
        default value. Terminals and submeshes held already are kept.

        Raises:
            KeyError: If there is no mesh for a domain they use.
            TypeError: For something that is not a component.
        """
        for component in components:
            coefficients, domains = _needs(component)
            for domain in domains:
                self._maps.ensure_mesh(domain)
            for terminal in coefficients:
                if terminal not in self._functions:
                    self._functions[terminal] = self._create(terminal)

    @property
    def maps(self) -> DomainMaps:
        """The meshes of the domains."""
        return self._maps

    @property
    def max_quadrature_degree(self) -> int | None:
        """Upper bound on estimated quadrature degrees, or ``None``."""
        return self._max_quadrature_degree

    @property
    def form_compiler_options(self) -> dict:
        """Options for FFCx."""
        return dict(self._form_compiler_options)

    @property
    def jit_options(self) -> dict:
        """Options for the JIT compilation of the generated code."""
        return dict(self._jit_options)

    @property
    def comm(self) -> MPI.Intracomm:
        """The MPI communicator of the meshes."""
        return self._maps.comm

    # Functions

    def _shared_space(
        self, domain: Domain, element: AbstractFiniteElement
    ) -> dolfinx.fem.FunctionSpace:
        key = (domain.name, element)
        space = self._spaces.get(key)
        if space is None:
            mesh = self._maps.mesh(domain)
            # On a manifold, the basis of a Piola-mapped element, e.g.
            # Raviart-Thomas, depends on the signs of the facets
            piola = {
                basix.MapType.contravariantPiola,
                basix.MapType.covariantPiola,
                basix.MapType.doubleContravariantPiola,
                basix.MapType.doubleCovariantPiola,
            }
            if mesh.geometry.dim > mesh.topology.dim and _map_types(element) & piola:
                mesh.topology.create_facet_orientations()
            space = dolfinx.fem.functionspace(mesh, element)
            self._spaces[key] = space
        return space

    def function_space(self, field: Field) -> dolfinx.fem.FunctionSpace:
        """The function space of a field, distinct from that of any other field."""
        space = self._field_spaces.get(field)
        if space is None:
            # Clones share the dofmap, but have their own identity
            space = self._shared_space(field.domain, field.ufl_element()).clone()
            self._field_spaces[field] = space
        return space

    def function(self, terminal: ufl.Coefficient) -> dolfinx.fem.Function:
        """The function of a field, parameter, known coefficient or history value.

        Raises:
            SystemBuildError: For the time-derivative placeholder
                ``field.dot``, which a time-stepping scheme must replace.
            KeyError: If the discretization does not hold the terminal.
        """
        function = self._functions.get(terminal)
        if function is None and _is_time_derivative(terminal):
            raise SystemBuildError(
                f"{getattr(terminal, 'name', terminal)!r} is a placeholder for a "
                "time derivative. Replace it with a time-stepping scheme before "
                "discretizing."
            )
        if function is None:
            raise KeyError(
                f"{getattr(terminal, 'name', terminal)!r} is not held by this "
                "discretization. Declare a component that uses it, in "
                "Discretization(meshes, components) or with add()."
            )
        return function

    def _create(self, terminal: ufl.Coefficient) -> dolfinx.fem.Function:
        """The function of a terminal. A parameter starts at its default value.

        Raises:
            SystemBuildError: For the time-derivative placeholder
                ``field.dot``, which a time-stepping scheme must replace.
            TypeError: For anything else.
        """
        if isinstance(terminal, Field):
            function = dolfinx.fem.Function(
                self.function_space(terminal), name=terminal.name
            )
        elif isinstance(terminal, FieldTerminal):
            if terminal.role is FieldRole.DOT:
                raise SystemBuildError(
                    f"{terminal.name!r} is a placeholder for a time derivative. "
                    "Replace it with a time-stepping scheme before discretizing."
                )
            function = dolfinx.fem.Function(
                self.function_space(terminal.field), name=terminal.name
            )
        elif isinstance(terminal, TimeDerivative):
            raise SystemBuildError(
                f"{terminal.name!r} is a placeholder for a time derivative. "
                "Replace it with a time-stepping scheme before discretizing."
            )
        elif isinstance(terminal, Parameter | KnownCoefficient):
            space = self._shared_space(
                domain_of(terminal.ufl_function_space()), terminal.ufl_element()
            )
            function = dolfinx.fem.Function(space, name=terminal.name)
            if isinstance(terminal, Parameter):
                if terminal.default is None:
                    self._unset.add(terminal)
                else:
                    assign(function, to_si(terminal.default, terminal.unit))
        else:
            raise TypeError(
                f"Cannot discretize {terminal!r}. Expected a {qualified_name(Field)}, "
                f"{qualified_name(Parameter)} or {qualified_name(KnownCoefficient)}."
            )
        return function

    def _by_name(
        self, kind: type[Field | Parameter | KnownCoefficient]
    ) -> Mapping[str, dolfinx.fem.Function]:
        return types.MappingProxyType(
            {
                typing.cast(Field | Parameter | KnownCoefficient, t).name: f
                for t, f in self._functions.items()
                if isinstance(t, kind)
            }
        )

    @property
    def fields(self) -> Mapping[str, dolfinx.fem.Function]:
        """The function of each field held, by field name."""
        return self._by_name(Field)

    @property
    def parameters(self) -> Mapping[str, dolfinx.fem.Function]:
        """The function of each parameter held, by parameter name."""
        return self._by_name(Parameter)

    @property
    def known_coefficients(self) -> Mapping[str, dolfinx.fem.Function]:
        """The function of each known coefficient held, by name."""
        return self._by_name(KnownCoefficient)

    def terminal(self, key: str | ufl.Coefficient) -> ufl.Coefficient:
        """The symbolic terminal with a name, among those held.

        A terminal given as an object is returned as it is.

        Raises:
            KeyError: If no terminal held has the name.
        """
        if not isinstance(key, str):
            return key
        matches = [t for t in self._functions if getattr(t, "name", None) == key]
        if len(matches) != 1:
            raise KeyError(
                f"No field, parameter or known coefficient named {key!r} is "
                "held by this discretization."
            )
        return matches[0]

    def __contains__(self, terminal: object) -> bool:
        """Whether the discretization holds a terminal."""
        return terminal in self._functions

    def __getitem__(self, key: typing.Any) -> typing.Any:
        """The discrete counterpart of a symbolic object.

        A terminal, given as the object or by name, gives its function. Any
        other object gives its :func:`~sumfem.discrete.counterpart.counterpart`
        with default options, compiled on first access and kept: a residual
        or a symbolic system its discrete
        :class:`~sumfem.discrete.system.System`, a Dirichlet condition its
        discrete condition, a transfer its discrete transfer, a form or
        functional its compiled form, an expression its expression in the
        discrete functions. The object need not be the one declared, but the
        terminals it uses must be held.
        """
        if isinstance(key, str) or (
            isinstance(key, ufl.Coefficient) and key in self._functions
        ):
            return self.function(self.terminal(key))
        if isinstance(key, Functional):
            key = key.form
        entry = self._counterparts.get(id(key))
        if entry is None:
            entry = (key, counterpart(key, self))
            self._counterparts[id(key)] = entry
        return entry[1]

    def set(
        self,
        values: Mapping[str | ufl.Coefficient, Value | sympy.Expr | ufl.core.expr.Expr],
    ) -> None:
        """Set the values of fields, parameters or known coefficients.

        Values are given in the unit of the terminal and stored in SI base
        units. A sympy quantity with its own units, e.g.
        ``5 * sympy.physics.units.centimeter``, is converted from those.

        A symbolic UFL expression, of fields, parameters, known coefficients
        and spatial coordinates, sets a field or known coefficient by
        interpolation, e.g. an initial condition. It evaluates in SI base
        units, as every symbolic expression does.

        Args:
            values: Values by terminal, given as the object or by name. See
                :data:`~sumfem.discrete.values.Value` for the accepted values.
                Parameters take numbers or arrays of their shape.

        Raises:
            ValueError: If a sympy quantity has the wrong dimension, or an
                expression the wrong shape.
            TypeError: For an expression given to a parameter.
        """
        for key, value in values.items():
            terminal = self.terminal(key)
            function = self.function(terminal)
            unit = _unit_of(terminal)
            if _is_symbolic(value):
                if isinstance(terminal, Parameter):
                    raise TypeError(
                        f"Parameter {terminal.name!r} takes a number or an array, "
                        "not an expression."
                    )
                expr = typing.cast(ufl.core.expr.Expr, value)
                if extract_domains(expr):
                    expr_ = self.compile_expression(expr, function.function_space)
                    assign(function, expr_)
                else:
                    assign(function, evaluate_constant(expr), si_factor(unit))
            elif has_units(value):
                assign(function, to_si(typing.cast(sympy.Expr, value), unit))
            elif isinstance(terminal, Parameter):
                array = _validate_value(
                    terminal.name, typing.cast(npt.ArrayLike, value), terminal.shape
                )
                assign(function, to_si(array, unit))
            else:
                assign(function, typing.cast(Value, value), si_factor(unit))
            if isinstance(terminal, Parameter):
                self._unset.discard(terminal)

    def compile_expression(
        self, expr: ufl.core.expr.Expr, space: dolfinx.fem.FunctionSpace
    ) -> dolfinx.fem.Expression:
        """Compile a symbolic expression for interpolation into a space.

        The expression is discretized with
        :func:`~sumfem.discrete.discretize.discretize` and compiled
        at the interpolation points of the space, with this
        discretization's compiler options and entity maps. On a domain whose
        coordinate system writes vectors in 3D, see
        :mod:`~sumfem.symbolic.coordinates`, the expression is evaluated in
        3D, and a space of the mesh's dimension receives its components in
        the mesh's coordinates, e.g. those of a stress but its hoop
        component.

        Raises:
            ValueError: If the expression's shape is not the space's.
        """
        domains = extract_domains(expr)
        domain = domains[0] if len(domains) == 1 else None
        if isinstance(domain, Domain):
            expr = _fitted(
                lift(domain.coordinates, expr, ufl.SpatialCoordinate(domain)),
                space.value_shape,
            )
        if expr.ufl_shape != space.value_shape:
            raise ValueError(
                f"An expression of shape {expr.ufl_shape} cannot fill a space of "
                f"shape {space.value_shape}."
            )
        return dolfinx.fem.Expression(
            counterpart(expr, self),
            space.element.interpolation_points,
            comm=space.mesh.comm,
            form_compiler_options=self.form_compiler_options,
            jit_options=self.jit_options,
            entity_maps=self._maps.entity_maps or None,
        )

    def value(self, parameter: Parameter | str) -> npt.NDArray[typing.Any]:
        """The current value of a parameter, in the parameter's unit."""
        terminal = self.terminal(parameter)
        if not isinstance(terminal, Parameter):
            raise TypeError(f"{terminal!r} is not a {qualified_name(Parameter)}.")
        array = self.function(terminal).x.array
        si = array[: max(1, int(np.prod(terminal.shape)))].reshape(terminal.shape)
        return si / si_factor(terminal.unit)

    def has_value(self, parameter: Parameter | str) -> bool:
        """Whether a parameter has a value, from its default or set explicitly."""
        terminal = self.terminal(parameter)
        self.function(terminal)
        return terminal not in self._unset

    def check_values(self, system: System) -> None:
        """Raise if a parameter of a system has no value.

        Raises:
            ValueError: Naming the parameters without a value.
        """
        missing = sorted(p.name for p in system.parameters.values() if p in self._unset)
        if missing:
            raise ValueError(
                f"The parameters {missing} have no value. Set them with "
                "Discretization.set()."
            )

    # History of fields for time stepping

    def history(self, field: Field) -> dict[int, dolfinx.fem.Function]:
        """The functions of the field's previous values held, by ``k``."""
        return {
            t.k: f
            for t, f in self._functions.items()
            if isinstance(t, FieldTerminal) and t.field is field
        }

    def initialize_history(self) -> None:
        """Set every previous value of every field to the field's current value."""
        for terminal, function in self._functions.items():
            if isinstance(terminal, FieldTerminal):
                function.x.array[:] = self.function(terminal.field).x.array

    def advance_history(self) -> None:
        """Shift the previous values of every field one step back in time.

        Afterwards ``previous(1)`` holds the current value of the field.
        """
        fields = {
            t.field
            for t in self._functions
            if isinstance(t, FieldTerminal) and t.role is FieldRole.PREVIOUS
        }
        for field in fields:
            history = self.history(field)
            for k in sorted(history, reverse=True):
                source = history.get(k - 1) if k > 1 else self.function(field)
                if source is not None:
                    history[k].x.array[:] = source.x.array

    # Functionals

    def assemble_scalar(self, form: ufl.Form | Functional) -> float:
        """Assemble a symbolic functional, summed over all processes.

        The functional is compiled on first use and kept, so assembling it
        again, e.g. at every time step, uses the current values of its
        terminals without compiling it again.

        Args:
            form: A form without test or trial functions over one domain,
                e.g. ``c * domain.ds("Membrane")``, or a functional.
        """
        local = dolfinx.fem.assemble_scalar(self[form])
        return float(self.comm.allreduce(local))


def _fitted(expr: ufl.core.expr.Expr, shape: tuple[int, ...]) -> ufl.core.expr.Expr:
    """An expression written in 3D, cut to the leading components of a shape.

    Expressions of another rank, or smaller than the shape, are kept.
    """
    if expr.ufl_shape == shape or len(expr.ufl_shape) != len(shape):
        return expr
    if any(n > m for n, m in zip(shape, expr.ufl_shape, strict=True)):
        return expr
    if not shape:
        return expr
    if len(shape) == 1:
        return ufl.as_vector([expr[i] for i in range(shape[0])])
    return ufl.as_matrix(
        [[expr[i, j] for j in range(shape[1])] for i in range(shape[0])]
    )


def discretize(
    components: Declarable | typing.Sequence[Declarable],
    maps: DomainMaps,
    max_quadrature_degree: int | None = None,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> Discretization:
    """The discretization of components on domain maps.

    Components that are discretized together share their functions, so
    declare every component that couples to another here, e.g. a flow and
    the transport it carries. :meth:`Discretization.add` declares later
    additions.

    Args:
        components: A component, or a list or tuple of them, see
            :meth:`Discretization.add`.
        maps: The meshes of the domains.
        max_quadrature_degree: Upper bound on estimated quadrature degrees,
            see :class:`Discretization`.
        form_compiler_options: Options for FFCx.
        jit_options: Options for the JIT compilation of the generated code.

    Example:
        A flow, and the half-cell it feeds::

            disc = discretize([flow, cell.model, *inlet], maps)
            Solver(disc[flow], [outlet]).solve()
            Simulation(disc, cell.model, inlet, clock=clock, scheme=BDF(1)).run()
    """
    disc = Discretization(
        maps, max_quadrature_degree, form_compiler_options, jit_options
    )
    disc.add(*(components if isinstance(components, list | tuple) else [components]))
    return disc


Declarable = typing.Union[
    Domain,
    Residual,
    Functional,
    Jacobian,
    System,
    DirichletBC,
    Transfer,
    Sequential,
    Clock,
    ufl.Form,
    ufl.core.expr.Expr,
]
"""What a discretization can be declared with, see :meth:`Discretization.add`."""


def _map_types(element: AbstractFiniteElement) -> set[basix.MapType]:
    """The map types of an element, or of the sub-elements of a mixed one."""
    if getattr(element, "is_mixed", False):
        return set().union(*(_map_types(e) for e in element.sub_elements))
    return {getattr(element, "map_type", basix.MapType.identity)}


def _needs(component: Declarable) -> tuple[list[ufl.Coefficient], list[Domain]]:
    """The terminals and domains a component needs, in a deterministic order."""
    if isinstance(component, Domain):
        return [], [component]
    if isinstance(component, Sequential):
        return _merge([_needs(stage) for stage in component.stages])
    if isinstance(component, Residual):
        terms = [
            t
            for m in component.components
            for t in (m.jacobian_term, m.preconditioner_term)
            if t is not None
        ]
        forms = [m.form for m in component.components] + [t.form for t in terms]
        return _merge([_form_needs(f) for f in forms] + [([], list(component.domains))])
    if isinstance(component, Functional | Jacobian):
        return _merge(
            [_form_needs(m.form) for m in component.components]
            + [([], list(component.domains))]
        )
    if isinstance(component, System):
        system_forms: list[ufl.Form] = []
        for blocks in (
            component.residual,
            component.jacobian,
            component.preconditioner,
        ):
            _collect_forms(blocks, system_forms)
        return _merge(
            [_form_needs(f) for f in system_forms]
            + [(list(component.unknowns), list(component.domains))]
        )
    if isinstance(component, DirichletBC):
        needs: list[tuple[list[ufl.Coefficient], list[Domain]]] = [
            ([component.field], [component.field.domain])
        ]
        if isinstance(component.value, ufl.core.expr.Expr):
            needs.append(_expr_needs(component.value))
        return _merge(needs)
    if isinstance(component, Transfer):
        return _merge([_expr_needs(component.source), _expr_needs(component.target)])
    if isinstance(component, Clock):
        return _merge([_expr_needs(component.t), _expr_needs(component.dt)])
    if isinstance(component, ufl.Form):
        return _form_needs(component)
    if isinstance(component, ufl.core.expr.Expr):
        return _expr_needs(component)
    raise TypeError(
        f"Cannot declare {component!r} of type {type(component).__name__} on a "
        "discretization."
    )


def _form_needs(form: FormOrZero) -> tuple[list[ufl.Coefficient], list[Domain]]:
    coefficients = [_holdable(c) for c in coefficients_through_time_derivatives(form)]
    fields = [field_of(a) for a in form.arguments()]
    terminals = coefficients + [f for f in fields if f is not None]
    domains = [d for d in form.ufl_domains() if isinstance(d, Domain)]
    return _merge([(terminals, domains), *(_expr_needs(t) for t in terminals)])


def _expr_needs(expr: ufl.core.expr.Expr) -> tuple[list[ufl.Coefficient], list[Domain]]:
    terminals = [_holdable(c) for c in extract_coefficients(expr)]
    domains = [domain_of(t.ufl_function_space()) for t in terminals]
    domains += [d for d in extract_domains(expr) if isinstance(d, Domain)]
    return terminals, domains


def _is_time_derivative(terminal: ufl.Coefficient) -> bool:
    return isinstance(terminal, TimeDerivative) or (
        isinstance(terminal, FieldTerminal) and terminal.role is FieldRole.DOT
    )


def _holdable(coefficient: ufl.Coefficient) -> ufl.Coefficient:
    """A coefficient, or the field of a time-derivative placeholder."""
    if isinstance(coefficient, FieldTerminal) and coefficient.role is FieldRole.DOT:
        return coefficient.field
    return coefficient


def _collect_forms(blocks: typing.Any, forms: list[ufl.Form]) -> None:
    if isinstance(blocks, ufl.Form):
        forms.append(blocks)
    elif isinstance(blocks, list | tuple):
        for block in blocks:
            _collect_forms(block, forms)


def _merge(
    needs: Iterable[tuple[list[ufl.Coefficient], list[Domain]]],
) -> tuple[list[ufl.Coefficient], list[Domain]]:
    terminals: dict[ufl.Coefficient, None] = {}
    domains: dict[str, Domain] = {}
    for coefficients, ds in needs:
        terminals.update(dict.fromkeys(coefficients))
        for d in ds:
            domains.setdefault(d.name, d)
    return list(terminals), list(domains.values())


def _unit_of(terminal: ufl.Coefficient) -> sympy.Expr:
    """The unit of a field, parameter, known coefficient or history value."""
    if isinstance(terminal, FieldTerminal):
        return terminal.field.unit
    if isinstance(terminal, Field | Parameter | KnownCoefficient):
        return terminal.unit
    raise TypeError(f"{terminal!r} has no unit.")


def _is_symbolic(value: typing.Any) -> bool:
    """Whether a value is a symbolic UFL expression rather than data."""
    return isinstance(value, ufl.core.expr.Expr) and not isinstance(
        value, dolfinx.fem.Function
    )
