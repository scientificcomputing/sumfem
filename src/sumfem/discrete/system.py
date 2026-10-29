"""Systems of equations discretized on meshes.

The discrete :class:`System` has the name of its symbolic counterpart,
:class:`sumfem.symbolic.system.System`, which it discretizes.
"""

from __future__ import annotations

import typing
import weakref
from collections.abc import Callable

import dolfinx.fem
import numpy as np
import numpy.typing as npt
import ufl

from ..error_handling import SystemBuildError, qualified_name
from ..symbolic import system as symbolic
from ..symbolic.components import Functional, Residual
from ..symbolic.coordinates import apply_coordinates
from ..symbolic.domains import (
    MEASURE_ENTITY_TYPE,
    Domain,
    MeasureName,
    place_subdomain_integrals,
)
from .counterpart import counterpart
from .meshes import INTEGRAL_TYPES

if typing.TYPE_CHECKING:
    from .discretization import Discretization

__all__ = ["System"]

# Compiled forms of a system, per compiler options and then by the id of the
# UFL form (the system keeps its forms alive). Compiling does not depend on
# the mesh, so a system is compiled once and can be bound to any mesh.
_compiled_systems: weakref.WeakKeyDictionary[
    symbolic.System, dict[str, dict[int, typing.Any]]
] = weakref.WeakKeyDictionary()


class System:
    """The residual, Jacobian and preconditioner of a symbolic system, compiled
    and bound to the meshes and functions of a discretization.

    The forms are evaluated at the current values of the functions, so a
    :class:`~sumfem.solvers.Solver` assembles them again at every Newton
    iteration. A linear system is the case where one iteration suffices.
    The forms are compiled with the discretization's compiler options, and
    cached per symbolic system and options.

    Args:
        discretization: The discretization holding the system's terminals.
        system: The symbolic system.
    """

    _symbolic: symbolic.System
    """The symbolic system."""

    _discretization: Discretization
    """The discretization owning the meshes and functions."""

    _residual: dolfinx.fem.Form | list[dolfinx.fem.Form]
    """Residual form, or one form per unknown."""

    _jacobian: dolfinx.fem.Form | list[list[dolfinx.fem.Form | None]]
    """Jacobian form, or blocks with ``None`` for zero blocks."""

    _preconditioner: dolfinx.fem.Form | list[list[dolfinx.fem.Form | None]] | None
    """Preconditioner forms, or ``None``."""

    _solution: tuple[dolfinx.fem.Function, ...]
    """The function of each unknown, in block order."""

    def __init__(self, discretization: Discretization, system: symbolic.System):
        compiled = _compile(discretization, system)
        coefficient_map = {
            c: discretization.function(c)
            for form in _forms(system)
            for c in form.coefficients()
        }

        def bind(form: ufl.Form) -> dolfinx.fem.Form:
            return _create_form(
                discretization, system, compiled[id(form)], form, coefficient_map
            )

        self._symbolic = system
        self._discretization = discretization
        self._residual = _map_blocks(bind, system.residual)
        self._jacobian = _map_blocks(bind, system.jacobian)
        self._preconditioner = _map_blocks(bind, system.preconditioner)
        self._solution = tuple(
            discretization.function(field) for field in system.unknowns
        )

    @property
    def symbolic(self) -> symbolic.System:
        """The symbolic system."""
        return self._symbolic

    @property
    def discretization(self) -> Discretization:
        """The discretization owning the meshes and functions."""
        return self._discretization

    @property
    def residual(self) -> dolfinx.fem.Form | list[dolfinx.fem.Form]:
        """Residual form, or one form per unknown."""
        return self._residual

    @property
    def jacobian(self) -> dolfinx.fem.Form | list[list[dolfinx.fem.Form | None]]:
        """Jacobian form, or blocks with ``None`` for zero blocks."""
        return self._jacobian

    @property
    def preconditioner(
        self,
    ) -> dolfinx.fem.Form | list[list[dolfinx.fem.Form | None]] | None:
        """Preconditioner forms, or ``None``."""
        return self._preconditioner

    @property
    def solution(self) -> tuple[dolfinx.fem.Function, ...]:
        """The function of each unknown, in block order."""
        return self._solution


@counterpart.register
def _(
    system: symbolic.System, discretization: Discretization, **options: typing.Any
) -> System:
    if options:
        raise ValueError(
            f"The options {sorted(options)} are those of build_system, which "
            "need a residual, not a system."
        )
    return System(discretization, system)


@counterpart.register
def _(
    residual: Residual, discretization: Discretization, **options: typing.Any
) -> System:
    """A residual's system, built with the options of ``build_system``, e.g.
    ``field_order`` or ``check_units``."""
    return System(discretization, symbolic.build_system(residual, **options))


@counterpart.register
def _(form: ufl.Form, discretization: Discretization) -> dolfinx.fem.Form:
    """A symbolic functional, compiled and discretized on the meshes.

    Each integral is weighted by the coordinate system of its domain.

    Args:
        discretization: The discretization holding the form's terminals.
        form: A form without test or trial functions, integrated over one
            domain.

    Raises:
        ValueError: If the form has arguments.
        SystemBuildError: If the form is integrated over several domains.
    """
    if form.arguments():
        raise ValueError("A functional cannot contain test or trial functions.")
    form = place_subdomain_integrals(apply_coordinates(form))
    compiled = dolfinx.fem.compile_form(
        discretization.comm,
        _bound_quadrature_degree(form, discretization.max_quadrature_degree),
        form_compiler_options=discretization.form_compiler_options,
        jit_options=discretization.jit_options,
    )
    domain = _form_domain(form)
    return dolfinx.fem.create_form(
        compiled,
        [],
        discretization.maps.mesh(domain),
        _subdomains(discretization, domain, form),
        {c: discretization.function(c) for c in form.coefficients()},
        {},
        discretization.maps.entity_maps,
    )


@counterpart.register
def _(
    functional: Functional,
    discretization: Discretization,
    **options: typing.Any,
) -> dolfinx.fem.Form:
    """A functional's form, compiled and discretized on the meshes."""
    return counterpart(functional.form, discretization, **options)


def _form_domain(form: ufl.Form) -> Domain:
    """The single domain a form is integrated over.

    Raises:
        SystemBuildError: If the form is integrated over several domains, or
            over a mesh that is not a domain.
    """
    domains = {integral.ufl_domain() for integral in form.integrals()}
    if len(domains) != 1:
        raise SystemBuildError(
            "Each block of a system must be integrated over a single domain, "
            f"got {sorted(str(d) for d in domains)}."
        )
    (domain,) = domains
    if not isinstance(domain, Domain):
        raise SystemBuildError(f"{domain!r} is not a {qualified_name(Domain)}.")
    return domain


def _compile(
    discretization: Discretization, system: symbolic.System
) -> dict[int, typing.Any]:
    """The compiled form of every form of a system, by the id of the form."""
    options = discretization.form_compiler_options
    jit_options = discretization.jit_options
    max_degree = discretization.max_quadrature_degree
    key = repr((sorted(options.items()), sorted(jit_options.items()), max_degree))
    per_options = _compiled_systems.setdefault(system, {})
    compiled = per_options.get(key)
    if compiled is None:
        compiled = {
            id(form): dolfinx.fem.compile_form(
                discretization.comm,
                _bound_quadrature_degree(form, max_degree),
                form_compiler_options=options,
                jit_options=jit_options,
            )
            for form in _forms(system)
        }
        per_options[key] = compiled
    return compiled


def _bound_quadrature_degree(form: ufl.Form, max_degree: int | None) -> ufl.Form:
    """Bound the estimated quadrature degree of the integrals of a form.

    Integrals with an explicit quadrature degree keep it, since FFCx rejects
    a bound together with a degree. The integrals keep their order and
    subdomains, so the compiled form matches the original one.
    """
    if max_degree is None:
        return form
    return ufl.Form(
        [
            integral
            if "quadrature_degree" in integral.metadata()
            else integral.reconstruct(
                metadata={**integral.metadata(), "max_quadrature_degree": max_degree}
            )
            for integral in form.integrals()
        ]
    )


def _create_form(
    discretization: Discretization,
    system: symbolic.System,
    compiled_form: typing.Any,
    ufl_form: ufl.Form,
    coefficient_map: dict[ufl.Coefficient, dolfinx.fem.Function],
) -> dolfinx.fem.Form:
    domain = _form_domain(ufl_form)
    arguments = sorted(ufl_form.arguments(), key=lambda a: a.number())
    spaces = [
        discretization.function_space(system.unknowns[a.part() or 0]) for a in arguments
    ]
    return dolfinx.fem.create_form(
        compiled_form,
        spaces,
        discretization.maps.mesh(domain),
        _subdomains(discretization, domain, ufl_form),
        coefficient_map,
        {},
        discretization.maps.entity_maps,
    )


def _subdomains(
    discretization: Discretization, domain: Domain, form: ufl.Form
) -> dict[dolfinx.fem.IntegralType, list[tuple[int, npt.NDArray[np.int32]]]]:
    """Integration entities of the marked integrals of a form."""
    names = {
        dim: {i: name for name, i in ids.items()} for dim, ids in domain.markers.items()
    }
    subdomains: dict[
        dolfinx.fem.IntegralType, list[tuple[int, npt.NDArray[np.int32]]]
    ] = {}
    for integral in form.integrals():
        subdomain_id = integral.subdomain_id()
        if subdomain_id in ("everywhere", "otherwise"):
            if integral.integral_type() == "interior_facet":
                discretization.maps.check_interior_facets(domain, None)
            continue
        integral_type = integral.integral_type()
        measure = typing.cast(
            MeasureName, ufl.measure.integral_type_to_measure_name[integral_type]
        )
        dim = domain.entity_dimension(MEASURE_ENTITY_TYPE[measure])
        ids = subdomain_id if isinstance(subdomain_id, tuple) else (subdomain_id,)
        entries = subdomains.setdefault(INTEGRAL_TYPES[integral_type], [])
        for i in ids:
            if any(i == j for j, _ in entries):
                continue
            entities = discretization.maps.integration_entities(
                domain, integral_type, names[dim][i]
            )
            entries.append((i, entities))
    return subdomains


def _map_blocks(
    function: Callable[[ufl.Form], typing.Any], blocks: typing.Any
) -> typing.Any:
    """Apply a function to every form in a block structure, keeping the structure.

    A block structure is a form, a sequence of block structures, or ``None``
    for a zero block.
    """
    if blocks is None:
        return None
    if isinstance(blocks, ufl.Form):
        return function(blocks)
    return [_map_blocks(function, block) for block in blocks]


def _forms(system: symbolic.System) -> list[ufl.Form]:
    """Every residual, Jacobian and preconditioner form of a system."""
    forms: list[ufl.Form] = []
    for blocks in (system.residual, system.jacobian, system.preconditioner):
        _map_blocks(forms.append, blocks)
    return forms
