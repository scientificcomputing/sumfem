"""Dirichlet boundary conditions on fields, given by marker name."""

from __future__ import annotations

import dataclasses
import typing

import ufl

from .domains import EntityType
from .terminals import Field

if typing.TYPE_CHECKING:
    from ..discrete.values import Value

__all__ = ["DirichletBC"]


@dataclasses.dataclass(frozen=True)
class DirichletBC:
    """A Dirichlet condition on a field, on the entities of a named marker.

    The condition is symbolic: a
    :class:`~sumfem.discrete.discretization.Discretization` turns it
    into a {py:class}`dolfinx.fem.DirichletBC` on the mesh of the field's domain.

    Attributes:
        field: The field the condition applies to.
        marker: Name of the marker on the field's mesh, e.g. ``"Inlet"``, or
            several names.
        value: The prescribed value. A UFL expression written with
            parameters, fields and the coordinates of the domain is evaluated
            at runtime, before every solve and every Newton iteration, so it
            follows time and other changing values. It is evaluated in SI
            base units. Any other
            :data:`~sumfem.discrete.values.Value` is evaluated once,
            in the field's unit.
        entity_type: The entities the marker refers to, facets by default.
    """

    field: Field
    marker: str | tuple[str, ...]
    value: Value | ufl.core.expr.Expr
    entity_type: EntityType = "facet"

    @property
    def markers(self) -> tuple[str, ...]:
        """The marker names, as a tuple."""
        return (self.marker,) if isinstance(self.marker, str) else tuple(self.marker)
