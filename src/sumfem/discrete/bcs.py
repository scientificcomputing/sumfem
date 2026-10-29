"""Dirichlet boundary conditions discretized on meshes.

The discrete :class:`DirichletBC` has the name of its symbolic counterpart,
:class:`sumfem.symbolic.bcs.DirichletBC`, which it discretizes.
"""

from __future__ import annotations

import typing

import dolfinx.fem
import numpy as np
import ufl
from ufl.domain import extract_domains

from ..symbolic import bcs as symbolic
from ..symbolic.expressions import evaluate_constant
from ..symbolic.si import si_factor
from .counterpart import counterpart
from .values import Evaluation, assign

if typing.TYPE_CHECKING:
    from .discretization import Discretization

__all__ = ["DirichletBC"]


class DirichletBC:
    """A :class:`~sumfem.symbolic.bcs.DirichletBC` discretized on its field's mesh.

    A UFL value is compiled into an expression, evaluated by :meth:`pack`.
    Any other value is assigned once.

    Args:
        discretization: The discretization of the field.
        condition: The symbolic condition.
    """

    _symbolic: symbolic.DirichletBC
    """The symbolic condition."""

    _function: dolfinx.fem.Function
    """The function holding the prescribed values."""

    _dolfinx_bc: dolfinx.fem.DirichletBC
    """The DOLFINx boundary condition."""

    _evaluation: Evaluation | None
    """The evaluation of a symbolic value, or ``None`` for a value assigned
    once."""

    def __init__(self, discretization: Discretization, condition: symbolic.DirichletBC):
        # Create a discrete function for the prescribed values
        field = condition.field
        markers = ",".join(condition.markers)
        space = discretization.function_space(field)
        function = dolfinx.fem.Function(space, name=f"{field.name}[{markers}]")

        # Extract the entities to apply the Dirichlet condition to
        domain = field.domain
        marked = discretization.maps.mesh_data(domain)
        dim = domain.entity_dimension(condition.entity_type)
        marked.mesh.topology.create_connectivity(dim, domain.tdim)
        entities = np.unique(
            np.concatenate([marked.entities(dim, m) for m in condition.markers])
        ).astype(np.int32)

        # Extract the degrees of freedom to apply the Dirichlet condition to
        dofs = dolfinx.fem.locate_dofs_topological(space, dim, entities)

        # Assign the prescribed values to the function, either directly
        # or by adding an evaluation so we can pack/update it in assembly
        value = condition.value
        evaluation = None
        if isinstance(value, ufl.core.expr.Expr) and not extract_domains(value):
            # A constant expression, in SI units like every symbolic value
            assign(function, evaluate_constant(value))
        elif isinstance(value, ufl.core.expr.Expr) and not isinstance(
            value, dolfinx.fem.Function
        ):
            # Kept, so that pack() evaluates it again with the current inputs
            try:
                expression = discretization.compile_expression(value, space)
            except ValueError as error:
                raise ValueError(
                    f"The value for {field.name!r} on {markers!r}: {error}"
                ) from error
            evaluation = Evaluation(function, expression)
            evaluation()
        else:
            assign(function, value, si_factor(field.unit))
        self._symbolic = condition
        self._function = function
        self._dolfinx_bc = dolfinx.fem.dirichletbc(function, dofs)
        self._evaluation = evaluation

    @property
    def symbolic(self) -> symbolic.DirichletBC:
        """The symbolic condition."""
        return self._symbolic

    @property
    def function(self) -> dolfinx.fem.Function:
        """The function holding the prescribed values."""
        return self._function

    @property
    def dolfinx_bc(self) -> dolfinx.fem.DirichletBC:
        """The DOLFINx boundary condition."""
        return self._dolfinx_bc

    @property
    def expression(self) -> dolfinx.fem.Expression | None:
        """The compiled expression of a symbolic value, or ``None``."""
        return None if self._evaluation is None else self._evaluation.expression

    def pack(self) -> None:
        """Evaluate a symbolic value with the current values of its inputs."""
        if self._evaluation is not None:
            self._evaluation()


@counterpart.register
def _(
    condition: symbolic.DirichletBC,
    discretization: Discretization,
    **options: typing.Any,
) -> DirichletBC:
    return DirichletBC(discretization, condition)
