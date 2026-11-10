"""Boundary terms: fluxes, outflow, and conditions with Lagrange multipliers."""

from __future__ import annotations

import numpy.typing as npt
import ufl

from sumfem.symbolic import Field, Residual, as_expr

__all__ = ["BoundaryConstraint", "BoundaryFlux", "DirichletMultiplier", "Outflow"]


class BoundaryFlux(Residual):
    """An inflow ``q`` into the domain through marked facets, ``-(q, v) ds``.

    Applied currents, the proton flux through a membrane and a prescribed
    inflow all take this form.

    The inflow is what enters per unit area of the boundary, of the field's
    shape, and is used as given: the model takes no normal component. For a
    scalar field with a flux vector ``j``, the inflow is ``q = -j . n``,
    e.g. ``-ufl.dot(j, ufl.FacetNormal(domain))``. For a vector field, it
    is a vector, e.g. a traction, tangential components included.
    :class:`Outflow`, by contrast, is given a velocity and takes its normal
    component itself.

    Args:
        name: Name of the model.
        field: The field whose equation receives the inflow.
        inflow: The inflow ``q``, per unit area, of the field's shape.
        markers: Facet markers of the boundary.

    Raises:
        ValueError: If the inflow has another shape than the field, e.g. a
            flux vector for a scalar field.
    """

    _field: Field
    """The field whose equation receives the inflow."""

    _inflow: ufl.core.expr.Expr
    """The inflow into the domain."""

    _markers: tuple[str, ...]
    """Facet markers of the boundary."""

    def __init__(
        self,
        name: str,
        field: Field,
        inflow: ufl.core.expr.Expr | npt.ArrayLike,
        markers: tuple[str, ...],
    ):
        super().__init__(name)
        self._field = field
        self._inflow = as_expr(inflow)
        if self._inflow.ufl_shape != field.ufl_shape:
            hint = (
                ": for a flux vector j, give the inflow -dot(j, n)"
                if field.ufl_shape == () and len(self._inflow.ufl_shape) == 1
                else ""
            )
            raise ValueError(
                f"The inflow of {name!r} has the shape {self._inflow.ufl_shape}, "
                f"but {field.name!r} has {field.ufl_shape}{hint}."
            )
        self._markers = tuple(markers)

    def residual(self) -> ufl.Form:
        u = self._field
        ds = u.domain.ds(*self._markers)
        return -ufl.inner(self._inflow, u.v) * ds


class Outflow(Residual):
    """Advection of a field out of the domain, ``(u . n c, w) ds``.

    Args:
        name: Name of the model.
        field: The advected field ``c``.
        velocity: The velocity ``u``.
        markers: Facet markers of the outlet.
    """

    _field: Field
    """The advected field."""

    _velocity: ufl.core.expr.Expr
    """The velocity."""

    _markers: tuple[str, ...]
    """Facet markers of the outlet."""

    def __init__(
        self,
        name: str,
        field: Field,
        velocity: ufl.core.expr.Expr,
        markers: tuple[str, ...],
    ):
        super().__init__(name)
        self._field = field
        self._velocity = as_expr(velocity)
        self._markers = tuple(markers)

    def residual(self) -> ufl.Form:
        c = self._field
        n = ufl.FacetNormal(c.domain)
        ds = c.domain.ds(*self._markers)
        return ufl.dot(self._velocity, n) * c * c.v * ds


class BoundaryConstraint(Residual):
    """A constraint ``c(u) = 0`` on marked facets, with a Lagrange multiplier.

    The constraint is imposed at the stationary point of the Lagrangian
    ``L(u, lambda) = (lambda, c(u)) ds``, so the model adds its variations::

        (lambda, c'(u)[v]) ds + (c(u), mu) ds

    with ``v`` the test function of the field ``u``, ``mu`` that of the
    multiplier ``lambda`` and ``c'(u)[v]`` the derivative of the constraint
    in the direction ``v``. The multiplier has the constraint's shape, e.g. a
    scalar for the normal component of a velocity, ``c(u) = u . n - g``. It
    usually lives on a submesh of the facets, see
    :class:`~sumfem.discrete.Discretization` for the entity maps between
    meshes. :class:`DirichletMultiplier` is the constraint ``u - g``.

    Args:
        name: Name of the model.
        field: The constrained field ``u``.
        multiplier: The multiplier ``lambda``, of the constraint's shape.
        constraint: The constraint ``c(u)``, an expression of the field.
            Subclasses may give ``None`` and override :meth:`constraint`.
        markers: Facet markers of the constrained boundary, on the field's
            domain.

    Raises:
        ValueError: If the multiplier has another shape than the constraint.

    Example:
        No flow through a wall, ``u . n = 0``, for a velocity ``u``::

            n = ufl.FacetNormal(domain)
            wall = BoundaryConstraint("wall", u, p_wall, ufl.dot(u, n), ("Wall",))
    """

    _field: Field
    """The constrained field."""

    _multiplier: Field
    """The multiplier."""

    _constraint: ufl.core.expr.Expr | None
    """The given constraint, or ``None`` for a subclass that builds it."""

    _markers: tuple[str, ...]
    """Facet markers of the constrained boundary."""

    def __init__(
        self,
        name: str,
        field: Field,
        multiplier: Field,
        constraint: ufl.core.expr.Expr | None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name)
        self._field = field
        self._multiplier = multiplier
        self._constraint = None if constraint is None else as_expr(constraint)
        self._markers = tuple(markers)
        if self._constraint is not None:
            self._check_shape(self._constraint)

    def _check_shape(self, constraint: ufl.core.expr.Expr) -> None:
        """Check that the multiplier has the shape of the constraint."""
        if self._multiplier.ufl_shape != constraint.ufl_shape:
            raise ValueError(
                f"The multiplier {self._multiplier.name!r} has shape "
                f"{self._multiplier.ufl_shape}, but the constraint of "
                f"{self.name!r} has {constraint.ufl_shape}."
            )

    def constraint(self) -> ufl.core.expr.Expr:
        """The constraint ``c(u)``."""
        assert self._constraint is not None
        return self._constraint

    def residual(self) -> ufl.Form:
        u, lam = self._field, self._multiplier
        ds = u.domain.ds(*self._markers)
        c = self.constraint()
        lagrangian = ufl.inner(lam, c) * ds
        return ufl.derivative(lagrangian, u, u.v) + ufl.inner(c, lam.v) * ds


class DirichletMultiplier(BoundaryConstraint):
    """A Dirichlet condition ``u = g`` imposed with a Lagrange multiplier.

    The :class:`BoundaryConstraint` ``c(u) = u - g``, which adds
    ``(lambda, v) + (u - g, mu)`` on the marked facets. Integrating by parts
    identifies the multiplier as the normal flux ``-grad(u) . n`` of a
    diffusion equation.

    Args:
        name: Name of the model.
        field: The constrained field ``u``.
        multiplier: The multiplier ``lambda``, of the field's shape.
        value: The prescribed value ``g``. Defaults to the parameter
            ``"<name>.value"``.
        markers: Facet markers of the constrained boundary, on the field's
            domain.

    Raises:
        ValueError: If the multiplier has another shape than the field.
    """

    _value: ufl.core.expr.Expr
    """The prescribed value."""

    def __init__(
        self,
        name: str,
        field: Field,
        multiplier: Field,
        value: ufl.core.expr.Expr | npt.ArrayLike | None = None,
        markers: tuple[str, ...] = (),
    ):
        super().__init__(name, field, multiplier, None, markers)
        self._value = self.input(
            "value", value, domain=field.domain, shape=field.ufl_shape, doc="Value"
        )
        self._check_shape(self.constraint())

    def constraint(self) -> ufl.core.expr.Expr:
        """The constraint ``u - g``."""
        return self._field - self._value
