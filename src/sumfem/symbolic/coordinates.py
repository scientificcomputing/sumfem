"""Coordinate systems of domains, which weight the integrals over them.

A body with rotational or radial symmetry is solved on a mesh of lower
dimension, with every integral weighted by the measure of the geometry that
the mesh sweeps: ``2 pi r`` for an axisymmetric body in ``(r, z)``, ``4 pi
r^2`` for a radially symmetric ball in ``r``. The weights include the
factors ``2 pi`` and ``4 pi``, so every integral is that of the body itself:
volumes, amounts, currents through boundaries and energies.

The weight alone is exact for scalar unknowns whose equations are written in
weak form, ``(flux, grad(v)) + (source, v)``, e.g. diffusion, heat
conduction, Poisson's equation or Nernst-Planck transport. The gradients
and divergences of vectors and tensors also have terms from the Christoffel
symbols of the coordinates, since their basis vectors turn with the position,
e.g. ``u_r / r`` in the divergence of a vector in cylindrical coordinates. In
cylindrical and spherical coordinates every integrand is therefore lifted to
3D: its vectors and tensors are written in physical components, in the
normalized basis of the coordinates, and its gradients and divergences gain
these terms, see :meth:`Cylindrical.lift`. Models written with ``grad``,
``div``, ``inner``, ``sym``, ``tr``, ``det`` and the like then hold
unchanged.

A field of the mesh's dimension is a vector in space unless it is created
with ``spatial=False``, e.g. a list of concentrations, whose gradient is that
of each scalar. Literals and data of the mesh's dimension, e.g. a source
vector or an anisotropic conductivity, have zero angular components, so they
act on vectors in space and on lists of scalars alike. Their inverses and
determinants would need the angular components, and are rejected.

The operators in physical components are those of R. B. Bird, W. E. Stewart
and E. N. Lightfoot, "Transport Phenomena", 2nd edition, Wiley, 2002,
Appendix A, and their derivation from the Christoffel symbols is in R. Aris,
"Vectors, Tensors and the Basic Equations of Fluid Mechanics", Dover, 1989.
"""

from __future__ import annotations

import dataclasses
import typing
from functools import singledispatchmethod

import numpy as np
import ufl
from ufl.algorithms import estimate_total_polynomial_degree, expand_derivatives
from ufl.classes import (
    CoefficientDerivative,
    Constant,
    ConstantValue,
    Curl,
    Determinant,
    Div,
    Expr,
    FacetNormal,
    FormArgument,
    Grad,
    Identity,
    Inverse,
    Label,
    ListTensor,
    MultiIndex,
    NablaDiv,
    NablaGrad,
    SpatialCoordinate,
    Terminal,
)
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.corealg.traversal import unique_pre_traversal

from ..error_handling import ComponentError
from .expressions import FormOrZero

__all__ = [
    "Cartesian",
    "CoordinateSystem",
    "Cylindrical",
    "Spherical",
    "Weighted",
    "apply_coordinates",
    "lift",
    "supports_vectors",
]


class CoordinateSystem(typing.Protocol):
    """How the integrals over a domain are weighted.

    A coordinate system may also write the vectors of integrands in its
    basis, with the method ``lift(integrand, x)``, see
    :meth:`Cylindrical.lift`, and say whether it supports vector-valued
    unknowns with the class attribute ``vectors``. Without them, integrands
    are kept as they are and only scalar unknowns are supported, see
    :func:`lift` and :func:`supports_vectors`.
    """

    def weight(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """The weight of an integral.

        Args:
            x: The spatial coordinate of the domain.
        """

    def check(self, gdim: int) -> None:
        """Check that a domain of geometric dimension ``gdim`` can use it.

        Raises:
            ComponentError: If it cannot.
        """


def lift(coordinates: CoordinateSystem, integrand: Expr, x: ufl.core.expr.Expr) -> Expr:
    """An integrand in a coordinate system's basis, if it has one.

    Args:
        coordinates: The coordinate system.
        integrand: The integrand, before derivatives are evaluated.
        x: The spatial coordinate of the domain.
    """
    method = getattr(coordinates, "lift", None)
    return integrand if method is None else method(integrand, x)


def supports_vectors(coordinates: CoordinateSystem) -> bool:
    """Whether a coordinate system supports vector-valued unknowns."""
    return bool(getattr(coordinates, "vectors", False))


@dataclasses.dataclass(frozen=True)
class Cartesian:
    """Integrals without a weight, the default of every domain.

    The weight is the integer one, which UFL drops from products, and
    integrands are kept as they are, so forms keep their integrands.
    """

    vectors: typing.ClassVar[bool] = True

    def weight(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return ufl.as_ufl(1)

    def check(self, gdim: int) -> None:
        """Any geometric dimension."""


def _derivative(f: Expr, i: int) -> Expr:
    """``f.dx(i)``, and zero for a literal, which has no domain to differentiate on."""
    return ufl.as_ufl(0) if isinstance(f, ConstantValue) else f.dx(i)


def _over_r(value: Expr, r: Expr) -> Expr:
    """``value / r``, and zero on the axis, where the weight is zero too."""
    return ufl.conditional(ufl.gt(r, 0), value / r, 0)


@dataclasses.dataclass(frozen=True)
class Cylindrical:
    """An axisymmetric body, with the weight ``2 pi r``.

    The mesh lies in the half-plane ``x[0] = r >= 0``, with ``x[1] = z`` on
    a 2D mesh. On a 1D mesh in ``r`` the integrals are per unit length of an
    infinite cylinder. On the axis the weight vanishes, so the symmetry
    condition of scalars there is the natural one. A vector's radial
    component vanishes on the axis, which is a Dirichlet condition.

    Vectors are written in the basis ``(r, z, theta)``: a field of shape
    ``(gdim,)`` has no swirl, and one of shape ``(3,)`` has the swirl
    ``u_theta`` as its last component. Fields do not depend on ``theta``.
    """

    vectors: typing.ClassVar[bool] = True

    def weight(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return 2 * np.pi * x[0]

    def lift(self, integrand: Expr, x: ufl.core.expr.Expr) -> Expr:
        """See :class:`_Lift`."""
        return _Lift(self, x)(integrand)

    def check(self, gdim: int) -> None:
        if gdim not in (1, 2):
            raise ComponentError(
                f"Cylindrical coordinates need a mesh in r or (r, z), got gdim {gdim}."
            )

    def gradient(self, f: Expr, x: ufl.core.expr.Expr) -> Expr:
        """The gradient of a scalar or a vector in ``(r, z, theta)``.

        Raises:
            ComponentError: For other shapes.
        """
        r = x[0]

        def d_r(g: Expr) -> Expr:
            return _derivative(g, 0)

        def d_z(g: Expr) -> Expr:
            return _derivative(g, 1) if x.ufl_shape[0] == 2 else ufl.as_ufl(0)

        if f.ufl_shape == ():
            return ufl.as_vector((d_r(f), d_z(f), 0))
        if f.ufl_shape == (3,):
            u_r, u_z, u_t = f[0], f[1], f[2]
            return ufl.as_matrix(
                (
                    (d_r(u_r), d_z(u_r), -_over_r(u_t, r)),
                    (d_r(u_z), d_z(u_z), 0),
                    (d_r(u_t), d_z(u_t), _over_r(u_r, r)),
                )
            )
        raise ComponentError(
            f"Cylindrical coordinates have gradients of scalars and vectors, not of "
            f"shape {f.ufl_shape}."
        )

    def divergence(self, f: Expr, x: ufl.core.expr.Expr) -> Expr:
        """The divergence of a vector or a tensor in ``(r, z, theta)``.

        Raises:
            ComponentError: For other shapes.
        """
        if f.ufl_shape == (3,):
            return ufl.tr(self.gradient(f, x))
        if f.ufl_shape == (3, 3):
            r = x[0]
            rows = [self.gradient(f[i, :], x) for i in range(3)]
            # The divergence of each row, and the Christoffel terms of the basis
            return ufl.as_vector(
                (
                    rows[0][0, 0] + rows[0][1, 1] + _over_r(f[0, 0] - f[2, 2], r),
                    rows[1][0, 0] + rows[1][1, 1] + _over_r(f[1, 0], r),
                    rows[2][0, 0] + rows[2][1, 1] + _over_r(f[0, 2] + f[2, 0], r),
                )
            )
        raise ComponentError(
            f"Cylindrical coordinates have divergences of vectors and tensors, not of "
            f"shape {f.ufl_shape}."
        )


@dataclasses.dataclass(frozen=True)
class Spherical:
    """A radially symmetric body, with the weight ``4 pi r^2``.

    The mesh is an interval in ``x[0] = r >= 0``. Vectors are written in the
    basis ``(r, theta, phi)``, and only have a radial component.
    """

    vectors: typing.ClassVar[bool] = True

    def weight(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return 4 * np.pi * x[0] ** 2

    def lift(self, integrand: Expr, x: ufl.core.expr.Expr) -> Expr:
        """See :class:`_Lift`."""
        return _Lift(self, x)(integrand)

    def check(self, gdim: int) -> None:
        if gdim != 1:
            raise ComponentError(
                f"Spherical coordinates need a mesh in r, got gdim {gdim}."
            )

    def gradient(self, f: Expr, x: ufl.core.expr.Expr) -> Expr:
        """The gradient of a scalar or a radial vector in ``(r, theta, phi)``.

        Raises:
            ComponentError: For other shapes.
        """
        if f.ufl_shape == ():
            return ufl.as_vector((_derivative(f, 0), 0, 0))
        if f.ufl_shape == (3,):
            turning = _over_r(f[0], x[0])
            return ufl.as_matrix(
                ((_derivative(f[0], 0), 0, 0), (0, turning, 0), (0, 0, turning))
            )
        raise ComponentError(
            f"Spherical coordinates have gradients of scalars and vectors, not of "
            f"shape {f.ufl_shape}."
        )

    def divergence(self, f: Expr, x: ufl.core.expr.Expr) -> Expr:
        """The divergence of a radial vector or a diagonal tensor.

        Raises:
            ComponentError: For other shapes.
        """
        if f.ufl_shape == (3,):
            return ufl.tr(self.gradient(f, x))
        if f.ufl_shape == (3, 3):
            turning = _over_r(2 * f[0, 0] - f[1, 1] - f[2, 2], x[0])
            return ufl.as_vector((_derivative(f[0, 0], 0) + turning, 0, 0))
        raise ComponentError(
            f"Spherical coordinates have divergences of vectors and tensors, not of "
            f"shape {f.ufl_shape}."
        )


@dataclasses.dataclass(frozen=True)
class Weighted:
    """Integrals with any weight of the spatial coordinate.

    E.g. ``Weighted(lambda x: 2 * pi * (1 - cos(theta)) * x[0] ** 2)`` for a
    cone of half-angle ``theta`` around its axis, in the distance ``x[0]``
    from its apex. Integrands are kept as they are, so only scalar unknowns
    are supported.

    Attributes:
        weight_of: The weight, as a function of the spatial coordinate.
    """

    vectors: typing.ClassVar[bool] = False

    weight_of: typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr | float]

    def weight(self, x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return ufl.as_ufl(self.weight_of(x))

    def check(self, gdim: int) -> None:
        """Any geometric dimension."""


def _field_of(o: Expr) -> typing.Any:
    """The field of a terminal, its test or its trial function, if any."""
    # Imported here, since terminals depend on domains, which depend on this
    from .terminals import field_of

    return field_of(o)


def _is_spatial(o: Expr) -> bool:
    """Whether a terminal of vectors is a vector in space.

    See :class:`~sumfem.symbolic.terminals.Field`. The test and trial
    functions of a field are as their field.
    """
    field = _field_of(o)
    return field.spatial if field is not None else bool(getattr(o, "spatial", True))


def _has_tensor_data(expr: Expr) -> bool:
    """Whether an expression has tensor-valued data or a literal matrix."""
    return any(
        (isinstance(node, FormArgument | Constant) and len(node.ufl_shape) > 1)
        or (isinstance(node, ListTensor) and len(node.ufl_shape) > 1)
        for node in unique_pre_traversal(expr)
    )


def _is_scaled_identity(o: ListTensor) -> bool:
    """Whether a literal matrix is a multiple of the identity."""
    n, m = o.ufl_shape
    entries = [[o[i, j] for j in range(m)] for i in range(n)]
    if not all(isinstance(e, ConstantValue) for row in entries for e in row):
        return False
    values = [[float(e) for e in row] for row in entries]
    diagonal = {values[i][i] for i in range(n)}
    off = [values[i][j] for i in range(n) for j in range(m) if i != j]
    return n == m and len(diagonal) == 1 and 0.0 not in diagonal and not any(off)


class _Lift(DAGTraverser):
    """Write the vectors and tensors of an integrand in 3D curvilinear coordinates.

    Spatial vectors, i.e. fields not marked ``spatial=False``, their test
    and trial functions, the facet normal and the position, gain zero
    components up to three, and ``Identity(gdim)`` becomes ``Identity(3)``.
    Gradients and divergences become those of the coordinates, with their
    Christoffel terms; the gradient of a list of scalars, a field with
    ``spatial=False``, is that of each scalar. Literals and data of the
    mesh's dimension, e.g. a source vector or an anisotropic conductivity,
    gain zero components where they meet a 3D operand, so they act on
    spatial vectors and on lists of scalars alike. All other algebra, e.g.
    ``inner``, ``sym``, ``tr`` and ``det``, then acts on the 3D vectors and
    tensors.

    It must run before UFL lowers the algebra to index sums of the mesh's
    dimension, i.e. before derivatives are evaluated. The coefficients of an
    unevaluated derivative are kept.

    Raises:
        ComponentError: For the inverse or determinant of tensor-valued data
            or literal matrices, whose angular components are unknown; for
            literal multiples of the identity, which are ``ufl.Identity``;
            for gradients and divergences of expressions with free indices,
            i.e. of components rather than vectors; for curls; and in
            spherical coordinates for vectors with angular components.
    """

    def __init__(self, coordinates: Cylindrical | Spherical, x: ufl.core.expr.Expr):
        super().__init__()
        self._coordinates = coordinates
        self._x = x

    @property
    def _gdim(self) -> int:
        """The geometric dimension of the mesh."""
        return self._x.ufl_shape[0]

    @property
    def _name(self) -> str:
        return f"{type(self._coordinates).__name__} coordinates"

    def _planar(self, expr: Expr) -> bool:
        """Whether an expression is a vector or matrix of the mesh's dimension."""
        shape = expr.ufl_shape
        return len(shape) in (1, 2) and all(n == self._gdim for n in shape)

    def _padded(self, tensor: Expr) -> Expr:
        """A vector or matrix of the mesh's dimension, with zeros up to three."""
        gdim = self._gdim
        if len(tensor.ufl_shape) == 1:
            return ufl.as_vector([tensor[i] for i in range(gdim)] + [0] * (3 - gdim))
        return ufl.as_matrix(
            [
                [tensor[i, j] if i < gdim and j < gdim else 0 for j in range(3)]
                for i in range(3)
            ]
        )

    def _operand(self, o: Expr) -> Expr:
        """The processed operand of a gradient or divergence, as a 3D tensor.

        Raises:
            ComponentError: If it has free indices, i.e. is a component.
        """
        (operand,) = o.ufl_operands
        if operand.ufl_free_indices:
            raise ComponentError(
                f"{type(o).__name__} of {operand}, an expression with free indices, "
                f"in {self._name}: the derivatives of components are not those of "
                "the vector. Write the operator on the vector, e.g. div(u) for "
                "u[i].dx(i)."
            )
        processed = self(operand)
        return self._padded(processed) if self._planar(processed) else processed

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        operands = [self(op) for op in o.ufl_operands]
        shaped = [op for op in operands if not isinstance(op, MultiIndex | Label)]
        if any(3 in op.ufl_shape for op in shaped) and any(
            self._planar(op) for op in shaped
        ):
            # Planar literals and data meet a 3D operand
            operands = [
                self._padded(op)
                if not isinstance(op, MultiIndex | Label) and self._planar(op)
                else op
                for op in operands
            ]
        if all(a is b for a, b in zip(operands, o.ufl_operands, strict=True)):
            return o
        return o._ufl_expr_reconstruct_(*operands)

    @process.register(Terminal)
    def _(self, o: Terminal) -> Expr:
        gdim = self._gdim
        if isinstance(o, Identity):
            return ufl.Identity(3) if o.ufl_shape == (gdim, gdim) else o
        if isinstance(o, SpatialCoordinate | FacetNormal):
            return self._padded(o)
        if not isinstance(o, FormArgument | Constant) or not _is_spatial(o):
            # e.g. literals, multi-indices and lists of scalars
            return o
        if o.ufl_shape == (3,) and isinstance(self._coordinates, Spherical):
            raise ComponentError(
                f"{o} has angular components in {self._name}, whose vectors are "
                "radial. Give it the shape (1,)."
            )
        if _field_of(o) is None:
            # Data is padded where it meets a 3D operand
            return o
        if o.ufl_shape == (gdim,):
            return self._padded(o)
        if len(o.ufl_shape) > 1:
            raise ComponentError(
                f"{o} of shape {o.ufl_shape} cannot be written in {self._name}."
            )
        return o

    @process.register(ListTensor)
    def _(self, o: ListTensor) -> Expr:
        if len(o.ufl_shape) == 2 and _is_scaled_identity(o):
            raise ComponentError(
                f"The literal matrix {o} is a multiple of the identity, whose "
                f"angular components in {self._name} are not zero. Write it with "
                "ufl.Identity."
            )
        return self.reuse_if_untouched(o)

    @process.register(Grad)
    def _(self, o: Grad) -> Expr:
        (operand,) = o.ufl_operands
        if isinstance(operand, FormArgument) and not _is_spatial(operand):
            # A list of scalars: the gradient of each
            return ufl.as_tensor(
                [
                    self._coordinates.gradient(operand[i], self._x)
                    for i in range(operand.ufl_shape[0])
                ]
            )
        return self._coordinates.gradient(self._operand(o), self._x)

    @process.register(NablaGrad)
    def _(self, o: NablaGrad) -> Expr:
        gradient = self._coordinates.gradient(self._operand(o), self._x)
        return gradient.T if len(gradient.ufl_shape) == 2 else gradient

    @process.register(Div)
    def _(self, o: Div) -> Expr:
        return self._coordinates.divergence(self._operand(o), self._x)

    @process.register(NablaDiv)
    def _(self, o: NablaDiv) -> Expr:
        f = self._operand(o)
        return self._coordinates.divergence(
            f.T if len(f.ufl_shape) == 2 else f, self._x
        )

    @process.register(Inverse)
    @process.register(Determinant)
    def _(self, o: Expr) -> Expr:
        if _has_tensor_data(o.ufl_operands[0]):
            raise ComponentError(
                f"{type(o).__name__} of {o.ufl_operands[0]} in {self._name} needs the "
                "angular components of its tensor-valued data, which are unknown. "
                "Give the data as a scalar."
            )
        return self.reuse_if_untouched(o)

    @process.register(Curl)
    def _(self, o: Curl) -> Expr:
        raise ComponentError(f"Curls are not available in {self._name}.")

    @process.register(CoefficientDerivative)
    def _(self, o: CoefficientDerivative) -> Expr:
        integrand, *derivative = o.ufl_operands
        return o._ufl_expr_reconstruct_(self(integrand), *derivative)


@typing.overload
def apply_coordinates(form: ufl.Form) -> ufl.Form: ...


@typing.overload
def apply_coordinates(form: ufl.ZeroBaseForm) -> ufl.ZeroBaseForm: ...


def apply_coordinates(form: FormOrZero) -> FormOrZero:
    """A form with each integral in its domain's coordinate system.

    Each integrand is lifted, see :meth:`CoordinateSystem.lift`, and
    weighted. Integrals over a mesh that is not a
    :class:`~sumfem.symbolic.domains.Domain` are Cartesian. On interior
    facets the weight is taken from the ``"+"`` side.
    """
    if isinstance(form, ufl.ZeroBaseForm):
        return form
    integrals = []
    for integral in form.integrals():
        domain = integral.ufl_domain()
        coordinates: CoordinateSystem = getattr(domain, "coordinates", Cartesian())
        x = ufl.SpatialCoordinate(domain)
        weight = coordinates.weight(x)
        if integral.integral_type() == "interior_facet":
            weight = weight("+")
        integrand = lift(coordinates, integral.integrand(), x)
        metadata = integral.metadata()
        if (
            integrand is not integral.integrand()
            and "quadrature_degree" not in metadata
        ):
            # The Christoffel terms divide by r, which UFL counts as a degree
            # of the denominator; the weight multiplies it back
            metadata = {**metadata, "quadrature_degree": _degree(weight, integral)}
        integrals.append(
            integral.reconstruct(integrand=weight * integrand, metadata=metadata)
        )
    return ufl.Form(integrals)


def _degree(weight: Expr, integral: ufl.Integral) -> int:
    """The quadrature degree of a lifted integral, from its integrand before.

    That of the weighted integrand before the lift, plus one for the
    Christoffel terms, which have the degree of a vector rather than of its
    gradient.
    """
    # The estimate needs the algebra lowered, so it is taken of a copy
    before = expand_derivatives(weight * integral.integrand())
    return estimate_total_polynomial_degree(before) + 1
