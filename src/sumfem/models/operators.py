"""Helpers for writing the expressions of models."""

from __future__ import annotations

import ufl

__all__ = [
    "apply_coefficient",
    "facet_size",
    "normal_outer",
    "normal_projection",
    "restricted",
    "tangential_projection",
]


def apply_coefficient(
    coefficient: ufl.core.expr.Expr, vector: ufl.core.expr.Expr
) -> ufl.core.expr.Expr:
    """A scalar or matrix coefficient applied to a vector.

    Args:
        coefficient: A scalar, such as an isotropic diffusivity, or a matrix,
            such as an anisotropic one.
        vector: The vector, such as a gradient or a facet normal.

    Raises:
        ValueError: If the coefficient is neither a scalar nor a matrix.
    """
    rank = len(coefficient.ufl_shape)
    if rank == 0:
        return coefficient * vector
    if rank == 2:
        return ufl.dot(coefficient, vector)
    raise ValueError(
        "A coefficient must be a scalar or a matrix, "
        f"got shape {coefficient.ufl_shape}."
    )


def facet_size(domain: ufl.Mesh) -> ufl.core.expr.Expr:
    """The volume of a cell over the area of its facet, a size normal to it.

    From the Jacobians, as UFL lowers ``CellVolume`` and ``FacetArea`` on
    affine cells, so it is exact there and the local value on other cells,
    e.g. quadrilaterals, where FFCx has neither. On intervals a facet is a
    point, of area one.
    """
    volume = abs(
        ufl.JacobianDeterminant(domain) * ufl.geometry.ReferenceCellVolume(domain)
    )
    if domain.topological_dimension == 1:
        return volume
    area = abs(
        ufl.geometry.FacetJacobianDeterminant(domain)
        * ufl.geometry.ReferenceFacetVolume(domain)
    )
    return volume / area


def normal_outer(w: ufl.core.expr.Expr, n: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """``w (x) n``, which is ``w n`` for a scalar ``w``."""
    return w * n if w.ufl_shape == () else ufl.outer(w, n)


def normal_projection(
    vector: ufl.core.expr.Expr, normal: ufl.core.expr.Expr
) -> ufl.core.expr.Expr:
    """The normal component ``(n (x) n) w = (w . n) n`` of a vector ``w``."""
    return ufl.dot(vector, normal) * normal


def tangential_projection(
    vector: ufl.core.expr.Expr, normal: ufl.core.expr.Expr
) -> ufl.core.expr.Expr:
    """The tangential component ``(I - n (x) n) w`` of a vector ``w``."""
    return vector - normal_projection(vector, normal)


def restricted(expr: ufl.core.expr.Expr, side: str) -> ufl.core.expr.Expr:
    """An expression restricted to a side of interior facets.

    Constants, which UFL does not restrict, are returned as they are.
    """
    if ufl.domain.extract_domains(expr):
        return expr(side)
    return expr
