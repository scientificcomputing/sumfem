"""Boundary terms: errors in their inputs."""

import pytest
import ufl

from sumfem.models import BoundaryConstraint, BoundaryFlux, DirichletMultiplier
from sumfem.symbolic import Domain, Field


def test_an_inflow_must_have_the_shape_of_its_field():
    """An inflow of another shape than its field is rejected when the model is
    created, with a hint for a flux vector given for a scalar field.
    """
    domain = Domain("box", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    u = Field("u", domain, ("Lagrange", 1, (2,)))
    flux = ufl.as_vector([1.0, 2.0])
    with pytest.raises(ValueError, match=r"give the inflow -dot\(j, n\)"):
        BoundaryFlux("in", c, flux, markers=("Left",))
    with pytest.raises(ValueError, match=r"has the shape \(\), but 'u' has \(2,\)\.$"):
        BoundaryFlux("traction", u, 1.0, markers=("Left",))
    assert BoundaryFlux("traction", u, flux, markers=("Left",)).equations == (u,)


def test_a_multiplier_must_have_the_shape_of_its_constraint():
    """A multiplier of another shape than its constraint is rejected when the
    model is created: a scalar one for the normal component of a vector field
    is accepted, and must be a vector for a Dirichlet condition on it.
    """
    domain = Domain("box", "triangle")
    u = Field("u", domain, ("Lagrange", 1, (2,)))
    scalar = Field("lambda", domain, ("Lagrange", 1))
    n = ufl.FacetNormal(domain)
    assert BoundaryConstraint("wall", u, scalar, ufl.dot(u, n), ("Left",))
    with pytest.raises(ValueError, match=r"has shape \(\), but the constraint"):
        BoundaryConstraint("wall", u, scalar, u, ("Left",))
    with pytest.raises(ValueError, match=r"has shape \(\), but the constraint"):
        DirichletMultiplier("dirichlet", u, scalar, markers=("Left",))
