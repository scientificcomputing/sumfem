"""Advection-diffusion with manufactured solutions."""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import (
    Advection,
    BoundaryFlux,
    ConservativeAdvection,
    ConvectiveAdvection,
    Diffusion,
    Outflow,
    Source,
)
from sumfem.post_processing import error
from sumfem.solvers import Solver
from sumfem.symbolic import DirichletBC, Domain, Field


def square(n: int) -> MeshData:
    """A unit square whose outlet is the side ``x = 1``, and inlet the rest.

    A boundary facet is marked if all its vertices are, so the inlet's
    locator takes the corners of the outlet too, for the facets beside them.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    return MeshData.from_locators(
        mesh,
        {
            1: {
                "Inlet": lambda x: (
                    ~np.isclose(x[0], 1.0)
                    | np.isclose(x[1], 0.0)
                    | np.isclose(x[1], 1.0)
                ),
                "Outlet": lambda x: np.isclose(x[0], 1.0),
            }
        },
        boundary=True,
    )


def exact(x):
    return 1 + ufl.sin(np.pi * x[0]) * x[1] ** 2


def solve(n: int, form: type[Advection], degree: int = 2) -> float:
    """Advection-diffusion with Dirichlet data on the inlet and a free outlet.

    The equation is ``-div(D grad c) + div(u c) = f`` for the conservative
    form and ``-div(D grad c) + u . grad(c) = f`` for the convective one. The
    velocity has a divergence, so the two differ. The outlet, ``x = 1``,
    gets the exact diffusive flux, and in conservative form the advective
    one by :class:`Outflow`.
    """
    domain = Domain("square", "triangle")
    c = Field("c", domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    normal = ufl.FacetNormal(domain)
    u = ufl.as_vector((1 + x[0], 0.5 * x[1]))
    D = 0.1
    c_exact = exact(x)
    if form is ConservativeAdvection:
        advection = ufl.div(u * c_exact)
    else:
        advection = ufl.dot(u, ufl.grad(c_exact))
    model = (
        Diffusion("diffusion", c, coefficient=D)
        + form("advection", c, u)
        + Source("source", c, source=-ufl.div(D * ufl.grad(c_exact)) + advection)
        + BoundaryFlux(
            "diffusive_flux", c, D * ufl.dot(ufl.grad(c_exact), normal), ("Outlet",)
        )
    )
    if form is ConservativeAdvection:
        model = model + Outflow("outflow", c, u, ("Outlet",))
    disc = discretize([model], DomainMaps({"square": square(n)}))
    (solution,) = Solver(disc[model], [DirichletBC(c, "Inlet", c_exact)]).solve()
    return error(solution, exact)


@pytest.mark.parametrize("form", [ConvectiveAdvection, ConservativeAdvection])
def test_advection_diffusion_converges(form):
    errors = np.array([solve(n, form) for n in (4, 8)])
    rate = np.log2(errors[0] / errors[1])
    assert rate > 2.8, (errors, rate)
