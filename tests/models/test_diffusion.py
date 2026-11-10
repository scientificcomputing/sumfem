"""Poisson's equation on one and two domains, and coupled through a source."""

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import Accumulation, Diffusion, Poisson, Source
from sumfem.post_processing import error
from sumfem.solvers import Solver
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    build_system,
)


def boundary(mesh: dolfinx.mesh.Mesh) -> MeshData:
    """A mesh with its whole boundary marked as ``"Boundary"``."""
    return MeshData.from_locators(
        mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
    )


def quadratic(x, shape=()):
    """``2 x^2 + y^2`` in every component, with Laplacian 6."""
    value = 2 * x[0] ** 2 + x[1] ** 2
    return np.tile(value, (int(np.prod(shape)), 1)) if shape else value


def tensor(shape):
    """The exact solution as a UFL tensor of the given shape."""

    def exact(x):
        value = 2 * x[0] ** 2 + x[1] ** 2
        return ufl.as_tensor(np.full(shape, value).tolist()) if shape else value

    return exact


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
@pytest.mark.parametrize("degree", [2, 4])
def test_poisson(shape, cell, degree):
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 5, 5, cell_type=dolfinx.mesh.to_type(cell)
    )
    domain = Domain.from_mesh("square", mesh)
    u = Field("u", domain, ("Lagrange", degree, shape))
    model = Poisson("poisson", u)
    disc = discretize([model], DomainMaps({domain: boundary(mesh)}))
    disc.set({model.source: np.full(shape, -6.0)})
    Solver(
        disc[model],
        [DirichletBC(u, "Boundary", lambda x: quadratic(x, shape))],
    ).solve()
    assert error(disc.function(u), tensor(shape)) < 1e-12


def test_poisson_is_diffusion_and_source():
    domain = Domain("square", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    poisson = Poisson("p", u, source=-6.0, coefficient=2.0)
    split = Diffusion("d", u, 2.0) + Source("s", u, -6.0)
    a = ufl.algorithms.expand_derivatives(poisson.form)
    b = ufl.algorithms.expand_derivatives(split.form)
    assert a.signature() == b.signature()


@pytest.mark.parametrize("shape", [(), (2,)])
@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_poisson_on_two_domains(shape, cell):
    """Two independent problems on the two halves of a square."""
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, cell_type=dolfinx.mesh.to_type(cell)
    )
    tdim = mesh.topology.dim
    left = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[0] <= 0.5 + 1e-10)
    right = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[0] >= 0.5 - 1e-10)
    meshes = {
        "left": dolfinx.mesh.create_submesh(mesh, tdim, left)[0],
        "right": dolfinx.mesh.create_submesh(mesh, tdim, right)[0],
    }
    fields, models, bcs = [], [], []
    for (name, submesh), degree in zip(meshes.items(), (2, 3), strict=True):
        domain = Domain.from_mesh(name, submesh)
        u = Field(f"u_{name}", domain, ("Lagrange", degree, shape))
        fields.append(u)
        models.append(Poisson(f"poisson_{name}", u, np.full(shape, -6.0)))
        bcs.append(DirichletBC(u, "Boundary", lambda x: quadratic(x, shape)))
    model = sum(models)
    disc = discretize(
        [model], DomainMaps({name: boundary(m) for name, m in meshes.items()})
    )
    solver = Solver(disc[model], bcs)
    assert solver.system.symbolic.domains == tuple(f.domain for f in fields)
    solver.solve()
    for u in fields:
        assert error(disc.function(u), tensor(shape)) < 1e-12


def test_accumulation_weight_and_markers():
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1, (2,)))
    porosity = Parameter("eps", domain, default=0.5)
    model = Accumulation("dcdt", c, weight=porosity, markers=("Electrode",))
    system = build_system(model)
    assert porosity in system.residual.coefficients()
    (integral,) = system.residual.integrals()
    # The measure holds a tuple of ids, the integrals in the form hold plain ids
    assert integral.subdomain_id() == domain.markers[domain.tdim]["Electrode"]


def test_accumulation_needs_a_field():
    domain = Domain("cell", "triangle")
    with pytest.raises(TypeError):
        Accumulation("dcdt", Parameter("p", domain))
