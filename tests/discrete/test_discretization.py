"""Binding systems to meshes and solving them."""

from mpi4py import MPI

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl

from sumfem.discrete import DomainMaps, MeshData, counterpart, discretize
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    Subdomain,
    SystemBuildError,
    UnitError,
)


def test_values_are_stored_in_si_units(toy, square, sides):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1), unit=syu.mol / syu.liter)
    length = Parameter("length", domain, default=5.0, unit=syu.cm)
    disc = discretize([c, length], DomainMaps({"cell": square()}))
    np.testing.assert_allclose(disc[length].x.array, 0.05)
    assert disc.value(length) == pytest.approx(5.0)
    disc.set({length: 3 * syu.mm})
    assert disc.value(length) == pytest.approx(0.3)
    np.testing.assert_allclose(disc[length].x.array, 0.003)
    with pytest.raises(ValueError, match="dimension"):
        disc.set({length: 3 * syu.second})
    # Field values are given in the field's unit, mol/L, and stored in mol/m^3
    disc.set({c: 2.0})
    np.testing.assert_allclose(disc.fields["c"].x.array, 2000.0)
    bc = disc[DirichletBC(c, sides, 1.5)]
    np.testing.assert_allclose(bc.function.x.array, 1500.0)


def test_expression_evaluates_symbolic_expressions(square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    scale = Parameter("scale", domain, default=2.0)
    x = ufl.SpatialCoordinate(domain)
    disc = discretize([scale * c + x[0]], DomainMaps({"cell": square()}))
    disc.set({c: 1.0})
    expr = disc[scale * c + x[0]]
    mesh = disc.maps.mesh(domain)
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr * ufl.dx(domain=mesh)))
    assert mesh.comm.allreduce(local, op=MPI.SUM) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="argument"):
        disc[c.v]


def test_mesh_must_match_the_domain(toy, square):
    domain = Domain("cell", "quadrilateral")
    c = Field("c", domain, ("Lagrange", 1))
    with pytest.raises(ValueError, match="coordinate element"):
        discretize([toy.Diffusion("d", c)], DomainMaps({"cell": square("triangle")}))
    with pytest.raises(KeyError, match="No mesh for domain 'other'"):
        DomainMaps({"cell": square()}).mesh(Domain("other", "triangle"))


def test_time_derivatives_need_a_scheme(toy, square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Accumulation("dcdt", c) + toy.Diffusion("d", c)
    disc = discretize([model], DomainMaps({"cell": square()}))
    with pytest.raises(SystemBuildError, match="time derivative"):
        disc[model]


def test_units_are_checked_on_request(toy, square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1), unit=syu.K)
    # The source has the wrong unit for an equation in kelvin
    source = Parameter("q", domain, default=1.0, unit=syu.W)
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=source)
    disc = discretize([model], DomainMaps({"cell": square()}))
    disc[model]
    with pytest.raises(UnitError, match="equation for 'c'"):
        counterpart(model, disc, check_units=True)


def test_max_quadrature_degree_bounds_estimated_degrees(square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))

    class Exponential(Residual):
        """``(exp(c), v)``, whose estimated quadrature degree is above one."""

        def __init__(self, name, degree=None):
            super().__init__(name)
            self.degree = degree

        def residual(self):
            return ufl.exp(c) * c.v * domain.dx(degree=self.degree)

    def residual(model, **options):
        disc = discretize([model], DomainMaps({domain: square()}), **options)
        disc.set({c: lambda x: x[0] + 2 * x[1]})
        problem = disc[model]
        return dolfinx.fem.assemble_vector(problem.residual).array.copy()

    estimated = residual(Exponential("e"))
    bounded = residual(Exponential("e"), max_quadrature_degree=1)
    explicit = residual(Exponential("e", degree=1))
    np.testing.assert_allclose(bounded, explicit, rtol=1e-14)
    assert not np.allclose(bounded, estimated, rtol=1e-6)
    # An explicit degree is kept under a bound
    kept = residual(Exponential("e", degree=3), max_quadrature_degree=1)
    np.testing.assert_allclose(kept, residual(Exponential("e", degree=3)), rtol=1e-14)
    with pytest.raises(ValueError, match="non-negative"):
        discretize([], DomainMaps({domain: square()}), max_quadrature_degree=-1)


def test_meshes_are_given_by_domain_or_name(square):
    domain = Domain("cell", "triangle")
    marked = square()
    by_domain = DomainMaps({domain: marked})
    by_name = DomainMaps({"cell": marked})
    assert by_domain.mesh(domain) is by_name.mesh(domain) is marked.mesh
    with pytest.raises(ValueError, match="given twice"):
        DomainMaps({domain: marked, "cell": marked})
    # A mesh given for a domain is checked at once, one given by name on use
    quadrilateral = Domain("cell", "quadrilateral")
    with pytest.raises(ValueError, match="coordinate element"):
        DomainMaps({quadrilateral: marked})
    with pytest.raises(ValueError, match="coordinate element"):
        by_name.mesh(quadrilateral)


def test_assemble_scalar_integrates_over_markers(square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    disc = discretize([c], DomainMaps({"cell": square()}))
    disc.set({c: lambda x: x[0]})
    left = 1.0 * domain.ds("Left")
    mean = c * domain.dx()
    assert disc.assemble_scalar(left) == pytest.approx(1.0)
    assert disc.assemble_scalar(mean) == pytest.approx(0.5)
    # The discretized functional follows the current values
    disc.set({c: 2.0})
    assert disc.assemble_scalar(mean) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="test or trial"):
        disc.assemble_scalar(c.v * domain.dx())


def test_unmarked_boundary_facets_are_found(square, sides):
    mesh_data = square()
    assert len(mesh_data.unmarked_boundary_facets(sides)) == 0
    unmarked = mesh_data.unmarked_boundary_facets(sides[1:])
    np.testing.assert_array_equal(unmarked, mesh_data.entities(1, sides[0]))


def test_symbolic_values_are_checked(square):
    domain = Domain("cell", "triangle")
    a = Field("a", domain, ("Lagrange", 1))
    k = Parameter("k", domain, default=1.0)
    x = ufl.SpatialCoordinate(domain)
    disc = discretize([a, k], DomainMaps({"cell": square()}))
    with pytest.raises(ValueError, match="shape"):
        disc.set({a: x})
    with pytest.raises(TypeError, match="number or an array"):
        disc.set({k: x[0]})


def test_a_discretization_holds_what_is_declared(toy, square, sides):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    other = Field("other", domain, ("Lagrange", 1))
    model = toy.Diffusion("d", c)
    disc = discretize([model], DomainMaps({"cell": square()}))
    assert c in disc and "d.k" in disc.parameters and other not in disc
    with pytest.raises(KeyError, match="not held"):
        disc.set({other: 1.0})
    with pytest.raises(KeyError, match="not held"):
        disc[toy.Diffusion("e", other)]
    # A later addition keeps what is held
    function = disc.function(c)
    disc.add(other)
    assert disc.function(c) is function and other in disc
    with pytest.raises(TypeError, match="Cannot declare"):
        disc.add("model")


def test_indexing_gives_kept_discrete_counterparts(toy, square, sides):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=1.0)
    disc = discretize(model, DomainMaps({"cell": square()}))
    system = disc[model]
    assert disc[model] is system and system.symbolic.unknowns == (c,)
    assert disc[c] is disc["c"] is system.solution[0]
    functional = c * domain.dx()
    assert disc[functional] is disc[functional]
    # Options give another counterpart, not kept
    assert counterpart(model, disc, field_order=[c]) is not system


def test_discretizations_on_the_same_maps_share_submeshes(toy):
    omega = Domain("omega", "tetrahedron")
    gamma = Subdomain("gamma", omega, 2, "Bottom")
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    bottom = {2: {"Bottom": lambda x: np.isclose(x[2], 0.0)}}
    maps = DomainMaps({omega: MeshData.from_locators(mesh, bottom, boundary=True)})
    s = Field("s", gamma, ("Lagrange", 1))
    first = discretize(toy.Diffusion("d", s), maps)
    second = discretize(s, maps)
    assert first.function(s) is not second.function(s)
    assert first.function(s).function_space.mesh is maps.mesh(gamma)
    assert second.function(s).function_space.mesh is maps.mesh(gamma)
