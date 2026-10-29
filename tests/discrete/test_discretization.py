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
from sumfem.post_processing import error, norm
from sumfem.solvers import Solver
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    Subdomain,
    SystemBuildError,
    UnitError,
    build_system,
)


def quadratic(x):
    """1 + x^2 + 2 y^2, with Laplacian 6, exact in degree-2 spaces."""
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


def integral(function):
    mesh = function.function_space.mesh
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(function * ufl.dx))
    return mesh.comm.allreduce(local, op=MPI.SUM)


@pytest.mark.parametrize("cell", ["triangle", "quadrilateral"])
def test_poisson_from_a_sum_of_models_is_exact(toy, square, sides, cell):
    domain = Domain("cell", cell)
    c = Field("c", domain, ("Lagrange", 2))
    model = toy.Diffusion("diffusion", c) + toy.Source("source", c, f=-6.0)
    disc = discretize([model], DomainMaps({"cell": square(cell)}))
    solver = Solver(disc[model], [DirichletBC(c, sides, quadratic)])
    (solution,) = solver.solve()
    assert solution is disc.fields["c"]
    assert error(solution, quadratic) < 1e-10


def test_fields_with_the_same_element_keep_their_own_conditions(toy, square, sides):
    # Both fields share element and mesh. Their conditions must not mix.
    domain = Domain("cell", "triangle")
    a = Field("a", domain, ("Lagrange", 2))
    b = Field("b", domain, ("Lagrange", 2))
    model = (
        toy.Diffusion("da", a)
        + toy.Source("sa", a, f=-6.0)
        + toy.Diffusion("db", b)
        + toy.Source("sb", b, f=-12.0)
    )
    disc = discretize([model], DomainMaps({"cell": square()}))
    bcs = [
        DirichletBC(a, sides, quadratic),
        DirichletBC(b, sides, lambda x: 2 * quadratic(x)),
    ]
    sol_a, sol_b = Solver(disc[model], bcs).solve()
    assert disc.function_space(a) is not disc.function_space(b)
    assert error(sol_a, quadratic) < 1e-10
    assert error(sol_b, lambda x: 2 * quadratic(x)) < 1e-10


def test_fields_on_disjoint_submeshes(toy):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = mesh.topology.dim
    left_cells = dolfinx.mesh.locate_entities(mesh, tdim, lambda x: x[0] <= 0.5 + 1e-10)
    right_cells = dolfinx.mesh.locate_entities(
        mesh, tdim, lambda x: x[0] >= 0.5 - 1e-10
    )
    meshes = {}
    for name, cells in (("left", left_cells), ("right", right_cells)):
        submesh = dolfinx.mesh.create_submesh(mesh, tdim, cells)[0]
        meshes[name] = MeshData.from_locators(
            submesh,
            {tdim - 1: {"Boundary": lambda x: np.full(x.shape[1], True)}},
            boundary=True,
        )
    left, right = Domain("left", "triangle"), Domain("right", "triangle")
    a = Field("a", left, ("Lagrange", 2))
    b = Field("b", right, ("Lagrange", 3))
    model = (
        toy.Diffusion("da", a)
        + toy.Source("sa", a, f=-6.0)
        + toy.Diffusion("db", b)
        + toy.Source("sb", b, f=-6.0)
    )
    disc = discretize([model], DomainMaps(meshes))
    bcs = [DirichletBC(a, "Boundary", quadratic), DirichletBC(b, "Boundary", quadratic)]
    sol_a, sol_b = Solver(disc[model], bcs).solve()
    assert error(sol_a, quadratic) < 1e-10
    assert error(sol_b, quadratic) < 1e-10


def test_parameters_change_without_recompiling(toy, square, sides, monkeypatch):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("diffusion", c) + toy.Source("source", c, f=1.0)
    system = build_system(model)
    disc = discretize([system], DomainMaps({"cell": square()}))
    solver = Solver(disc[system], [DirichletBC(c, sides, 0.0)])
    (solution,) = solver.solve()
    unit_k = solution.x.array.copy()
    disc.set({"diffusion.k": 2.0})
    solver.solve()
    np.testing.assert_allclose(solution.x.array, unit_k / 2, atol=1e-13)
    # The parameter is a Real-space function, the same on every process
    values = solution.function_space.mesh.comm.allgather(disc.value("diffusion.k"))
    assert all(v == 2.0 for v in values)

    # Binding the system again, even to another discretization, reuses the
    # compiled forms
    compiled = []
    original = dolfinx.fem.compile_form
    monkeypatch.setattr(
        dolfinx.fem,
        "compile_form",
        lambda *a, **k: compiled.append(1) or original(*a, **k),
    )
    (discretize([system], DomainMaps({"cell": square()})))[system]
    assert compiled == []


def test_parameters_without_value_are_reported(toy, square, sides):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    f = Parameter("f", domain)
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=f)
    disc = discretize([model], DomainMaps({"cell": square()}))
    solver = Solver(disc[model], [DirichletBC(c, sides, 0.0)])
    assert not disc.has_value(f)
    with pytest.raises(ValueError, match=r"\['f'\] have no value"):
        solver.solve()
    disc.set({f: 1.0})
    solver.solve()


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


def test_marked_integrals_use_the_named_cells(square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("DG", 0))

    class IndicatorProjection(Residual):
        def residual(self):
            return c * c.v * domain.dx() - c.v * domain.dx("LeftHalf")

    projection = IndicatorProjection("p")
    disc = discretize([projection], DomainMaps({"cell": square()}))
    (solution,) = Solver(disc[projection]).solve()
    assert integral(solution) == pytest.approx(0.5)


def test_markers_from_gmsh_physical_groups(toy):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if MPI.COMM_WORLD.rank == 0:
            gmsh.model.add("rectangle")
            surface = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
            gmsh.model.occ.synchronize()
            left = [
                tag
                for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False)
                if np.isclose(gmsh.model.occ.getCenterOfMass(dim, tag)[0], 0.0)
            ]
            gmsh.model.addPhysicalGroup(2, [surface], name="Domain")
            gmsh.model.addPhysicalGroup(1, left, name="Inlet")
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
            gmsh.model.mesh.generate(2)
        data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    finally:
        gmsh.finalize()
    marked = MeshData.from_gmsh(data)
    assert set(marked.markers[1]) == {"Inlet"} and set(marked.markers[2]) == {"Domain"}

    # gmsh meshes use equispaced coordinate elements, unlike the domain default
    with pytest.raises(ValueError, match="lagrange_variant"):
        DomainMaps({"cell": marked}).mesh(Domain("cell", "triangle"))
    domain = Domain.from_mesh("cell", marked.mesh)
    equispaced = Domain("other", "triangle", lagrange_variant="equispaced")
    assert domain.ufl_coordinate_element() == equispaced.ufl_coordinate_element()
    c = Field("c", domain, ("Lagrange", 1))
    diffusion = toy.Diffusion("d", c)
    disc = discretize([diffusion], DomainMaps({domain: marked}))
    # c = 1 on the inlet with no flux elsewhere gives c = 1 everywhere
    (solution,) = Solver(disc[diffusion], [DirichletBC(c, "Inlet", 1.0)]).solve()
    np.testing.assert_allclose(solution.x.array, 1.0, atol=1e-10)


def test_runtime_boundary_values_follow_parameters(toy, square, sides):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 2))
    g = Parameter("g", domain, default=1.0)
    x = ufl.SpatialCoordinate(domain)
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=-6.0 * g)
    disc = discretize([model], DomainMaps({"cell": square()}))
    solver = Solver(disc[model], [DirichletBC(c, sides, g * quadratic(x))])
    (solution,) = solver.solve()
    assert error(solution, quadratic) < 1e-10
    # The boundary value is evaluated again before the next solve
    disc.set({g: 3.0})
    solver.solve()
    assert error(solution, lambda x: 3 * quadratic(x)) < 1e-10


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


def test_boundary_condition_errors(toy, square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    known = Field("known", domain, ("Lagrange", 1))
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=known)
    disc = discretize([model], DomainMaps({"cell": square()}))
    problem = disc[model]
    with pytest.raises(KeyError, match="no marker 'Nowhere'"):
        Solver(problem, [DirichletBC(c, "Nowhere", 0.0)])
    with pytest.raises(ValueError, match="not on a field this system solves for"):
        Solver(problem, [DirichletBC(known, "Left", 0.0)])


def test_factorized_solve_matches_the_dimensional_solve(square, sides):
    # Concentrations of order 1e3 mol/m^3 diffusing with D = 1e-9 m^2/s give
    # residual entries of order 1e-6, far from one
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 2), unit=syu.mol / syu.m**3, scale=1e3)
    c0 = 1e3

    class Diffusion(Residual):
        def residual(self):
            D = self.parameter("D", domain, default=1e-9, unit=syu.m**2 / syu.s)
            r = self.parameter(
                "r", domain, default=-6e-9 * c0, unit=syu.mol / (syu.m**3 * syu.s)
            )
            dx = domain.dx()
            return ufl.inner(D * ufl.grad(c), ufl.grad(c.v)) * dx - r * c.v * dx

    def exact(x):
        return c0 * quadratic(x)

    solutions = {}
    for factorize in (False, True):
        model = Diffusion("d")
        disc = discretize([model], DomainMaps({"cell": square()}))
        problem = counterpart(model, disc, factorize_equations=factorize)
        (solution,) = Solver(problem, [DirichletBC(c, sides, exact)]).solve()
        assert solution is disc.fields["c"]
        assert set(disc.fields) == {"c"}
        assert error(solution, exact) < 1e-10 * c0
        solutions[factorize] = solution.x.array.copy()
    np.testing.assert_allclose(solutions[True], solutions[False], rtol=1e-10)
    # Only the equation is scaled, so c is the unknown in both solves
    assert problem.symbolic.unknowns == (c,)
    assert set(problem.symbolic.equation_scales) == {"c"}


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


def test_fields_are_set_from_symbolic_expressions(square):
    """An initial condition from coordinates, parameters and other fields."""
    domain = Domain("cell", "triangle")
    a = Field("a", domain, ("Lagrange", 2))
    b = Field("b", domain, ("Lagrange", 2, (2,)))
    k = Parameter("k", domain, default=3.0)
    x = ufl.SpatialCoordinate(domain)
    disc = discretize([a, b, k], DomainMaps({"cell": square()}))
    disc.set({a: k * x[0] ** 2})
    disc.set({b: ufl.as_vector((a, x[1]))})
    mesh = disc.function(a).function_space.mesh
    X = ufl.SpatialCoordinate(mesh)
    assert norm(disc.function(b) - ufl.as_vector((3.0 * X[0] ** 2, X[1]))) < 1e-12


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
