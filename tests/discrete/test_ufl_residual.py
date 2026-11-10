"""A model in plain UFL and from components discretizes to the same system.

Each test writes a model as the UFL file of a DOLFINx C++ demo would, with
plain UFL objects, turns it into a ``UFLResidual``, and compares it with the
same model built from components on fields of their own. Both are discretized on
the same mesh data, so their fields have the same layout and the data of one can
be copied to the other: the assembled residual, Jacobian and preconditioner agree
to machine precision, and so do the solutions.
"""

import numpy as np
import pytest
import ufl
from basix.ufl import element

from sumfem.discrete import Discretization, DomainMaps, discretize
from sumfem.solvers import Solver
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Functional,
    Jacobian,
    Residual,
    UFLResidual,
)

CELLS = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]
"""The cell types each comparison runs on."""

DOMAIN = "box"
"""The name of the symbolic domain of both formulations.

The plain forms name their mesh so with ``names``, the components create their
domain with it, and ``DomainMaps`` binds the mesh data to the domain by it.
"""


def source(x: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """A source that is not constant, the same in both formulations."""
    y = x[1] if x.ufl_shape[0] > 1 else x[0]
    return 1 + x[0] ** 2 + ufl.sin(3 * y)


def diffusion(
    k: ufl.core.expr.Expr,
    u: ufl.core.expr.Expr,
    v: ufl.core.expr.Expr,
    dx: ufl.Measure,
) -> ufl.Form:
    """``(k grad u, grad v)``, written once for both formulations.

    The plain form calls it with the plain coefficient, test function and
    measure, the component with its field, ``field.v`` and the domain's ``dx``,
    so the two can only differ by what ``UFLResidual`` does with the form.
    """
    return ufl.inner(k * ufl.grad(u), ufl.grad(v)) * dx


def assign_random_data(disc: Discretization, field: Field) -> None:
    """Set the field to random values, so that nonlinear terms are exercised."""
    function = disc.function(field)
    rng = np.random.default_rng(seed=disc.comm.rank)
    function.x.array[:] = rng.uniform(-1.0, 1.0, function.x.array.size)
    function.x.scatter_forward()


@pytest.mark.parametrize("cell", CELLS)
def test_poisson_matches_its_components(
    ufl_functionspace, cell, toy, box, same_assembly
):
    """A ufl.Constant of the form becomes the parameter "plain.kappa", set by name.
    The residual, Jacobian and solution agree with the model from components.
    """
    mesh, V = ufl_functionspace(cell, degree=2)
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    kappa = ufl.Constant(mesh)
    f = source(ufl.SpatialCoordinate(mesh))
    F = kappa * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    plain = UFLResidual("plain", F, uh, names={DOMAIN: mesh, "kappa": kappa})

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 2))
    components = toy.Diffusion("d", u) + toy.Source(
        "s", u, f=source(ufl.SpatialCoordinate(domain))
    )

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    # The sides the box marks: two on an interval, four otherwise
    sides = list(mesh_data.markers[mesh_data.mesh.topology.dim - 1])
    disc_plain.set({"plain.kappa": 2.5})
    disc_components.set({"d.k": 2.5})
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    same_assembly(disc_components[components], disc_plain[plain])

    plain_u.x.array[:] = 0.0
    components_u.x.array[:] = 0.0
    Solver(disc_plain[plain], [DirichletBC(plain.field, sides, 0.0)]).solve()
    Solver(disc_components[components], [DirichletBC(u, sides, 0.0)]).solve()
    np.testing.assert_allclose(plain_u.x.array, components_u.x.array, atol=1e-12)


@pytest.mark.parametrize("cell", CELLS)
def test_a_linear_problem_is_given_as_an_action(
    ufl_functionspace, cell, toy, box, same_assembly
):
    """A linear problem given as its action, with the bilinear form as Jacobian.
    The given Jacobian is used as written and matches the derived one.
    """
    mesh, V = ufl_functionspace(cell)
    uh, v, du = ufl.Coefficient(V), ufl.TestFunction(V), ufl.TrialFunction(V)
    a = ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx
    L = 1.0 * v * ufl.dx
    plain = UFLResidual("plain", ufl.action(a, uh) - L, uh, a, names={DOMAIN: mesh})

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 1))
    components = toy.Diffusion("d", u) + toy.Source("s", u, f=1.0)

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    same_assembly(disc_components[components], disc_plain[plain])


class NonlinearDiffusion(Residual):
    """The :func:`diffusion` of a field with ``k = 1 + u^2``, as a component."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u = self.field
        return diffusion(1 + u**2, u, u.v, u.domain.dx())


@pytest.mark.parametrize("given", [False, True], ids=["derived", "given"])
@pytest.mark.parametrize("cell", CELLS)
def test_a_nonlinear_jacobian_matches(
    ufl_functionspace, cell, given, toy, box, same_assembly
):
    """The Jacobian of a nonlinear residual, derived by sumfem or given in UFL.
    Both match the components' derived Jacobian, and so do the solutions.
    """
    mesh, V = ufl_functionspace(cell)
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    F = diffusion(1 + uh**2, uh, v, ufl.dx) - v * ufl.dx
    J = ufl.derivative(F, uh, ufl.TrialFunction(V)) if given else None
    plain = UFLResidual("plain", F, uh, J, names={DOMAIN: mesh})

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 1))
    components = NonlinearDiffusion("n", u) + toy.Source("s", u, f=1.0)

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    # The sides the box marks: two on an interval, four otherwise
    sides = list(mesh_data.markers[mesh_data.mesh.topology.dim - 1])
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    same_assembly(disc_components[components], disc_plain[plain])

    plain_u.x.array[:] = 0.0
    components_u.x.array[:] = 0.0
    Solver(disc_plain[plain], [DirichletBC(plain.field, sides, 0.0)]).solve()
    Solver(disc_components[components], [DirichletBC(u, sides, 0.0)]).solve()
    np.testing.assert_allclose(plain_u.x.array, components_u.x.array, atol=1e-12)


class ConstantLaplacian(Jacobian):
    """The :func:`diffusion` of ``du`` with ``k = 1``, as a preconditioner."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def jacobian(self):
        u = self.field
        return diffusion(1.0, u.du, u.v, u.domain.dx())


@pytest.mark.parametrize("cell", CELLS)
def test_a_preconditioner_matches(ufl_functionspace, cell, toy, box, same_assembly):
    """A preconditioner given in plain UFL, against with_preconditioner.
    Both assemble alike and give the same solution with a Krylov solver.
    """
    mesh, V = ufl_functionspace(cell)
    uh, v, du = ufl.Coefficient(V), ufl.TestFunction(V), ufl.TrialFunction(V)
    xh = ufl.SpatialCoordinate(mesh)
    F = diffusion(1 + 100 * xh[0] ** 2, uh, v, ufl.dx) - v * ufl.dx
    P = diffusion(1.0, du, v, ufl.dx)
    plain = UFLResidual("plain", F, uh, preconditioner=P, names={DOMAIN: mesh})

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)

    class VariableDiffusion(Residual):
        def residual(self):
            return diffusion(1 + 100 * x[0] ** 2, u, u.v, domain.dx())

    components = (
        VariableDiffusion("vd") + toy.Source("s", u, f=1.0)
    ).with_preconditioner(ConstantLaplacian("p", u))

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    # The sides the box marks: two on an interval, four otherwise
    sides = list(mesh_data.markers[mesh_data.mesh.topology.dim - 1])
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    same_assembly(disc_components[components], disc_plain[plain])

    # CG on the Jacobian, preconditioned by a factorization of the Laplacian. A
    # direct solve would factorize the preconditioner instead of the Jacobian.
    options = {"ksp_type": "cg", "pc_type": "lu", "ksp_rtol": 1e-12}
    plain_u.x.array[:] = 0.0
    components_u.x.array[:] = 0.0
    Solver(
        disc_plain[plain], [DirichletBC(plain.field, sides, 0.0)], petsc_options=options
    ).solve()
    Solver(
        disc_components[components], [DirichletBC(u, sides, 0.0)], petsc_options=options
    ).solve()
    np.testing.assert_allclose(plain_u.x.array, components_u.x.array, atol=1e-10)


class DirichletEnergy(Functional):
    """``1/2 |grad u|^2 - f u``, whose stationary point solves Poisson's equation."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def functional(self):
        u = self.field
        f = source(ufl.SpatialCoordinate(u.domain))
        return (0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) - f * u) * u.domain.dx()


@pytest.mark.parametrize("cell", CELLS)
def test_the_variation_of_an_energy_matches(
    ufl_functionspace, cell, box, same_assembly
):
    """A residual given as the derivative of a plain UFL energy.
    It matches Functional.to_residual of the same energy as a component.
    """
    mesh, V = ufl_functionspace(cell)
    uh = ufl.Coefficient(V)
    f = source(ufl.SpatialCoordinate(mesh))
    E = (0.5 * ufl.inner(ufl.grad(uh), ufl.grad(uh)) - f * uh) * ufl.dx
    F = ufl.derivative(E, uh, ufl.TestFunction(V))
    plain = UFLResidual("plain", F, uh, names={DOMAIN: mesh})

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 1))
    components = DirichletEnergy("energy", u).to_residual()

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    same_assembly(disc_components[components], disc_plain[plain])


@pytest.mark.parametrize("cell", CELLS)
def test_marked_boundary_integrals_match(
    ufl_functionspace, cell, toy, box, same_assembly
):
    """Integer subdomain ids are mapped to the marker names given in markers.
    The integrals are over the same facets as the components' named ones.
    """
    # Different fluxes on the two sides, so that swapped markers would differ
    flux_left, flux_right = 2.0, 3.0

    mesh, V = ufl_functionspace(cell)
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", mesh)
    F = (
        ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        - flux_left * v * ds(1)
        - flux_right * v * ds(2)
    )
    plain = UFLResidual(
        "plain", F, uh, names={DOMAIN: mesh}, markers={"ds": {1: "Left", 2: "Right"}}
    )

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 1))

    class Flux(Residual):
        def residual(self):
            return -flux_left * u.v * domain.ds("Left") - flux_right * u.v * domain.ds(
                "Right"
            )

    components = toy.Diffusion("d", u) + Flux("flux")

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    same_assembly(disc_components[components], disc_plain[plain])


def test_different_systems_are_told_apart(ufl_functionspace, toy, box, same_assembly):
    """same_assembly fails for systems that differ, here in a parameter value.
    It guards the comparison that the other tests rely on.
    """
    cell = "triangle"
    mesh, V = ufl_functionspace(cell)
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    kappa = ufl.Constant(mesh)
    F = kappa * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - v * ufl.dx
    plain = UFLResidual("plain", F, uh, names={DOMAIN: mesh, "kappa": kappa})

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 1))
    components = toy.Diffusion("d", u) + toy.Source("s", u, f=1.0)

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    plain_u, components_u = (
        disc_plain.function(plain.field),
        disc_components.function(u),
    )
    disc_plain.set({"plain.kappa": 2.6})
    disc_components.set({"d.k": 2.5})
    assign_random_data(disc_plain, plain.field)
    components_u.x.array[:] = plain_u.x.array
    with pytest.raises(AssertionError, match="residuals differ"):
        same_assembly(disc_components[components], disc_plain[plain])


class Momentum(Residual):
    """``(grad u, grad v) + ((grad u) u, v) - (p, div v)``."""

    def __init__(self, name, velocity, pressure):
        super().__init__(name)
        self.velocity, self.pressure = velocity, pressure

    def residual(self):
        u, p, v = self.velocity, self.pressure, self.velocity.v
        convection = ufl.inner(ufl.grad(u) * u, v)
        return (ufl.inner(ufl.grad(u), ufl.grad(v)) + convection - p * ufl.div(v)) * (
            u.domain.dx()
        )


class Continuity(Residual):
    """``-(div u, q)``."""

    def __init__(self, name, velocity, pressure):
        super().__init__(name)
        self.velocity, self.pressure = velocity, pressure

    def residual(self):
        u, q = self.velocity, self.pressure.v
        return -ufl.div(u) * q * u.domain.dx()


class StokesPreconditioner(Jacobian):
    """``(grad du, grad v) + (dp, q)``: the Laplacian and the pressure mass matrix."""

    def __init__(self, name, velocity, pressure):
        super().__init__(name)
        self.velocity, self.pressure = velocity, pressure

    def jacobian(self):
        u, p = self.velocity, self.pressure
        return (ufl.inner(ufl.grad(u.du), ufl.grad(u.v)) + p.du * p.v) * u.domain.dx()


@pytest.mark.parametrize("variant", ["derived", "given", "preconditioner"])
@pytest.mark.parametrize("cell", CELLS)
def test_a_mixed_space_matches_its_components(
    ufl_functionspace, cell, variant, box, same_assembly
):
    """Two unknowns on a ufl.MixedFunctionSpace become two fields, matched by part.
    The block residual, Jacobian and preconditioner match those of components.
    """
    # Navier-Stokes in plain UFL: one form for both equations, tested with the
    # test functions of the two sub-spaces. The Jacobian is derived by sumfem,
    # or given in plain UFL, and the preconditioner is given for one variant.
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell, degree=2, shape=(gdim,))
    Q = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, Q)
    uh, ph = ufl.Coefficient(V), ufl.Coefficient(Q)
    v, q = ufl.TestFunctions(W)
    du, dp = ufl.TrialFunctions(W)
    F = (
        ufl.inner(ufl.grad(uh), ufl.grad(v))
        + ufl.inner(ufl.grad(uh) * uh, v)
        - ph * ufl.div(v)
        - ufl.div(uh) * q
    ) * ufl.dx
    J = ufl.derivative(F, uh, du) + ufl.derivative(F, ph, dp)
    P = (ufl.inner(ufl.grad(du), ufl.grad(v)) + dp * q) * ufl.dx
    plain = UFLResidual(
        "plain",
        F,
        [uh, ph],
        J if variant == "given" else None,
        P if variant == "preconditioner" else None,
        names={DOMAIN: mesh, "u": uh, "p": ph},
    )

    # The same equations as two components, one per test function, on fields
    # of their own. Their Jacobian is always derived.
    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 2, (gdim,)))
    p = Field("p", domain, ("Lagrange", 1))
    components = Momentum("m", u, p) + Continuity("c", u, p)
    if variant == "preconditioner":
        components = components.with_preconditioner(StokesPreconditioner("s", u, p))

    # The same mesh data for both, so the fields have the same layout
    mesh_data = box(cell)
    disc_plain = discretize([plain], DomainMaps({DOMAIN: mesh_data}))
    disc_components = discretize([components], DomainMaps({DOMAIN: mesh_data}))
    # Both at the same random velocity and pressure, nonzero so that the
    # convection and its derivative contribute. The unknowns of the plain form
    # are in the order of the sub-spaces, as (u, p).
    for plain_field, field in zip(plain.unknowns, (u, p), strict=True):
        assign_random_data(disc_plain, plain_field)
        values = disc_plain.function(plain_field).x.array
        disc_components.function(field).x.array[:] = values
    # same_assembly assembles the residual, Jacobian and preconditioner of both
    # systems, blocked by field, and compares them: a test or trial function
    # mapped to the wrong field would put its terms in another block
    same_assembly(disc_components[components], disc_plain[plain])
