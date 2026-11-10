"""Residuals from plain UFL forms: naming, rebinding and errors."""

import typing

import pytest
import ufl
from basix.ufl import element

from sumfem.error_handling import ComponentError
from sumfem.symbolic import Domain, Field, UFLResidual

CELLS = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]
"""The cell types of the tests that create domains and fields."""

UFLSpaceFactory = typing.Callable[..., tuple[ufl.Mesh, ufl.FunctionSpace]]
"""The ``ufl_functionspace`` fixture: a plain UFL mesh and space for a cell."""

DOMAIN = "box"
"""The name of the plain mesh in ``names``, and so of the domain created for it."""


def poisson_form(
    ufl_functionspace: UFLSpaceFactory, cell: str = "triangle"
) -> tuple[ufl.Form, dict[str, typing.Any]]:
    """``kappa (grad u, grad v) - (f, v)`` in plain UFL, and its variable names."""
    mesh, V = ufl_functionspace(cell)
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    kappa, f = ufl.Constant(mesh), ufl.Coefficient(V)
    F = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    return F, {DOMAIN: mesh, "u": u, "kappa": kappa, "f": f}


@pytest.mark.parametrize("cell", CELLS)
def test_terminals_become_named_sumfem_terminals(ufl_functionspace, cell):
    """The unknown, constant, coefficient and mesh become a field, a parameter, a
    known coefficient and a domain of the mesh's cell type, named by ``names``.
    """
    F, names = poisson_form(ufl_functionspace, cell)
    residual = UFLResidual("poisson", F, names["u"], names=names)
    assert isinstance(residual.field, Field)
    assert residual.field.name == "u"
    assert residual.field.domain.name == DOMAIN
    assert residual.field.domain.cell_name == cell
    assert [f.name for f in residual.equations] == ["u"]
    assert list(residual.parameters) == ["poisson.kappa"]
    assert list(residual.known_coefficients) == ["poisson.f"]
    assert residual.form.arguments() == (residual.field.v,)


@pytest.mark.parametrize("cell", CELLS)
def test_a_given_domain_and_field_are_used(ufl_functionspace, cell):
    """A field given for the unknown in ``terminals`` is used as it is.
    Its domain becomes the domain of the plain mesh, instead of a new one.
    """
    F, names = poisson_form(ufl_functionspace, cell)
    domain = Domain(DOMAIN, cell)
    u = Field("w", domain, ("Lagrange", 1))
    residual = UFLResidual(
        "poisson", F, names["u"], names=names, terminals={names["u"]: u}
    )
    assert residual.field is u
    assert residual.domains == (domain,)


def test_given_forms_are_rebound_to_the_field(ufl_functionspace):
    """A given Jacobian and preconditioner are rebound like the residual.
    Their test and trial functions become the field's ``v`` and ``du``.
    """
    mesh, V = ufl_functionspace()
    u, v, du = ufl.Coefficient(V), ufl.TestFunction(V), ufl.TrialFunction(V)
    F = (1 + u**2) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    J = ufl.derivative(F, u, du)
    P = ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx
    residual = UFLResidual("nonlinear", F, u, J, P)
    field = residual.field
    assert residual.jacobian().arguments() == (field.v, field.du)
    assert residual.preconditioner().arguments() == (field.v, field.du)
    assert field in residual.jacobian().coefficients()


@pytest.mark.parametrize("cell", CELLS)
def test_marked_integrals_get_marker_names(ufl_functionspace, cell):
    """Integer subdomain ids become the domain's ids of their marker names.
    Each id needs a name, and a name of its own.
    """
    mesh, V = ufl_functionspace(cell)
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", mesh)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - v * ds((1, 2))
    residual = UFLResidual("flux", F, u, markers={"ds": {1: "Left", 2: "Right"}})
    domain = residual.field.domain
    # UFL splits ds((1, 2)) into an integral per id
    ids = [
        i.subdomain_id()
        for i in residual.form.integrals()
        if i.integral_type() == "exterior_facet"
    ]
    assert ids == [domain.marker_ids([n], "facet") for n in ("Left", "Right")]
    with pytest.raises(ComponentError, match="no marker name"):
        UFLResidual("flux", F, u, markers={"ds": {1: "Left"}})
    # A shared name would integrate each integral over both ids
    with pytest.raises(ComponentError, match=r"ids \[1, 2\] of ds .* share"):
        UFLResidual("flux", F, u, markers={"ds": {1: "Side", 2: "Side"}})


@pytest.mark.parametrize("cell", CELLS)
def test_earlier_steps_become_the_history_of_the_field(ufl_functionspace, cell):
    """A coefficient given in ``previous`` becomes the field one step back.
    The time step, a constant, becomes a parameter as any other.
    """
    mesh, V = ufl_functionspace(cell)
    u, u_n, v = ufl.Coefficient(V), ufl.Coefficient(V), ufl.TestFunction(V)
    dt = ufl.Constant(mesh)
    F = (u - u_n) * v * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    residual = UFLResidual("heat", F, u, names={"dt": dt}, previous=(u_n,))
    assert residual.field.previous(1) in residual.form.coefficients()
    assert list(residual.parameters) == ["heat.dt"]


def test_forms_on_other_spaces_are_rejected(ufl_functionspace):
    """An unknown that is no coefficient, and forms with other arguments than
    those of the unknown's space, or of the wrong arity, are rejected.
    """
    F, names = poisson_form(ufl_functionspace)
    u = names["u"]
    _, W = ufl_functionspace(degree=2)
    w, q = ufl.TrialFunction(W), ufl.TestFunction(W)
    with pytest.raises(ComponentError, match="must be a ufl.Coefficient"):
        UFLResidual("bad", F, names["kappa"], names=names)
    with pytest.raises(ComponentError, match="linear in a test function"):
        UFLResidual("bad", ufl.inner(w, q) * ufl.dx, u, names=names)
    with pytest.raises(ComponentError, match="not on the space of the unknown"):
        UFLResidual("bad", q * ufl.dx, u, names=names)
    with pytest.raises(ComponentError, match="not on the space of the unknown"):
        UFLResidual("bad", F, u, ufl.inner(w, q) * ufl.dx, names=names)
    with pytest.raises(ComponentError, match="The jacobian of 'bad' must be bilinear"):
        UFLResidual("bad", F, u, F, names=names)
    with pytest.raises(ComponentError, match="The preconditioner of 'bad'"):
        UFLResidual("bad", F, u, preconditioner=F, names=names)


def test_unnamed_and_mismatched_terminals_are_rejected(ufl_functionspace):
    """A coefficient without a name, and replacements and earlier steps that do
    not fit what they replace, are rejected.
    """
    F, names = poisson_form(ufl_functionspace)
    u, kappa = names["u"], names["kappa"]
    with pytest.raises(ComponentError, match="has no name"):
        UFLResidual("poisson", F, u)
    domain = Domain(DOMAIN, "triangle")
    vector = Field("w", domain, ("Lagrange", 1, (2,)))
    with pytest.raises(ComponentError, match="is an unknown: it must be given"):
        UFLResidual("poisson", F, u, names=names, terminals={u: vector})
    with pytest.raises(ComponentError, match="has the shape"):
        UFLResidual("poisson", F, u, names=names, terminals={kappa: vector.v})
    _, W = ufl_functionspace(degree=2)
    with pytest.raises(ComponentError, match="another element than its unknown"):
        UFLResidual("poisson", F, u, names=names, previous=(ufl.Coefficient(W),))


def stokes_form(
    ufl_functionspace: UFLSpaceFactory, cell: str = "triangle"
) -> tuple[ufl.Form, ufl.MixedFunctionSpace, ufl.Coefficient, ufl.Coefficient]:
    """Stokes flow on a ufl.MixedFunctionSpace, as a DOLFINx C++ demo has it."""
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell, degree=2, shape=(gdim,))
    Q = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, Q)
    u, p = ufl.Coefficient(V), ufl.Coefficient(Q)
    v, q = ufl.TestFunctions(W)
    F = (ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - ufl.div(u) * q) * ufl.dx
    return F, W, u, p


@pytest.mark.parametrize("cell", CELLS)
def test_a_mixed_space_gives_a_field_per_unknown(ufl_functionspace, cell):
    """Each unknown of a mixed space becomes a field, and each test and trial
    function the ``v`` or ``du`` of the field of its sub-space.
    """
    F, W, u, p = stokes_form(ufl_functionspace, cell)
    v, q = ufl.TestFunctions(W)
    du, dp = ufl.TrialFunctions(W)
    P = (ufl.inner(ufl.grad(du), ufl.grad(v)) + dp * q) * ufl.dx
    residual = UFLResidual(
        "stokes", F, [u, p], preconditioner=P, names={"u": u, "p": p}
    )
    velocity, pressure = residual.unknowns
    assert (velocity.name, pressure.name) == ("u", "p")
    assert residual.equations == (velocity, pressure)
    assert set(residual.form.arguments()) == {velocity.v, pressure.v}
    assert set(residual.preconditioner_term.form.arguments()) == {
        velocity.v,
        velocity.du,
        pressure.v,
        pressure.du,
    }
    with pytest.raises(ComponentError, match="has 2 unknowns"):
        residual.field
    unnamed = UFLResidual("stokes", F, [u, p])
    assert [field.name for field in unnamed.unknowns] == ["u0", "u1"]


def test_unknowns_must_match_the_sub_spaces(ufl_functionspace):
    """Unknowns in another order than the sub-spaces, fewer unknowns than
    sub-spaces, and test functions without a part are rejected.
    """
    F, W, u, p = stokes_form(ufl_functionspace)
    with pytest.raises(ComponentError, match="in the order of the sub-spaces"):
        UFLResidual("stokes", F, [p, u])
    with pytest.raises(ComponentError, match="which has no unknown"):
        UFLResidual("stokes", F, u)
    # A test function without a part, with several unknowns
    G = ufl.inner(ufl.grad(u), ufl.grad(ufl.TestFunction(u.ufl_function_space())))
    with pytest.raises(ComponentError, match="has no part, but there are 2"):
        UFLResidual("stokes", G * ufl.dx, [u, p])


def test_earlier_steps_are_given_by_unknown(ufl_functionspace):
    """With several unknowns, ``previous`` maps each unknown to its steps, in
    any order and with a single step alone; a sequence is rejected.
    """
    F, W, u, p = stokes_form(ufl_functionspace)
    v, q = ufl.TestFunctions(W)
    u_n = ufl.Coefficient(u.ufl_function_space())
    G = F + ufl.inner(u - u_n, v) * ufl.dx
    residual = UFLResidual("unsteady", G, [u, p], previous={u: (u_n,)})
    velocity = residual.unknowns[0]
    assert velocity.previous(1) in residual.form.coefficients()
    # Keyed by unknown, so in any order, and a single step without a tuple
    p_n = ufl.Coefficient(p.ufl_function_space())
    G = G + (p - p_n) * q * ufl.dx
    reordered = UFLResidual("unsteady", G, [u, p], previous={p: p_n, u: (u_n,)})
    velocity, pressure = reordered.unknowns
    coefficients = reordered.form.coefficients()
    assert velocity.previous(1) in coefficients
    assert pressure.previous(1) in coefficients
    with pytest.raises(ComponentError, match="give previous by unknown"):
        UFLResidual("unsteady", G, [u, p], previous=(u_n,))
    with pytest.raises(ComponentError, match="not an unknown"):
        UFLResidual("unsteady", G, [u, p], previous={u_n: (u_n,)})


@pytest.mark.parametrize("cell", CELLS)
def test_terminals_may_be_expressions(ufl_functionspace, cell):
    """A coefficient or constant may be replaced by any expression of its shape,
    e.g. of another field; an unknown only by a field with its element.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell)
    B = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1, shape=(gdim,)))
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    b, kappa = ufl.Coefficient(B), ufl.Constant(mesh)
    F = (
        kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(b, ufl.grad(u)) * v
    ) * ufl.dx
    domain = Domain(DOMAIN, cell)
    T = Field("T", domain, ("Lagrange", 2))
    replacements = {b: -ufl.grad(T), kappa: 2.0}
    residual = UFLResidual("convection", F, u, terminals=replacements)
    assert T in residual.fields
    assert residual.parameters == {} and residual.known_coefficients == {}
    assert residual.field.domain is domain
    with pytest.raises(ComponentError, match="has the shape"):
        UFLResidual("convection", F, u, terminals={b: T, kappa: 2.0})
    with pytest.raises(ComponentError, match="is an unknown: it must be given"):
        UFLResidual("convection", F, u, terminals={**replacements, u: T**2})
