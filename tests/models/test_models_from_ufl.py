"""Each model of sumfem.models against its formulation in plain UFL.

Each test writes a model's terms as the UFL file of a DOLFINx C++ demo would,
turns them into a ``UFLResidual``, and compares it with the model on fields of
its own. Both are discretized on the same mesh data, so the fields have the
same layout and the random data of one is copied to the other: the assembled
residual, Jacobian and preconditioner agree to machine precision.
"""

import numpy as np
import pytest
import sympy.physics.units as syu
import ufl
from basix.ufl import element

from sumfem.discrete import DomainMaps, discretize
from sumfem.models import (
    Accumulation,
    BoundaryConstraint,
    BoundaryFlux,
    Brinkman,
    BrinkmanForchheimer,
    Darcy,
    Diffusion,
    DirichletMultiplier,
    GradDiv,
    NavierStokes,
    Outflow,
    Poisson,
    RateReaction,
    Reaction,
    Source,
    Stokes,
    arrhenius,
)
from sumfem.solvers import BDF
from sumfem.symbolic import (
    BOLTZMANN,
    Domain,
    Field,
    UFLResidual,
    boltzmann_constant,
    build_system,
)
from sumfem.symbolic.si import to_si

CELLS = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]
"""The cell types each comparison runs on."""

DOMAIN = "box"
"""The name of the symbolic domain of both formulations, see ``compare``."""


@pytest.mark.parametrize("coefficient", ["parameter", "matrix", "nonlinear"])
@pytest.mark.parametrize("cell", CELLS)
def test_diffusion(cell, coefficient, ufl_functionspace, box, compare):
    """Diffusion with a scalar parameter, a matrix, or a coefficient of the
    field, restricted to the cells of a marker.
    """
    mesh, V = ufl_functionspace(cell, degree=2)
    gdim = ufl.Cell(cell).topological_dimension
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    kappa = ufl.Constant(mesh)
    matrix = ufl.as_matrix(
        [[2.0 + i + j if i == j else 0.5 for j in range(gdim)] for i in range(gdim)]
    )
    k = {"parameter": kappa, "matrix": matrix, "nonlinear": 1 + uh**2}[coefficient]
    grad = ufl.dot(k, ufl.grad(uh)) if coefficient == "matrix" else k * ufl.grad(uh)
    F = ufl.inner(grad, ufl.grad(v)) * ufl.dx(1)
    plain = UFLResidual(
        "plain",
        F,
        uh,
        names={DOMAIN: mesh, "kappa": kappa},
        markers={"dx": {1: "LeftHalf"}},
    )

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 2))
    model_k = {"parameter": None, "matrix": matrix, "nonlinear": 1 + u**2}[coefficient]
    model = Diffusion("d", u, model_k, markers=("LeftHalf",))
    is_parameter = coefficient == "parameter"
    values = [({"plain.kappa": 2.5}, {"d.coefficient": 2.5})] if is_parameter else []
    compare(plain, [plain.field], model, [u], box(cell), values)


@pytest.mark.parametrize("cell", CELLS)
def test_source_of_a_vector_field(cell, ufl_functionspace, box, compare):
    """A source of a vector field, a vector parameter by default, with diffusion:
    a source alone does not depend on the field, so its Jacobian is empty.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell, degree=1, shape=(gdim,))
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    f = ufl.Constant(mesh, shape=(gdim,))
    F = ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
    plain = UFLResidual("plain", F, uh, names={DOMAIN: mesh, "f": f})

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 1, (gdim,)))
    model = Diffusion("d", u, 1.0) + Source("s", u)
    value = np.arange(1.0, gdim + 1.0)
    values = [({"plain.f": value}, {"s.source": value})]
    compare(plain, [plain.field], model, [u], box(cell), values)


@pytest.mark.parametrize("cell", CELLS)
def test_poisson(cell, ufl_functionspace, box, compare):
    """Poisson's equation, with its coefficient one by default and a source
    that depends on the coordinates.
    """
    mesh, V = ufl_functionspace(cell, degree=2)
    uh, v = ufl.Coefficient(V), ufl.TestFunction(V)
    xh = ufl.SpatialCoordinate(mesh)
    F = ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.sin(3 * xh[0]) * v * ufl.dx
    plain = UFLResidual("plain", F, uh, names={DOMAIN: mesh})

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 2))
    x = ufl.SpatialCoordinate(domain)
    model = Poisson("p", u, source=ufl.sin(3 * x[0]))
    compare(plain, [plain.field], model, [u], box(cell))


@pytest.mark.parametrize("cell", CELLS)
def test_boundary_flux_and_outflow(cell, ufl_functionspace, box, compare):
    """A flux into the domain through one side, and advection out of another,
    on the facets of their markers.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell)
    ch, v = ufl.Coefficient(V), ufl.TestFunction(V)
    q = ufl.Constant(mesh)
    n = ufl.FacetNormal(mesh)
    velocity = ufl.as_vector([1.0 + i for i in range(gdim)])
    ds = ufl.Measure("ds", mesh)
    F = -q * v * ds(1) + ufl.dot(velocity, n) * ch * v * ds(2)
    plain = UFLResidual(
        "plain",
        F,
        ch,
        names={DOMAIN: mesh, "q": q},
        markers={"ds": {1: "Left", 2: "Right"}},
    )

    c = Field("c", Domain(DOMAIN, cell), ("Lagrange", 1))
    inflow = BoundaryFlux("in", c, 2.0, markers=("Left",))
    model = inflow + Outflow("out", c, velocity, markers=("Right",))
    values = [({"plain.q": 2.0}, {})]
    compare(plain, [plain.field], model, [c], box(cell), values)


@pytest.mark.parametrize("cell", CELLS)
def test_dirichlet_multiplier(cell, ufl_functionspace, box, compare):
    """A Dirichlet condition with a Lagrange multiplier, the two fields on a
    mixed space, the multiplier here on the whole mesh.
    """
    mesh, V = ufl_functionspace(cell)
    Q = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, Q)
    uh, lh = ufl.Coefficient(V), ufl.Coefficient(Q)
    v, mu = ufl.TestFunctions(W)
    g = ufl.Constant(mesh)
    ds = ufl.Measure("ds", mesh)
    F = ufl.inner(lh, v) * ds(1) + ufl.inner(uh - g, mu) * ds(1)
    plain = UFLResidual(
        "plain",
        F,
        [uh, lh],
        names={DOMAIN: mesh, "u": uh, "lambda": lh, "g": g},
        markers={"ds": {1: "Left"}},
    )

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 1))
    lam = Field("lambda", domain, ("Lagrange", 1))
    model = DirichletMultiplier("m", u, lam, markers=("Left",))
    values = [({"plain.g": 0.5}, {"m.value": 0.5})]
    compare(plain, plain.unknowns, model, [u, lam], box(cell), values)


@pytest.mark.parametrize("constraint", ["normal", "nonlinear"])
@pytest.mark.parametrize("cell", CELLS)
def test_boundary_constraint(cell, constraint, ufl_functionspace, box, compare):
    """A constraint with a Lagrange multiplier of its own shape: the normal
    component of a vector field, or a nonlinear constraint of a scalar one,
    whose derivative enters the equation of the field.
    """
    gdim = ufl.Cell(cell).topological_dimension
    shape = (gdim,) if constraint == "normal" else ()
    mesh, V = ufl_functionspace(cell, degree=1, shape=shape)
    Q = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, Q)
    uh, lh = ufl.Coefficient(V), ufl.Coefficient(Q)
    v, mu = ufl.TestFunctions(W)
    g = ufl.Constant(mesh)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", mesh)
    if constraint == "normal":
        c, dc = ufl.dot(uh, n) - g, ufl.dot(v, n)
    else:
        c, dc = uh**2 - g, 2 * uh * v
    F = lh * dc * ds(1) + c * mu * ds(1)
    names = {DOMAIN: mesh, "u": uh, "lambda": lh, "g": g}
    plain = UFLResidual("plain", F, [uh, lh], names=names, markers={"ds": {1: "Left"}})

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 1, shape))
    lam = Field("lambda", domain, ("Lagrange", 1))
    n_model = ufl.FacetNormal(domain)
    model_c = ufl.dot(u, n_model) - 0.5 if constraint == "normal" else u**2 - 0.5
    model = BoundaryConstraint("c", u, lam, model_c, markers=("Left",))
    values = [({"plain.g": 0.5}, {})]
    compare(plain, plain.unknowns, model, [u, lam], box(cell), values)


@pytest.mark.parametrize("reaction", ["mass action", "rate"])
@pytest.mark.parametrize("cell", CELLS)
def test_reactions(cell, reaction, ufl_functionspace, box, compare):
    """``A + 2 B <-> C``: each species gets its coefficient times the rate in
    its equation, reactants with a plus and products with a minus sign.
    """
    mesh, V = ufl_functionspace(cell)
    W = ufl.MixedFunctionSpace(V, V, V)
    ah, bh, ch = ufl.Coefficient(V), ufl.Coefficient(V), ufl.Coefficient(V)
    va, vb, vc = ufl.TestFunctions(W)
    kf = ufl.Constant(mesh)
    kb = 0.3
    if reaction == "mass action":
        R = kf * ah * bh**2 - kb * ch
    else:
        # A rate of every species, so no block of the Jacobian is empty
        R = kf * ah * ufl.exp(-bh) - kb * ch**2
    F = R * (va + 2 * vb - vc) * ufl.dx
    names = {DOMAIN: mesh, "A": ah, "B": bh, "C": ch, "kf": kf}
    plain = UFLResidual("plain", F, [ah, bh, ch], names=names)

    domain = Domain(DOMAIN, cell)
    a, b, c = (Field(name, domain, ("Lagrange", 1)) for name in "ABC")
    reactants, products = {a: 1, b: 2}, {c: 1}
    if reaction == "mass action":
        model = Reaction("r", reactants, products, backward_rate=kb)
        values = [({"plain.kf": 1.7}, {"r.forward_rate": 1.7})]
    else:
        model = RateReaction(
            "r", reactants, products, rate=1.7 * a * ufl.exp(-b) - kb * c**2
        )
        values = [({"plain.kf": 1.7}, {})]
    compare(plain, plain.unknowns, model, [a, b, c], box(cell), values)


def test_arrhenius_is_the_arrhenius_law():
    """``k_0 exp(-E / (k_B T))``, with the Boltzmann constant in J/K by default.

    For a temperature without a domain, a number, it is its SI value; on a
    domain, the parameter of :func:`boltzmann_constant` there, so units are
    checked.
    """
    k_B = to_si(BOLTZMANN, syu.joule / syu.kelvin)
    assert k_B == 1.380649e-23
    rate = arrhenius(2.0, 0.5 * k_B * 300.0, 300.0)
    assert float(rate) == pytest.approx(2.0 * np.exp(-0.5))
    domain = Domain("bar", "interval")
    T = Field("T", domain, ("Lagrange", 1), unit=syu.kelvin)
    on_domain = arrhenius(2.0, 1e-20, T)
    parameter = boltzmann_constant(domain)
    assert parameter in ufl.algorithms.extract_coefficients(on_domain)
    assert parameter.name == "bar.k_B" and boltzmann_constant(domain) is parameter
    in_ev = arrhenius(2.0, 0.1, 300.0, boltzmann=8.617333262e-5)
    assert float(in_ev) == pytest.approx(2.0 * np.exp(-0.1 / (8.617333262e-5 * 300.0)))


@pytest.mark.parametrize("cell", CELLS)
def test_accumulation_by_backward_euler(cell, ufl_functionspace, box, same_assembly):
    """A weighted time derivative on the cells of a marker, as one step of
    backward Euler: ``(w (u - u_n) / dt, v)``.

    The model's time derivative becomes the weighted history of
    :class:`~sumfem.solvers.BDF`, whose weights the scheme sets for a step.
    """
    mesh, V = ufl_functionspace(cell, degree=2)
    uh, un, v = ufl.Coefficient(V), ufl.Coefficient(V), ufl.TestFunction(V)
    dt = ufl.Constant(mesh)
    F = 2.5 * (uh - un) / dt * v * ufl.dx(1)
    plain = UFLResidual(
        "plain",
        F,
        uh,
        names={DOMAIN: mesh, "u": uh, "dt": dt},
        markers={"dx": {1: "LeftHalf"}},
        previous={uh: (un,)},
    )
    disc_plain = discretize([plain], DomainMaps({DOMAIN: box(cell)}))
    disc_plain.set({"plain.dt": 0.1})

    u = Field("u", Domain(DOMAIN, cell), ("Lagrange", 2))
    model = Accumulation("a", u, 2.5, markers=("LeftHalf",))
    scheme = BDF(1)
    system = build_system(model, substitutions=scheme.substitutions(model))
    disc_model = discretize([system], DomainMaps({DOMAIN: box(cell)}))
    scheme.set_step(1, 0.1, disc_model)

    rng = np.random.default_rng(seed=7 + 100 * disc_plain.comm.rank)
    for plain_terminal, terminal in (
        (plain.field, u),
        (plain.field.previous(1), u.previous(1)),
    ):
        plain_values = disc_plain.function(plain_terminal)
        plain_values.x.array[:] = rng.uniform(0.5, 1.5, plain_values.x.array.size)
        plain_values.x.scatter_forward()
        disc_model.function(terminal).x.array[:] = plain_values.x.array
    same_assembly(disc_model[system], disc_plain[plain])


@pytest.mark.parametrize("cell", CELLS)
def test_darcy(cell, ufl_functionspace, box, compare):
    """Darcy flow ``(K / mu grad(p), grad(q))`` on the cells of a marker."""
    mesh, Q = ufl_functionspace(cell, degree=2)
    ph, q = ufl.Coefficient(Q), ufl.TestFunction(Q)
    K, mu = ufl.Constant(mesh), ufl.Constant(mesh)
    F = K / mu * ufl.inner(ufl.grad(ph), ufl.grad(q)) * ufl.dx(1)
    names = {DOMAIN: mesh, "p": ph, "K": K, "mu": mu}
    plain = UFLResidual("plain", F, ph, names=names, markers={"dx": {1: "LeftHalf"}})

    p = Field("p", Domain(DOMAIN, cell), ("Lagrange", 2))
    model = Darcy("darcy", p, markers=("LeftHalf",))
    values = [
        (
            {"plain.K": 1e-2, "plain.mu": 3.0},
            {"darcy.permeability": 1e-2, "darcy.viscosity": 3.0},
        )
    ]
    compare(plain, [plain.field], model, [p], box(cell), values)


FLOW_CELLS = ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
"""The cell types of the comparisons of flows of a vector velocity."""


@pytest.mark.parametrize(
    "flow", ["stokes", "navier-stokes", "brinkman", "brinkman-forchheimer"]
)
@pytest.mark.parametrize("cell", FLOW_CELLS)
def test_flow(cell, flow, ufl_functionspace, box, compare):
    """Taylor-Hood flows on the cells of a marker: the viscous term of the
    symmetric gradient, in the pores ``mu / eps``, with convection
    ``rho grad(u) u``, in the pores ``rho / eps^2``, Darcy's drag ``mu / K u``
    and Forchheimer's ``rho beta |u| u``, and the pressure and continuity
    terms.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell, degree=2, shape=(gdim,))
    Q = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    uh, ph = ufl.Coefficient(V), ufl.Coefficient(Q)
    w, q = ufl.TestFunctions(ufl.MixedFunctionSpace(V, Q))
    mu, rho, eps, K, beta, floor = (ufl.Constant(mesh) for _ in range(6))
    dx = ufl.Measure("dx", mesh)(1)
    porous = flow in ("brinkman", "brinkman-forchheimer")
    convection = ufl.dot(ufl.dot(ufl.grad(uh), uh), w) * dx
    viscosity = mu / eps if porous else mu
    F = 2 * viscosity * ufl.inner(ufl.sym(ufl.grad(uh)), ufl.sym(ufl.grad(w))) * dx
    if flow == "navier-stokes":
        F += rho * convection
    if porous:
        F += mu / K * ufl.dot(uh, w) * dx
    if flow == "brinkman-forchheimer":
        speed = ufl.sqrt(ufl.dot(uh, uh) + floor**2)
        F += rho / eps**2 * convection + rho * beta * ufl.dot(speed * uh, w) * dx
    F += -ph * ufl.div(w) * dx - ufl.div(uh) * q * dx
    names = {DOMAIN: mesh, "u": uh, "p": ph, "mu": mu, "rho": rho, "eps": eps}
    names.update(K=K, beta=beta, floor=floor)
    plain = UFLResidual(
        "plain", F, [uh, ph], names=names, markers={"dx": {1: "LeftHalf"}}
    )

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 2, (gdim,)))
    p = Field("p", domain, ("Lagrange", 1))
    markers = ("LeftHalf",)
    model = {
        "stokes": lambda: Stokes("f", u, p, markers=markers),
        "navier-stokes": lambda: NavierStokes("f", u, p, markers=markers),
        "brinkman": lambda: Brinkman("f", u, p, markers=markers),
        "brinkman-forchheimer": lambda: BrinkmanForchheimer("f", u, p, markers=markers),
    }[flow]()
    used = {"mu": ("viscosity", 2.0)}
    if flow in ("navier-stokes", "brinkman-forchheimer"):
        used["rho"] = ("density", 3.0)
    if porous:
        used.update(eps=("porosity", 0.7), K=("permeability", 0.5))
    if flow == "brinkman-forchheimer":
        used.update(beta=("forchheimer", 4.0), floor=("speed_floor", 1e-12))
    values = [
        (
            {f"plain.{name}": value for name, (_, value) in used.items()},
            {f"f.{parameter}": value for parameter, value in used.values()},
        )
    ]
    compare(plain, plain.unknowns, model, [u, p], box(cell), values)


@pytest.mark.parametrize("cell", FLOW_CELLS)
def test_stokes_with_grad_div(cell, ufl_functionspace, box, compare):
    """Grad-div stabilization ``gamma (div(u), div(w))`` in the residual."""
    gdim = ufl.Cell(cell).topological_dimension
    mesh, V = ufl_functionspace(cell, degree=2, shape=(gdim,))
    Q = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    uh, ph = ufl.Coefficient(V), ufl.Coefficient(Q)
    w, q = ufl.TestFunctions(ufl.MixedFunctionSpace(V, Q))
    F = (
        2 * ufl.inner(ufl.sym(ufl.grad(uh)), ufl.sym(ufl.grad(w))) * ufl.dx
        - ph * ufl.div(w) * ufl.dx
        - ufl.div(uh) * q * ufl.dx
        + 5.0 * ufl.div(uh) * ufl.div(w) * ufl.dx
    )
    plain = UFLResidual("plain", F, [uh, ph], names={DOMAIN: mesh, "u": uh, "p": ph})

    domain = Domain(DOMAIN, cell)
    u = Field("u", domain, ("Lagrange", 2, (gdim,)))
    p = Field("p", domain, ("Lagrange", 1))
    model = Stokes("f", u, p, 1.0) + GradDiv("grad_div", u, 5.0)
    compare(plain, plain.unknowns, model, [u, p], box(cell))
