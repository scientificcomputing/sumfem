# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # A model as a component
#
# This page illustrates how to:
#
# - Describe a problem symbolically, without mesh data, with a
#   {py:class}`sumfem.symbolic.Domain`, a {py:class}`sumfem.symbolic.Field`
#   and a {py:class}`sumfem.symbolic.Residual`
# - Extend a model by inheritance
# - Sum components into a model with `+`
# - Keep the equation of a contribution that is zero
#
# A plain UFL form, as in {doc}`01_poisson_in_ufl`, is fixed: to change a
# term, the form is written again. A component is a class instead, so a
# model is extended by inheritance and summed with other models with `+`.
#
# We solve Poisson's equation $-\nabla \cdot (k \nabla u) = f$ on the unit
# square, with $u = u_D = 1 + x^2 + 2y^2$ on the boundary: first with $k = 1$
# and $f = -6$, then with a conductivity that varies. The exact solution is
# $u = u_D$ in both.

# +
from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.post_processing import error

# -

# ## Symbolic objects
#
# A {py:class}`Domain <sumfem.symbolic.Domain>` is a symbolic description of
# a type of mesh: its name and its cell type, without mesh data. A model is
# written on domains with three kinds of named objects, each a
# {py:class}`UFL coefficient <ufl.Coefficient>` without data:
#
# - a {py:class}`Field <sumfem.symbolic.Field>`, an unknown with its finite
#   element. It carries its own test function `u.v` and trial function `u.du`;
# - a {py:class}`Parameter <sumfem.symbolic.Parameter>`, a value constant in
#   space, such as a source or a time step;
# - a {py:class}`KnownCoefficient <sumfem.symbolic.KnownCoefficient>`, given
#   data that varies in space, such as an imposed velocity.
#
# In UFL terms they are terminals, the leaves of an expression.
# {py:func}`discretize <sumfem.discrete.discretize>` gives each a DOLFINx
# function on the mesh of its domain. {py:class}`UFLResidual
# <sumfem.symbolic.UFLResidual>` creates them from the plain UFL objects of
# {doc}`01_poisson_in_ufl`, or takes existing ones in its `terminals`
# argument. Here the field is created directly.

cell_name = "triangle"
domain = symbolic.Domain("square", cell_name)
u = symbolic.Field("u", domain, ("Lagrange", 2))

# ## The model
#
# A model is a {py:class}`Residual <sumfem.symbolic.Residual>`: a component that
# returns its contribution to the weak form in {py:meth}`residual
# <sumfem.symbolic.Residual.residual>`. Its inputs are given to the constructor.
# {py:meth}`input <sumfem.symbolic.Component.input>` turns a value into a UFL
# expression, or, when no value is given, creates a named
# {py:class}`Parameter <sumfem.symbolic.Parameter>` that can be set later. The
# conductivity is a method of its own, so that a subclass can change it.


# +
class Poisson(symbolic.Residual):
    """``(k grad(u), grad(v)) - (f, v)``, with ``k = 1``.

    Args:
        name: Name of the model.
        field: The unknown ``u``.
        source: The source ``f``. Defaults to the parameter ``"<name>.f"``.
    """

    def __init__(self, name, field, source=None):
        super().__init__(name)
        self.field = field
        self.source = self.input("f", source, domain=field.domain, doc="Source")

    def conductivity(self):
        return 1.0

    def residual(self):
        u, v = self.field, self.field.v
        dx = u.domain.dx()
        k = self.conductivity()
        return ufl.inner(k * ufl.grad(u), ufl.grad(v)) * dx - self.source * v * dx


# -

# The model knows what it uses: the fields it writes equations for, and its
# parameters.

model = Poisson("poisson", u)
print("Equations for:", [field.name for field in model.equations])
print("Parameters:", list(model.parameters))

# It gives the same discrete system as the plain form of
# {doc}`01_poisson_in_ufl`: the tests of sumfem check models this way, by
# assembling both. It is discretized and solved in the same way.

# +
square = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 16, 16, cell_type=dolfinx.mesh.to_type(cell_name)
)
marked = discrete.MeshData.from_locators(
    square, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)
maps = discrete.DomainMaps({domain: marked})
x = ufl.SpatialCoordinate(domain)
u_D = 1 + x[0] ** 2 + 2 * x[1] ** 2
bcs = [symbolic.DirichletBC(u, "Boundary", u_D)]


def exact(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


disc = discrete.discretize([model], maps)
disc.set({"poisson.f": -6.0})
solvers.Solver(disc[model], bcs).solve()
print(f"L2 error: {error(disc[u], exact):.2e}")
assert error(disc[u], exact) < 1e-10
# -

# ## Extending the model
#
# A material whose conductivity varies, $k = 1 + x^2$, only overrides
# `conductivity`. For the same exact solution, the source becomes
# $f = -\nabla \cdot (k \nabla u_D)$, a UFL expression given to the
# constructor.


# +
class VariablePoisson(Poisson):
    """Poisson's equation with the conductivity ``1 + x^2``."""

    def conductivity(self):
        x = ufl.SpatialCoordinate(self.field.domain)
        return 1 + x[0] ** 2


# -

# The source for the same exact solution, and the solve:

k = 1 + x[0] ** 2
variable = VariablePoisson("variable", u, source=-ufl.div(k * ufl.grad(u_D)))
disc = discrete.discretize([variable], maps)
solvers.Solver(disc[variable], bcs).solve()
print(f"L2 error: {error(disc[u], exact):.2e}")
assert error(disc[u], exact) < 1e-10

# ## A nonlinear model
#
# The conductivity may also depend on the solution, $q(u) = 1 + u^2$, as in
# the [nonlinear Poisson equation of the FEniCSx
# tutorial](https://jsdokken.com/dolfinx-tutorial/chapter2/nonlinpoisson.html):
#
# $$
#   -\nabla \cdot (q(u) \nabla u) = f.
# $$
#
# It is the same override. Newton's method uses the Jacobian derived from the
# residual, which includes the derivative of the conductivity, so nothing
# else changes. As in the FEniCSx tutorial, the exact solution is
# $u = 1 + x + 2y$, which gives the source $f = -\nabla \cdot (q(u) \nabla u)$
# and the boundary values.


# +
class NonlinearPoisson(Poisson):
    """Poisson's equation with the conductivity ``1 + u^2``."""

    def conductivity(self):
        return 1 + self.field**2


u_ex = 1 + x[0] + 2 * x[1]
nonlinear = NonlinearPoisson(
    "nonlinear", u, source=-ufl.div((1 + u_ex**2) * ufl.grad(u_ex))
)
disc = discrete.discretize([nonlinear], maps)
solver = solvers.Solver(disc[nonlinear], [symbolic.DirichletBC(u, "Boundary", u_ex)])
solver.solve()
error_nonlinear = error(disc[u], lambda x: 1 + x[0] + 2 * x[1])
print(f"Newton iterations: {solver.iterations}, L2 error: {error_nonlinear:.2e}")
assert error_nonlinear < 1e-10
# -

# ## Summing components
#
# A model may also be split into components, one per term, and summed with
# `+`. Here the source is a component of its own, added to a diffusion
# component without one, so either can be replaced without touching the other.
# The sum is a residual like any other, with the summands' parameters:


# +
class Diffusion(symbolic.Residual):
    """``(grad(u), grad(v))``."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u = self.field
        return ufl.inner(ufl.grad(u), ufl.grad(u.v)) * u.domain.dx()


class Source(symbolic.Residual):
    """``-(f, v)``, with ``f`` the parameter ``"<name>.f"`` unless given."""

    def __init__(self, name, field, source=None):
        super().__init__(name)
        self.field = field
        self.source = self.input("f", source, domain=field.domain, doc="Source")

    def residual(self):
        u = self.field
        return -self.source * u.v * u.domain.dx()


summed = Diffusion("diffusion", u) + Source("source", u)
print("Summands:", [component.name for component in summed.components])
print("Parameters:", list(summed.parameters))
# -

# It is the first problem of this page again, and solved the same way:

disc = discrete.discretize([summed], maps)
disc.set({"source.f": -6.0})
solvers.Solver(disc[summed], bcs).solve()
print(f"L2 error: {error(disc[u], exact):.2e}")
assert error(disc[u], exact) < 1e-10

# ## Contributions that are zero
#
# UFL drops the test function from a form whose terms all vanish, e.g. a
# boundary flux with a prescribed value of zero, and with it the equation
# the form belongs to. A model whose residual is zero for its inputs returns
# `ufl.ZeroBaseForm((field.v,))` instead, which keeps its equation:


# +
class Flux(symbolic.Residual):
    """A prescribed flux ``-(q, v) ds``, zero when ``q`` is."""

    def __init__(self, name, field, q):
        super().__init__(name)
        self.field, self.q = field, symbolic.as_expr(q)

    def residual(self):
        u = self.field
        # A zero given as a number or as UFL is UFL's Zero
        if isinstance(self.q, ufl.constantvalue.Zero):
            return ufl.ZeroBaseForm((u.v,))
        return -self.q * u.v * u.domain.ds()


# -

# With no flux, the model still has the equation of its field:

print("Equations:", [f.name for f in Flux("no_flux", u, 0).equations])

# {doc}`03_preconditioners` solves the variable problem with an iterative
# solver.
