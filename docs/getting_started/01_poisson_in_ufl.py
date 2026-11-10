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

# # Poisson's equation in UFL
#
# This page illustrates how to:
#
# - Bring a problem written in plain UFL, as for the DOLFINx C++ interface,
#   into sumfem with {py:class}`sumfem.symbolic.UFLResidual`
# - Bind it to a mesh with {py:func}`sumfem.discrete.discretize`
# - Solve it with a {py:class}`sumfem.solvers.Solver`
#
# ## The problem
#
# We solve Poisson's equation on the unit square $\Omega$,
#
# $$
# \begin{aligned}
#   -\nabla^2 u &= f \quad \text{in } \Omega, \\
#   u &= u_D \quad \text{on } \partial\Omega,
# \end{aligned}
# $$
#
# whose weak form is: find $u$ with $u = u_D$ on $\partial\Omega$ such that
#
# $$
#   F(u; v) = \int_\Omega \nabla u \cdot \nabla v - f v \, \mathrm{d}x = 0
# $$
#
# for all test functions $v$ that vanish on $\partial\Omega$. With
# $u_D = 1 + x^2 + 2y^2$ and $f = -6$, the exact solution is $u = u_D$.
#
# ## Three layers
#
# {py:mod}`sumfem.symbolic` describes domains, fields and residuals
# symbolically, without mesh data, so a model is compiled once and solved on
# any mesh. {py:mod}`sumfem.discrete` binds it to a mesh, and
# {py:mod}`sumfem.solvers` solves it.

# +
from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl
from basix.ufl import element

from sumfem import discrete, solvers, symbolic
from sumfem.post_processing import error

# -

# ## The model in plain UFL
#
# The UFL file of a DOLFINx C++ program writes the weak form, without data,
# with a {py:class}`ufl.Mesh`, a {py:class}`ufl.FunctionSpace` on it, the
# unknown as a {py:class}`ufl.Coefficient`, a test function and the source as
# a {py:class}`ufl.Constant`:

cell = "triangle"
mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", cell, 2))
uh = ufl.Coefficient(V)
v = ufl.TestFunction(V)
f = ufl.Constant(mesh)
F = ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# {py:class}`UFLResidual <sumfem.symbolic.UFLResidual>` turns the residual
# into a sumfem model. The unknown becomes a field, and the constant a named
# parameter of the model, `"poisson.f"`, which can be set later. `names` gives
# the variable name of each, and of the mesh, whose symbolic domain is then
# `"square"`. A Jacobian and a preconditioner can be given too; without them,
# the Jacobian is derived.

poisson = symbolic.UFLResidual(
    "poisson", F, uh, names={"square": mesh, "u": uh, "f": f}
)
print("Equations for:", [field.name for field in poisson.equations])
print("Parameters:", list(poisson.parameters))

# ## The mesh
#
# So far, we have not considered an actual discrete domain: a
# {py:class}`dolfinx.mesh.Mesh` with its
# {py:class}`MeshTags <dolfinx.mesh.MeshTags>`. In {py:mod}`sumfem` we store
# these in a {py:class}`sumfem.discrete.MeshData`, a dataclass that also names
# the tags: for each entity dimension, a marker name maps to tag values. It is
# created from locator functions, from existing mesh tags or from a Gmsh model,
# see {doc}`05_mesh_data`. Here the whole boundary is marked `"Boundary"` by a
# locator, a function of the coordinates as
# {py:func}`dolfinx.mesh.locate_entities` takes:

square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
marked = discrete.MeshData.from_locators(
    square, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)

# ## Discretize and solve
#
# {py:class}`DomainMaps <sumfem.discrete.DomainMaps>` pairs each symbolic domain
# with its `MeshData`, and {py:func}`discretize <sumfem.discrete.discretize>`
# creates one function for every field and parameter of the given components.
# The result, a {py:class}`Discretization <sumfem.discrete.Discretization>`,
# holds them all.

u = poisson.field
domain = u.domain
domain_maps = discrete.DomainMaps({domain: marked})
components = [poisson]
disc = discrete.discretize(components, domain_maps)

# Parameters are set by object or by name:

disc.set({"poisson.f": -6.0})

# Indexing the discretization gives the discrete counterpart of a symbolic
# object: the DOLFINx function of a field, or the discrete system of a model,
# compiled on first use. A {py:class}`DirichletBC <sumfem.symbolic.DirichletBC>`
# is given by the field, the marker name and the value, here a
# {py:class}`UFL expression <ufl.core.expr.Expr>` of the
# {py:class}`spatial coordinate <ufl.SpatialCoordinate>` of the domain.

x = ufl.SpatialCoordinate(domain)
u_D = 1 + x[0] ** 2 + 2 * x[1] ** 2
bcs = [symbolic.DirichletBC(u, "Boundary", u_D)]
solver = solvers.Solver(disc[poisson], bcs)
solver.solve()
print(f"Newton iterations: {solver.iterations}")

# ## Check the solution
#
# The exact solution is quadratic, so P2 elements reproduce it to round-off.
# {py:func}`error <sumfem.post_processing.error>` takes the DOLFINx
# function and the exact solution as a function of the mesh's coordinates.


# +
def exact(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


print(f"L2 error: {error(disc[u], exact):.2e}")
assert error(disc[u], exact) < 1e-10
# -

# The same symbolic model can be discretized again on another mesh, without
# writing it again; only the forms are compiled for the new mesh's cell.
# {doc}`02_components` writes the same model as a component, which other
# models can extend.
