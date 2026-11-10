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

# # Energy functionals
#
# This page illustrates how to:
#
# - Describe a problem by an energy, a {py:class}`sumfem.symbolic.Functional`,
#   whose variation is the residual
# - Do the same in plain UFL with {py:class}`sumfem.symbolic.UFLResidual`
# - Assemble a functional
#
# ## The problem
#
# Many problems minimize an energy. Poisson's equation $-\nabla^2 u = f$ on
# the unit square, with $u = u_D = 1 + x^2 + 2y^2$ on the boundary and
# $f = -6$, is the condition for a stationary point of the Dirichlet energy
#
# $$
#   J(u) = \int_\Omega \frac{1}{2} |\nabla u|^2 - f u \,\mathrm{d}x,
# $$
#
# whose variation in the direction $v$ is the weak form of
# {doc}`01_poisson_in_ufl`.

# +
from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl
from basix.ufl import element

from sumfem import discrete, solvers, symbolic
from sumfem.post_processing import error

# -

# ## The energy as a component
#
# A {py:class}`Functional <sumfem.symbolic.Functional>` returns the energy in
# {py:meth}`functional <sumfem.symbolic.Functional.functional>`:


class DirichletEnergy(symbolic.Functional):
    """``int 1/2 |grad(u)|^2 - f u dx``."""

    def __init__(self, name, field, source):
        super().__init__(name)
        self.field, self.source = field, source

    def functional(self):
        u = self.field
        energy = 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) - self.source * u
        return energy * u.domain.dx()


# {py:meth}`to_residual <sumfem.symbolic.Functional.to_residual>` derives the
# residual, the derivative of the energy in the direction of the test function
# of the field, which is solved like any other:

domain = symbolic.Domain("square", "triangle")
u = symbolic.Field("u", domain, ("Lagrange", 2))
energy = DirichletEnergy("energy", u, -6.0)
residual = energy.to_residual()

# On the unit square, with the exact solution on the boundary:


# +
def exact(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
marked = discrete.MeshData.from_locators(
    square, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)
x = ufl.SpatialCoordinate(domain)
bcs = [symbolic.DirichletBC(u, "Boundary", exact(x))]
# -

# Both the energy and its residual are discretized, and the residual is solved
# like any other. The exact solution is quadratic, so the P2 solution matches
# it to round-off:

maps = discrete.DomainMaps({domain: marked})
disc = discrete.discretize([energy, residual], maps)
solvers.Solver(disc[residual], bcs).solve()
print(f"L2 error: {error(disc[u], exact):.2e}")
assert error(disc[u], exact) < 1e-10

# ## Assembling the energy
#
# The discrete counterpart of the energy, `disc[energy]`, is its compiled
# {py:class}`dolfinx.fem.Form`, which gives the integral over the cells of one
# process. {py:meth}`assemble_scalar
# <sumfem.discrete.Discretization.assemble_scalar>` assembles it and sums over
# all processes, here the energy of the solution,
# $\int \frac{1}{2}(4x^2 + 16y^2) + 6(1 + x^2 + 2y^2) \,\mathrm{d}x = 46/3$:

J = disc.assemble_scalar(energy)
print(f"J(u) = {J:.6f}, expected {46 / 3:.6f}")
assert np.isclose(J, 46 / 3)

# ## The energy in plain UFL
#
# In plain UFL, the residual is the derivative of the energy in the direction
# of a test function, which {py:class}`UFLResidual <sumfem.symbolic.UFLResidual>`
# takes like any residual. `terminals` gives it the field of the component
# above, so both are solved for the same unknown.

# +
mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))
uh = ufl.Coefficient(V)
E = (0.5 * ufl.inner(ufl.grad(uh), ufl.grad(uh)) + 6.0 * uh) * ufl.dx
F = ufl.derivative(E, uh, ufl.TestFunction(V))
plain = symbolic.UFLResidual("plain", F, uh, terminals={uh: u})

disc = discrete.discretize([plain], maps)
solvers.Solver(disc[plain], bcs).solve()
print(f"L2 error: {error(disc[u], exact):.2e}")
assert error(disc[u], exact) < 1e-10
# -
