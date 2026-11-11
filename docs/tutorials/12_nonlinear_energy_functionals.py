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

# # Nonlinear energy functionals
#
# This tutorial illustrates how to:
#
# - Describe a nonlinear problem by an energy, a {py:class}`Functional
#   <sumfem.symbolic.Functional>`, and solve its variation
# - See why the Jacobian of an energy is symmetric, and that of a problem
#   without an energy is not
#
# {doc}`../getting_started/04_energy_functionals` describes Poisson's
# equation by its energy, whose variation is linear. Here the energy is
# nonlinear, and Newton's method needs its Jacobian, see {doc}`06_jacobians`.
#
# ## A nonlinear energy
#
# The energy
#
# $$
#   J(u) = \int_\Omega \frac{1}{2} (1 + u^2) |\nabla u|^2 - g u \,\mathrm{d}x
# $$ (nonlinear-energy)
#
# has the variation
#
# $$
#   J'(u)[v] = \int_\Omega (1 + u^2) \nabla u \cdot \nabla v + u |\nabla u|^2 v
#   - g v \,\mathrm{d}x,
# $$
#
# the weak form of a nonlinear diffusion,
#
# $$
#   -\nabla \cdot ((1 + u^2) \nabla u) + u |\nabla u|^2 = g.
# $$ (energy-diffusion)

# +
from mpi4py import MPI

import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Source
from sumfem.post_processing import error

# -

# ### The problem
#
# The unknown is in quadratic Lagrange elements on a square. The source $g$
# comes from a manufactured solution, a quadratic, which these elements
# represent exactly, so the error shows the solver alone.


# +
def exact(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


cell_name = "triangle"
domain = symbolic.Domain("square", cell_name)
u = symbolic.Field("u", domain, ("Lagrange", 2))
x = ufl.SpatialCoordinate(domain)
u_exact = exact(x)
gradient = ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact))
g = -ufl.div((1 + u_exact**2) * ufl.grad(u_exact)) + u_exact * gradient
# -

# The mesh of the square, of the cells of the domain, has its boundary
# marked `"Boundary"`, where the exact solution is prescribed.

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 16, 16, cell_type=dolfinx.mesh.to_type(cell_name)
)
marked = discrete.MeshData.from_locators(
    mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)
bcs = [symbolic.DirichletBC(u, "Boundary", u_exact)]

# ### Solving it
#
# The energy is a {py:class}`Functional <sumfem.symbolic.Functional>` that
# returns {eq}`nonlinear-energy`, and {py:meth}`to_residual
# <sumfem.symbolic.Functional.to_residual>` gives its variation, which is
# solved like any other residual, starting Newton's method from a constant:


# +
class NonlinearEnergy(symbolic.Functional):
    """``int 1/2 (1 + u^2) |grad(u)|^2 - g u dx``."""

    def __init__(self, name, field, source):
        super().__init__(name)
        self.field, self.source = field, source

    def functional(self):
        u = self.field
        gradient = ufl.inner(ufl.grad(u), ufl.grad(u))
        return (0.5 * (1 + u**2) * gradient - self.source * u) * u.domain.dx()


variational = NonlinearEnergy("energy", u, g).to_residual()
disc = discrete.discretize([variational], discrete.DomainMaps({domain: marked}))
disc.set({u: 1.0})
solver = solvers.Solver(disc[variational], bcs)
solver.solve()
print(f"{solver.iterations} Newton iterations, error {error(disc[u], exact):.1e}")
assert error(disc[u], exact) < 1e-10
# -

# ## Symmetric Jacobians
#
# The Jacobian of the variation is the second variation of the energy,
#
# $$
#   J''(u)[v, \delta u] = \int_\Omega (1 + u^2) \nabla \delta u \cdot \nabla v
#   + 2 u\, \delta u\, \nabla u \cdot \nabla v
#   + 2 u\, v\, \nabla u \cdot \nabla \delta u
#   + |\nabla u|^2 \delta u\, v \,\mathrm{d}x,
# $$ (energy-jacobian)
#
# which is symmetric in $v$ and $\delta u$. The nonlinear diffusion of
# {doc}`06_jacobians`,
#
# $$
#   -\nabla \cdot ((1 + u^2) \nabla u) = f,
# $$ (diffusion-without-energy)
#
# is not the variation of an energy, and its Jacobian, the first two terms
# of {eq}`energy-jacobian`, is therefore not symmetric: the term
# $2 u\, \delta u\, \nabla u \cdot \nabla v$ changes when $\delta u$ and $v$
# swap, and has no partner. For the same exact solution, the diffusion is a
# {py:class}`Residual <sumfem.symbolic.Residual>` with a source:


# +
class NonlinearDiffusion(symbolic.Residual):
    """``((1 + u^2) grad(u), grad(v))``."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u = self.field
        return ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(u.v)) * u.domain.dx()


f = -ufl.div((1 + u_exact**2) * ufl.grad(u_exact))
diffusion = NonlinearDiffusion("diffusion", u) + Source("source", u, f)
# -

# The Jacobians of the two problems, assembled at the exact solution, show
# the difference:


# +
def symmetric_jacobian(model):
    disc = discrete.discretize([model], discrete.DomainMaps({domain: marked}))
    disc.function(u).interpolate(exact)
    A = dolfinx.fem.petsc.assemble_matrix(disc[model].jacobian)
    A.assemble()
    symmetric = A.isSymmetric(1e-10)
    A.destroy()
    return symmetric


print("Symmetric Jacobian of the energy:", symmetric_jacobian(variational))
print("Symmetric Jacobian of the diffusion:", symmetric_jacobian(diffusion))
assert symmetric_jacobian(variational) and not symmetric_jacobian(diffusion)
# -

# A symmetric Jacobian can be solved with methods for symmetric matrices,
# e.g. a Cholesky factorization, or conjugate gradients where it is also
# positive definite.
