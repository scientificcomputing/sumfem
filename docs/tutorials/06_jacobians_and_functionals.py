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

# # Jacobians and functionals
#
# This tutorial illustrates how to:
#
# - Replace the derived Jacobian of a model by a hand-written one
# - Write a Jacobian as a component of its own, a
#   {py:class}`Jacobian <sumfem.symbolic.Jacobian>`, and attach it to a residual
# - Describe a problem by an energy, a {py:class}`Functional
#   <sumfem.symbolic.Functional>`, whose Jacobian is symmetric
# - Keep the equation of a contribution that is zero
#
# ## Derived Jacobians
#
# Newton's method needs the Jacobian of the residual, its derivative with
# respect to the unknowns. sumfem derives it from the residual of the
# whole system, after any substitutions, e.g. of a time-stepping scheme.
# For the nonlinear diffusion
#
# $$
#   -\nabla \cdot ((1 + u^2) \nabla u) = f
# $$
#
# it is $\int (1 + u^2) \nabla \delta u \cdot \nabla v + 2 u\, \delta u\,
# \nabla u \cdot \nabla v \,\mathrm{d}x$, and Newton's method converges
# quadratically.

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

domain = symbolic.Domain("square", "triangle")
u = symbolic.Field("u", domain, ("Lagrange", 2))
x = ufl.SpatialCoordinate(domain)
u_exact = 1 + x[0] ** 2 + 2 * x[1] ** 2
f = -ufl.div((1 + u_exact**2) * ufl.grad(u_exact))

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
marked = discrete.MeshData.from_locators(
    mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)
bcs = [symbolic.DirichletBC(u, "Boundary", u_exact)]


class NonlinearDiffusion(symbolic.Residual):
    """``((1 + u^2) grad(u), grad(v))``."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u = self.field
        return ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(u.v)) * u.domain.dx()


def solve(model):
    disc = discrete.discretize([model], discrete.DomainMaps({domain: marked}))
    disc.set({u: 1.0})
    solver = solvers.Solver(disc[model], bcs)
    solver.solve()
    accuracy = error(disc[u], lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
    return solver.iterations, accuracy


newton = NonlinearDiffusion("diffusion", u) + Source("source", u, f)
print("Newton: %d iterations, error %.1e" % solve(newton))

# ## A hand-written Jacobian
#
# A model may give its own linearization in {py:meth}`jacobian
# <sumfem.symbolic.Residual.jacobian>`, as a form with the trial function `u.du`.
# The Picard linearization freezes the coefficient, $\int (1 + u^2)\nabla
# \delta u \cdot \nabla v \,\mathrm{d}x$: each iteration is cheaper to
# assemble, but convergence is only linear. The solution is the same, to
# the solver's tolerance, since only the path to it changes.


class PicardDiffusion(NonlinearDiffusion):
    """Nonlinear diffusion, linearized with its coefficient frozen."""

    def jacobian(self):
        u = self.field
        return ufl.inner((1 + u**2) * ufl.grad(u.du), ufl.grad(u.v)) * u.domain.dx()


picard = PicardDiffusion("diffusion", u) + Source("source", u, f)
picard_result = solve(picard)
print("Picard: %d iterations, error %.1e" % picard_result)

# Summands without a hand-written Jacobian, like the source here, keep
# their derived ones.
#
# ## Jacobians as components
#
# A linearization can also be a component of its own, a {py:class}`Jacobian
# <sumfem.symbolic.Jacobian>`, that returns a bilinear form. Jacobians sum with
# `+` like residuals, and {py:meth}`with_jacobian
# <sumfem.symbolic.Residual.with_jacobian>` attaches one to any residual, e.g. one
# from a library, without subclassing it. {py:meth}`to_jacobian
# <sumfem.symbolic.Residual.to_jacobian>` gives the derived linearization of a
# residual as a Jacobian, to combine with others. A preconditioner is given
# the same way, with {py:meth}`with_preconditioner
# <sumfem.symbolic.Residual.with_preconditioner>`, see
# {doc}`../getting_started/03_preconditioners`.


# +
class FrozenStiffness(symbolic.Jacobian):
    """``((1 + u^2) grad(du), grad(v))``, the Picard linearization."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def jacobian(self):
        u = self.field
        return ufl.inner((1 + u**2) * ufl.grad(u.du), ufl.grad(u.v)) * u.domain.dx()


attached = newton.with_jacobian(FrozenStiffness("picard", u))
attached_result = solve(attached)
print("Attached: %d iterations, error %.1e" % attached_result)
assert attached_result == picard_result
# -

# The two give the same linearization, so they take the same iterations to
# the same solution. They differ in where it lives:
#
# - `PicardDiffusion` is a model of its own. Its hand-written Jacobian
#   replaces only its own derived one, in any sum it is part of: the source
#   keeps its derived Jacobian.
# - `FrozenStiffness` is attached to an existing residual, without a new
#   model class. It replaces the Jacobian of all of that residual, here the
#   sum `newton`, so it must linearize every summand. It can do without the
#   source's term only because the source does not depend on `u`. Attached
#   to the diffusion alone, `NonlinearDiffusion("diffusion",
#   u).with_jacobian(...) + Source("source", u, f)`, it would replace the
#   Jacobian of that summand only, as `PicardDiffusion` does.

# ## Functionals
#
# A problem can also be described by an energy, a {py:class}`Functional
# <sumfem.symbolic.Functional>`, whose variation is the residual, see
# {doc}`../getting_started/04_energy_functionals`. The Jacobian is then the
# second variation of the energy, which is symmetric. The nonlinear
# diffusion above is the variation of no energy, and its Jacobian is not
# symmetric: its term $2 u\, \delta u\, \nabla u \cdot \nabla v$ changes
# when $\delta u$ and $v$ swap. The energy
#
# $$
#   J(u) = \int_\Omega \frac{1}{2} (1 + u^2) |\nabla u|^2 - g u \,\mathrm{d}x
# $$
#
# has the variation $\int (1 + u^2) \nabla u \cdot \nabla v + u |\nabla u|^2
# v - g v \,\mathrm{d}x$, of a nonlinear diffusion too,
# $-\nabla \cdot ((1 + u^2) \nabla u) + u |\nabla u|^2 = g$. With $g$ for
# the same exact solution, its residual from {py:meth}`to_residual
# <sumfem.symbolic.Functional.to_residual>` is solved like any other:


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


g = f + u_exact * ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact))
variational = NonlinearEnergy("energy", u, g).to_residual()
print("Energy: %d iterations, error %.1e" % solve(variational))
# -

# The Jacobians of the two problems, assembled at the exact solution, show
# the difference:


# +
def symmetric_jacobian(model):
    disc = discrete.discretize([model], discrete.DomainMaps({domain: marked}))
    disc.function(u).interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
    A = dolfinx.fem.petsc.assemble_matrix(disc[model].jacobian)
    A.assemble()
    symmetric = A.isSymmetric(1e-10)
    A.destroy()
    return symmetric


print("Symmetric Jacobian of the energy:", symmetric_jacobian(variational))
print("Symmetric Jacobian of the nonlinear diffusion:", symmetric_jacobian(newton))
assert symmetric_jacobian(variational) and not symmetric_jacobian(newton)
# -

# A symmetric Jacobian can be solved with methods for symmetric matrices,
# e.g. a Cholesky factorization, or conjugate gradients where it is also
# positive definite.
#
# ## Contributions that are zero
#
# UFL drops the test function from a form whose terms all vanish, e.g. a
# boundary flux with a prescribed value of zero. A model whose residual is
# zero for its inputs returns `ufl.ZeroBaseForm((field.v,))` instead, which
# keeps its equation:


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


print("Equations:", [f.name for f in Flux("no_flux", u, 0).equations])
