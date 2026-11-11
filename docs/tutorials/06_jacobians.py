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

# # Jacobians
#
# This tutorial illustrates how to:
#
# - Solve a nonlinear model with the Jacobian sumfem derives
# - Replace the derived Jacobian of a model by a hand-written one
# - Write a Jacobian as a component of its own, a
#   {py:class}`Jacobian <sumfem.symbolic.Jacobian>`, and attach it to a residual
#
# A problem described by an energy has a symmetric Jacobian, see
# {doc}`12_nonlinear_energy_functionals`.
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
# $$ (nonlinear-diffusion)
#
# the residual and its Jacobian, the derivative of the residual in the
# direction $\delta u$, are
#
# $$
# \begin{aligned}
#   F(u; v) &= \int_\Omega (1 + u^2) \nabla u \cdot \nabla v - f v
#   \,\mathrm{d}x, \\
#   F'(u; v)[\delta u]
#   &= \frac{\mathrm{d}}{\mathrm{d}\epsilon}
#   F(u + \epsilon\, \delta u; v) \Big|_{\epsilon = 0} \\
#   &= \int_\Omega (1 + u^2) \nabla \delta u \cdot \nabla v
#   + 2 u\, \delta u\, \nabla u \cdot \nabla v \,\mathrm{d}x,
# \end{aligned}
# $$ (newton-jacobian)
#
# and Newton's method converges quadratically.

# +
from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Source
from sumfem.post_processing import error

# -

# ### The problem
#
# The unknown is in quadratic Lagrange elements on a square. The source $f$
# comes from a manufactured solution, a quadratic, which these elements
# represent exactly, so the error shows the solver alone. The exact
# solution is a function of the coordinates, for the symbolic source as for
# the error and the interpolation later.


# +
def exact(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


cell_name = "triangle"
domain = symbolic.Domain("square", cell_name)
u = symbolic.Field("u", domain, ("Lagrange", 2))
x = ufl.SpatialCoordinate(domain)
u_exact = exact(x)
f = -ufl.div((1 + u_exact**2) * ufl.grad(u_exact))
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

# A helper discretizes a model on the mesh, starts Newton's method from a
# constant, solves, and returns the number of iterations and the error.


# +
def solve(model):
    disc = discrete.discretize([model], discrete.DomainMaps({domain: marked}))
    disc.set({u: 1.0})
    solver = solvers.Solver(disc[model], bcs)
    solver.solve()
    accuracy = error(disc[u], exact)
    return solver.iterations, accuracy


# -

# The nonlinear diffusion is a {py:class}`Residual <sumfem.symbolic.Residual>`
# that returns its weak form; the source is the
# {py:class}`Source <sumfem.models.Source>` model.


# +
class NonlinearDiffusion(symbolic.Residual):
    """``((1 + u^2) grad(u), grad(v))``."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u = self.field
        return ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(u.v)) * u.domain.dx()


# -

# ### Newton's method
#
# The diffusion and the source, summed, solved with the derived Jacobian:

newton = NonlinearDiffusion("diffusion", u) + Source("source", u, f)
print("Newton: %d iterations, error %.1e" % solve(newton))

# ## Hand-written Jacobians
#
# A model may give its own linearization in {py:meth}`jacobian
# <sumfem.symbolic.Residual.jacobian>`, as a form with the trial function
# {py:attr}`u.du <sumfem.symbolic.Field.du>`.
#
# ### Picard's method, in a model
#
# The Picard linearization freezes the coefficient,
#
# $$
#   \int_\Omega (1 + u^2) \nabla \delta u \cdot \nabla v \,\mathrm{d}x,
# $$ (picard-jacobian)
#
# dropping the second term of {eq}`newton-jacobian`: each iteration is
# cheaper to assemble, but convergence is only linear. The solution is the same, to
# the solver's tolerance, since only the path to it changes.
#
# With this Jacobian, a step of Newton's method is a Picard iteration. The
# residual is $A(u) u - b$, with $A(w)$ the diffusion with the coefficient
# $1 + w^2$ and $b$ the source, and the Jacobian is $A(u_n)$, so
#
# $$
#   u_{n+1} = u_n - A(u_n)^{-1} \left(A(u_n) u_n - b\right) = A(u_n)^{-1} b,
# $$
#
# with the full step of Newton's method without a line search, sumfem's
# default. That is, $u_{n+1}$ solves
#
# $$
#   -\nabla \cdot ((1 + u_n^2) \nabla u_{n+1}) = f.
# $$ (picard-iteration)


# +
class PicardDiffusion(NonlinearDiffusion):
    """Nonlinear diffusion, linearized with its coefficient frozen."""

    def jacobian(self):
        u = self.field
        return ufl.inner((1 + u**2) * ufl.grad(u.du), ufl.grad(u.v)) * u.domain.dx()


# -

# Summed with the source and solved as before:

picard = PicardDiffusion("diffusion", u) + Source("source", u, f)
picard_result = solve(picard)
print("Picard: %d iterations, error %.1e" % picard_result)

# Summands without a hand-written Jacobian, like the source here, keep
# their derived ones.
#
# ### Picard's method, as a component
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
