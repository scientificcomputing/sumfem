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

# # Preconditioners
#
# This page illustrates how to:
#
# - Give a model a preconditioner, as a form to
#   {py:class}`sumfem.symbolic.UFLResidual` or as a
#   {py:class}`sumfem.symbolic.Jacobian` component
# - Solve with an iterative solver through PETSc options
#
# ## The problem
#
# We solve $-\nabla \cdot (k \nabla u) = f$ on the unit square with the
# conductivity $k = 1 + x^2$, $u = u_D = 1 + x^2 + 2y^2$ on the boundary, and
# $f = -\nabla \cdot (k \nabla u_D)$, so that the exact solution is $u_D$.
#
# Newton's method solves a linear system with the Jacobian in every
# iteration. An iterative solver needs a preconditioner, which is built from
# the Jacobian unless the model gives another operator. The Laplacian with a
# constant conductivity, $\int \nabla \delta u \cdot \nabla v \,\mathrm{d}x$, is
# spectrally equivalent to the Jacobian, with a condition number bounded by
# the ratio of the largest and smallest conductivity. It does not change with
# the conductivity, so in a nonlinear or time-dependent problem it is set up
# once.

# +
from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl
from basix.ufl import element

from sumfem import discrete, solvers, symbolic
from sumfem.post_processing import error

# -

# ## A preconditioner in plain UFL
#
# {py:class}`UFLResidual <sumfem.symbolic.UFLResidual>` takes the
# preconditioner as a bilinear form, with the same arguments as the Jacobian:

mesh = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, element("Lagrange", "triangle", 2))
uh, v, du = ufl.Coefficient(V), ufl.TestFunction(V), ufl.TrialFunction(V)
xh = ufl.SpatialCoordinate(mesh)
k = 1 + xh[0] ** 2
u_D = 1 + xh[0] ** 2 + 2 * xh[1] ** 2
f = -ufl.div(k * ufl.grad(u_D))
F = k * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
P = ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx

# From the same residual we create two models: a standard one, without a
# preconditioner, and one preconditioned by `P`, the Laplacian with a constant
# conductivity.

standard_problem = symbolic.UFLResidual("standard", F, uh, names={"square": mesh})
preconditioned_problem = symbolic.UFLResidual(
    "preconditioned", F, uh, preconditioner=P, names={"square": mesh}
)

# ## A preconditioner as a component
#
# A preconditioner is also a {py:class}`Jacobian <sumfem.symbolic.Jacobian>`,
# a component that returns a bilinear form with the trial function `u.du`,
# and {py:meth}`with_preconditioner
# <sumfem.symbolic.Residual.with_preconditioner>` attaches it to any
# residual, here the one without a preconditioner:


# +
class Laplacian(symbolic.Jacobian):
    """``(grad(du), grad(v))``."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def jacobian(self):
        u = self.field
        return ufl.inner(ufl.grad(u.du), ufl.grad(u.v)) * u.domain.dx()


laplacian = Laplacian("laplacian", standard_problem.field)
hook_preconditioned_problem = standard_problem.with_preconditioner(laplacian)
# -

# ## Solving
#
# The two plain models share no field, as each `UFLResidual` creates its own
# (`terminals` would give the second the field of the first), and the
# component shares the field of the model it is attached to. So each model is
# discretized on its own, with the Dirichlet condition and the error of its own
# field.

square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
marked = discrete.MeshData.from_locators(
    square, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)

# By default, {py:class}`Solver <sumfem.solvers.Solver>` solves directly, with
# an LU factorization. PETSc factorizes the preconditioner matrix, so with a
# preconditioner it would solve with the Laplacian instead of the Jacobian.
#
# The defaults are overridden by passing PETSc options to the solver. Here
# they choose conjugate gradients, preconditioned by algebraic multigrid
# (GAMG) of the preconditioner, which smooths with the Jacobian unless
# `pc_use_amat` is false. The problem is linear, so the nonlinear solver that
# wraps it does a single linear solve, `ksponly`, and both raise an error if
# they do not converge.

options = {
    "snes_type": "ksponly",
    "snes_error_if_not_converged": True,
    "ksp_type": "cg",
    "pc_type": "gamg",
    "pc_use_amat": False,
    "ksp_rtol": 1e-12,
    "ksp_error_if_not_converged": True,
}

# Each problem is then solved in turn. For each, we take its field, set it to
# the exact solution on the boundary, bind its domain to the mesh, discretize
# it and solve it with these options. The number of CG iterations is read from
# the Krylov solver of the PETSc nonlinear solver, and the solution is compared
# with the exact one.


def exact(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


# The domains of the plain models are named `"square"`, so one map binds the
# mesh data to all of them by name:

maps = discrete.DomainMaps({"square": marked})
cases = [
    ("the Jacobian", standard_problem),
    ("the Laplacian form", preconditioned_problem),
    ("the Laplacian component", hook_preconditioned_problem),
]
for name, residual in cases:
    (u,) = residual.equations
    x = ufl.SpatialCoordinate(u.domain)
    bcs = [symbolic.DirichletBC(u, "Boundary", exact(x))]
    disc = discrete.discretize([residual], maps)
    solver = solvers.Solver(disc[residual], bcs, petsc_options=options)
    solver.solve()
    krylov = solver.nonlinear_problem.solver.getKSP().getIterationNumber()
    print(
        f"Preconditioned by {name}: {krylov} CG iterations, "
        f"L2 error {error(disc[u], exact):.2e}"
    )
    assert error(disc[u], exact) < 1e-8

# The form and the component are the same preconditioner, so they take the
# same iterations. The Laplacian is only an approximation of the Jacobian, so
# CG needs more iterations with it, the more the larger the contrast of the
# conductivity: for $k = 1 + 100 x^2$ it takes about five times as many as with
# the Jacobian (137 against 27).
#
# What such a preconditioner offers is that it depends neither on the solution
# nor on $k$. In a nonlinear or time-dependent problem, its multigrid setup can
# then be kept while the Jacobian changes, e.g. with the PETSc option
# `snes_lag_preconditioner`, which a single linear solve as here has no use
# for.
