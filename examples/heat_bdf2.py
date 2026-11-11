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

# # A time loop written by hand
#
# This demo illustrates how to:
#
# - Turn a transient residual into the residual of one time step, with the
#   multistep scheme {py:class}`BDF <sumfem.solvers.BDF>` of order two
# - Build that step's system once and solve it with a
#   {py:class}`Solver <sumfem.solvers.Solver>`
# - Write the time loop yourself: set the time and the scheme's weights,
#   solve, and shift the history of previous values
#
# A {py:class}`Simulation <sumfem.solvers.Simulation>` runs the same loop for
# you, and adds output, metadata and adaptive steps, see
# {doc}`../docs/tutorials/04_time_stepping`. A problem that needs none of
# that needs no simulation.
#
# ## Equation and problem definition
#
# The heat equation on the unit square $\Omega$: find $u$ such that
#
# $$
# \begin{aligned}
#   \frac{\partial u}{\partial t} - \Delta u &= f \quad \text{in } \Omega, \\
#   u &= u_D \quad \text{on } \partial\Omega,
# \end{aligned}
# $$ (heat-bdf2-equation)
#
# in weak form, for all $v$ that vanish on the boundary,
#
# $$
# \int_\Omega \frac{\partial u}{\partial t} v~\mathrm{d}x
# + \int_\Omega \nabla u \cdot \nabla v~\mathrm{d}x
# - \int_\Omega f v~\mathrm{d}x = 0.
# $$ (heat-bdf2-weak)
#
# The exact solution and its source are
#
# $$
# \begin{aligned}
#   u_D &= 1 + x^2 + 3 y^2 + 1.2 t, \\
#   f &= \frac{\partial u_D}{\partial t} - \Delta u_D.
# \end{aligned}
# $$ (heat-bdf2-exact)
#
# Quadratic elements are exact in space, and the BDF formulas are exact for
# a solution linear in time, also in the first step, which has only one
# previous value and uses backward Euler. The error is therefore at
# round-off in every step.
#
# ## Implementation
#
# The modules that are used are imported:

# +
from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Accumulation, Diffusion, Source
from sumfem.post_processing import error

# -

# The mesh, with its boundary marked as `"Boundary"`:

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
marked = discrete.MeshData.from_locators(
    mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)

# ### The symbolic description
#
# The residual of {eq}`heat-bdf2-weak` is a sum of three components on a
# field $u$. A {py:class}`Clock <sumfem.symbolic.Clock>` holds the time $t$
# as a parameter, so that the boundary value of {eq}`heat-bdf2-exact` can
# depend on it. The exact solution is one function, of UFL or of NumPy
# coordinates, and UFL derives the source from it, with the time as a
# {py:func}`variable <ufl.variable>`:


# +
def u_exact(x, t):
    return 1 + x[0] ** 2 + 3 * x[1] ** 2 + 1.2 * t


domain = symbolic.Domain.from_mesh("square", mesh)
u = symbolic.Field("u", domain, ("Lagrange", 2))
clock = symbolic.Clock(domain)
x = ufl.SpatialCoordinate(domain)
t = ufl.variable(clock.t)
source = ufl.diff(u_exact(x, t), t) - ufl.div(ufl.grad(u_exact(x, t)))
residual = (
    Accumulation("accumulation", u)
    + Diffusion("diffusion", u, 1.0)
    + Source("source", u, source)
)
bc = symbolic.DirichletBC(u, "Boundary", u_exact(x, clock.t))
# -

# ### One time step
#
# BDF2 replaces the time derivative `u.dot` by
#
# $$
# w_0 u + w_1 u^{n} + w_2 u^{n-1},
# $$ (heat-bdf2-formula)
#
# with the previous values `u.previous(1)` and `u.previous(2)` and weights
# that are parameters. {py:meth}`prepare <sumfem.solvers.BDF.prepare>` gives
# the residual and those substitutions, and
# {py:func}`build_system <sumfem.symbolic.build_system>` the step's system,
# whose Jacobian is derived after the substitution:

scheme = solvers.BDF(2)
step, substitutions = scheme.prepare(residual, clock)
system = symbolic.build_system(step, substitutions=substitutions)

# {py:class}`DomainMaps <sumfem.discrete.DomainMaps>` give the mesh of the
# domain. {py:func}`discretize <sumfem.discrete.discretize>` creates on it
# what the system and the boundary condition need: a function for $u$, its
# two previous values, the BDF weights and the time. Indexing the
# discretization with the system gives the discrete system, compiled once,
# and the solver evaluates the boundary value at the current time before
# every solve:

maps = discrete.DomainMaps({domain: marked})
disc = discrete.discretize([system, bc], maps)
solver = solvers.Solver(disc[system], [bc])

# ### The time loop
#
# Before the first step every previous value is the initial condition. Each
# step then sets the time, sets the scheme's weights for the step, which
# BDF2 lowers to backward Euler in the first step, solves, and shifts the
# history: `u.previous(2)` takes the value of `u.previous(1)`, which takes
# the new solution. Every step is exact to round-off:

dt, num_steps = 0.1, 10
disc.set({u: u_exact(x, 0.0)})
disc.initialize_history()
for n in range(1, num_steps + 1):
    t = n * dt
    disc.set({clock.t: t})
    scheme.set_step(n, dt, disc)
    solver.solve()
    disc.advance_history()
    e = error(disc.function(u), lambda x: u_exact(x, t))
    if MPI.COMM_WORLD.rank == 0:
        print(f"t = {t:.1f}: {solver.iterations} Newton iterations, L2 error {e:.2e}")
    assert e < 1e-10
