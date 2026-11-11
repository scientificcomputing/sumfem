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

# # Time stepping
#
# This tutorial illustrates how to:
#
# - Write a time derivative in UFL with the placeholder `u.dot`, and with
#   {py:class}`Accumulation <sumfem.models.Accumulation>`
# - Step in time with a {py:class}`Simulation <sumfem.solvers.Simulation>`,
#   with a backward differentiation formula or a theta scheme: backward
#   Euler, Crank-Nicolson or forward Euler
# - Write output and metadata, and adapt the time step
#
# ## The problem
#
# The heat equation on the unit square $\Omega$,
#
# $$
# \begin{aligned}
#   \frac{\partial u}{\partial t} - \Delta u &= f \quad \text{in } \Omega, \\
#   u &= u_{\text{exact}} \quad \text{on } \partial\Omega,
# \end{aligned}
# $$ (heat-equation)
#
# has the exact solution
#
# $$
# \begin{aligned}
#   u_{\text{exact}} &= (1 + x^2 + 2 y^2)(1 + \sin t), \\
#   f &= \frac{\partial u_{\text{exact}}}{\partial t} - \Delta u_{\text{exact}}.
# \end{aligned}
# $$ (heat-exact)
#
# Quadratic Lagrange elements represent the solution exactly in space, so
# all of the error comes from the time stepping. The modules used are
# imported:

# +
import json
import pathlib
import tempfile

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Accumulation, Diffusion, Source
from sumfem.post_processing import error

# -

# ## Time derivatives
#
# The time derivative of a field `u` is `u.dot`, a placeholder that a
# time-stepping scheme replaces by differences of the solution at the
# current and previous steps. The time-derivative term of the weak form of
# {eq}`heat-equation`, $(\partial u / \partial t, v)$, is written in UFL
# with it like any other term, here as a component, see
# {doc}`../getting_started/02_components`:

# +
domain = symbolic.Domain("square", "triangle")
u = symbolic.Field("u", domain, ("Lagrange", 2))


class TimeDerivative(symbolic.Residual):
    def residual(self) -> ufl.Form:
        return ufl.inner(u.dot, u.v) * domain.dx()


# -

# {py:class}`Accumulation <sumfem.models.Accumulation>` is this term in
# general, for any field, with a weight such as a heat capacity or a
# porosity, and on the cells of given markers. Without them it is the same
# form:

assert (
    TimeDerivative("dudt").form.signature() == Accumulation("dudt", u).form.signature()
)

# A {py:class}`Clock <sumfem.symbolic.Clock>` holds the time `clock.t` and the
# step `clock.dt` as parameters, so models can use them in expressions, e.g.
# a source or a boundary value that changes in time. UFL derives the source
# of {eq}`heat-exact` from the exact solution: with the time as a
# {py:func}`variable <ufl.variable>`, {py:func}`diff <ufl.diff>`
# differentiates with respect to it, and the divergence of the gradient is
# the Laplacian. The model of {eq}`heat-equation`, with this source and the
# exact solution on the boundary:

clock = symbolic.Clock(domain, dt=0.1)
x = ufl.SpatialCoordinate(domain)
t = ufl.variable(clock.t)
spatial = 1 + x[0] ** 2 + 2 * x[1] ** 2
exact = spatial * (1 + ufl.sin(t))
source = ufl.diff(exact, t) - ufl.div(ufl.grad(exact))
model = TimeDerivative("dudt") + Diffusion("diffusion", u, 1.0) + Source("f", u, source)
bcs = [symbolic.DirichletBC(u, "Boundary", exact)]

# The mesh, with its whole boundary marked:

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
marked = discrete.MeshData.from_locators(
    mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
)


# ## A simulation
#
# A {py:class}`Simulation <sumfem.solvers.Simulation>` builds the system of
# one step once and takes the steps. Boundary values that depend on the time
# are evaluated again before every step. The clock is declared with the
# model, so that its step can be set, and the initial value is set with an
# expression of the spatial coordinate. The function below runs to a given
# time with a time-stepping scheme and a step, and returns the error in the
# $L^2$ norm:


# +
def run(scheme: solvers.TimeScheme, dt: float, t_end: float = 1.0) -> float:
    disc = discrete.discretize([model, clock], discrete.DomainMaps({domain: marked}))
    disc.set({clock.dt: dt})
    simulation = solvers.Simulation(disc, model, bcs, clock=clock, scheme=scheme)
    disc.set({u: spatial})
    simulation.run(t_end=t_end)
    return error(
        disc[u], lambda x: (1 + x[0] ** 2 + 2 * x[1] ** 2) * (1 + np.sin(t_end))
    )


# -

# The backward differentiation formula {py:class}`BDF <sumfem.solvers.BDF>`
# of order $p$ has an error proportional to $\Delta t^p$, so halving the
# step divides the error by $2^p$. The observed rates, the base-two
# logarithms of the ratios of successive errors, approach the order:

for order in (1, 2):
    errors = np.array([run(solvers.BDF(order), dt) for dt in (0.1, 0.05, 0.025)])
    rates = np.log2(errors[:-1] / errors[1:])
    print(f"BDF{order}: errors {errors}, rates {rates.round(2)}")
    assert np.all(rates > order - 0.15)

# ## Theta schemes
#
# The {py:class}`Theta <sumfem.solvers.Theta>` method is implemented too.
# Writing a model's residual as $M(\partial u / \partial t) + F(u, t)$, with
# $M$ linear in the time derivative, one step is
#
# $$
# M\left(\frac{u - u^n}{\Delta t}\right)
# + \theta F(u, t) + (1 - \theta) F(u^n, t^n) = 0,
# $$ (theta-step)
#
# which is backward Euler for $\theta = 1$, the same step as BDF of order one,
# {py:class}`CrankNicolson <sumfem.solvers.CrankNicolson>`, of second order,
# for $\theta = 1/2$, and {py:class}`ForwardEuler <sumfem.solvers.ForwardEuler>`
# for $\theta = 0$. Every other $\theta$ is of first order. Forward Euler is
# explicit: each step solves with the time-derivative terms alone, and is
# stable only for steps far below the ones above. Crank-Nicolson converges at
# second order:

errors = np.array([run(solvers.CrankNicolson(), dt) for dt in (0.1, 0.05, 0.025)])
rates = np.log2(errors[:-1] / errors[1:])
print(f"Crank-Nicolson: errors {errors}, rates {rates.round(2)}")
assert np.all(rates > 2 - 0.15)

# ## Output and metadata
#
# With an {py:class}`Output <sumfem.solvers.Output>`, a simulation writes its
# fields with ADIOS2 and a `metadata.json` that records the versions, the
# parameters with their units, the model, the time stepping and the meshes.
# A callback is called after every step, e.g. to post-process. The output
# goes to a temporary directory here, removed at the end:

with tempfile.TemporaryDirectory() as temporary:
    directory = pathlib.Path(MPI.COMM_WORLD.bcast(temporary, root=0))
    disc = discrete.discretize([model], discrete.DomainMaps({domain: marked}))
    simulation = solvers.Simulation(
        disc,
        model,
        bcs,
        clock=clock,
        scheme=solvers.BDF(2),
        output=solvers.Output(directory, every=5),
    )
    disc.set({u: spatial})
    simulation.run(
        t_end=1.0,
        callback=lambda t, step: step % 5 or print(f"t = {t:.1f}, step {step}"),
    )
    MPI.COMM_WORLD.barrier()
    files = sorted(p.name for p in directory.iterdir())
    metadata = json.loads((directory / "metadata.json").read_text())
    print(files)
    print(metadata["time"])
    assert files == ["metadata.json", "u.bp"]
    assert metadata["time"]["t0"] == 0.0
    MPI.COMM_WORLD.barrier()

# ## Adaptive steps
#
# An {py:class}`AdaptiveSimulation <sumfem.solvers.AdaptiveSimulation>`
# grows the step while Newton's method converges easily and cuts it when it
# struggles, retrying failed steps, by the rules of an
# {py:class}`AdaptiveTimeStep <sumfem.solvers.AdaptiveTimeStep>`. Steps end
# at milestones, e.g. output times, and at the end time. This problem is
# linear, so every step takes one iteration, and the steps grow up to the
# largest one allowed:

disc = discrete.discretize([model], discrete.DomainMaps({domain: marked}))
rules = solvers.AdaptiveTimeStep(
    initial=0.01, growth_factor=1.5, target_iterations=4, max_step=0.2, milestones=[0.5]
)
adaptive = solvers.AdaptiveSimulation(
    disc, model, rules, clock=clock, scheme=solvers.BDF(1), bcs=bcs
)
disc.set({u: spatial})
times: list[float] = []
adaptive.run(t_end=1.0, callback=lambda t, step: times.append(t))
print("Times:", np.round(times, 3))
steps = np.diff(times)
assert steps[1] > steps[0] and steps.max() <= 0.2 + 1e-12
assert np.isclose(times, 0.5).any() and np.isclose(times[-1], 1.0)

# The steps of a simulation can also be written out by hand, with a
# {py:class}`Solver <sumfem.solvers.Solver>`, see the demo
# {doc}`../../examples/heat_bdf2`.
