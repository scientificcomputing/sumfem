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

# # Units
#
# This tutorial illustrates how to:
#
# - Give fields and parameters physical units, with
#   [sympy.physics.units](https://docs.sympy.org/latest/modules/physics/units/)
# - Check that every term of an equation has the same dimension
# - Scale the equations of a system by their typical sizes
# - Find the dimensionless groups of a model
# - See what scaling changes for Newton's method
#
# The dimensional analysis, the scaling and the dimensionless groups come
# from [ufl-units](https://github.com/fenics-dolfiny/ufl-units)
# {cite}`Habera_ufl-units_Software_for`, which analyzes UFL forms; sumfem
# gives it the units and scales of the fields, parameters and domains of a
# model, one equation at a time.
#
# ## Values in SI, units in the description
#
# A field or parameter carries the unit it is written in. Its values are
# stored in SI base units, so models never convert, and values are set in
# the unit of the terminal, or as a sympy quantity with its own unit.
#
# We consider a species that diffuses along a rod of length $L = 1$ µm and
# decays at the rate $k$,
#
# $$
#   -D \frac{\mathrm{d}^2 c}{\mathrm{d}x^2} + k c = 0, \qquad c(0) = c_0, \qquad
#   \frac{\mathrm{d}c}{\mathrm{d}x}(L) = 0,
# $$
#
# with the solution $c = c_0 \cosh((L - x) / \lambda) / \cosh(L / \lambda)$
# and the decay length $\lambda = \sqrt{D / k}$.

# +
import time

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import sympy.physics.units as syu
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Diffusion, RateReaction, Reaction, Source, arrhenius
from sumfem.post_processing import error
from sumfem.symbolic.si import si_factor

um = syu.micrometer
uM = syu.micro * syu.mol / syu.liter
s = syu.second
# -

# The mesh is in metres, as every value. The domain's `length_scale` is the
# typical length of the problem, which the dimensional analysis uses for
# gradients.

L = 1e-6
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 50, [0.0, L])
marked = discrete.MeshData.from_locators(
    mesh, {0: {"Left": lambda x: np.isclose(x[0], 0.0)}}
)
domain = symbolic.Domain.from_mesh("rod", mesh, length_scale=L)

# The concentration is in micromolar, and its `scale` is its typical size,
# in that unit. The parameters' defaults are in their own units.

c = symbolic.Field("c", domain, ("Lagrange", 2), unit=uM, scale=1.0)
D = symbolic.Parameter("D", domain, default=10.0, unit=um**2 / s, doc="Diffusivity")
k = symbolic.Parameter("k", domain, default=10.0, unit=1 / s, doc="Decay rate")
model = Diffusion("diffusion", c, D) + Reaction("decay", {c: 1}, {}, k)

# ## Checking units
#
# {py:func}`build_system <sumfem.symbolic.build_system>` collects the equations of
# a model. {py:func}`check_units <sumfem.symbolic.check_units>` finds the
# dimension of every term and raises a
# {py:class}`UnitError <sumfem.symbolic.UnitError>` naming the equation where they
# differ.

system = symbolic.build_system(model)
symbolic.check_units(system)
print("Dimension of each equation:", symbolic.equation_dimensions(system))

# A decay rate given per metre instead of per second does not fit:

wrong = symbolic.Parameter("k_wrong", domain, default=10.0, unit=1 / syu.m)
try:
    symbolic.check_units(
        symbolic.build_system(Diffusion("d", c, D) + Reaction("r", {c: 1}, {}, wrong))
    )
except symbolic.UnitError as e:
    print("UnitError:", str(e).splitlines()[0])

# ## Scaling the equations
#
# In SI units the terms of this equation are tiny: a rate of about
# $10^{-2}$ mol/(m$^3$ s), integrated over a rod of a micrometre. Newton's
# tolerances are absolute and relative, and tiny residuals make the absolute
# one meaningless.
# {py:func}`equation_scales <sumfem.symbolic.equation_scales>` estimates the size
# of each equation from the scales of its quantities, and
# `factorize_equations=True` divides each equation by it. The unknowns keep
# their SI values.

# +
print("Typical size of each equation:", symbolic.equation_scales(system))
scaled = symbolic.build_system(model, check_units=True, factorize_equations=True)

disc = discrete.discretize([scaled], discrete.DomainMaps({domain: marked}))
solvers.Solver(disc[scaled], [symbolic.DirichletBC(c, "Left", 1.0)]).solve()
# -

# Values come back in SI: mol/m$^3$ for the concentration. Dividing by the
# SI factor of µM gives micromolar.

# +
decay_length = np.sqrt(10.0 / 10.0) * 1e-6


def exact(x):
    return ufl.cosh((L - x[0]) / decay_length) / np.cosh(L / decay_length)


relative = error(disc[c], lambda x: si_factor(uM) * exact(x), relative=True)
print(f"Relative L2 error: {relative:.2e}")
assert relative < 1e-5
# -

# Values can also be set with units of their own, which are converted:

disc.set({k: 600.0 / syu.minute})
print(f"k in SI: {float(disc.value(k)):.1f} 1/s")

# ## Dimensionless groups
#
# {py:func}`pi_groups <sumfem.symbolic.pi_groups>` finds the dimensionless groups of
# a system by Buckingham's Pi theorem {cite}`Buckingham1914`, with their
# values from the parameters' defaults and the scales. Here it is the ratio
# of the length of the domain to the decay length, the Thiele modulus.

for group in symbolic.pi_groups(system):
    print(f"{group.expression} = {group.value:.3g}")

# ## Scaling and Newton's method
#
# Newton's method takes the same steps whether the equations are scaled or
# not: scaling an equation scales its row of the Jacobian as well, so the
# update is the same {cite}`Deuflhard2011` (Sections 1.2.2 and 2.1). What
# scaling changes is when Newton stops. We consider heat and hydrogen in a
# wall, as in [FESTIM](https://github.com/festim-dev/festim): the
# temperature is of $10^3$ K, the concentration of hydrogen of $10^{22}$
# m$^{-3}$, and the diffusivity of hydrogen follows the temperature by the
# Arrhenius law,
#
# $$
#   -\frac{\mathrm{d}}{\mathrm{d}x}\left(\kappa \frac{\mathrm{d}T}{\mathrm{d}x}\right)
#   + \sigma T^4 = f, \qquad
#   -\frac{\mathrm{d}}{\mathrm{d}x}\left(D_0 e^{-E / (k_B T)}
#   \frac{\mathrm{d}c}{\mathrm{d}x}\right) = g,
# $$
#
# on a wall of thickness $L$. The sources $f$ and $g$ are chosen so that
#
# $$
#   T = T_c + (T_h - T_c) \cos\left(\frac{\pi x}{2 L}\right), \qquad
#   c = c_h \cos\left(\frac{\pi x}{2 L}\right),
# $$
#
# with the hot face at $x = 0$ and the cold face at $x = L$, where the
# temperature and the concentration are given. The sources are written with
# the parameters of the model, so their units are checked as well, and the
# Boltzmann constant of {py:func}`arrhenius <sumfem.models.arrhenius>` is
# {py:func}`boltzmann_constant <sumfem.symbolic.boltzmann_constant>`, in J/K.

# +
K, W, m, J = syu.kelvin, syu.watt, syu.meter, syu.joule

thickness = 1e-2
wall_mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 100, [0.0, thickness])
wall_marked = discrete.MeshData.from_locators(
    wall_mesh,
    {
        0: {
            "Hot": lambda x: np.isclose(x[0], 0.0),
            "Cold": lambda x: np.isclose(x[0], thickness),
        }
    },
)
wall = symbolic.Domain.from_mesh("wall", wall_mesh, length_scale=thickness)
T = symbolic.Field("T", wall, ("Lagrange", 1), unit=K, scale=1000.0)
c_H = symbolic.Field("c_H", wall, ("Lagrange", 1), unit=1 / m**3, scale=1e22)
kappa = symbolic.Parameter(
    "kappa", wall, default=400.0, unit=W / (m * K), doc="Thermal conductivity"
)
sigma = symbolic.Parameter(
    "sigma", wall, default=1e-4, unit=W / (m**3 * K**4), doc="Loss coefficient"
)
D_0 = symbolic.Parameter("D_0", wall, default=1e-7, unit=m**2 / s)
E = symbolic.Parameter("E", wall, default=3.2e-20, unit=J, doc="Activation energy")
L_wall = symbolic.Parameter("L", wall, default=thickness, unit=m, doc="Thickness")
T_hot = symbolic.Parameter("T_hot", wall, default=1000.0, unit=K)
T_cold = symbolic.Parameter("T_cold", wall, default=300.0, unit=K)
c_hot = symbolic.Parameter("c_hot", wall, default=1e22, unit=1 / m**3)


def diffusivity(temperature):
    return arrhenius(D_0, E, temperature)


x = ufl.SpatialCoordinate(wall)
profile = ufl.cos(ufl.pi * x[0] / (2 * L_wall))
T_exact = T_cold + (T_hot - T_cold) * profile
c_exact = c_hot * profile
transport = (
    Diffusion("conduction", T, kappa)
    + RateReaction("loss", {T: 1}, {}, rate=sigma * T**4)
    + Source("f", T, -ufl.div(kappa * ufl.grad(T_exact)) + sigma * T_exact**4)
    + Diffusion("hydrogen", c_H, diffusivity(T))
    + Source("g", c_H, -ufl.div(diffusivity(T_exact) * ufl.grad(c_exact)))
)
wall_bcs = [
    symbolic.DirichletBC(T, "Hot", T_hot),
    symbolic.DirichletBC(T, "Cold", T_cold),
    symbolic.DirichletBC(c_H, "Hot", c_hot),
    symbolic.DirichletBC(c_H, "Cold", 0.0),
]
# -

# We sweep the temperature of the hot face, each solve starting from the
# last solution, as in a parameter study or a time step. The first solve
# starts from linear profiles, which meet the Dirichlet conditions:
# unscaled, their rows of the residual are in SI, and would otherwise
# dominate it. The sweep returns the total number of Newton iterations, the
# time it took, and the discretization holding the last solution.


# +
def sweep(factorize: bool, hot_temperatures: np.ndarray):
    system = symbolic.build_system(
        transport, check_units=True, factorize_equations=factorize
    )
    disc = discrete.discretize([system], discrete.DomainMaps({wall: wall_marked}))
    solver = solvers.Solver(disc[system], wall_bcs)
    disc.function(T).interpolate(lambda x: 1000.0 - 700.0 * x[0] / thickness)
    disc.function(c_H).interpolate(lambda x: 1e22 * (1 - x[0] / thickness))
    iterations = 0
    start = time.perf_counter()
    for value in hot_temperatures:
        disc.set({T_hot: value})
        solver.solve()
        iterations += solver.iterations
    elapsed = time.perf_counter() - start
    return iterations, elapsed, disc


# -

# We sweep the hot face from 1000 to 1100 K in steps of 10 K, unscaled and
# scaled, and compare the last solutions with the exact ones, at
# $T_h = 1100$ K.

# +
hot_temperatures = 1000.0 + 10.0 * np.arange(11)


def T_last(x):
    return 300.0 + 800.0 * ufl.cos(ufl.pi * x[0] / (2 * thickness))


def c_last(x):
    return 1e22 * ufl.cos(ufl.pi * x[0] / (2 * thickness))


iterations, errors = {}, {}
for factorize in (False, True):
    iterations[factorize], elapsed, disc = sweep(factorize, hot_temperatures)
    errors[factorize] = (
        error(disc[T], T_last, relative=True),
        error(disc[c_H], c_last, relative=True),
    )
    print(
        f"factorize_equations={factorize}: {iterations[factorize]} Newton "
        f"iterations, {elapsed * 1e3:.0f} ms, relative L2 errors "
        f"{errors[factorize][0]:.6e} (T), {errors[factorize][1]:.6e} (c_H)"
    )
assert iterations[True] < iterations[False]
assert np.allclose(errors[True], errors[False], rtol=1e-4)
# -

# Both solutions are as accurate, to the digits of the error of the
# discretization, but the scaled sweep takes fewer iterations. Unscaled, the
# absolute tolerance is in SI and means nothing, so Newton stops when its
# update is small next to the solution, a norm over both fields that the
# concentration dominates. It takes one more iteration in each solve to see
# that, which only reduces an error already far below that of the
# discretization. Scaled, Newton iterates on the unknowns divided by their
# scales, and the rows of the Dirichlet conditions are divided by them too,
# so the residual of every equation is of size one, and Newton stops as soon
# as it is below the absolute tolerance of
# {py:data}`FACTORIZED_PETSC_OPTIONS <sumfem.solvers.solver.FACTORIZED_PETSC_OPTIONS>`.
# The entries of the Jacobian are then of size one as well, instead of
# around $10^{-22}$ for the concentration, which a direct solver with an
# absolute test for zero pivots, such as PETSc's own LU, needs. An iterative
# solver or a line search also sees the sizes of the equations.

# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
