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
# - Find the dimensionless groups of a model
# - Scale the equations of a system by their typical sizes
# - See what scaling changes for Newton's method
#
# The dimensional analysis, the scaling and the dimensionless groups come
# from [ufl-units](https://github.com/fenics-dolfiny/ufl-units)
# {cite}`Habera_ufl-units_Software_for`, which analyzes UFL forms; sumfem
# gives it the units and scales of the fields, parameters and domains of a
# model, one equation at a time.
#
# ## Heat and hydrogen in a wall
#
# We consider heat and hydrogen in a wall of thickness $L$, as in
# [FESTIM](https://github.com/festim-dev/festim). Heat is conducted and lost
# at a rate $\sigma T^4$, and hydrogen diffuses with a diffusivity that
# follows the temperature by the Arrhenius law,
#
# $$
#   -\frac{\mathrm{d}}{\mathrm{d}x}\left(\kappa \frac{\mathrm{d}T}{\mathrm{d}x}\right)
#   + \sigma T^4 = 0, \qquad
#   -\frac{\mathrm{d}}{\mathrm{d}x}\left(D_0 e^{-E / (k_B T)}
#   \frac{\mathrm{d}c}{\mathrm{d}x}\right) = 0,
# $$
#
# with the temperature and the concentration given on both faces. The
# temperature is of $10^3$ K, and the concentration of $10^{22}$ m$^{-3}$.

# +
import logging
import time

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import sympy.physics.units as syu
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Diffusion, RateReaction, Source, arrhenius
from sumfem.post_processing import error

K, W, m, s = syu.kelvin, syu.watt, syu.meter, syu.second
# -

# ## Values in SI, units in the description
#
# A field or parameter carries the unit it is written in. Its values are
# stored in SI base units, so models never convert, and values are set in
# the unit of the terminal, or as a sympy quantity with its own unit.
#
# The mesh is in metres, as every value. The domain's `length_scale` is the
# typical length of the problem, which the dimensional analysis uses for
# gradients.

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

# The fields' `scale` is their typical size, in their unit. The parameters'
# defaults are in their own units: the diffusivity in mm$^2$/s and the
# activation energy in eV.

# +
T = symbolic.Field("T", wall, ("Lagrange", 1), unit=K, scale=1000.0)
c_H = symbolic.Field("c_H", wall, ("Lagrange", 1), unit=1 / m**3, scale=1e22)
kappa = symbolic.Parameter(
    "kappa", wall, default=400.0, unit=W / (m * K), doc="Thermal conductivity"
)
sigma = symbolic.Parameter(
    "sigma", wall, default=1e-4, unit=W / (m**3 * K**4), doc="Loss coefficient"
)
D_0 = symbolic.Parameter(
    "D_0", wall, default=0.1, unit=syu.millimeter**2 / s, doc="Diffusivity factor"
)
E = symbolic.Parameter(
    "E", wall, default=0.2, unit=syu.electronvolt, doc="Activation energy"
)


def diffusivity(temperature):
    return arrhenius(D_0, E, temperature)


transport = (
    Diffusion("conduction", T, kappa)
    + RateReaction("loss", {T: 1}, {}, rate=sigma * T**4)
    + Diffusion("hydrogen", c_H, diffusivity(T))
)
# -

# The Boltzmann constant of {py:func}`arrhenius <sumfem.models.arrhenius>` is
# {py:func}`boltzmann_constant <sumfem.symbolic.boltzmann_constant>`, the
# parameter `"wall.k_B"` in J/K, so the exponent is checked to be
# dimensionless as well.
#
# ## Checking units
#
# {py:func}`build_system <sumfem.symbolic.build_system>` collects the equations of
# a model. {py:func}`check_units <sumfem.symbolic.check_units>` finds the
# dimension of every term and raises a
# {py:class}`UnitError <sumfem.symbolic.UnitError>` naming the equation where they
# differ.

system = symbolic.build_system(transport)
symbolic.check_units(system)
print("Dimension of each equation:", symbolic.equation_dimensions(system))

# A loss coefficient in W/(m$^2$ K$^4$), the unit of the Stefan–Boltzmann
# constant for radiation from a surface, does not fit a loss per volume:

per_area = symbolic.Parameter("sigma_area", wall, default=1e-4, unit=W / (m**2 * K**4))
try:
    symbolic.check_units(
        symbolic.build_system(
            Diffusion("d", T, kappa) + RateReaction("r", {T: 1}, {}, per_area * T**4)
        )
    )
except symbolic.UnitError as e:
    print("UnitError:", str(e).splitlines()[0])

# ## Dimensionless groups
#
# {py:func}`pi_groups <sumfem.symbolic.pi_groups>` finds the dimensionless groups of
# a system by Buckingham's Pi theorem {cite}`Buckingham1914`, with their
# values from the parameters' defaults and the scales. Any product of powers
# of dimensionless groups is one too, so they are not unique. By default,
# they are those ufl-units finds, which depend on the order in which the
# quantities appear in the forms. ufl-units logs its analysis, and flags
# groups far from the others; we print the groups ourselves.

logging.getLogger("ufl_units").setLevel(logging.ERROR)
for group in symbolic.pi_groups(system):
    print(f"{group.expression} = {group.value:.3g}")

# Which groups describe the problem is clearer from repeating quantities:
# every group is then one other quantity times powers of these. With the
# thickness, the temperature, the conductivity and the Boltzmann constant:

repeating = [wall, T, kappa, symbolic.boltzmann_constant(wall)]
for group in symbolic.pi_groups(system, repeating=repeating):
    print(f"{group.expression} = {group.value:.3g}")

# The first is the ratio of the loss to conduction, small here, and the
# third the Arrhenius number, the activation energy over the thermal
# energy. The other two cannot change the solution, which dimensional
# analysis cannot know: the hydrogen equation is linear in the
# concentration, and the diffusivity factor divides out of it.
#
# ## Scaling the equations
#
# In SI units the terms of these equations are of very different sizes: a
# heat flux of some $10^7$ W/m$^2$, and a flux of hydrogen of some $10^{17}$
# per m$^2$ and second. Newton's tolerances are absolute and relative, and
# such residuals make the absolute one meaningless.
# {py:func}`equation_scales <sumfem.symbolic.equation_scales>` estimates the size
# of each equation from the scales of its quantities, and
# `factorize_equations=True` divides each equation by it.

print("Typical size of each equation:", symbolic.equation_scales(system))

# To see how accurate the solution is, we add sources $f$ and $g$ to the
# equations, so that the solution is
#
# $$
#   T = T_c + (T_h - T_c) \cos\left(\frac{\pi x}{2 L}\right), \qquad
#   c = c_h \cos\left(\frac{\pi x}{2 L}\right),
# $$
#
# with the hot face at $x = 0$ and the cold face at $x = L$. The sources are
# written with parameters of their own, so their units are checked as well;
# the thickness is given in centimetres.

# +
L_wall = symbolic.Parameter("L", wall, default=1.0, unit=syu.centimeter)
T_hot = symbolic.Parameter("T_hot", wall, default=1000.0, unit=K)
T_cold = symbolic.Parameter("T_cold", wall, default=300.0, unit=K)
c_hot = symbolic.Parameter("c_hot", wall, default=1e22, unit=1 / m**3)

x = ufl.SpatialCoordinate(wall)
profile = ufl.cos(ufl.pi * x[0] / (2 * L_wall))
T_exact = T_cold + (T_hot - T_cold) * profile
c_exact = c_hot * profile
manufactured = (
    transport
    + Source("f", T, -ufl.div(kappa * ufl.grad(T_exact)) + sigma * T_exact**4)
    + Source("g", c_H, -ufl.div(diffusivity(T_exact) * ufl.grad(c_exact)))
)
wall_bcs = [
    symbolic.DirichletBC(T, "Hot", T_hot),
    symbolic.DirichletBC(T, "Cold", T_cold),
    symbolic.DirichletBC(c_H, "Hot", c_hot),
    symbolic.DirichletBC(c_H, "Cold", 0.0),
]
# -

# The scaled system is solved from a uniform 300 K and no hydrogen, far from
# the values on the hot face. Newton iterates on the unknowns divided by
# their scales, and the rows of the Dirichlet conditions are divided by them
# too, so these rows are of size one like the others.

# +
scaled = symbolic.build_system(manufactured, check_units=True, factorize_equations=True)
disc = discrete.discretize([scaled], discrete.DomainMaps({wall: wall_marked}))
disc.function(T).x.array[:] = 300.0
solver = solvers.Solver(disc[scaled], wall_bcs)
solver.solve()


def T_solution(x, hot=1000.0):
    return 300.0 + (hot - 300.0) * ufl.cos(ufl.pi * x[0] / (2 * thickness))


def c_solution(x):
    return 1e22 * ufl.cos(ufl.pi * x[0] / (2 * thickness))


T_error = error(disc[T], T_solution, relative=True)
c_error = error(disc[c_H], c_solution, relative=True)
print(
    f"{solver.iterations} Newton iterations, relative L2 errors {T_error:.2e} (T), "
    f"{c_error:.2e} (c_H)"
)
assert T_error < 1e-4 and c_error < 1e-4
# -

# {py:meth}`value <sumfem.discrete.Discretization.value>` gives a parameter's
# value in its own unit. The function holding it, which the forms use, has
# its SI value: the thickness in metres, the diffusivity factor in m$^2$/s
# and the activation energy in J.

for parameter in (L_wall, D_0, E):
    own, si = float(disc.value(parameter)), disc.function(parameter).x.array[0]
    print(f"{parameter.name} = {own:.3g} in its unit, {si:.4g} in SI")

# Values can also be set with units of their own, which are converted:

disc.set({kappa: 4.0 * W / (syu.centimeter * K)})
print(f"kappa = {float(disc.value(kappa)):.1f} W/(m K)")

# ## Scaling and Newton's method
#
# Newton's method takes the same steps whether the equations are scaled or
# not: scaling an equation scales its row of the Jacobian as well, so the
# update is the same {cite}`Deuflhard2011` (Sections 1.2.2 and 2.1). What
# scaling changes is when Newton stops. We sweep the temperature of the hot
# face, each solve starting from the last solution, as in a parameter study
# or a time step. The first solve starts from linear profiles, which meet
# the Dirichlet conditions: unscaled, their rows of the residual are in SI,
# and would otherwise dominate it. The sweep returns the total number of
# Newton iterations, the time it took, and the discretization holding the
# last solution.


# +
def sweep(factorize: bool, hot_temperatures: np.ndarray):
    system = symbolic.build_system(
        manufactured, check_units=True, factorize_equations=factorize
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
iterations, errors = {}, {}
for factorize in (False, True):
    iterations[factorize], elapsed, disc = sweep(factorize, hot_temperatures)
    errors[factorize] = (
        error(disc[T], lambda x: T_solution(x, hot=1100.0), relative=True),
        error(disc[c_H], c_solution, relative=True),
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
# discretization. Scaled, the residual of every equation is of size one, and
# Newton stops as soon as it is below the absolute tolerance of
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
