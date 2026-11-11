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

# # Units and dimensional analysis
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
# {cite}`Habera_ufl-units_Software_for`, which analyzes UFL forms
# {cite}`Habera2026`; sumfem gives it the units and scales of the fields,
# parameters and domains of a model, one equation at a time.
#
# ## Heat and hydrogen in a wall
#
# We consider heat and hydrogen in a wall of thickness $L$, as in
# [FESTIM](https://github.com/festim-dev/festim). Heat is conducted and lost
# at a rate $\sigma T^4$, and hydrogen diffuses with a diffusivity that
# follows the temperature by the Arrhenius law {cite}`Arrhenius1889,Logan1982`,
#
# $$
# \begin{aligned}
#   -\frac{\mathrm{d}}{\mathrm{d}x}\left(\kappa \frac{\mathrm{d}T}{\mathrm{d}x}\right)
#   + \sigma T^4 &= 0, \\
#   -\frac{\mathrm{d}}{\mathrm{d}x}\left(D_0 e^{-E / (k_B T)}
#   \frac{\mathrm{d}c}{\mathrm{d}x}\right) &= 0,
# \end{aligned}
# $$ (wall-equations)
#
# with the temperature and the concentration given on both faces. In SI
# units, the two fields differ by many orders of magnitude.

# +
import logging
import time

from mpi4py import MPI

import dolfinx
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
# Every value sumfem stores is in SI base units, so models never convert:
# meshes are in metres, and fields and parameters hold SI values. Units
# describe the values: a {py:class}`Field <sumfem.symbolic.Field>` or
# {py:class}`Parameter <sumfem.symbolic.Parameter>` carries the unit it is
# written in, values are set in that unit, or as a sympy {py:class}`quantity
# <sympy.physics.units.quantities.Quantity>` with its own unit, and the
# dimensional analysis uses the units and a typical size of everything in
# the equations. The domain, the fields and the parameters each give that
# size in their own way.
#
# (units-length-scale)=
# ### A domain and its length scale
#
# The {py:attr}`length_scale <sumfem.symbolic.Domain.length_scale>` of a
# {py:class}`Domain <sumfem.symbolic.Domain>`, in metres, is the typical
# length of the problem, from which the analysis sizes gradients and
# integrals {cite}`Habera2026` (Section 3.3). It defaults to one metre, so a
# domain of a micrometre or of a kilometre must give its own.
#
# A mesh of nanometres is no problem for the arithmetic: floating point
# numbers are as precise, relative to their size, at $10^{-9}$ as at one.
# What does not scale are absolute thresholds. Newton's tolerances are one,
# which the scaling of the equations below makes meaningful, given this
# length scale. Geometric tolerances are another, which scaling does not
# change: e.g. {py:func}`np.isclose <numpy.isclose>` has the absolute
# tolerance $10^{-8}$ by default, so the locators of the boundaries below
# use one relative to the thickness and to the precision of the coordinates.

thickness = 1e-2
# The precision of PETSc's real type, which assembly and solvers use
dtype = dolfinx.default_real_type
wall_mesh = dolfinx.mesh.create_interval(
    MPI.COMM_WORLD, 100, [0.0, thickness], dtype=dtype
)
# Above the round-off of the coordinates and far below the size of a cell
tolerance = 100 * np.finfo(dtype).eps * thickness
wall_marked = discrete.MeshData.from_locators(
    wall_mesh,
    {
        0: {
            "Hot": lambda x: np.isclose(x[0], 0.0, atol=tolerance),
            "Cold": lambda x: np.isclose(x[0], thickness, atol=tolerance),
        }
    },
)
wall = symbolic.Domain.from_mesh("wall", wall_mesh, length_scale=thickness)

# ### Fields: a unit and a scale
#
# A field is an unknown: it has no value until it is solved for, so it gives
# its typical size, {py:attr}`scale <sumfem.symbolic.Field.scale>`, in its
# own unit. Its order of magnitude is enough: the scale is only used by the
# dimensional analysis and the scaling of the equations, never to compute
# the solution. Values of a field, e.g. those of its Dirichlet conditions,
# are given in its unit.

T = symbolic.Field("T", wall, ("Lagrange", 1), unit=K, scale=1e3)
c_H = symbolic.Field("c_H", wall, ("Lagrange", 1), unit=1 / m**3, scale=1e22)

# ### Parameters: a unit and a value
#
# A parameter has a value, its default or one set later, in its own unit,
# here the diffusivity factor in mm$^2$/s and the activation energy in eV.
# The size of its default is its scale for the dimensional analysis, so give
# a default of the right size: a parameter without one counts as of size
# one.

# +
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
# (units-checking)=
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

# Each equation has a dimension of its own: the heat equation is a power per
# area, W/m$^2$, and the hydrogen equation a number of atoms per area and
# second. The terms within an equation must agree, but different equations
# need not, so each is checked on its own, and later scaled by its own
# size.
#
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
# {py:func}`pi_groups <sumfem.symbolic.pi_groups>` finds the dimensionless
# groups of a system by Buckingham's Pi theorem {cite}`Buckingham1914`, with
# their values from the parameters' defaults and the scales. Any product of
# powers of dimensionless groups is one too, so they are not unique
# {cite}`Habera2026` (Section 2.3), and by default they are hard to read:

# ufl-units logs its analysis; the groups are printed here instead
logging.getLogger("ufl_units").setLevel(logging.ERROR)
for group in symbolic.pi_groups(system):
    print(f"{group.expression} = {group.value:.3g}")

# Which groups describe the problem is clearer from repeating quantities,
# as in the method of repeating variables {cite}`CengelCimbala2010`
# (Section 7.4): every group is then one other quantity times powers of
# these. With the thickness, the temperature, the conductivity and the
# Boltzmann constant:

repeating = [wall, T, kappa, symbolic.boltzmann_constant(wall)]
groups = symbolic.pi_groups(system, repeating=repeating)
for group in groups:
    print(f"{group.expression} = {group.value:.3g}")

# The value of a group is its expression at the defaults and the scales, and
# it compares two effects. About one, both matter; much smaller than one,
# the effect in the numerator is a small correction to that in the
# denominator; much larger, it dominates. The first group is the ratio of
# the loss to conduction: much smaller than one, so the loss is a small
# correction, and the temperature is close to that of conduction alone. The third is the
# Arrhenius number, the activation energy over the thermal energy at the
# reference temperature $T_{ref}$, the scale of the temperature. In terms of
# it the diffusivity is
#
# $$
# \begin{aligned}
#   \mathrm{Ar} &= \frac{E}{k_B T_{ref}}, \\
#   D &= D_0 e^{-\mathrm{Ar}\, T_{ref} / T},
# \end{aligned}
# $$ (arrhenius-number)
#
# so at a temperature $c\, T_{ref}$ it is
#
# $$
#   D = D_0 e^{-\mathrm{Ar} / c},
# $$
#
# so a factor $c$ in the temperature divides the exponent by $c$, and with
# $\mathrm{Ar}$ of one or more the diffusivity changes strongly with the
# temperature. Here, at the reference temperature and at the cold face:

# +
# The Arrhenius number is the group with the activation energy
arrhenius_number = next(
    g.value for g in groups if "E" in {str(q) for q in g.expression.free_symbols}
)
for label, temperature in (("T_ref", T.scale), ("the cold face", 300.0)):
    c = temperature / T.scale
    print(f"D at {label}: {np.exp(-arrhenius_number / c):.1e} D_0")
# -

# Values many orders of magnitude from one are not round-off, but ratios of
# scales, and such a value means one of two things: either the group
# compares two effects that compete in the equations, and one is negligible
# beside the other, or its quantities never compete in the equations, so its
# value cannot change the solution, however large or small. The Pi theorem
# uses the dimensions of the quantities, not the equations, so it finds
# groups of both kinds. Two of the groups here are of the second kind,
# $D_0 k_B / (L^3 \kappa)$ and $L^3 c_{ref}$: the steady hydrogen equation,
# the second of {eq}`wall-equations`, is linear in the concentration, so the
# scale of the concentration divides out of it, and so does $D_0$.
#
# (units-scaling)=
# ## Scaling the equations
#
# In SI units these equations are of very different sizes, as their typical
# sizes, printed below, show. Newton's tolerances are absolute and relative, and
# such residuals make the absolute one meaningless.
# `factorize_equations=True` divides each equation by its typical size, that
# of its largest term, which {py:func}`equation_scales
# <sumfem.symbolic.equation_scales>` gives {cite}`Habera2026` (Sections 3.5
# and 3.6). For scaling PDEs by hand, see {cite}`Langtangen2016` (Chapters 3
# and 4).

print("Typical size of each equation:", symbolic.equation_scales(system))

# To see how accurate the solution is, we add sources $f$ and $g$ to the
# equations, so that the solution is
#
# $$
# \begin{aligned}
#   T &= T_c + (T_h - T_c) \cos\left(\frac{\pi x}{2 L}\right), \\
#   c &= c_h \cos\left(\frac{\pi x}{2 L}\right),
# \end{aligned}
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

# Scaled, the solve may start far from the Dirichlet values, here from a
# uniform temperature and no hydrogen.

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

# {py:meth}`value <sumfem.discrete.Discretization.value>` gives a
# parameter's value in its own unit; the forms use its SI value, here in
# metres, m$^2$/s and J.

for parameter in (L_wall, D_0, E):
    own, si = float(disc.value(parameter)), disc.function(parameter).x.array[0]
    print(f"{parameter.name} = {own:.3g} in its unit, {si:.4g} in SI")

# Values can also be set with units of their own, which are converted:

disc.set({kappa: 4.0 * W / (syu.centimeter * K)})
print(f"kappa = {float(disc.value(kappa)):.1f} W/(m K)")

# ## Scaling and Newton's method
#
# Newton's method takes the same steps whether the equations are scaled or
# not {cite}`Deuflhard2011` (Sections 1.2.2 and 2.1): what scaling changes
# is when it stops. We sweep the temperature of the hot face, each solve
# starting from the last solution, as in a parameter study or a time step.
# Unscaled, a start must meet the Dirichlet conditions, so the first solve
# starts from linear profiles. The sweep returns the total number of Newton
# iterations, the time it took, and the discretization holding the last
# solution.


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

# We sweep the temperature of the hot face, unscaled and scaled, and compare
# the last solutions with the exact one, at the last temperature.

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
# absolute tolerance is in SI and means nothing, and Newton takes more
# iterations, to stop on the size of its update instead. Scaled,
# every equation is of size one, and the absolute tolerance of
# {py:data}`FACTORIZED_PETSC_OPTIONS
# <sumfem.solvers.solver.FACTORIZED_PETSC_OPTIONS>` means the same for all
# of them. Use `factorize_equations=True` when the fields or the equations
# of a model differ by orders of magnitude; it also suits direct solvers
# that test for small pivots, such as PETSc's own LU, and iterative solvers.

# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
