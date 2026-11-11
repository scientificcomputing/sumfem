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

# # A porous electrode benchmark
#
# This demo solves the porous electrode of the benchmark of
# {cite}`Wang2025`, in one, two and three dimensions, and validates it
# against a reference solution of the problem reduced to one dimension, see
# {ref}`electrode-validation`. It shows how to:
#
# - Build a battery model with sumfem's
#   {ref}`models of electrochemistry <electrode-electrochemistry>`, then
#   with its {ref}`generic components <electrode-generic>` alone, and in
#   {ref}`plain UFL <electrode-ufl>`
# - Give its parameters units, check them, and solve the model scaled
# - Reduce the model to
#   {ref}`one equation for the overpotential <electrode-one-equation>`
#
# ## The model
#
# In porous electrode theory {cite}`NewmanTiedemann1975`, a macrohomogeneous
# model, the pores of the electrode are not resolved. Averaged over a
# volume, the solid and the electrolyte become two continua that coexist
# at every point, each with its own potential: the dual-continuum model of
# {cite}`Wang2025`. In an electrode of width $W$, the solid has the potential
# $\phi_e$ and the conductivity $\sigma$, and the electrolyte the potential
# $\phi_l$ and the conductivity $\kappa$. The reaction at their interface,
# of specific area $a$, carries the current $a\, i(\eta)$ from the solid to
# the electrolyte, by Butler-Volmer kinetics
# {cite}`Butler1924,ErdeyGruzVolmer1930` with the exchange current density
# $i_0$,
#
# $$
# \begin{aligned}
#   -\nabla \cdot (\sigma \nabla \phi_e) + a\, i(\eta) &= 0, \\
#   -\nabla \cdot (\kappa \nabla \phi_l) - a\, i(\eta) &= 0, \\
#   i(\eta) &= i_0 \left(e^{F \eta / (2 R T)} - e^{-F \eta / (2 R T)}\right), \\
#   \eta &= \phi_e - \phi_l - E_{eq},
# \end{aligned}
# $$ (electrode)
#
# with the overpotential $\eta$ and the equilibrium potential $E_{eq}$. The
# applied current density $j$ enters the solid through the current
# collector at $x = 0$, and leaves the electrolyte through the membrane at
# $x = W$:
#
# $$
# \begin{aligned}
#   \sigma \nabla \phi_e \cdot n &= j \quad \text{on the current collector}, \\
#   \kappa \nabla \phi_l \cdot n &= -j \quad \text{on the membrane},
# \end{aligned}
# $$ (electrode-currents)
#
# and no current crosses the other boundaries.
#
# We import the relevant packages required for this modelling

# + tags=["hide-input"]
import os

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import sympy.physics.units as syu
import ufl
from basix.ufl import element
from scipy.integrate import solve_bvp

from sumfem import discrete, solvers, symbolic
from sumfem.electrochemistry import Electrolyte, ExchangeCurrentButlerVolmer
from sumfem.models import (
    BoundaryFlux,
    Conduction,
    Diffusion,
    Source,
    butler_volmer,
)
from sumfem.plotting import gather_grid
from sumfem.post_processing import interpolate_to_line
from sumfem.symbolic.si import to_si

# Images in the book, unless PYVISTA_JUPYTER_BACKEND chooses another backend
if "PYVISTA_JUPYTER_BACKEND" not in os.environ:
    pyvista.set_jupyter_backend("static")
# -

# ## The electrode
#
# The electrode is a rectangle of width $W$ and height $H$, of
# quadrilaterals, in the precision of PETSc's real type, which assembly and
# solvers use. The potentials are of degree two. The number of cells across
# the electrode is shared with the meshes of the other dimensions below:

cell_name = "quadrilateral"
degree = 2
width, height = 5e-3, 1e-1
cells_across = 50
dtype = dolfinx.default_real_type
rectangle = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [width, height]],
    [cells_across, 40],
    cell_type=dolfinx.mesh.to_type(cell_name),
    dtype=dtype,
)

# The current collector and the membrane are located by their coordinates.
# The tolerance of the locators is relative to the width and to the
# precision of the coordinates, as an absolute one, such as the default of
# {py:func}`np.isclose <numpy.isclose>`, does not scale with the size of the
# electrode, see {ref}`units-length-scale`. The symbolic domain has the width
# as its length scale.


# +
def mark(mesh, dim, width, tolerance):
    """The current collector at x = 0 and the membrane at x = width."""
    return discrete.MeshData.from_locators(
        mesh,
        {
            dim: {
                "CurrentCollector": lambda x: np.isclose(x[0], 0.0, atol=tolerance),
                "Membrane": lambda x: np.isclose(x[0], width, atol=tolerance),
            }
        },
        boundary=dim > 0,
    )


tolerance = 100 * np.finfo(dtype).eps * width
marked = mark(rectangle, 1, width, tolerance)
domain = symbolic.Domain.from_mesh("electrode", rectangle, length_scale=width)
# -

# The parameters have the values of the benchmark, in their units, so that
# sumfem can check the units, see {ref}`units-checking`, and scale the
# equations, see {ref}`units-scaling`. The potentials are fields in volts,
# of a typical size of a tenth of a volt:

# +
S, m, A, V = syu.siemens, syu.meter, syu.ampere, syu.volt


def parameters(domain):
    """The parameters of the benchmark on a domain."""
    values = {
        "sigma": (103.1891, S / m, "Conductivity of the solid"),
        "kappa": (5.9514, S / m, "Conductivity of the electrolyte"),
        "a": (1.64e4, 1 / m, "Specific area"),
        "i0": (2.7657, A / m**2, "Exchange current density"),
        "E_eq": (-0.1609, V, "Equilibrium potential"),
        "j": (-400.0, A / m**2, "Applied current density"),
    }
    return {
        name: symbolic.Parameter(name, domain, default=value, unit=unit, doc=doc)
        for name, (value, unit, doc) in values.items()
    }


def potentials(domain, degree):
    """The potentials of the solid and of the electrolyte."""
    return tuple(
        symbolic.Field(name, domain, ("Lagrange", degree), unit=V, scale=0.1)
        for name in ("phi_e", "phi_l")
    )


p = parameters(domain)
phi_e_field, phi_l_field = potentials(domain, degree)
maps = discrete.DomainMaps({domain: marked})
# -

# The model is written three times below, from the top down: with the
# models of electrochemistry, with sumfem's generic components alone, and in
# plain UFL. Each gives the same solution.
#
# (electrode-electrochemistry)=
# ## With the models of electrochemistry
#
# {py:mod}`sumfem.electrochemistry` holds the models of electrochemistry. An
# {py:class}`~sumfem.electrochemistry.Electrolyte` holds Faraday's constant
# $F$, the gas constant $R$ and the temperature $T$, as parameters with
# units that the models built on it share; here it is of uniform
# composition, without ions. The reaction of {eq}`electrode` is
# {py:class}`~sumfem.electrochemistry.ExchangeCurrentButlerVolmer`, which
# adds the current $a\, i(\eta)$ to the equations of both potentials. Ohmic
# conduction in the solid and the electrolyte,
# {py:class}`Conduction <sumfem.models.Conduction>`, and the applied current
# on the two boundaries of {eq}`electrode-currents`,
# {py:class}`BoundaryFlux <sumfem.models.BoundaryFlux>`, are generic
# components of {py:mod}`sumfem.models`. The model is their sum:


# +
def electrode(p, electrolyte, phi_e, phi_l):
    reaction = ExchangeCurrentButlerVolmer(
        "reaction",
        phi_e,
        phi_l,
        electrolyte,
        exchange_current=p["i0"],
        equilibrium_potential=p["E_eq"],
        anodic_transfer=0.5,
        cathodic_transfer=0.5,
        specific_area=p["a"],
    )
    model = (
        Conduction("solid", phi_e, p["sigma"])
        + Conduction("electrolyte", phi_l, p["kappa"])
        + reaction
        + BoundaryFlux("collector", phi_e, p["j"], ("CurrentCollector",))
        + BoundaryFlux("membrane", phi_l, -p["j"], ("Membrane",))
    )
    return model, reaction


electrolyte = Electrolyte("electrolyte", domain)
model, reaction = electrode(p, electrolyte, phi_e_field, phi_l_field)
# -

# The units are checked, the equations scaled, and the model solved:


# +
def solve(model, maps):
    system = symbolic.build_system(model, check_units=True, factorize_equations=True)
    disc = discrete.discretize([system], maps)
    solvers.Solver(disc[system]).solve()
    return disc


disc = solve(model, maps)
# -

# The reaction carries all of the applied current, which enters through the
# current collector of height $H$:

# +
reacted = disc.assemble_scalar(reaction.source() * domain.dx())
applied = p["j"].default_si * height
print(f"Reacted over applied current: {reacted / applied}")
assert np.isclose(reacted, applied, rtol=1e-8)
# -

# The overpotential varies across the electrode alone. It is plotted with
# [PyVista](https://pyvista.org), gathered from all processes by
# {py:func}`gather_grid <sumfem.plotting.gather_grid>`:


# +
def overpotential(disc, phi_e, reaction):
    space = disc.function_space(phi_e)
    eta = dolfinx.fem.Function(space, name="eta")
    expression = disc[reaction.overpotential()]
    eta.interpolate(
        dolfinx.fem.Expression(expression, space.element.interpolation_points)
    )
    return eta


eta_2d = overpotential(disc, phi_e_field, reaction)
grid = gather_grid(eta_2d)
if MPI.COMM_WORLD.rank == 0:
    plotter = pyvista.Plotter(window_size=(640, 480))
    plotter.add_mesh(grid, scalars="eta", scalar_bar_args={"title": "eta (V)"})
    plotter.view_xy()
    plotter.show()
# -

# Only currents are prescribed, and the potentials enter the reaction only
# through their difference, so both potentials are determined up to the
# same constant: their difference is what the model determines, and what the
# formulations below are compared by:


# +
def potential_difference(disc):
    return disc.function(phi_e_field).x.array - disc.function(phi_l_field).x.array


def relative_difference(disc, reference):
    difference = np.abs(potential_difference(disc) - reference).max()
    return difference / np.abs(reference).max()


reference_difference = potential_difference(disc)
# -

# (electrode-generic)=
# ## With generic components
#
# Without the models of electrochemistry, $F$, $R$ and $T$ are parameters
# of their own: the exact constants
# {py:data}`FARADAY <sumfem.symbolic.FARADAY>` and
# {py:data}`GAS_CONSTANT <sumfem.symbolic.GAS_CONSTANT>` in SI units, and the
# room temperature, the default of the electrolyte above:

# +
C, J, K, mol = syu.coulomb, syu.joule, syu.kelvin, syu.mol
F = symbolic.Parameter(
    "F", domain, default=to_si(symbolic.FARADAY, C / mol), unit=C / mol
)
R = symbolic.Parameter(
    "R", domain, default=to_si(symbolic.GAS_CONSTANT, J / (mol * K)), unit=J / (mol * K)
)
T = symbolic.Parameter("T", domain, default=298.15, unit=K)
# -

# The reaction is then a source in each equation of {eq}`electrode`, with the
# current density of {py:func}`butler_volmer <sumfem.models.butler_volmer>`,
# which takes the overpotential over the thermal voltage $R T / F$, and its
# transfer coefficients one half by default:

# +
eta = phi_e_field - phi_l_field - p["E_eq"]
current = p["a"] * butler_volmer(p["i0"], eta / (R * T / F))
generic = (
    Conduction("solid", phi_e_field, p["sigma"])
    + Conduction("electrolyte", phi_l_field, p["kappa"])
    + Source("solid_reaction", phi_e_field, -current)
    + Source("electrolyte_reaction", phi_l_field, current)
    + BoundaryFlux("collector", phi_e_field, p["j"], ("CurrentCollector",))
    + BoundaryFlux("membrane", phi_l_field, -p["j"], ("Membrane",))
)
generic_difference = relative_difference(solve(generic, maps), reference_difference)
print(f"Relative difference of phi_e - phi_l: {generic_difference:.1e}")
assert generic_difference < 1e-10
# -

# (electrode-ufl)=
# ## In plain UFL
#
# The weak form of {eq}`electrode` and {eq}`electrode-currents` is written in
# plain UFL, as in a UFL file of DOLFINx, on a mesh of the same cells, with
# the current collector and the membrane as the boundary markers 1 and 2:

# +
mesh = ufl.Mesh(element("Lagrange", cell_name, 1, shape=(2,)))
Q_e = ufl.FunctionSpace(mesh, element("Lagrange", cell_name, degree))
Q_l = ufl.FunctionSpace(mesh, element("Lagrange", cell_name, degree))
phi_e, phi_l = ufl.Coefficient(Q_e), ufl.Coefficient(Q_l)
v_e, v_l = ufl.TestFunctions(ufl.MixedFunctionSpace(Q_e, Q_l))
sigma, kappa, a, i0, E_eq, j, F_c, R_c, T_c = (ufl.Constant(mesh) for _ in range(9))

eta_ufl = phi_e - phi_l - E_eq
x = F_c * eta_ufl / (R_c * T_c)
reaction_ufl = a * i0 * (ufl.exp(x / 2) - ufl.exp(-x / 2))
ds = ufl.Measure("ds", domain=mesh)
residual = (
    ufl.inner(sigma * ufl.grad(phi_e), ufl.grad(v_e)) * ufl.dx
    + ufl.inner(kappa * ufl.grad(phi_l), ufl.grad(v_l)) * ufl.dx
    + reaction_ufl * v_e * ufl.dx
    - reaction_ufl * v_l * ufl.dx
    - j * v_e * ds(1)
    + j * v_l * ds(2)
)
# -

# {py:class}`UFLResidual <sumfem.symbolic.UFLResidual>` turns it into a
# residual of sumfem, as introduced in
# {doc}`../../docs/getting_started/01_poisson_in_ufl`: the coefficients become
# the potentials, the constants the parameters above, the UFL mesh the
# domain, and the boundary markers 1 and 2 the current collector and the
# membrane:

# +
plain = symbolic.UFLResidual(
    "electrode",
    residual,
    [phi_e, phi_l],
    domains={mesh: domain},
    markers={"ds": {1: "CurrentCollector", 2: "Membrane"}},
    terminals={
        phi_e: phi_e_field,
        phi_l: phi_l_field,
        sigma: p["sigma"],
        kappa: p["kappa"],
        a: p["a"],
        i0: p["i0"],
        E_eq: p["E_eq"],
        j: p["j"],
        F_c: F,
        R_c: R,
        T_c: T,
    },
)
plain_difference = relative_difference(solve(plain, maps), reference_difference)
print(f"Relative difference of phi_e - phi_l: {plain_difference:.1e}")
assert plain_difference < 1e-10
# -

# ## Across the electrode
#
# In one dimension, the same components solve the problem across an
# electrode of a given width, on an interval of a given number of cells,
# marked with a tolerance as above. The overpotential and the current that
# reacts are returned, so that the solution can be checked:


# +
def across(width, cells, degree, tolerance):
    interval = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, cells, [0.0, width], dtype=dtype
    )
    line = symbolic.Domain.from_mesh("line", interval, length_scale=width)
    phi_e, phi_l = potentials(line, degree)
    model, reaction = electrode(
        parameters(line), Electrolyte("electrolyte", line), phi_e, phi_l
    )
    disc = solve(
        model, discrete.DomainMaps({line: mark(interval, 0, width, tolerance)})
    )
    reacted = disc.assemble_scalar(reaction.source() * line.dx())
    return overpotential(disc, phi_e, reaction), reacted


# -

# With the width of the rectangle and as many cells across it, the reaction
# carries the applied current density:

# +
eta_1d, reacted_1d = across(width, cells_across, degree, tolerance)
print(f"Reacted over applied current: {reacted_1d / p['j'].default_si}")
assert np.isclose(reacted_1d, p["j"].default_si, rtol=1e-8)
# -

# ## In three dimensions
#
# A box of hexahedra, finely resolved across the electrode and coarsely
# along it, where the solution is constant. The reaction carries the applied
# current, through the current collector of area $H^2$:

# +
box = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [np.zeros(3), np.array([width, height, height])],
    [cells_across, 4, 4],
    cell_type=dolfinx.mesh.CellType.hexahedron,
    dtype=dtype,
)
solid = symbolic.Domain.from_mesh("box", box, length_scale=width)
p_3d = parameters(solid)
phi_e_3d, phi_l_3d = potentials(solid, degree)
model_3d, reaction_3d = electrode(
    p_3d, Electrolyte("electrolyte", solid), phi_e_3d, phi_l_3d
)
disc_3d = solve(model_3d, discrete.DomainMaps({solid: mark(box, 2, width, tolerance)}))
eta_3d = overpotential(disc_3d, phi_e_3d, reaction_3d)

reacted_3d = disc_3d.assemble_scalar(reaction_3d.source() * solid.dx())
applied_3d = p_3d["j"].default_si * height**2
print(f"Reacted over applied current: {reacted_3d / applied_3d}")
assert np.isclose(reacted_3d, applied_3d, rtol=1e-8)
# -

# (electrode-one-equation)=
# ## One equation for the overpotential
#
# With constant conductivities, the two potentials can be eliminated:
# subtracting the equations of {eq}`electrode` divided by their
# conductivities gives one equation for the overpotential,
#
# $$
# \begin{aligned}
#   -\Delta \eta + \left(\frac{1}{\sigma} + \frac{1}{\kappa}\right) a\, i(\eta)
#   &= 0, \\
#   \nabla \eta \cdot n &= \frac{j}{\sigma}
#   \quad \text{on the current collector}, \\
#   \nabla \eta \cdot n &= \frac{j}{\kappa} \quad \text{on the membrane},
# \end{aligned}
# $$ (overpotential-equation)
#
# a diffusion with a source, written with
# {py:class}`Diffusion <sumfem.models.Diffusion>`,
# {py:class}`Source <sumfem.models.Source>` and
# {py:class}`BoundaryFlux <sumfem.models.BoundaryFlux>`, on the rectangle.
# Integrating it shows that the reaction carries the applied current here
# too:

# +
eta_field = symbolic.Field("eta", domain, ("Lagrange", degree), unit=V, scale=0.1)
thermal = electrolyte.R * electrolyte.T / electrolyte.F
current_density = butler_volmer(p["i0"], eta_field / thermal, 0.5, 0.5)
resistivity = 1 / p["sigma"] + 1 / p["kappa"]
reduced = (
    Diffusion("diffusion", eta_field, 1.0)
    + Source("reaction", eta_field, -resistivity * p["a"] * current_density)
    + BoundaryFlux("collector", eta_field, p["j"] / p["sigma"], ("CurrentCollector",))
    + BoundaryFlux("membrane", eta_field, p["j"] / p["kappa"], ("Membrane",))
)
disc_eta = solve(reduced, maps)
eta_reduced = disc_eta.function(eta_field)

reacted_eta = disc_eta.assemble_scalar(p["a"] * current_density * domain.dx())
print(f"Reacted over applied current: {reacted_eta / applied}")
assert np.isclose(reacted_eta, applied, rtol=1e-8)
# -

# (electrode-validation)=
# ## Validation
#
# The solution depends on $x$ alone. Across the electrode, the current
# $i_l$ of the electrolyte grows from zero at the current collector to $j$
# at the membrane, and with constant conductivities the overpotential and
# $i_l$ solve
#
# $$
# \begin{aligned}
#   \eta' &= \frac{i_l - j}{\sigma} + \frac{i_l}{\kappa}, \\
#   i_l' &= a\, i(\eta), \quad i_l(0) = 0, \quad i_l(W) = j,
# \end{aligned}
# $$ (electrode-reduced)
#
# a boundary value problem for {py:func}`scipy.integrate.solve_bvp`, with
# the values of the parameters and of the electrolyte in SI. It starts from
# a mesh of the given points, refines it until the residual is below the
# tolerance, and stops with an error past the given number of nodes:


# +
def reference_solution(p, electrolyte, points, tolerance, max_nodes):
    sigma, kappa, a, i0, j = (
        p[n].default_si for n in ("sigma", "kappa", "a", "i0", "j")
    )
    F, R, T = (c.default_si for c in (electrolyte.F, electrolyte.R, electrolyte.T))

    def system(x, y):
        eta, current = y
        reaction = i0 * 2 * np.sinh(F * eta / (2 * R * T))
        return np.vstack([(current - j) / sigma + current / kappa, a * reaction])

    def boundary(y0, yW):
        return np.array([y0[1], yW[1] - j])

    solution = solve_bvp(
        system,
        boundary,
        points,
        np.zeros((2, points.size)),
        tol=tolerance,
        bc_tol=tolerance,
        max_nodes=max_nodes,
    )
    assert solution.success, solution.message
    return lambda x: solution.sol(x)[0]


# -

# The tolerance is far below the errors of the finite elements, so that the
# reference is exact for the comparison. The solution varies most in a
# reaction layer at the membrane, which the solver resolves by refining; a
# start from points as fine as the finite elements, and a limit of nodes far
# above what the refinement needs, leave it room to:

reference = reference_solution(
    p, electrolyte, np.linspace(0, width, 100), tolerance=1e-9, max_nodes=100_000
)

# Each solution is compared with it on a line across the middle of its
# electrode, from the current collector to the membrane, by the relative
# root mean square of the difference. The solution is interpolated on the
# line with its own degree, at a number of points that resolves it:


# +
def line_error(eta, reference, start, end, degree, points):
    line = interpolate_to_line(eta, start, end, points, ("Lagrange", degree))
    x = line.function_space.tabulate_dof_coordinates()[:, 0]
    expected = dolfinx.fem.Function(line.function_space)
    expected.x.array[:] = reference(x)
    form = dolfinx.fem.form((line - expected) ** 2 * ufl.dx)
    comm = line.function_space.mesh.comm
    squared = comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
    length = np.linalg.norm(np.subtract(end, start))
    scale = np.abs(reference(np.linspace(start[0], end[0], 200))).max()
    return np.sqrt(squared / length) / scale


middle = height / 2
lines = {
    "2D": (eta_2d, [0.0, middle], [width, middle]),
    "1D": (eta_1d, [0.0], [width]),
    "3D": (eta_3d, [0.0, middle, middle], [width, middle, middle]),
    "One equation": (eta_reduced, [0.0, middle], [width, middle]),
}
errors = {
    name: line_error(eta, reference, start, end, degree, points=25)
    for name, (eta, start, end) in lines.items()
}
for name, value in errors.items():
    print(f"{name}: relative error {value:.1e}")
assert all(value < 1e-5 for value in errors.values())
# -

# Across the electrode, the solution in one dimension and the reference:

x_dofs = eta_1d.function_space.tabulate_dof_coordinates()[:, 0]
order = np.argsort(x_dofs)
x_plot = np.linspace(0, width, 200)
plt.plot(1e3 * x_plot, reference(x_plot), "k-", label="Reference")
plt.plot(1e3 * x_dofs[order], eta_1d.x.array[order], "o", label="Finite elements")
plt.xlabel("x (mm)")
plt.ylabel("Overpotential (V)")
plt.legend()
plt.show()

# (electrode-convergence)=
# ### Convergence
#
# For a smooth solution, the error of Lagrange elements of degree $p$ in the
# $L^2$ norm falls as $h^{p+1}$ with the cell size $h$, see the duality
# estimates in Chapter 0 of {cite}`BrennerScott2008`. Halving the cell
# size divides the error by $2^{p+1}$, so the observed rate
#
# $$
# r = \log_2 \frac{e_h}{e_{h/2}}
# $$ (convergence-rate)
#
# approaches $p + 1$. The fixed points of the line above would limit the
# error under refinement, so the solution across the electrode is compared
# with the reference in the relative $L^2$ norm over the whole interval. The
# reference is interpolated in a degree above that of the solution, so that
# its interpolation error falls faster than the error measured:
#


# +
def l2_error(eta, reference, degree):
    mesh = eta.function_space.mesh
    space = dolfinx.fem.functionspace(mesh, ("Lagrange", degree + 2))
    expected = dolfinx.fem.Function(space)
    expected.interpolate(lambda x: reference(x[0]))

    def integral(expression):
        form = dolfinx.fem.form(expression**2 * ufl.dx)
        return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    return np.sqrt(integral(eta - expected) / integral(expected))


# -

# The interval is refined twice from a coarse mesh, for the degrees one to
# three. The coarsest mesh resolves the reaction layer at the membrane, and
# the finest stays far above the tolerance of the reference. The rate
# {eq}`convergence-rate` exceeds $p + 1/2$ on every refinement:

# +
refinements = (8, 16, 32)
scientific = {"float_kind": lambda value: f"{value:.1e}"}
for p_degree in (1, 2, 3):
    errors_h = np.array(
        [
            l2_error(across(width, n, p_degree, tolerance)[0], reference, p_degree)
            for n in refinements
        ]
    )
    rates = np.log2(errors_h[:-1] / errors_h[1:])
    print(
        f"Degree {p_degree}: errors {np.array2string(errors_h, formatter=scientific)}, "
        f"rates {np.array2string(rates, precision=2)}"
    )
    assert (rates > p_degree + 0.5).all()
# -

# ### A thicker electrode
#
# In a thicker electrode, the conductivity of the electrolyte, lower than
# that of the solid, confines the reaction to a layer at the membrane. The
# overpotential is nearly uniform across most of the electrode and varies
# steeply in the layer. The parameters, not the electrode, set the thickness
# of the layer, so the thicker the electrode, the smaller the part of it the
# layer takes. The same components solve the problem, from the same initial
# guess, on a thicker electrode. The same number of cells across would leave
# few cells in the layer, so the mesh has more, and the reference starts
# from points across the thicker electrode:

# +
thick = 10 * width
thick_tolerance = 100 * np.finfo(dtype).eps * thick
eta_thick, _ = across(thick, 4 * cells_across, degree, thick_tolerance)
reference_thick = reference_solution(
    p, electrolyte, np.linspace(0, thick, 100), tolerance=1e-9, max_nodes=100_000
)
error_thick = l2_error(eta_thick, reference_thick, degree)
print(f"Thick electrode: relative error {error_thick:.1e}")
assert error_thick < 1e-4
# -

# The current balance is not checked here: the steep layer needs a higher
# quadrature degree for the integral of the reaction than the default
# estimate gives. Across the thick electrode, the overpotential and its
# reaction layer at the membrane:

x_thick = eta_thick.function_space.tabulate_dof_coordinates()[:, 0]
order_thick = np.argsort(x_thick)
x_plot_thick = np.linspace(0, thick, 400)
plt.plot(1e3 * x_plot_thick, reference_thick(x_plot_thick), "k-", label="Reference")
plt.plot(
    1e3 * x_thick[order_thick],
    eta_thick.x.array[order_thick],
    ".",
    label="Finite elements",
)
plt.xlabel("x (mm)")
plt.ylabel("Overpotential (V)")
plt.legend()
plt.show()

# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
