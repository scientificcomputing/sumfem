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

# # Constraints with Lagrange multipliers
#
# This tutorial illustrates how to:
#
# - Impose a Dirichlet condition with a Lagrange multiplier,
#   {py:class}`sumfem.models.DirichletMultiplier`, whose multiplier is the
#   flux leaving the domain
# - Impose a nonlinear constraint on a boundary with a Lagrange multiplier,
#   {py:class}`sumfem.models.BoundaryConstraint`, here the contact of an
#   elastic body with a curved rigid obstacle
# - Impose a nonlinear condition across an interface between two materials,
#   {py:class}`sumfem.models.InterfaceConstraint`, here the continuity of the
#   partial pressure of hydrogen
#
# A Lagrange multiplier imposes a constraint exactly, at the price of an
# unknown of its own, and has a meaning of its own: a contact pressure, or a
# flux. The examples show that the multiplier enters the equations
# differently on a boundary and on an interface.

# +
import os

from mpi4py import MPI

import dolfinx.io.gmsh
import dolfinx.mesh
import gmsh
import numpy as np
import pyvista
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import (
    BoundaryConstraint,
    Diffusion,
    DirichletMultiplier,
    InterfaceConstraint,
    Poisson,
)
from sumfem.plotting import gather_grid
from sumfem.post_processing import error

# -

# ## A Dirichlet condition
#
# The method goes back to Babuška {cite}`Babuska1973`: a Dirichlet condition
# $u = g$ on a part $\Gamma$ of the boundary is imposed at the stationary
# point of $(\lambda, u - g)$ on $\Gamma$, with a multiplier $\lambda$ that
# lives on $\Gamma$ alone. For Poisson's equation $-\Delta u = f$, this adds
#
# $$
#   (\lambda, v) + (u - g, \mu)
# $$
#
# on $\Gamma$ to the weak form, with $v$ and $\mu$ the test functions of $u$
# and $\lambda$. Integrating $(\nabla u, \nabla v)$ by parts shows that the
# multiplier is the flux leaving the domain, $\lambda = -\nabla u \cdot n$
# {cite}`Babuska1973,GunzburgerHou1992`, computed with the solution and no
# post-processing.
#
# As in the
# [mortar demo of dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc/blob/18a40cbcee4761d5136a57d917e3dae2399ccf05/python/demos/demo_mortar.py),
# we take the unit square, the left edge
# as $\Gamma$, and the exact solution $u = (x - 1)^2 \cos(2 \pi y)$, whose
# normal derivative vanishes on the other edges, so that their natural
# condition holds. On $\Gamma$, $n = (-1, 0)$ and $\lambda = -2 \cos(2 \pi
# y)$. The multiplier is a field on a
# {py:class}`Subdomain <sumfem.symbolic.Subdomain>` of the facets of
# $\Gamma$:

# +
plane = symbolic.Domain("plane", "triangle")
left = symbolic.Subdomain("left", plane, 1, "Left")
u = symbolic.Field("u", plane, ("Lagrange", 1))
outflow = symbolic.Field("outflow", left, ("Lagrange", 1))


def exact_u(x):
    return (x[0] - 1) ** 2 * ufl.cos(2 * ufl.pi * x[1])


def exact_outflow(x):
    return -2 * ufl.cos(2 * ufl.pi * x[1])


x = ufl.SpatialCoordinate(plane)
source = -ufl.div(ufl.grad(exact_u(x)))
model = Poisson("poisson", u, source) + DirichletMultiplier(
    "dirichlet", u, outflow, exact_u(x), markers=("Left",)
)
# -

# The problem is linear, so Newton's method converges in one iteration. We
# solve it on three meshes, each square cut into four triangles by its
# diagonals, and compare $u$ and $\lambda$ with the exact solution:

previous = None
for n in (8, 16, 32):
    unit_square = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, diagonal=dolfinx.mesh.DiagonalType.crossed
    )
    marked = discrete.MeshData.from_locators(
        unit_square, {1: {"Left": lambda x: np.isclose(x[0], 0.0)}}
    )
    disc = discrete.discretize([model], discrete.DomainMaps({plane: marked}))
    solver = solvers.Solver(disc[model], [])
    solver.solve()
    errors = np.array([error(disc[u], exact_u), error(disc[outflow], exact_outflow)])
    rates = (
        ""
        if previous is None
        else ", rates %.2f and %.2f" % tuple(np.log2(previous / errors))
    )
    print(f"n = {n}: L2 errors {errors[0]:.1e} and {errors[1]:.1e}{rates}")
    previous = errors
assert errors[0] < 1e-3 and errors[1] < 0.1

# Both converge at second order. The mesh matters for the multiplier: with
# one diagonal per square, the default of
# {py:func}`dolfinx.mesh.create_unit_square`, its error falls at second order
# in the interior of $\Gamma$ but only at first order at the two ends of
# $\Gamma$, where it meets the edges with the natural condition. On two
# elements of length $h$, that is $h^{3/2}$ in the $L^2$ norm, and the rate
# falls towards $3/2$ on finer meshes. On the crossed mesh, which is
# symmetric about every grid line, the ends converge at second order too.
# In three dimensions, a hexahedral mesh of the unit cube does the same,
# while on the tetrahedra of {py:func}`dolfinx.mesh.create_unit_cube` the
# multiplier loses the order on the edges of $\Gamma$.
#
# With elements of degree $p$ for both, $u$ converges at order $p + 1$ and
# the multiplier, a derivative of $u$, at order $p$ from $p = 2$ on, as the
# gradient does; the second order of P1 on the crossed mesh is a bonus of
# its symmetry. The pair is stable: the multiplier space is the trace of the
# space of $u$ on $\Gamma$, for which the discrete inf-sup condition holds
# uniformly {cite}`Chouly2023` (Proposition 7.5), and its constant stays
# between 0.48 and 0.82 under refinement on all of these meshes, for
# $p = 1, 2, 3$.
#
# ## Contact with a curved rigid obstacle
#
# An elastic body in frictionless contact with a rigid obstacle may not
# penetrate it, and where it touches the obstacle, the obstacle pushes back
# with a contact pressure. With the pressure as a Lagrange multiplier, the
# contact conditions are imposed on the boundary alone
# {cite}`Wohlmuth2011`, {cite}`Chouly2023` (Section 7.1). For an obstacle
# that is a circle of radius $R$ about the origin, the gap of a boundary
# point $X$, displaced by $u$, is its distance to the circle
# {cite}`Wriggers2006` (Chapter 4),
#
# $$
#   g(u) = |X + u| - R,
# $$
#
# which is nonlinear in the displacement: as in contact at large strains,
# the gap follows the deformed boundary {cite}`Chouly2023` (Section 11.1.2).
#
# We fit an elastic disc of radius $r_0 = 1$ into a ring of radius
# $R = 0.9$, a shrink fit, so its whole rim stays in contact. The contact
# zone is then known in advance, a bilateral contact, and nonpenetration,
# $g(u) \le 0$, holds with equality, $g(u) = 0$ {cite}`Chouly2023`
# (Section 1.3.1). When the contact zone is not known, unilateral contact,
# the inequality needs a nonsmooth reformulation, such as Nitsche's method
# or an augmented Lagrangian {cite}`Chouly2023` (Chapters 6 and 7), which a
# {py:class}`~sumfem.models.BoundaryConstraint` does not provide.
#
# The constraint is imposed at the stationary point of $(p, g(u))$ on the
# rim, so the multiplier $p$ enters the equation of $u$ through the
# derivative of the gap, $g'(u; v) = n \cdot v$ with the current normal
# $n = (X + u)/|X + u|$. The boundary term of elasticity is
# $(\sigma(u) n, v)$, so the multiplier is minus the normal stress,
# $p = -\sigma_n \ge 0$, the contact pressure. Note that
# {cite}`Chouly2023` (Sections 1.3.2 and 7.1) takes the multiplier as
# $\sigma_n \le 0$ instead.
#
# The body is linearly elastic, with the Lamé parameters $\mu$ and $\lambda_L$:


class Elasticity(symbolic.Residual):
    """``(sigma(u), grad(v))`` with ``sigma = 2 mu eps + lambda_L tr(eps) I``."""

    def __init__(self, name, field, mu, lmbda):
        super().__init__(name)
        self.field, self.mu, self.lmbda = field, mu, lmbda

    def residual(self):
        u = self.field
        eps = ufl.sym(ufl.grad(u))
        sigma = 2 * self.mu * eps + self.lmbda * ufl.tr(eps) * ufl.Identity(2)
        return ufl.inner(sigma, ufl.grad(u.v)) * u.domain.dx()


# Without friction or other loads, the disc contracts uniformly, $u = \alpha
# X$ with $\alpha = R / r_0 - 1$, and the contact pressure is uniform,
# $p = -2(\mu + \lambda_L)\alpha$. The rim is a polygon on a mesh, so the
# discrete solution converges to it as the mesh is refined. The disc is
# meshed with Gmsh, its rim marked `"Rim"`, and the multiplier lives on the
# rim, a {py:class}`Subdomain <sumfem.symbolic.Subdomain>` of its facets:


def disc_mesh(size: float) -> discrete.MeshData:
    """The unit disc, with its cells marked "Disc" and its rim "Rim"."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.add("disc")
        disc = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        rim = [tag for _, tag in gmsh.model.getBoundary([(2, disc)], oriented=False)]
        gmsh.model.addPhysicalGroup(2, [disc], tag=1, name="Disc")
        gmsh.model.addPhysicalGroup(1, rim, tag=2, name="Rim")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.model.mesh.generate(2)
    data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return discrete.MeshData.from_gmsh(data)


# +
mu, lmbda, R = 1.0, 2.0, 0.9
alpha = R - 1.0

# Gmsh meshes have equispaced coordinate elements
domain = symbolic.Domain("disc", "triangle", lagrange_variant="equispaced")
rim = symbolic.Subdomain("rim", domain, 1, "Rim")
u = symbolic.Field("u", domain, ("Lagrange", 1, (2,)))
p = symbolic.Field("p", rim, ("Lagrange", 1))
X = ufl.SpatialCoordinate(domain)
gap = ufl.sqrt(ufl.inner(X + u, X + u)) - R
model = Elasticity("solid", u, mu, lmbda) + BoundaryConstraint(
    "contact", u, p, gap, markers=("Rim",)
)
# -

# The constraint is nonlinear, so Newton's method needs a few iterations on
# each mesh, with the Jacobian derived from the residual:

for size in (0.2, 0.1, 0.05):
    disc = discrete.discretize([model], discrete.DomainMaps({domain: disc_mesh(size)}))
    solver = solvers.Solver(disc[model], [])
    solver.solve()
    displacement_error = error(
        disc[u], lambda x: ufl.as_vector([alpha * x[0], alpha * x[1]])
    )
    pressure = disc.function(p).x.array
    print(
        f"h = {size}: {solver.iterations} Newton iterations, "
        f"L2 error {displacement_error:.1e}, "
        f"pressure {pressure.mean():.4f}, exact {-2 * (mu + lmbda) * alpha:.4f}"
    )
assert displacement_error < 1e-3
assert np.allclose(pressure, -2 * (mu + lmbda) * alpha, rtol=1e-2)

# The error falls by a factor of four with the mesh size, at second order,
# and the multiplier is the contact pressure.
#
# The solution on the finest mesh is plotted with
# [PyVista](https://pyvista.org). Each process holds the cells it owns, and
# {py:func}`gather_grid <sumfem.plotting.gather_grid>` gathers a grid of a
# function over them on one process, so the plot is the same on any number
# of processes. The disc is drawn where it started, as a grey wireframe,
# and displaced by its solution, coloured by the size of the displacement.
# Its displaced rim lies on the obstacle, drawn in black: the whole rim is in
# contact, as in a shrink fit.

# +
# Images in the book, unless PYVISTA_JUPYTER_BACKEND chooses another
# backend, e.g. "html" for interactive plots in the book, or "trame" in a
# notebook
if "PYVISTA_JUPYTER_BACKEND" not in os.environ:
    pyvista.set_jupyter_backend("static")

displacement = gather_grid(disc[u], name="u")
if MPI.COMM_WORLD.rank == 0:
    plotter = pyvista.Plotter(window_size=(640, 560))
    # The disc before, as a wireframe, and after, displaced by u
    plotter.add_mesh(displacement, style="wireframe", color="lightgrey")
    plotter.add_mesh(
        displacement.warp_by_vector("u"),
        scalars="u",
        show_edges=True,
        edge_color="dimgrey",
        scalar_bar_args={
            "title": "displacement",
            "vertical": True,
            "height": 0.6,
            "position_x": 0.86,
            "position_y": 0.2,
        },
    )
    obstacle = pyvista.Polygon(radius=R, n_sides=200, fill=False)
    plotter.add_mesh(obstacle, color="black", line_width=2)
    plotter.view_xy()
    plotter.show()
# -

# ## An interface condition in hydrogen transport
#
# Across an interface between two materials, hydrogen's concentration jumps,
# while its partial pressure $P$ is continuous {cite}`DelaporteMathurin2021`.
# In a metal, the concentration follows Sieverts' law, $c = K_S \sqrt{P}$,
# and in e.g. a liquid metal Henry's law, $c = K_H P$, so at an interface
# between the two {cite}`FESTIMTheory`,
#
# $$
#   \left(\frac{c_0}{K_S}\right)^2 = \frac{c_1}{K_H},
# $$
#
# a nonlinear condition between the concentrations $c_0$ of the metal and
# $c_1$ of the liquid.
#
# The multiplier of a constraint $g(c_0, c_1) = 0$, as in the sections
# above, enters through the derivative of $(\lambda, g)$. For this
# condition that gives the metal the flux $2 \lambda c_0 / K_S^2$ and the
# liquid the flux $\lambda / K_H$: different fluxes on the two sides, which
# create or destroy hydrogen at the interface. Instead, the multiplier here
# is the flux itself, which leaves the metal and enters the liquid
# unchanged, so the hydrogen is conserved:
#
# $$
#   (\lambda, v_0) - (\lambda, v_1)
#   + \left(\left(\frac{c_0}{K_S}\right)^2 - \frac{c_1}{K_H}, \mu\right)
# $$
#
# on the interface, with the test functions $v_0$, $v_1$ and $\mu$. The
# last term still imposes the condition, but the Jacobian is not symmetric.
# The two multipliers only coincide for a condition whose derivative is the
# same on both sides, such as continuity, $c_0 = c_1$.
#
# Partitioned methods for interface problems use such a flux multiplier
# {cite}`Peterson2019,Sockwell2020`, and so does a Lagrange multiplier
# method for fluid-structure interaction, whose multiplier is the traction
# on the interface {cite}`deCastro2025`. Contact between deformable bodies
# at large strains takes the same route: the multiplier acts on the two
# bodies with opposite signs, instead of through the derivative of the
# nonlinear gap {cite}`Poulios2015`, {cite}`Chouly2023` (Section 11.2.2).
#
# The unit square holds the metal left of $x = 1/2$ and the liquid right of
# it, each a subdomain of cells, and their interface is a subdomain of
# facets, which holds the multiplier:

# +
eps = 1e-10
square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
halves = discrete.MeshData.from_locators(
    square,
    {
        2: {
            "Metal": lambda x: x[0] <= 0.5 + eps,
            "Liquid": lambda x: x[0] >= 0.5 - eps,
        },
        1: {
            "Interface": lambda x: np.isclose(x[0], 0.5),
            "Inlet": lambda x: np.isclose(x[0], 0.0),
            "Outlet": lambda x: np.isclose(x[0], 1.0),
        },
    },
)
cell = symbolic.Domain("cell", "triangle")
metal = symbolic.Subdomain("metal", cell, 2, "Metal")
liquid = symbolic.Subdomain("liquid", cell, 2, "Liquid")
interface = symbolic.Subdomain("interface", cell, 1, "Interface")
# -

# Hydrogen diffuses in each material, enters the metal at $c_0 = a$ and
# leaves the liquid at $c_1 = b$:

# +
D_0, D_1, K_S, K_H, a, b = 2.0, 0.5, 1.5, 0.8, 3.0, 1.0
c_0 = symbolic.Field("c_0", metal, ("Lagrange", 1))
c_1 = symbolic.Field("c_1", liquid, ("Lagrange", 1))
flux = symbolic.Field("flux", interface, ("Lagrange", 1))
pressure_continuity = InterfaceConstraint(
    "pressure",
    c_0,
    c_1,
    (c_0 / K_S) ** 2,
    c_1 / K_H,
    flux,
    cell.interface(metal, liquid, "Interface"),
)
model = Diffusion("metal", c_0, D_0) + Diffusion("liquid", c_1, D_1)
model = model + pressure_continuity
bcs = [symbolic.DirichletBC(c_0, "Inlet", a), symbolic.DirichletBC(c_1, "Outlet", b)]
disc = discrete.discretize([model], discrete.DomainMaps({cell: halves}))
disc.function(c_0).x.array[:] = a
disc.function(c_1).x.array[:] = b
solver = solvers.Solver(disc[model], bcs)
solver.solve()
# -

# The profiles are linear, with slopes $s_0$ in the metal and $s_1 = D_0 s_0
# / D_1$ in the liquid, the flux being continuous, and the concentration
# $y = a + s_0 / 2$ of the metal at the interface solves the quadratic
# $y^2 / K_S^2 + r y / K_H - (b + r a) / K_H = 0$, $r = D_0 / D_1$. P1
# elements are exact for linear profiles, so the solution matches it to
# round-off, and the multiplier is the flux $-D_0 s_0$:

# +
r = D_0 / D_1
A, B, C = 1 / K_S**2, r / K_H, -(b + r * a) / K_H
y = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
s_0 = 2 * (y - a)
s_1 = r * s_0
x_0, x_1 = ufl.SpatialCoordinate(metal), ufl.SpatialCoordinate(liquid)
error_0 = np.sqrt(disc.assemble_scalar((c_0 - (a + s_0 * x_0[0])) ** 2 * metal.dx()))
error_1 = np.sqrt(
    disc.assemble_scalar((c_1 - (b + s_1 * (x_1[0] - 1))) ** 2 * liquid.dx())
)
print(
    f"{solver.iterations} Newton iterations, L2 errors {error_0:.1e} and {error_1:.1e}"
)
print(f"At the interface: c_0 = {y:.4f} in the metal, c_1 = {b - s_1 / 2:.4f}")
print(f"Flux: {disc.function(flux).x.array.mean():.4f}, exact {-D_0 * s_0:.4f}")
assert error_0 < 1e-10 and error_1 < 1e-10
assert np.allclose(disc.function(flux).x.array, -D_0 * s_0)
# -

# Plotted as surfaces raised by the concentration, in the metal and the
# liquid on one colour scale, the solution is linear on each side and jumps
# at the interface, where the partial pressure is continuous:

# +
metal_grid = gather_grid(disc[c_0], name="c")
liquid_grid = gather_grid(disc[c_1], name="c")
if MPI.COMM_WORLD.rank == 0:
    plotter = pyvista.Plotter(window_size=(640, 480))
    limits = [
        min(g["c"].min() for g in (metal_grid, liquid_grid)),
        max(g["c"].max() for g in (metal_grid, liquid_grid)),
    ]
    for label, grid in (("metal", metal_grid), ("liquid", liquid_grid)):
        plotter.add_mesh(
            grid.warp_by_scalar("c", factor=0.25),
            scalars="c",
            clim=limits,
            show_edges=True,
            show_scalar_bar=label == "metal",
            scalar_bar_args={"title": "c", "vertical": True},
        )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.show()
# -

# The concentration jumps at the interface, while the partial pressure, and
# the flux, are continuous.
#
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
