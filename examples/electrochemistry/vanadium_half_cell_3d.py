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

# # A vanadium half-cell in three dimensions, with flow channels
#
# This demo runs the half-cell of {doc}`vanadium_half_cell_2d` in a
# flow-through electrode with two channels, as the 3D example of
# [RFBniCS](https://github.com/Ah-Aghabarari/RFBniCS) {footcite}`Aghabarari2026`.
# It shows how to:
#
# - Build a mesh with physical groups in [Gmsh](https://gmsh.info), in
#   memory, and mark it with them
# - Solve the flow of the electrolyte through the electrode and along the
#   channels, Brinkman-Forchheimer in the electrode and Navier-Stokes in the
#   channels, from a creeping-flow start
# - Couple the half-cell to the flow in the electrode and the channels, with
#   the current of the cell balanced between the membrane and the smaller
#   current collector
#
# ## The model
#
# The electrode is a thin plate. Two channels lie against its front: the
# electrolyte enters through the inlet channel, crosses the electrode along
# its height and leaves through the outlet channel. The front of the
# electrode that the channels leave free is the current collector, and its
# back the membrane. In the electrode, of porosity $\varepsilon$, the
# superficial velocity $\mathbf{u}$ and the pressure $p$ follow
# Brinkman-Forchheimer flow, with the Darcy drag of the permeability $K$
# and the inertial drag of the Forchheimer coefficient $\beta$:
#
# $$
# \begin{aligned}
#   \frac{\rho}{\varepsilon^2} (\mathbf{u} \cdot \nabla) \mathbf{u}
#   - \nabla \cdot \left( \frac{2 \mu}{\varepsilon}
#   \boldsymbol{\epsilon}(\mathbf{u}) \right) + \nabla p
#   + \frac{\mu}{K} \mathbf{u} + \rho \beta |\mathbf{u}| \mathbf{u}
#   &= 0, \\
#   \nabla \cdot \mathbf{u} &= 0,
# \end{aligned}
# $$ (brinkman-forchheimer)
#
# with the symmetric gradient $\boldsymbol{\epsilon}$, see
# {py:class}`BrinkmanForchheimer <sumfem.models.BrinkmanForchheimer>`. In
# the channels, $\varepsilon = 1$ and the drag vanishes: the flow is
# Navier-Stokes flow. The velocity is given at the inlet, vanishes on the
# walls, the current collector and the membrane, and the outlet is free of
# stress. The Forchheimer coefficient is Ergun's, an estimate for fibres,
# see {py:func}`ergun_coefficient <sumfem.models.ergun_coefficient>`.
#
# The half-cell is that of {doc}`vanadium_half_cell_2d`, with the
# electrolyte also in the channels, where it is stored and transported
# without porosity and without reaction.
#
# We import the relevant packages:

# + tags=["hide-input"]
from mpi4py import MPI

import dolfinx.fem
import dolfinx.io.gmsh
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import sympy.physics.units as syu
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.electrochemistry import Ion, PorousElectrode, RedoxHalfCell
from sumfem.models import Brinkman, BrinkmanForchheimer, GradDiv, NavierStokes, Stokes
from sumfem.plotting import gather_grid

# -

# ## The mesh
#
# The geometry is the flow-through flow field of RfbFoam
# {footcite}`Barzegari2026`, as RFBniCS {footcite}`Aghabarari2026` (Table III) builds
# it, from whose example (MIT licence) the code below is adapted, in
# millimetres: a plate electrode, and two channels with rounded ends that
# open at its front. A channel is a box with a cylinder at each end:


# +
def rounded_channel(
    x_start: float, x_depth: float, y_center: float, z_start: float, z_end: float
) -> int:
    """A channel of unit width along z with rounded ends, open along x."""
    radius = 0.5
    box = gmsh.model.occ.addBox(
        x_start, y_center - radius, z_start, x_depth, 2 * radius, z_end - z_start
    )
    caps = [
        gmsh.model.occ.addCylinder(x_start, y_center, z, x_depth, 0.0, 0.0, radius)
        for z in (z_start, z_end)
    ]
    fused, _ = gmsh.model.occ.fuse([(3, box)], [(3, cap) for cap in caps])
    return fused[0][1]


# -

# The electrode and the channels are fragmented into one conforming mesh,
# with the electrode and each channel a physical group of cells. The
# boundary surfaces are told apart by where they lie: the open ends of the
# channels are the inlet and the outlet, the free front of the electrode the
# current collector, its back the membrane, and the rest walls:


# +
def flow_through_electrode(sizes: tuple[float, float]) -> None:
    """The Gmsh model of the cell, meshed with cells of the given sizes."""
    height, width, thickness, depth = 15.0, 17.0, 0.42, 1.0
    z_start, z_end, y_inlet, y_outlet = 1.5, 13.5, 1.0, 16.0
    electrode = gmsh.model.occ.addBox(depth, 0.0, 0.0, thickness, width, height)
    channels = [
        rounded_channel(0.0, depth, y, z_start, z_end) for y in (y_inlet, y_outlet)
    ]
    gmsh.model.occ.synchronize()
    _, fragments = gmsh.model.occ.fragment([(3, electrode)], [(3, c) for c in channels])
    gmsh.model.occ.synchronize()
    for name, entities in zip(
        ("Electrode", "InletChannel", "OutletChannel"), fragments, strict=True
    ):
        tags = [tag for dim, tag in entities if dim == 3]
        gmsh.model.addPhysicalGroup(3, tags, name=name)
    surfaces: dict[str, list[int]] = {
        "Inlet": [],
        "Outlet": [],
        "CurrentCollector": [],
        "Membrane": [],
        "Walls": [],
    }
    tol = 1e-4
    for dim, tag in gmsh.model.getEntities(2):
        cells, _ = gmsh.model.getAdjacencies(dim, tag)
        if len(cells) == 2:
            # An interface between the electrode and a channel
            continue
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        if np.isclose(xmin, 0.0, atol=tol) and np.isclose(xmax, 0.0, atol=tol):
            y = 0.5 * (ymin + ymax)
            near_inlet = abs(y - y_inlet) < abs(y - y_outlet)
            surfaces["Inlet" if near_inlet else "Outlet"].append(tag)
        elif np.isclose(xmin, depth, atol=tol) and np.isclose(xmax, depth, atol=tol):
            surfaces["CurrentCollector"].append(tag)
        elif np.isclose(xmin, depth + thickness, atol=tol) and np.isclose(
            xmax, depth + thickness, atol=tol
        ):
            surfaces["Membrane"].append(tag)
        else:
            surfaces["Walls"].append(tag)
    for name, tags in surfaces.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.option.setNumber("Mesh.MeshSizeMin", sizes[0])
    gmsh.option.setNumber("Mesh.MeshSizeMax", sizes[1])
    gmsh.model.mesh.generate(3)


# -

# The first process builds the model, and
# {py:func}`model_to_mesh <dolfinx.io.gmsh.model_to_mesh>` distributes it
# without a file. The mesh is coarse, a few cells across the electrode, to
# keep the demo short; RFBniCS meshes far finer. Gmsh scales coordinates only
# when it writes a file, so they are scaled from millimetres to metres here.
# {py:meth}`MeshData.from_gmsh <sumfem.discrete.MeshData.from_gmsh>` names
# the markers after the physical groups:

# +
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
if MPI.COMM_WORLD.rank == 0:
    flow_through_electrode(sizes=(0.2, 0.5))
data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
gmsh.finalize()
data.mesh.geometry.x[:] *= 1e-3
marked = discrete.MeshData.from_gmsh(data)
thickness = 0.42e-3
domain = symbolic.Domain.from_mesh("cell", data.mesh, length_scale=thickness)
num_cells = data.mesh.topology.index_map(3).size_global
print(f"Cells: {num_cells}")
# -

# ## The electrode and the electrolyte
#
# The electrode of {doc}`vanadium_half_cell_2d` fills the cells marked
# `Electrode`, the channels the others. The density and the viscosity of the
# electrolyte are shared by the flow and the half-cell:

# +
electrode = PorousElectrode(
    "electrode",
    domain,
    markers=("Electrode",),
    porosity=0.94,
    conductivity=3402.1 * syu.siemens / syu.m,
    specific_area=2.3e4 / syu.m,
    fiber_diameter=8.0 * syu.micrometer,  # RFBniCS: 0.5 micrometre
    kozeny_carman_constant=5.55,
    pore_length=100.0 * syu.micrometer,
)
channels = ("InletChannel", "OutletChannel")
viscosity = symbolic.Parameter(
    "electrolyte.viscosity",
    domain,
    default=6.963e-3 * syu.pascal * syu.second,
    unit=syu.pascal * syu.second,
    doc="Dynamic viscosity of the electrolyte",
)
density = symbolic.Parameter(
    "electrolyte.density",
    domain,
    default=1399.8 * syu.kg / syu.m**3,
    unit=syu.kg / syu.m**3,
    doc="Density of the electrolyte",
)
inlet_velocity = symbolic.Parameter(
    "flow.inlet_velocity",
    domain,
    default=1e-7 * syu.m / syu.second,
    unit=syu.m / syu.second,
    doc="Velocity at the inlet",
)
# -

# ## The flow
#
# The velocity is of quadratic and the pressure of linear elements, the
# Taylor-Hood pair. The inlet velocity scales the velocity, and the pressure
# drop of Darcy flow at it across the height of the electrode, evaluated at
# the defaults of the parameters with
# {py:func}`value_at_defaults <sumfem.symbolic.value_at_defaults>`, scales
# the pressure:

# +
height = 15e-3
pressure_scale = symbolic.value_at_defaults(
    viscosity * inlet_velocity * height / electrode.permeability()
)
u = symbolic.Field(
    "u",
    domain,
    ("Lagrange", 2, (3,)),
    unit=syu.m / syu.second,
    scale=symbolic.value_at_defaults(inlet_velocity),
)
p = symbolic.Field("p", domain, ("Lagrange", 1), unit=syu.pascal, scale=pressure_scale)
# -

# The flow of {eq}`brinkman-forchheimer` is a sum of
# {py:class}`BrinkmanForchheimer <sumfem.models.BrinkmanForchheimer>` in the
# electrode and {py:class}`NavierStokes <sumfem.models.NavierStokes>` in the
# channels, with grad-div stabilization,
# {py:class}`GradDiv <sumfem.models.GradDiv>`, whose weight RFBniCS takes as
# a small part of the viscosity. Its creeping-flow part,
# {py:class}`Brinkman <sumfem.models.Brinkman>` and
# {py:class}`Stokes <sumfem.models.Stokes>`, is linear, and its solution the
# start of Newton's method for the whole flow. Both are checked for units
# and solved in dimensionless form:

# +
grad_div = GradDiv("grad_div", u, 1e-3 * viscosity)
creeping = (
    Brinkman(
        "electrode_creeping_flow",
        u,
        p,
        electrode.porosity,
        electrode.permeability(),
        viscosity,
        markers=electrode.markers,
    )
    + Stokes("channel_creeping_flow", u, p, viscosity, markers=channels)
    + grad_div
)
flow = (
    BrinkmanForchheimer(
        "electrode_flow",
        u,
        p,
        electrode.porosity,
        electrode.permeability(),
        electrode.forchheimer(),
        viscosity,
        density,
        markers=electrode.markers,
    )
    + NavierStokes("channel_flow", u, p, viscosity, density, markers=channels)
    + grad_div
)
flow_systems = [
    symbolic.build_system(model, check_units=True, factorize_equations=True)
    for model in (creeping, flow)
]
flow_bcs = [
    symbolic.DirichletBC(u, "Inlet", ufl.as_vector([inlet_velocity, 0.0, 0.0])),
    symbolic.DirichletBC(u, ("Walls", "CurrentCollector", "Membrane"), 0.0),
]
# -

# ## The half-cell
#
# The ions, the reservoir and the half-cell are those of
# {doc}`vanadium_half_cell_2d`. The electrolyte fills the `channels` too.
# The membrane is larger than the current collector, which the channels
# partly cover, so the cell's current density, given at the membrane, is
# scaled at the collector by the ratio of their areas:

# +
diffusivity = syu.m**2 / syu.second
concentration = syu.mol / syu.m**3
ions = (
    Ion("V2", 2, 8.10e-12 * diffusivity, positive=True),
    Ion("V3", 3, 1.65e-11 * diffusivity, positive=True),
    Ion("H", 1, 9.312e-9 * diffusivity),
    Ion("HSO4", -1, 1.33e-9 * diffusivity),
    Ion("SO4", -2, 1.065e-9 * diffusivity),
)
reservoir = {
    "V2": 300.0 * concentration,
    "V3": 300.0 * concentration,
    "H": 1000.0 * concentration,
    "HSO4": 1000.0 * concentration,
}
cell = RedoxHalfCell(
    domain,
    ions,
    reservoir,
    reduced="V2",
    oxidized="V3",
    electrons=1,
    electrode=electrode,
    oxidation_on_discharge=True,
    rate_constant=2.5e-7 * syu.m / syu.second,
    formal_potential=-0.36 * syu.volt,
    channels=channels,
    velocity=u,
    outlet=("Outlet",),
    proton_flux=True,
    density=density,
    viscosity=viscosity,
)
inlet = cell.inlet_conditions("Inlet")
# -

# ## Solving the flow
#
# Everything is discretized together, and the creeping flow and then the
# whole flow are solved:

# +
disc = discrete.discretize(
    [*flow_systems, *flow_bcs, cell.model, *inlet],
    discrete.DomainMaps({domain: marked}),
    max_quadrature_degree=4,
)
for name, system in zip(("Creeping flow", "Flow"), flow_systems, strict=True):
    solver = solvers.Solver(disc[system], flow_bcs)
    solver.solve()
    print(f"{name}: {solver.iterations} Newton iterations")
# -

# The flow is so slow that its inertia, and the Forchheimer drag, change it
# by less than Newton's tolerance: the creeping flow is already the flow. It
# is incompressible, so what enters at the inlet leaves at the outlet. The
# velocity of Taylor-Hood elements is not divergence-free cell by cell, but
# the constants are among the test functions of the pressure, so it
# conserves mass over the whole cell, and the outflow matches the inflow to
# round-off:

# +
n = ufl.FacetNormal(domain)
inflow = -disc.assemble_scalar(ufl.dot(u, n) * domain.ds("Inlet"))
outflow = disc.assemble_scalar(ufl.dot(u, n) * domain.ds("Outlet"))
print(f"Outflow over inflow: {outflow / inflow:.8f}")
assert inflow > 0 and np.isclose(outflow, inflow, rtol=1e-8)
# -

# ## Charge, rest and discharge
#
# The cell starts at rest, with the current balanced between the membrane
# and the collector, and follows the schedule of
# {doc}`vanadium_half_cell_2d`. As there, the simulation's model is the
# half-cell alone, and every step takes the velocity solved above as known:

# +
cell.balance_currents(disc)
cell.initialize(disc)
schedule = (
    ("charge", 0.5, ((20, 0.1), (10, 1.0), (10, 10.0))),
    ("rest", 0.0, ((5, 0.1), (5, 1.0), (10, 10.0))),
    ("discharge", 0.5, ((20, 0.1), (10, 1.0), (10, 10.0))),
)
sign = {"charge": -1.0, "rest": 0.0, "discharge": 1.0}
largest_step = max(dt for *_, steps in schedule for _, dt in steps)
clock = symbolic.Clock(domain, dt=schedule[0][2][0][1])
simulation = solvers.Simulation(
    disc,
    cell.model,
    inlet,
    clock=clock,
    scheme=solvers.BDF(1),
    petsc_options={"snes_linesearch_type": "bt", "snes_max_it": 100},
    check_units=True,
    factorize_equations=True,
    time_scale=largest_step,
)
# -

# The simulation runs in parts, as in {doc}`vanadium_half_cell_2d`, and keeps
# the local state of charge at the end of the charge:

# +
local = cell.local_state_of_charge()
space = disc.function_space(cell.concentrations["V2"])
charged = dolfinx.fem.Function(space, name="soc")
records: list[tuple[float, str, float, float]] = []
for mode, current, steps in schedule:
    disc.set({cell.current: sign[mode] * current})
    for count, dt in steps:
        disc.set({clock.dt: dt})
        simulation.run(
            num_steps=count,
            callback=lambda t, step, mode=mode: records.append(
                (t, mode, cell.voltage(disc), cell.state_of_charge(disc))
            ),
        )
    if mode == "charge":
        charged.interpolate(
            dolfinx.fem.Expression(disc[local], space.element.interpolation_points)
        )
times = np.array([r[0] for r in records])
modes = np.array([r[1] for r in records])
voltage = np.array([r[2] for r in records])
charge = np.array([r[3] for r in records])
# -

# Charging raises the state of charge and discharging lowers it, and the
# voltage is lower on charge than on discharge:

# +
steps = {mode: np.diff(charge[modes == mode]) for mode in ("charge", "discharge")}
print(f"State of charge after charge: {charge[modes == 'charge'][-1]:.6f}")
print(f"State of charge at the end: {charge[-1]:.6f}")
assert np.all(steps["charge"] > 0) and np.all(steps["discharge"] < 0)
assert voltage[modes == "charge"].mean() < voltage[modes == "discharge"].mean()
assert np.isfinite(voltage).all()
# -

# The voltage and the state of charge through the schedule:

fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
for mode in ("charge", "rest", "discharge"):
    top.plot(times[modes == mode], 1e3 * voltage[modes == mode], ".", label=mode)
    bottom.plot(times[modes == mode], charge[modes == mode], ".", label=mode)
top.set_ylabel("Voltage (mV)")
bottom.set_ylabel("State of charge")
bottom.set_xlabel("Time (s)")
top.legend()
plt.show()

# The local state of charge at the end of the charge, on the membrane side
# of the electrode, in millimetres:

grid = gather_grid(charged)
if MPI.COMM_WORLD.rank == 0:
    grid.points[:] *= 1e3
    plotter = pyvista.Plotter(window_size=[640, 640])
    plotter.add_mesh(grid, scalars="soc", scalar_bar_args={"title": "SOC"})
    plotter.view_yz(negative=True)
    plotter.show()

# ## References
#
# ```{footbibliography}
# ```
