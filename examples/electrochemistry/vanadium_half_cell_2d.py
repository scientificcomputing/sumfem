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

# # A vanadium half-cell in two dimensions, with flow
#
# This demo extends {doc}`vanadium_half_cell_1d` with the flow of the
# electrolyte through the porous electrode, as the 2D example of
# [RFBniCS](https://github.com/Ah-Aghabarari/RFBniCS) {footcite}`Aghabarari2026`.
# It shows how to:
#
# - Solve Darcy flow through the electrode with
#   {py:class}`Darcy <sumfem.models.Darcy>`, and check it against its exact
#   solution
# - Couple the half-cell to the flow, which carries the ions in at the inlet
#   and out at the outlet, and limits the kinetics by mass transfer to the
#   fibres
# - Name the dimensionless numbers of the flow from the expressions of the
#   model
# - Run the half-cell through a charging and discharging schedule, and plot
#   its local state of charge
#
# ## The model
#
# The electrode spans the width $W$ from the current collector at $x = 0$ to
# the membrane at $x = W$, and the height $H$ from the inlet at $y = 0$ to
# the outlet at $y = H$. The electrolyte enters at the inlet with the
# superficial velocity $v_{\text{in}}$ and leaves at the outlet, held at the
# pressure $p_{\text{out}}$. In the pores, Darcy's law gives the
# superficial velocity $\mathbf{u}$ from the pressure $p$, with the
# permeability $K$ of the electrode and the viscosity $\mu$ of the
# electrolyte:
#
# $$
# \begin{aligned}
#   \mathbf{u} &= -\frac{K}{\mu} \nabla p, \\
#   \nabla \cdot \mathbf{u} &= 0, \\
#   -\mathbf{u} \cdot \mathbf{n} &= v_{\text{in}} \quad \text{at the inlet}, \\
#   p &= p_{\text{out}} \quad \text{at the outlet},
# \end{aligned}
# $$ (darcy)
#
# and no flow through the other sides. Darcy's law is the balance of the
# pressure gradient and the Darcy drag in the volume-averaged Navier-Stokes
# equations, whose inertia, viscous diffusion and Forchheimer drag are
# negligible in the electrode {footcite}`Aghabarari2026` (Section II B 5). The
# permeability follows from the porosity and the fibre diameter by the
# Kozeny-Carman relation, see
# {py:func}`kozeny_carman <sumfem.models.kozeny_carman>`.
#
# The half-cell is that of {eq}`half-cell` in {doc}`vanadium_half_cell_1d`,
# with three changes. The Nernst-Planck fluxes carry the ions with
# $\mathbf{u}$. The concentrations of the reservoir hold at the inlet
# instead of the membrane, the ions leave with the flow at the outlet, and
# the membrane passes the current of the cell as protons. The reaction takes
# the concentrations at the surface of the fibres, which mass transfer from
# the pores, at the rate $k_m (c - c_s)$, ties to the concentrations $c$ of
# the pores, with a mass-transfer coefficient $k_m$ from the correlation of
# {footcite}`Krishnamurthy2011`, see
# {py:class}`~sumfem.electrochemistry.MassTransferButlerVolmer`. These are the
# boundary conditions and kinetics of {footcite}`Aghabarari2026` (Sections II B 4
# and II C).
# The flow is steady and does not depend on the electrochemistry, so it is
# solved once, first.
#
# We import the relevant packages:

# + tags=["hide-input"]
from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import sympy.physics.units as syu
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.electrochemistry import Ion, PorousElectrode, RedoxHalfCell
from sumfem.models import BoundaryFlux, Darcy
from sumfem.plotting import gather_grid

# -

# ## The electrode
#
# The cell is a rectangle of quadrilaterals, with its sides marked. They are
# located with a tolerance relative to the height, see
# {ref}`units-length-scale`:

# +
width, height, cells = 2.5e-3, 1e-2, (25, 100)
dtype = dolfinx.default_real_type
rectangle = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    np.array([[0.0, 0.0], [width, height]]),
    list(cells),
    cell_type=dolfinx.mesh.CellType.quadrilateral,
    dtype=dtype,
)
tolerance = 100 * float(np.finfo(dtype).eps) * height
marked = discrete.MeshData.from_locators(
    rectangle,
    {
        1: {
            "CurrentCollector": lambda x: np.isclose(x[0], 0.0, atol=tolerance),
            "Membrane": lambda x: np.isclose(x[0], width, atol=tolerance),
            "Inlet": lambda x: np.isclose(x[1], 0.0, atol=tolerance),
            "Outlet": lambda x: np.isclose(x[1], height, atol=tolerance),
        }
    },
    boundary=True,
)
domain = symbolic.Domain.from_mesh("cell", rectangle, length_scale=width)
# -

# The porous electrode is the carbon felt of {doc}`vanadium_half_cell_1d`,
# whose fibres of several micrometres {footcite}`Barzegari2026` now matter: their
# diameter sets the permeability and the mass transfer. RFBniCS
# {footcite}`Aghabarari2026` uses a diameter far below that range, kept in a
# comment:

electrode = PorousElectrode(
    "electrode",
    domain,
    porosity=0.94,
    conductivity=3402.1 * syu.siemens / syu.m,
    specific_area=2.3e4 / syu.m,
    fiber_diameter=8.0 * syu.micrometer,  # RFBniCS: 0.5 micrometre
    kozeny_carman_constant=5.55,
    pore_length=100.0 * syu.micrometer,
)

# ## The flow
#
# The density and the viscosity of the electrolyte are parameters that the
# flow and the half-cell share, as are the inlet velocity and the outlet
# pressure. The flow depends on the gradient of the pressure alone, so the
# pressure is the gauge pressure, zero at the outlet: an offset as large as
# the atmospheric pressure would cost the small pressure drop its digits.

# +
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
    doc="Superficial velocity at the inlet",
)
outlet_pressure = symbolic.Parameter(
    "flow.outlet_pressure",
    domain,
    default=0.0 * syu.pascal,
    unit=syu.pascal,
    doc="Gauge pressure at the outlet",
)
# -

# The pressure drop $\mu v_{\text{in}} H / K$ of the uniform flow through the
# electrode is the scale of the pressure.
# {py:func}`value_at_defaults <sumfem.symbolic.value_at_defaults>` evaluates
# it, an expression of parameters, at their defaults:

# +
pressure_drop = symbolic.value_at_defaults(
    viscosity * inlet_velocity * height / electrode.permeability()
)
p = symbolic.Field("p", domain, ("Lagrange", 1), unit=syu.pascal, scale=pressure_drop)
# -

# {py:class}`Darcy <sumfem.models.Darcy>` is the flow of {eq}`darcy`, and a
# {py:class}`BoundaryFlux <sumfem.models.BoundaryFlux>` its inflow. Its system
# is checked for units and solved in dimensionless form, see
# {ref}`units-scaling`:

darcy = Darcy("darcy", p, electrode.permeability(), viscosity)
flow = darcy + BoundaryFlux("inflow", p, inlet_velocity, ("Inlet",))
flow_system = symbolic.build_system(flow, check_units=True, factorize_equations=True)
outlet = symbolic.DirichletBC(p, "Outlet", outlet_pressure)

# ## The half-cell
#
# The ions and the reservoir are those of {doc}`vanadium_half_cell_1d`, with
# sulfate from electroneutrality:

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
# -

# The half-cell is the negative one of {doc}`vanadium_half_cell_1d`. The
# superficial velocity of the flow carries the ions and drives the mass
# transfer to the fibres, which a velocity switches on. The flow carries the
# ions out at the `outlet`, the membrane passes the current as protons with
# `proton_flux`, and the reservoir concentrations hold at the inlet:

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
    velocity=darcy.superficial_velocity,
    outlet=("Outlet",),
    proton_flux=True,
    density=density,
    viscosity=viscosity,
)
inlet = cell.inlet_conditions("Inlet")

# The flow and the half-cell are discretized together, so they share their
# parameters, with the quadrature degree of the nonlinear kinetics bounded as
# in {doc}`vanadium_half_cell_1d`. A {py:class}`Solver <sumfem.solvers.Solver>`
# solves the flow:

disc = discrete.discretize(
    [flow_system, outlet, cell.model, *inlet],
    discrete.DomainMaps({domain: marked}),
    max_quadrature_degree=4,
)
solvers.Solver(disc[flow_system], [outlet]).solve()

# With a uniform permeability, the flow is uniform, and the pressure falls
# linearly from the inlet to the outlet by $\mu v_{\text{in}} H / K$. Linear
# elements hold it exactly, so the pressure drop and the outflow, the flux of
# the superficial velocity through the outlet, match it to round-off:

# +
n = ufl.FacetNormal(domain)
ds_inlet, ds_outlet = domain.ds("Inlet"), domain.ds("Outlet")
area = disc.assemble_scalar(1.0 * domain.dx())
v_in = float(disc.value(inlet_velocity))
drop = disc.assemble_scalar((p - outlet_pressure) * ds_inlet) / width
outflow = disc.assemble_scalar(ufl.dot(darcy.superficial_velocity, n) * ds_outlet)
print(f"Pressure drop: {drop:.6g} Pa, exact {pressure_drop:.6g} Pa")
print(f"Outflow over inflow: {outflow / (v_in * width):.12f}")
assert np.isclose(drop, pressure_drop, rtol=1e-8)
assert np.isclose(outflow, v_in * width, rtol=1e-8)
# -

# ## Dimensionless numbers of the flow
#
# The flow past the fibres has the [Reynolds
# number](https://dimless.org/docs/numbers/reynolds/) $\rho |\mathbf{u}| d /
# \mu$ of the fibre diameter $d$, which the kinetics compute, and sets the
# [Sherwood number](https://dimless.org/docs/numbers/sherwood/) $k_m d / D$
# of the mass transfer by the correlation of
# {py:class}`~sumfem.electrochemistry.MassTransferButlerVolmer`.
# Along the electrode, the [mass Péclet
# number](https://dimless.org/docs/numbers/mass-peclet/) $v_{\text{in}} H / D$
# of V²⁺ compares the transport by the flow with diffusion. Their means over
# the electrode:

# +
reaction = cell.reaction
reynolds = disc.assemble_scalar(reaction.reynolds_number() * domain.dx()) / area
d_V2 = float(disc.value(cell.electrolyte.diffusivity("V2")))
d_fibre = float(disc.value(electrode.fiber_diameter))
(k_V2, _) = reaction.mass_transfer_coefficients()
sherwood = disc.assemble_scalar(k_V2 * domain.dx()) / area * d_fibre / d_V2
peclet = v_in * height / d_V2
print(f"Reynolds number of the fibres: {reynolds:.3g}")
print(f"Sherwood number of V2+: {sherwood:.3g}")
print(f"Mass Peclet number of V2+ along the electrode: {peclet:.3g}")
assert reynolds < 1 and peclet > 1
# -

# The flow is creeping, so Darcy's law holds, yet it carries the ions along
# the electrode faster than they diffuse.
#
# ## Charge, rest and discharge
#
# The schedule of {doc}`vanadium_half_cell_1d`, run by a
# {py:class}`Simulation <sumfem.solvers.Simulation>` in parts, one for each
# current and step size. The simulation's model is the half-cell alone: the
# flow does not depend on the electrochemistry and its inputs are fixed, so
# every step takes the pressure solved above as known, and the velocity from
# it, without solving the flow again. The initial state is at rest, and the
# reference time of the time derivatives is the largest step:

# +
schedule = (
    ("charge", 0.5, ((20, 0.1), (10, 1.0), (10, 10.0))),
    ("rest", 0.0, ((5, 0.1), (5, 1.0), (10, 10.0))),
    ("discharge", 0.5, ((20, 0.1), (10, 1.0), (10, 10.0))),
)
sign = {"charge": -1.0, "rest": 0.0, "discharge": 1.0}
largest_step = max(dt for *_, steps in schedule for _, dt in steps)
cell.initialize(disc)
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

# After every step, a callback records the time, the mode, the voltage and
# the state of charge. At the end of the charge, the local state of charge,
# the share of V²⁺ in the vanadium at each point, is kept for a plot:

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

# Charging raises the state of charge and discharging lowers it. The flow
# now renews the electrolyte, so at rest it carries the charged electrolyte
# out and brings the reservoir's in: the state of charge falls back towards
# that of the reservoir:

# +
steps = {mode: np.diff(charge[modes == mode]) for mode in ("charge", "discharge")}
rest = charge[modes == "rest"]
print(f"State of charge after charge: {charge[modes == 'charge'][-1]:.6f}")
print(f"State of charge after rest: {rest[-1]:.6f}")
print(f"State of charge at the end: {charge[-1]:.6f}")
assert np.all(steps["charge"] > 0) and np.all(steps["discharge"] < 0)
assert np.all(np.diff(rest) < 0)
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

# The local state of charge at the end of the charge, gathered from all
# processes with {py:func}`gather_grid <sumfem.plotting.gather_grid>`. The
# electrode is drawn in millimetres, widened to show its width:

grid = gather_grid(charged)
if MPI.COMM_WORLD.rank == 0:
    grid.points[:, 0] *= 4e3
    grid.points[:, 1] *= 1e3
    plotter = pyvista.Plotter(window_size=[480, 640])
    plotter.add_mesh(grid, scalars="soc", scalar_bar_args={"title": "SOC"})
    plotter.view_xy()
    plotter.show()

# ## References
#
# ```{footbibliography}
# ```
