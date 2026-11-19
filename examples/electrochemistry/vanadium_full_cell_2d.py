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

# # A vanadium redox flow cell in two dimensions
#
# This demo builds a full cell model of Shah, Watt-Smith and Walsh
# {footcite}`Shah2008`, with a slight correction, see
# {ref}`full-cell-corrections`: two half-cells of
# {doc}`vanadium_half_cell_2d`, a proton-exchange membrane between them and
# a current collector on either side, with the electrolytes flowing through
# the electrodes from their tanks. It shows how to:
#
# - Describe each side of the cell with a
#   {py:class}`HalfCellSpec <sumfem.electrochemistry.HalfCellSpec>`, the
#   membrane with a
#   {py:class}`MembraneSpec <sumfem.electrochemistry.MembraneSpec>` and the
#   flow with a {py:class}`FlowSpec <sumfem.electrochemistry.FlowSpec>`
# - Build the cell with {py:class}`FullCell <sumfem.electrochemistry.FullCell>`
#   on a mesh of its layers, each unknown on the layers where it lives
# - Charge, rest and discharge the cell, with the tanks mixing what the
#   electrodes return
# - Check that the cell and its tanks conserve the vanadium, and that each
#   couple turns over the charge that passes
#
# ## The model
#
# The cell has five layers, from left to right: the negative current
# collector, the negative porous electrode, the membrane, the positive
# porous electrode and the positive current collector. Each half-cell is
# that of {doc}`vanadium_half_cell_1d`, {eq}`redox-half-cell`, for its couple:
# V²⁺/V³⁺ on the negative side and VO²⁺/VO₂⁺, which takes up water and gives
# off protons, {eq}`butler-volmer-couple`, on the positive side, with the
# piston velocities of Shah et al., {eq}`piston-velocity`. The membrane
# conducts protons only, {eq}`proton-exchange-membrane`, and lets water
# through by Schlögl's flow, {eq}`schloegl-velocity`. The electrolytes
# enter at the bottom of the electrodes and leave at the top,
# {eq}`full-cell-flow`, and return to their tanks, {eq}`full-cell-tank`.
#
# (full-cell-corrections)=
# ### Corrections to Shah et al.
#
# The model is that of Shah et al., with one correction to the model and
# some to the paper. The correction to the model is at the membrane:
#
# - At the faces of the membrane the protons of the electrodes carry its
#   current, {eq}`membrane-proton-flux`, and no other ion crosses. Shah et
#   al. hold the protons at the fixed charge of the membrane there, which
#   lets sulfate, whose concentration follows from electroneutrality, carry
#   part of the current through the faces.
#
# The corrections to the paper are:
#
# - Their Table 1 puts the protons that the positive couple makes in the
#   negative electrode.
# - Their eq. 22 has the decimal logarithm in place of the natural one, and
#   the ratio of V(IV) to itself in the positive potential, which should be
#   that of V(V) to V(IV). The text after it gives the formal potentials in
#   their Table 3; they are in their Table 4.
# - Their eq. 23 pairs the rate constant of each couple with the formal
#   potential of the other; their eq. 24 has them right.
# - Their Table 3 gives the volume of electrolyte of a half-cell in metres.
#   Millilitres match the charge time and the state of charge at the end of
#   the charge of their Figure 2.
# - Their Table 2 gives an initial water concentration ten times smaller
#   than the one their Figure 7 starts from, and their Figure 6 shows.
# - Their Table 5 gives the diffusivities of bisulfate and sulfate in the
#   membrane, which holds no anions in their model, and none of the
#   protons, bisulfate and sulfate in the electrolyte, which their eqs. 5
#   and 10 need. Those are taken from Knehr et al. {footcite}`Knehr2012`.
#
# The paper leaves two choices open. It does not say how the electrolyte
# potential, the water and the pressure meet across the faces of the
# membrane: they are continuous here. It relates its inlet velocity to the
# flow rate through the porosity, as a velocity in the pores, but uses it in
# Darcy's law, whose velocity is superficial: here it is the flow rate over
# the area of the inlet. The inlets also take in the flux of the
# electrolyte, rather than holding its concentration, so that the cell and
# its tanks exchange exactly what they hold.
#
# The model leaves out the Donnan potentials at the faces of the membrane
# and the protons in the potential of the positive couple, which Shah et al.
# make up for with a fitted contact resistance. Their measured curves are
# not compared with.

# +
from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import sympy.physics.units as syu

from sumfem import discrete, solvers, symbolic
from sumfem.electrochemistry import (
    DarcyFlow,
    FlowSpec,
    FullCell,
    HalfCellSpec,
    Ion,
    MembraneSpec,
    RedoxCouple,
)
from sumfem.plotting import gather_grid
from sumfem.symbolic.si import to_si

# -

# ## The cell
#
# The cell lives on a symbolic domain of quadrilaterals. Its models name the
# parts of the cell by markers, the layers and their faces, so any mesh with
# those markers discretizes it, see
# {py:meth}`mesh_data <sumfem.electrochemistry.FullCell.mesh_data>`. Its
# length scale is the thickness of an electrode, Shah et al.'s Table 3,
# which with the height and the width of the cell also sets the flow below:

# +
thickness = 4.0 * syu.millimeter
height, width = 10.0 * syu.centimeter, 10.0 * syu.centimeter
domain = symbolic.Domain("cell", "quadrilateral", length_scale=to_si(thickness, syu.m))
# -

# ## The sides
#
# Each side has a
# {py:class}`RedoxCouple <sumfem.electrochemistry.RedoxCouple>`, as the
# half-cell of {ref}`vanadium_half_cell_1d <vanadium-half-cell-couple>`:
# those of Shah et al.'s eqs. 1 and 2, with the formal potentials and the
# rate constants of their Table 4. Their eq. 20 has the porosity times the
# rate constant where the half-cell has its rate constant
# {eq}`butler-volmer-couple`, with the same specific area, so the rate
# constants are multiplied by the porosity.
#
# The positive couple, $\mathrm{VO^{2+} + H_2O \rightleftharpoons VO_2^+ +
# 2 H^+ + e^-}$, also takes up water and gives off protons when it is
# oxidized, on charge. Its `reactants` and `products` give every species of
# the reaction with its coefficient, the ions of the couple included, and
# the couple checks that the reaction conserves charge. The coefficients
# are those of the sources of the ions in each half-cell,
# {eq}`redox-half-cell`:

# +
porosity = 0.68
negative_couple = RedoxCouple(
    reduced="V2",
    oxidized="V3",
    rate_constant=porosity * 1.75e-7 * syu.m / syu.second,
    formal_potential=-0.255 * syu.volt,
    oxidation_on_discharge=True,
)
positive_couple = RedoxCouple(
    reduced="VO",
    oxidized="VO2",
    rate_constant=porosity * 3e-9 * syu.m / syu.second,
    formal_potential=1.004 * syu.volt,
    oxidation_on_discharge=False,
    reactants={"VO": 1, "H2O": 1},
    products={"VO2": 1, "H": 2},
)
# -

# Each electrolyte holds its couple, protons, bisulfate and water, with
# sulfate by electroneutrality. The diffusivities of the vanadium ions and
# the water are those of Shah et al.'s Table 5, and those of the protons,
# bisulfate and sulfate those of Knehr et al.'s Table II
# {footcite}`Knehr2012`, see {ref}`full-cell-corrections`:

# +
diffusivity = syu.m**2 / syu.second


def create_ions(reduced, oxidized, charges, couple_diffusivity):
    """The ions of a side: its couple, protons, bisulfate, water and sulfate.

    Args:
        reduced: The name of the couple's reduced ion.
        oxidized: The name of the couple's oxidized ion.
        charges: The charge numbers of the reduced and oxidized ion.
        couple_diffusivity: The diffusivity of both ions of the couple.
    """
    return (
        Ion(reduced, charges[0], couple_diffusivity),
        Ion(oxidized, charges[1], couple_diffusivity),
        Ion("H", 1, 9.312e-9 * diffusivity),
        Ion("HSO4", -1, 1.33e-9 * diffusivity),
        Ion("H2O", 0, 2.3e-9 * diffusivity),
        Ion("SO4", -2, 1.065e-9 * diffusivity),
    )


# -

# Both sides start at the state of charge of Shah et al.'s Table 2: the
# concentrations of the discharged and the charged ion of each couple, and
# those of the acid and the water, the latter corrected:

# +
concentration = syu.mol / syu.m**3
discharged, charged = 1053.0 * concentration, 27.0 * concentration
acid = {
    "H": 1200.0 * concentration,
    "HSO4": 1200.0 * concentration,
    "H2O": 4.2e4 * concentration,  # Shah et al., Table 2: 4.2e3
}
# -

# A {py:class}`HalfCellSpec <sumfem.electrochemistry.HalfCellSpec>` gathers
# what a side of the cell is made of: the ions of its electrolyte and their
# initial concentrations, its couple, and the properties of its electrode.
# The full cell makes a
# {py:class}`RedoxHalfCell <sumfem.electrochemistry.RedoxHalfCell>` of each
# side, see {doc}`vanadium_half_cell_1d`. Both electrodes are the carbon
# felt of Shah et al.'s Tables 3 and 5: its porosity, the conductivity of
# its fibres and their specific area, the distance between the fibres,
# which sets the piston velocities, {eq}`piston-velocity`, and their
# diameter and the Kozeny-Carman constant, which set the permeability of the
# flow, {py:func}`kozeny_carman <sumfem.models.kozeny_carman>`. The electrolytes have
# the viscosity of water, as theirs:

# +
carbon_felt = dict(
    porosity=porosity,
    conductivity=500.0 * syu.siemens / syu.m,
    specific_area=2e6 / syu.m,
    fiber_distance=10.0 * syu.micrometer,
    fiber_diameter=10.0 * syu.micrometer,
    kozeny_carman_constant=5.55,
    viscosity=1e-3 * syu.pascal * syu.second,
)
negative = HalfCellSpec(
    ions=create_ions("V2", "V3", (2, 3), 2.4e-10 * diffusivity),
    reservoir={"V2": charged, "V3": discharged, **acid},
    couple=negative_couple,
    **carbon_felt,
)
positive = HalfCellSpec(
    ions=create_ions("VO", "VO2", (2, 1), 3.9e-10 * diffusivity),
    reservoir={"VO": discharged, "VO2": charged, **acid},
    couple=positive_couple,
    **carbon_felt,
)
# -

# ## The membrane and the flow
#
# A {py:class}`MembraneSpec <sumfem.electrochemistry.MembraneSpec>` gives the
# properties of the
# {py:class}`ProtonExchangeMembrane <sumfem.electrochemistry.ProtonExchangeMembrane>`:
# the concentration of its fixed charges and the diffusivity of its
# protons, which set its conductivity, {eq}`proton-exchange-membrane`, the
# diffusivity of its water, and the permeabilities and the viscosity of
# Schlögl's flow through it, {eq}`schloegl-velocity`. Those of Shah et al.'s
# Tables 4 and 5:

membrane = MembraneSpec(
    fixed_charge=1200.0 * concentration,
    proton_diffusivity=1.4e-9 * diffusivity,
    water_diffusivity=5.75e-10 * diffusivity,
    electrokinetic_permeability=1.13e-19 * syu.m**2,
    hydraulic_permeability=1.58e-18 * syu.m**2,
    viscosity=1e-3 * syu.pascal * syu.second,
)

# A {py:class}`FlowSpec <sumfem.electrochemistry.FlowSpec>` adds the flow of
# the electrolytes to the cell. Its `model` is the flow, an
# {py:class}`ElectrodeFlow <sumfem.electrochemistry.ElectrodeFlow>`, which
# gives the superficial velocity of each electrode: any flow plugs in, e.g.
# Brinkman-Forchheimer or transient Stokes flow. Shah et al.'s is
# {py:class}`DarcyFlow <sumfem.electrochemistry.DarcyFlow>`: one pressure
# lives on the electrodes and the membrane, with
# {py:class}`Darcy <sumfem.models.Darcy>` flow in the electrodes,
# {eq}`darcy`, and
# {py:class}`Schlögl's <sumfem.electrochemistry.SchloeglFlow>` in the
# membrane, {eq}`schloegl-velocity`.
# The electrolyte of each side enters its electrode at the inlet, at the
# inlet velocity and with the concentrations of its tank, and leaves at the
# outlet, held at the outlet pressure, {eq}`full-cell-flow`. Each tank,
# whose volume is given relative to the pores of its electrode, mixes what
# the outlet returns, {eq}`full-cell-tank`.
#
# Shah et al. pump each electrolyte at the flow rate of their Table 2. The
# inlet velocity is the superficial velocity of that flow rate, over the
# inlet, the thickness of the electrode times the width of the cell. Each
# tank holds the volume of electrolyte of a half-cell, their Table 3, but
# for what fills the pores of the electrode:

# +
flow_rate, half_cell_volume = 1.0 * syu.milliliter / syu.second, 250 * syu.milliliter
inlet = thickness * width
pores = to_si(porosity * height * inlet, syu.m**3)
tank = to_si(half_cell_volume, syu.m**3) - pores
flow = FlowSpec(
    inlet_velocity=flow_rate / inlet, tank_ratio=tank / pores, model=DarcyFlow
)
# -

# {py:class}`FullCell <sumfem.electrochemistry.FullCell>` builds the cell
# from its sides, its membrane and its flow, with the conductivity of the
# collectors of Shah et al.'s Table 5. Its Dirichlet conditions,
# {py:meth}`conditions <sumfem.electrochemistry.FullCell.conditions>`, ground
# the negative terminal and hold the pressure at the outlets:

# +
cell = FullCell(
    domain,
    negative,
    positive,
    membrane=membrane,
    collector_conductivity=1000.0 * syu.siemens / syu.m,
    flow=flow,
)
conditions = cell.conditions()
# -

# ## The mesh
#
# The cell is discretized on the geometry of Shah et al., their Table 3: the
# layers with their thicknesses and the number of columns of cells across
# each, and the rows of cells over the height of the cell, from the inlets
# to the outlets:

# +
layers = (
    ("NegativeCollector", 6.0 * syu.millimeter, 3),
    ("NegativeElectrode", thickness, 16),
    ("Membrane", 180.0 * syu.micrometer, 3),
    ("PositiveElectrode", thickness, 16),
    ("PositiveCollector", 6.0 * syu.millimeter, 3),
)
rows = 40
# -

# A rectangle of quadrilaterals, whose columns are moved onto the edges of
# the layers, with the markers of the cell: the cells of each layer, the
# outer faces of the collectors, the faces of the membrane and the ends of
# the electrodes. They are located with a tolerance relative to the thinnest layer,
# see {ref}`units-length-scale`. The cell's unknowns live on unions of the
# layers, which
# {py:meth}`mesh_data <sumfem.electrochemistry.FullCell.mesh_data>` adds:

# +
edges = np.cumsum([0.0, *(to_si(thickness, syu.m) for _, thickness, _ in layers)])
columns = np.unique(
    np.concatenate(
        [
            np.linspace(a, b, n + 1)
            for a, b, (*_, n) in zip(edges[:-1], edges[1:], layers, strict=True)
        ]
    )
)
dtype = dolfinx.default_real_type
rectangle = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    np.array([[0.0, 0.0], [1.0, to_si(height, syu.m)]]),
    [len(columns) - 1, rows],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
    dtype=dtype,
)
index = np.rint(rectangle.geometry.x[:, 0] * (len(columns) - 1)).astype(int)
rectangle.geometry.x[:, 0] = columns[index]
tolerance = 100 * float(np.finfo(dtype).eps) * to_si(layers[2][1], syu.m)


def between(a, b):
    """The points from ``x = a`` to ``x = b``.

    Args:
        a: The left edge, in m.
        b: The right edge, in m.
    """
    return lambda x: (x[0] >= a - tolerance) & (x[0] <= b + tolerance)


def at(a):
    """The points at ``x = a``.

    Args:
        a: The position, in m.
    """
    return lambda x: np.isclose(x[0], a, atol=tolerance)


def end(a, b, y):
    """The points from ``x = a`` to ``x = b`` at the height ``y``.

    Args:
        a: The left edge, in m.
        b: The right edge, in m.
        y: The height, in m.
    """
    return lambda x: between(a, b)(x) & np.isclose(x[1], y, atol=tolerance)


top = to_si(height, syu.m)
marked = FullCell.mesh_data(
    discrete.MeshData.from_locators(
        rectangle,
        {
            2: {
                name: between(a, b)
                for (name, *_), a, b in zip(layers, edges[:-1], edges[1:], strict=True)
            },
            1: {
                "NegativeTerminal": at(edges[0]),
                "NegativeFace": at(edges[2]),
                "PositiveFace": at(edges[3]),
                "PositiveTerminal": at(edges[5]),
                "NegativeInlet": end(edges[1], edges[2], 0.0),
                "NegativeOutlet": end(edges[1], edges[2], top),
                "PositiveInlet": end(edges[3], edges[4], 0.0),
                "PositiveOutlet": end(edges[3], edges[4], top),
            },
        },
    )
)
# -

# ## The discrete problem
#
# The discretization pairs the symbolic domain with the marked mesh. It
# bounds the quadrature degree of the nonlinear kinetics, as in
# {doc}`vanadium_half_cell_1d`, whose integrals no degree makes exact; a
# higher bound changes the voltage and the states of charge only at
# round-off. The initial state is at rest, Shah et al.'s eq. 32: the
# concentrations of the tanks, the electrolyte at zero and each solid at the
# equilibrium potential of its couple:

disc = discrete.discretize(
    [cell.model, *conditions],
    discrete.DomainMaps({domain: marked}),
    max_quadrature_degree=4,
)
cell.initialize(disc)

# ## Charge, rest and discharge
#
# The current of Shah et al.'s Table 2 enters and leaves through the outer
# faces of the collectors, their eq. 31, whose area is the height times the
# width of the cell. The cell is charged for the time of
# their Figure 2, rested, and discharged until its voltage falls to a
# cut-off or the time of the charge has passed. Each operation gives its
# mode and its time steps, as pairs of a number of steps and their size in
# seconds:

# +
current = 10.0 * syu.ampere / (height * width)
density = to_si(current, syu.ampere / syu.m**2)
charging = ((7, 1.0), (10, 5.0), (98, 20.0))
schedule = (
    ("charge", -density, charging),
    ("rest", 0.0, ((12, 10.0),)),
    ("discharge", density, charging),
)
cutoff = 1.0
largest_step = max(dt for *_, steps in schedule for _, dt in steps)
# -

# A {py:class}`Simulation <sumfem.solvers.Simulation>` takes the steps with
# backward Euler and solves each in dimensionless form, see
# {ref}`units-scaling`. After every step, a callback takes a step of the
# tanks, {py:meth}`update_tanks
# <sumfem.electrochemistry.FullCell.update_tanks>`, and records the time,
# the mode, the voltage and the state of charge of each side, tank included,
# {py:meth}`state_of_charge
# <sumfem.electrochemistry.FullCell.state_of_charge>`:

# +
clock = symbolic.Clock(domain, dt=schedule[0][2][0][1])
simulation = solvers.Simulation(
    disc,
    cell.model,
    conditions,
    clock=clock,
    scheme=solvers.BDF(1),
    petsc_options={"snes_linesearch_type": "bt", "snes_max_it": 50},
    check_units=True,
    factorize_equations=True,
    time_scale=largest_step,
)
records: list[tuple[float, str, float, float, float]] = []


def record(t: float, mode: str, dt: float) -> None:
    """Take a step of the tanks, and record the state of the cell.

    Args:
        t: The time, in s.
        mode: The operation.
        dt: The size of the step, in s.
    """
    cell.update_tanks(disc, dt)
    records.append(
        (
            t,
            mode,
            cell.voltage(disc),
            cell.state_of_charge(disc, "negative"),
            cell.state_of_charge(disc, "positive"),
        )
    )


# -

# The amounts of vanadium on each side, in the electrode and its tank, before
# the schedule, and the charge that passes through the cell, per unit of its
# width, are kept for the checks below. At the end of the charge, the local
# state of charge of the negative electrode is kept for a plot:


# +
def vanadium(side: str) -> float:
    """The amount of vanadium of a side, in its electrode and its tank.

    Args:
        side: ``"negative"`` or ``"positive"``.
    """
    couple = cell.half_cells[side].couple
    return sum(
        cell.amount(disc, side, ion) for ion in (couple.reduced, couple.oxidized)
    )


start = {side: vanadium(side) for side in ("negative", "positive")}
reduced = cell.amount(disc, "negative", "V2")
negative_cell = cell.half_cells["negative"]
local = negative_cell.local_state_of_charge()
space = disc.function_space(negative_cell.concentrations["V2"])
local_charge = dolfinx.fem.Function(space, name="soc")
passed = {"charge": 0.0, "discharge": 0.0}
for mode, density_of_mode, steps in schedule:
    disc.set({cell.current: density_of_mode})
    stop = False
    for count, dt in steps:
        disc.set({clock.dt: dt})
        for _ in range(count):
            simulation.run(
                num_steps=1,
                callback=lambda t, step, mode=mode, dt=dt: record(t, mode, dt),
            )
            if mode in passed:
                passed[mode] += abs(density_of_mode) * dt * top
            if mode == "discharge" and records[-1][2] < cutoff:
                stop = True
                break
        if stop:
            break
    if mode == "charge":
        local_charge.interpolate(
            dolfinx.fem.Expression(disc[local], space.element.interpolation_points)
        )
times = np.array([r[0] for r in records])
modes = np.array([r[1] for r in records])
voltage = np.array([r[2] for r in records])
soc = {
    "negative": np.array([r[3] for r in records]),
    "positive": np.array([r[4] for r in records]),
}
# -

# ## Checks
#
# The cell and its tanks keep the vanadium of each side, and the negative
# couple turns over the charge that passed, divided by Faraday's constant.
# At rest the states of charge, tanks included, stay as they are, while the
# flow mixes the electrolytes of the electrodes and the tanks. Sulfate,
# which follows from electroneutrality, stays positive on both sides, see
# {py:meth}`minimum_balanced_concentration
# <sumfem.electrochemistry.RedoxHalfCell.minimum_balanced_concentration>`:

# +
faraday = float(disc.value(negative_cell.electrolyte.F))
turned_over = (passed["charge"] - passed["discharge"]) / faraday
for side in ("negative", "positive"):
    print(
        f"Vanadium of the {side} side: {vanadium(side) / start[side]:.12f} of the start"
    )
    assert np.isclose(vanadium(side), start[side], rtol=1e-9)
change = cell.amount(disc, "negative", "V2") - reduced
print(f"V2+ made over the charge passed: {change / turned_over:.9f}")
assert np.isclose(change, turned_over, rtol=1e-6)
rest = soc["negative"][modes == "rest"]
assert np.ptp(rest) < 1e-9 * rest.mean()
assert np.all(np.diff(soc["negative"][modes == "charge"]) > 0)
assert np.all(np.diff(soc["negative"][modes == "discharge"]) < 0)
assert voltage[modes == "charge"].mean() > voltage[modes == "discharge"].mean()
for half in cell.half_cells.values():
    assert half.minimum_balanced_concentration(disc) > 0
# -

# The voltage and the states of charge through the schedule. Charging raises
# the voltage above the difference of the formal potentials of the couples,
# and discharging lowers it below, by the overpotentials and the Ohmic
# losses of the cell. The states of charge of the two sides follow each
# other, since each electron that one couple takes the other gives:

fig, (upper, lower) = plt.subplots(2, 1, sharex=True)
for mode in ("charge", "rest", "discharge"):
    upper.plot(times[modes == mode], voltage[modes == mode], ".", label=mode)
    lower.plot(times[modes == mode], soc["negative"][modes == mode], ".", label=mode)
lower.plot(times, soc["positive"], "k--", label="positive side")
upper.set_ylabel("Voltage (V)")
lower.set_ylabel("State of charge")
lower.set_xlabel("Time (s)")
upper.legend()
lower.legend()
plt.show()

# The local state of charge of the negative electrode at the end of the
# charge, gathered from all processes with
# {py:func}`gather_grid <sumfem.plotting.gather_grid>`. The electrolyte
# flows up from the inlet at the bottom, so it is charged more towards the
# outlet. The electrode is drawn in millimetres, widened to show its
# thickness:

grid = gather_grid(local_charge)
if MPI.COMM_WORLD.rank == 0:
    grid.points[:, 0] *= 1e4
    grid.points[:, 1] *= 1e3
    plotter = pyvista.Plotter(window_size=[480, 640])
    plotter.add_mesh(grid, scalars="soc", scalar_bar_args={"title": "SOC"})
    plotter.view_xy()
    plotter.show()

# ## References
#
# ```{footbibliography}
# ```
