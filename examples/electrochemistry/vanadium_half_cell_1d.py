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

# # A vanadium half-cell in one dimension
#
# This demo charges, rests and discharges the negative half-cell of a
# vanadium redox flow battery across its porous electrode, as the 1D example
# of [RFBniCS](https://github.com/Ah-Aghabarari/RFBniCS). It shows how to:
#
# - Build a half-cell of any redox couple and electrolyte, negative or
#   positive, with
#   {py:class}`RedoxHalfCell <sumfem.electrochemistry.RedoxHalfCell>`
# - Run it through a galvanostatic schedule of charge, rest and discharge,
#   with a {py:class}`Simulation <sumfem.solvers.Simulation>`
# - Follow the voltage and the state of charge of the half-cell
#
# ## The model
#
# The porous electrode holds the solid, of potential $\phi_s$, and in its pores
# of porosity $\varepsilon$ the electrolyte, of potential $\phi_l$, with the
# ions V²⁺, V³⁺, H⁺ and HSO₄⁻ of concentrations $c_k$, and sulfate SO₄²⁻ by
# electroneutrality. At the fibres of specific area $a$, the reaction
# $\mathrm{V^{2+} \rightleftharpoons V^{3+} + e^-}$ carries the current
# density $i$ from the electrolyte to the solid. Charge is conserved in both
# phases, and each ion is stored, transported and produced:
#
# $$
# \begin{aligned}
#   -\nabla \cdot (\sigma_{\text{eff}} \nabla \phi_s) + a i &= 0, \\
#   \nabla \cdot i_l - a i &= 0, \\
#   \varepsilon \frac{\partial c_k}{\partial t} + \nabla \cdot N_k &= s_k,
#   \\
#   \sum_k z_k c_k &= 0,
# \end{aligned}
# $$ (half-cell)
#
# with the effective conductivity of the solid $\sigma_{\text{eff}}$, the
# current density of the electrolyte $i_l$ and the Nernst-Planck fluxes
# $N_k$, whose diffusivities are corrected for the porosity. The reaction
# consumes V²⁺ and produces V³⁺, $s_{\mathrm{V^{2+}}} = -a i / F$ and
# $s_{\mathrm{V^{3+}}} = a i / F$, at the Butler-Volmer current density
#
# $$
# \begin{aligned}
#   i &= F k_0 \left(
#   e^{\alpha_a F \eta / (R T)} c_{\mathrm{V^{2+}}}
#   - e^{-\alpha_c F \eta / (R T)} c_{\mathrm{V^{3+}}} \right), \\
#   \eta &= \phi_s - \phi_l - E_0,
# \end{aligned}
# $$ (half-cell-kinetics)
#
# with the formal potential $E_0$. The current density $j$ of the cell,
# positive on discharge, enters the solid at the current collector and leaves
# the electrolyte at the membrane, where the electrolyte has the
# concentrations of the reservoir. The voltage of the half-cell is the mean
# solid potential at the current collector less the mean electrolyte
# potential at the membrane. The parameters are those of the RFBniCS
# examples.
#
# We import the relevant packages:

# + tags=["hide-input"]
from mpi4py import MPI

import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import sympy.physics.units as syu

from sumfem import discrete, solvers, symbolic
from sumfem.electrochemistry import Ion, PorousElectrode, RedoxHalfCell

# -

# ## The electrode
#
# Across the electrode, from the current collector at $x = 0$ to the
# membrane at $x = W$, the cell is an interval. Its ends are located with a
# tolerance relative to the width, see {ref}`units-length-scale`:

# +
width, num_cells = 2.5e-3, 250
dtype = dolfinx.default_real_type
interval = dolfinx.mesh.create_interval(
    MPI.COMM_WORLD, num_cells, [0.0, width], dtype=dtype
)
tolerance = 100 * float(np.finfo(dtype).eps) * width
marked = discrete.MeshData.from_locators(
    interval,
    {
        0: {
            "CurrentCollector": lambda x: np.isclose(x[0], 0.0, atol=tolerance),
            "Membrane": lambda x: np.isclose(x[0], width, atol=tolerance),
        }
    },
    boundary=True,
)
domain = symbolic.Domain.from_mesh("cell", interval, length_scale=width)
# -

# ## The half-cell
#
# The ions of the electrolyte are given with their charges and diffusivities,
# in their units, which sumfem converts to SI, see {ref}`units-checking`. The
# vanadium ions take the positive part of their concentrations in the
# kinetics and the conductivity, as they may be depleted. The concentrations
# of the reservoir name the ions that are transported, so sulfate, which has
# none, follows from electroneutrality:

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

# The porous electrode is a carbon felt, a mat of carbon fibres, with its
# porosity, the conductivity of its fibres, their specific area and diameter,
# see {py:class}`PorousElectrode <sumfem.electrochemistry.PorousElectrode>`.
# The constants of the permeability of its pores and of the inertial drag
# matter only with flow, and this demo has none:

electrode = PorousElectrode(
    "electrode",
    domain,
    porosity=0.94,
    conductivity=3402.1 * syu.siemens / syu.m,
    specific_area=2.3e4 / syu.m,
    fiber_diameter=0.5 * syu.micrometer,
    kozeny_carman_constant=5.55,
    pore_length=100.0 * syu.micrometer,
)

# {py:class}`RedoxHalfCell <sumfem.electrochemistry.RedoxHalfCell>` sums the
# models of {eq}`half-cell` for a couple of these ions, with its standard rate
# constant $k_0$ and formal potential $E_0$. Discharge oxidizes V²⁺ in the
# negative half-cell; the positive half-cell of the battery, with the couple
# VO²⁺/VO₂⁺, reduces its couple on discharge, which reverses the current. The
# concentrations of the reservoir are Dirichlet conditions at the membrane:

cell = RedoxHalfCell(
    domain,
    ions,
    reservoir,
    reduced="V2",
    oxidized="V3",
    electrode=electrode,
    oxidation_on_discharge=True,
    rate_constant=2.5e-7 * syu.m / syu.second,
    formal_potential=-0.36 * syu.volt,
)
inlet = cell.inlet_conditions("Membrane")

# The discretization bounds the quadrature degree of the nonlinear kinetics,
# as RFBniCS does. The initial state is at rest: the concentrations of the
# reservoir, the solid at zero and the electrolyte at minus the formal
# potential:

disc = discrete.discretize(
    [cell.model, *inlet],
    discrete.DomainMaps({domain: marked}),
    max_quadrature_degree=4,
)
cell.initialize(disc)

# ## The schedule
#
# The cell is charged, rested and discharged, each with time steps that
# start short, while the cell responds fast, and then grow. Each operation
# gives its mode, the magnitude of the current density in A/m², and its time
# steps, as pairs of a number of steps and their size in seconds. The
# current of the cell is positive on discharge, so the mode gives its sign:

# +
schedule = (
    ("charge", 0.5, ((20, 0.1), (10, 1.0), (10, 10.0))),
    ("rest", 0.0, ((5, 0.1), (10, 1.0), (20, 2.0))),
    ("discharge", 0.5, ((20, 0.1), (10, 1.0), (10, 10.0))),
)
sign = {"charge": -1.0, "rest": 0.0, "discharge": 1.0}
# -

# A {py:class}`Simulation <sumfem.solvers.Simulation>` takes the steps with
# backward Euler, checks the units, and solves every time step in
# dimensionless form, see {ref}`units-scaling`. The steps of the schedule
# change in size, so the reference time of the time derivatives is the
# largest of them: a shorter step only makes Newton's absolute tolerance
# stricter, see {doc}`../cooling_plate`:

# +
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

# ## Charge, rest and discharge
#
# Each operation sets the current of the cell, and each part of it the size
# of the time step, before the simulation takes its steps. After every step,
# a callback records the time, the mode, the voltage of the half-cell and its
# state of charge, the share of V²⁺ in the vanadium of the electrode:

# +
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
times = np.array([r[0] for r in records])
modes = np.array([r[1] for r in records])
voltage = np.array([r[2] for r in records])
charge = np.array([r[3] for r in records])
# -

# Charging reduces V³⁺ to V²⁺, so the state of charge grows, and discharging
# turns it back. At rest only the exchange with the reservoir at the membrane
# changes the electrode, slowly. The voltage is lower on charge than on
# discharge, by the overpotentials that drive the reaction either way:

# +
steps = {mode: np.diff(charge[modes == mode]) for mode in ("charge", "discharge")}
print(f"State of charge after charge: {charge[modes == 'charge'][-1]:.6f}")
print(f"State of charge at the end: {charge[-1]:.6f}")
assert np.all(steps["charge"] > 0) and np.all(steps["discharge"] < 0)
rest = charge[modes == "rest"]
assert np.abs(np.diff(rest)).max() < np.abs(steps["charge"]).min()
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
