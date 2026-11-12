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
# of [RFBniCS](https://github.com/Ah-Aghabarari/RFBniCS)
# {footcite}`Aghabarari2026`. It shows how to:
#
# - Build a half-cell of any redox couple and electrolyte, negative or
#   positive, with
#   {py:class}`RedoxHalfCell <sumfem.electrochemistry.RedoxHalfCell>`
# - Run it through a galvanostatic schedule of charge, rest and discharge,
#   with a {py:class}`Simulation <sumfem.solvers.Simulation>`
# - Follow the voltage and the state of charge of the half-cell
#
# ## The mathematical model
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
# potential at the membrane. The model is the macro-homogeneous
# porous-electrode model of RFBniCS {footcite}`Aghabarari2026` (Section II),
# after {footcite}`NewmanTiedemann1975`, and its parameters those of Table I
# there, several from {footcite}`MunozPerales2022`.
#
# We import the relevant packages:

# + tags=["hide-input"]
from mpi4py import MPI

import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import sympy.physics.units as syu

from sumfem import discrete, solvers, symbolic
from sumfem.electrochemistry import Ion, PorousElectrode, RedoxCouple, RedoxHalfCell

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

# The porous electrode, which fills the interval, is a carbon felt, a mat of
# carbon fibres, with its porosity, the conductivity of its fibres, their
# specific area and diameter, see
# {py:class}`PorousElectrode <sumfem.electrochemistry.PorousElectrode>`.
# The fibres of commercial carbon felts, papers and cloths share a diameter
# of several micrometres {footcite}`Barzegari2026`; RFBniCS
# {footcite}`Aghabarari2026` uses a diameter far below that range, kept in a
# comment. The diameter and the
# constants of the permeability of the pores and of the inertial drag matter
# only with flow, and this demo has none:

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


# ## The half-cell
#
# The ions of the electrolyte are given with their charges and diffusivities,
# in their units, which sumfem converts to SI, see {ref}`units-checking`. The
# kinetics take the positive part of the vanadium concentrations, as they may
# be depleted. The concentrations
# of the reservoir name the ions that are transported, each with a field and
# the equations of {eq}`half-cell`. Sulfate has none: electroneutrality,
# the last equation of {eq}`half-cell`, with the charges of the ions,
#
# $$
# \begin{aligned}
#   2 c_{\mathrm{V^{2+}}} + 3 c_{\mathrm{V^{3+}}} + c_{\mathrm{H^+}}
#   - c_{\mathrm{HSO_4^-}} - 2 c_{\mathrm{SO_4^{2-}}} &= 0,
# \end{aligned}
# $$ (electroneutrality)
#
# is solved for its concentration, which is then no unknown, but an
# expression of the others:
#
# $$
# \begin{aligned}
#   c_{\mathrm{SO_4^{2-}}} &= \frac{1}{2} \left( 2 c_{\mathrm{V^{2+}}}
#   + 3 c_{\mathrm{V^{3+}}} + c_{\mathrm{H^+}}
#   - c_{\mathrm{HSO_4^-}} \right).
# \end{aligned}
# $$ (sulfate)
#
# It enters the conductivity and the diffusion current of the electrolyte
# like the concentrations of the other ions. The half-cell checks that the
# reservoir lacks exactly one ion, and that this ion is charged:

# +
diffusivity = syu.m**2 / syu.second
concentration = syu.mol / syu.m**3
ions = (
    Ion("V2", 2, 8.10e-12 * diffusivity),
    Ion("V3", 3, 1.65e-11 * diffusivity),
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

# A {py:class}`RedoxCouple <sumfem.electrochemistry.RedoxCouple>` gives the
# reaction $\mathrm{V^{2+} \rightleftharpoons V^{3+} + e^-}$ of
# {eq}`half-cell-kinetics`: the ion `reduced`, V²⁺, the ion `oxidized`, V³⁺,
# and the number of `electrons` the reaction transfers, which the charges of
# the ions must match, with the standard rate constant $k_0$ and the formal
# potential $E_0$ of its kinetics. Discharge oxidizes V²⁺ in this negative
# half-cell; the positive half-cell of the battery, with the couple
# VO²⁺/VO₂⁺, reduces its couple on discharge, which reverses the current:

couple = RedoxCouple(
    reduced="V2",
    oxidized="V3",
    electrons=1,
    rate_constant=2.5e-7 * syu.m / syu.second,
    formal_potential=-0.36 * syu.volt,
    oxidation_on_discharge=True,
)

# {py:class}`RedoxHalfCell <sumfem.electrochemistry.RedoxHalfCell>` sums the
# models of {eq}`half-cell` for the couple. The other ions take no part in
# the reaction; they carry current and keep the electrolyte neutral. The
# concentrations of the reservoir are Dirichlet conditions at the membrane:

cell = RedoxHalfCell(domain, ions, reservoir, couple, electrode)
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

# ## The charging and discharging schedule
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

# ## Dimensionless groups
#
# The model is semi-discrete, so its units are checked on its system alone,
# see {ref}`units-checking`, with the length of the whole schedule as the
# reference time of the time derivatives. With the width of the electrode,
# the scales of the concentration of V²⁺ and of the solid potential,
# Faraday's constant, the temperature and the diffusivity of V²⁺ as the
# repeating quantities, {py:func}`pi_groups <sumfem.symbolic.pi_groups>`
# gives one group for every other quantity of the model:

# +
duration = sum(count * dt for *_, steps in schedule for count, dt in steps)
semi_discrete = symbolic.build_system(cell.model, check_units=True, time_scale=duration)
electrolyte = cell.electrolyte
repeating = [
    domain,
    cell.concentrations["V2"],
    cell.phi_s,
    electrolyte.F,
    electrolyte.T,
    electrolyte.diffusivity("V2"),
]
groups = symbolic.pi_groups(semi_discrete, time_scale=duration, repeating=repeating)
for group in groups:
    print(f"{group.expression} = {group.value:.3g}")


def value_of(name: str) -> float:
    """The value of the one group of the quantity of a name."""
    (value,) = [g.value for g in groups if name in str(g.expression)]
    return value


# -

# Most groups are ratios of like quantities: of the diffusivities and the
# concentrations of the ions, the porosity and the transfer coefficients.
# Four describe the behaviour of the cell:
#
# - The group of the reference time is the [mass Fourier
#   number](https://dimless.org/docs/numbers/mass-fourier/) $D t / W^2$ of
#   V²⁺ over the whole schedule. It is far below one: the ions diffuse only
#   a small part of the way across the electrode, so the electrode exchanges
#   little with the reservoir at the membrane.
# - The groups of the rate constant and the specific area multiply to the
#   Damköhler number $a k_0 W^2 / D$ {footcite}`Fogler2020`, the rate of the
#   reaction over that of diffusion across the electrode. Its square root is
#   the Thiele modulus {footcite}`Thiele1939`, the width of the electrode over
#   the depth a reactant supplied from the membrane would reach before it
#   reacts. Both are far above one.
# - The group of the current of the cell is the current over the limiting
#   current of diffusion across the electrode, $F D c / W$
#   {footcite}`NewmanBalsara2021`. Above one, the cell runs on the vanadium
#   stored in its pores, not on what diffuses in from the reservoir. The
#   current has no default, so the analysis takes a unit current density,
#   and the group is scaled with the current of the schedule.
# - The groups of the conductivity of the solid and of the gas constant
#   multiply to $\sigma R T / (F^2 D c)$, the conductivity of the solid over
#   that V²⁺ alone would give the electrolyte.
#
# Neither the Damköhler number nor the Thiele modulus is yet on
# [dimless.org](https://dimless.org/docs/numbers/):

# +
fourier = value_of("t_ref")
damkohler = value_of("rate_constant") * value_of("specific_area")
largest_current = max(current for _, current, _ in schedule)
current_ratio = value_of("cell.current") * largest_current
conductivity_ratio = value_of("electrode.conductivity") * value_of("electrolyte.R")
print(f"Mass Fourier number of the schedule: {fourier:.3g}")
print(f"Damkoehler number: {damkohler:.3g}, Thiele modulus: {np.sqrt(damkohler):.3g}")
print(f"Current over the limiting current of diffusion: {current_ratio:.3g}")
print(f"Conductivity of the solid over that of V2+: {conductivity_ratio:.3g}")
assert fourier < 1e-2 and damkohler > 1e2 and current_ratio > 1
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
# discharge, by the overpotentials that drive the reaction either way.
# Sulfate, which follows from electroneutrality, stays positive, see
# {py:meth}`minimum_balanced_concentration
# <sumfem.electrochemistry.RedoxHalfCell.minimum_balanced_concentration>`:

# +
steps = {mode: np.diff(charge[modes == mode]) for mode in ("charge", "discharge")}
print(f"State of charge after charge: {charge[modes == 'charge'][-1]:.6f}")
print(f"State of charge at the end: {charge[-1]:.6f}")
assert np.all(steps["charge"] > 0) and np.all(steps["discharge"] < 0)
rest = charge[modes == "rest"]
assert np.abs(np.diff(rest)).max() < np.abs(steps["charge"]).min()
assert voltage[modes == "charge"].mean() < voltage[modes == "discharge"].mean()
assert np.isfinite(voltage).all()
assert cell.minimum_balanced_concentration(disc) > 0
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

# ## References
#
# ```{footbibliography}
# ```
