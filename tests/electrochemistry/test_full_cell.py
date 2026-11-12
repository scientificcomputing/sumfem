"""The full cell of Shah et al. (2008), corrected, in one and two dimensions."""

import dataclasses

from mpi4py import MPI

import basix.ufl
import dolfinx.mesh
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.electrochemistry import (
    DarcyFlow,
    FlowSpec,
    FullCell,
    HalfCellSpec,
    Ion,
    MembraneSpec,
    RedoxCouple,
)
from sumfem.models import Accumulation, Diffusion
from sumfem.solvers import BDF, Simulation
from sumfem.symbolic import (
    Clock,
    Domain,
    Field,
    Parameter,
    build_system,
    value_at_defaults,
)

DIFFUSIVITY = syu.m**2 / syu.second
CONCENTRATION = syu.mol / syu.m**3

LAYERS = (
    ("NegativeCollector", 6e-3, 6),
    ("NegativeElectrode", 4e-3, 40),
    ("Membrane", 180e-6, 6),
    ("PositiveElectrode", 4e-3, 40),
    ("PositiveCollector", 6e-3, 6),
)
"""The layers of the cell of Shah et al., Table 3: name, thickness in m and
number of cells."""

ACID = {
    "H": 1200.0 * CONCENTRATION,
    "HSO4": 1200.0 * CONCENTRATION,
    "H2O": 4.2e4 * CONCENTRATION,
}
"""The acid and the water of both electrolytes, Table 2, with the water
corrected from 4.2e3 to 4.2e4 mol/m^3."""


def create_ions(reduced, oxidized, charges, diffusivity):
    """The ions of a side: its couple, protons, bisulfate, water and sulfate."""
    return (
        Ion(reduced, charges[0], diffusivity * DIFFUSIVITY),
        Ion(oxidized, charges[1], diffusivity * DIFFUSIVITY),
        Ion("H", 1, 9.312e-9 * DIFFUSIVITY),
        Ion("HSO4", -1, 1.33e-9 * DIFFUSIVITY),
        Ion("H2O", 0, 2.3e-9 * DIFFUSIVITY),
        Ion("SO4", -2, 1.065e-9 * DIFFUSIVITY),
    )


ELECTRODE = dict(
    porosity=0.68,
    conductivity=500.0,
    specific_area=2e6,
    fiber_distance=10 * syu.micrometer,
)
"""The carbon felt of both sides, Tables 3 and 5."""

NEGATIVE = HalfCellSpec(
    ions=create_ions("V2", "V3", (2, 3), 2.4e-10),
    reservoir={"V2": 27.0 * CONCENTRATION, "V3": 1053.0 * CONCENTRATION, **ACID},
    couple=RedoxCouple("V2", "V3", 0.68 * 1.75e-7, -0.255, True),
    **ELECTRODE,
)
"""The negative side at 5% state of charge."""

POSITIVE = HalfCellSpec(
    ions=create_ions("VO", "VO2", (2, 1), 3.9e-10),
    reservoir={"VO": 1053.0 * CONCENTRATION, "VO2": 27.0 * CONCENTRATION, **ACID},
    couple=RedoxCouple(
        "VO",
        "VO2",
        0.68 * 3e-9,
        1.004,
        False,
        reactants={"VO": 1, "H2O": 1},
        products={"VO2": 1, "H": 2},
    ),
    **ELECTRODE,
)
"""The positive side at 5% state of charge."""

MEMBRANE = MembraneSpec(
    fixed_charge=1200.0, proton_diffusivity=1.4e-9, water_diffusivity=5.75e-10
)
"""The membrane, Tables 4 and 5."""


def create_cell_mesh(comm=MPI.COMM_WORLD) -> MeshData:
    """The layers of the cell on an interval, with their markers."""
    edges = np.cumsum([0.0, *(width for _, width, _ in LAYERS)])
    if comm.rank == 0:
        x = np.unique(
            np.concatenate(
                [
                    np.linspace(a, b, n + 1)
                    for (a, b), (*_, n) in zip(zip(edges, edges[1:]), LAYERS)
                ]
            )
        ).reshape(-1, 1)
        cells = np.stack([np.arange(len(x) - 1), np.arange(1, len(x))], axis=1)
    else:
        x, cells = np.empty((0, 1)), np.empty((0, 2), dtype=np.int64)
    element = basix.ufl.element("Lagrange", "interval", 1, shape=(1,))
    mesh = dolfinx.mesh.create_mesh(comm, cells, element, x)
    tolerance = 1e-9

    def between(a, b):
        return lambda x: (x[0] >= a - tolerance) & (x[0] <= b + tolerance)

    def at(a):
        return lambda x: np.isclose(x[0], a, atol=tolerance)

    layers = {name: between(a, b) for (name, *_), a, b in zip(LAYERS, edges, edges[1:])}
    facets = {
        "NegativeTerminal": at(edges[0]),
        "NegativeFace": at(edges[2]),
        "PositiveFace": at(edges[3]),
        "PositiveTerminal": at(edges[5]),
    }
    return FullCell.mesh_data(MeshData.from_locators(mesh, {1: layers, 0: facets}))


def create_full_cell():
    """The cell of Shah et al. on its domain."""
    domain = Domain("cell", "interval", length_scale=4e-3)
    return FullCell(domain, NEGATIVE, POSITIVE, membrane=MEMBRANE), domain


def test_the_full_cell_has_consistent_units_and_shared_fields():
    cell, _ = create_full_cell()
    system = build_system(cell.model, check_units=True)
    names = sorted(f.name for f in system.unknowns)
    assert names.count("phi_l") == 1 and names.count("c_H2O") == 1
    assert {"negative_phi_s", "positive_phi_s", "negative_c_H", "positive_c_H"} <= {
        *names
    }
    assert "cell.collector_conductivity" in system.parameters
    models = [m.name for m in cell.model.components]
    assert {"cell_negative_protons", "cell_positive_protons"} <= {*models}
    assert not any("membrane_current" in m for m in models)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: {k: v for k, v in r.items() if k != "H"}, "transport protons"),
        (lambda r: {k: v for k, v in r.items() if k != "H2O"}, "both sides"),
    ],
)
def test_the_sides_must_share_protons_and_water(change, message):
    domain = Domain("cell", "interval")
    ions = tuple(
        i for i in POSITIVE.ions if i.name in (*change(POSITIVE.reservoir), "SO4")
    )
    with pytest.raises(ValueError, match=message):
        positive = HalfCellSpec(
            ions=ions,
            reservoir=change(POSITIVE.reservoir),
            couple=POSITIVE.couple,
            **ELECTRODE,
        )
        FullCell(domain, NEGATIVE, positive, membrane=MEMBRANE)


def simulate(current, steps, dt=1.0):
    """Run the cell at a current density from rest, with what the tests need."""
    cell, domain = create_full_cell()
    marked = create_cell_mesh()
    disc = discretize(
        [cell.model, *cell.conditions()],
        DomainMaps({domain: marked}),
        max_quadrature_degree=4,
    )
    cell.initialize(disc)
    clock = Clock(domain, dt=dt)
    simulation = Simulation(
        disc,
        cell.model,
        cell.conditions(),
        clock=clock,
        scheme=BDF(1),
        petsc_options={"snes_linesearch_type": "bt", "snes_max_it": 50},
        check_units=True,
        factorize_equations=True,
        time_scale=dt,
    )
    disc.set({cell.current: current})
    return cell, disc, simulation


def inventories(cell, disc):
    """The amounts per unit area of what the cell conserves, by name."""
    amounts = {}
    for side, half in cell.half_cells.items():
        e = half.electrolyte
        dx = half.concentrations["H"].domain.dx()
        storage = half.electrode.porosity
        for ion in e.ions:
            if ion == "H2O":
                continue
            amounts[f"{side}_{ion}"] = disc.assemble_scalar(
                storage * e.concentration(ion) * dx
            )
    water = cell.water
    electrolyte = cell.subdomain("Electrolyte")
    amounts["water"] = sum(
        disc.assemble_scalar(weight * water * electrolyte.dx(marker))
        for weight, marker in (
            (cell.half_cells["negative"].electrode.porosity, "NegativeElectrode"),
            (1.0, "Membrane"),
            (cell.half_cells["positive"].electrode.porosity, "PositiveElectrode"),
        )
    )
    return amounts


def test_a_cell_at_rest_stays_at_rest():
    """Started at the equilibrium potentials of its couples, without current,
    the cell does not change, and its voltage is the difference of those
    potentials."""
    cell, disc, simulation = simulate(0.0, steps=2)
    voltage = cell.voltage(disc)
    before = inventories(cell, disc)
    simulation.run(num_steps=2)
    assert cell.voltage(disc) == pytest.approx(voltage, abs=1e-9)
    after = inventories(cell, disc)
    for name, amount in before.items():
        assert after[name] == pytest.approx(amount, rel=1e-10), name
    RT_F = 8.314462618 * 298.15 / 96485.33212
    expected = 1.004 + 0.255 + 2 * RT_F * np.log(27.0 / 1053.0)
    assert voltage == pytest.approx(expected, rel=1e-6)


def test_a_discharging_cell_conserves_its_species():
    """On discharge, V2+ is oxidized and VO2+ reduced, each by the charge
    ``Q = j t`` per unit area; the protons carry the current through the
    membrane, so sulfate and bisulfate stay on their sides, each side loses
    ``Q / F`` protons, and the water gains what VO2+ loses.

    In one dimension the balances hold to the tolerance of Newton's method:
    the electrolyte potential's test functions of an electrode and its face
    sum to one in the electrode, so the current of the membrane cell at the
    face is the electrode's total reaction current, which the protons carry.
    """
    current, steps, dt = 100.0, 5, 1.0
    cell, disc, simulation = simulate(current, steps, dt)
    before = inventories(cell, disc)
    simulation.run(num_steps=steps)
    after = inventories(cell, disc)
    change = {name: after[name] - before[name] for name in before}
    moles = current * steps * dt / 96485.33212

    for side, reduced, oxidized, sign in (
        ("negative", "V2", "V3", 1.0),
        ("positive", "VO", "VO2", -1.0),
    ):
        # Vanadium stays on its side, and the couple turns over Q / F
        assert change[f"{side}_{reduced}"] == pytest.approx(-sign * moles, rel=1e-8)
        assert change[f"{side}_{oxidized}"] == pytest.approx(sign * moles, rel=1e-8)
        # Neither anion crosses the membrane, and each side loses Q / F protons
        assert change[f"{side}_HSO4"] == pytest.approx(0.0, abs=1e-8 * moles)
        assert change[f"{side}_SO4"] == pytest.approx(0.0, abs=1e-8 * moles)
        assert change[f"{side}_H"] == pytest.approx(-moles, rel=1e-8)
    # One water per VO2+ reduced
    assert change["water"] == pytest.approx(moles, rel=1e-8)
    assert cell.voltage(disc) < 1.004 + 0.255


def test_the_protons_cross_a_face_from_their_own_side():
    """The proton flux is a condition of an electrode's protons on its face,
    with the membrane on the other side; a face seen from the wrong side, or
    a measure that knows no sides, is rejected."""
    cell, domain = create_full_cell()
    negative = cell.half_cells["negative"].concentrations["H"]
    electrode = cell.subdomain("NegativeElectrode")
    membrane = cell.subdomain("Membrane")
    proton_flux = cell.membrane.proton_flux
    with pytest.raises(ValueError, match="membrane on its '-' side"):
        proton_flux(
            "wrong", negative, domain.interface(membrane, electrode, "NegativeFace")
        )
    with pytest.raises(ValueError, match="'\\+' side"):
        proton_flux(
            "wrong",
            negative,
            domain.interface(
                cell.subdomain("PositiveElectrode"), membrane, "PositiveFace"
            ),
        )
    with pytest.raises(ValueError, match="Domain.interface"):
        proton_flux("wrong", negative, domain.dS("NegativeFace"))


HEIGHT = 1e-2
"""The height of the 2D cell, from the inlets to the outlets, in m."""

FLOW = FlowSpec(inlet_velocity=2.5e-3, tank_ratio=8.0)
"""Shah et al.'s 1 mL/s through an electrode of 4 mm by 10 cm, Table 2,
and tanks of about the paper's 250 mL each, Table 3."""

FLOWING = dict(fiber_diameter=10 * syu.micrometer, kozeny_carman_constant=5.55)
"""What the flow needs of the felt of both sides, Tables 3 and 5."""

MEMBRANE_FLOW = MembraneSpec(
    fixed_charge=1200.0,
    proton_diffusivity=1.4e-9,
    water_diffusivity=5.75e-10,
    electrokinetic_permeability=1.13e-19,
    hydraulic_permeability=1.58e-18,
    viscosity=1e-3,
)
"""The membrane with the flow of its water, Table 5."""


def create_flowing_mesh(
    comm=MPI.COMM_WORLD, columns=(2, 6, 2, 6, 2), rows=4
) -> MeshData:
    """The layers of the cell on a rectangle of the cell's height, with the
    inlets of the electrodes at the bottom and their outlets at the top."""
    edges = np.cumsum([0.0, *(width for _, width, _ in LAYERS)])
    xs = np.unique(
        np.concatenate(
            [np.linspace(a, b, n + 1) for a, b, n in zip(edges, edges[1:], columns)]
        )
    )
    mesh = dolfinx.mesh.create_rectangle(
        comm, [[0.0, 0.0], [1.0, HEIGHT]], [len(xs) - 1, rows]
    )
    # Move the vertices of the uniform grid onto the edges of the layers
    index = np.rint(mesh.geometry.x[:, 0] * (len(xs) - 1)).astype(int)
    mesh.geometry.x[:, 0] = xs[index]
    tolerance = 1e-9

    def between(a, b):
        return lambda x: (x[0] >= a - tolerance) & (x[0] <= b + tolerance)

    def end(a, b, y):
        return lambda x: between(a, b)(x) & np.isclose(x[1], y, atol=tolerance)

    layers = {name: between(a, b) for (name, *_), a, b in zip(LAYERS, edges, edges[1:])}
    facets = {
        "NegativeTerminal": lambda x: np.isclose(x[0], edges[0], atol=tolerance),
        "NegativeFace": lambda x: np.isclose(x[0], edges[2], atol=tolerance),
        "PositiveFace": lambda x: np.isclose(x[0], edges[3], atol=tolerance),
        "PositiveTerminal": lambda x: np.isclose(x[0], edges[5], atol=tolerance),
        "NegativeInlet": end(edges[1], edges[2], 0.0),
        "NegativeOutlet": end(edges[1], edges[2], HEIGHT),
        "PositiveInlet": end(edges[3], edges[4], 0.0),
        "PositiveOutlet": end(edges[3], edges[4], HEIGHT),
    }
    marked = MeshData.from_locators(mesh, {2: layers, 1: facets})
    return FullCell.mesh_data(marked)


def create_flowing_cell(current, dt=1.0, flow=FLOW):
    """The cell of Shah et al. in 2D, with flow and tanks, from rest."""
    domain = Domain("cell", "triangle", length_scale=HEIGHT)
    sides = [dataclasses.replace(s, **FLOWING) for s in (NEGATIVE, POSITIVE)]
    cell = FullCell(domain, *sides, membrane=MEMBRANE_FLOW, flow=flow)
    disc = discretize(
        [cell.model, *cell.conditions()],
        DomainMaps({domain: create_flowing_mesh()}),
        max_quadrature_degree=4,
    )
    cell.initialize(disc)
    simulation = Simulation(
        disc,
        cell.model,
        cell.conditions(),
        clock=Clock(domain, dt=dt),
        scheme=BDF(1),
        petsc_options={"snes_linesearch_type": "bt", "snes_max_it": 50},
        check_units=True,
        factorize_equations=True,
        time_scale=dt,
    )
    disc.set({cell.current: current})
    return cell, disc, simulation, domain


def totals(cell, disc):
    """The amounts per unit depth in the cell and its tanks, by name."""
    amounts = inventories(cell, disc)
    water = 0.0
    for side, half in cell.half_cells.items():
        volume = cell.tank_volume(disc, side)
        for ion in half.concentrations:
            tank = volume * float(disc.value(half.inlet[ion]))
            if ion == "H2O":
                water += tank
            else:
                amounts[f"{side}_{ion}"] += tank
    amounts["water"] += water
    return amounts


def test_a_cell_at_rest_with_flow_stays_at_rest():
    """The tanks hold the electrolyte the cell starts with, so without current
    the flow changes nothing, and the pressure falls through each electrode
    by the drop ``mu v H / K`` of a uniform flow."""
    cell, disc, simulation, domain = create_flowing_cell(0.0)
    before, voltage = totals(cell, disc), cell.voltage(disc)
    simulation.run(num_steps=2, callback=lambda t, step: cell.update_tanks(disc, 1.0))
    after = totals(cell, disc)
    for name, amount in before.items():
        assert after[name] == pytest.approx(amount, rel=1e-8), name
    assert cell.voltage(disc) == pytest.approx(voltage, abs=1e-8)

    electrode = cell.half_cells["negative"].electrode
    viscosity = cell.half_cells["negative"].electrolyte.viscosity
    drop = value_at_defaults(viscosity * 2.5e-3 * HEIGHT / electrode.permeability())
    inlet = cell.subdomain("NegativeElectrode").ds("NegativeInlet")
    mean = disc.assemble_scalar(cell.pressure * inlet) / disc.assemble_scalar(
        1.0 * inlet
    )
    assert mean == pytest.approx(drop, rel=1e-6)


def test_a_discharging_cell_and_its_tanks_conserve_every_ion():
    """With flow and tanks, the cell and its tanks together keep the vanadium
    and the bisulfate of each side, turn over ``Q / F`` of each couple, with
    ``Q`` the charge through the terminals of the cell's height, and gain the
    water that the positive couple makes.

    Each side loses the protons that carry the current through the membrane,
    ``Q / F``, only to first order in the mesh size in two dimensions: the
    current of the membrane's cells at a face differs from the current the
    electrode's reactions send to it by the discretization error. It halves
    as the mesh is refined.
    """
    current, steps, dt = 100.0, 3, 1.0
    cell, disc, simulation, _ = create_flowing_cell(current, dt)
    before = totals(cell, disc)
    simulation.run(
        num_steps=steps, callback=lambda t, step: cell.update_tanks(disc, dt)
    )
    after = totals(cell, disc)
    change = {name: after[name] - before[name] for name in before}
    moles = current * HEIGHT * steps * dt / 96485.33212
    for side, reduced, oxidized, sign in (
        ("negative", "V2", "V3", 1.0),
        ("positive", "VO", "VO2", -1.0),
    ):
        assert change[f"{side}_{reduced}"] == pytest.approx(-sign * moles, rel=1e-8)
        assert change[f"{side}_{oxidized}"] == pytest.approx(sign * moles, rel=1e-8)
        assert change[f"{side}_HSO4"] == pytest.approx(0.0, abs=1e-8 * moles)
        assert change[f"{side}_H"] == pytest.approx(-moles, rel=2e-2)
    assert change["water"] == pytest.approx(moles, rel=1e-8)


class StoredDarcyFlow(DarcyFlow):
    """Darcy flow with a storage of the pressure in the electrodes: a flow
    that changes in time, plugged into the cell in a few lines."""

    @property
    def models(self):
        storage = Accumulation(
            f"{self.setup.name}_storage",
            self.pressure,
            weight=Parameter(
                "storage.storativity",
                self.setup.domain,
                default=1e-3,
                unit=1 / syu.pascal,
            ),
            markers=("NegativeElectrode", "PositiveElectrode"),
        )
        return [*super().models, storage]


def test_any_flow_plugs_into_the_cell():
    """A flow of the cell's own making, here one that changes in time, takes
    the place of Darcy flow. Its storage keeps the pressure below the steady
    drop of Darcy flow over the steps, so its velocity is not divergence
    free; the cell and its tanks still conserve the vanadium and turn over
    ``Q / F`` of each couple, as the species balances hold the flux of
    whatever velocity the flow gives."""
    current, steps, dt = 100.0, 3, 1.0
    flow = dataclasses.replace(FLOW, model=StoredDarcyFlow)
    cell, disc, simulation, _ = create_flowing_cell(current, dt, flow=flow)
    assert isinstance(cell.flow, StoredDarcyFlow)
    assert "cell_storage" in [m.name for m in cell.model.components]
    before = totals(cell, disc)
    simulation.run(
        num_steps=steps, callback=lambda t, step: cell.update_tanks(disc, dt)
    )
    after = totals(cell, disc)
    moles = current * HEIGHT * steps * dt / 96485.33212
    for side, reduced, oxidized, sign in (
        ("negative", "V2", "V3", 1.0),
        ("positive", "VO", "VO2", -1.0),
    ):
        change = after[f"{side}_{oxidized}"] - before[f"{side}_{oxidized}"]
        assert change == pytest.approx(sign * moles, rel=1e-8)
        vanadium = (after[f"{side}_{reduced}"] + after[f"{side}_{oxidized}"]) - (
            before[f"{side}_{reduced}"] + before[f"{side}_{oxidized}"]
        )
        assert vanadium == pytest.approx(0.0, abs=1e-8 * moles)
    electrode = cell.half_cells["negative"].electrode
    viscosity = cell.half_cells["negative"].electrolyte.viscosity
    drop = value_at_defaults(viscosity * 2.5e-3 * HEIGHT / electrode.permeability())
    inlet = cell.subdomain("NegativeElectrode").ds("NegativeInlet")
    mean = disc.assemble_scalar(cell.pressure * inlet) / disc.assemble_scalar(
        1.0 * inlet
    )
    assert 0 < mean < 0.99 * drop


def test_inputs_can_be_fields_of_other_models():
    """The properties of the electrodes and the couples may be expressions of
    the fields of other models: here a temperature, which sets the
    conductivity of the fibres, an Arrhenius rate constant and a formal
    potential that changes with it, and a compression, which sets the
    porosity. Their parameters are then not created, the units are
    consistent, and a cell at rest starts and stays at the Nernst voltage of
    that temperature."""
    domain = Domain("cell", "interval", length_scale=4e-3)
    T = Field("T", domain, ("Lagrange", 1), unit=syu.kelvin, scale=300.0)
    strain = Field("strain", domain, ("Lagrange", 1))
    reference = 298.15 * syu.kelvin
    slope = 1e-3 * syu.volt / syu.kelvin
    activation = 4e3 * syu.kelvin
    sides = []
    for spec in (NEGATIVE, POSITIVE):
        couple = dataclasses.replace(
            spec.couple,
            rate_constant=spec.couple.rate_constant
            * ufl.exp(-activation * (1 / T - 1 / reference)),
            formal_potential=spec.couple.formal_potential + slope * (T - reference),
            reactants=None if spec is NEGATIVE else spec.couple.reactants,
            products=None if spec is NEGATIVE else spec.couple.products,
        )
        sides.append(
            dataclasses.replace(
                spec,
                couple=couple,
                porosity=0.68 * (1 - 0.1 * strain),
                conductivity=500.0 * syu.siemens / syu.m * (T / reference),
            )
        )
    cell = FullCell(domain, *sides, membrane=MEMBRANE)
    heat = Diffusion(
        "heat",
        T,
        Parameter("heat.k", domain, default=0.6, unit=syu.watt / (syu.m * syu.kelvin)),
    )
    compression = Diffusion("compression", strain, 1.0)
    model = cell.model + heat + compression
    system = build_system(model, check_units=True)
    assert "negative_electrode.porosity" not in system.parameters
    assert "positive.rate_constant" not in system.parameters
    assert {T, strain} <= set(system.unknowns)

    disc = discretize(
        [model, *cell.conditions()],
        DomainMaps({domain: create_cell_mesh()}),
        max_quadrature_degree=4,
    )
    temperature = 308.15
    disc.set({T: temperature, strain: 0.1})
    cell.initialize(disc)
    simulation = Simulation(
        disc,
        model,
        cell.conditions(),
        clock=Clock(domain, dt=1.0),
        scheme=BDF(1),
        petsc_options={"snes_linesearch_type": "bt", "snes_max_it": 50},
        check_units=True,
        factorize_equations=True,
        time_scale=1.0,
    )
    RT_F = 8.314462618 * temperature / 96485.33212
    shift = 2 * 1e-3 * (temperature - 298.15)
    expected = 1.004 + 0.255 + shift + 2 * RT_F * np.log(27.0 / 1053.0)
    assert cell.voltage(disc) == pytest.approx(expected, rel=1e-9)
    simulation.run(num_steps=1)
    assert cell.voltage(disc) == pytest.approx(expected, rel=1e-8)
