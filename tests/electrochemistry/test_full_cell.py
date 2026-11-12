"""The full cell of Shah et al. (2008), corrected, in one dimension."""

from mpi4py import MPI

import basix.ufl
import dolfinx.mesh
import numpy as np
import pytest
import sympy.physics.units as syu

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.electrochemistry import (
    FullCell,
    HalfCellSpec,
    Ion,
    MembraneSpec,
    RedoxCouple,
)
from sumfem.solvers import BDF, Simulation
from sumfem.symbolic import Clock, Domain, build_system

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


def ions(reduced, oxidized, charges, diffusivity):
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
    ions=ions("V2", "V3", (2, 3), 2.4e-10),
    reservoir={"V2": 27.0 * CONCENTRATION, "V3": 1053.0 * CONCENTRATION, **ACID},
    couple=RedoxCouple("V2", "V3", 0.68 * 1.75e-7, -0.255, True),
    **ELECTRODE,
)
"""The negative side at 5% state of charge."""

POSITIVE = HalfCellSpec(
    ions=ions("VO", "VO2", (2, 1), 3.9e-10),
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


def cell_mesh(comm=MPI.COMM_WORLD) -> MeshData:
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


def full_cell():
    """The cell of Shah et al. on its domain."""
    domain = Domain("cell", "interval", length_scale=4e-3)
    return FullCell(domain, NEGATIVE, POSITIVE, membrane=MEMBRANE), domain


def test_the_full_cell_has_consistent_units_and_shared_fields():
    cell, _ = full_cell()
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
    positive = HalfCellSpec(
        ions=ions,
        reservoir=change(POSITIVE.reservoir),
        couple=POSITIVE.couple,
        **ELECTRODE,
    )
    with pytest.raises(ValueError, match=message):
        FullCell(domain, NEGATIVE, positive, membrane=MEMBRANE)


def simulate(current, steps, dt=1.0):
    """Run the cell at a current density from rest, with what the tests need."""
    cell, domain = full_cell()
    marked = cell_mesh()
    disc = discretize(
        [cell.model, *cell.ground()],
        DomainMaps({domain: marked}),
        max_quadrature_degree=4,
    )
    cell.initialize(disc)
    clock = Clock(domain, dt=dt)
    simulation = Simulation(
        disc,
        cell.model,
        cell.ground(),
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
    cell, domain = full_cell()
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
