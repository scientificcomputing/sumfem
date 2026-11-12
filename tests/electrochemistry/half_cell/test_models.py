"""The RFBniCS models: units, structure, schedules and post-processing."""

import dataclasses

from mpi4py import MPI

import dolfinx.fem
import dolfinx.la
import numpy as np
import pytest
import sympy.physics.units as syu

from sumfem.discrete import DomainMaps, discretize
from sumfem.electrochemistry import (
    Ion,
    PistonVelocityButlerVolmer,
    PorousElectrode,
    RedoxCouple,
    RedoxHalfCell,
)
from sumfem.models import (
    BoundaryFlux,
    BrinkmanForchheimer,
    Darcy,
    GradDiv,
    NavierStokes,
)
from sumfem.post_processing import evaluate_over_line, function_range
from sumfem.solvers import BDF
from sumfem.symbolic import Domain, Field, Parameter, build_system


def fluid(domain):
    return (
        Parameter(
            "electrolyte.viscosity",
            domain,
            default=6.963e-3,
            unit=syu.pascal * syu.second,
        ),
        Parameter(
            "electrolyte.density", domain, default=1399.8, unit=syu.kg / syu.m**3
        ),
    )


def assert_consistent(model):
    """Units are consistent with and without time stepping, and when factorized."""
    build_system(model, check_units=True)
    build_system(model, substitutions=BDF(1).substitutions(model), check_units=True)
    scaled = build_system(model, factorize_equations=True)
    assert all(scale > 0 for scale in scaled.equation_scales.values())


def test_half_cells_have_consistent_units(reference):
    line = Domain("cell", "interval")
    cell = reference.half_cell(line)
    assert [f.name for f in cell.model.equations] == [
        "phi_s",
        "phi_l",
        "c_V2",
        "c_V3",
        "c_H",
        "c_HSO4",
    ]
    assert_consistent(cell.model)

    rect = Domain("cell", "quadrilateral")
    viscosity, density = fluid(rect)
    electrode = reference.electrode(rect)
    p = Field("p", rect, ("Lagrange", 1), unit=syu.pascal)
    inflow = Parameter("u_in", rect, default=1e-7, unit=syu.m / syu.second)
    darcy = Darcy("darcy", p, electrode.permeability(), viscosity)
    assert_consistent(darcy + BoundaryFlux("inflow", p, inflow, ("Inlet",)))
    cell = reference.half_cell(
        rect,
        electrode,
        velocity=darcy.superficial_velocity,
        outlet=("Outlet",),
        proton_flux=True,
        density=density,
        viscosity=viscosity,
    )
    assert_consistent(cell.model)
    # The flow is data for the electrochemistry
    assert list(build_system(cell.model).known_coefficients) == ["p"]

    box = Domain("cell", "tetrahedron")
    viscosity, density = fluid(box)
    electrode = reference.electrode(box, ("Electrode",))
    channels = ("InletChannel", "OutletChannel")
    u = Field("u", box, ("Lagrange", 2, (3,)), unit=syu.m / syu.second, scale=1e-7)
    p = Field("p", box, ("Lagrange", 1), unit=syu.pascal)
    flow = BrinkmanForchheimer(
        "porous_flow",
        u,
        p,
        electrode.porosity,
        electrode.permeability(),
        electrode.forchheimer(),
        viscosity,
        density,
        markers=("Electrode",),
    ) + NavierStokes("channel_flow", u, p, viscosity, density, markers=channels)
    flow += GradDiv("grad_div", u, 1e-3 * viscosity)
    assert_consistent(flow)
    cell = reference.half_cell(
        box,
        electrode,
        channels=channels,
        velocity=u,
        outlet=("Outlet",),
        proton_flux=True,
        density=density,
        viscosity=viscosity,
    )
    assert_consistent(cell.model)
    names = [m.name for m in cell.model.components]
    assert "cell_V2_channels_transport" in names
    assert "cell_H_channels_storage" in names
    assert "cell_electrolyte_channels" in names


def test_shared_parameters_are_used_by_flow_and_kinetics(reference):
    rect = Domain("cell", "quadrilateral")
    viscosity, density = fluid(rect)
    p = Field("p", rect, ("Lagrange", 1), unit=syu.pascal)
    electrode = reference.electrode(rect)
    darcy = Darcy("darcy", p, electrode.permeability(), viscosity)
    cell = reference.half_cell(
        rect,
        electrode,
        velocity=darcy.superficial_velocity,
        density=density,
        viscosity=viscosity,
    )
    parameters = build_system(cell.model + darcy).parameters
    assert parameters["electrolyte.viscosity"] is viscosity
    assert parameters["electrode.porosity"] is electrode.porosity
    assert "darcy.viscosity" not in parameters


def test_half_cell_errors(reference):
    domain = Domain("cell", "interval")
    with pytest.raises(ValueError, match="need a velocity"):
        reference.half_cell(domain, mass_transfer=True)
    with pytest.raises(ValueError, match="need a velocity"):
        reference.half_cell(domain, outlet=("Outlet",))


def test_the_couple_is_oxidized_or_reduced_on_discharge(reference):
    """On discharge the current enters the solid at the current collector of a
    half-cell that oxidizes its couple, and leaves it at that of one that
    reduces it: the equations of the potentials, with only the current in
    them, are opposite."""
    marked = reference.interval(num_cells=4)
    residuals = []
    for oxidation in (True, False):
        domain = Domain.from_mesh("cell", marked.mesh)
        couple = RedoxCouple("V2", "V3", 0.0, 0.0, oxidation_on_discharge=oxidation)
        cell = reference.half_cell(domain, couple=couple)
        scheme = BDF(1)
        system = build_system(
            cell.model, substitutions=scheme.substitutions(cell.model)
        )
        disc = discretize([system], DomainMaps({domain: marked}))
        scheme.set_step(1, 1.0, disc)
        # Without gradients, rate or change, only the currents are left in
        # the equations of the potentials
        disc.set({cell.current: 0.5})
        problem = disc[system]
        residuals.append([owned_residual(form) for form in problem.residual[:2]])
    # Each process may hold only one of the boundaries, so the checks are
    # over all processes, which all reach the same verdict
    comm = marked.mesh.comm
    for oxidizing, reducing in zip(*residuals, strict=True):
        largest = comm.allreduce(np.abs(oxidizing).max(initial=0.0), op=MPI.MAX)
        mismatch = comm.allreduce(
            np.abs(oxidizing + reducing).max(initial=0.0), op=MPI.MAX
        )
        assert largest > 0
        assert mismatch <= 1e-12 * largest


def owned_residual(form: dolfinx.fem.Form) -> np.ndarray:
    """A residual vector with its ghost contributions added to their owners,
    restricted to the entries this process owns."""
    vector = dolfinx.fem.assemble_vector(form)
    vector.scatter_reverse(dolfinx.la.InsertMode.add)
    size = vector.index_map.size_local * vector.block_size
    return vector.array[:size].copy()


def test_the_reservoir_names_ions_of_the_electrolyte(reference):
    domain = Domain("cell", "interval")
    with pytest.raises(ValueError, match="no ions"):
        RedoxHalfCell(
            domain,
            reference.IONS,
            {**reference.RESERVOIR, "Cl": 1.0},
            reference.COUPLE,
            reference.electrode(domain),
        )


def test_post_processing(reference):
    marked = reference.interval(num_cells=10)
    domain = Domain.from_mesh("cell", marked.mesh)
    c = Field("c", domain, ("Lagrange", 1))
    disc = discretize([c], DomainMaps({domain: marked}))
    disc.set({c: lambda x: 1.0 + x[0] / reference.W})
    function = disc.function(c)
    assert function_range(function) == pytest.approx((1.0, 2.0))
    points, values = evaluate_over_line(function, [0.0], [reference.W], 5)
    np.testing.assert_allclose(values[:, 0], 1.0 + points[:, 0] / reference.W)
    assert values.shape[1] == 1


def test_the_couple_conserves_charge(reference):
    """V2 <-> V3 + 2 e- does not conserve charge. SO4 <-> HSO4 + e- does, but
    sulfate follows from electroneutrality, so it has no field to react."""
    domain = Domain("cell", "interval")
    with pytest.raises(ValueError, match="does not conserve charge"):
        reference.half_cell(
            domain, couple=dataclasses.replace(reference.COUPLE, electrons=2)
        )
    with pytest.raises(ValueError, match="needs a reservoir"):
        reference.half_cell(
            domain,
            couple=dataclasses.replace(
                reference.COUPLE,
                reduced="SO4",
                oxidized="HSO4",
                reactants=None,
                products=None,
            ),
        )


@pytest.mark.parametrize("missing", [(), ("H", "SO4")], ids=["none", "two"])
def test_the_reservoir_lacks_exactly_one_ion(reference, missing):
    """One ion, and only one, follows from electroneutrality."""
    domain = Domain("cell", "interval")
    reservoir = {**reference.RESERVOIR, "SO4": 100.0}
    reservoir = {k: v for k, v in reservoir.items() if k not in missing}
    with pytest.raises(ValueError, match="all ions but one"):
        RedoxHalfCell(
            domain,
            reference.IONS,
            reservoir,
            reference.COUPLE,
            reference.electrode(domain),
        )


CONCENTRATION_UNIT = syu.mol / syu.m**3
DIFFUSIVITY_UNIT = syu.m**2 / syu.second

POSITIVE_IONS = (
    Ion("VO", 2, 3.9e-10 * DIFFUSIVITY_UNIT, positive=True),
    Ion("VO2", 1, 3.9e-10 * DIFFUSIVITY_UNIT, positive=True),
    Ion("H", 1, 9.312e-9 * DIFFUSIVITY_UNIT),
    Ion("HSO4", -1, 1.33e-9 * DIFFUSIVITY_UNIT),
    Ion("H2O", 0, 2.3e-9 * DIFFUSIVITY_UNIT),
    Ion("SO4", -2, 1.065e-9 * DIFFUSIVITY_UNIT),
)
"""The positive electrolyte of Shah et al. (2008): VO2+, VO2+, protons,
bisulfate and water, sulfate by electroneutrality."""

POSITIVE_RESERVOIR = {
    "VO": 1053.0 * CONCENTRATION_UNIT,
    "VO2": 27.0 * CONCENTRATION_UNIT,
    "H": 1200.0 * CONCENTRATION_UNIT,
    "HSO4": 1200.0 * CONCENTRATION_UNIT,
    "H2O": 4.2e4 * CONCENTRATION_UNIT,
}


POSITIVE_COUPLE = RedoxCouple(
    reduced="VO",
    oxidized="VO2",
    rate_constant=0.68 * 3e-9 * syu.m / syu.second,
    formal_potential=1.004 * syu.volt,
    oxidation_on_discharge=False,
    reactants={"VO": 1, "H2O": 1},
    products={"VO2": 1, "H": 2},
)
"""VO2+ + H2O <-> VO2+ + 2 H+ + e- of Shah et al., reduced on discharge."""


def positive_half_cell(domain, electrode, couple=POSITIVE_COUPLE, **kwargs):
    """The positive half-cell of Shah et al."""
    return RedoxHalfCell(
        domain, POSITIVE_IONS, POSITIVE_RESERVOIR, couple, electrode, **kwargs
    )


def test_a_positive_half_cell_produces_protons_and_consumes_water(reference):
    """VO2+ + H2O <-> VO2+ + 2 H+ + e- conserves charge only with its protons,
    and diffusion between the fibres limits it without flow."""
    domain = Domain("cell", "interval")
    electrode = reference.electrode(domain)
    cell = positive_half_cell(domain, electrode, fiber_distance=10 * syu.micrometer)
    assert isinstance(cell.reaction, PistonVelocityButlerVolmer)
    assert_consistent(cell.model)
    with pytest.raises(ValueError, match="does not conserve charge"):
        positive_half_cell(
            domain,
            electrode,
            dataclasses.replace(POSITIVE_COUPLE, reactants=None, products=None),
        )
    with pytest.raises(ValueError, match="has no ions"):
        # An electrolyte without water
        RedoxHalfCell(
            domain,
            tuple(ion for ion in POSITIVE_IONS if ion.name != "H2O"),
            {k: v for k, v in POSITIVE_RESERVOIR.items() if k != "H2O"},
            POSITIVE_COUPLE,
            electrode,
        )


def test_existing_fields_are_used_as_given(reference):
    domain = Domain("cell", "interval")
    phi_s = Field("psi", domain, ("Lagrange", 1), unit=syu.volt)
    phi_l = Field("phi", domain, ("Lagrange", 1), unit=syu.volt)
    protons = Field("c_protons", domain, ("Lagrange", 1), unit=CONCENTRATION_UNIT)
    cell = reference.half_cell(domain, phi_s=phi_s, phi_l=phi_l, fields={"H": protons})
    assert cell.phi_s is phi_s and cell.phi_l is phi_l
    assert cell.concentrations["H"] is protons
    assert cell.electrolyte["H"].field is protons
    assert cell.concentrations["V2"].name == "c_V2"


def test_a_resolved_collector_and_membrane_carry_no_currents(reference):
    """In a full cell the collector and the membrane are domains of their
    own, so the half-cell puts no currents on its boundary."""
    domain = Domain("cell", "interval")
    cell = reference.half_cell(domain, collector=None, membrane=None)
    names = [m.name for m in cell.model.components]
    assert not any("current" in name for name in names)
    with pytest.raises(ValueError, match="resolves them"):
        cell.voltage(None)
    with pytest.raises(ValueError, match="needs its markers"):
        reference.half_cell(domain, membrane=None, proton_flux=True)


def test_two_half_cells_share_the_electrolyte_potential_and_water(reference):
    """A negative and a positive half-cell side by side, with one electrolyte
    potential and one water field, as in a full cell: their parameters and
    models have names of their own, and their sum has consistent units."""
    domain = Domain("cell", "interval")
    phi_l = Field("phi_l", domain, ("Lagrange", 1), unit=syu.volt)
    water = Field("c_H2O", domain, ("Lagrange", 1), unit=CONCENTRATION_UNIT)
    sides = {}
    for side in ("negative", "positive"):
        fields = {
            ion: Field(
                f"{side}_c_{ion}", domain, ("Lagrange", 1), unit=CONCENTRATION_UNIT
            )
            for ion in ("H", "HSO4")
        }
        common = dict(
            phi_s=Field(f"{side}_phi_s", domain, ("Lagrange", 1), unit=syu.volt),
            phi_l=phi_l,
            collector=None,
            membrane=None,
            fiber_distance=10 * syu.micrometer,
            name=side,
            electrolyte_name=f"{side}_electrolyte",
        )
        electrode = PorousElectrode(
            f"{side}_electrode",
            domain,
            (side.capitalize(),),
            porosity=0.68,
            conductivity=500.0,
            specific_area=2e6,
        )
        if side == "negative":
            ions = (*reference.IONS[:4], Ion("H2O", 0, 2.3e-9), reference.IONS[4])
            reservoir = {**reference.RESERVOIR, "H2O": 4.2e4}
            couple = RedoxCouple("V2", "V3", 0.68 * 1.75e-7, -0.255, True)
            sides[side] = RedoxHalfCell(
                domain,
                ions,
                reservoir,
                couple,
                electrode,
                fields={**fields, "H2O": water},
                **common,
            )
        else:
            sides[side] = positive_half_cell(
                domain, electrode, fields={**fields, "H2O": water}, **common
            )
    model = sides["negative"].model + sides["positive"].model
    system = build_system(model, check_units=True)
    names = [f.name for f in system.unknowns]
    assert names.count("phi_l") == 1 and names.count("c_H2O") == 1
    assert {"negative_c_H", "positive_c_H", "c_VO", "c_V2"} <= set(names)
    assert "negative.rate_constant" in system.parameters
    assert "positive_electrolyte.D_H" in system.parameters
