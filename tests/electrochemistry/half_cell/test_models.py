"""The RFBniCS models: units, structure, schedules and post-processing."""

import dolfinx.fem
import numpy as np
import pytest
import sympy.physics.units as syu

from sumfem.discrete import DomainMaps, discretize
from sumfem.electrochemistry import RedoxHalfCell
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
    assert "transport_V2_channels" in names and "storage_H_channels" in names
    assert "electrolyte_channels" in names


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
        cell = RedoxHalfCell(
            domain,
            reference.IONS,
            reference.RESERVOIR,
            "V2",
            "V3",
            reference.electrode(domain),
            oxidation_on_discharge=oxidation,
            rate_constant=0.0,
            formal_potential=0.0,
        )
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
        residuals.append(
            [dolfinx.fem.assemble_vector(form).array for form in problem.residual[:2]]
        )
    for oxidizing, reducing in zip(*residuals, strict=True):
        assert np.abs(oxidizing).max() > 0
        np.testing.assert_allclose(oxidizing, -reducing)


def test_the_reservoir_names_ions_of_the_electrolyte(reference):
    domain = Domain("cell", "interval")
    with pytest.raises(ValueError, match="no ions"):
        RedoxHalfCell(
            domain,
            reference.IONS,
            {**reference.RESERVOIR, "Cl": 1.0},
            "V2",
            "V3",
            reference.electrode(domain),
            oxidation_on_discharge=True,
            rate_constant=2.5e-7,
            formal_potential=-0.36,
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
