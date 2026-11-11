"""The models reproduce the forms and solutions of the RFBniCS examples."""

import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl
import ufl.algorithms

from sumfem.discrete import DomainMaps, discretize
from sumfem.models import BoundaryFlux, Darcy
from sumfem.solvers import BDF, Simulation
from sumfem.symbolic import Clock, Domain, Field, Parameter, build_system

IONS = ("V2", "V3", "H", "HSO4")


def fluid(domain):
    """Viscosity and density of the examples, shared by flow and kinetics."""
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


def unknowns(cell):
    return [cell.phi_s, cell.phi_l, *(cell.concentrations[ion] for ion in IONS)]


def set_state(disc, cell, reference):
    """A smooth state away from equilibrium, with a different previous step."""
    W = reference.W
    disc.set(
        {
            cell.phi_s: lambda x: 0.01 * np.sin(np.pi * x[0] / W),
            cell.phi_l: lambda x: 0.36 + 0.005 * x[0] / W,
            cell.concentrations["V2"]: lambda x: (
                300 * (1 + 0.1 * np.sin(2 * np.pi * x[0] / W))
            ),
            cell.concentrations["V3"]: lambda x: (
                300 * (1 - 0.1 * np.cos(np.pi * x[0] / W))
            ),
            cell.concentrations["H"]: lambda x: 1000 * (1 + 0.05 * x[0] / W),
            cell.concentrations["HSO4"]: lambda x: 1000 * (1 - 0.02 * x[0] / W),
        }
    )
    for ion in IONS:
        field = cell.concentrations[ion]
        previous = disc.function(field.previous(1))
        previous.x.array[:] = 0.99 * disc.function(field).x.array


def compare(disc, cell, problem, reference, dt, current, **options):
    """Assemble both residuals, block by block, and compare them."""
    fields = unknowns(cell)
    spaces = [disc.function_space(f) for f in fields]
    tests = ufl.TestFunctions(ufl.MixedFunctionSpace(*spaces))
    marked = disc.maps.mesh_data(cell.phi_s.domain)
    residual = reference.electrochemistry(
        marked,
        [disc.function(f) for f in fields],
        [disc.function(cell.concentrations[ion].previous(1)) for ion in IONS],
        tests,
        dt,
        current,
        **options,
    )
    expected = ufl.extract_blocks(residual)
    for field, ours, theirs in zip(fields, problem.residual, expected, strict=True):
        a = dolfinx.fem.assemble_vector(ours).array
        b = dolfinx.fem.assemble_vector(dolfinx.fem.form(theirs)).array
        scale = np.abs(b).max()
        assert scale > 0, field.name
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-11 * scale, err_msg=field.name
        )


def backward_euler(cell):
    """The scheme, and the system of one backward Euler step of the cell's model."""
    scheme = BDF(1)
    return scheme, build_system(
        cell.model, substitutions=scheme.substitutions(cell.model)
    )


def transient(disc, scheme, system, cell, dt, current):
    """Bind the step's system, and set the step and current."""
    problem = disc[system]
    scheme.set_step(1, dt, disc)
    disc.set({cell.current: current})
    return problem


def test_1d_residual_matches_rfbnics(reference):
    marked = reference.interval()
    domain = Domain.from_mesh("cell", marked.mesh)
    cell = reference.half_cell(domain)
    scheme, system = backward_euler(cell)
    disc = discretize(
        [system],
        DomainMaps({domain: marked}),
        max_quadrature_degree=reference.QUADRATURE_DEGREE,
    )
    set_state(disc, cell, reference)
    problem = transient(disc, scheme, system, cell, dt=0.1, current=-0.5)
    compare(disc, cell, problem, reference, dt=0.1, current=-0.5)


def test_2d_residual_matches_rfbnics(reference):
    marked = reference.rectangle()
    domain = Domain.from_mesh("cell", marked.mesh)
    viscosity, density = fluid(domain)
    electrode = reference.electrode(domain)
    p = Field("p", domain, ("Lagrange", 1), unit=syu.pascal)
    flow = Darcy("darcy", p, electrode.permeability(), viscosity)
    cell = reference.half_cell(
        domain,
        electrode,
        velocity=flow.superficial_velocity,
        outlet=("Outlet",),
        proton_flux=True,
        density=density,
        viscosity=viscosity,
    )
    scheme, system = backward_euler(cell)
    disc = discretize(
        [system, p],
        DomainMaps({domain: marked}),
        max_quadrature_degree=reference.QUADRATURE_DEGREE,
    )
    set_state(disc, cell, reference)
    # A pressure drop of 1 kPa from inlet to outlet
    disc.set({p: lambda x: 3e5 + 1e3 * (1 - x[1] / reference.H)})
    problem = transient(disc, scheme, system, cell, dt=0.5, current=0.5)
    velocity = reference.darcy_velocity(marked, disc.function(p))
    compare(
        disc,
        cell,
        problem,
        reference,
        dt=0.5,
        current=0.5,
        velocity=velocity,
        membrane_flux=True,
        outlet=True,
    )


def test_2d_pressure_matches_rfbnics(reference):
    marked = reference.rectangle()
    domain = Domain.from_mesh("cell", marked.mesh)
    viscosity, _ = fluid(domain)
    electrode = reference.electrode(domain)
    p = Field("p", domain, ("Lagrange", 1), unit=syu.pascal)
    inflow = Parameter("u_in", domain, default=1e-7, unit=syu.m / syu.second)
    model = Darcy("darcy", p, electrode.permeability(), viscosity) + BoundaryFlux(
        "inflow", p, inflow, ("Inlet",)
    )
    disc = discretize(
        [model],
        DomainMaps({domain: marked}),
        max_quadrature_degree=reference.QUADRATURE_DEGREE,
    )
    disc.set({p: lambda x: 3e5 + 1e3 * x[0] * x[1] / (reference.W * reference.H)})
    problem = disc[model]
    q = ufl.TestFunction(disc.function_space(p))
    expected = reference.darcy(marked, disc.function(p), q)
    a = dolfinx.fem.assemble_vector(problem.residual).array
    b = dolfinx.fem.assemble_vector(dolfinx.fem.form(expected)).array
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())


def test_3d_residual_matches_rfbnics(reference):
    marked = reference.box()
    domain = Domain.from_mesh("cell", marked.mesh)
    viscosity, density = fluid(domain)
    electrode = reference.electrode(domain, ("Electrode",))
    u = Field("u", domain, ("Lagrange", 2, (3,)), unit=syu.m / syu.second)
    cell = reference.half_cell(
        domain,
        electrode,
        channels=("InletChannel", "OutletChannel"),
        velocity=u,
        outlet=("Outlet",),
        proton_flux=True,
        density=density,
        viscosity=viscosity,
    )
    scheme, system = backward_euler(cell)
    disc = discretize(
        [system],
        DomainMaps({domain: marked}),
        max_quadrature_degree=reference.QUADRATURE_DEGREE,
    )
    # The cube is 1 m wide, so scale the state of the 2.5 mm electrode
    set_state(disc, cell, type(reference)(**{**vars(reference), "W": 1.0}))
    disc.set({u: lambda x: (1e-4 * (1 + x[1]), 2e-5 * x[2], -1e-5 * x[0])})
    cell.balance_currents(disc)
    ratio = float(disc.value(cell.area_ratio))
    assert ratio == pytest.approx(2.0)
    problem = transient(disc, scheme, system, cell, dt=0.2, current=-0.5)
    compare(
        disc,
        cell,
        problem,
        reference,
        dt=0.2,
        current=-0.5,
        velocity=disc.function(u),
        regions=True,
        area_ratio=ratio,
        membrane_flux=True,
        outlet=True,
    )


def rfbnics_1d(marked, reference, schedule):
    """The time loop of the 1D example of RFBniCS, on a blocked function."""
    mesh = marked.mesh
    el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, basix.ufl.blocked_element(el, shape=(6,)))
    X = dolfinx.fem.Function(V)
    X_old = dolfinx.fem.Function(V)
    parameters = reference.PARAMETERS
    for k, ion in enumerate(IONS, start=2):
        X.sub(k).interpolate(
            lambda x, c=parameters[f"c0_{ion}"]: np.full(x.shape[1], c)
        )
    X.sub(1).interpolate(lambda x: np.full(x.shape[1], -parameters["E_formal"]))
    X_old.x.array[:] = X.x.array
    u = ufl.split(X)
    dt = dolfinx.fem.Constant(mesh, 0.1)
    current = dolfinx.fem.Constant(mesh, 0.0)
    residual = reference.electrochemistry(
        marked, u, ufl.split(X_old)[2:], ufl.TestFunctions(V), dt, current
    )
    tdim = mesh.topology.dim
    membrane = marked.entities(tdim - 1, "Membrane")
    bcs = [
        dolfinx.fem.dirichletbc(
            np.float64(parameters[f"c0_{ion}"]),
            dolfinx.fem.locate_dofs_topological(V.sub(k), tdim - 1, membrane),
            V.sub(k),
        )
        for k, ion in enumerate(IONS, start=2)
    ]
    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual,
        X,
        bcs=bcs,
        petsc_options_prefix="rfbnics_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1e-8,
            "snes_rtol": 1e-12,
            "snes_stol": 1e-14,
            "snes_error_if_not_converged": True,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            # No potential is fixed, so the Jacobian is singular
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_25": 0,
        },
    )
    for signed_current, steps in schedule:
        current.value = signed_current
        for count, step in steps:
            dt.value = step
            for _ in range(count):
                problem.solve()
                X_old.x.array[:] = X.x.array
    return X, V


def test_1d_charge_and_discharge_match_rfbnics(reference):
    marked = reference.interval(num_cells=30)
    # The signed current density, and the number and size of the time steps
    schedule = (
        (-0.5, ((3, 0.1), (2, 1.0))),
        (0.0, ((2, 1.0),)),
        (0.5, ((3, 1.0),)),
    )
    X, V = rfbnics_1d(marked, reference, schedule)

    domain = Domain.from_mesh("cell", marked.mesh)
    cell = reference.half_cell(domain)
    inlet = cell.inlet_conditions("Membrane")
    disc = discretize(
        [cell.model, *inlet],
        DomainMaps({domain: marked}),
        max_quadrature_degree=reference.QUADRATURE_DEGREE,
    )
    cell.initialize(disc)
    clock = Clock(domain)
    simulation = Simulation(
        disc,
        cell.model,
        inlet,
        clock=clock,
        scheme=BDF(1),
        petsc_options={"snes_atol": 1e-8, "snes_rtol": 1e-12, "snes_stol": 1e-14},
    )
    voltages, charge = [], []

    def monitor(t, step):
        voltages.append(cell.voltage(disc))
        charge.append(cell.state_of_charge(disc))

    for signed_current, steps in schedule:
        disc.set({cell.current: signed_current})
        for count, dt in steps:
            disc.set({clock.dt: dt})
            simulation.run(num_steps=count, callback=monitor)
    assert len(voltages) == 10
    assert float(disc.value(clock.t)) == pytest.approx(0.3 + 2.0 + 2.0 + 3.0)
    values = {}
    for k, field in enumerate(unknowns(cell)):
        space, dofs = V.sub(k).collapse()
        expected = dolfinx.fem.Function(space)
        expected.x.array[:] = X.x.array[dofs]
        ours = disc.function(field)
        # Both are degree-one Lagrange on one interval, so order by vertex
        order_ours = np.argsort(ours.function_space.tabulate_dof_coordinates()[:, 0])
        order_theirs = np.argsort(
            expected.function_space.tabulate_dof_coordinates()[:, 0]
        )
        values[field.name] = ours.x.array[order_ours], expected.x.array[order_theirs]
    # Only differences of potentials are defined, so measure both potentials
    # from the electrolyte potential at the membrane
    for name in ("phi_s", "phi_l"):
        values[name] = tuple(values[name][i] - values["phi_l"][i][-1] for i in range(2))
    for name, (ours, theirs) in values.items():
        scale = max(np.abs(theirs).max(), 1e-3)
        np.testing.assert_allclose(
            ours, theirs, rtol=1e-7, atol=1e-9 * scale, err_msg=name
        )
    # Charging turns V3 into V2, and discharging turns it back
    assert charge[4] > 0.5
    assert charge[-1] < charge[6]
    assert charge[6] == pytest.approx(charge[4], rel=1e-6)
    assert np.isfinite(voltages).all()
