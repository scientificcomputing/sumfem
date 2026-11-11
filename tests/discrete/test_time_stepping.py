"""Time stepping with BDF schemes through a Simulation."""

from mpi4py import MPI

import dolfinx.fem
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl
from ufl.algorithms import extract_coefficients

from sumfem.discrete import DomainMaps, discretize
from sumfem.solvers import (
    BDF,
    AdaptiveSimulation,
    AdaptiveTimeStep,
    CrankNicolson,
    ForwardEuler,
    Simulation,
    Solver,
    Theta,
)
from sumfem.symbolic import (
    Clock,
    DirichletBC,
    Domain,
    Dt,
    Field,
    Residual,
    build_system,
)


def spatial(x):
    """Quadratic in space, so degree-2 elements leave only the time error."""
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


def solve_heat(toy, square, sides, scheme, dt, t_end=1.0):
    """du/dt - lap(u) = f with exact solution u = spatial(x) (1 + sin t).

    ``scheme`` is a BDF order or a time scheme. A scheme with a given
    history starts from the exact solution's past values.
    """
    scheme = BDF(scheme) if isinstance(scheme, int) else scheme
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 2))
    clock = Clock(domain, dt=dt)
    x = ufl.SpatialCoordinate(domain)
    exact = spatial(x) * (1 + ufl.sin(clock.t))
    source = spatial(x) * ufl.cos(clock.t) - 6 * (1 + ufl.sin(clock.t))
    model = (
        toy.Accumulation("dudt", u)
        + toy.Diffusion("diffusion", u)
        + toy.Source("source", u, f=source)
    )
    disc = discretize([], DomainMaps({"cell": square()}))
    simulation = Simulation(
        disc, model, bcs=[DirichletBC(u, sides, exact)], clock=clock, scheme=scheme
    )
    disc.set({u: spatial})
    if scheme.given_history:
        disc.set(
            {
                u.previous(k): (lambda x, k=k: spatial(x) * (1 + np.sin(-k * dt)))
                for k in range(1, scheme.order + 1)
            }
        )
    simulation.run(t_end=t_end)
    solution = disc.fields["u"]
    mesh = solution.function_space.mesh
    xm = ufl.SpatialCoordinate(mesh)
    error = (solution - spatial(xm) * (1 + np.sin(t_end))) ** 2 * ufl.dx
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(error))
    return np.sqrt(mesh.comm.allreduce(local, op=MPI.SUM)), disc, clock


@pytest.mark.parametrize(
    "scheme, order",
    [
        (lambda: BDF(1), 1),
        (lambda: BDF(2), 2),
        (lambda: BDF(3, given_history=True), 3),
        (lambda: CrankNicolson(), 2),
        (lambda: Theta(0.75), 1),
        (lambda: BDF(2, blend=0.5), 1),
    ],
    ids=["bdf1", "bdf2", "bdf3", "crank-nicolson", "theta", "blend"],
)
def test_schemes_reach_their_order(toy, square, sides, scheme, order):
    errors = [
        solve_heat(toy, square, sides, scheme(), dt)[0] for dt in (0.1, 0.05, 0.025)
    ]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > order - 0.15), rates


def test_forward_euler_reaches_order_one(toy, square, sides):
    """With steps below the stability limit of the explicit method."""
    errors = np.array(
        [
            solve_heat(toy, square, sides, ForwardEuler(), dt, t_end=0.1)[0]
            for dt in (1e-3, 5e-4, 2.5e-4)
        ]
    )
    # An unstable run would show as a large error, not as a fast rate
    assert errors.max() < 1e-5, errors
    rates = np.log2(errors[:-1] / errors[1:])
    assert np.all(np.abs(rates - 1) < 0.15), rates


def test_bdf3_without_history_starts_at_lower_order(toy, square, sides):
    """The started BDF3 converges at least at second order.

    For this solution ``u''(0) = 0``, so the first, backward Euler step adds
    no leading error and the rate even stays near three.
    """
    errors = [solve_heat(toy, square, sides, 3, dt)[0] for dt in (0.1, 0.05, 0.025)]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 1.85), rates


def test_theta_one_is_backward_euler(toy, square, sides):
    theta, _, _ = solve_heat(toy, square, sides, Theta(1.0), 0.1)
    bdf, _, _ = solve_heat(toy, square, sides, 1, 0.1)
    np.testing.assert_allclose(theta, bdf, rtol=1e-10)


def test_clock_is_advanced(toy, square, sides):
    _, disc, clock = solve_heat(toy, square, sides, 1, dt=0.25, t_end=0.5)
    assert disc.value(clock.t) == pytest.approx(0.5)


def test_time_arguments_are_checked(toy, square, sides):
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    model = toy.Accumulation("dudt", u) + toy.Diffusion("diffusion", u)
    disc = discretize([], DomainMaps({"cell": square()}))
    with pytest.raises(ValueError, match="needs a clock"):
        Simulation(disc, model)
    clock = Clock(domain, dt=0.3)
    with pytest.raises(ValueError, match="needs a time-stepping scheme"):
        Simulation(disc, model, clock=clock)
    simulation = Simulation(
        disc, model, bcs=[DirichletBC(u, sides, 0.0)], clock=clock, scheme=BDF()
    )
    with pytest.raises(ValueError, match="whole number of steps"):
        simulation.run(t_end=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        simulation.run()


def test_bdf_replaces_time_derivatives_with_weighted_history(toy):
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    substitutions = BDF(2).substitutions(
        toy.Accumulation("dudt", u) + toy.Diffusion("d", u)
    )
    assert list(substitutions) == [u.dot]
    terminals = {c.name: c for c in extract_coefficients(substitutions[u.dot])}
    weights = {"bdf.cell.w0", "bdf.cell.w1", "bdf.cell.w2"}
    assert set(terminals) == {"u", "u.previous[1]", "u.previous[2]", *weights}
    assert all(terminals[w].unit == 1 / syu.second for w in weights)
    with pytest.raises(ValueError):
        BDF(4)
    with pytest.raises(ValueError, match="order 2"):
        BDF(3, blend=0.5)
    with pytest.raises(ValueError):
        Theta(1.5)


def test_factorized_time_stepping_matches_the_dimensional_run(toy, square, sides):
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1), unit=syu.K, scale=10.0)

    class Heat(Residual):
        def residual(self):
            D = self.parameter("D", domain, default=1e-2, unit=syu.m**2 / syu.s)
            r = self.parameter("r", domain, default=5.0, unit=syu.K / syu.s)
            dx = domain.dx()
            return D * ufl.inner(ufl.grad(u), ufl.grad(u.v)) * dx - r * u.v * dx

    model = toy.Accumulation("dudt", u) + Heat("heat")
    results = []
    for factorize in (False, True):
        disc = discretize([], DomainMaps({"cell": square()}))
        simulation = Simulation(
            disc,
            model,
            bcs=[DirichletBC(u, sides, 0.0)],
            clock=Clock(domain, dt=0.1),
            scheme=BDF(2),
            factorize_equations=factorize,
        )
        disc.set({u: 1.0})
        simulation.run(num_steps=4)
        results.append(disc.fields["u"].x.array.copy())
    assert np.max(np.abs(results[0])) > 0.1
    np.testing.assert_allclose(results[1], results[0], rtol=1e-9, atol=1e-12)


class Squared(Residual):
    """``(Dt(c^2) - rate, v)``, whose exact solution has ``c^2`` linear in time."""

    def __init__(self, name, field, rate):
        super().__init__(name)
        self.field = field
        self.rate = rate

    def residual(self):
        c = self.field
        return ufl.inner(Dt(c**2) - self.rate, c.v) * c.domain.dx()


@pytest.mark.parametrize(
    "scheme",
    [BDF(1), BDF(2), BDF(2, blend=0.5), BDF(3), CrankNicolson(), ForwardEuler()],
    ids=["bdf1", "bdf2", "blend", "bdf3", "crank-nicolson", "forward-euler"],
)
def test_time_derivative_of_an_expression_is_exact_for_linear_change(square, scheme):
    """Every scheme differentiates ``c^2 = 1 + 2 t`` exactly.

    The chain rule ``2 c Dt(c)`` would not, so this checks that the whole
    expression is differenced over the time steps.
    """
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    clock = Clock(domain, dt=0.1)
    disc = discretize([], DomainMaps({"cell": square()}))
    simulation = Simulation(
        disc, Squared("squared", c, 2.0), clock=clock, scheme=scheme
    )
    disc.set({c: 1.0})
    simulation.run(num_steps=5)
    np.testing.assert_allclose(disc.function(c).x.array, np.sqrt(1 + 2 * 0.5))


def test_time_derivative_of_a_field_is_its_dot():
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    assert Dt(c) is c.dot


def test_time_derivative_of_an_expression_is_replaced_by_the_scheme(toy):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    model = Squared("squared", c, 2.0)
    substitutions = BDF(2).substitutions(model)
    (dot,) = substitutions
    assert dot.fields == (c,)
    replaced = extract_coefficients(substitutions[dot])
    assert c in replaced and c.previous(1) in replaced and c.previous(2) in replaced


def test_copies_differentiate_their_own_fields():
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    copy = Squared("squared", c, 2.0).copy("other")
    (dot,) = BDF(1).substitutions(copy)
    assert dot.fields != (c,)


def test_expression_time_derivatives_need_a_scheme(square):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    with pytest.raises(ValueError, match="time derivatives"):
        Simulation(
            discretize([], DomainMaps({"cell": square()})),
            Squared("squared", c, 2.0),
            clock=Clock(domain, dt=0.1),
        )


@pytest.mark.parametrize("blend", [-0.1, 1.5])
def test_blend_is_checked(blend):
    with pytest.raises(ValueError, match="blend"):
        BDF(2, blend=blend)
    with pytest.raises(ValueError, match="order 2"):
        BDF(1, blend=0.5)


def test_adaptive_time_step_rules():
    step = AdaptiveTimeStep(
        1.0,
        growth_factor=1.2,
        cutback_factor=0.5,
        target_iterations=4,
        max_step=lambda t: 3.0,
        min_step=0.2,
        milestones=[10.0],
        max_retries=2,
    )
    assert step.next(1.0, 2) == pytest.approx(1.2)
    assert step.next(1.0, 6) == pytest.approx(0.5)
    assert step.next(1.0, 4) == pytest.approx(1.0)
    assert step.limit(5.0, 0.0, 100.0) == pytest.approx(3.0)
    assert step.limit(2.0, 9.0, 100.0) == pytest.approx(1.0)
    assert step.limit(2.0, 10.0, 11.0) == pytest.approx(1.0)
    assert step.retry(1.0) == pytest.approx(0.5)
    assert step.retry(0.3) is None
    with pytest.raises(ValueError, match="cutback"):
        AdaptiveTimeStep(1.0, max_retries=1)


def adaptive_heat(toy, square, sides, time_step, scheme=None):
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    clock = Clock(domain, dt=0.1)
    model = toy.Accumulation("dudt", u) + toy.Diffusion("diffusion", u)
    disc = discretize([], DomainMaps({"cell": square()}))
    simulation = AdaptiveSimulation(
        disc,
        model,
        time_step,
        clock,
        scheme=BDF(1) if scheme is None else scheme,
        bcs=[DirichletBC(u, sides, 0.0)],
    )
    disc.set({u: lambda x: x[0] * (1 - x[0]) * x[1] * (1 - x[1])})
    return simulation, disc, u, clock


def test_adaptive_steps_end_at_milestones_and_the_end(toy, square, sides):
    step = AdaptiveTimeStep(
        0.05, growth_factor=2.0, target_iterations=3, milestones=[0.3]
    )
    simulation, disc, _, clock = adaptive_heat(toy, square, sides, step)
    times = []
    simulation.run(t_end=0.7, callback=lambda t, n: times.append(t))
    # A linear problem takes one Newton iteration, so the steps grow
    assert np.diff(times)[1] > np.diff(times)[0]
    assert any(np.isclose(t, 0.3) for t in times)
    assert times[-1] == pytest.approx(0.7)
    assert disc.value(clock.t) == pytest.approx(0.7)


def test_failed_steps_are_retried_with_shorter_steps(toy, square, sides, monkeypatch):
    """A solve that fails restores the unknowns and halves the step."""
    reference, ref_disc, u, _ = adaptive_heat(
        toy, square, sides, AdaptiveTimeStep(0.05)
    )
    reference.run(t_end=0.2)
    simulation, disc, u2, clock = adaptive_heat(
        toy,
        square,
        sides,
        AdaptiveTimeStep(0.1, cutback_factor=0.5, max_retries=1),
    )
    solver = simulation.solver
    solve = type(solver).solve
    failures = []

    def failing_solve(self):
        # Every attempt with a step of 0.1 fails, as Newton would
        if self is solver and disc.value(clock.dt) > 0.075:
            disc.function(u2).x.array[:] = np.nan
            failures.append(float(disc.value(clock.t)))
            raise RuntimeError("Newton did not converge")
        return solve(self)

    # On the class, since a solver that refers to itself is left to the
    # garbage collector
    monkeypatch.setattr(type(solver), "solve", failing_solve)
    times = []
    simulation.run(t_end=0.2, callback=lambda t, n: times.append(t))
    # Only the first step tries 0.1: the step stays at 0.05 after its retry
    assert failures == [pytest.approx(0.1)]
    np.testing.assert_allclose(times, [0.05, 0.1, 0.15, 0.2])
    np.testing.assert_allclose(
        disc.function(u2).x.array, ref_disc.function(u).x.array, rtol=1e-12
    )


def test_adapted_steps_need_a_one_step_scheme(toy, square, sides):
    with pytest.raises(ValueError, match="equal steps"):
        adaptive_heat(toy, square, sides, AdaptiveTimeStep(0.1), scheme=BDF(2))


def test_steps_that_keep_failing_raise(toy, square, sides):
    simulation, *_ = adaptive_heat(
        toy,
        square,
        sides,
        AdaptiveTimeStep(0.1, cutback_factor=0.5, max_retries=1),
    )

    def failing_solve():
        raise RuntimeError("Newton did not converge")

    simulation.solver.solve = failing_solve
    with pytest.raises(RuntimeError, match="after 1 retries"):
        simulation.run(t_end=0.2)


def test_a_hand_written_loop_matches_the_simulation(toy, square, sides):
    """The steps of a simulation, written out with a solver."""
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    clock = Clock(domain, dt=0.05)
    x = ufl.SpatialCoordinate(domain)
    residual = (
        toy.Accumulation("dudt", u)
        + toy.Diffusion("diffusion", u)
        + toy.Source("source", u, f=ufl.sin(3 * x[0]) * (1 + clock.t**2))
    )
    bcs = [DirichletBC(u, sides, clock.t * x[1])]
    solutions = []

    # By a simulation
    disc = discretize([residual], DomainMaps({"cell": square()}))
    disc.set({u: x[0] * (1 - x[0])})
    Simulation(disc, residual, bcs, clock=clock, scheme=BDF(2)).run(num_steps=6)
    solutions.append(disc.function(u).x.array.copy())

    # By hand
    scheme = BDF(2)
    step, substitutions = scheme.prepare(residual, clock)
    system = build_system(step, substitutions=substitutions)
    disc = discretize([system, clock, *bcs], DomainMaps({"cell": square()}))
    solver = Solver(disc[system], bcs)
    disc.set({u: x[0] * (1 - x[0])})
    disc.initialize_history()
    for n in range(1, 7):
        disc.set({clock.t: n * 0.05, clock.dt: 0.05})
        scheme.set_step(n, 0.05, disc)
        solver.solve()
        disc.advance_history()
    solutions.append(disc.function(u).x.array.copy())

    np.testing.assert_allclose(solutions[1], solutions[0], rtol=1e-12, atol=1e-14)


def test_a_zero_residual_changes_nothing(toy, square, sides):
    """A transient model with a zero summand steps as the model without it."""

    class Zero(Residual):
        def residual(self):
            return ufl.ZeroBaseForm((u.v,))

    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    model = (
        toy.Accumulation("dudt", u) + toy.Diffusion("d", u) + toy.Source("s", u, f=1.0)
    )
    values = []
    for summed in (model, model + Zero("zero")):
        disc = discretize([], DomainMaps({"cell": square()}))
        bcs = [DirichletBC(u, sides, 0.0)]
        clock = Clock(domain, dt=0.1)
        Simulation(disc, summed, bcs=bcs, clock=clock, scheme=BDF()).run(num_steps=2)
        values.append(disc.function(u).x.array.copy())
    np.testing.assert_allclose(values[1], values[0], atol=1e-14)
