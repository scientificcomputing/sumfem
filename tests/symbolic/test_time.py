import numpy as np
import ufl

from sumfem.symbolic import Clock, Field, build_system


def names(form):
    return {getattr(w, "name", None) for w in form.coefficients()}


def test_time_derivative_makes_a_steady_model_transient(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    system = build_system(toy.Accumulation("dcdt", c) + toy.Diffusion("diff", c))
    assert system.unknowns == (c,)
    assert {"c", "c.dot", "diff.k"} <= names(system.residual)


def test_time_derivative_lands_in_its_own_equation(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    model = (
        toy.Accumulation("dadt", a)
        + toy.Diffusion("da", a)
        + toy.Diffusion("db", b)
        + toy.Reaction("bv", a, b)
    )
    system = build_system(model)
    assert "a.dot" in names(system.residual[0])
    assert "a.dot" not in names(system.residual[1])
    # The time derivative is linear in u's placeholder, not in u
    assert system.jacobian[1][1] is not None


def test_clock(domain):
    clock = Clock(domain, t0=1.5, dt=0.1)
    assert clock.t.name == "clock.t" and clock.dt.name == "clock.dt"
    np.testing.assert_allclose(clock.t.default, 1.5)
    np.testing.assert_allclose(clock.dt.default, 0.1)
    assert Clock(domain, name="flow").dt.default is None


def test_clock_parameters_can_appear_in_sources(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    clock = Clock(domain, dt=0.1)
    model = toy.Accumulation("dcdt", c) + toy.Source("src", c, f=ufl.sin(clock.t))
    system = build_system(model)
    assert "clock.t" in system.parameters
