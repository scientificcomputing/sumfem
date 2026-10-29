import dataclasses

import pytest
import ufl

from sumfem.symbolic import (
    Domain,
    Field,
    Parameter,
    Residual,
    System,
    SystemBuildError,
    build_system,
)


def pattern(blocks):
    """Sparsity pattern of a block matrix."""
    return [[block is not None for block in row] for row in blocks]


def labels(form):
    """Labels of the spaces of a form's arguments, by argument number."""
    return [
        a.ufl_function_space().label()
        for a in sorted(form.arguments(), key=lambda a: a.number())
    ]


class OneWay(Residual):
    """Adds ``(source, v)`` to the equation of ``target``."""

    def __init__(self, name, target, source):
        super().__init__(name)
        self.target, self.source = target, source

    def residual(self):
        return -self.source * self.target.v * self.target.domain.dx()


class Picard(Residual):
    """``(u^2 grad u, grad v)`` with a Picard linearization that freezes ``u^2``."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u, v = self.field, self.field.v
        return u**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * self.field.domain.dx()

    def jacobian(self):
        u, v, du = self.field, self.field.v, self.field.du
        return u**2 * ufl.inner(ufl.grad(du), ufl.grad(v)) * self.field.domain.dx()


class Mass(Residual):
    """``(u, v)`` with a mass-matrix preconditioner."""

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        return self.field * self.field.v * self.field.domain.dx()

    def preconditioner(self):
        return self.field.du * self.field.v * self.field.domain.dx()


def test_single_unknown_gives_unblocked_forms(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    system = build_system(toy.Diffusion("d", c) + toy.Source("s", c))
    assert not system.blocked
    assert isinstance(system.residual, ufl.Form)
    assert isinstance(system.jacobian, ufl.Form)
    (test,) = system.residual.arguments()
    assert test.number() == 0 and test.part() is None
    assert [(a.number(), a.part()) for a in system.jacobian.arguments()] == [
        (0, None),
        (1, None),
    ]
    assert system.preconditioner is None
    assert system.index("c") == system.index(c) == 0


def test_two_fields_with_the_same_element_get_separate_blocks(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    system = build_system(toy.Diffusion("da", a) + toy.Diffusion("db", b))
    assert system.blocked
    assert system.unknowns == (a, b)
    assert [labels(F_i) for F_i in system.residual] == [
        [a.ufl_function_space().label()],
        [b.ufl_function_space().label()],
    ]
    assert pattern(system.jacobian) == [[True, False], [False, True]]


def test_coupling_gives_off_diagonal_blocks(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    one_way = toy.Diffusion("da", a) + toy.Diffusion("db", b) + OneWay("ab", b, a)
    assert pattern(build_system(one_way).jacobian) == [[True, False], [True, True]]
    two_way = toy.Diffusion("da", a) + toy.Diffusion("db", b) + toy.Reaction("bv", a, b)
    assert pattern(build_system(two_way).jacobian) == [[True, True], [True, True]]


def test_rfb_like_reaction_contributes_to_every_equation(toy, domain):
    phi_s = Field("phi_s", domain, ("Lagrange", 1))
    phi_l = Field("phi_l", domain, ("Lagrange", 1))
    c2 = Field("c_V2", domain, ("Lagrange", 1))
    c3 = Field("c_V3", domain, ("Lagrange", 1))

    class ButlerVolmer(Residual):
        def residual(self):
            j0 = self.parameter("j0", domain, default=1.0)
            rate = j0 * (c2 - c3) * ufl.sinh(phi_s - phi_l)
            dx = domain.dx("Electrode")
            return (rate * phi_s.v - rate * phi_l.v - rate * c2.v + rate * c3.v) * dx

    model = (
        toy.Diffusion("solid", phi_s)
        + toy.Diffusion("liquid", phi_l)
        + toy.Diffusion("V2", c2)
        + toy.Diffusion("V3", c3)
        + ButlerVolmer("bv")
    )
    system = build_system(model)
    assert system.unknowns == (phi_s, phi_l, c2, c3)
    assert pattern(system.jacobian) == [[True] * 4] * 4
    assert all(block is not None for block in system.derivative("bv.j0"))
    solid_k = system.derivative("solid.k")
    assert solid_k[0] is not None
    assert solid_k[1:] == (None, None, None)


def test_field_order(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("DG", 1))
    model = toy.Diffusion("da", a) + toy.InteriorPenalty("ipb", b) + OneWay("ab", b, a)
    system = build_system(model, field_order=[b, "a"])
    assert system.unknowns == (b, a)
    assert labels(system.residual[0]) == [b.ufl_function_space().label()]
    assert system.residual[0].integrals_by_type("interior_facet")
    assert not system.residual[1].integrals_by_type("interior_facet")
    assert pattern(system.jacobian) == [[True, True], [False, True]]


def test_fields_on_two_domains(toy):
    left = Domain("left", "triangle")
    right = Domain("right", "triangle")
    a = Field("a", left, ("Lagrange", 1))
    b = Field("b", right, ("Lagrange", 2))
    # The coupling term is integrated over the left domain, using b from the right
    model = toy.Diffusion("da", a) + toy.Diffusion("db", b) + OneWay("ba", a, b)
    system = build_system(model, field_order=[b, a])
    assert system.domains == (left, right)
    assert pattern(system.jacobian) == [[True, False], [True, True]]
    # a's equation, including the coupling term, is integrated over the left domain
    assert {i.ufl_domain() for i in system.residual[1].integrals()} == {left}
    assert b in system.residual[1].coefficients()


def test_fields_without_an_equation_are_known_coefficients(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    velocity = Field("velocity", domain, ("Lagrange", 1))
    system = build_system(toy.Diffusion("da", a) + toy.Source("src", a, f=velocity))
    assert system.unknowns == (a,)
    assert list(system.known_coefficients) == ["velocity"]
    assert system.known_coefficients["velocity"] is velocity
    assert velocity in system.residual.coefficients()
    assert system.derivative(velocity) is not None
    assert system.derivative("velocity") is not None


def test_inconsistent_field_order_is_an_error(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    other = Field("other", domain, ("Lagrange", 1))
    model = toy.Diffusion("da", a) + toy.Diffusion("db", b)
    with pytest.raises(SystemBuildError, match="listed twice"):
        build_system(model, field_order=[a, a, b])
    with pytest.raises(SystemBuildError, match="every unknown"):
        build_system(model, field_order=[a])
    with pytest.raises(SystemBuildError, match="'other' in field_order is not an"):
        build_system(model, field_order=[a, b, other])
    # A field without an equation is in the model, but is not an unknown
    known = toy.Diffusion("da", a) + toy.Source("s", a, f=b)
    with pytest.raises(SystemBuildError, match="'b' in field_order is not an"):
        build_system(known, field_order=[a, "b"])


def test_parameters_are_real_coefficients_in_the_system(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    system = build_system(toy.Diffusion("d", c))
    k = system.parameters["d.k"]
    assert k in system.residual.coefficients()
    sensitivity = system.derivative("d.k")
    assert sensitivity.signature() == system.derivative(k).signature()
    assert labels(sensitivity) == [
        c.ufl_function_space().label(),
        k.ufl_function_space().label(),
    ]


def test_derivative_lookup_errors(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    system = build_system(toy.Diffusion("d", c))
    with pytest.raises(KeyError, match="is an unknown"):
        system.derivative(c)
    with pytest.raises(KeyError):
        system.derivative("missing")
    with pytest.raises(KeyError):
        system.derivative(Parameter("d.k", domain))
    with pytest.raises(KeyError):
        system.derivative(Field("unused", domain, ("Lagrange", 1)))
    with pytest.raises(KeyError):
        system.index("missing")


def test_real_field_couples_like_any_other_field(toy, domain):
    phi = Field("phi", domain, ("Lagrange", 1))
    voltage = Field("V_cell", domain, "Real")
    current = Parameter("I", domain, default=1.0)

    class Galvanostatic(Residual):
        """Set the potential level so that the total current equals I."""

        def residual(self):
            dx = domain.dx()
            return (phi - voltage) * phi.v * dx + (phi - current) * voltage.v * dx

    system = build_system(toy.Diffusion("d", phi) + Galvanostatic("galv"))
    assert system.unknowns == (phi, voltage)
    assert pattern(system.jacobian) == [[True, True], [True, False]]


def test_hand_written_jacobian_is_used(domain):
    c = Field("c", domain, ("Lagrange", 1))
    automatic = build_system(Picard("p", c))
    # Compare against the derivative of the residual
    exact = ufl.algorithms.expand_derivatives(
        ufl.derivative(automatic.residual, c, automatic.trial_functions[0])
    )
    assert automatic.jacobian.signature() != exact.signature()


def test_hand_written_and_automatic_jacobians_are_summed(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    system = build_system(Picard("p", a) + toy.Diffusion("db", b) + OneWay("ab", b, a))
    assert pattern(system.jacobian) == [[True, False], [True, True]]


def test_hand_written_jacobian_of_a_known_field_is_an_error(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))

    class Coupling(OneWay):
        # Linearizes with respect to the source, which is not an unknown below
        def jacobian(self):
            return -self.source.du * self.target.v * self.target.domain.dx()

    model = toy.Diffusion("da", a) + Coupling("ba", a, b)
    with pytest.raises(SystemBuildError, match="b.du"):
        build_system(model)


def test_preconditioner(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    assert build_system(toy.Diffusion("da", a)).preconditioner is None
    system = build_system(Mass("ma", a) + toy.Diffusion("db", b) + OneWay("ab", b, a))
    assert pattern(system.preconditioner) == [[True, False], [True, True]]


def test_building_does_not_modify_the_model(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("d", c) + toy.Source("s", c)
    form = model.form
    first = build_system(model)
    second = build_system(model)
    assert model.form is form
    assert first.residual.signature() == second.residual.signature()
    assert c.v in form.arguments()


def test_block_arguments_are_those_of_the_forms(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("DG", 0))
    blocked = build_system(toy.Diffusion("da", a) + toy.InteriorPenalty("ipb", b))
    for i, F_i in enumerate(blocked.residual):
        assert F_i.arguments() == (blocked.test_functions[i],)
        J_ii = blocked.jacobian[i][i]
        assert J_ii.arguments() == (
            blocked.test_functions[i],
            blocked.trial_functions[i],
        )
    single = build_system(toy.Diffusion("d", a))
    assert single.residual.arguments() == single.test_functions
    assert single.jacobian.arguments() == (
        *single.test_functions,
        *single.trial_functions,
    )


def test_block_arguments_are_numbered_0_or_1(domain):
    from sumfem.symbolic.system import _block_arguments

    a = Field("a", domain, ("Lagrange", 1))
    with pytest.raises(ValueError, match="got 2"):
        _block_arguments((a,), 2)


def test_a_system_stores_its_inputs_and_derives_the_rest(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    model = toy.Diffusion("da", a) + toy.Diffusion("db", b)
    stored = [f.name for f in dataclasses.fields(System)]
    assert stored == ["component", "field_order", "substitutions", "equation_scales"]
    system = System(model, ("b", "a"))
    assert system.unknowns == (b, a)
    assert sorted(system.parameters) == ["da.k", "db.k"]
    reordered = dataclasses.replace(system, field_order=None)
    assert reordered.unknowns == (a, b)
    with pytest.raises(SystemBuildError, match="a scale for each unknown"):
        System(model, equation_scales={"a": 1.0})
    with pytest.raises(SystemBuildError, match="must be positive"):
        System(model, equation_scales={"a": 1.0, "b": 0.0})
