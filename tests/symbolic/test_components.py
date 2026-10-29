import re

import basix.ufl
import pytest
import sympy.physics.units as syu
import ufl

from sumfem.error_handling import qualified_name
from sumfem.symbolic import (
    ComponentError,
    Domain,
    Field,
    Functional,
    FunctionalSum,
    KnownCoefficient,
    Parameter,
    Residual,
    ResidualSum,
    SystemBuildError,
    build_system,
)


def test_terminals_are_discovered_from_the_residual(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("diff", c)
    assert model.fields == (c,)
    assert model.equations == (c,)
    assert list(model.parameters) == ["diff.k"]
    assert model.parameters["diff.k"].doc == "Conductivity"
    assert dict(model.known_coefficients) == {}
    assert model.domains == (domain,)
    assert model.components == (model,)


def test_a_sum_is_the_union_of_its_summands(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    model = (
        toy.Diffusion("solid", a)
        + toy.Diffusion("liquid", b)
        + toy.Reaction("bv", a, b)
    )
    assert isinstance(model, ResidualSum)
    assert [m.name for m in model.components] == ["solid", "liquid", "bv"]
    assert model.name == "solid+liquid+bv"
    assert model.fields == (a, b)
    assert model.equations == (a, b)
    assert list(model.parameters) == ["bv.j0", "liquid.k", "solid.k"]


def test_summands_add_into_the_same_equation(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("diff", c) + toy.Source("src", c)
    system = build_system(model)
    # One unknown, so one equation that contains both terms
    assert system.unknowns == (c,)
    assert isinstance(system.residual, ufl.Form)
    coefficient_names = {
        getattr(w, "name", None) for w in system.residual.coefficients()
    }
    assert {"diff.k", "src.f"} <= coefficient_names


def test_sums_are_flattened(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    a, b, s = (
        toy.Diffusion("a", c),
        toy.Source("s", c),
        toy.Diffusion("b", c, markers=("X",)),
    )
    left = (a + b) + s
    right = a + (b + s)
    summed = sum([a, b, s])
    assert left.components == right.components == summed.components == (a, b, s)
    signatures = {build_system(m).residual.signature() for m in (left, right, summed)}
    assert len(signatures) == 1


def test_a_model_can_take_part_in_several_sums(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    diffusion = toy.Diffusion("diff", c)
    first = diffusion + toy.Source("s1", c)
    second = diffusion + toy.Source("s2", c)
    assert first.components[0] is second.components[0]


def test_adding_the_same_model_twice_is_an_error(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    diffusion = toy.Diffusion("diff", c)
    with pytest.raises(ComponentError, match="more than once"):
        diffusion + diffusion
    with pytest.raises(ComponentError, match="more than once"):
        (diffusion + toy.Source("s", c)) + diffusion


def test_summands_may_share_a_name_unless_their_parameters_clash(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    summed = toy.Diffusion("diff", c) + toy.Source("diff", c)
    assert sorted(summed.parameters) == ["diff.f", "diff.k"]
    with pytest.raises(ComponentError, match="two different objects"):
        (toy.Diffusion("diff", c) + toy.Diffusion("diff", c)).inventory


def test_two_fields_with_the_same_name_are_an_error(toy, domain):
    first = Field("c", domain, ("Lagrange", 1))
    second = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("a", first) + toy.Diffusion("b", second)
    with pytest.raises(ComponentError, match="'c' is used by two different objects"):
        model.fields


def test_two_parameters_with_the_same_name_are_an_error(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Source("a", c, f=Parameter("f", domain)) + toy.Source(
        "b", c, f=Parameter("f", domain)
    )
    with pytest.raises(ComponentError, match="'f' is used by two different objects"):
        model.parameters


def test_a_shared_parameter_is_one_parameter(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    faraday = Parameter("F", domain, default=96485.33212)
    model = toy.Source("a", c, f=faraday) + toy.Source("b", c, f=2 * faraday)
    assert list(model.parameters) == ["F"]
    assert model.parameters["F"] is faraday


class _RawTerminals(Residual):
    def __init__(self, name, field, kind):
        super().__init__(name)
        self.field, self.kind = field, kind

    def residual(self):
        domain = self.field.domain
        v = self.field.v
        if self.kind == "constant":
            return ufl.Constant(domain) * v * domain.dx()
        if self.kind == "coefficient":
            space = ufl.FunctionSpace(domain, basix.ufl.element("DG", "triangle", 0))
            return ufl.Coefficient(space) * v * domain.dx()
        if self.kind == "test_function":
            # A plain test function on the field's own space is still rejected
            return ufl.TestFunction(self.field.ufl_function_space()) * domain.dx()
        if self.kind == "foreign_test_function":
            space = ufl.FunctionSpace(
                domain, basix.ufl.element("Lagrange", "triangle", 1)
            )
            return self.field * ufl.TestFunction(space) * domain.dx()
        if self.kind == "mesh":
            mesh = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
            return self.field * v * ufl.Measure("dx", domain=mesh)
        if self.kind == "trial":
            return self.field * self.field.du * v * domain.dx()
        if self.kind == "not_a_form":
            return self.field * v
        raise ValueError(self.kind)


@pytest.mark.parametrize(
    "kind, message",
    [
        ("constant", "ufl.Constant"),
        ("coefficient", "unnamed coefficient"),
        ("test_function", re.escape(f"does not belong to a {qualified_name(Field)}")),
        (
            "foreign_test_function",
            re.escape(f"does not belong to a {qualified_name(Field)}"),
        ),
        ("mesh", re.escape(f"not a {qualified_name(Domain)}")),
        ("trial", "must have rank 1"),
        ("not_a_form", "must return a ufl.Form"),
    ],
)
def test_unnamed_or_invalid_terminals_are_rejected(domain, kind, message):
    c = Field("c", domain, ("Lagrange", 1))
    with pytest.raises(ComponentError, match=message):
        _RawTerminals("raw", c, kind).fields


def test_two_domains_with_the_same_name_are_an_error(toy):
    first = Field("a", Domain("cell", "triangle"), ("Lagrange", 1))
    second = Field("b", Domain("cell", "triangle"), ("Lagrange", 1))
    with pytest.raises(ComponentError, match="domain:cell"):
        (toy.Diffusion("a", first) + toy.Diffusion("b", second)).fields


def test_overriding_one_term_needs_no_bookkeeping(toy, domain):
    class VariableDiffusion(toy.Diffusion):
        def conductivity(self):
            return self.known_coefficient("kappa", self.field.domain, ("DG", 0))

    c = Field("c", domain, ("Lagrange", 1))
    model = VariableDiffusion("diff", c)
    assert dict(model.parameters) == {}
    assert list(model.known_coefficients) == ["diff.kappa"]
    assert isinstance(model.known_coefficients["diff.kappa"], KnownCoefficient)


def test_parameter_helper_is_idempotent(domain):
    model = Residual("m")
    k = model.parameter("k", domain, default=1.0)
    assert model.parameter("k", domain) is k
    assert k.name == "m.k"
    with pytest.raises(ComponentError, match="already created"):
        model.parameter("k", domain, shape=(2,))
    with pytest.raises(ComponentError, match="already created"):
        model.parameter("k", Domain("other", "triangle"))
    with pytest.raises(ComponentError):
        model.parameter("a.b", domain)


def test_known_coefficient_helper_is_idempotent(domain):
    model = Residual("m")
    h = model.known_coefficient("h", domain, ("DG", 0))
    assert model.known_coefficient("h", domain, ("DG", 0)) is h
    assert h.name == "m.h"
    with pytest.raises(ComponentError, match="already created"):
        model.known_coefficient("h", Domain("other", "triangle"), ("DG", 0))


def test_inputs_default_to_named_placeholders(toy, domain):
    c = Field("c", domain, ("Lagrange", 1, (2,)))
    source = toy.Source("src", c)
    assert isinstance(source.f, Parameter)
    assert source.f.name == "src.f" and source.f.shape == (2,)
    model = Residual("m")
    velocity = model.input("velocity", domain=domain, element=("Lagrange", 1, (2,)))
    assert isinstance(velocity, KnownCoefficient) and velocity.name == "m.velocity"


def test_passing_a_field_as_input_couples_the_models(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    source = toy.Source("src", a, f=b)
    assert source.f is b
    assert source.fields == (a, b)
    assert source.equations == (a,)


def test_model_names_cannot_contain_dots():
    with pytest.raises(ComponentError):
        Residual("a.b")


def test_base_model_has_no_residual(domain):
    with pytest.raises(NotImplementedError):
        Residual("m").form


def test_only_models_can_be_summed(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    with pytest.raises(TypeError):
        toy.Diffusion("d", c) + 1
    with pytest.raises(TypeError):
        1 + toy.Diffusion("d", c)
    with pytest.raises(TypeError):
        ResidualSum([toy.Diffusion("d", c), "model"])
    with pytest.raises(ComponentError):
        ResidualSum([])


def test_residual_is_built_once(toy, domain):
    calls = []

    class Counting(toy.Diffusion):
        def residual(self):
            calls.append(1)
            return super().residual()

    model = Counting("d", Field("c", domain, ("Lagrange", 1)))
    model.fields
    model.parameters
    build_system(model)
    assert len(calls) == 1


def jacobian_pattern(system):
    return [[block is not None for block in row] for row in system.jacobian]


def test_copy_solves_for_new_fields(toy, domain):
    a = Field("a", domain, ("Lagrange", 2), unit=syu.K, scale=300.0)
    b = Field("b", domain, ("Lagrange", 1))
    velocity = Field("velocity", domain, ("Lagrange", 1))
    model = (
        toy.Diffusion("da", a)
        + toy.Diffusion("db", b)
        + toy.Reaction("r", a, b)
        + toy.Source("s", a, f=velocity)
    )
    copy = model.copy("_pos")
    a_pos, b_pos = copy.equations
    assert (a_pos.name, b_pos.name) == ("a_pos", "b_pos")
    assert a_pos.element == a.element
    assert a_pos.unit == syu.K and a_pos.scale == 300.0
    assert [m.name for m in copy.components] == ["da_pos", "db_pos", "r_pos", "s_pos"]
    assert copy.name == "da_pos+db_pos+r_pos+s_pos"
    # A field without an equation is shared
    assert copy.fields == (velocity, a_pos, b_pos)
    # Parameters the summands created are copied under the new names
    assert sorted(copy.parameters) == ["da_pos.k", "db_pos.k", "r_pos.j0"]
    assert copy.parameters["da_pos.k"].doc == "Conductivity"
    assert sorted(model.parameters) == ["da.k", "db.k", "r.j0"]

    system = build_system(model + copy)
    assert [f.name for f in system.unknowns] == ["a", "b", "a_pos", "b_pos"]
    assert list(system.known_coefficients) == ["velocity"]
    assert jacobian_pattern(system) == [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ]


def test_copy_replaces_time_derivatives_history_and_jacobians(domain):
    c = Field("c", domain, ("Lagrange", 1))
    rate = Parameter("rate", domain, default=2.0)

    class Relaxation(Residual):
        """``(c.dot + rate (c - c_prev), v)``, linearized by hand."""

        def residual(self):
            dx = domain.dx()
            return (c.dot + rate * (c - c.previous(1))) * c.v * dx

        def jacobian(self):
            return rate * c.du * c.v * domain.dx()

    copy = Relaxation("relax").copy("_2")
    (c_2,) = copy.equations
    names = {t.name for t in copy.form.coefficients()}
    assert names == {"c_2", "c_2.dot", "c_2.previous[1]", "rate"}
    # An input given to the model is shared
    assert copy.parameters["rate"] is rate
    assert {a.part() for a in copy.jacobian_term.form.arguments()} == {c_2.count()}
    assert re.fullmatch(
        r"_ResidualCopy\('relax_2', original=.*Relaxation\('relax'\)\)", repr(copy)
    )


def test_copy_names(toy, domain):
    a = Field("a", domain, ("Lagrange", 1))
    diffusion = toy.Diffusion("d", a)
    with pytest.raises(ComponentError, match="non-empty suffix"):
        diffusion.copy("")
    named = ResidualSum([toy.Diffusion("da", a), toy.Source("s", a)], name="cell")
    assert named.copy("_pos").name == "cell_pos"
    # A default input is a parameter the model created, so it is copied
    assert sorted(named.copy("_neg").parameters) == ["da_neg.k", "s_neg.f"]
    # Two copies with the same suffix create fields and parameters of the
    # same names
    with pytest.raises(ComponentError, match="two different objects"):
        (diffusion.copy("_x") + diffusion.copy("_x")).inventory


class GradientEnergy(Functional):
    """``int k / 2 |grad u|^2 dx``, with a parameter ``k``.

    Args:
        name: Name of the functional.
        field: The field ``u``.
    """

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def functional(self):
        k = self.parameter("k", self.field.domain, default=1.0)
        u = self.field
        return 0.5 * k * ufl.inner(ufl.grad(u), ufl.grad(u)) * u.domain.dx()


class Load(Functional):
    """``-int f u dx``.

    Args:
        name: Name of the functional.
        field: The field ``u``.
        f: The source ``f``.
    """

    def __init__(self, name, field, f):
        super().__init__(name)
        self.field, self.f = field, f

    def functional(self):
        return -self.f * self.field * self.field.domain.dx()


def test_functionals_sum_and_vary_into_residuals(domain):
    u = Field("u", domain, ("Lagrange", 1))
    a = Field("a", domain, ("Lagrange", 1))
    energy = GradientEnergy("energy", u) + Load("load", u, a)
    assert isinstance(energy, FunctionalSum)
    assert [c.name for c in (energy + Load("more", u, 1.0)).components] == [
        "energy",
        "load",
        "more",
    ]
    assert energy.fields == (u, a)
    assert list(energy.parameters) == ["energy.k"]
    # Varying u only: a is data, as in a staggered stage
    residual = energy.to_residual([u], name="poisson")
    assert isinstance(residual, Residual)
    assert residual.name == "poisson"
    assert residual.equations == (u,)
    assert build_system(residual).unknowns == (u,)
    # Varying every field gives a an equation too
    assert energy.to_residual().equations == (u, a)


def test_a_functional_has_rank_zero(domain):
    u = Field("u", domain, ("Lagrange", 1))

    class NotScalar(Functional):
        def functional(self):
            return u * u.v * domain.dx()

    with pytest.raises(ValueError, match="rank"):
        NotScalar("bad").form
    with pytest.raises(TypeError):
        GradientEnergy("e", u) + toy_residual(u)
    with pytest.raises(ComponentError, match="varies no field"):
        GradientEnergy("e", u).to_residual([])


def toy_residual(u):
    class Mass(Residual):
        def residual(self):
            return u * u.v * u.domain.dx()

    return Mass("mass")


class _Zero(Residual):
    """A contribution to the equation of a field that is zero."""

    def __init__(self, name: str, field: Field):
        super().__init__(name)
        self.field = field

    def residual(self):
        return ufl.ZeroBaseForm((self.field.v,))


def test_a_zero_residual_keeps_its_equation(toy, domain):
    c = Field("c", domain, ("Lagrange", 1))
    zero = _Zero("zero", c)
    assert zero.equations == (c,)
    model = toy.Diffusion("d", c) + zero
    assert model.equations == (c,)
    assert build_system(model).unknowns == (c,)
    with pytest.raises(SystemBuildError, match="equations for \\['c'\\] are empty"):
        build_system(zero).residual


def test_a_vanishing_residual_asks_for_a_zero_form(domain):
    c = Field("c", domain, ("Lagrange", 1))

    class Vanishing(Residual):
        def residual(self):
            return ufl.inner(ufl.zero(), c.v) * domain.dx()

    with pytest.raises(ComponentError, match="ZeroBaseForm"):
        Vanishing("vanishing").fields
