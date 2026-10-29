import basix.ufl
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl
from ufl.domain import extract_unique_domain

from sumfem.symbolic import (
    ComponentError,
    Field,
    KnownCoefficient,
    Parameter,
    as_expr,
)
from sumfem.symbolic.elements import as_element
from sumfem.symbolic.terminals import FieldRole, FieldTerminal, field_of


def test_fields_with_the_same_element_are_distinct(domain):
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    assert a.element == b.element
    assert a.ufl_function_space() != b.ufl_function_space()
    assert a != b
    assert a.v != b.v
    assert a.du != b.du
    assert a.v.number() == 0 and a.du.number() == 1
    assert a.v.part() == a.du.part()
    assert a.v.part() != b.v.part()


def test_field_counts_are_unique_across_subclasses(domain):
    class Concentration(Field):
        pass

    fields = [
        Field("a", domain, ("Lagrange", 1)),
        Concentration("b", domain, ("Lagrange", 1)),
        Field("c", domain, ("Lagrange", 1)),
        Concentration("d", domain, ("Lagrange", 1)),
    ]
    counts = [f.count() for f in fields]
    # All UFL coefficients share one counter, so counts increase and never repeat
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    for f in fields:
        assert f.v.part() == f.du.part() == f.count()


def test_forms_of_two_fields_can_be_added(domain):
    # Plain TestFunctions on two spaces cannot be summed in UFL. Field test
    # functions can, because each field has its own part number.
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 2))
    form = a * a.v * domain.dx() + b * b.v * domain.dx()
    assert {arg.part() for arg in form.arguments()} == {a.v.part(), b.v.part()}


def test_field_is_its_own_solution_coefficient(domain):
    c = Field("c", domain, ("Lagrange", 1))
    assert isinstance(c, ufl.Coefficient)
    assert c.name == "c" and c.domain is domain
    # Fields are used directly in expressions
    assert ufl.grad(c).ufl_shape == (2,)


def test_field_terminals(domain):
    c = Field("c", domain, ("Lagrange", 1))
    assert c.dot is c.dot and c.dot.role is FieldRole.DOT and c.dot.name == "c.dot"
    assert c.previous() is c.previous(1)
    assert c.previous(2).role is FieldRole.PREVIOUS
    assert c.previous(2).k == 2 and c.previous(2).name == "c.previous[2]"
    assert c.dot.k == 0
    assert c.previous(1) != c.previous(2) != c
    with pytest.raises(ValueError):
        c.previous(0)
    with pytest.raises(TypeError, match="FieldRole"):
        FieldTerminal(c, "value")


def test_field_of(domain):
    c = Field("c", domain, ("Lagrange", 1))
    for obj in (c, c.v, c.du, c.dot, c.previous(3)):
        assert field_of(obj) is c
    # Only the field's own arguments belong to it
    assert field_of(ufl.TestFunction(c.ufl_function_space())) is None
    assert field_of(ufl.Argument(c.ufl_function_space(), 0, part=c.count() + 1)) is None
    assert field_of(ufl.SpatialCoordinate(domain)) is None


def test_field_spaces_are_labeled_with_the_name(domain):
    # Labels enter form signatures, so they must not depend on creation order
    assert Field("c", domain, ("Lagrange", 1)).ufl_function_space().label() == "c"


@pytest.mark.parametrize(
    "spec, shape",
    [
        (("Lagrange", 2), ()),
        (("Lagrange", 1, (2,)), (2,)),
        (("DG", 0, (2, 2)), (2, 2)),
        ("Real", ()),
        (("Real", 0, (3,)), (3,)),
    ],
)
def test_element_specs(domain, spec, shape):
    element = as_element(spec, domain)
    assert element.reference_value_shape == shape


def test_element_objects_are_checked_against_the_domain(domain):
    element = basix.ufl.element("Lagrange", "triangle", 3)
    assert as_element(element, domain) is element
    with pytest.raises(ComponentError):
        as_element(basix.ufl.element("Lagrange", "quadrilateral", 1), domain)
    with pytest.raises(TypeError):
        as_element(("Lagrange",), domain)
    with pytest.raises(TypeError):
        as_element(3, domain)


def test_real_field(domain):
    voltage = Field("V_cell", domain, "Real")
    assert voltage.ufl_shape == ()
    assert voltage.element.reference_value_shape == ()


def test_terminals_require_a_sumfem_domain():
    mesh = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    with pytest.raises(TypeError):
        Field("u", mesh, ("Lagrange", 1))
    with pytest.raises(TypeError):
        Parameter("k", mesh)
    with pytest.raises(TypeError):
        KnownCoefficient("h", mesh, ("DG", 0))


def test_parameter_lives_on_a_real_space(domain):
    k = Parameter(
        "k", domain, shape=(2, 2), default=np.eye(2), unit=syu.m**2, doc="Tensor"
    )
    assert isinstance(k, ufl.Coefficient)
    assert k.name == "k"
    assert k.shape == (2, 2) and k.ufl_shape == (2, 2)
    assert k.doc == "Tensor" and k.unit == syu.m**2
    np.testing.assert_allclose(k.default, np.eye(2))
    # The default is copied, so it cannot be changed through the property
    k.default[0, 0] = 5
    np.testing.assert_allclose(k.default, np.eye(2))
    element = k.ufl_function_space().ufl_element()
    assert type(element) is type(basix.ufl.real_element("triangle", (2, 2)))


def test_parameter_without_default(domain):
    assert Parameter("k", domain).default is None


def test_scalar_default_sets_every_component(domain):
    np.testing.assert_allclose(
        Parameter("g", domain, shape=(2,), default=3.0).default, [3, 3]
    )


def test_parameter_default_is_validated(domain):
    with pytest.raises(ComponentError):
        Parameter("k", domain, shape=(2,), default=[1.0, 2.0, 3.0])
    with pytest.raises(ComponentError):
        Parameter("k", domain, default="one")


def test_residual_can_be_differentiated_with_respect_to_a_parameter(domain):
    k = Parameter("k", domain, default=1.0)
    u = Field("u", domain, ("Lagrange", 1))
    form = k * ufl.inner(ufl.grad(u), ufl.grad(u.v)) * domain.dx()
    dk = ufl.TrialFunction(k.ufl_function_space())
    sensitivity = ufl.algorithms.expand_derivatives(ufl.derivative(form, k, dk))
    assert len(sensitivity.arguments()) == 2
    assert not sensitivity.empty()


def test_known_coefficient(domain):
    h = KnownCoefficient("h", domain, ("DG", 0), doc="Cell size")
    assert isinstance(h, ufl.Coefficient)
    assert h.name == "h" and h.doc == "Cell size" and extract_unique_domain(h) is domain
    assert h.element == as_element(("DG", 0), domain)


@pytest.mark.parametrize("name", ["", "a b"])
def test_invalid_terminal_names(domain, name):
    with pytest.raises(ComponentError):
        Field(name, domain, ("Lagrange", 1))
    with pytest.raises(ComponentError):
        Parameter(name, domain)
    with pytest.raises(ComponentError):
        KnownCoefficient(name, domain, ("DG", 0))


def test_as_expr(domain):
    c = Field("c", domain, ("Lagrange", 1))
    k = Parameter("k", domain)
    x = ufl.SpatialCoordinate(domain)
    assert as_expr(c) is c
    assert as_expr(k) is k
    assert as_expr(x) is x
    assert as_expr(2.0) == ufl.as_ufl(2.0)
    # A zero-dimensional array is a number
    assert as_expr(np.full((), -6.0)) == ufl.as_ufl(-6.0)
    assert as_expr([1.0, 2.0]).ufl_shape == (2,)
    assert as_expr(np.eye(2)).ufl_shape == (2, 2)
    with pytest.raises(TypeError):
        as_expr("velocity")


def test_parameter_names_may_be_qualified(domain):
    assert Parameter("solid.sigma", domain).name == "solid.sigma"
