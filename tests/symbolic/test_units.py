"""Units on terminals, conversion to SI, and dimensional analysis."""

import numpy as np
import pytest
import sympy
import sympy.physics.units as syu
import ufl
from ufl.algorithms import expand_derivatives
from ufl_units import QuantityMixin

from sumfem.symbolic import (
    Domain,
    Field,
    KnownCoefficient,
    Parameter,
    Residual,
    UnitError,
    build_system,
    check_units,
    equation_dimensions,
    equation_scales,
    pi_groups,
)
from sumfem.symbolic.si import has_units, si_factor, to_si


class HeatConduction(Residual):
    """``-div(kappa grad T) = q``, with optional rho c_p dT/dt."""

    def __init__(self, name, T, source_unit=syu.W / syu.m**3):
        super().__init__(name)
        self.T, self.source_unit = T, source_unit

    def residual(self):
        domain = self.T.domain
        kappa = self.parameter(
            "kappa", domain, default=0.6, unit=syu.W / (syu.K * syu.m)
        )
        q = self.parameter("q", domain, default=1e3, unit=self.source_unit)
        dx = domain.dx()
        return (
            ufl.inner(kappa * ufl.grad(self.T), ufl.grad(self.T.v)) * dx
            - q * self.T.v * dx
        )


@pytest.fixture
def rod():
    return Domain("rod", "interval", length_scale=0.05)


def test_terminals_carry_units(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    assert T.unit == syu.K and T.scale == 10.0
    assert Field("phi", rod, ("Lagrange", 1)).unit == 1
    kappa = Parameter("kappa", rod, default=0.6, unit=syu.W / (syu.K * syu.m))
    assert isinstance(kappa, QuantityMixin)
    assert kappa.unit == syu.W / (syu.K * syu.m) and kappa.scale == 0.6
    # Without a non-zero default the scale for analysis is 1
    assert Parameter("E0", rod, default=0.0, unit=syu.V).scale == 1.0
    u = KnownCoefficient("u", rod, ("Lagrange", 1), unit=syu.m / syu.s, scale=1e-3)
    assert isinstance(u, QuantityMixin) and u.scale == 1e-3
    assert rod.length_scale == 0.05


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_scales_must_be_positive(rod, scale):
    with pytest.raises(ValueError):
        Field("T", rod, ("Lagrange", 1), scale=scale)
    with pytest.raises(ValueError):
        KnownCoefficient("u", rod, ("DG", 0), scale=scale)
    with pytest.raises(ValueError):
        Domain("d", "triangle", length_scale=scale)


def test_conversion_to_si():
    assert si_factor(None) == 1.0
    assert si_factor(syu.cm) == pytest.approx(0.01)
    assert si_factor(syu.mol / syu.liter) == pytest.approx(1e3)
    np.testing.assert_allclose(to_si([1.0, 2.0], syu.cm), [0.01, 0.02])
    assert has_units(5 * syu.cm) and not has_units(sympy.sympify(5))
    # A quantity with its own units is converted from them
    assert to_si(5 * syu.mm, syu.cm) == pytest.approx(0.005)
    with pytest.raises(ValueError, match="dimension"):
        to_si(5 * syu.second, syu.cm)


def test_consistent_equation_passes(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    system = build_system(HeatConduction("heat", T))
    check_units(system)
    dimensions = equation_dimensions(system)
    assert set(dimensions) == {"T"}


def test_inconsistent_equation_is_rejected(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    system = build_system(HeatConduction("heat", T, source_unit=syu.W))
    with pytest.raises(UnitError, match="equation for 'T'"):
        check_units(system)
    with pytest.raises(UnitError):
        equation_dimensions(system)


def test_equations_are_checked_separately(rod):
    # Two equations of different dimension in one system are fine
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    c = Field("c", rod, ("Lagrange", 1), unit=syu.mol / syu.m**3, scale=1e3)

    class Diffusion(Residual):
        def residual(self):
            D = self.parameter("D", rod, default=1e-9, unit=syu.m**2 / syu.s)
            r = self.parameter(
                "r", rod, default=1e-3, unit=syu.mol / (syu.m**3 * syu.s)
            )
            dx = rod.dx()
            return ufl.inner(D * ufl.grad(c), ufl.grad(c.v)) * dx - r * c.v * dx

    system = build_system(HeatConduction("heat", T) + Diffusion("diffusion"))
    check_units(system)
    dimensions = equation_dimensions(system)
    assert dimensions["T"] != dimensions["c"]


def test_time_derivatives_use_the_reference_time(toy, rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    rho_cp = Parameter("rho_cp", rod, default=4e6, unit=syu.J / (syu.K * syu.m**3))
    system = build_system(
        toy.Accumulation("dTdt", T, weight=rho_cp) + HeatConduction("heat", T)
    )
    check_units(system, time_scale=100.0)


def test_pi_groups(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    groups = pi_groups(build_system(HeatConduction("heat", T)))
    assert len(groups) == 1
    # The heat source number q L^2 / (kappa T_ref), up to an exponent
    (group,) = groups
    symbols = {str(s) for s in group.expression.free_symbols}
    assert symbols == {"heat.q", "heat.kappa", "L_rod", "T_ref"}
    expected = 1e3 * 0.05**2 / (0.6 * 10.0)
    assert group.value == pytest.approx(expected) or group.value == pytest.approx(
        1 / expected
    )


def test_equation_scales_use_the_largest_term(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    scales = equation_scales(build_system(HeatConduction("heat", T)))
    # kappa T_ref / L = 120 for the diffusion term, q L = 50 for the source
    assert scales == {"T": pytest.approx(120.0)}


def test_build_system_checks_units_on_request(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    model = HeatConduction("heat", T, source_unit=syu.W)
    build_system(model)
    with pytest.raises(UnitError, match="equation for 'T'"):
        build_system(model, check_units=True)
    with pytest.raises(UnitError):
        build_system(model, factorize_equations=True)


def test_factorized_system_divides_each_equation_by_its_size(rod):
    T = Field("T", rod, ("Lagrange", 1), unit=syu.K, scale=10.0)
    model = HeatConduction("heat", T)
    plain = build_system(model)
    system = build_system(model, factorize_equations=True)
    assert plain.equation_scales == {}
    assert system.equation_scales == {"T": pytest.approx(120.0)}
    # The unknown keeps its SI values, and only the equation is scaled
    assert system.unknowns == (T,)
    (test,) = plain.test_functions
    scaled = expand_derivatives(ufl.replace(plain.residual, {test: test / 120.0}))
    assert system.residual.signature() == scaled.signature()
