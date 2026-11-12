"""An electrolyte of ions: the ions as given, what the electrolyte derives
from them, and the models of a vanadium electrode built on it."""

import pytest
import sympy.physics.units as syu
import ufl

from sumfem.discrete import DomainMaps, discretize
from sumfem.electrochemistry import (
    Electrolyte,
    ElectrolyteConduction,
    Ion,
    MassTransferButlerVolmer,
    NernstPlanck,
    PorousElectrode,
    Region,
    TransientNernstPlanck,
)
from sumfem.models import Accumulation
from sumfem.symbolic import Domain, Field, build_system

CONCENTRATION = syu.mol / syu.m**3

VALUES = {"V2": 1.0, "V3": 2.0, "H": 3.0}
"""Uniform concentrations of the transported ions, in mol/m^3."""


def vanadium(domain):
    """The ions of a negative half-cell, sulfate by electroneutrality."""
    c2, c3, h = (
        Field(name, domain, ("Lagrange", 1), unit=CONCENTRATION)
        for name in ("c2", "c3", "h")
    )
    return [
        Ion("V2", 2, 2.4e-10, c2),
        Ion("V3", 3, 2.4e-10, c3),
        Ion("H", 1, 9.3e-9, h),
        Ion("SO4", -2, 1.065e-9),
    ]


def potentials(domain):
    """The solid and the electrolyte potential."""
    return (
        Field(name, domain, ("Lagrange", 1), unit=syu.volt)
        for name in ("phi_s", "phi_l")
    )


def mean(disc, domain, expression):
    """The mean of an expression over the unit square."""
    return disc.assemble_scalar(expression * domain.dx())


def uniform(domain, square, values):
    """The electrolyte, discretized with the given uniform concentrations."""
    ions = vanadium(domain)
    electrolyte = Electrolyte("e", domain, ions)
    _, phi_l = potentials(domain)
    # The conduction holds the parameters of every ion
    model = ElectrolyteConduction("charge", phi_l, electrolyte)
    for ion in ions[:3]:
        model += NernstPlanck(ion.name, ion, electrolyte, phi_l)
    disc = discretize([model], DomainMaps({domain: square()}))
    for ion in ions[:3]:
        disc.function(ion.field).x.array[:] = values[ion.name]
    return electrolyte, disc


def test_the_electrolyte_keeps_the_ions_as_given(domain):
    ions = vanadium(domain)
    electrolyte = Electrolyte("e", domain, ions)
    assert list(electrolyte.ions) == [ion.name for ion in ions]
    assert all(electrolyte[ion.name] is ion for ion in ions)
    assert electrolyte["SO4"].concentration is None


def test_the_diffusivity_is_a_parameter_of_the_electrolyte(domain):
    electrolyte = Electrolyte("e", domain, vanadium(domain))
    diffusivity = electrolyte.diffusivity("H")
    assert diffusivity.name == "e.D_H"
    assert diffusivity.default_si == pytest.approx(9.3e-9)
    assert diffusivity is electrolyte.diffusivity("H")


def test_the_balanced_ion_follows_from_electroneutrality(domain, square):
    electrolyte, disc = uniform(domain, square, VALUES)
    charge = sum(electrolyte[n].charge * c for n, c in VALUES.items())
    sulfate = mean(disc, domain, electrolyte.concentration("SO4"))
    assert sulfate == pytest.approx(-charge / electrolyte["SO4"].charge)


def test_concentrations_keep_their_sign(domain, square):
    """The electrolyte gives a concentration that undershoots zero as it is,
    and the balanced one from it."""
    electrolyte, disc = uniform(domain, square, {**VALUES, "V2": -1.0})
    assert mean(disc, domain, electrolyte.concentration("V2")) == pytest.approx(-1.0)
    sulfate = (2 * -1.0 + 3 * VALUES["V3"] + VALUES["H"]) / 2
    assert mean(disc, domain, electrolyte.concentration("SO4")) == pytest.approx(
        sulfate
    )


def test_the_current_is_the_charge_the_ions_carry(domain, square):
    """The electrolyte current equals ``F sum_j z_j N_j``, with the
    Nernst-Planck fluxes of every ion, sulfate's from electroneutrality, in
    a porous region with advection, also where V2+ undershoots zero.

    A regression test: the conductivity once took the positive part of the
    vanadium concentrations while migration did not, so conduction did not
    carry the charge the ions moved where a concentration was negative.
    """
    ions = vanadium(domain)
    electrolyte = Electrolyte("e", domain, ions)
    _, phi_l = potentials(domain)
    region = Region(porosity=0.6)
    velocity = ufl.as_vector((1e-3, 2e-3))
    transport = [
        NernstPlanck(ion.name, ion, electrolyte, phi_l, region, velocity)
        for ion in ions[:3]
    ]
    model = ElectrolyteConduction("charge", phi_l, electrolyte, region)
    for summand in transport:
        model += summand
    disc = discretize([model], DomainMaps({domain: square()}))
    profiles = {
        ions[0].field: lambda x: -1.0 + 3.0 * x[0],
        ions[1].field: lambda x: 2.0 + x[1],
        ions[2].field: lambda x: 3.0 + x[0] * x[1],
        phi_l: lambda x: x[0] ** 2 + 0.5 * x[1],
    }
    for field, profile in profiles.items():
        disc.function(field).interpolate(profile)

    e, sulfate = electrolyte, ions[3]
    c = e.concentration(sulfate.name)
    D = region.transport * e.diffusivity(sulfate.name)
    migration = D * sulfate.charge * e.F * c / (e.R * e.T) * ufl.grad(phi_l)
    fluxes = [m.flux() for m in transport]
    fluxes.append(-D * ufl.grad(c) - migration + velocity * c)
    charges = [e.F * ion.charge * N for ion, N in zip(ions, fluxes)]
    current = e.current_density(phi_l, region.transport)

    # The charge fluxes of the ions largely cancel, so the round-off is
    # relative to them, not to the current
    difference = current - sum(charges)
    error = mean(disc, domain, ufl.inner(difference, difference))
    size = sum(mean(disc, domain, ufl.inner(q, q)) for q in charges)
    assert mean(disc, domain, ufl.inner(current, current)) > 0
    assert error <= 1e-24 * size


@pytest.mark.parametrize(
    "ions, message",
    [
        (lambda c: [Ion("H", 1, 1.0, c), Ion("H", 1, 1.0, c)], "unique"),
        (
            lambda c: [Ion("H", 1, 1.0, c), Ion("A", -1, 1.0), Ion("B", -1, 1.0)],
            "At most",
        ),
        (lambda c: [Ion("A", -1, 1.0)], "at least one"),
    ],
)
def test_invalid_ions_are_rejected(domain, ions, message):
    c = Field("c", domain, ("Lagrange", 1), unit=CONCENTRATION)
    with pytest.raises(ValueError, match=message):
        Electrolyte("e", domain, ions(c))


def test_transport_needs_a_transported_ion_of_the_electrolyte(domain):
    ions = vanadium(domain)
    electrolyte = Electrolyte("e", domain, ions)
    _, phi_l = potentials(domain)
    with pytest.raises(TypeError, match="electroneutrality"):
        NernstPlanck("SO4", ions[3], electrolyte, phi_l)
    with pytest.raises(TypeError, match="electroneutrality"):
        ions[3].field
    stranger = Ion("V2", 2, 2.4e-10, ions[0].field)
    with pytest.raises(ValueError, match="not in"):
        NernstPlanck("V2", stranger, electrolyte, phi_l)


def test_transient_transport_is_storage_and_transport(domain):
    """The balance of an ion is its storage, weighted with the porosity, and
    its Nernst-Planck transport, and needs a transported ion."""
    ions = vanadium(domain)
    electrolyte = Electrolyte("e", domain, ions)
    _, phi_l = potentials(domain)
    region = Region(("Electrode",), porosity=0.7)
    balance = TransientNernstPlanck("V2", ions[0], electrolyte, phi_l, region)
    storage, transport = balance.components
    assert [m.name for m in balance.components] == ["V2_storage", "V2_transport"]
    assert isinstance(storage, Accumulation) and isinstance(transport, NernstPlanck)
    assert balance.ion is ions[0]
    with pytest.raises(TypeError, match="electroneutrality"):
        TransientNernstPlanck("SO4", ions[3], electrolyte, phi_l)


def test_a_vanadium_electrode_has_consistent_units(domain):
    ions = vanadium(domain)
    electrolyte = Electrolyte("e", domain, ions, density=1e3, viscosity=1e-3)
    electrode = PorousElectrode(
        "electrode", domain, porosity=0.9, specific_area=1e4, fiber_diameter=1e-5
    )
    phi_s, phi_l = potentials(domain)
    u = Field("u", domain, ("Lagrange", 1, (2,)), unit=syu.m / syu.second)
    region = electrode.region
    model = ElectrolyteConduction("charge", phi_l, electrolyte, region)
    for ion in ions[:3]:
        model += NernstPlanck(ion.name, ion, electrolyte, phi_l, region, velocity=u)
    model += MassTransferButlerVolmer(
        "reaction",
        phi_s,
        phi_l,
        electrolyte,
        "V2",
        "V3",
        velocity=u,
        specific_area=electrode.specific_area,
        fiber_diameter=electrode.fiber_diameter,
    )
    system = build_system(model, check_units=True)
    assert {f.name for f in system.unknowns} >= {"phi_s", "phi_l", "c2", "c3", "h"}


def test_the_balanced_ion_needs_a_charge():
    domain = Domain("cell", "interval")
    c = Field("c", domain, ("Lagrange", 1))
    with pytest.raises(ValueError, match="has no charge"):
        Electrolyte("e", domain, [Ion("A", 1, 1e-9, c), Ion("N", 0, 1e-9)])
