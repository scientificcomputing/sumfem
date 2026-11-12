"""The models of sumfem.electrochemistry against their formulations in plain UFL.

Only the models that the demo ``examples/electrochemistry/porous_electrode.py``
does not compare with plain UFL: the conduction and transport of an
electrolyte of ions in a porous region, and the kinetics of a redox couple,
with and without mass transfer. Each test writes the terms as the UFL file of
a DOLFINx C++ demo would and compares the assembled residual, Jacobian and
preconditioner with the model's, at the same random values of the fields.
"""

import pytest
import ufl
from basix.ufl import element

from sumfem.electrochemistry import (
    ButlerVolmer,
    Electrolyte,
    ElectrolyteConduction,
    Ion,
    MassTransferButlerVolmer,
    NernstPlanck,
    PistonVelocityButlerVolmer,
    PorousElectrode,
)
from sumfem.symbolic import Domain, Field, Residual, UFLResidual

CELLS = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]
"""The cell types each comparison runs on."""

DOMAIN = "box"
"""The name of the symbolic domain of both formulations, see ``compare``."""


def constants(electrolyte_values):
    """The values of the electrolyte's parameters, for the plain constants."""
    return {f"plain.{name}": value for name, value in electrolyte_values.items()}


def positive_part(c, floor):
    """The electrolyte's smooth positive part ``(c + sqrt(c^2 + floor^2)) / 2``."""
    return 0.5 * (c + ufl.sqrt(c * c + floor * floor))


def velocity(gdim):
    """A uniform velocity, the same expression in both formulations."""
    return ufl.as_vector([1e-3 * (1.0 + i) for i in range(gdim)])


@pytest.mark.parametrize("cell", CELLS)
def test_conduction_and_transport_in_a_porous_region(cell, box, compare):
    """Conduction of an electrolyte of V2+, V3+ and H+, with SO4 2- by
    electroneutrality, and Nernst-Planck transport of the three, with
    advection of V2+, in the pores of an electrode on the left half.

    The conductivity takes the concentrations as they are, and every
    diffusivity Bruggeman's correction of the porosity.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(gdim,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, V, V, V)
    phi, c2, c3, h = (ufl.Coefficient(V) for _ in range(4))
    v_phi, w2, w3, wh = ufl.TestFunctions(W)
    F, R, T, eps, D2, D3, DH, DS = (ufl.Constant(mesh) for _ in range(8))
    sulfate = -(2 * c2 + 3 * c3 + h) / -2
    correction = eps**1.5
    conductivity = (
        F**2
        / (R * T)
        * correction
        * (4 * D2 * c2 + 9 * D3 * c3 + DH * h + 4 * DS * sulfate)
    )
    diffusion = (
        F
        * correction
        * (
            2 * D2 * ufl.grad(c2)
            + 3 * D3 * ufl.grad(c3)
            + DH * ufl.grad(h)
            - 2 * DS * ufl.grad(sulfate)
        )
    )
    current = -conductivity * ufl.grad(phi) - diffusion
    u = velocity(gdim)

    def flux(c, z, D, advected=False):
        D = correction * D
        N = -D * ufl.grad(c) - D * z * F * c / (R * T) * ufl.grad(phi)
        return N + u * c if advected else N

    dx = ufl.Measure("dx", mesh)(1)
    residual = (
        ufl.inner(-current, ufl.grad(v_phi)) * dx
        - ufl.inner(flux(c2, 2, D2, advected=True), ufl.grad(w2)) * dx
        - ufl.inner(flux(c3, 3, D3), ufl.grad(w3)) * dx
        - ufl.inner(flux(h, 1, DH), ufl.grad(wh)) * dx
    )
    names = {
        DOMAIN: mesh,
        **{n: c for n, c in zip(("phi", "c2", "c3", "h"), (phi, c2, c3, h))},
        **dict(F=F, R=R, T=T, eps=eps, D2=D2, D3=D3, DH=DH, DS=DS),
    }
    plain = UFLResidual(
        "plain",
        residual,
        [phi, c2, c3, h],
        names=names,
        markers={"dx": {1: "LeftHalf"}},
    )

    domain = Domain(DOMAIN, cell)
    potential = Field("phi", domain, ("Lagrange", 1))
    fields = [Field(name, domain, ("Lagrange", 1)) for name in ("c2", "c3", "h")]
    ions = [
        Ion("V2", 2, 2.4e-10, fields[0]),
        Ion("V3", 3, 2.4e-10, fields[1]),
        Ion("H", 1, 9.3e-9, fields[2]),
        Ion("SO4", -2, 1.065e-9),
    ]
    electrolyte = Electrolyte("e", domain, ions)
    region = PorousElectrode("el", domain, ("LeftHalf",), porosity=0.8).region
    model = ElectrolyteConduction("charge", potential, electrolyte, region)
    model += NernstPlanck("V2", ions[0], electrolyte, potential, region, velocity=u)
    for ion in ions[1:3]:
        model += NernstPlanck(ion.name, ion, electrolyte, potential, region)

    values = {
        "F": electrolyte.F.default_si,
        "R": electrolyte.R.default_si,
        "T": electrolyte.T.default_si,
        "eps": 0.8,
        "D2": 2.4e-10,
        "D3": 2.4e-10,
        "DH": 9.3e-9,
        "DS": 1.065e-9,
    }
    compare(
        plain,
        plain.unknowns,
        model,
        [potential, *fields],
        box(cell),
        [(constants(values), {})],
    )


@pytest.mark.parametrize("mass_transfer", [False, True])
@pytest.mark.parametrize("cell", CELLS)
def test_kinetics_of_a_redox_couple(cell, mass_transfer, box, compare):
    """V2+ <-> V3+ + e-, by Butler-Volmer kinetics from the positive parts of
    the concentrations, or at the surface concentrations that balance the
    reaction with mass transfer to fibers in a uniform flow.

    The current enters the equation of the solid potential and leaves that of
    the electrolyte; the reduced ion is consumed and the oxidized produced at
    the current over Faraday's constant.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(gdim,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, V, V, V)
    phi_s, phi_l, c2, c3 = (ufl.Coefficient(V) for _ in range(4))
    v_s, v_l, w2, w3 = ufl.TestFunctions(W)
    F, R, T, floor, a, k0, E0 = (ufl.Constant(mesh) for _ in range(7))
    d, rho, mu, D2, D3 = (ufl.Constant(mesh) for _ in range(5))
    factor = F / (R * T) * (phi_s - phi_l - E0)
    k_a, k_c = k0 * ufl.exp(0.5 * factor), k0 * ufl.exp(-0.5 * factor)
    c_red, c_ox = positive_part(c2, floor), positive_part(c3, floor)
    if mass_transfer:
        u = velocity(gdim)
        reynolds = rho * ufl.sqrt(ufl.dot(u, u) + 1e-16**2) * d / mu
        sherwood = 7.0 * reynolds**0.4
        m_red = sherwood * D2 / d + 1e-14
        m_ox = sherwood * D3 / d + 1e-14
        denominator = 1.0 + k_a / m_red + k_c / m_ox
        c_ox, c_red = (
            ((1.0 + k_a / m_red) * c_ox + k_a / m_ox * c_red) / denominator,
            ((1.0 + k_c / m_ox) * c_red + k_c / m_red * c_ox) / denominator,
        )
    f = a * F * (k_a * c_red - k_c * c_ox)
    rate = f / F
    residual = (f * v_s - f * v_l + rate * w2 - rate * w3) * ufl.dx
    names = {
        DOMAIN: mesh,
        **dict(phi_s=phi_s, phi_l=phi_l, c2=c2, c3=c3),
        **dict(F=F, R=R, T=T, floor=floor, a=a, k0=k0, E0=E0),
        **dict(d=d, rho=rho, mu=mu, D2=D2, D3=D3),
    }
    plain = UFLResidual("plain", residual, [phi_s, phi_l, c2, c3], names=names)

    domain = Domain(DOMAIN, cell)
    potentials = [Field(name, domain, ("Lagrange", 1)) for name in ("phi_s", "phi_l")]
    fields = [Field(name, domain, ("Lagrange", 1)) for name in ("c2", "c3")]
    ions = [
        Ion("V2", 2, 2.4e-10, fields[0]),
        Ion("V3", 3, 2.4e-10, fields[1]),
    ]
    electrolyte = Electrolyte(
        "e", domain, ions, density=1.3e3, viscosity=5e-3, concentration_floor=0.1
    )
    kinetics = dict(specific_area=1e3, rate_constant=1e-6, formal_potential=0.2)
    if mass_transfer:
        model = MassTransferButlerVolmer(
            "bv",
            *potentials,
            electrolyte,
            "V2",
            "V3",
            velocity(gdim),
            fiber_diameter=1e-5,
            **kinetics,
        )
    else:
        model = ButlerVolmer("bv", *potentials, electrolyte, "V2", "V3", **kinetics)

    values = {
        "F": electrolyte.F.default_si,
        "R": electrolyte.R.default_si,
        "T": electrolyte.T.default_si,
        "floor": 0.1,
        "a": 1e3,
        "k0": 1e-6,
        "E0": 0.2,
    }
    if mass_transfer:
        values.update(d=1e-5, rho=1.3e3, mu=5e-3, D2=2.4e-10, D3=2.4e-10)
    compare(
        plain,
        plain.unknowns,
        model,
        [*potentials, *fields],
        box(cell),
        [(constants(values), {})],
    )


@pytest.mark.parametrize("cell", CELLS)
def test_positive_couple_with_protons_water_and_piston_velocities(cell, box, compare):
    """VO2+ + H2O <-> VO2+ + 2 H+ + e- of Shah et al. (2008), limited by the
    piston velocities ``D / delta`` of each vanadium ion.

    The reaction produces two protons and consumes one water molecule per
    vanadium ion oxidized.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(gdim,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, V, V, V, V, V)
    phi_s, phi_l, c4, c5, ch, cw = (ufl.Coefficient(V) for _ in range(6))
    v_s, v_l, w4, w5, wh, ww = ufl.TestFunctions(W)
    F, R, T, floor, a, k0, E0 = (ufl.Constant(mesh) for _ in range(7))
    delta, D4, D5 = (ufl.Constant(mesh) for _ in range(3))
    factor = F / (R * T) * (phi_s - phi_l - E0)
    k_a, k_c = k0 * ufl.exp(0.5 * factor), k0 * ufl.exp(-0.5 * factor)
    c_red, c_ox = positive_part(c4, floor), positive_part(c5, floor)
    m_red = D4 / delta + 1e-14
    m_ox = D5 / delta + 1e-14
    denominator = 1.0 + k_a / m_red + k_c / m_ox
    c_ox, c_red = (
        ((1.0 + k_a / m_red) * c_ox + k_a / m_ox * c_red) / denominator,
        ((1.0 + k_c / m_ox) * c_red + k_c / m_red * c_ox) / denominator,
    )
    f = a * F * (k_a * c_red - k_c * c_ox)
    rate = f / F
    # A mass term for the protons and water, which only the reaction would
    # otherwise reach, so that each has a Jacobian block of its own
    residual = (
        f * v_s - f * v_l + rate * w4 - rate * w5 - 2 * rate * wh + rate * ww
    ) * ufl.dx + (ch * wh + cw * ww) * ufl.dx
    names = {
        DOMAIN: mesh,
        **dict(phi_s=phi_s, phi_l=phi_l, c4=c4, c5=c5, ch=ch, cw=cw),
        **dict(F=F, R=R, T=T, floor=floor, a=a, k0=k0, E0=E0),
        **dict(delta=delta, D4=D4, D5=D5),
    }
    unknowns = [phi_s, phi_l, c4, c5, ch, cw]
    plain = UFLResidual("plain", residual, unknowns, names=names)

    domain = Domain(DOMAIN, cell)
    potentials = [Field(name, domain, ("Lagrange", 1)) for name in ("phi_s", "phi_l")]
    fields = [Field(name, domain, ("Lagrange", 1)) for name in ("c4", "c5", "ch", "cw")]
    ions = [
        Ion("VO", 2, 3.9e-10, fields[0]),
        Ion("VO2", 1, 3.9e-10, fields[1]),
        Ion("H", 1, 9.3e-9, fields[2]),
        Ion("H2O", 0, 2.3e-9, fields[3]),
        Ion("SO4", -2, 1.1e-9),
    ]
    electrolyte = Electrolyte("e", domain, ions, concentration_floor=0.1)
    model = PistonVelocityButlerVolmer(
        "bv",
        *potentials,
        electrolyte,
        "VO",
        "VO2",
        fiber_distance=1e-5,
        specific_area=1e3,
        rate_constant=1e-6,
        formal_potential=1.0,
        reactants={"VO": 1, "H2O": 1},
        products={"VO2": 1, "H": 2},
    )

    class Mass(Residual):
        def residual(self):
            ch, cw = fields[2], fields[3]
            return (ch * ch.v + cw * cw.v) * domain.dx()

    model = model + Mass("mass")
    values = {
        "F": electrolyte.F.default_si,
        "R": electrolyte.R.default_si,
        "T": electrolyte.T.default_si,
        "floor": 0.1,
        "a": 1e3,
        "k0": 1e-6,
        "E0": 1.0,
        "delta": 1e-5,
        "D4": 3.9e-10,
        "D5": 3.9e-10,
    }
    compare(
        plain,
        plain.unknowns,
        model,
        [*potentials, *fields],
        box(cell),
        [(constants(values), {})],
    )


def test_the_reaction_needs_its_couple_and_ions_with_fields():
    domain = Domain(DOMAIN, "interval")
    potentials = [Field(name, domain, ("Lagrange", 1)) for name in ("phi_s", "phi_l")]
    c = [Field(name, domain, ("Lagrange", 1)) for name in ("c4", "c5")]
    ions = [
        Ion("VO", 2, 3.9e-10, c[0]),
        Ion("VO2", 1, 3.9e-10, c[1]),
        Ion("H", 1, 9.3e-9),
    ]
    electrolyte = Electrolyte("e", domain, ions)
    with pytest.raises(ValueError, match="electroneutrality"):
        ButlerVolmer(
            "bv",
            *potentials,
            electrolyte,
            "VO",
            "VO2",
            products={"VO2": 1, "H": 2},
        )
    with pytest.raises(ValueError, match="must hold the reduced ion"):
        ButlerVolmer("bv", *potentials, electrolyte, "VO", "VO2", reactants={"H": 1})


def test_mass_transfer_takes_the_properties_of_the_electrolyte():
    """Without a density and viscosity of its own, the mass transfer uses the
    electrolyte's, so models of one electrolyte share them."""
    domain = Domain(DOMAIN, "triangle")
    phi_s, phi_l, c2, c3 = (
        Field(name, domain, ("Lagrange", 1)) for name in ("phi_s", "phi_l", "c2", "c3")
    )
    ions = [Ion("V2", 2, 2.4e-10, c2), Ion("V3", 3, 2.4e-10, c3)]
    electrolyte = Electrolyte("e", domain, ions, density=1.3e3, viscosity=5e-3)
    model = MassTransferButlerVolmer(
        "bv", phi_s, phi_l, electrolyte, "V2", "V3", velocity(2)
    )
    coefficients = ufl.algorithms.extract_coefficients(model.reynolds_number())
    assert electrolyte.density in coefficients
    assert electrolyte.viscosity in coefficients
    names = {c.name for c in coefficients}
    assert not {"bv.density", "bv.viscosity"} & names
