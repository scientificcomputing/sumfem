"""The proton-exchange membrane of Bernardi and Verbrugge, with Schlögl's flow."""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl
from basix.ufl import element

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.electrochemistry import ProtonExchangeMembrane
from sumfem.solvers import BDF, Solver
from sumfem.symbolic import (
    FARADAY,
    GAS_CONSTANT,
    DirichletBC,
    Domain,
    Field,
    ResidualSum,
    UFLResidual,
    build_system,
)
from sumfem.symbolic.si import to_si

CELLS = ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"]

DOMAIN = "box"
"""The name of the symbolic domain of both formulations, see ``compare``."""

VALUES = {
    "fixed_charge": 1200.0,
    "proton_diffusivity": 1.4e-9,
    "water_diffusivity": 5.75e-10,
    "electrokinetic_permeability": 1.13e-19,
    "hydraulic_permeability": 1.58e-18,
    "viscosity": 1e-3,
}
"""The membrane of Shah et al. (2008), Tables 4 and 5."""

F = float(to_si(FARADAY, syu.coulomb / syu.mol))
R = float(to_si(GAS_CONSTANT, syu.joule / (syu.mol * syu.kelvin)))
T = 298.15


def create_steady_membrane(model):
    """The membrane without the storage of water."""
    return ResidualSum(
        [m for m in model.components if not m.name.endswith("storage")],
        name="steady",
    )


def create_membrane(domain, water=True, pressure=True):
    phi = Field("phi", domain, ("Lagrange", 1), unit=syu.volt)
    c_w = Field("c_w", domain, ("Lagrange", 1), unit=syu.mol / syu.m**3, scale=4e4)
    p = Field("p", domain, ("Lagrange", 1), unit=syu.pascal, scale=1e3)
    model = ProtonExchangeMembrane(
        "membrane",
        phi,
        (),
        water=c_w if water else None,
        pressure=p if pressure else None,
        **VALUES,
    )
    return model, phi, c_w, p


@pytest.mark.parametrize("cell", CELLS)
def test_the_membrane_against_plain_ufl(cell, box, compare):
    """Ohmic conduction with the conductivity of the fixed protons, Schlögl's
    flow and the transport of water by it."""
    gdim = ufl.Cell(cell).topological_dimension
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(gdim,)))
    V = ufl.FunctionSpace(mesh, element("Lagrange", cell, 1))
    W = ufl.MixedFunctionSpace(V, V, V)
    phi, c_w, p = (ufl.Coefficient(V) for _ in range(3))
    v_phi, w, q = ufl.TestFunctions(W)
    Fc, Rc, Tc, cf, DH, Dw, kphi, kp, mu = (ufl.Constant(mesh) for _ in range(9))
    kappa = Fc**2 / (Rc * Tc) * DH * cf
    v = -kphi / mu * Fc * cf * ufl.grad(phi) - kp / mu * ufl.grad(p)
    residual = (
        kappa * ufl.inner(ufl.grad(phi), ufl.grad(v_phi))
        - ufl.inner(v, ufl.grad(q))
        - ufl.inner(-Dw * ufl.grad(c_w) + v * c_w, ufl.grad(w))
    ) * ufl.dx
    names = {
        DOMAIN: mesh,
        **dict(phi=phi, c_w=c_w, p=p),
        **dict(F=Fc, R=Rc, T=Tc, cf=cf, DH=DH, Dw=Dw, kphi=kphi, kp=kp, mu=mu),
    }
    plain = UFLResidual("plain", residual, [phi, c_w, p], names=names)

    domain = Domain(DOMAIN, cell)
    model, *fields = create_membrane(domain)
    values = {
        "plain.F": F,
        "plain.R": R,
        "plain.T": T,
        "plain.cf": VALUES["fixed_charge"],
        "plain.DH": VALUES["proton_diffusivity"],
        "plain.Dw": VALUES["water_diffusivity"],
        "plain.kphi": VALUES["electrokinetic_permeability"],
        "plain.kp": VALUES["hydraulic_permeability"],
        "plain.mu": VALUES["viscosity"],
    }
    compare(
        plain,
        plain.unknowns,
        create_steady_membrane(model),
        fields,
        box(cell),
        [(values, {})],
    )


def test_the_membrane_has_consistent_units():
    domain = Domain("cell", "interval")
    model, *_ = create_membrane(domain)
    build_system(model, check_units=True)
    build_system(model, substitutions=BDF(1).substitutions(model), check_units=True)
    names = [m.name for m in model.components]
    assert names == [
        "membrane_conduction",
        "membrane_flow",
        "membrane_water_storage",
        "membrane_water_diffusion",
        "membrane_water_advection",
    ]
    still, *_ = create_membrane(domain, pressure=False)
    assert [m.name for m in still.components] == [
        "membrane_conduction",
        "membrane_water_storage",
        "membrane_water_diffusion",
    ]
    bare, *_ = create_membrane(domain, water=False, pressure=False)
    assert [m.name for m in bare.components] == ["membrane_conduction"]


def test_a_membrane_between_two_potentials_and_pressures():
    """Across a membrane of thickness L, with the potential, the pressure and
    the water held at both faces, the potential and the pressure are linear,
    so the current density and Schlögl's velocity are uniform, and the water
    follows, at the nodes, the steady advection-diffusion profile
    ``c(x) = c_0 + (c_L - c_0) (exp(v x / D) - 1) / (exp(v L / D) - 1)``.
    """
    L, cells = 180e-6, 400
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, cells, [0.0, L])
    marked = MeshData.from_locators(
        mesh,
        {
            0: {
                "Negative": lambda x: np.isclose(x[0], 0.0),
                "Positive": lambda x: np.isclose(x[0], L),
            }
        },
        boundary=True,
    )
    domain = Domain.from_mesh("membrane", mesh, length_scale=L)
    model, phi, c_w, p = create_membrane(domain)
    dphi, dp, c0, cL = 0.01, -5e4, 4.2e4, 4.1e4
    bcs = [
        DirichletBC(phi, "Negative", 0.0),
        DirichletBC(phi, "Positive", dphi),
        DirichletBC(p, "Negative", 0.0),
        DirichletBC(p, "Positive", dp),
        DirichletBC(c_w, "Negative", c0),
        DirichletBC(c_w, "Positive", cL),
    ]
    # The pressure's equation is some fifteen orders of magnitude smaller
    # than the potential's, so the equations are solved in dimensionless form
    system = build_system(
        create_steady_membrane(model), check_units=True, factorize_equations=True
    )
    disc = discretize([system, *bcs], DomainMaps({domain: marked}))
    Solver(disc[system], bcs).solve()

    kappa = F**2 / (R * T) * VALUES["proton_diffusivity"] * VALUES["fixed_charge"]
    mu = VALUES["viscosity"]
    v = (
        -VALUES["electrokinetic_permeability"] / mu * F * VALUES["fixed_charge"] * dphi
        - VALUES["hydraulic_permeability"] / mu * dp
    ) / L
    current = disc.assemble_scalar(-kappa * phi.dx(0) * domain.dx()) / L
    assert current == pytest.approx(-kappa * dphi / L, rel=1e-10)
    velocity = disc.assemble_scalar(model.velocity[0] * domain.dx()) / L
    assert velocity == pytest.approx(v, rel=1e-10)

    # Linear elements are as good as exact at the nodes of this 1D problem;
    # compared there, the exponential profile needs no quadrature
    D = VALUES["water_diffusivity"]
    water = disc.function(c_w)
    owned = water.function_space.dofmap.index_map.size_local
    x = water.function_space.tabulate_dof_coordinates()[:owned, 0]
    exact = c0 + (cL - c0) * np.expm1(v * x / D) / np.expm1(v * L / D)
    error = np.abs(water.x.array[:owned] - exact).max(initial=0.0)
    assert mesh.comm.allreduce(error, op=MPI.MAX) < 1e-9 * abs(cL - c0)


def test_the_membrane_holds_the_protons_at_its_faces():
    domain = Domain("cell", "interval")
    model, *_ = create_membrane(domain)
    protons = [Field(f"c_H_{side}", domain, ("Lagrange", 1)) for side in "ab"]
    bcs = model.proton_conditions(protons, ("Negative", "Positive"))
    assert [bc.field for bc in bcs] == protons
    assert all(bc.value is model.fixed_charge for bc in bcs)
