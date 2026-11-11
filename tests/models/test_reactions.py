"""Reactions at any rate, and reactions across a membrane between two volumes."""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import Accumulation, Diffusion, RateReaction, Reaction
from sumfem.solvers import BDF, Simulation, Solver
from sumfem.symbolic import Clock, DirichletBC, Domain, Field, Subdomain

TOLERANCES = {"snes_atol": 1e-12, "snes_rtol": 1e-12}


def halves(n: int = 8) -> MeshData:
    """The unit square: the ER left of ``x = 1/2``, the cytosol right of it.

    The facets between them are the ER membrane, ``"ERm"``.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    eps = 1e-10
    return MeshData.from_locators(
        mesh,
        {
            2: {
                "ER": lambda x: x[0] <= 0.5 + eps,
                "Cyto": lambda x: x[0] >= 0.5 - eps,
            },
            1: {
                "ERm": lambda x: np.isclose(x[0], 0.5),
                "Left": lambda x: np.isclose(x[0], 0.0),
                "Right": lambda x: np.isclose(x[0], 1.0),
            },
        },
    )


def compartments():
    cell = Domain("cell", "triangle")
    er = Subdomain("er", cell, 2, "ER")
    cyto = Subdomain("cyto", cell, 2, "Cyto")
    membrane = Subdomain("membrane", cell, 1, "ERm")
    return cell, er, cyto, membrane


def test_a_rate_reaction_is_mass_action_with_its_rate():
    domain = Domain("cell", "triangle")
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    mass_action = Reaction("r", {a: 2}, {b: 1}, 3.0, 0.5)
    rate = RateReaction("r", {a: 2}, {b: 1}, 3.0 * a**2 - 0.5 * b)
    assert mass_action.form.signature() == rate.form.signature()
    with pytest.raises(ValueError, match="several domains"):
        RateReaction(
            "r", {Field("c", Domain("other", "triangle"), ("DG", 0)): 1}, {a: 1}, 1.0
        )


def test_release_across_a_membrane_is_exact():
    """``u_ER`` and ``u`` diffuse on either side, and receptors on the membrane
    release ``J = k R (u_ER - u)`` from the ER into the cytosol.

    With ``u_ER(0) = a`` and ``u(1) = b``, both profiles are linear with the
    slope ``s = -k R (a - b) / (D + k R)`` of the continuous flux, so P1 is
    exact.
    """
    D, k, R, a, b = 2.0, 3.0, 0.5, 1.0, 0.25
    cell, er, cyto, membrane = compartments()
    u_er = Field("u_er", er, ("Lagrange", 1))
    u = Field("u", cyto, ("Lagrange", 1))
    receptors = Field("receptors", membrane, ("Lagrange", 1))
    release = RateReaction(
        "release",
        {u_er: 1},
        {u: 1},
        k * receptors * (u_er - u),
        measure=cell.interface(er, cyto, "ERm"),
    )
    model = Diffusion("d_er", u_er, D) + Diffusion("d", u, D) + release
    bcs = [DirichletBC(u_er, "Left", a), DirichletBC(u, "Right", b)]
    disc = discretize([model, receptors], DomainMaps({cell: halves()}))
    disc.set({receptors: R})
    Solver(disc[model], bcs, petsc_options=TOLERANCES).solve()

    s = -k * R * (a - b) / (D + k * R)
    for field, exact, dx in (
        (u_er, lambda x: a + s * x[0], er.dx()),
        (u, lambda x: b + s * (x[0] - 1), cyto.dx()),
    ):
        x = ufl.SpatialCoordinate(field.domain)
        squared = disc.assemble_scalar((field - exact(x)) ** 2 * dx)
        assert np.sqrt(squared) < 1e-12


def test_membrane_reactions_conserve_molecules():
    """SMART's Example 5 on two halves: ``B + R <-> R_o`` at the ER membrane,
    and the release ``k R_o (A_ER - A)`` of ``A`` from the ER.

    Molecules of ``A`` in both volumes, and of the receptors, are conserved.
    """
    cell, er, cyto, membrane = compartments()
    A = Field("A", cyto, ("Lagrange", 1))
    B = Field("B", cyto, ("Lagrange", 1))
    A_er = Field("A_er", er, ("Lagrange", 1))
    R = Field("R", membrane, ("Lagrange", 1))
    R_o = Field("R_o", membrane, ("Lagrange", 1))
    interface = cell.interface(er, cyto, "ERm")
    model = (
        sum(
            (Accumulation(f"d{f.name}", f) for f in (A, B, A_er, R, R_o)),
            start=Diffusion("diffusion_A", A, 1.0),
        )
        + Diffusion("diffusion_B", B, 1.0)
        + Diffusion("diffusion_A_er", A_er, 5.0)
        + Diffusion("diffusion_R", R, 0.02)
        + Diffusion("diffusion_R_o", R_o, 0.02)
        + Reaction("opening", {B: 1, R: 1}, {R_o: 1}, 100.0, 100.0, measure=interface)
        + RateReaction(
            "release", {A_er: 1}, {A: 1}, 20.0 * R_o * (A_er - A), measure=interface
        )
    )
    clock = Clock(cell, dt=0.01)
    disc = discretize(model, DomainMaps({cell: halves()}))
    simulation = Simulation(
        disc, model, clock=clock, scheme=BDF(1), petsc_options=TOLERANCES
    )
    disc.set({A: 0.01, B: 1.0, A_er: 2.0, R: 1.0, R_o: 0.0})
    totals = (
        A * cyto.dx() + A_er * er.dx(),
        R * membrane.dx() + R_o * membrane.dx(),
        B * cyto.dx() + R_o * membrane.dx(),
    )
    initial = [disc.assemble_scalar(total) for total in totals]
    simulation.run(num_steps=10)
    for total, start in zip(totals, initial, strict=True):
        assert disc.assemble_scalar(total) == pytest.approx(start, rel=1e-10)
    # Receptors opened and A was released into the cytosol
    assert disc.assemble_scalar(R_o * membrane.dx()) > 0.01
    assert disc.assemble_scalar(A * cyto.dx()) > 0.01 * 0.5 * 1.01
