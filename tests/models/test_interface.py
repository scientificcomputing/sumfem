"""Conditions across an interface between two subdomains."""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import (
    Diffusion,
    InterfaceLagrangeMultiplier,
    InterfacePenalty,
    InterfacePenaltyEnergy,
)
from sumfem.solvers import Solver
from sumfem.symbolic import ComponentError, DirichletBC, Domain, Field, Subdomain

TOLERANCES = {"snes_atol": 1e-12, "snes_rtol": 1e-12}


def halves(n: int = 8) -> MeshData:
    """The unit square: one material left of ``x = 1/2``, another right of it.

    The facets between them are the interface, ``"Interface"``.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    eps = 1e-10
    return MeshData.from_locators(
        mesh,
        {
            2: {
                "Left": lambda x: x[0] <= 0.5 + eps,
                "Right": lambda x: x[0] >= 0.5 - eps,
            },
            1: {
                "Interface": lambda x: np.isclose(x[0], 0.5),
                "Inlet": lambda x: np.isclose(x[0], 0.0),
                "Outlet": lambda x: np.isclose(x[0], 1.0),
            },
        },
    )


def test_a_sieverts_henry_interface_is_exact():
    """Hydrogen diffuses through a Sieverts material into a Henry material.

    The partial pressure is continuous, ``(c_0 / K_S)^2 = c_1 / K_H``, and so
    is the flux, the multiplier. With ``c_0(0) = a`` and ``c_1(1) = b`` both
    profiles are linear, with slopes ``s_0`` and ``s_1 = D_0 s_0 / D_1``, and
    ``y = c_0(1/2)`` solves a quadratic, so P1 is exact.
    """
    D_0, D_1, K_S, K_H, a, b = 2.0, 0.5, 1.5, 0.8, 3.0, 1.0
    cell = Domain("cell", "triangle")
    left = Subdomain("left", cell, 2, "Left")
    right = Subdomain("right", cell, 2, "Right")
    interface = Subdomain("interface", cell, 1, "Interface")
    c_0 = Field("c_0", left, ("Lagrange", 1))
    c_1 = Field("c_1", right, ("Lagrange", 1))
    flux = Field("flux", interface, ("Lagrange", 1))
    condition = InterfaceLagrangeMultiplier(
        "pressure",
        c_0,
        c_1,
        (c_0 / K_S) ** 2,
        c_1 / K_H,
        flux,
        cell.interface(left, right, "Interface"),
    )
    model = Diffusion("d_0", c_0, D_0) + Diffusion("d_1", c_1, D_1) + condition
    bcs = [DirichletBC(c_0, "Inlet", a), DirichletBC(c_1, "Outlet", b)]
    disc = discretize([model], DomainMaps({cell: halves()}))
    # A start where the pressure condition is nearly met
    disc.function(c_0).x.array[:] = a
    disc.function(c_1).x.array[:] = b
    Solver(disc[model], bcs, petsc_options=TOLERANCES).solve()

    # y^2 / K_S^2 + r y / K_H - (b + r a) / K_H = 0 with r = D_0 / D_1
    r = D_0 / D_1
    A, B, C = 1 / K_S**2, r / K_H, -(b + r * a) / K_H
    y = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    s_0 = 2 * (y - a)
    s_1 = r * s_0
    for field, exact, dx in (
        (c_0, lambda x: a + s_0 * x[0], left.dx()),
        (c_1, lambda x: b + s_1 * (x[0] - 1), right.dx()),
    ):
        x = ufl.SpatialCoordinate(field.domain)
        squared = disc.assemble_scalar((field - exact(x)) ** 2 * dx)
        assert np.sqrt(squared) < 1e-10
    # The multiplier is the flux from the first material to the second
    assert np.allclose(disc.function(flux).x.array, -D_0 * s_0, atol=1e-10)


def test_the_quantities_and_the_multiplier_must_be_scalars():
    """A vector quantity or multiplier is rejected when the model is created."""
    cell = Domain("cell", "triangle")
    left = Subdomain("left", cell, 2, "Left")
    right = Subdomain("right", cell, 2, "Right")
    interface = Subdomain("interface", cell, 1, "Interface")
    c_0 = Field("c_0", left, ("Lagrange", 1))
    c_1 = Field("c_1", right, ("Lagrange", 1))
    vector = Field("flux", interface, ("Lagrange", 1, (2,)))
    measure = cell.interface(left, right, "Interface")
    with pytest.raises(ValueError, match="multiplier of 'c' must be a scalar"):
        InterfaceLagrangeMultiplier("c", c_0, c_1, c_0, c_1, vector, measure)


D_BOTTOM, K_BOTTOM = 5.0, 6.0
D_TOP, K_TOP = 2.0, 3.0


def two_materials(coupling: str, alpha: float, a=None, b=None):
    """FESTIM's interface test, coupled by a penalty flux or energy.

    The unit square is split at ``y = 1/2``: the bottom has ``D = 5`` and
    solubility ``K = 6``, the top ``D = 2`` and ``K = 3``. The bottom wall is
    held at ``c = K``, the top wall at zero. The solution is piecewise linear
    in ``y``, so P1 elements give it exactly. The condition is
    ``c_0 / K_0 = c_1 / K_1`` unless the quantities ``a`` and ``b`` of the
    two sides are given as functions of the concentrations.

    Returns:
        The concentrations on both sides of the interface, and the flux that
        leaves the bottom and the one that enters the top.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    marked = MeshData.from_locators(
        mesh,
        {
            2: {
                "Bottom": lambda x: x[1] <= 0.5 + 1e-10,
                "Top": lambda x: x[1] >= 0.5 - 1e-10,
            },
            1: {
                "Interface": lambda x: np.isclose(x[1], 0.5),
                "BottomWall": lambda x: np.isclose(x[1], 0.0),
                "TopWall": lambda x: np.isclose(x[1], 1.0),
            },
        },
    )
    square = Domain("square", "triangle")
    bottom = Subdomain("bottom", square, 2, "Bottom")
    top = Subdomain("top", square, 2, "Top")
    c_0 = Field("c_bottom", bottom, ("Lagrange", 1))
    c_1 = Field("c_top", top, ("Lagrange", 1))
    dS = square.interface(bottom, top, "Interface")
    first = c_0 / K_BOTTOM if a is None else a(c_0)
    second = c_1 / K_TOP if b is None else b(c_1)
    if coupling == "flux":
        interface = InterfacePenalty("interface", c_0, c_1, first, second, alpha, dS)
    else:
        energy = InterfacePenaltyEnergy("interface", c_0, c_1, first, second, alpha, dS)
        interface = energy.to_residual([c_0, c_1])
    model = (
        Diffusion("bottom_diffusion", c_0, coefficient=D_BOTTOM)
        + Diffusion("top_diffusion", c_1, coefficient=D_TOP)
        + interface
    )
    disc = discretize([model], DomainMaps({"square": marked}))
    bcs = [DirichletBC(c_0, "BottomWall", K_BOTTOM), DirichletBC(c_1, "TopWall", 0.0)]
    Solver(disc[model], bcs, petsc_options=TOLERANCES).solve()
    c_bottom = disc.assemble_scalar(c_0("+") * dS)
    c_top = disc.assemble_scalar(c_1("-") * dS)
    leaving = D_BOTTOM * (K_BOTTOM - c_bottom) / 0.5
    entering = D_TOP * c_top / 0.5
    return c_bottom, c_top, leaving, entering


def test_penalty_coupling_gives_the_analytic_jump():
    """A conductance of 40: ``J = 40 (c_0 / 6 - c_1 / 3) = 8``, and the flux
    that leaves the bottom enters the top."""
    result = two_materials("flux", 40.0)
    np.testing.assert_allclose(result, [26 / 5, 2.0, 8.0, 8.0], rtol=1e-9)


def test_penalty_energy_gives_its_variation():
    """``alpha (a - b) (v_0 / K_0 - v_1 / K_1)``: the flux that enters the top
    is ``K_0 / K_1`` times the one that leaves the bottom, so the energy does
    not conserve the quantity for different solubilities."""
    alpha = 40.0
    c_bottom, c_top, leaving, entering = two_materials("energy", alpha)
    # The two fluxes through the walls balance the penalty terms of each side
    matrix = np.array(
        [
            [D_BOTTOM / 0.5 + alpha / K_BOTTOM**2, -alpha / (K_BOTTOM * K_TOP)],
            [-alpha / (K_BOTTOM * K_TOP), D_TOP / 0.5 + alpha / K_TOP**2],
        ]
    )
    expected = np.linalg.solve(matrix, [D_BOTTOM * K_BOTTOM / 0.5, 0.0])
    np.testing.assert_allclose([c_bottom, c_top], expected, rtol=1e-9)
    assert entering == pytest.approx(K_BOTTOM / K_TOP * leaving, rel=1e-9)


def test_penalty_flux_and_energy_agree_for_equal_derivatives():
    """For ``a = c_0`` and ``b = c_1``, the variation of the energy is the
    penalty flux on both sides."""
    flux = two_materials("flux", 40.0, a=lambda c: c, b=lambda c: c)
    energy = two_materials("energy", 40.0, a=lambda c: c, b=lambda c: c)
    np.testing.assert_allclose(energy, flux, rtol=1e-10)


def test_interface_sides_must_be_subdomains_of_cells():
    square = Domain("square", "triangle")
    bottom = Subdomain("bottom", square, 2, "Bottom")
    wall = Subdomain("wall", square, 1, "BottomWall")
    with pytest.raises(ComponentError, match="subdomain of cells"):
        square.interface(bottom, wall, "Interface")
