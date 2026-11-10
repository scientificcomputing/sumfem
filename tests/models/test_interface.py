"""Conditions across an interface between two subdomains."""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import Diffusion, InterfaceConstraint
from sumfem.solvers import Solver
from sumfem.symbolic import DirichletBC, Domain, Field, Subdomain

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
    condition = InterfaceConstraint(
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
        InterfaceConstraint("c", c_0, c_1, c_0, c_1, vector, measure)
