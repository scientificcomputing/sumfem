"""Convergence of Brinkman-Forchheimer flow to a manufactured solution."""

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import sympy.physics.units as syu
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import (
    Brinkman,
    BrinkmanForchheimer,
    GradDiv,
    NavierStokes,
    Source,
    Stokes,
)
from sumfem.post_processing import integrate, norm
from sumfem.solvers import Solver
from sumfem.symbolic import DirichletBC, Domain, Field, Residual, build_system


def test_brinkman_forchheimer_converges():
    """The solution of Yoon and Mallikarjunaiah, doi:10.48550/arXiv.2501.04041."""
    domain = Domain("square", "quadrilateral")
    u = Field("u", domain, ("Lagrange", 2, (2,)))
    p = Field("p", domain, ("Lagrange", 1))
    porosity, viscosity, density, permeability, drag = 0.2, 1.0, 1.0, 1e-2, 5.0

    def exact(x):
        velocity = ufl.as_vector(
            (ufl.sin(ufl.pi * x[0]), -ufl.pi * x[1] * ufl.cos(ufl.pi * x[0]))
        )
        return velocity, ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    u_e, p_e = exact(ufl.SpatialCoordinate(domain))
    # The strong form of the model, applied to the exact solution
    force = (
        density / porosity**2 * ufl.dot(ufl.grad(u_e), u_e)
        + ufl.grad(p_e)
        - viscosity / porosity * ufl.div(ufl.grad(u_e))
        + viscosity / permeability * u_e
        + density * drag * ufl.sqrt(ufl.dot(u_e, u_e)) * u_e
    )
    model = BrinkmanForchheimer(
        "flow", u, p, porosity, permeability, drag, viscosity, density
    ) + Source("force", u, force)

    sizes = np.array([4, 8, 16, 32])
    errors = np.zeros((len(sizes), 2))
    for i, n in enumerate(sizes):
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral
        )
        marked = MeshData.from_locators(
            mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
        )
        disc = discretize([model], DomainMaps({"square": marked}))
        Solver(disc[model], [DirichletBC(u, "Boundary", u_e)]).solve()
        u_h, p_h = disc.function(u), disc.function(p)
        velocity, pressure = exact(ufl.SpatialCoordinate(mesh))
        # The pressure is unique up to a constant, so compare without means
        volume = integrate(1.0, mesh)
        shift = (integrate(p_h) - integrate(pressure)) / volume
        errors[i] = (
            norm(u_h - velocity),
            norm(p_h - shift - pressure),
        )
    rates = np.log2(errors[:-1] / errors[1:])
    # Taylor-Hood elements: third order in velocity, second in pressure
    assert rates[-1, 0] > 2.8
    assert rates[-1, 1] > 1.8


def test_stokes_converges():
    domain = Domain("square", "triangle")
    u = Field("u", domain, ("Lagrange", 2, (2,)))
    p = Field("p", domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    u_e = ufl.as_vector(
        (ufl.sin(ufl.pi * x[0]), -ufl.pi * x[1] * ufl.cos(ufl.pi * x[0]))
    )
    p_e = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    viscosity = 2.0
    force = -viscosity * ufl.div(ufl.grad(u_e)) + ufl.grad(p_e)
    model = Stokes("stokes", u, p, viscosity) + Source("force", u, force)
    errors = []
    for n in (8, 16, 32):
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
        marked = MeshData.from_locators(
            mesh, {1: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
        )
        disc = discretize([model], DomainMaps({"square": marked}))
        Solver(disc[model], [DirichletBC(u, "Boundary", u_e)]).solve()
        velocity = ufl.as_vector(
            (
                ufl.sin(ufl.pi * ufl.SpatialCoordinate(mesh)[0]),
                -ufl.pi
                * ufl.SpatialCoordinate(mesh)[1]
                * ufl.cos(ufl.pi * ufl.SpatialCoordinate(mesh)[0]),
            )
        )
        error = disc.function(u) - velocity
        errors.append(norm(error))
    assert (np.log2(np.array(errors[:-1]) / np.array(errors[1:])) > 2.8).all()


def test_flow_models_extend_stokes():
    """Each model adds its terms to the one it builds on."""
    domain = Domain("square", "triangle")
    u = Field("u", domain, ("Lagrange", 2, (2,)))
    p = Field("p", domain, ("Lagrange", 1))
    w = u.v

    class Extra(Residual):
        """A term added to the velocity equation."""

        def __init__(self, name, term):
            super().__init__(name)
            self.term = term

        def residual(self):
            return ufl.inner(self.term, w) * domain.dx()

    convection = ufl.dot(ufl.grad(u), u)
    pairs = [
        (
            NavierStokes("ns", u, p, 1.0, 3.0),
            Stokes("s", u, p, 1.0) + Extra("c", 3.0 * convection),
        ),
        (
            Brinkman("b", u, p, 0.5, 2.0, 1.0),
            Stokes("s", u, p, 1.0 / 0.5) + Extra("d", u / 2.0),
        ),
        (
            BrinkmanForchheimer("bf", u, p, 0.5, 2.0, 4.0, 1.0, 3.0, speed_floor=0.0),
            Brinkman("b", u, p, 0.5, 2.0, 1.0)
            + Extra(
                "f", 3.0 / 0.25 * convection + 3.0 * 4.0 * ufl.sqrt(ufl.dot(u, u)) * u
            ),
        ),
    ]
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    for model, expected in pairs:
        disc = discretize([model, expected], DomainMaps({"square": mesh}))
        disc.set({u: lambda x: (1 + x[0] * x[1], x[0] - x[1]), p: lambda x: x[0]})
        for a, b in zip(
            disc[model].residual,
            disc[expected].residual,
            strict=True,
        ):
            np.testing.assert_allclose(
                dolfinx.fem.assemble_vector(a).array,
                dolfinx.fem.assemble_vector(b).array,
                atol=1e-12,
            )


def test_named_flows_are_sums_with_consistent_units():
    """Each named flow sums its components under its name, whose parameters
    are named after it, and every equation of the sums is consistent in SI.
    """
    domain = Domain("square", "triangle")
    u = Field("u", domain, ("Lagrange", 2, (2,)), unit=syu.m / syu.s)
    p = Field("p", domain, ("Lagrange", 1), unit=syu.pascal)
    flows = {
        NavierStokes("ns", u, p): (
            ["ns_stokes", "ns_convection"],
            {"ns.viscosity", "ns.density"},
        ),
        Brinkman("b", u, p): (
            ["b_stokes", "b_drag"],
            {"b.porosity", "b.permeability", "b.viscosity"},
        ),
        BrinkmanForchheimer("bf", u, p): (
            ["bf_stokes", "bf_drag", "bf_convection", "bf_forchheimer"],
            {
                "bf.porosity",
                "bf.permeability",
                "bf.viscosity",
                "bf.density",
                "bf.forchheimer",
                "bf.speed_floor",
            },
        ),
    }
    for flow, (summands, parameters) in flows.items():
        assert [m.name for m in flow.components] == summands
        assert flow.velocity is u and flow.pressure is p
        system = build_system(flow + GradDiv("grad_div", u), check_units=True)
        assert parameters | {"grad_div.weight"} == set(system.parameters)
