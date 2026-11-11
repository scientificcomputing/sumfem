"""Solvers: the lifetime of their PETSc objects, and scaled unknowns."""

import gc
import weakref

from mpi4py import MPI

from petsc4py import PETSc

import dolfinx.mesh
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl

from sumfem.discrete import DomainMaps, MeshData, discretize
from sumfem.models import Diffusion, Reaction, Source, arrhenius
from sumfem.post_processing import error
from sumfem.solvers import Solver
from sumfem.symbolic import (
    DirichletBC,
    Domain,
    Field,
    Parameter,
    build_system,
)


@pytest.mark.parametrize("symbolic", [False, True])
def test_solver_is_freed_without_the_garbage_collector(toy, square, sides, symbolic):
    """A solver must not be part of a reference cycle.

    Destroying its PETSc objects is collective. The garbage collector runs
    at different times on different ranks, so a solver that only it can
    free deadlocks parallel runs, e.g. while one rank compiles a form.
    Symbolic boundary values install a callback on the SNES, which once
    referred back to the solver.
    """
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    value = x[0] if symbolic else 0.0
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=1.0)
    enabled = gc.isenabled()
    gc.disable()
    try:
        disc = discretize([model], DomainMaps({"cell": square()}))
        solver = Solver(disc[model], [DirichletBC(c, sides, value)])
        solver.solve()
        problem = weakref.ref(solver.nonlinear_problem)
        del solver
        assert problem() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize(
    "snes_type, line_search", [("newtonls", True), ("ksponly", False)]
)
def test_the_default_line_search_is_only_set_for_line_search_methods(
    snes_type, line_search, toy, square, sides
):
    """The default line search is set for a method that uses one only.
    PETSc would report it as an unused option otherwise, e.g. for ksponly.
    """
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=1.0)
    disc = discretize([model], DomainMaps({"cell": square()}))
    prefix = f"line_search_{snes_type}_"
    solver = Solver(
        disc[model],
        [DirichletBC(c, sides, 0.0)],
        petsc_options={"snes_type": snes_type},
        petsc_options_prefix=prefix,
    )
    solver.solve()
    assert PETSc.Options().hasName(f"{prefix}snes_linesearch_type") == line_search


def wall_problem(factorize: bool, kind: str | None, petsc_options: dict):
    """Heat and hydrogen in a wall, of very different magnitudes.

    The temperature is of 1e3 K, the hydrogen of 1e22 m^-3, and its
    diffusivity follows the temperature by the Arrhenius law. The unknowns
    start far from their hot Dirichlet values, at 300 K and no hydrogen.
    """
    K, W, m, s, J = syu.kelvin, syu.watt, syu.meter, syu.second, syu.joule
    thickness = 1e-2
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 50, [0.0, thickness])
    marked = MeshData.from_locators(
        mesh,
        {
            0: {
                "Hot": lambda x: np.isclose(x[0], 0.0),
                "Cold": lambda x: np.isclose(x[0], thickness),
            }
        },
    )
    wall = Domain.from_mesh("wall", mesh, length_scale=thickness)
    T = Field("T", wall, ("Lagrange", 2), unit=K, scale=1e3)
    c = Field("c", wall, ("Lagrange", 2), unit=1 / m**3, scale=1e22)
    kappa = Parameter("kappa", wall, default=400.0, unit=W / (m * K))
    sigma = Parameter("sigma", wall, default=1e-4, unit=W / (m**3 * K**4))
    D_0 = Parameter("D_0", wall, default=1e-7, unit=m**2 / s)
    E = Parameter("E", wall, default=3.2e-20, unit=J)
    model = (
        Diffusion("conduction", T, kappa)
        + Reaction("loss", {T: 4}, {}, sigma)
        + Diffusion("hydrogen", c, arrhenius(D_0, E, T))
    )
    bcs = [
        DirichletBC(T, "Hot", 1000.0),
        DirichletBC(T, "Cold", 300.0),
        DirichletBC(c, "Hot", 1e22),
        DirichletBC(c, "Cold", 0.0),
    ]
    system = build_system(model, check_units=True, factorize_equations=factorize)
    disc = discretize([system], DomainMaps({wall: marked}))
    solver = Solver(disc[system], bcs, petsc_options=petsc_options, kind=kind)
    disc.function(T).x.array[:] = 300.0
    solver.solve()
    return solver, disc.function(T).x.array, disc.function(c).x.array


NEST = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-14,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "fieldsplit_T_0_ksp_type": "preonly",
    "fieldsplit_T_0_pc_type": "lu",
    "fieldsplit_c_1_ksp_type": "preonly",
    "fieldsplit_c_1_pc_type": "lu",
}
"""Field-split LU of a nested matrix: the temperature does not depend on
the hydrogen, so the multiplicative split is exact."""


@pytest.mark.parametrize(
    "kind, options",
    [
        (None, {}),
        pytest.param(
            None,
            {"pc_factor_mat_solver_type": "petsc"},
            marks=pytest.mark.skipif(
                MPI.COMM_WORLD.size > 1, reason="PETSc's own LU is serial"
            ),
        ),
        ("nest", NEST),
    ],
    ids=["default-lu", "petsc-lu", "nest"],
)
def test_newton_iterates_on_scaled_unknowns(kind, options):
    """Factorized equations are solved for the unknowns over their scales.

    The constrained rows, ``x - g`` in SI, are divided by the scales too, so
    the residual of a start far from the Dirichlet values does not dominate
    the relative test. Once it did, and Newton stopped with the hydrogen 61 %
    off. The Jacobian's entries are of size one, which PETSc's own LU, with
    its absolute zero-pivot test, needs: it failed on the SI unknowns.
    """
    reference = {"snes_rtol": 0.0, "snes_atol": 0.0, "snes_stol": 0.0}
    reference |= {"snes_max_it": 12, "snes_error_if_not_converged": False}
    _, T_ref, c_ref = wall_problem(False, None, reference)
    solver, T_h, c_h = wall_problem(True, kind, options)
    assert solver.converged_to_tolerance
    assert np.max(np.abs(T_h - T_ref)) < 1e-8 * 1000.0
    assert np.max(np.abs(c_h - c_ref)) < 1e-8 * 1e22


def test_a_nanometre_domain_converges_when_scaled():
    """Nonlinear diffusion on a cube of a nanometre, from zero.

    In SI the rows of the diffusion are of some 1e-9 and those of the
    Dirichlet conditions of one volt, so the relative test, against a first
    residual of the Dirichlet rows, stops Newton far from the solution:
    unscaled, 6 % off. Scaled, every row is of size one, but only with the
    length scale of the domain: with the default of one metre, the size of
    the equation, which goes as the length in 3D, was 1e9 too large, its
    rows stayed of some 1e-9, and the solution was 49 % off.
    """
    L = 1e-9
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    mesh.geometry.x[:] *= L
    marked = MeshData.from_locators(
        mesh, {2: {"Boundary": lambda x: np.full(x.shape[1], True)}}, boundary=True
    )
    cube = Domain.from_mesh("cube", mesh, length_scale=L)
    u = Field("u", cube, ("Lagrange", 2), unit=syu.volt, scale=1.0)
    k = Parameter("k", cube, default=1.0, unit=syu.farad / syu.meter)
    length = Parameter("L", cube, default=L, unit=syu.meter)
    volt = Parameter("V", cube, default=1.0, unit=syu.volt)
    x = ufl.SpatialCoordinate(cube)
    # u = V (x^2 + 2 y^2) / L^2, written with parameters to keep its units
    u_exact = volt * ((x[0] / length) ** 2 + 2 * (x[1] / length) ** 2)
    source = -ufl.div(k * (1 + (u_exact / volt) ** 2) * ufl.grad(u_exact))
    model = Diffusion("d", u, k * (1 + (u / volt) ** 2)) + Source("s", u, source)

    def exact(x):
        return (x[0] / L) ** 2 + 2 * (x[1] / L) ** 2

    system = build_system(model, check_units=True, factorize_equations=True)
    disc = discretize([system], DomainMaps({cube: marked}))
    solver = Solver(disc[system], [DirichletBC(u, "Boundary", exact)])
    solver.solve()
    assert solver.converged_to_tolerance
    assert error(disc[u], exact, relative=True) < 1e-8
