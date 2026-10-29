"""Lifetime of the PETSc objects of solvers."""

import gc
import weakref

import pytest
import ufl

from sumfem.discrete import DomainMaps, discretize
from sumfem.solvers import Solver
from sumfem.symbolic import DirichletBC, Domain, Field


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
