"""Jacobians as components: hand-written, derived, and as preconditioners."""

import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, discretize
from sumfem.solvers import Solver
from sumfem.symbolic import (
    ComponentError,
    Domain,
    Field,
    Jacobian,
    JacobianSum,
    Residual,
    build_system,
)


class MassMatrix(Jacobian):
    """``(w du, v)``.

    Args:
        name: Name of the Jacobian.
        field: The field of the test and trial functions.
        weight: The weight ``w``.
    """

    def __init__(self, name, field, weight=1.0):
        super().__init__(name)
        self.field, self.weight = field, weight

    def jacobian(self):
        u = self.field
        return self.weight * u.du * u.v * u.domain.dx()


class Cubic(Residual):
    """``(u^3, v)``, with a hand-written Jacobian twice the true one.

    Args:
        name: Name of the residual.
        field: The field ``u``.
    """

    def __init__(self, name, field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u = self.field
        return u**3 * u.v * u.domain.dx()

    def jacobian(self):
        u = self.field
        return 6 * u**2 * u.du * u.v * u.domain.dx()


def matrix(form):
    A = dolfinx.fem.petsc.assemble_matrix(form)
    A.assemble()
    return A


def same_matrix(a, b):
    A, B = matrix(a), matrix(b)
    A.axpy(-1.0, B)
    return A.norm() <= 1e-12 * max(B.norm(), 1.0)


def test_wrapping_keeps_hand_written_jacobians(toy, square):
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    residual = Cubic("cubic", u) + toy.Diffusion("diffusion", u)
    disc = discretize([residual], DomainMaps({"cell": square()}))
    disc.set({u: 1.0})
    plain = disc[residual]
    wrapped = disc[residual.with_preconditioner(MassMatrix("m", u))]
    assert same_matrix(wrapped.jacobian, plain.jacobian)
    # A given Jacobian replaces them, here by the derived one
    derived = disc[residual.with_jacobian(residual.to_jacobian())]
    assert not same_matrix(derived.jacobian, plain.jacobian)


def test_jacobians_sum_copy_and_are_checked(toy):
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    a = Field("a", domain, ("Lagrange", 1))
    mass = MassMatrix("m", u, weight=a) + MassMatrix("n", u)
    assert isinstance(mass, JacobianSum)
    assert mass.fields == (u, a)
    residual = toy.Diffusion("d", u).with_preconditioner(mass)
    system = build_system(residual)
    assert system.preconditioner is not None
    copy = residual.copy("_c")
    assert sorted(copy.parameters) == ["d_c.k"]
    (u_c,) = copy.equations
    assert u_c.name == "u_c"
    # The preconditioner of the copy acts on the new field
    assert {f.name for f in copy.preconditioner_term.fields} == {"u_c", "a"}

    class NotBilinear(Jacobian):
        def jacobian(self):
            return u * u.v * domain.dx()

    with pytest.raises(ComponentError, match="rank"):
        NotBilinear("bad").form
    with pytest.raises(TypeError):
        MassMatrix("m", u) + toy.Diffusion("d", u)


def test_a_preconditioned_solve_matches_the_plain_one(toy, square, sides):
    from sumfem.symbolic import DirichletBC

    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 1))
    residual = toy.Diffusion("d", u) + toy.Source("s", u, f=1.0)
    bcs = [DirichletBC(u, sides, 0.0)]
    options = {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12}
    solutions = []
    for coupling in (residual, residual.with_preconditioner(residual.to_jacobian())):
        disc = discretize([coupling], DomainMaps({"cell": square()}))
        Solver(disc[coupling], bcs, petsc_options=options).solve()
        solutions.append(disc.function(u).x.array.copy())
    np.testing.assert_allclose(solutions[0], solutions[1], atol=1e-10)


def test_derived_jacobians_need_a_field(domain):
    u = Field("u", domain, ("Lagrange", 1))

    class Bare(Residual):
        def residual(self):
            return u * u.v * domain.dx()

    with pytest.raises(ComponentError, match="at least one field"):
        Bare("b").to_jacobian([]).form
    assert ufl.algorithms.expand_derivatives(Bare("b").to_jacobian().form)
