"""Functionals: solving for their stationary point, and assembling them."""

import numpy as np
import pytest
import ufl

from sumfem.discrete import DomainMaps, discretize
from sumfem.solvers import Solver
from sumfem.symbolic import DirichletBC, Domain, Field, Functional


class PoissonEnergy(Functional):
    """``int 1/2 |grad u|^2 - f u dx``, stationary where ``-div(grad(u)) = f``.

    Args:
        name: Name of the functional.
        field: The field ``u``.
        f: The source ``f``.
    """

    def __init__(self, name, field, f):
        super().__init__(name)
        self.field, self.f = field, f

    def functional(self):
        u = self.field
        return (0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) - self.f * u) * u.domain.dx()


def test_the_stationary_point_of_an_energy_solves_its_equation(square, sides):
    """``u = 1 + x^2 + 2 y^2`` minimizes the energy with ``f = -6``, exactly in P2."""
    domain = Domain("cell", "triangle")
    u = Field("u", domain, ("Lagrange", 2))
    energy = PoissonEnergy("energy", u, -6.0)
    residual = energy.to_residual()
    disc = discretize([residual], DomainMaps({"cell": square()}))

    def exact(x):
        return 1 + x[0] ** 2 + 2 * x[1] ** 2

    Solver(disc[residual], [DirichletBC(u, sides, exact)]).solve()
    function = disc.function(u)
    expected = dolfinx_values(function, exact)
    np.testing.assert_allclose(function.x.array, expected, atol=1e-10)
    # int 1/2 (4 x^2 + 16 y^2) + 6 (1 + x^2 + 2 y^2) = 10/3 + 12
    assert disc.assemble_scalar(energy) == pytest.approx(46 / 3)


def dolfinx_values(function, expression):
    reference = function.copy()
    reference.interpolate(expression)
    return reference.x.array
