"""Toy models and meshes shared by the tests of the symbolic and discrete packages.

The tests use pytest's importlib import mode, so the models are handed to
tests through the ``toy`` fixture instead of being imported.
"""

import types

import pytest
import ufl

from sumfem.symbolic import Domain, Field, Residual


class Accumulation(Residual):
    """The term ``(w du/dt, v)``, with an optional weight ``w``."""

    def __init__(self, name: str, field: Field, weight=None):
        super().__init__(name)
        self.field = field
        self.weight = 1.0 if weight is None else weight

    def residual(self):
        u = self.field
        return ufl.inner(self.weight * u.dot, u.v) * u.domain.dx()


class Diffusion(Residual):
    """The term ``(k grad u, grad v)`` with a parameter ``k``."""

    def __init__(self, name: str, field: Field, markers: tuple[str, ...] = ()):
        super().__init__(name)
        self.field = field
        self.markers = markers

    def conductivity(self):
        return self.parameter("k", self.field.domain, default=1.0, doc="Conductivity")

    def residual(self):
        u, v = self.field, self.field.v
        dx = self.field.domain.dx(*self.markers)
        return ufl.inner(self.conductivity() * ufl.grad(u), ufl.grad(v)) * dx


class Source(Residual):
    """The term ``-(f, v)``, where ``f`` is an input."""

    def __init__(self, name: str, field: Field, f=None):
        super().__init__(name)
        self.field = field
        self.f = self.input(
            "f", f, domain=field.domain, shape=field.ufl_shape, default=0.0
        )

    def residual(self):
        return -ufl.inner(self.f, self.field.v) * self.field.domain.dx()


class Reaction(Residual):
    """Exchange ``r = j0 sinh(a - b)``: ``+r`` in a's equation, ``-r`` in b's."""

    def __init__(self, name: str, a: Field, b: Field, marker: str = "Electrode"):
        super().__init__(name)
        self.a, self.b, self.marker = a, b, marker

    def residual(self):
        j0 = self.parameter("j0", self.a.domain, default=1.0)
        rate = j0 * ufl.sinh(self.a - self.b)
        dx = self.a.domain.dx(self.marker)
        return rate * self.a.v * dx - rate * self.b.v * dx


class InteriorPenalty(Residual):
    """A DG interior-penalty Laplacian for a discontinuous field."""

    def __init__(self, name: str, field: Field):
        super().__init__(name)
        self.field = field

    def residual(self):
        u, v, domain = self.field, self.field.v, self.field.domain
        n = ufl.FacetNormal(domain)
        h = ufl.CellDiameter(domain)
        alpha = self.parameter("alpha", domain, default=10.0)
        dx, dS = domain.dx(), domain.dS()
        return (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            - ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * dS
            - ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * dS
            + alpha / ufl.avg(h) * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
        )


@pytest.fixture
def toy():
    """Namespace of toy model classes."""
    return types.SimpleNamespace(
        Accumulation=Accumulation,
        Diffusion=Diffusion,
        Source=Source,
        Reaction=Reaction,
        InteriorPenalty=InteriorPenalty,
    )


@pytest.fixture
def domain():
    """A triangle domain."""
    return Domain("cell", "triangle")
