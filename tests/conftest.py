"""Toy models and meshes shared by the tests of the symbolic and discrete packages.

The tests use pytest's importlib import mode, so the models are handed to
tests through the ``toy`` fixture instead of being imported.
"""

import gc
import types

from mpi4py import MPI

from petsc4py import PETSc

import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from basix.ufl import element

from sumfem.discrete import MeshData
from sumfem.symbolic import Domain, Field, Residual

SIDES = ("Left", "Right", "Bottom", "Top")


@pytest.fixture(autouse=True)
def problems_are_freed_by_reference_counting():
    """Fail a test that leaves a DOLFINx problem to the garbage collector.

    A problem destroys its PETSc objects when it is deleted, and destroying
    them is collective. A problem in a reference cycle is deleted by the
    garbage collector, which runs at different times on different
    processes, so a parallel run deadlocks, often in a later test. A test
    that replaces a method of a solver with a closure over the solver makes
    such a cycle; patch it with ``monkeypatch``, which breaks the cycle at
    teardown.
    """
    yield
    problems = (dolfinx.fem.petsc.NonlinearProblem, dolfinx.fem.petsc.LinearProblem)
    left = sum(isinstance(o, problems) for o in gc.get_objects())
    gc.collect()
    if left:
        pytest.fail(f"{left} DOLFINx problem(s) left to the garbage collector.")


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


def create_unit_box(cell: str = "triangle", n: int = 4) -> MeshData:
    """A unit interval, square or cube with markers for sides and halves.

    The cell type decides the dimension. The cells are marked ``"LeftHalf"``
    and ``"RightHalf"``, and the sides normal to x ``"Left"`` and ``"Right"``.
    A square or cube also has the sides normal to y, ``"Bottom"`` and
    ``"Top"``, so ``SIDES`` are all sides of a square and four of a cube.
    ``markers[tdim - 1]`` of the result names the sides marked.
    """
    cell_type = dolfinx.mesh.to_type(cell)
    tdim = dolfinx.mesh.cell_dim(cell_type)
    if tdim == 1:
        mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, n)
    elif tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type)
    eps = 1e-10
    sides = {
        "Left": lambda x: np.isclose(x[0], 0.0),
        "Right": lambda x: np.isclose(x[0], 1.0),
    }
    if tdim > 1:
        # On an interval x[1] is 0 everywhere, so these would mark both ends
        sides["Bottom"] = lambda x: np.isclose(x[1], 0.0)
        sides["Top"] = lambda x: np.isclose(x[1], 1.0)
    return MeshData.from_locators(
        mesh,
        {
            tdim: {
                "LeftHalf": lambda x: x[0] <= 0.5 + eps,
                "RightHalf": lambda x: x[0] >= 0.5 - eps,
            },
            tdim - 1: sides,
        },
        boundary=True,
    )


def create_ufl_functionspace(
    cell: str = "triangle", degree: int = 1, shape: tuple[int, ...] = ()
) -> tuple[ufl.Mesh, ufl.FunctionSpace]:
    """A plain UFL mesh and a Lagrange space on it, as a DOLFINx C++ demo has them.

    The mesh has the dimension of the cell.
    """
    gdim = ufl.Cell(cell).topological_dimension
    mesh = ufl.Mesh(element("Lagrange", cell, 1, shape=(gdim,)))
    return mesh, ufl.FunctionSpace(mesh, element("Lagrange", cell, degree, shape=shape))


@pytest.fixture
def ufl_functionspace():
    """Factory for plain UFL meshes and spaces, see :func:`create_ufl_functionspace`."""
    return create_ufl_functionspace


@pytest.fixture
def square():
    """Factory for unit squares with side and half markers.

    See :func:`create_unit_box`.
    """
    return create_unit_box


@pytest.fixture
def box():
    """Factory for unit intervals, squares or cubes, see :func:`create_unit_box`."""
    return create_unit_box


@pytest.fixture
def sides():
    """Marker names of the four sides of the unit square."""
    return SIDES


def _assembled(system):
    """The residual vector and Jacobian and preconditioner matrices of a system.

    A system of several fields is assembled into one blocked vector and matrix.
    """
    kind = "mpi" if isinstance(system.residual, list) else None
    b = dolfinx.fem.petsc.assemble_vector(system.residual, kind=kind)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    matrices = []
    for form in (system.jacobian, system.preconditioner):
        if form is None:
            matrices.append(None)
            continue
        A = dolfinx.fem.petsc.assemble_matrix(form, kind=kind)
        A.assemble()
        matrices.append(A)
    return [b, *matrices]


def assert_same_assembly(first, second, rtol: float = 1e-12):
    """Assert that two discrete systems assemble to the same vectors and matrices.

    The residual vectors and the Jacobian matrices must agree to ``rtol``,
    relative to the norm of the first, and so must the preconditioners, if
    either system has one. Discretize both on the same functions, e.g. by
    building the components on the field of a
    :class:`~sumfem.symbolic.UFLResidual`, so that they are assembled at the
    same values.
    """
    tensors = list(zip(_assembled(first), _assembled(second), strict=True))
    try:
        for what, (x, y) in zip(("residual", "Jacobian", "preconditioner"), tensors):
            if x is None and y is None:
                continue
            assert x is not None and y is not None, f"Only one system has a {what}."
            reference = x.norm()
            if isinstance(x, PETSc.Mat):
                x.axpy(-1.0, y, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            else:
                x.axpy(-1.0, y)
            difference = x.norm()
            assert difference <= rtol * reference, (
                f"The {what}s differ by {difference:.3e}, relative to {reference:.3e}."
            )
    finally:
        for pair in tensors:
            for tensor in pair:
                if tensor is not None:
                    tensor.destroy()


@pytest.fixture
def same_assembly():
    """The comparison of two discrete systems, :func:`assert_same_assembly`."""
    return assert_same_assembly
