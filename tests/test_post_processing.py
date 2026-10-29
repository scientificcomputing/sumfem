"""Norms, errors, integrals and point values of DOLFINx functions."""

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import ufl

from sumfem.post_processing import (
    NormType,
    error,
    evaluate_at_points,
    integrate,
    norm,
)


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)


def test_norms_of_known_functions(mesh):
    x = ufl.SpatialCoordinate(mesh)
    # f = x: (f, f) = 1/3 and (grad f, grad f) = 1
    np.testing.assert_allclose(norm(x[0]), np.sqrt(1 / 3))
    np.testing.assert_allclose(norm(x[0], NormType.H10), 1.0)
    np.testing.assert_allclose(norm(x[0], "H1"), np.sqrt(4 / 3))
    # f = (x, y): div f = 2 and curl f = 0
    v = ufl.as_vector((x[0], x[1]))
    np.testing.assert_allclose(norm(v, "Hdiv"), np.sqrt(2 / 3 + 4))
    np.testing.assert_allclose(norm(v, "Hcurl"), np.sqrt(2 / 3))
    # f = (-y, x): curl f = 2
    w = ufl.as_vector((-x[1], x[0]))
    np.testing.assert_allclose(norm(w, "Hcurl"), np.sqrt(2 / 3 + 4))


def test_l2_norm_counts_every_degree_of_freedom_once(mesh):
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1.0
    size = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    np.testing.assert_allclose(norm(u, "l2"), np.sqrt(size))
    with pytest.raises(TypeError):
        norm(ufl.SpatialCoordinate(mesh)[0], "l2")


def test_errors_against_exact_solutions(mesh):
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2)
    assert error(u, lambda x: x[0] ** 2) < 1e-14
    assert error(u, lambda x: x[0] ** 2, "H1") < 1e-13
    # Against zero, the relative error is one
    x = ufl.SpatialCoordinate(mesh)
    np.testing.assert_allclose(error(u, 2 * x[0] ** 2, relative=True), 0.5)


def test_integrals_need_a_mesh_for_constants(mesh):
    np.testing.assert_allclose(integrate(1.0, mesh), 1.0)
    with pytest.raises(ValueError, match="mesh"):
        integrate(1.0)
    with pytest.raises(ValueError, match="Unknown norm"):
        norm(ufl.SpatialCoordinate(mesh)[0], "W1p")


def test_custom_norms(mesh):
    """A weighted norm: (x f, f) for f = 1 is the integral of x, 1/2."""
    x = ufl.SpatialCoordinate(mesh)

    def weighted(f):
        return x[0] * ufl.inner(f, f)

    np.testing.assert_allclose(norm(ufl.as_ufl(1.0), weighted, mesh), np.sqrt(0.5))
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    np.testing.assert_allclose(error(u, 1.0 + 0 * x[0], weighted), np.sqrt(0.5))


@pytest.mark.parametrize("corner", [False, True])
def test_points_are_evaluated_once(mesh, corner):
    """Each point in the mesh on one process, also when a process has none."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: np.vstack((x[0] + 2 * x[1], x[0] - x[1])))
    points = np.array([[0.05, 0.02], [0.01, 0.03]])
    if not corner:
        # A vertex, shared by processes, and a point outside the mesh
        points = np.vstack((points, [[0.5, 0.5], [0.9, 0.3], [2.0, 2.0]]))
    indices, values = evaluate_at_points(f, points)
    assert values.shape == (len(indices), 2)
    gathered = mesh.comm.allgather((indices, values))
    found = np.concatenate([i for i, _ in gathered])
    inside = np.flatnonzero(points.max(axis=1) <= 1.0)
    assert sorted(found) == list(inside)
    for i, v in gathered:
        x, y = points[i, 0], points[i, 1]
        np.testing.assert_allclose(v, np.column_stack((x + 2 * y, x - y)), atol=1e-14)
