"""PyVista grids of functions, gathered from all processes."""

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from sumfem.plotting import gather_grid  # noqa: E402


@pytest.mark.parametrize("shape, degree", [((), 1), ((2,), 1), ((), 2)])
def test_every_owned_cell_is_gathered_once(shape, degree):
    """The grid has every cell once, and the function's values at its points.

    A vector is padded to three components."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, shape))
    u = dolfinx.fem.Function(V, name="u")
    if shape:
        u.interpolate(lambda x: np.vstack([x[0] + 2 * x[1], -x[1]]))
    else:
        u.interpolate(lambda x: x[0] + 2 * x[1])
    grid = gather_grid(u, root=MPI.COMM_WORLD.size - 1)
    if MPI.COMM_WORLD.rank != MPI.COMM_WORLD.size - 1:
        assert grid is None
        return
    assert grid.n_cells == 2 * 4 * 4
    x, y = grid.points[:, 0], grid.points[:, 1]
    values = grid.point_data["u"]
    if shape:
        assert values.shape == (grid.n_points, 3)
        assert np.allclose(values, np.column_stack([x + 2 * y, -y, 0 * x]))
    else:
        assert np.allclose(values, x + 2 * y)


def test_a_function_on_a_submesh_of_facets():
    """E.g. a multiplier on part of the boundary, which some processes hold
    no cells of."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.isclose(x[0], 0.0)
    )
    submesh, *_ = dolfinx.mesh.create_submesh(mesh, 1, facets)
    Q = dolfinx.fem.functionspace(submesh, ("Lagrange", 1))
    q = dolfinx.fem.Function(Q)
    q.interpolate(lambda x: x[1])
    grid = gather_grid(q, name="pressure")
    if MPI.COMM_WORLD.rank == 0:
        assert grid.n_cells == 4
        assert np.allclose(grid.point_data["pressure"], grid.points[:, 1])
