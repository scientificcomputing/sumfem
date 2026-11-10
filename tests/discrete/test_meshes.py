"""Meshes with named markers."""

from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest

from sumfem.discrete import MeshData


def test_mesh_tags_are_named_in_their_own_dimension():
    """Existing mesh tags of cells and facets are named by their tag values.
    Their dimension is their own, and a name may stand for several values.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    def left_half(x):
        return x[0] <= 0.5 + 1e-10

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    left_tag, right_tag, half_tag = 1, 2, 3
    cells = dolfinx.mesh.locate_entities(mesh, tdim, left_half)
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cells, np.full_like(cells, half_tag))
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, right)
    facets = np.concatenate([left_facets, right_facets])
    values = np.concatenate(
        [np.full_like(left_facets, left_tag), np.full_like(right_facets, right_tag)]
    )
    order = np.argsort(facets)
    facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, facets[order], values[order])

    data = MeshData.from_meshtags(
        mesh,
        {
            cell_tags: {"LeftHalf": (half_tag,)},
            facet_tags: {"Left": (left_tag,), "Sides": (left_tag, right_tag)},
        },
    )
    assert data.tags == {tdim: cell_tags, tdim - 1: facet_tags}
    assert data.markers == {
        tdim: {"LeftHalf": (half_tag,)},
        tdim - 1: {"Left": (left_tag,), "Sides": (left_tag, right_tag)},
    }

    # They mark the entities that locators find. Locators mark each entity once,
    # so the sides are compared with the union of left and right.
    located = MeshData.from_locators(
        mesh,
        {tdim: {"LeftHalf": left_half}, tdim - 1: {"Left": left, "Right": right}},
        boundary=True,
    )

    def entities(mesh_data, dim, *names):
        return np.sort(np.concatenate([mesh_data.entities(dim, n) for n in names]))

    assert np.array_equal(
        entities(data, tdim, "LeftHalf"), entities(located, tdim, "LeftHalf")
    )
    assert np.array_equal(
        entities(data, tdim - 1, "Left"), entities(located, tdim - 1, "Left")
    )
    assert np.array_equal(
        entities(data, tdim - 1, "Sides"), entities(located, tdim - 1, "Left", "Right")
    )


def test_mesh_tags_of_one_dimension_are_given_once():
    """Two mesh tags of the same dimension are rejected, as markers are kept by
    dimension: the names of both belong in one mesh tags.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    first = dolfinx.mesh.meshtags(mesh, tdim - 1, facets, np.full_like(facets, 1))
    second = dolfinx.mesh.meshtags(mesh, tdim - 1, facets, np.full_like(facets, 2))
    with pytest.raises(ValueError, match="Two mesh tags of dimension 1"):
        MeshData.from_meshtags(mesh, {first: {"A": (1,)}, second: {"B": (2,)}})


def test_mesh_tags_of_another_mesh_are_rejected():
    """Mesh tags whose topology is not the mesh's are rejected, as their entity
    indices refer to another mesh, even one with the same cells.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    other = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    tdim = other.topology.dim
    cells = np.arange(other.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_tags = dolfinx.mesh.meshtags(other, tdim, cells, np.full_like(cells, 1))
    with pytest.raises(ValueError, match="are of another mesh"):
        MeshData.from_meshtags(mesh, {cell_tags: {"All": (1,)}})
    assert (
        MeshData.from_meshtags(other, {cell_tags: {"All": (1,)}}).tags[tdim]
        is cell_tags
    )


def test_overlapping_locators_are_rejected():
    """Locators of two markers that mark the same entity are rejected, on every
    process, as mesh tags hold one value per entity.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = mesh.topology.dim
    with pytest.raises(ValueError, match="'Left' and 'Sides' of dimension 1 mark"):
        MeshData.from_locators(
            mesh,
            {
                tdim - 1: {
                    "Left": lambda x: np.isclose(x[0], 0.0),
                    "Sides": lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
                }
            },
            boundary=True,
        )
    # Only the right side is on some processes; all of them raise
    with pytest.raises(ValueError, match="mark the same entities"):
        MeshData.from_locators(
            mesh,
            {tdim: {"Right": lambda x: x[0] >= 0.75, "All": lambda x: x[0] > -1}},
        )
