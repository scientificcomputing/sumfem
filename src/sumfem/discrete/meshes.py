"""Meshes with their mesh tags, and names for their markers."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Iterable, Mapping

import basix
import basix.ufl
import dolfinx.fem
import dolfinx.graph
import dolfinx.io
import dolfinx.io.gmsh
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl

__all__ = [
    "INTEGRAL_TYPES",
    "Locator",
    "MeshData",
    "MeshLike",
    "as_mesh_data",
    "ghost_by_vertex",
]

INTEGRAL_TYPES: dict[str, dolfinx.fem.IntegralType] = {
    "cell": dolfinx.fem.IntegralType.cell,
    "exterior_facet": dolfinx.fem.IntegralType.exterior_facet,
    "interior_facet": dolfinx.fem.IntegralType.interior_facet,
    "ridge": dolfinx.fem.IntegralType.ridge,
    "vertex": dolfinx.fem.IntegralType.vertex,
}
"""The DOLFINx integral type of each UFL integral type."""

Locator = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]]
"""A function of the coordinates ``x``, shape ``(3, num_points)``, that
returns ``True`` for the points of the entities to mark."""


@dataclasses.dataclass(frozen=True)
class MeshData:
    """One mesh with its mesh tags, and names for its markers.

    It knows nothing of symbolic domains: :class:`DomainMaps` pairs domains
    with meshes. For each entity dimension, a marker name maps to tag values. Create it
    from a locator function for each dimension and name with
    :meth:`from_locators`, from existing mesh tags for each dimension and the
    integer tag values of each name directly, e.g.
    ``MeshData(mesh, {1: facet_tags}, {1: {"Inlet": (1,), "Walls": (2, 3)}})``,
    or from the physical groups of a mesh read with :mod:`dolfinx.io.gmsh`
    with :meth:`from_gmsh`.

    Attributes:
        mesh: The mesh.
        tags: Mesh tags, by entity dimension.
        markers: The tag values of each named marker, by entity dimension.

    Raises:
        ValueError: If mesh tags are given for another dimension than their
            own, or markers for a dimension without mesh tags.
    """

    mesh: dolfinx.mesh.Mesh
    tags: Mapping[int, dolfinx.mesh.MeshTags] = types.MappingProxyType({})
    markers: Mapping[int, Mapping[str, tuple[int, ...]]] = types.MappingProxyType({})

    def __post_init__(self) -> None:
        for dim, tags in self.tags.items():
            if tags.dim != dim:
                raise ValueError(f"Mesh tags of dimension {tags.dim} given as {dim}.")
        for dim in self.markers:
            if dim not in self.tags:
                raise ValueError(f"Markers of dimension {dim} have no mesh tags.")

    @classmethod
    def from_gmsh(cls, data: dolfinx.io.gmsh.MeshData) -> MeshData:
        """Use the physical groups of a mesh read with :mod:`dolfinx.io.gmsh`.

        Each physical group becomes a marker with the group's name.

        Args:
            data: The mesh, its mesh tags and its physical groups, e.g. from
                :func:`dolfinx.io.gmsh.read_from_msh`.
        """
        tags = {
            t.dim: t
            for t in (data.cell_tags, data.facet_tags, data.ridge_tags, data.peak_tags)
            if t is not None
        }
        markers: dict[int, dict[str, tuple[int, ...]]] = {}
        for name, group in data.physical_groups.items():
            markers.setdefault(group.dim, {})[name] = (group.tag,)
        return cls(data.mesh, tags, markers)

    @classmethod
    def from_locators(
        cls,
        mesh: dolfinx.mesh.Mesh,
        locators: Mapping[int, Mapping[str, Locator]],
        boundary: bool = False,
    ) -> MeshData:
        """Mark entities geometrically, as :func:`dolfinx.mesh.locate_entities`.

        Args:
            mesh: The mesh.
            locators: For each entity dimension, a locator per marker name.
                Where locators overlap, the later marker wins.
            boundary: Mark only entities on the boundary of the mesh, for
                dimensions below the cell dimension. Use this for boundary
                markers, so that interior facets touching the boundary are
                not marked.
        """
        tags: dict[int, dolfinx.mesh.MeshTags] = {}
        markers: dict[int, dict[str, tuple[int, ...]]] = {}
        tdim = mesh.topology.dim
        if boundary:
            mesh.topology.create_connectivity(tdim - 1, tdim)
        for dim, named in locators.items():
            mesh.topology.create_entities(dim)
            index_map = mesh.topology.index_map(dim)
            values = np.zeros(index_map.size_local + index_map.num_ghosts, np.int32)
            markers[dim] = {}
            locate = (
                dolfinx.mesh.locate_entities_boundary
                if boundary and dim < tdim
                else dolfinx.mesh.locate_entities
            )
            for value, (name, locator) in enumerate(named.items(), start=1):
                values[locate(mesh, dim, locator)] = value
                markers[dim][name] = (value,)
            indices = np.flatnonzero(values).astype(np.int32)
            tags[dim] = dolfinx.mesh.meshtags(mesh, dim, indices, values[indices])
        return cls(mesh, tags, markers)

    def entities(self, dim: int, marker: str) -> npt.NDArray[np.int32]:
        """Local indices of the entities of dimension ``dim`` with a marker.

        Raises:
            KeyError: If the mesh has no such marker.
        """
        values = self.markers.get(dim, {}).get(marker)
        if values is None:
            known = sorted(self.markers.get(dim, {}))
            raise KeyError(
                f"The mesh has no marker {marker!r} of dimension {dim}. "
                f"Markers of that dimension: {known}."
            )
        tags = self.tags[dim]
        return tags.indices[np.isin(tags.values, values)].astype(np.int32)

    def unmarked_boundary_facets(self, markers: Iterable[str]) -> npt.NDArray[np.int32]:
        """Local indices of the boundary facets that none of the markers covers.

        A hyperbolic system needs a boundary flux on every boundary facet,
        so the markers of its boundary models should leave none.
        """
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, fdim + 1)
        boundary = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        marked = [self.entities(fdim, marker) for marker in markers]
        covered = np.concatenate(marked) if marked else np.empty(0, np.int32)
        return np.setdiff1d(boundary, covered).astype(np.int32)

    def submesh(self, dim: int, marker: str) -> tuple[MeshData, dolfinx.mesh.EntityMap]:
        """The submesh of the entities of a dimension with a marker.

        The submesh keeps the markers of its entities: the parent's markers
        of dimension ``dim`` and lower become those of the submesh, e.g. a
        surface of a 3D mesh has the parent's facet markers as cell markers
        and its ridge markers as facet markers.

        Returns:
            The submesh, and the entity map from its cells to the parent's
            entities.
        """
        mesh = self.mesh
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        submesh, entity_map, vertex_map = dolfinx.mesh.create_submesh(
            mesh, dim, self.entities(dim, marker)
        )[:3]
        tags: dict[int, dolfinx.mesh.MeshTags] = {}
        markers: dict[int, Mapping[str, tuple[int, ...]]] = {}
        for d, parent_tags in self.tags.items():
            if d > dim:
                continue
            mesh.topology.create_connectivity(d, 0)
            tags[d] = dolfinx.mesh.transfer_meshtags_to_submesh(
                parent_tags, submesh, entity_map, vertex_map
            )
            markers[d] = dict(self.markers.get(d, {}))
        return MeshData(submesh, tags, markers), entity_map

    def ghost_by_vertex(self) -> MeshData:
        """The mesh with every cell ghosted on each process sharing its vertices.

        See :func:`ghost_by_vertex`. The markers carry over, onto the new
        ghost entities too. On one process the mesh is returned.
        """
        if self.mesh.comm.size == 1:
            return self
        ghosted = ghost_by_vertex(self.mesh)
        tags = {
            d: _redistribute_tags(t, self.mesh, ghosted) for d, t in self.tags.items()
        }
        return MeshData(ghosted, tags, self.markers)


def ghost_by_vertex(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.Mesh:
    """The mesh, with every cell ghosted on each process sharing its vertices.

    Interior facet integrals on a submesh of facets need both cells of each
    of its facets on the process, but cells that share only a ridge are not
    ghosted by default. With this ghost layer, all cells around each owned
    vertex are on the process. The cells keep their owners. The new mesh
    has the global node numbers of ``mesh`` as its input node indices, see
    ``mesh.geometry.input_global_indices``.

    After J. S. Dokken,
    https://gist.github.com/jorgensd/1e3d89b8c46a05fce36eb2fe88cfe9a6.
    """
    topology = mesh.topology
    tdim = topology.dim
    topology.create_connectivity(tdim, 0)
    num_owned = topology.index_map(tdim).size_local
    cell_to_vertex = topology.connectivity(tdim, 0)
    dest, offset = topology.index_map(0).index_to_dest_ranks()
    rank = mesh.comm.rank
    destinations, counts = [], []
    for cell in range(num_owned):
        sharing: set[int] = set()
        for v in cell_to_vertex.links(cell):
            sharing.update(dest[offset[v] : offset[v + 1]])
        sharing.discard(rank)
        # The owner first, then the processes that ghost the cell
        destinations.extend([rank, *sorted(sharing)])
        counts.append(1 + len(sharing))
    offsets = np.zeros(num_owned + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    layout = dolfinx.graph.adjacencylist(
        np.asarray(destinations, dtype=np.int32), offsets
    )

    def keep_partition(
        comm: object,
        nparts: object,
        dual_graph: object,
        node_weights: object,
        edge_weights: object,
        ghosting: object,
    ) -> dolfinx.graph.AdjacencyList:
        # The cells are handed back in the order they are passed
        return layout

    # create_mesh numbers the given points by position, so pass the owned
    # nodes and refer to them by their global index, whose local ranges are
    # that same running count
    geometry = mesh.geometry
    node_map = geometry.index_map()
    cells = node_map.local_to_global(
        geometry.dofmaps[0][:num_owned].reshape(-1).astype(np.int32)
    ).reshape(num_owned, -1)
    cmap = geometry.cmaps[0]
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            topology.cell_name(),
            cmap.degree,
            basix.LagrangeVariant(cmap.variant),
            shape=(geometry.dim,),
            dtype=geometry.x.dtype,
        )
    )
    return dolfinx.mesh.create_mesh(
        mesh.comm,
        cells,
        domain,
        geometry.x[: node_map.size_local],
        partitioner=keep_partition,
    )


def _redistribute_tags(
    tags: dolfinx.mesh.MeshTags, mesh: dolfinx.mesh.Mesh, other: dolfinx.mesh.Mesh
) -> dolfinx.mesh.MeshTags:
    """The tags of a mesh on the same mesh distributed differently.

    The input nodes of ``other`` must be the global nodes of ``mesh``, as
    from :func:`ghost_by_vertex`. Ghost entities get tags too.
    """
    dim = tags.dim
    mesh.topology.create_connectivity(dim, mesh.topology.dim)
    owned = tags.indices < mesh.topology.index_map(dim).size_local
    nodes = dolfinx.mesh.entities_to_geometry(
        mesh, dim, tags.indices[owned], permute=False
    )
    entities = (
        mesh.geometry.index_map()
        .local_to_global(nodes.reshape(-1).astype(np.int32))
        .reshape(nodes.shape)
        .astype(np.int64)
    )
    other.topology.create_entities(dim)
    other.topology.create_connectivity(dim, 0)
    other.topology.create_connectivity(dim, other.topology.dim)
    local, values = dolfinx.io.distribute_entity_data(
        other, dim, entities, tags.values[owned]
    )
    return dolfinx.mesh.meshtags_from_entities(
        other, dim, dolfinx.graph.adjacencylist(local), values
    )


MeshLike = MeshData | dolfinx.mesh.Mesh | dolfinx.io.gmsh.MeshData
"""A mesh, possibly with markers. A plain mesh has no markers."""


def as_mesh_data(mesh: MeshLike) -> MeshData:
    """A mesh with markers, from any :data:`MeshLike`."""
    if isinstance(mesh, MeshData):
        return mesh
    if isinstance(mesh, dolfinx.io.gmsh.MeshData):
        return MeshData.from_gmsh(mesh)
    if isinstance(mesh, dolfinx.mesh.Mesh):
        return MeshData(mesh)
    raise TypeError(f"Expected a mesh, got {type(mesh).__name__}.")
