"""Meshes of symbolic domains, and what is derived from the meshes alone."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl

from ..error_handling import ComponentError
from ..symbolic.domains import MEASURE_ENTITY_TYPE, Domain, MeasureName, Subdomain
from .meshes import INTEGRAL_TYPES, MeshData, MeshLike, as_mesh_data

__all__ = ["DomainMaps"]


class DomainMaps:
    """Pairs each symbolic domain with a :class:`MeshData`, and checks that they match.

    The maps also hold the submesh of each subdomain, created with its
    markers and its entity map from the parent's mesh when a discretization
    first needs it, unless a mesh is given for the subdomain, and shared by
    every discretization on the maps. Integration entities of the markers
    are computed on first use.

    Args:
        meshes: The mesh of each domain, by the domain or its name. A mesh
            given for a domain is checked against it here, one given by name
            when the domain is first used.
        entity_maps: Entity maps between meshes, for forms that couple
            fields on different meshes.

    Raises:
        ValueError: If there are no meshes, a domain is given twice, or a
            mesh does not match its domain.
    """

    _meshes: dict[str, MeshData]
    """The mesh of each domain, by domain name."""

    _entity_maps: list[dolfinx.mesh.EntityMap]
    """Entity maps between meshes, passed to every form."""

    _checked_domains: dict[str, Domain]
    """The domain each mesh was last checked against, by domain name."""

    _submesh_maps: dict[str, dolfinx.mesh.EntityMap]
    """Entity maps of the submeshes created here, by domain name."""

    _integration_entities: dict[tuple[str, str, str], npt.NDArray[np.int32]]
    """Integration entities, by domain name, integral type and marker name."""

    _checked_interior_facets: set[tuple[str, str | None]]
    """The domain names and facet markers, ``None`` for all facets, whose
    interior facets passed :meth:`check_interior_facets`."""

    def __init__(
        self,
        meshes: Mapping[Domain | str, MeshLike],
        entity_maps: Iterable[dolfinx.mesh.EntityMap] = (),
    ):
        if len(meshes) == 0:
            raise ValueError("Domain maps need at least one mesh.")
        self._meshes = {}
        self._checked_domains = {}
        for key, mesh in meshes.items():
            name = key.name if isinstance(key, Domain) else key
            if name in self._meshes:
                raise ValueError(f"The mesh of domain {name!r} is given twice.")
            self._meshes[name] = as_mesh_data(mesh)
            if isinstance(key, Domain):
                _check_mesh(key, self._meshes[name])
                self._checked_domains[name] = key
        self._entity_maps = list(entity_maps)
        self._submesh_maps = {}
        self._integration_entities = {}
        self._checked_interior_facets = set()

    @property
    def entity_maps(self) -> list[dolfinx.mesh.EntityMap]:
        """Entity maps between meshes, passed to every form."""
        return list(self._entity_maps)

    @property
    def comm(self) -> MPI.Intracomm:
        """The MPI communicator of the meshes."""
        # DOLFINx types a mesh's communicator as MPI.Comm, but it is an
        # intracommunicator, which dolfinx.fem.compile_form requires
        return typing.cast(MPI.Intracomm, next(iter(self._meshes.values())).mesh.comm)

    def ensure_mesh(self, domain: Domain) -> None:
        """Make sure a domain has a mesh, creating the submesh of a subdomain if needed.

        Collective: a submesh is created on every process together. Use
        :meth:`mesh_data` or :meth:`mesh` to get the mesh, which never create
        one, and so can be called on any process.

        Raises:
            KeyError: If there is no mesh for a domain that is no subdomain.
            ValueError: If the mesh does not match the domain.
        """
        if domain.name not in self._meshes and isinstance(domain, Subdomain):
            self.ensure_mesh(domain.parent)
            self._create_submesh(domain)
        self.mesh_data(domain)

    def mesh_data(self, domain: Domain) -> MeshData:
        """The mesh of a domain, with its markers.

        The submesh of a subdomain must have been created by
        :meth:`ensure_mesh`, which a discretization does for the domains of
        what it holds.

        Raises:
            KeyError: If there is no mesh for the domain.
            ValueError: If the mesh does not match the domain's cell type,
                coordinate degree or geometric dimension.
        """
        marked = self._meshes.get(domain.name)
        if marked is None:
            raise KeyError(
                f"No mesh for domain {domain.name!r}. Meshes are given for "
                f"{sorted(self._meshes)}, and created for the subdomains "
                "that discretizations need."
            )
        if self._checked_domains.get(domain.name) is not domain:
            _check_mesh(domain, marked)
            self._checked_domains[domain.name] = domain
        return marked

    def _create_submesh(self, domain: Subdomain) -> MeshData:
        """The submesh of a subdomain's entities, with its entity map."""
        marked, entity_map = self.mesh_data(domain.parent).submesh(
            domain.tdim, domain.marker
        )
        self._meshes[domain.name] = marked
        self._submesh_maps[domain.name] = entity_map
        self._entity_maps.append(entity_map)
        return marked

    def mesh(self, domain: Domain) -> dolfinx.mesh.Mesh:
        """The mesh of a domain."""
        return self.mesh_data(domain).mesh

    # Integration entities

    def integration_entities(
        self, domain: Domain, integral_type: str, marker: str
    ) -> npt.NDArray[np.int32]:
        """Integration entities of a named marker, for a UFL integral type.

        The entities are in the flattened layout DOLFINx assembles over, e.g.
        cells for a cell integral, ``(cell, local facet)`` pairs for an exterior
        facet integral, and ``(cell, local facet, cell, local facet)`` for an
        interior facet integral. The marker is one of:

        - a marker of the domain's mesh, see :attr:`MeshData.markers`;
        - a one-sided marker ``"<subdomain>.<marker>"`` of an exterior facet
          integral moved from a subdomain to its parent, see
          :meth:`~sumfem.symbolic.domains.Domain.one_sided`;
        - an interface marker of an interior facet integral between two
          subdomains, see :meth:`~sumfem.symbolic.domains.Domain.interface`.

        They are computed on first use and cached.

        Raises:
            KeyError: If there is no mesh for the domain.
            ComponentError: If the facets of the marker cannot be integrated
                over with the integral type, e.g. an exterior facet marker with
                facets inside the mesh.
        """
        key = (domain.name, integral_type, marker)
        entities = self._integration_entities.get(key)
        if entities is None:
            marked = self.mesh_data(domain)
            topology = marked.mesh.topology

            # The dimension of the integrated entities, e.g. facets for "ds"
            measure = typing.cast(
                MeasureName, ufl.measure.integral_type_to_measure_name[integral_type]
            )
            dim = domain.entity_dimension(MEASURE_ENTITY_TYPE[measure])

            # Each entity is integrated over from a cell next to it, so
            # connectivity is needed both ways
            topology.create_entities(dim)
            topology.create_connectivity(dim, domain.tdim)
            topology.create_connectivity(domain.tdim, dim)

            one_sided = domain.one_sided(marker)
            sides = domain.interface_sides(marker)
            if integral_type == "exterior_facet" and one_sided is not None:
                # Facets seen from a subdomain's side, which can be interior
                # facets of the parent, so DOLFINx cannot pick the cell
                entities = self._one_sided_entities(*one_sided)
            elif integral_type == "interior_facet" and sides is not None:
                # Facets between two subdomains, oriented so that "+" is in
                # the first
                entities = self._interface_entities(domain, *sides)
            else:
                # A marker of the mesh: check that its facets suit the
                # integral type, then let DOLFINx pair them with their cells
                marked_entities = marked.entities(dim, marker)
                if integral_type == "exterior_facet":
                    self._check_exterior(domain, marker, marked_entities)
                elif integral_type == "interior_facet":
                    self.check_interior_facets(domain, marker)
                entities = dolfinx.fem.compute_integration_domains(
                    INTEGRAL_TYPES[integral_type], topology, marked_entities
                )
            self._integration_entities[key] = entities
        return entities

    def _interface_entities(
        self, domain: Domain, first: Subdomain, second: Subdomain, marker: str
    ) -> npt.NDArray[np.int32]:
        """The interior facets of an interface, with the first subdomain's cell first.

        Each facet is ``(cell, local facet, cell, local facet)`` of the parent,
        ordered so that the ``"+"`` restriction is in ``first``.

        Raises:
            ComponentError: On every process, if a facet of the marker does not
                lie between the two subdomains.
        """
        marked = self.mesh_data(domain)
        topology = marked.mesh.topology
        tdim = domain.tdim

        # Both cells of each facet must be on the process
        self.check_interior_facets(domain, marker)

        # The marked facets as interior facets of the parent, one row
        # (cell, local facet, cell, local facet) each, in DOLFINx's order
        entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.interior_facet,
            topology,
            marked.entities(tdim - 1, marker),
        ).reshape(-1, 4)

        # Which cells on the process, owned and ghosts, lie in each subdomain
        cell_map = topology.index_map(tdim)
        in_first = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=bool)
        in_first[marked.entities(tdim, first.marker)] = True
        in_second = np.zeros_like(in_first)
        in_second[marked.entities(tdim, second.marker)] = True

        # Swap the two sides of the rows that start outside the first
        # subdomain, so that the first cell, the "+" side, is in it
        swap = ~in_first[entities[:, 0]]
        entities[swap] = entities[swap][:, [2, 3, 0, 1]]

        # Every facet must now have its "+" cell in the first subdomain and
        # its "-" cell in the second. Counted over all processes, so that
        # every process raises, and none waits in a later collective call
        wrong = ~(in_first[entities[:, 0]] & in_second[entities[:, 2]])
        if self.comm.allreduce(int(np.count_nonzero(wrong))) > 0:
            raise ComponentError(
                f"Facet marker {marker!r} of domain {domain.name!r} has facets that "
                f"do not lie between {first.name!r} and {second.name!r}."
            )
        return entities.reshape(-1).astype(np.int32)

    def _one_sided_entities(
        self, subdomain: Subdomain, marker: str | None
    ) -> npt.NDArray[np.int32]:
        """The parent's (cell, local facet) pairs of a subdomain's boundary.

        The facets of the subdomain's boundary in the marker, or all of it,
        each with the cell of the subdomain next to it, as a cell of the
        parent. The local facet index is the same in both.
        """
        marked = self.mesh_data(subdomain)

        # Only a submesh created here has an entity map to its parent
        entity_map = self._submesh_maps.get(subdomain.name)
        if entity_map is None:
            raise ComponentError(
                f"The mesh of subdomain {subdomain.name!r} was given, so its "
                "facets cannot be found on the parent. Let the domain maps "
                "create it."
            )
        # On the submesh, the facets are on the boundary, so each has one
        # cell, found through the connectivity in both directions
        topology = marked.mesh.topology
        fdim = subdomain.tdim - 1
        topology.create_connectivity(fdim, subdomain.tdim)
        topology.create_connectivity(subdomain.tdim, fdim)

        # The whole boundary of the subdomain, or the facets of its marker
        facets = (
            dolfinx.mesh.exterior_facet_indices(topology)
            if marker is None
            else marked.entities(fdim, marker)
        )

        # (cell, local facet) pairs on the submesh, with each cell then
        # replaced by its cell in the parent; the local facet index is kept
        pairs = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet, topology, facets
        ).reshape(-1, 2)
        pairs[:, 0] = entity_map.sub_topology_to_topology(pairs[:, 0], inverse=False)
        return pairs.reshape(-1)

    def _check_exterior(
        self, domain: Domain, marker: str, facets: npt.NDArray[np.int32]
    ) -> None:
        """Raise unless the facets of a marker are all on the boundary.

        Raises:
            ComponentError: On every process, if a process has a facet of the
                marker inside the mesh.
        """
        topology = self.mesh(domain).topology
        owned = facets[facets < topology.index_map(domain.tdim - 1).size_local]
        exterior = dolfinx.mesh.exterior_facet_indices(topology)
        inside = np.setdiff1d(owned, exterior).size
        if self.comm.allreduce(inside) > 0:
            raise ComponentError(
                f"Marker {marker!r} of domain {domain.name!r} has facets inside "
                "the mesh, which an exterior facet integral cannot cover. For a "
                "subdomain of such facets, couple to the parent with two-sided "
                "interface terms instead."
            )

    def check_interior_facets(self, domain: Domain, marker: str | None) -> None:
        """Raise unless the interior facets of a domain can be integrated over.

        Each needs exactly two cells, and both on the process. A submesh of
        facets can have more, where three or more of its cells meet, which
        an interior facet integral does not cover. In parallel, the cells of
        a submesh of codimension one or more that meet at a facet on a
        process boundary share only a ridge of the parent, so the parent
        must be ghosted by vertex, see :meth:`MeshData.ghost_by_vertex`.

        Args:
            domain: The domain.
            marker: A facet marker of the domain, or ``None`` for all facets.

        Raises:
            ComponentError: On every process, if a process has such a facet.
        """
        # A domain and marker passed once are not checked again
        key = (domain.name, marker)
        if key in self._checked_interior_facets:
            return
        marked = self.mesh_data(domain)
        topology = marked.mesh.topology
        fdim = domain.tdim - 1
        topology.create_entities(fdim)
        topology.create_connectivity(fdim, domain.tdim)

        # Only owned facets, so that each facet is counted on one process
        num_owned = topology.index_map(fdim).size_local
        if marker is None:
            facets = np.arange(num_owned, dtype=np.int32)
        else:
            facets = marked.entities(fdim, marker)
            facets = facets[facets < num_owned]

        # The number of cells of each facet on this process, owned and ghosts
        connectivity = topology.connectivity(fdim, domain.tdim)
        counts = np.diff(connectivity.offsets)[facets]

        # More than two cells: a branching joint, e.g. of a submesh of facets
        branched = int(np.count_nonzero(counts > 2))

        # One cell, but not on the boundary of the mesh: the other cell is on
        # another process and not ghosted here
        one_sided = facets[counts == 1]
        exterior = dolfinx.mesh.exterior_facet_indices(topology)
        unghosted = np.setdiff1d(one_sided, exterior).size

        # Summed over all processes, so that every process raises, and none
        # waits in a later collective call
        branched, unghosted = self.comm.allreduce(np.array([branched, unghosted]))
        where = f"domain {domain.name!r}" + (f", marker {marker!r}" if marker else "")
        if branched > 0:
            raise ComponentError(
                f"{branched} facets of {where} have more than two cells, which "
                "an interior facet integral cannot cover. Integrate over the "
                "other facets only, with a marker."
            )
        if unghosted > 0:
            raise ComponentError(
                f"{unghosted} facets of {where} lie on a process boundary "
                "without the cell on the other side, so interior facet "
                "integrals cannot be computed there. Ghost the parent mesh by "
                "vertex, with MeshData.ghost_by_vertex(), before creating the "
                "discretization."
            )
        self._checked_interior_facets.add(key)


def _check_mesh(domain: Domain, marked: MeshData) -> None:
    """Check that a mesh has the coordinate element of its domain.

    Raises:
        ValueError: If it does not.
    """
    expected = domain.ufl_coordinate_element()
    mesh_domain = marked.mesh.ufl_domain()
    if mesh_domain is None:
        raise ValueError(f"The mesh for domain {domain.name!r} has no UFL domain.")
    actual = mesh_domain.ufl_coordinate_element()
    # Compiled forms embed the coordinate element, so it must match exactly,
    # including the Lagrange variant
    if expected != actual:
        raise ValueError(
            f"The mesh for domain {domain.name!r} has coordinate element "
            f"{actual!r}, but the domain has {expected!r}. Create the domain "
            "with Domain.from_mesh(), or with a matching cell, degree, gdim and "
            'lagrange_variant. Meshes read from gmsh use lagrange_variant="equispaced".'
        )
