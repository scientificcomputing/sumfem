from __future__ import annotations

import functools
import typing

import basix
import dolfinx
import dolfinx.mesh
import ufl
from basix.ufl import element
from ufl.domain import extract_domains
from ufl.finiteelement import AbstractFiniteElement

from ..error_handling import ComponentError, check_name, qualified_name
from .coordinates import Cartesian, CoordinateSystem

__all__ = [
    "MEASURE_ENTITY_TYPE",
    "EntityType",
    "MeasureName",
    "create_coordinate_element",
    "Domain",
    "Subdomain",
    "domain_of",
    "place_subdomain_integrals",
]

CellType = typing.Literal[
    "point",
    "interval",
    "triangle",
    "quadrilateral",
    "tetrahedron",
    "hexahedron",
    "prism",
    "pyramid",
]
"""Cell types, by the names of :class:`basix.CellType` and
:class:`dolfinx.mesh.CellType`. UFL calls ``point`` cells ``vertex``."""

EntityType = typing.Literal["cell", "facet", "ridge", "peak", "vertex"]
"""Mesh entities that markers can refer to. ``cell``, ``facet``, ``ridge``
and ``peak`` have codimension 0, 1, 2 and 3. ``vertex`` has dimension 0,
that is codimension ``tdim``."""

MeasureName = typing.Literal["dx", "ds", "dS", "dr", "dP"]
"""UFL measures supported by :meth:`Domain.measure`."""

MEASURE_ENTITY_TYPE: dict[MeasureName, EntityType] = {
    "dx": "cell",
    "ds": "facet",
    "dS": "facet",
    "dr": "ridge",
    "dP": "vertex",
}
"""The entity type that the markers of each measure refer to."""

_CODIMENSION: dict[EntityType, int] = {"cell": 0, "facet": 1, "ridge": 2, "peak": 3}


def create_coordinate_element(
    cell_type: CellType,
    coordinate_element_degree: int,
    gdim: int | None = None,
    lagrange_variant: basix.LagrangeVariant | str | None = None,
) -> AbstractFiniteElement:
    """Create a coordinate element for a given cell type and degree.

    Args:
        cell_type: The type of the cell (e.g., "triangle", "tetrahedron").
        coordinate_element_degree: The degree of the coordinate element.
        gdim: The geometric dimension. If None, it is inferred from the cell type
        lagrange_variant: The Lagrange variant, e.g. ``"equispaced"`` as used
            by meshes read from gmsh. Defaults to Basix's default, as used by
            the built-in DOLFINx meshes.
    """

    tdim = dolfinx.cpp.mesh.cell_dim(dolfinx.mesh.to_type(cell_type))
    gdim = tdim if gdim is None else gdim
    if lagrange_variant is None:
        return element("Lagrange", cell_type, coordinate_element_degree, shape=(gdim,))
    if isinstance(lagrange_variant, str):
        lagrange_variant = basix.LagrangeVariant[lagrange_variant]
    return element(
        "Lagrange",
        cell_type,
        coordinate_element_degree,
        lagrange_variant,
        shape=(gdim,),
    )


class Domain(ufl.Mesh):
    """A symbolic description of a type of mesh.

    A domain has a name, a cell type and named markers, but does not store
    any data.

    A domain is named independently of the models that use it, and is
    shared between models by identity. Several models written on the same
    ``Domain`` are discretized on the same mesh.

    Subdomains are referred to by marker names, see :meth:`measure`. The
    discretization later maps the names to the tags of an actual mesh.

    Args:
        name: Name of the domain, used to pair it with a mesh.
        cell: Cell type, e.g. ``"triangle"`` or ``"hexahedron"``.
        gdim: Geometric dimension. Defaults to the topological dimension.
        degree: Degree of the coordinate element.
        length_scale: Typical length of the domain in metres, used as
            reference length in dimensional analysis. Mesh coordinates are in
            metres.
        lagrange_variant: Lagrange variant of the coordinate element. It must
            match the mesh, e.g. ``"equispaced"`` for meshes read from gmsh.
            :meth:`from_mesh` takes it from a mesh.
        coordinates: The coordinate system, which weights the integrals over
            the domain, e.g.
            :class:`~sumfem.symbolic.coordinates.Cylindrical` for an
            axisymmetric body. Defaults to
            :class:`~sumfem.symbolic.coordinates.Cartesian`.

    Raises:
        ComponentError: For a name that is not an identifier, a length scale
            that is not positive, or a coordinate system the geometric
            dimension does not fit.
    """

    _name: str
    """Name of the domain, used to pair it with a mesh."""

    _markers: dict[int, dict[str, int]]
    """Marker name to integer id, per entity dimension ``0, ..., tdim``.

    The id is the ``subdomain_id`` of the UFL measures, not a mesh tag value.
    A name may stand for several tag values of a mesh, see
    :attr:`sumfem.discrete.MeshData.markers`."""

    _measures: dict[MeasureName, ufl.Measure]
    """Measure over the whole domain, per integral type, created on first use."""

    _one_sided: dict[str, tuple[Subdomain, str | None]]
    """Facet markers seen from one side: the subdomain of cells on that side,
    and its own facet marker, or ``None`` for its whole boundary."""

    _interfaces: dict[str, tuple[Subdomain, Subdomain, str]]
    """Interior facet markers between two subdomains of cells, whose ``"+"``
    side is the first: the two subdomains and the facet marker."""

    _length_scale: float
    """Typical length of the domain in metres, for dimensional analysis."""

    _coordinates: CoordinateSystem
    """The coordinate system, which weights the integrals."""

    def __init__(
        self,
        name: str,
        cell: CellType,
        gdim: int | None = None,
        degree: int = 1,
        length_scale: float = 1.0,
        lagrange_variant: basix.LagrangeVariant | str | None = None,
        coordinates: CoordinateSystem | None = None,
    ):
        check_name(name, type(self).__name__, allow_dots=False)
        if not float(length_scale) > 0:
            raise ComponentError(
                f"length_scale must be positive, got {length_scale!r}."
            )
        super().__init__(
            create_coordinate_element(cell, degree, gdim, lagrange_variant)
        )
        self._name = name
        self._markers = {dim: {} for dim in range(self.tdim + 1)}
        self._measures = {}
        self._one_sided = {}
        self._interfaces = {}
        self._length_scale = float(length_scale)
        self._coordinates = Cartesian() if coordinates is None else coordinates
        self._coordinates.check(self.gdim)

    @classmethod
    def from_mesh(
        cls,
        name: str,
        mesh: dolfinx.mesh.Mesh,
        length_scale: float = 1.0,
        coordinates: CoordinateSystem | None = None,
    ) -> Domain:
        """A domain whose coordinate element matches an existing mesh.

        Args:
            name: Name of the domain.
            mesh: The mesh, which provides cell type, geometric dimension,
                coordinate degree and Lagrange variant.
            length_scale: See :class:`Domain`.
            coordinates: See :class:`Domain`.
        """
        mesh_domain = mesh.ufl_domain()
        if mesh_domain is None:
            raise ValueError("The mesh has no UFL domain.")
        element_ = mesh_domain.ufl_coordinate_element()
        return cls(
            name,
            typing.cast(CellType, element_.cell_type.name),
            gdim=element_.reference_value_shape[0],
            degree=element_.embedded_superdegree,
            length_scale=length_scale,
            lagrange_variant=element_.lagrange_variant,
            coordinates=coordinates,
        )

    @property
    def name(self) -> str:
        """Name of the domain."""
        return self._name

    @property
    def cell_name(self) -> str:
        """Name of the cell type, e.g. ``"triangle"``."""
        return self.ufl_coordinate_element().cell_type.name

    @property
    def length_scale(self) -> float:
        """Typical length of the domain in metres, for dimensional analysis."""
        return self._length_scale

    @property
    def coordinates(self) -> CoordinateSystem:
        """The coordinate system, which weights the integrals."""
        return self._coordinates

    @property
    def tdim(self) -> int:
        """Topological dimension."""
        return self.topological_dimension

    @property
    def gdim(self) -> int:
        """Geometric dimension."""
        return self.geometric_dimension

    def entity_dimension(self, entity_type: EntityType) -> int:
        """Topological dimension of an entity type on this domain.

        Raises:
            ComponentError: If the entity type does not exist on this domain,
                for instance a ridge on an interval mesh.
        """
        if entity_type == "vertex":
            return 0
        if entity_type not in _CODIMENSION:
            raise ComponentError(
                f"Unknown entity type {entity_type!r}, expected one of "
                f"{sorted([*_CODIMENSION, 'vertex'])}."
            )
        dim = self.tdim - _CODIMENSION[entity_type]
        if dim < 0:
            raise ComponentError(
                f"Domain {self._name!r} has topological dimension {self.tdim} "
                f"and therefore no entities of type {entity_type!r}."
            )
        return dim

    def marker_ids(
        self, markers: str | typing.Sequence[str], entity_type: EntityType
    ) -> tuple[int, ...]:
        """Return the integer ids of named markers, assigning ids to new names.

        The ids are numbered separately per entity dimension, so entity types
        that coincide share their markers: on a triangle mesh a ridge is a
        vertex, and ``"A"`` has the same id for both.

        Args:
            markers: Name of a subdomain, e.g. ``"Electrode"``, or a sequence
                of names.
            entity_type: The entities the markers refer to.

        Returns:
            One id per marker, in the given order.
        """
        if isinstance(markers, str):
            markers = (markers,)
        ids = self._markers[self.entity_dimension(entity_type)]
        for marker in markers:
            check_name(marker, "Marker")
            if marker not in ids:
                ids[marker] = len(ids) + 1
        return tuple(ids[marker] for marker in markers)

    @property
    def markers(self) -> dict[int, dict[str, int]]:
        """Copy of the marker name to integer id maps, per entity dimension."""
        return {dim: dict(ids) for dim, ids in self._markers.items()}

    def measure(
        self,
        integral_type: MeasureName,
        *markers: str,
        degree: int | None = None,
        metadata: dict | None = None,
    ) -> ufl.Measure:
        """Create a measure over the whole domain or over named subdomains.

        Subdomains are referred to by marker name, e.g.
        ``domain.measure("dx", "Electrode")``, or with the shortcuts
        ``domain.dx("Electrode")`` and ``domain.ds("Inlet")``. Entity types
        that coincide share their markers, see :meth:`marker_ids`: on a
        triangle mesh ``dr("A")`` and ``dP("A")`` refer to the same vertices.

        Args:
            integral_type: One of ``"dx"``, ``"ds"``, ``"dS"``, ``"dr"`` or
                ``"dP"``.
            markers: Names of the subdomains to integrate over. With no
                names, the measure covers the whole domain.
            degree: Quadrature degree. Shorthand for
                ``metadata={"quadrature_degree": degree}``.
            metadata: Form compiler metadata for the measure.
        """
        if integral_type not in MEASURE_ENTITY_TYPE:
            raise ComponentError(
                f"Unknown integral type {integral_type!r}, "
                f"expected one of {sorted(MEASURE_ENTITY_TYPE)}."
            )
        entity_type = MEASURE_ENTITY_TYPE[integral_type]
        base = self._measures.get(integral_type)
        if base is None:
            # Fails for measures the domain cannot have, e.g. dr on intervals
            self.entity_dimension(entity_type)
            base = ufl.Measure(integral_type, domain=self)
            self._measures[integral_type] = base
        if len(markers) == 0 and degree is None and metadata is None:
            return base
        # A subdomain id of None keeps the whole domain, and UFL merges the
        # degree into the metadata
        subdomain_id = self.marker_ids(markers, entity_type) if markers else None
        return base(subdomain_id, metadata=metadata, degree=degree)

    # Shortcuts for measure() with the integral type bound. Each takes
    # (*markers, degree=None, metadata=None).
    dx = functools.partialmethod(measure, "dx")
    """Cell measure over the domain, or over the named cell markers."""

    ds = functools.partialmethod(measure, "ds")
    """Exterior facet measure, optionally restricted to named facet markers."""

    dS = functools.partialmethod(measure, "dS")
    """Interior facet measure, optionally restricted to named facet markers."""

    dr = functools.partialmethod(measure, "dr")
    """Ridge measure (codimension 2), optionally restricted to named ridges."""

    dP = functools.partialmethod(measure, "dP")
    """Vertex measure, optionally restricted to named vertex markers."""

    def one_sided(self, marker: str) -> tuple[Subdomain, str | None] | None:
        """The subdomain a facet marker sees its facets from, if one-sided.

        Facet integrals of a subdomain of cells move to its parent as
        integrals over facets seen from the subdomain's side, see
        :meth:`Subdomain.on_parent`, under markers named
        ``"<subdomain>.<marker>"``. Inside the parent, such facets may have
        cells on several sides, e.g. at the joint of a branching surface.

        Returns:
            The subdomain and its own facet marker, ``None`` for its whole
            boundary, or ``None`` if the marker is not one-sided.
        """
        return self._one_sided.get(marker)

    def interface(
        self,
        first: Subdomain,
        second: Subdomain,
        marker: str,
        degree: int | None = None,
        metadata: dict | None = None,
    ) -> ufl.Measure:
        """Interior facet measure of an interface between two subdomains.

        Its ``"+"`` side is always in ``first`` and its ``"-"`` side in
        ``second``, so the functions of each subdomain can be restricted to
        their own side, e.g. ``u_0("+")`` and ``u_1("-")`` for an exchange
        flux between two materials.

        Args:
            first: The subdomain of cells on the ``"+"`` side.
            second: The subdomain of cells on the ``"-"`` side.
            marker: This domain's facet marker of the interface.
            degree: Quadrature degree, see :meth:`measure`.
            metadata: Form compiler metadata, see :meth:`measure`.

        Raises:
            ComponentError: If a subdomain is not a subdomain of cells of this
                domain.
        """
        for subdomain in (first, second):
            if not isinstance(subdomain, Subdomain) or subdomain.parent is not self:
                raise ComponentError(
                    f"{subdomain} is not a subdomain of {self._name!r}, so it has "
                    "no side of its interfaces."
                )
            if subdomain.codim != 0:
                raise ComponentError(
                    f"Subdomain {subdomain.name!r} is not a subdomain of cells, "
                    "so it has no side of an interface."
                )
        name = f"{first.name}.{second.name}.{marker}"
        self._interfaces[name] = (first, second, marker)
        return self.measure("dS", name, degree=degree, metadata=metadata)

    def interface_sides(self, marker: str) -> tuple[Subdomain, Subdomain, str] | None:
        """The subdomains and facet marker of an interface measure's marker.

        See :meth:`interface`. ``None`` if the marker is not an interface.
        """
        return self._interfaces.get(marker)

    def __str__(self) -> str:
        return f"<{type(self).__name__} {self._name!r} ({self.cell_name})>"


class Subdomain(Domain):
    """Marked entities of a domain, as a domain of their own.

    Fields on a subdomain live on a submesh of a subset of the parent's mesh
    entities, those with the subdomain's marker. These are cells, e.g. the
    electrolyte of a cell, or facets, e.g. the concentration of a species on
    a surface, or the unknown of a PDE on a manifold.
    ``dx`` integrates over the subdomain, ``ds`` over its boundary and
    ``dS`` over its interior facets. Markers are named after the parent's:
    those of ``dx`` are parent markers of the subdomain's dimension, and
    those of ``ds`` and ``dS`` parent markers of one dimension lower. For a
    surface of a 3D domain, these are facet and ridge markers. Its
    coordinates are the parent's, and so is its coordinate system.

    Whether a form is integrated on the submesh or on the parent is decided
    per form, see :func:`place_subdomain_integrals`.

    Args:
        name: Name of the domain.
        parent: The domain the entities belong to.
        dim: Topological dimension of the entities, e.g. ``parent.tdim - 1``
            for facets.
        marker: The parent's marker of the entities, e.g. ``"Membrane"``.
        length_scale: See :class:`Domain`. Defaults to the parent's.
    """

    _parent: Domain
    """The domain the entities belong to."""

    _marker: str
    """The parent's marker of the entities."""

    def __init__(
        self,
        name: str,
        parent: Domain,
        dim: int,
        marker: str,
        length_scale: float | None = None,
    ):
        if not 1 <= dim <= parent.tdim:
            raise ComponentError(
                f"Subdomains of {parent.name!r} have dimension 1 to {parent.tdim}, "
                f"got {dim}."
            )
        cell = _entity_cell(parent.cell_name, dim)
        element_ = parent.ufl_coordinate_element()
        super().__init__(
            name,
            cell,
            gdim=parent.gdim,
            degree=element_.embedded_superdegree,
            length_scale=parent.length_scale if length_scale is None else length_scale,
            lagrange_variant=getattr(element_, "lagrange_variant", None),
        )
        self._parent = parent
        self._marker = marker
        parent.marker_ids(marker, _entity_type(parent, dim))

    @property
    def parent(self) -> Domain:
        """The domain the entities belong to."""
        return self._parent

    @property
    def marker(self) -> str:
        """The parent's marker of the entities."""
        return self._marker

    @property
    def codim(self) -> int:
        """Codimension of the subdomain in its parent."""
        return self._parent.tdim - self.tdim

    @property
    def coordinates(self) -> CoordinateSystem:
        """The parent's coordinate system, since the coordinates are the same."""
        return self._parent.coordinates

    def measure(
        self,
        integral_type: MeasureName,
        *markers: str,
        degree: int | None = None,
        metadata: dict | None = None,
    ) -> ufl.Measure:
        """A measure of this domain, as :meth:`Domain.measure`.

        The markers are the parent's markers of the measure's entities,
        which have the same dimension in the parent: e.g. for a surface of
        a 3D domain, facet markers for ``dx``, ridge markers for ``ds`` and
        ``dS``, and vertex markers for ``dr``.
        """
        if markers:
            dim = self.entity_dimension(MEASURE_ENTITY_TYPE[integral_type])
            self._parent.marker_ids(markers, _entity_type(self._parent, dim))
        return super().measure(
            integral_type, *markers, degree=degree, metadata=metadata
        )

    # Bound again, so that they reach the override
    dx = functools.partialmethod(measure, "dx")
    """Cell measure of this domain, optionally restricted to named parts."""

    ds = functools.partialmethod(measure, "ds")
    """Measure of this domain's boundary, optionally of named parts of it."""

    dS = functools.partialmethod(measure, "dS")
    """Measure of this domain's interior facets, optionally of named ones."""

    dr = functools.partialmethod(measure, "dr")
    """Ridge measure of this domain, optionally restricted to named ridges."""

    dP = functools.partialmethod(measure, "dP")
    """Vertex measure of this domain, optionally restricted to named vertices."""

    def on_parent(self, integral: ufl.Integral) -> ufl.Integral:
        """An integral of this domain, as one on the parent's entities.

        A cell integral moves to the parent's cells for a subdomain of
        cells, and to its exterior facets for a subdomain of facets. For a
        subdomain of facets between two subdomains of cells, an interface
        of the parent, see :meth:`Domain.interface`, an integral over the
        whole subdomain moves to the interior facets of the interface, on
        its ``"+"`` side, which holds the same values of the facets'
        functions as the other. An
        exterior facet integral of a subdomain of cells moves to the
        parent's facets seen from the subdomain's side, under a one-sided
        marker, see :meth:`Domain.one_sided`: e.g. the joint of a branching
        surface, seen from one branch. The functions and the geometry of
        the submesh in it, e.g. its spatial coordinate, are mapped by the
        entity map, and its facet normal is the parent's.

        Raises:
            ComponentError: For other integrals, or subdomains of codimension
                above one.
        """
        integral_type = integral.integral_type()
        if integral_type == "exterior_facet" and self.codim == 0:
            # UFL takes facet geometry only from the integration domain. The
            # cells are the parent's, so the normal is the same.
            integrand = ufl.replace(
                integral.integrand(),
                {ufl.FacetNormal(self): ufl.FacetNormal(self._parent)},
            )
            return integral.reconstruct(
                integrand=integrand,
                domain=self._parent,
                subdomain_id=self._one_sided_ids(integral.subdomain_id()),
            )
        if integral_type != "cell":
            raise ComponentError(
                f"The {integral_type} integrals of subdomain {self.name!r}, of "
                f"codimension {self.codim}, cannot be integrated on the parent."
            )
        parent_type = {0: "cell", 1: "exterior_facet"}.get(self.codim)
        if parent_type is None:
            raise ComponentError(
                f"Subdomain {self.name!r} has codimension {self.codim}, and "
                "DOLFINx cannot integrate its functions on the parent."
            )
        entity_type = _entity_type(self._parent, self.tdim)
        subdomain_id = integral.subdomain_id()
        interface = self._interface()
        if interface is not None and subdomain_id in ("everywhere", "otherwise"):
            return integral.reconstruct(
                integrand=integral.integrand()("+"),
                integral_type="interior_facet",
                domain=self._parent,
                subdomain_id=self._parent.marker_ids(interface, entity_type),
            )
        if subdomain_id in ("everywhere", "otherwise"):
            ids = self._parent.marker_ids(self._marker, entity_type)
        else:
            names = {i: name for name, i in self._markers[self.tdim].items()}
            ids_ = subdomain_id if isinstance(subdomain_id, tuple) else (subdomain_id,)
            ids = self._parent.marker_ids([names[i] for i in ids_], entity_type)
        return integral.reconstruct(
            integral_type=parent_type,
            domain=self._parent,
            subdomain_id=ids,
        )

    def _interface(self) -> str | None:
        """The parent's interface on this subdomain's facets, if it has one."""
        if self.codim != 1:
            return None
        for name, (_, _, marker) in self._parent._interfaces.items():
            if marker == self._marker:
                return name
        return None

    def _one_sided_ids(self, subdomain_id: typing.Any) -> tuple[int, ...]:
        """Parent ids of one-sided markers for this domain's facet markers."""
        if subdomain_id in ("everywhere", "otherwise"):
            markers: tuple[str | None, ...] = (None,)
        else:
            names = {i: name for name, i in self._markers[self.tdim - 1].items()}
            ids = subdomain_id if isinstance(subdomain_id, tuple) else (subdomain_id,)
            markers = tuple(names[i] for i in ids)
        parent_names = []
        for marker in markers:
            name = f"{self.name}.{'boundary' if marker is None else marker}"
            self._parent._one_sided[name] = (self, marker)
            parent_names.append(name)
        return self._parent.marker_ids(parent_names, "facet")


def place_subdomain_integrals(form: ufl.Form) -> ufl.Form:
    """Place the integrals of subdomains on a mesh DOLFINx can assemble.

    A form is assembled over one mesh. Integrals of a :class:`Subdomain`
    stay on its submesh when the form involves nothing else. When the form
    also has integrals of the parent, or the integrands use functions of
    the parent, the cell integrals move to the parent, see
    :meth:`Subdomain.on_parent`, which gives the same values.

    Raises:
        ComponentError: If such a form also has other integrals of a
            subdomain, e.g. over the facets of a subdomain of facets.
            DOLFINx cannot integrate parent functions on a submesh, nor
            functions of a submesh of facets on the parent's ridges, so the
            two kinds cannot share a form.
    """
    integrals = form.integrals()
    if not any(isinstance(i.ufl_domain(), Subdomain) for i in integrals):
        return form
    native = len({i.ufl_domain() for i in integrals}) == 1 and all(
        set(extract_domains(i.integrand())) <= {i.ufl_domain()} for i in integrals
    )
    if native:
        return form
    placed = []
    for integral in integrals:
        domain = integral.ufl_domain()
        if not isinstance(domain, Subdomain):
            placed.append(integral)
        elif integral.integral_type() == "cell" or (
            integral.integral_type() == "exterior_facet" and domain.codim == 0
        ):
            placed.append(domain.on_parent(integral))
        else:
            raise ComponentError(
                f"A form integrates over {integral.integral_type()} entities of "
                f"{domain.name!r} and involves the parent {domain.parent.name!r}. "
                "DOLFINx assembles a form over one mesh, and cannot integrate "
                "parent functions on a submesh, or functions of a submesh of "
                "facets on the parent's ridges. Write the coupling to the parent "
                "and these terms of the subdomain into different equations."
            )
    return ufl.Form(placed)


def _entity_cell(cell_name: str, dim: int) -> CellType:
    """The cell type of the entities of a dimension of a cell type.

    Raises:
        ComponentError: If those entities are not all of one type, e.g. the
            facets of a prism.
    """
    cell = dolfinx.mesh.to_type(cell_name)
    types = {
        dolfinx.mesh.cell_entity_type(cell, dim, i)
        for i in range(dolfinx.mesh.cell_num_entities(cell, dim))
    }
    if len(types) != 1:
        raise ComponentError(
            f"The entities of dimension {dim} of {cell_name!r} cells are not all "
            "of one type."
        )
    return typing.cast(CellType, types.pop().name)


def _entity_type(domain: Domain, dim: int) -> EntityType:
    """The entity type of a dimension of a domain, named by codimension."""
    codim_types: tuple[EntityType, ...] = ("cell", "facet", "ridge", "peak")
    return codim_types[domain.tdim - dim]


def domain_of(space: ufl.FunctionSpace) -> Domain:
    """The :class:`Domain` a symbolic function space is defined on.

    Raises:
        RuntimeError: If the space is defined on a mesh that is not a
            :class:`Domain`.
    """
    domain = space.ufl_domain()
    if not isinstance(domain, Domain):
        raise RuntimeError(
            f"Function space {space!r} is defined on {domain!r}, "
            f"which is not a {qualified_name(Domain)}."
        )
    return domain
