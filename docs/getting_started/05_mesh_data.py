# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Meshes and their markers
#
# This page illustrates how to:
#
# - Create a {py:class}`sumfem.discrete.MeshData`, a mesh with named markers,
#   from locator functions, from existing mesh tags, and from a Gmsh model
# - Look up the entities of a marker
#
# A model refers to parts of its domain by marker names, e.g. a Dirichlet
# condition on `"Boundary"` or an integral `domain.ds("Inlet")`. A
# {py:class}`MeshData <sumfem.discrete.MeshData>` stores a
# {py:class}`dolfinx.mesh.Mesh` with its
# {py:class}`MeshTags <dolfinx.mesh.MeshTags>`, and for each entity dimension
# the tag values of each marker name. Below, each way of creating one marks the
# boundary of the unit square `"Boundary"`.

# +
from mpi4py import MPI

import dolfinx.io.gmsh
import dolfinx.mesh
import gmsh
import numpy as np

from sumfem import discrete

# -

# ## From locator functions
#
# {py:meth}`from_locators <sumfem.discrete.MeshData.from_locators>` takes, for
# each entity dimension and marker name, a function of the coordinates as
# {py:func}`dolfinx.mesh.locate_entities` takes. With `boundary=True`, the
# facets are only looked for on the boundary. Here the cells of the left half
# are marked too:

square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
tdim = square.topology.dim
from_locators = discrete.MeshData.from_locators(
    square,
    {
        tdim: {"LeftHalf": lambda x: x[0] <= 0.5 + 1e-10},
        tdim - 1: {"Boundary": lambda x: np.full(x.shape[1], True)},
    },
    boundary=True,
)

# Each marker name stands for tag values in the mesh tags, which
# `from_locators` chooses itself: a model only uses the names. What a name
# marks is looked up with {py:meth}`entities <sumfem.discrete.MeshData.entities>`,
# which gives the local indices of its entities. Here we print how many there
# are on this process:


def print_markers(mesh_data):
    """The name, entity dimension and number of local entities of each marker."""
    for dim, names in sorted(mesh_data.markers.items()):
        for name in names:
            count = mesh_data.entities(dim, name).size
            print(f"{name!r}: {count} entities of dimension {dim}")


print_markers(from_locators)

# ## From existing mesh tags
#
# Mesh tags made with DOLFINx, or read from a file, come with tag values of
# their own, here `boundary_tag` on every boundary facet.
# {py:meth}`from_meshtags <sumfem.discrete.MeshData.from_meshtags>` names them:
# for each mesh tags, the tag values of each marker name, here `"Boundary"` for
# `boundary_tag`. The dimension of a marker is that of its mesh tags, and a
# marker may stand for several values, e.g. `{"Walls": (2, 3)}`.

boundary_tag = 1
square.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(square.topology)
facet_tags = dolfinx.mesh.meshtags(
    square, tdim - 1, boundary_facets, np.full_like(boundary_facets, boundary_tag)
)
from_tags = discrete.MeshData.from_meshtags(
    square, {facet_tags: {"Boundary": (boundary_tag,)}}
)

# The tag values differ from those `from_locators` chose, but the name marks
# the same facets of the same mesh:

print_markers(from_tags)
boundary = np.sort(from_locators.entities(tdim - 1, "Boundary"))
assert np.array_equal(boundary, np.sort(from_tags.entities(tdim - 1, "Boundary")))

# ## From a Gmsh model
#
# The physical groups of a mesh made with [Gmsh](https://gmsh.info) become
# markers with the groups' names, with
# {py:meth}`from_gmsh <sumfem.discrete.MeshData.from_gmsh>`.
# {py:func}`dolfinx.io.gmsh.model_to_mesh` turns the Gmsh model, made on one
# process, into a distributed mesh with its tags; a mesh read with
# {py:func}`dolfinx.io.gmsh.read_from_msh` is used the same way.

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
if MPI.COMM_WORLD.rank == 0:
    gmsh.model.add("square")
    surface = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    sides = [tag for _, tag in gmsh.model.getBoundary([(2, surface)], oriented=False)]
    gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="Square")
    gmsh.model.addPhysicalGroup(1, sides, tag=2, name="Boundary")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1 / 16)
    gmsh.model.mesh.generate(2)
gmsh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
from_gmsh = discrete.MeshData.from_gmsh(gmsh_data)

# The mesh is another one, but its boundary is marked by the same name, so a
# model written with `"Boundary"` is discretized on any of these:

print_markers(from_gmsh)
