"""Binding symbolic components to meshes, as their discrete counterparts.

- :class:`MeshData`: one mesh with its mesh tags, and names for its
  markers. It knows nothing of symbolic domains.
- :class:`DomainMaps`: pairs each symbolic domain with a :class:`MeshData`,
  checks that they match, and creates the submeshes of subdomains.
- :func:`discretize`: the :class:`Discretization` of symbolic components on
  domain maps, holding one DOLFINx function per terminal, shared by all of
  them. ``disc[component]`` gives a component's discrete counterpart.
- :class:`System`: the residual, Jacobian and preconditioner of a symbolic
  system, compiled and bound to the meshes and functions.
- :class:`DirichletBC`: a symbolic boundary condition on the meshes.
- :func:`counterpart`: the discrete counterpart of a symbolic object, with
  options.

Solving the counterparts is left to :mod:`sumfem.solvers`.

A discrete counterpart has the name of its symbolic one, and is made by
indexing a discretization, by :func:`counterpart`, or by its own
constructor from the discretization and the symbolic object. ``sumfem``
exports the symbolic ones.
"""

from .bcs import DirichletBC
from .counterpart import counterpart
from .discretization import Discretization, discretize
from .domain_maps import DomainMaps
from .meshes import MeshData, ghost_by_vertex
from .system import System

__all__ = [
    "System",
    "Discretization",
    "DomainMaps",
    "MeshData",
    "ghost_by_vertex",
    "DirichletBC",
    "counterpart",
    "discretize",
]
