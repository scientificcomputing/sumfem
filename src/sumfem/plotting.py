"""Plotting with `PyVista <https://pyvista.org>`_, on any number of processes.

Each process holds the cells it owns, so a plot of a distributed function
gathers them on one process: :func:`gather_grid` builds a PyVista grid of
the owned cells on each process, with the function's values at its points,
and merges the grids on one. PyVista is an optional dependency, imported
when a grid is built.
"""

from __future__ import annotations

import typing

import dolfinx.fem
import dolfinx.plot
import numpy as np

if typing.TYPE_CHECKING:
    import pyvista

__all__ = ["gather_grid"]


def _pyvista() -> typing.Any:
    """PyVista, or an error saying how to install it."""
    try:
        import pyvista
    except ImportError as error:
        raise ImportError(
            "Plotting needs PyVista, e.g. pip install sumfem[docs]."
        ) from error
    return pyvista


def gather_grid(
    function: dolfinx.fem.Function, name: str | None = None, root: int = 0
) -> pyvista.UnstructuredGrid | None:
    """A PyVista grid of a function, gathered on one process.

    The grid has the cells of the function space, e.g. of degree two for a
    quadratic Lagrange space, and the function's values at its points. Each
    process contributes the cells it owns, so every cell appears once. A
    vector is padded to three components, as PyVista expects, e.g. for
    ``warp_by_vector``.

    Every process must call this, as it is collective.

    Args:
        function: A function on a Lagrange space, of any shape, on a mesh
            or a submesh, e.g. a multiplier on facets.
        name: Name of the values in the grid's ``point_data``. Defaults to
            the function's name.
        root: The process to gather on.

    Returns:
        The grid on ``root``, ``None`` on the other processes.
    """
    pyvista = _pyvista()
    space = function.function_space
    mesh = space.mesh
    tdim = mesh.topology.dim
    owned = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(space, entities=owned))
    # The points are the degrees of freedom of the process, in order, each
    # with a block of components
    components = space.dofmap.index_map_bs
    values = function.x.array.real.reshape(-1, components)
    if values.shape[1] in (2, 3):
        padding = np.zeros((values.shape[0], 3 - values.shape[1]))
        values = np.hstack([values, padding])
    elif values.shape[1] == 1:
        values = values[:, 0]
    grid.point_data[function.name if name is None else name] = values
    grids = mesh.comm.gather(grid, root=root)
    if grids is None:
        return None
    pieces = [g for g in grids if g.n_cells > 0]
    if len(pieces) <= 1:
        return pieces[0] if pieces else pyvista.UnstructuredGrid()
    return pyvista.merge(pieces, merge_points=False)
