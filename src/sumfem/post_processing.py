"""Post-processing of DOLFINx functions: integrals, norms, lines and ranges."""

import enum
import typing

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem
import dolfinx.geometry
import numpy as np
import numpy.typing as npt
import ufl
from ufl.domain import extract_unique_domain

from .symbolic.coordinates import Cartesian, CoordinateSystem, lift

__all__ = [
    "create_line",
    "evaluate_at_points",
    "evaluate_over_line",
    "function_range",
    "NormType",
    "Squared",
    "error",
    "integrate",
    "interpolate_to_line",
    "norm",
]


def create_line(
    p0: npt.NDArray[np.floating],
    p1: npt.NDArray[np.floating],
    num_points: int,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> dolfinx.mesh.Mesh:
    """A mesh of interval cells on the line segment from ``p0`` to ``p1``.

    The points are created on process 0, and DOLFINx distributes the cells.

    Args:
        p0: The first point, with as many coordinates as the line's geometry.
        p1: The last point.
        num_points: Number of points, including both ends.
        comm: The communicator of the line mesh.
    """
    _p0 = np.asarray(p0)
    _p1 = np.asarray(p1)
    assert _p0.shape == _p1.shape, "Points must have the same dimension"
    if comm.rank == 0:
        weights = np.linspace(0, 1, num_points, endpoint=True)
        points = (_p0 + np.outer(weights, (_p1 - _p0))).astype(_p0.dtype)
        cells = (
            np.repeat(np.arange(points.shape[0]), 2)[1:-1]
            .reshape(points.shape[0] - 1, 2)
            .astype(np.int64)
        )
    else:
        points = np.empty((0, _p0.shape[0]), dtype=np.float64)
        cells = np.empty((0, 2), dtype=np.int64)
    c_el = basix.ufl.element("Lagrange", "interval", 1, shape=(_p0.shape[0],))
    return dolfinx.mesh.create_mesh(comm, cells=cells, x=points, e=ufl.Mesh(c_el))


def interpolate_to_line(
    expr: ufl.core.expr.Expr,
    p0: npt.NDArray[np.floating],
    p1: npt.NDArray[np.floating],
    num_points: int,
    element: dolfinx.fem.ElementMetaData,
) -> dolfinx.fem.Function:
    """Interpolate an expression into a function on a line segment.

    The line is ``L(alpha) = p0 + alpha (p1 - p0)`` for ``alpha`` in
    ``[0, 1]``, meshed by :func:`create_line`. The expression is first
    interpolated into a space of the element on its own mesh, and then
    moved to the line by non-matching interpolation.

    Args:
        expr: A UFL expression on a single mesh.
        p0: The first point of the line.
        p1: The last point of the line.
        num_points: Number of points on the line, including both ends.
        element: The element of the function on the line, and of the
            intermediate space on the expression's mesh.

    Returns:
        The function on the line mesh.

    Raises:
        ValueError: If the expression is not on a single mesh.
    """
    line_mesh = create_line(p0, p1, num_points)
    ufl_domain = ufl.domain.extract_unique_domain(expr)
    if not isinstance(ufl_domain, ufl.Mesh):
        raise ValueError(f"{expr!r} is not defined on a single mesh.")
    mesh = dolfinx.mesh.Mesh(ufl_domain.ufl_cargo(), ufl_domain)
    input_space = dolfinx.fem.functionspace(mesh, element)
    input_expr = dolfinx.fem.Expression(expr, input_space.element.interpolation_points)
    input_func = dolfinx.fem.Function(input_space)
    input_func.interpolate(input_expr)

    V_line = dolfinx.fem.functionspace(line_mesh, element)
    eta_line = dolfinx.fem.Function(V_line)

    fine_mesh_cell_map = line_mesh.topology.index_map(line_mesh.topology.dim)
    num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = dolfinx.fem.create_interpolation_data(
        V_line, input_space, cells, padding=1e-6
    )

    eta_line.interpolate_nonmatching(input_func, cells, interpolation_data)
    eta_line.x.scatter_forward()
    return eta_line


def evaluate_at_points(
    function: dolfinx.fem.Function, points: npt.ArrayLike
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """Evaluate a function at points, each on the process that owns it.

    :func:`dolfinx.geometry.determine_point_ownership` gives each point in
    the mesh one owning process, also on the boundary between processes.
    Points outside the mesh are evaluated nowhere. Collective.

    Args:
        function: The function.
        points: The points, of shape ``(n, gdim)``, the same on every process.

    Returns:
        The indices of the points evaluated on this process, increasing,
        and the values there, of shape ``(len(indices), value size)``.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    padded = np.zeros((len(coordinates), 3))
    padded[:, : coordinates.shape[1]] = coordinates
    mesh = function.function_space.mesh
    comm = mesh.comm
    # Sent from one process only, so each owner receives each point once
    sent = padded if comm.rank == 0 else np.zeros((0, 3))
    ownership = dolfinx.geometry.determine_point_ownership(
        mesh, sent, padding=1e-8, find_closest_cell=False
    )
    owners = comm.bcast(np.asarray(ownership.src_owner) if comm.rank == 0 else None)
    found = np.flatnonzero(owners == comm.rank).astype(np.int32)
    size = int(np.prod(function.ufl_shape))
    if len(found) == 0:
        return found, np.zeros((0, size))
    cells = np.asarray(ownership.dest_cells, dtype=np.int32)
    values = function.eval(padded[found], cells)
    return found, np.asarray(values).reshape(len(found), size)


def evaluate_over_line(
    function: dolfinx.fem.Function,
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    num_points: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate a function at equidistant points on a line segment.

    Each point is evaluated on the process that owns it, see
    :func:`evaluate_at_points`. Collective.

    Args:
        function: The function.
        start: The first point, with as many coordinates as the mesh.
        end: The last point.
        num_points: Number of points, including both ends.

    Returns:
        The points on this process, of shape ``(n, 3)``, and the values
        there, of shape ``(n, value size)``.
    """
    weights = np.linspace(0.0, 1.0, num_points)
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    points = np.zeros((num_points, 3))
    points[:, : a.size] = np.outer(1 - weights, a) + np.outer(weights, b)
    indices, values = evaluate_at_points(function, points)
    return points[indices], values


def function_range(
    function: dolfinx.fem.Function,
    entities: npt.NDArray[np.int32] | None = None,
    dim: int | None = None,
) -> tuple[float, float]:
    """The global minimum and maximum of the values of a scalar function.

    These are the extreme values of the degrees of freedom, which bound the
    function for Lagrange elements of degree one or zero.

    Args:
        function: The function.
        entities: Local indices of mesh entities whose degrees of freedom
            bound the range, e.g. the cells of a subdomain or the facets of
            a surface. ``None`` for the whole mesh.
        dim: Topological dimension of the entities. Defaults to the cells.

    Raises:
        ValueError: For a function with several components.
    """
    space = function.function_space
    if space.dofmap.index_map_bs != 1:
        raise ValueError(
            f"Function {function.name!r} has {space.dofmap.index_map_bs} components."
        )
    num_owned = space.dofmap.index_map.size_local
    if entities is None:
        owned = function.x.array[:num_owned].real
    else:
        mesh = space.mesh
        dim = mesh.topology.dim if dim is None else dim
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        dofs = dolfinx.fem.locate_dofs_topological(space, dim, entities)
        owned = function.x.array[dofs[dofs < num_owned]].real
    local_min = float(owned.min()) if owned.size else np.inf
    local_max = float(owned.max()) if owned.size else -np.inf
    comm = space.mesh.comm
    return comm.allreduce(local_min, op=MPI.MIN), comm.allreduce(local_max, op=MPI.MAX)


Squared = typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr]
"""A custom norm: the square of the integrand of a quantity ``f``."""


class NormType(enum.StrEnum):
    """The kind of a :func:`norm`. Plain strings with these values work too."""

    l2 = "l2"
    """The Euclidean norm of the degrees of freedom of a function."""

    L2 = "L2"
    """``sqrt((f, f))``."""

    H1 = "H1"
    """``sqrt((f, f) + (grad f, grad f))``."""

    H10 = "H10"
    """The seminorm ``sqrt((grad f, grad f))``."""

    Hdiv = "Hdiv"
    """``sqrt((f, f) + (div f, div f))``, for vectors."""

    Hcurl = "Hcurl"
    """``sqrt((f, f) + (curl f, curl f))``, for vectors."""


def integrate(
    expr: ufl.core.expr.Expr | float,
    mesh: dolfinx.mesh.Mesh | None = None,
    coordinates: CoordinateSystem | None = None,
) -> float:
    """The integral of an integrand over its whole mesh, on every process.

    Args:
        expr: A scalar integrand of DOLFINx functions, constants and the
            mesh's coordinates, e.g. the square of an error.
        mesh: The mesh, needed when the integrand does not name one, such
            as a number to measure the mesh's volume.
        coordinates: The coordinate system of the mesh, which weights the
            integral and writes its vectors, e.g. that of the domain the mesh
            belongs to. Cartesian by default.
    """
    integrand = ufl.as_ufl(expr)
    domain = mesh.ufl_domain() if mesh is not None else extract_unique_domain(integrand)
    if domain is None:
        raise ValueError("Give the mesh to integrate an integrand without one over.")
    coordinates = Cartesian() if coordinates is None else coordinates
    x = ufl.SpatialCoordinate(domain)
    integrand = coordinates.weight(x) * lift(coordinates, integrand, x)
    form = dolfinx.fem.form(integrand * ufl.dx(domain=domain))
    return form.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def norm(
    expr: ufl.core.expr.Expr,
    kind: NormType | str | Squared = NormType.L2,
    mesh: dolfinx.mesh.Mesh | None = None,
    coordinates: CoordinateSystem | None = None,
) -> float:
    """A norm of an expression over its whole mesh, on every process.

    Args:
        expr: A scalar, vector or tensor expression, e.g. an error.
        kind: Which norm, see :class:`NormType`. For ``l2``, ``expr`` must
            be a function. A callable gives a custom norm: it takes ``f``
            and returns the integrand of the squared norm, e.g.
            ``lambda f: w * ufl.inner(f, f)`` for an L2 norm with the weight
            ``w``.
        mesh: See :func:`integrate`.
        coordinates: See :func:`integrate`. It does not weight ``l2``.

    Raises:
        TypeError: For ``"l2"`` of anything but a function.
        ValueError: For an unknown kind.
    """
    if callable(kind):
        return float(np.sqrt(integrate(kind(expr), mesh, coordinates)))
    try:
        kind = NormType(kind)
    except ValueError:
        known = ", ".join(repr(k.value) for k in NormType)
        raise ValueError(f"Unknown norm {kind!r}, expected one of {known}.") from None
    if kind is NormType.l2:
        if not isinstance(expr, dolfinx.fem.Function):
            raise TypeError("The l2 norm is that of a function's degrees of freedom.")
        dofmap = expr.function_space.dofmap
        owned = expr.x.array[: dofmap.index_map.size_local * dofmap.index_map_bs]
        comm = expr.function_space.mesh.comm
        return float(np.sqrt(comm.allreduce(np.sum(np.abs(owned) ** 2), op=MPI.SUM)))
    squares = {
        NormType.L2: lambda f: ufl.inner(f, f),
        NormType.H1: lambda f: ufl.inner(f, f) + ufl.inner(ufl.grad(f), ufl.grad(f)),
        NormType.H10: lambda f: ufl.inner(ufl.grad(f), ufl.grad(f)),
        NormType.Hdiv: lambda f: ufl.inner(f, f) + ufl.div(f) ** 2,
        NormType.Hcurl: lambda f: ufl.inner(f, f) + ufl.inner(ufl.curl(f), ufl.curl(f)),
    }
    return float(np.sqrt(integrate(squares[kind](expr), mesh, coordinates)))


def error(
    function: dolfinx.fem.Function,
    exact: ufl.core.expr.Expr
    | typing.Callable[[ufl.SpatialCoordinate], ufl.core.expr.Expr],
    kind: NormType | str | Squared = NormType.L2,
    relative: bool = False,
    coordinates: CoordinateSystem | None = None,
) -> float:
    """The error of a function against an exact solution, in a :func:`norm`.

    Args:
        function: The approximation.
        exact: The exact solution, as an expression on the function's mesh
            or as a callable of the mesh's spatial coordinate.
        kind: The norm, see :class:`NormType`. Not ``l2``.
        relative: Divide by the norm of the exact solution.
        coordinates: See :func:`integrate`.
    """
    mesh = function.function_space.mesh
    if callable(exact) and not isinstance(exact, ufl.core.expr.Expr):
        exact = exact(ufl.SpatialCoordinate(mesh))
    if isinstance(kind, str) and kind == NormType.l2:
        raise ValueError("The error of an exact solution has no degrees of freedom.")
    value = norm(function - exact, kind, mesh, coordinates)
    return value / norm(exact, kind, mesh, coordinates) if relative else value
