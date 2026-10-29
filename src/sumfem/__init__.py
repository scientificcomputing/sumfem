"""Finite element models written as summable components, on FEniCSx.

The generic framework has three layers, imported as subpackages:

- :mod:`sumfem.symbolic`, the symbolic description, without mesh data:
  domains, fields, parameters, residuals and functionals, systems of
  equations, sequences of stages, time derivatives, boundary conditions,
  transfers and dimensional analysis.
- :mod:`sumfem.discrete`, which binds them to meshes: discretizations,
  and the discrete systems and boundary conditions.
- :mod:`sumfem.solvers`, which solves the discrete systems.

The names users need most are exported here too. Where a symbolic and a
discrete class share a name, such as ``System`` or ``DirichletBC``, the
symbolic one is exported.

Battery physics and the other model families live in
:mod:`sumfem.models`, and post-processing in
:mod:`sumfem.post_processing`::

    import sumfem

    domain = sumfem.Domain("square", "triangle")
    u = sumfem.Field("u", domain, ("Lagrange", 1))
"""

from . import discrete, solvers, symbolic
from .discrete import (
    Discretization,
    DomainMaps,
    MeshData,
    counterpart,
    discretize,
    ghost_by_vertex,
)
from .solvers import Solver
from .symbolic import (
    Assignment,
    Cartesian,
    Clock,
    Component,
    ComponentError,
    CoordinateSystem,
    Cylindrical,
    DirichletBC,
    Domain,
    Dt,
    Field,
    Functional,
    FunctionalSum,
    Interpolation,
    Jacobian,
    JacobianSum,
    KnownCoefficient,
    Parameter,
    PiGroup,
    Residual,
    ResidualSum,
    Sequential,
    Spherical,
    Subdomain,
    System,
    SystemBuildError,
    Transfer,
    Transferable,
    UnitError,
    Weighted,
    apply_coordinates,
    as_expr,
    build_system,
    check_units,
    equation_dimensions,
    equation_scales,
    pi_groups,
)

__all__ = [
    "discrete",
    "solvers",
    "symbolic",
    "Discretization",
    "DomainMaps",
    "MeshData",
    "counterpart",
    "discretize",
    "ghost_by_vertex",
    "Solver",
    "apply_coordinates",
    "as_expr",
    "Assignment",
    "build_system",
    "Cartesian",
    "check_units",
    "Clock",
    "Component",
    "ComponentError",
    "CoordinateSystem",
    "Cylindrical",
    "DirichletBC",
    "Domain",
    "Dt",
    "equation_dimensions",
    "equation_scales",
    "Field",
    "Functional",
    "FunctionalSum",
    "Interpolation",
    "Jacobian",
    "JacobianSum",
    "KnownCoefficient",
    "Parameter",
    "pi_groups",
    "PiGroup",
    "Residual",
    "ResidualSum",
    "Sequential",
    "Spherical",
    "Subdomain",
    "System",
    "SystemBuildError",
    "Transfer",
    "Transferable",
    "UnitError",
    "Weighted",
]
