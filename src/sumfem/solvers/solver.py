"""Newton solver for discrete systems, built on DOLFINx's NonlinearProblem."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable, Mapping, Sequence

from mpi4py import MPI

from petsc4py import PETSc

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import numpy as np

from .. import symbolic
from ..discrete import bcs as discrete_bcs
from ..discrete import system as discrete_system
from ..symbolic.si import si_factor

__all__ = [
    "DEFAULT_PETSC_OPTIONS",
    "FACTORIZED_ABSOLUTE_TOLERANCES",
    "FACTORIZED_PETSC_OPTIONS",
    "NEWTON_TOLERANCES",
    "Solver",
    "direct_solver_options",
]

NEWTON_TOLERANCES: Mapping[np.dtype, Mapping[str, float]] = {
    np.dtype(np.float64): {"snes_rtol": 1e-10, "snes_atol": 1e-12},
    np.dtype(np.float32): {"snes_rtol": 1e-5, "snes_atol": 1e-6},
}
"""Newton's tolerances in each precision of PETSc's real type, which
:data:`DEFAULT_PETSC_OPTIONS` takes from :data:`dolfinx.default_real_type`.
The absolute tolerance is in the units of the residual, SI unless the
equations are factorized, so a problem may need its own."""

DEFAULT_PETSC_OPTIONS: Mapping[str, typing.Any] = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    **NEWTON_TOLERANCES[np.dtype(dolfinx.default_real_type)],
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
}
"""Newton's method with a direct LU solve, raising if it fails to converge.
Options passed to :class:`Solver` are applied on top. The LU package is
chosen by :func:`direct_solver_options`, unless the options name one."""

_WITHOUT_LINE_SEARCH = frozenset(
    {"ksponly", "ksptransposeonly", "newtontr", "newtontrdc"}
)
"""SNES types that use no line search, for which the default one is dropped."""

FACTORIZED_ABSOLUTE_TOLERANCES: Mapping[np.dtype, float] = {
    np.dtype(np.float64): 1e-10,
    np.dtype(np.float32): 1e-3,
}
"""Newton's absolute tolerance for factorized equations, in each precision
of PETSc's real type. Their residuals are dimensionless, of size one, and
stall at round-off some thousand machine epsilons above zero, so the
absolute tolerance is well above that."""

FACTORIZED_PETSC_OPTIONS: Mapping[str, typing.Any] = {
    "snes_stol": 0.0,
    "snes_atol": FACTORIZED_ABSOLUTE_TOLERANCES[np.dtype(dolfinx.default_real_type)],
}
"""Defaults for factorized equations, on top of :data:`DEFAULT_PETSC_OPTIONS`.

Their residuals are dimensionless, and so are the unknowns that Newton
iterates on, see :class:`Solver`, so Newton converges on the residual alone,
with the absolute tolerance of :data:`FACTORIZED_ABSOLUTE_TOLERANCES`, and
the step-size test is off."""

# Unique PETSc options prefixes for solvers created without one
_prefix_counter = itertools.count()


def _has_package(name: str) -> bool:
    """Whether PETSc was built with an external package, e.g. ``"mumps"``."""
    return bool(PETSc.Sys.hasExternalPackage(name))


def direct_solver_options(comm: MPI.Intracomm) -> dict[str, typing.Any]:
    """PETSc options choosing the LU package, by what PETSc was built with.

    MUMPS is preferred, with null pivot detection so that systems with a
    null space, e.g. a pure Neumann problem, can be factorized. Without it,
    SuperLU_dist is used, and in serial without either, PETSc's own LU.

    Args:
        comm: The communicator of the system to solve.

    Raises:
        RuntimeError: In parallel, if PETSc has neither MUMPS nor
            SuperLU_dist, since PETSc's own LU runs in serial only.
    """
    if _has_package("mumps"):
        return {
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_25": 0,
        }
    if _has_package("superlu_dist"):
        return {"pc_factor_mat_solver_type": "superlu_dist"}
    if comm.size == 1:
        return {"pc_factor_mat_solver_type": "petsc"}
    raise RuntimeError(
        "PETSc has neither MUMPS nor SuperLU_dist, which an LU solve on "
        f"{comm.size} processes needs. Build PETSc with one of them, or pass "
        "an iterative solver in petsc_options."
    )


def _like(vector: PETSc.Vec, value: float = 0.0) -> PETSc.Vec:
    """A vector of the layout of ``vector``, ghosts and blocks included."""
    result = vector.duplicate()
    layout = vector.getAttr("_blocks")
    if layout is not None:
        result.setAttr("_blocks", layout)
    result.set(value)
    return result


def _unknown_scales(
    problem: dolfinx.fem.petsc.NonlinearProblem, scales: Sequence[float]
) -> PETSc.Vec:
    """The SI scale of its unknown on each degree of freedom.

    Args:
        problem: The problem, whose unknowns are in block order.
        scales: The SI scale of each unknown.
    """
    u = problem.u
    functions = list(u) if isinstance(u, Sequence) else [u]
    filled = []
    for function, scale in zip(functions, scales, strict=True):
        constant = dolfinx.fem.Function(function.function_space)
        constant.x.array[:] = scale
        filled.append(constant)
    result = _like(problem.b)
    dolfinx.fem.petsc.assign(filled if isinstance(u, Sequence) else filled[0], result)
    return result


def _dirichlet_row_scaling(
    problem: dolfinx.fem.petsc.NonlinearProblem,
    bcs: Sequence[dolfinx.fem.DirichletBC],
    scales: Sequence[float],
) -> PETSc.Vec:
    """One over the SI scale of its unknown on each constrained row, one on
    the others.

    The rows are found by :func:`dolfinx.fem.petsc.set_bc` itself, so any
    layout of the problem, single, blocked or nested, is covered: setting
    ``g - x0`` with ``x0 = g`` zeroes exactly the constrained rows of a
    vector of ones.

    Args:
        problem: The problem, whose unknowns are in block order.
        bcs: The Dirichlet conditions.
        scales: The SI scale of each unknown.
    """
    blocked = isinstance(problem.F, Sequence)
    if blocked:
        spaces: typing.Any = dolfinx.fem.extract_function_spaces(problem.F)
        by_block = dolfinx.fem.bcs_by_block(spaces, bcs)
    else:
        by_block = [list(bcs)]
    rows, values, free = _like(problem.b, 1.0), _like(problem.b), _like(problem.b)
    for block, scale in enumerate(scales):
        conditions: typing.Any = (
            [c if b == block else [] for b, c in enumerate(by_block)]
            if blocked
            else by_block[0]
        )
        values.set(0.0)
        dolfinx.fem.petsc.set_bc(values, conditions)
        free.set(1.0)
        dolfinx.fem.petsc.set_bc(free, conditions, x0=values)
        # One minus free is exactly one on the constrained rows of the
        # block and zero elsewhere, so the rows there go from one to 1 / X
        # without the round-off of adding 1 / X - 1 to one
        free.scale(-1.0)
        free.shift(1.0)
        rows.axpy(-1.0, free)
        rows.axpy(1.0 / scale, free)
    for work in (values, free):
        work.destroy()
    return rows


def _scale_unknowns(
    problem: dolfinx.fem.petsc.NonlinearProblem, scales: PETSc.Vec, rows: PETSc.Vec
) -> None:
    """Let Newton iterate on the unknowns divided by their scales.

    The SNES iterate is then ``x / X``, from which the residual and the
    Jacobian are assembled at ``x``. The columns of the Jacobian are
    multiplied by ``X``, and the rows of the residual and the Jacobian by
    ``rows``: DOLFINx sets a constrained row to ``x - g`` and the identity,
    in SI values, which ``rows`` divides by ``X``. The Newton update is the
    same, since the system is only scaled, but every entry is now of the
    size of the factorized equations, and so are the norms of the residual
    and of the step.
    """
    # petsc4py's type hints do not cover the callbacks the getters return
    snes: typing.Any = problem.solver
    vector, (residual, residual_args, residual_kargs) = snes.getFunction()
    matrix, preconditioner, (jacobian, jacobian_args, jacobian_kargs) = (
        snes.getJacobian()
    )
    si = _like(problem.b)

    def scaled_residual(snes_: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        si.pointwiseMult(x, scales)
        residual(snes_, si, b, *residual_args, **residual_kargs)
        b.pointwiseMult(b, rows)

    def scaled_jacobian(
        snes_: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat
    ) -> None:
        si.pointwiseMult(x, scales)
        jacobian(snes_, si, J, P, *jacobian_args, **jacobian_kargs)
        J.diagonalScale(L=rows, R=scales)
        if P is not None and P.handle != J.handle:
            P.diagonalScale(L=rows, R=scales)

    snes.setFunction(scaled_residual, vector)
    snes.setJacobian(scaled_jacobian, matrix, preconditioner)


class Solver:
    """Solve a :class:`~sumfem.discrete.system.System`.

    Args:
        system: The discrete system.
        bcs: Dirichlet conditions, symbolic or as DOLFINx objects. Symbolic
            conditions must be on unknowns of the system.
        petsc_options: PETSc options for the SNES solver, on top of
            :data:`DEFAULT_PETSC_OPTIONS`, and of
            :data:`FACTORIZED_PETSC_OPTIONS` for factorized equations. An LU
            solve without ``pc_factor_mat_solver_type`` gets the package of
            :func:`direct_solver_options`.
        petsc_options_prefix: Options prefix of the PETSc objects. Defaults
            to a unique prefix.
        kind: PETSc matrix and vector kind, e.g. ``"nest"`` for field-split
            preconditioning.

    Symbolic boundary values are evaluated before every solve and every
    Newton iteration, so they follow parameters, time and known coefficients, as
    in Irksome's DOLFINx backend.

    For factorized equations, Newton iterates on the unknowns divided by their
    scales, in SI, and the rows of the Dirichlet conditions are divided by the
    scale of their unknown, so that the whole system is dimensionless: the
    residual, the step and every entry of the Jacobian are of size one. The
    Newton update is the same; the norms that stop Newton are not, nor are the
    pivots of an LU factorization. The functions of the unknowns hold SI values
    before and after a solve, which must go through :meth:`solve`, not the
    :attr:`nonlinear_problem`.
    """

    _system: discrete_system.System
    """The discrete system being solved."""

    _bcs: tuple[discrete_bcs.DirichletBC, ...]
    """The symbolic Dirichlet conditions, discretized."""

    _nonlinear_problem: dolfinx.fem.petsc.NonlinearProblem
    """The DOLFINx problem holding the PETSc objects."""

    _unknown_scales: PETSc.Vec | None
    """The SI scale of its unknown on each degree of freedom, for factorized
    equations, by which Newton's iterate is divided."""

    def __init__(
        self,
        system: discrete_system.System,
        bcs: Iterable[symbolic.DirichletBC | dolfinx.fem.DirichletBC] = (),
        petsc_options: Mapping[str, typing.Any] | None = None,
        petsc_options_prefix: str | None = None,
        kind: str | None = None,
    ):
        symbolic_system = system.symbolic
        discretized: list[discrete_bcs.DirichletBC] = []
        dolfinx_bcs: list[dolfinx.fem.DirichletBC] = []
        for bc in bcs:
            if isinstance(bc, symbolic.DirichletBC):
                if not any(bc.field is f for f in symbolic_system.unknowns):
                    raise ValueError(
                        f"The boundary condition on {bc.field.name!r} is not on a "
                        "field this system solves for, "
                        f"{[f.name for f in symbolic_system.unknowns]}."
                    )
                discrete_bc = discrete_bcs.DirichletBC(system.discretization, bc)
                discretized.append(discrete_bc)
                dolfinx_bcs.append(discrete_bc.dolfinx_bc)
            else:
                dolfinx_bcs.append(bc)
        self._system = system
        self._bcs = tuple(discretized)
        self._dolfinx_bcs = tuple(dolfinx_bcs)
        if petsc_options_prefix is None:
            petsc_options_prefix = f"sumfem_{next(_prefix_counter)}_"
        blocked = symbolic_system.blocked
        options = {
            **DEFAULT_PETSC_OPTIONS,
            **(FACTORIZED_PETSC_OPTIONS if symbolic_system.equation_scales else {}),
            **(petsc_options or {}),
        }
        # Only a line search method reads the default line search: PETSc
        # reports it as unused otherwise
        line_search_given = "snes_linesearch_type" in (petsc_options or {})
        if options["snes_type"] in _WITHOUT_LINE_SEARCH and not line_search_given:
            options.pop("snes_linesearch_type", None)
        # Only an LU solve needs a package, which the caller may have chosen
        if (
            options.get("pc_type") == "lu"
            and "pc_factor_mat_solver_type" not in options
        ):
            options = {**direct_solver_options(system.discretization.comm), **options}
        # NonlinearProblem accepts compiled forms, which its type hints do
        # not cover.
        forms: typing.Any = (system.residual, system.jacobian, system.preconditioner)
        self._nonlinear_problem = dolfinx.fem.petsc.NonlinearProblem(
            forms[0],
            list(system.solution) if blocked else system.solution[0],
            J=forms[1],
            P=forms[2],
            bcs=dolfinx_bcs,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=options,
            kind=kind,
        )
        # Newton iterates on unknowns divided by their scales when the
        # equations are, see _scale_unknowns
        self._unknown_scales = None
        if symbolic_system.equation_scales:
            scales = [
                field.scale * si_factor(field.unit)
                for field in symbolic_system.unknowns
            ]
            problem = self._nonlinear_problem
            self._unknown_scales = _unknown_scales(problem, scales)
            rows = _dirichlet_row_scaling(problem, self._dolfinx_bcs, scales)
            _scale_unknowns(problem, self._unknown_scales, rows)
        if any(bc.expression is not None for bc in self._bcs):
            # The callback must not refer to the solver: the cycle through
            # the SNES would leave the PETSc objects to the garbage
            # collector, whose collective destroy calls then run at
            # different times on different ranks and deadlock.
            discrete_bcs_ = self._bcs
            self._nonlinear_problem.set_update(lambda _iteration: _pack(discrete_bcs_))

    @property
    def system(self) -> discrete_system.System:
        """The discrete system being solved."""
        return self._system

    @property
    def nonlinear_problem(self) -> dolfinx.fem.petsc.NonlinearProblem:
        """The DOLFINx problem, giving access to the PETSc objects.

        For factorized equations, its SNES iterates on scaled unknowns, see
        :class:`Solver`.
        """
        return self._nonlinear_problem

    @property
    def boundary_values(
        self,
    ) -> tuple[dolfinx.fem.Function | dolfinx.fem.Constant, ...]:
        """The functions or constants holding the prescribed boundary values."""
        return tuple(bc.g for bc in self._dolfinx_bcs)

    @property
    def iterations(self) -> int:
        """Newton iterations of the last solve."""
        return int(self._nonlinear_problem.solver.getIterationNumber())

    @property
    def converged(self) -> bool:
        """Whether the last solve converged."""
        reason: typing.Any = self._nonlinear_problem.solver.getConvergedReason()
        return int(reason) > 0

    @property
    def converged_to_tolerance(self) -> bool:
        """Whether the last solve stopped at a residual or step tolerance.

        Solving again from its result then changes nothing, unlike after a
        solve stopped at its iteration count, e.g. a smoother.
        """
        reason = self._nonlinear_problem.solver.getConvergedReason()
        return reason in _AT_TOLERANCE

    def pack_bcs(self) -> None:
        """Evaluate symbolic boundary values with the current values of their inputs."""
        _pack(self._bcs)

    def solve(self) -> tuple[dolfinx.fem.Function, ...]:
        """Solve the system, starting from the current values of the unknowns.

        Returns:
            The function of each unknown, in block order, holding the
            solution in SI units.

        Raises:
            ValueError: If a parameter of the system has no value.
        """
        system = self._system
        system.discretization.check_values(system.symbolic)
        self.pack_bcs()
        problem = self._nonlinear_problem
        scales = self._unknown_scales
        if scales is None:
            problem.solve()
        else:
            u = problem.u
            dolfinx.fem.petsc.assign(u, problem.x)
            problem.x.pointwiseDivide(problem.x, scales)
            problem.solver.solve(None, problem.x)
            problem.x.pointwiseMult(problem.x, scales)
            dolfinx.fem.petsc.assign(problem.x, u)
            for function in system.solution:
                function.x.scatter_forward()
        return system.solution


_AT_TOLERANCE = {
    PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS,
    PETSc.SNES.ConvergedReason.CONVERGED_FNORM_RELATIVE,
    PETSc.SNES.ConvergedReason.CONVERGED_SNORM_RELATIVE,
}


def _pack(bcs: typing.Iterable[discrete_bcs.DirichletBC]) -> None:
    """Evaluate the symbolic values of boundary conditions."""
    for bc in bcs:
        bc.pack()
