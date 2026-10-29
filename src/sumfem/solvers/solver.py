"""Newton solver for discrete systems, built on DOLFINx's NonlinearProblem."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable, Mapping

from mpi4py import MPI

from petsc4py import PETSc

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import numpy as np

from ..discrete import bcs as discrete_bcs
from ..discrete import system as discrete_system
from ..symbolic.bcs import DirichletBC

__all__ = [
    "DEFAULT_PETSC_OPTIONS",
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

FACTORIZED_PETSC_OPTIONS: Mapping[str, typing.Any] = {"snes_stol": 0.0}
"""Defaults for factorized equations, on top of :data:`DEFAULT_PETSC_OPTIONS`.

Their residuals are dimensionless, so Newton converges on the residual
alone. The step-size test is off, since it compares the update with the
solution over all unknowns at once, in SI values of different magnitudes."""

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
    """

    _system: discrete_system.System
    """The discrete system being solved."""

    _bcs: tuple[discrete_bcs.DirichletBC, ...]
    """The symbolic Dirichlet conditions, discretized."""

    _nonlinear_problem: dolfinx.fem.petsc.NonlinearProblem
    """The DOLFINx problem holding the PETSc objects."""

    def __init__(
        self,
        system: discrete_system.System,
        bcs: Iterable[DirichletBC | dolfinx.fem.DirichletBC] = (),
        petsc_options: Mapping[str, typing.Any] | None = None,
        petsc_options_prefix: str | None = None,
        kind: str | None = None,
    ):
        symbolic = system.symbolic
        discretized: list[discrete_bcs.DirichletBC] = []
        dolfinx_bcs: list[dolfinx.fem.DirichletBC] = []
        for bc in bcs:
            if isinstance(bc, DirichletBC):
                if not any(bc.field is f for f in symbolic.unknowns):
                    raise ValueError(
                        f"The boundary condition on {bc.field.name!r} is not on a "
                        "field this system solves for, "
                        f"{[f.name for f in symbolic.unknowns]}."
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
        blocked = symbolic.blocked
        options = {
            **DEFAULT_PETSC_OPTIONS,
            **(FACTORIZED_PETSC_OPTIONS if symbolic.equation_scales else {}),
            **(petsc_options or {}),
        }
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
        """The DOLFINx problem, giving access to the PETSc objects."""
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
        self._nonlinear_problem.solve()
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
