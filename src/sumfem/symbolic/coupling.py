"""Sequential (operator-split) coupling of stages."""

from __future__ import annotations

import typing

from .components import Residual
from .system import System
from .transfer import Transfer

__all__ = ["Sequential", "Stage"]

Stage = Residual | System | Transfer
"""A stage of a sequence: a model, a system built from one, or a transfer."""


class Sequential:
    """Solve stages one after another, each with the latest values of the others.

    A model is a stage that solves for every field it writes an equation
    for, and a :class:`~sumfem.symbolic.system.System` one built with
    options of its own, e.g. a time-stepping scheme's substitutions. The
    other fields a stage uses are data, taken from the stages that solve for
    them. A :class:`~sumfem.symbolic.transfer.Transfer` is a stage too,
    which writes data from one field to another, e.g. a velocity solved on
    one mesh into a known coefficient of another.

    By default the sequence runs once, which is weak coupling. With a
    tolerance it is repeated until the solution stops changing, which is
    strong coupling.

    A stage may itself be a sequence, which runs to its own tolerance
    whenever the outer sequence reaches it, e.g. the iterations of an
    enthalpy method inside a pass that then solves for the flow.

    Args:
        stages: The models, systems, transfers and nested sequences, in the
            order they run.
        max_iterations: Number of passes through the stages without a
            tolerance, or the maximum number with one.
        tol: Relative change of the unknowns of its stages between two
            passes below which strong coupling has converged.

    Raises:
        ValueError: For an empty sequence, a non-positive number of
            iterations, or a non-positive tolerance.
        TypeError: If a stage is not a model, a system, a transfer or a
            sequence.
    """

    _stages: tuple[Stage | Sequential, ...]
    """The stages and nested sequences, in the order they run."""

    _max_iterations: int
    """Number of passes, or maximum number of passes with a tolerance."""

    _tol: float | None
    """Tolerance of the relative change for strong coupling."""

    def __init__(
        self,
        stages: typing.Sequence[Stage | Sequential],
        max_iterations: int = 1,
        tol: float | None = None,
    ):
        if len(stages) == 0:
            raise ValueError("A sequence needs at least one stage.")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}.")
        if tol is not None and tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        for stage in stages:
            if not isinstance(stage, Residual | System | Transfer | Sequential):
                raise TypeError(
                    "A stage must be a model, a system, a transfer or a sequence, "
                    f"got {stage!r}."
                )
        self._stages = tuple(stages)
        self._max_iterations = max_iterations
        self._tol = tol

    @property
    def stages(self) -> tuple[Stage | Sequential, ...]:
        """The stages and nested sequences, in the order they run."""
        return self._stages

    @property
    def max_iterations(self) -> int:
        """Number of passes, or maximum number of passes with a tolerance."""
        return self._max_iterations

    @property
    def tol(self) -> float | None:
        """Tolerance of the relative change for strong coupling, if any."""
        return self._tol

    def leaves(self) -> tuple[Stage, ...]:
        """The stages of this and the nested sequences, in order."""
        return tuple(
            leaf
            for stage in self._stages
            for leaf in (stage.leaves() if isinstance(stage, Sequential) else (stage,))
        )

    def map(self, function: typing.Callable[[Stage], Stage]) -> Sequential:
        """The same sequence, with each stage replaced by ``function(stage)``."""
        return Sequential(
            [
                stage.map(function)
                if isinstance(stage, Sequential)
                else function(stage)
                for stage in self._stages
            ],
            self._max_iterations,
            self._tol,
        )

    def description(self) -> dict[str, typing.Any]:
        """The structure of the sequence by stage names, e.g. for metadata."""
        return {
            "max_iterations": self._max_iterations,
            "tol": self._tol,
            "stages": [
                stage.description()
                if isinstance(stage, Sequential)
                else stage_name(stage)
                for stage in self._stages
            ],
        }


def stage_name(stage: Stage) -> str:
    """The name of a stage: its own, or that of a system's model."""
    return stage.component.name if isinstance(stage, System) else stage.name
