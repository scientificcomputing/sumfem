"""Step a model in time."""

from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import typing
from collections.abc import Iterable, Mapping

import petsc4py
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem
import dolfinx.io
import ffcx
import numpy as np
import numpy.typing as npt
import ufl

from .. import symbolic
from ..discrete import Discretization
from ..symbolic.components import Residual
from ..symbolic.system import build_system
from ..symbolic.terminals import Parameter
from ..symbolic.time import Clock, time_derivative_expressions, time_derivative_fields
from .solver import DEFAULT_PETSC_OPTIONS, FACTORIZED_PETSC_OPTIONS, Solver
from .time_stepping import BDF, AdaptiveTimeStep, TimeScheme

Callback = typing.Callable[[float, int], None]
"""Called after a step with the time and the step number."""

__all__ = ["AdaptiveSimulation", "Output", "Simulation", "owned", "versions"]


@dataclasses.dataclass(frozen=True)
class Output:
    """Where and what a simulation writes.

    Attributes:
        directory: Output directory, created if needed.
        fields: Names of the fields to write. Defaults to every field.
        every: Write every ``every``-th time step.
    """

    directory: str | os.PathLike
    fields: tuple[str, ...] | None = None
    every: int = 1


class Simulation:
    """Step a model in time.

    The model is prepared with the time-stepping scheme and built into a
    system once. Each step sets the clock and the scheme's weights, solves
    the system with a :class:`~sumfem.solvers.solver.Solver`, and advances
    the history. A steady problem needs no simulation: solve the model with
    a :class:`~sumfem.solvers.solver.Solver`.

    Args:
        discretization: The meshes and the functions of all terminals.
        model: The model, solved monolithically.
        bcs: Dirichlet conditions, symbolic or as DOLFINx objects.
        clock: The time and time step.
        scheme: Time-stepping scheme, needed if the model has time
            derivatives.
        petsc_options: PETSc options of the solver.
        kind: PETSc matrix and vector kind.
        output: Where and what to write. Nothing is written without it.
        check_units: Check that every equation is dimensionally consistent,
            see :func:`~sumfem.symbolic.units.check_units`.
        factorize_equations: Solve in dimensionless form, see
            :func:`~sumfem.symbolic.system.build_system`. Fields are still
            stored and written in SI units.
        time_scale: Reference time in seconds of the time derivatives, for
            the unit check and the sizes of factorized equations. Defaults to
            the clock's time step, which makes the change over one step, the
            dominant term of a step, of order one; without a step, to one
            second. The size divides each equation by one number, so it does
            not change the solution or Newton's iterates, but it gives
            Newton's absolute tolerance its meaning: a reference time far
            below the steps makes the scaled residual small, and stops
            Newton's method early.

    The simulation declares on the discretization what it creates: the
    clock, the system with the terminals of the time-stepping scheme, and
    its symbolic boundary conditions.

    Raises:
        ValueError: Without a clock, or if the model has time derivatives but
            no scheme is given.
    """

    _discretization: Discretization
    """The meshes and the functions of all terminals."""

    _model: Residual
    """The model as given, before preparation with the scheme."""

    _clock: Clock
    """The time and time step."""

    _scheme: TimeScheme | None
    """The time-stepping scheme, if any."""

    _solver: Solver
    """The solver of one step."""

    _petsc_options: dict[str, typing.Any]
    """The PETSc options given, for the metadata."""

    _output: Output | None
    """Where and what to write, if anything."""

    _time_scale: float
    """Reference time of the time derivatives, in seconds."""

    def __init__(
        self,
        discretization: Discretization,
        model: Residual,
        bcs: Iterable[symbolic.DirichletBC | dolfinx.fem.DirichletBC] = (),
        clock: Clock | None = None,
        scheme: TimeScheme | None = None,
        petsc_options: Mapping[str, typing.Any] | None = None,
        kind: str | None = None,
        output: Output | None = None,
        check_units: bool = False,
        factorize_equations: bool = False,
        time_scale: float | None = None,
    ):
        if clock is None:
            raise ValueError(
                "A simulation steps in time, so it needs a clock. Solve a steady "
                "model with a Solver."
            )
        self._discretization = discretization
        self._model = model
        self._clock = clock
        self._scheme = scheme
        self._petsc_options = dict(petsc_options or {})
        self._output = output
        bcs = list(bcs)
        discretization.add(clock)
        discretization.add(*(bc for bc in bcs if isinstance(bc, symbolic.DirichletBC)))
        if time_scale is None:
            time_scale = 1.0
            if discretization.has_value(clock.dt):
                time_scale = float(discretization.value(clock.dt))
        if not time_scale > 0:
            raise ValueError(f"The time scale must be positive, got {time_scale}.")
        self._time_scale = time_scale
        if scheme is None:
            if time_derivative_fields(model) or time_derivative_expressions(model):
                raise ValueError(
                    f"Component {model.name!r} has time derivatives, so the "
                    "simulation needs a time-stepping scheme."
                )
            step, substitutions = model, None
        else:
            step, substitutions = scheme.prepare(model, clock)
        system = build_system(
            step,
            substitutions=substitutions,
            check_units=check_units,
            factorize_equations=factorize_equations,
            time_scale=time_scale,
        )
        discretization.add(system)
        self._solver = Solver(discretization[system], bcs, petsc_options, kind=kind)

    @property
    def discretization(self) -> Discretization:
        """The meshes and the functions of all terminals."""
        return self._discretization

    @property
    def solver(self) -> Solver:
        """The solver of one step."""
        return self._solver

    def run(
        self,
        num_steps: int | None = None,
        t_end: float | None = None,
        callback: Callback | None = None,
    ) -> None:
        """Take time steps of the clock's step size.

        Args:
            num_steps: Number of time steps.
            t_end: End time, a whole number of steps after the start. Give
                exactly one of ``num_steps`` and ``t_end``.
            callback: Called after each step with the time and the step
                number, e.g. to post-process.

        Raises:
            ValueError: For inconsistent time arguments, or without a time
                step size.
        """
        disc, clock = self._discretization, self._clock
        if not disc.has_value(clock.dt):
            raise ValueError(f"Set the time step {clock.dt.name!r} before running.")
        t0 = self._start()
        dt = float(disc.value(clock.dt))
        num_steps = _number_of_steps(t0, dt, num_steps, t_end)
        with _Writer(self) as writer:
            writer.write(t0)
            for step in range(1, num_steps + 1):
                t = t0 + step * dt
                self._set_time(step, t, dt)
                self._solver.solve()
                self._finish_step(t, step, writer, callback)

    def _start(self) -> float:
        """Start a run: set up the history, and return the time."""
        disc = self._discretization
        if self._scheme is None or not self._scheme.given_history:
            disc.initialize_history()
        return float(disc.value(self._clock.t))

    def _set_time(self, step: int, t: float, dt: float) -> None:
        """Set the time and step size of a step, and the scheme's weights."""
        disc = self._discretization
        disc.set({self._clock.t: t, self._clock.dt: dt})
        if self._scheme is not None:
            self._scheme.set_step(step, dt, disc)

    def _finish_step(
        self, t: float, step: int, writer: _Writer, callback: Callback | None
    ) -> None:
        """Advance the history, write, and call back after a step."""
        self._discretization.advance_history()
        writer.write(t, step)
        if callback is not None:
            callback(t, step)

    def metadata(self) -> dict[str, typing.Any]:
        """A description of the simulation, for archiving with its results."""
        disc = self._discretization
        system = self._solver.system.symbolic
        meshes = {}
        for domain in system.domains:
            mesh = disc.maps.mesh(domain)
            cells = mesh.topology.index_map(mesh.topology.dim).size_global
            meshes[domain.name] = {"cell_type": domain.cell_name, "cells": cells}
        parameters = {}
        for name in sorted(disc.parameters):
            parameter = typing.cast(Parameter, disc.terminal(name))
            parameters[name] = {
                "value": disc.value(name).tolist() if disc.has_value(name) else None,
                "unit": str(parameter.unit),
                "doc": parameter.doc,
            }
        options = {
            **DEFAULT_PETSC_OPTIONS,
            **(FACTORIZED_PETSC_OPTIONS if system.equation_scales else {}),
            **self._petsc_options,
        }
        time = {
            "t0": float(disc.value(self._clock.t)),
            "dt": float(disc.value(self._clock.dt))
            if disc.has_value(self._clock.dt)
            else None,
            "scheme": None
            if self._scheme is None
            else {"type": type(self._scheme).__name__, "order": self._scheme.order},
            "time_scale": self._time_scale,
        }
        return {
            "created": datetime.datetime.now(datetime.UTC).isoformat(),
            "versions": versions(),
            "model": {
                "component": self._model.name,
                "summands": [m.name for m in self._model.components],
                "unknowns": [f.name for f in system.unknowns],
                "known_coefficients": list(system.known_coefficients),
                "equation_scales": dict(system.equation_scales),
            },
            "parameters": parameters,
            "meshes": meshes,
            "petsc_options": {k: _json_value(v) for k, v in options.items()},
            "time": time,
        }


class AdaptiveSimulation(Simulation):
    """A simulation whose time steps adapt to Newton's iterations.

    Steps grow or shrink by the iterations of the last step and stop at
    milestones and the end time. A failed step is retried from the previous
    values with a shorter step. The rules are those of an
    :class:`~sumfem.solvers.time_stepping.AdaptiveTimeStep`.

    Args:
        discretization: See :class:`Simulation`.
        model: See :class:`Simulation`.
        time_step: The rules of the steps.
        clock: The time and time step.
        scheme: A one-step scheme: BDF of order one, or a theta method.
        **kwargs: The other arguments of :class:`Simulation`. The
            ``time_scale`` defaults to the largest step, if it is a number,
            so that the absolute tolerance of Newton's method is no looser
            for any step than for the largest; otherwise to the first step.

    Raises:
        ValueError: With a scheme of more than one step.
    """

    _time_step: AdaptiveTimeStep
    """The rules of the steps."""

    def __init__(
        self,
        discretization: Discretization,
        model: Residual,
        time_step: AdaptiveTimeStep,
        clock: Clock,
        scheme: TimeScheme | None = None,
        **kwargs: typing.Any,
    ):
        if isinstance(scheme, BDF) and scheme.order > 1:
            raise ValueError(
                f"BDF of order {scheme.order} assumes equal steps. Adapt the steps "
                "with BDF(1) or a theta method."
            )
        if kwargs.get("time_scale") is None:
            largest = time_step.max_step
            kwargs["time_scale"] = (
                float(largest)
                if isinstance(largest, int | float)
                else time_step.initial
            )
        super().__init__(discretization, model, clock=clock, scheme=scheme, **kwargs)
        self._time_step = time_step

    def run(  # type: ignore[override]
        self, t_end: float, callback: Callback | None = None
    ) -> None:
        """Step to the end time.

        Args:
            t_end: End time.
            callback: Called after each step with the time and the step
                number, e.g. to post-process.

        Raises:
            RuntimeError: If a step fails after its retries.
        """
        rules = self._time_step
        t = self._start()
        dt = rules.initial
        step = 0
        with _Writer(self) as writer:
            writer.write(t)
            while t < t_end and not math.isclose(
                t, t_end, rel_tol=rules.milestone_tolerance
            ):
                dt = rules.limit(dt, t, t_end)
                before = [owned(f).copy() for f in self._solver.system.solution]
                for attempt in range(rules.max_retries + 1):
                    self._set_time(step + 1, t + dt, dt)
                    if self._try_step():
                        break
                    self._restore(before)
                    retried = rules.retry(dt)
                    if attempt == rules.max_retries or retried is None:
                        raise RuntimeError(
                            f"The step from t = {t} failed with dt = {dt}, after "
                            f"{attempt} retries."
                        )
                    dt = retried
                t += dt
                step += 1
                self._finish_step(t, step, writer, callback)
                dt = rules.next(dt, self._solver.iterations)

    def _try_step(self) -> bool:
        """Solve a step, and whether the solve converged."""
        try:
            self._solver.solve()
        except (PETSc.Error, RuntimeError):
            return False
        return self._solver.converged

    def _restore(self, values: list[npt.NDArray[typing.Any]]) -> None:
        """Set the unknowns back to earlier values of their owned entries."""
        for function, values_ in zip(self._solver.system.solution, values, strict=True):
            function.x.array[: values_.size] = values_
            function.x.scatter_forward()


def owned(function: dolfinx.fem.Function) -> npt.NDArray[typing.Any]:
    """The values of a function's degrees of freedom owned by this process."""
    index_map = function.function_space.dofmap.index_map
    bs = function.function_space.dofmap.index_map_bs
    return function.x.array[: index_map.size_local * bs]


def versions() -> dict[str, str]:
    """The versions of sumfem and the libraries it builds on."""
    try:
        sumfem_version = importlib.metadata.version("sumfem")
    except importlib.metadata.PackageNotFoundError:
        sumfem_version = "unknown"
    return {
        "sumfem": sumfem_version,
        "dolfinx": dolfinx.__version__,
        "ufl": ufl.__version__,
        "basix": basix.__version__,
        "ffcx": ffcx.__version__,
        "petsc4py": petsc4py.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def _json_value(value: typing.Any) -> typing.Any:
    return (
        value
        if isinstance(value, bool | int | float | str | type(None))
        else repr(value)
    )


def _number_of_steps(
    t0: float, dt: float, num_steps: int | None, t_end: float | None
) -> int:
    if dt <= 0:
        raise ValueError(f"The time step must be positive, got {dt}.")
    if (num_steps is None) == (t_end is None):
        raise ValueError("Give exactly one of num_steps and t_end.")
    if num_steps is not None:
        if num_steps < 0:
            raise ValueError(f"num_steps must not be negative, got {num_steps}.")
        return num_steps
    assert t_end is not None
    steps = round((t_end - t0) / dt)
    if steps < 0 or not math.isclose(
        t0 + steps * dt, t_end, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(
            f"t_end = {t_end} is not a whole number of steps of {dt} after t0 = {t0}."
        )
    return steps


class _Writer:
    """Writes fields and metadata for a simulation, if it has an output."""

    def __init__(self, simulation: Simulation):
        self._simulation = simulation
        self._output = simulation._output
        self._vtx: list[dolfinx.io.VTXWriter] = []
        self._real: dict[str, dolfinx.fem.Function] = {}
        self._real_rows: list[list[float]] = []

    def __enter__(self) -> _Writer:
        output = self._output
        if output is None:
            return self
        disc = self._simulation.discretization
        comm = next(iter(disc.fields.values())).function_space.mesh.comm
        directory = pathlib.Path(output.directory)
        if comm.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
            metadata = self._simulation.metadata()
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        comm.barrier()
        names = sorted(disc.fields) if output.fields is None else list(output.fields)
        for name in names:
            function = disc.fields[name]
            element = function.function_space.ufl_element()
            if getattr(element, "is_real", False):
                self._real[name] = function
            else:
                self._vtx.append(
                    dolfinx.io.VTXWriter(comm, directory / f"{name}.bp", [function])
                )
        return self

    def write(self, t: float, step: int = 0) -> None:
        if self._output is None or step % self._output.every != 0:
            return
        for writer in self._vtx:
            writer.write(t)
        if self._real:
            self._real_rows.append(
                [t, *(float(f.x.array[0]) for f in self._real.values())]
            )

    def __exit__(self, *exc: typing.Any) -> None:
        for writer in self._vtx:
            writer.close()
        if self._output is not None and self._real:
            disc = self._simulation.discretization
            comm = next(iter(disc.fields.values())).function_space.mesh.comm
            if comm.rank == 0:
                path = pathlib.Path(self._output.directory) / "real_fields.csv"
                header = ",".join(["t", *self._real])
                np.savetxt(
                    path,
                    np.array(self._real_rows),
                    delimiter=",",
                    header=header,
                    comments="",
                )
