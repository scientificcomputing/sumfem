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
import weakref
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
from .time_stepping import AdaptiveTimeStep, TimeScheme

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

    A simulation may run in parts, e.g. to change a load or the step size
    between them. The first run starts the history from the current values;
    later runs continue from where the last one ended, with the history, the
    numbering of the steps and the output of the earlier runs, so a
    multistep scheme keeps its order across them. :meth:`restart` starts
    afresh, e.g. from new initial values.

    Output stays open across the runs. Close it with :meth:`close`, or by
    using the simulation in a ``with`` block. Every process must close it at
    the same point, as writing is collective; an output left open is closed
    when the simulation is garbage collected or the program ends.

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

    _started: bool
    """Whether a run has started the history."""

    _steps_taken: int
    """Number of steps since the start."""

    _step_sizes: list[float]
    """Sizes of the latest steps, most recent first."""

    _writer: _Writer | None
    """The open output, if any."""

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
        self._started = False
        self._steps_taken = 0
        self._step_sizes = []
        self._writer = None
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

    @property
    def steps_taken(self) -> int:
        """Number of steps since the start, over all runs."""
        return self._steps_taken

    def restart(self) -> None:
        """Start afresh at the next run.

        The next run starts the history from the current values, e.g. new
        initial values, numbers its steps from one, and opens the output
        again, which overwrites the earlier files.
        """
        self.close()
        self._started = False
        self._steps_taken = 0
        self._step_sizes = []

    def close(self) -> None:
        """Close the output. A later run opens it again, overwriting it."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> Simulation:
        return self

    def __exit__(self, *exc: typing.Any) -> None:
        self.close()

    def run(
        self,
        num_steps: int | None = None,
        t_end: float | None = None,
        callback: Callback | None = None,
    ) -> None:
        """Take time steps of the clock's step size, from the clock's time.

        Args:
            num_steps: Number of time steps.
            t_end: End time, a whole number of steps after the start. Give
                exactly one of ``num_steps`` and ``t_end``.
            callback: Called after each step with the time and the step
                number, counted over all runs since the start, e.g. to
                post-process.

        Raises:
            ValueError: For inconsistent time arguments, or without a time
                step size.
        """
        disc, clock = self._discretization, self._clock
        if not disc.has_value(clock.dt):
            raise ValueError(f"Set the time step {clock.dt.name!r} before running.")
        dt = float(disc.value(clock.dt))
        t0 = float(disc.value(clock.t))
        num_steps = _number_of_steps(t0, dt, num_steps, t_end)
        self._start()
        for k in range(1, num_steps + 1):
            t = t0 + k * dt
            self._set_time(t, dt)
            self._solver.solve()
            self._finish_step(t, dt, callback)
        self._end()

    def _start(self) -> float:
        """Start a run, and return the time.

        The first run after the start sets up the history, unless the scheme
        has a given history, and opens the output with the initial values.
        """
        disc = self._discretization
        t = float(disc.value(self._clock.t))
        if not self._started:
            if self._scheme is None or not self._scheme.given_history:
                disc.initialize_history()
            self._started = True
            if self._output is not None:
                self._writer = _Writer(self._output, disc)
                self._writer.open(self.metadata())
                self._writer.write(t)
        return t

    def _set_time(self, t: float, dt: float) -> None:
        """Set the time and step size of the next step, and the scheme's weights."""
        disc = self._discretization
        disc.set({self._clock.t: t, self._clock.dt: dt})
        if self._scheme is not None:
            self._scheme.set_step(
                self._steps_taken + 1, dt, disc, tuple(self._step_sizes)
            )

    def _finish_step(self, t: float, dt: float, callback: Callback | None) -> None:
        """Advance the history, write, and call back after a step."""
        self._discretization.advance_history()
        self._steps_taken += 1
        # BDF3, the longest history, needs the two steps before the next one
        self._step_sizes = [dt, *self._step_sizes][:2]
        if self._writer is not None:
            self._writer.write(t, self._steps_taken)
        if callback is not None:
            callback(t, self._steps_taken)

    def _end(self) -> None:
        """End a run: write the files that are rewritten after every run."""
        if self._writer is not None:
            self._writer.flush()

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
        scheme: A time-stepping scheme. BDF of order two or three adapts
            its weights to the changing steps; it stays stable while the
            steps grow slowly, see :class:`~sumfem.solvers.time_stepping.BDF`.
        **kwargs: The other arguments of :class:`Simulation`. The
            ``time_scale`` defaults to the largest step, if it is a number,
            so that the absolute tolerance of Newton's method is no looser
            for any step than for the largest; otherwise to the first step.

    Like a :class:`Simulation`, it may run in parts. Each run starts with
    the initial step of the rules, as a run usually starts at a change, e.g.
    of a load, that a long step could miss.
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
        """Step from the clock's time to the end time.

        Args:
            t_end: End time.
            callback: Called after each step with the time and the step
                number, counted over all runs since the start, e.g. to
                post-process.

        Raises:
            RuntimeError: If a step fails after its retries.
        """
        rules = self._time_step
        t = self._start()
        dt = rules.initial
        while t < t_end and not math.isclose(
            t, t_end, rel_tol=rules.milestone_tolerance
        ):
            dt = rules.limit(dt, t, t_end)
            before = [owned(f).copy() for f in self._solver.system.solution]
            for attempt in range(rules.max_retries + 1):
                self._set_time(t + dt, dt)
                if self._try_step():
                    break
                self._restore(before)
                retried = rules.retry(dt)
                if attempt == rules.max_retries or retried is None:
                    self._end()
                    raise RuntimeError(
                        f"The step from t = {t} failed with dt = {dt}, after "
                        f"{attempt} retries."
                    )
                dt = retried
            t += dt
            self._finish_step(t, dt, callback)
            dt = rules.next(dt, self._solver.iterations)
        self._end()

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
    """Writes the fields of a discretization, and metadata, to an output."""

    def __init__(self, output: Output, discretization: Discretization):
        self._output = output
        self._discretization = discretization
        self._comm = next(iter(discretization.fields.values())).function_space.mesh.comm
        self._vtx: list[dolfinx.io.VTXWriter] = []
        self._real: dict[str, dolfinx.fem.Function] = {}
        self._real_rows: list[list[float]] = []
        self._finalizer: weakref.finalize | None = None

    def open(self, metadata: dict[str, typing.Any]) -> None:
        """Create the directory, write the metadata and open the writers."""
        output, disc, comm = self._output, self._discretization, self._comm
        directory = pathlib.Path(output.directory)
        if comm.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
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
        # Close the writers if the simulation is dropped without closing them
        self._finalizer = weakref.finalize(self, _close_writers, self._vtx)

    def write(self, t: float, step: int = 0) -> None:
        """Write the fields at a time, at every chosen step."""
        if step % self._output.every != 0:
            return
        for writer in self._vtx:
            writer.write(t)
        if self._real:
            self._real_rows.append(
                [t, *(float(f.x.array[0]) for f in self._real.values())]
            )

    def flush(self) -> None:
        """Write the values of the real fields so far."""
        if not self._real or self._comm.rank != 0:
            return
        path = pathlib.Path(self._output.directory) / "real_fields.csv"
        header = ",".join(["t", *self._real])
        np.savetxt(
            path, np.array(self._real_rows), delimiter=",", header=header, comments=""
        )

    def close(self) -> None:
        """Write the real fields and close the writers."""
        self.flush()
        if self._finalizer is not None:
            self._finalizer()


def _close_writers(writers: list[dolfinx.io.VTXWriter]) -> None:
    for writer in writers:
        writer.close()
