"""Time-stepping schemes, which discretize time derivatives.

Models stay semi-discrete, with the placeholders ``field.dot`` and
``Dt(expr)``. A :class:`TimeScheme` turns a model into one step:

- :class:`BDF` of order 1 (backward Euler), 2 or 3 replaces the placeholders
  by backward differences, and evaluates everything else at the new time.
- :class:`Theta`, with :class:`CrankNicolson` and :class:`ForwardEuler`,
  also weighs the other terms between the new and the old time.
"""

from __future__ import annotations

import functools
import math
import typing
from collections.abc import Sequence

import sympy.physics.units as syu
import ufl

from ..symbolic.components import Residual, ResidualSum
from ..symbolic.domains import Domain, domain_of
from ..symbolic.expressions import FormOrZero
from ..symbolic.terminals import Field, Parameter
from ..symbolic.time import Clock, time_derivative_expressions, time_derivative_fields

if typing.TYPE_CHECKING:
    from ..discrete import Discretization

__all__ = [
    "BDF",
    "AdaptiveTimeStep",
    "CrankNicolson",
    "ForwardEuler",
    "Theta",
    "TimeScheme",
]


_BDF_ORDERS = (1, 2, 3)
"""The orders of the BDF schemes."""


def _bdf_weights(steps: Sequence[float]) -> tuple[float, ...]:
    """Weights of ``u_{n+1}, u_n, u_{n-1}, ...`` in a BDF formula.

    The derivative at the new time of the polynomial that interpolates the
    values at the new time and at the previous ones, so the formula is exact
    for polynomials in time of degree up to the number of steps.

    Args:
        steps: The sizes of the current step and of the previous ones, most
            recent first; their number is the order.
    """
    # The times of the values, relative to the new time
    times = [0.0, *(-sum(steps[: j + 1]) for j in range(len(steps)))]
    weights = [sum(1.0 / (0.0 - tm) for tm in times[1:])]
    for j, tj in enumerate(times[1:], start=1):
        others = [tm for m, tm in enumerate(times) if m != j]
        numerator = math.prod(0.0 - tm for tm in others[1:])
        weights.append(numerator / math.prod(tj - tm for tm in others))
    return tuple(weights)


Substitutions = dict[ufl.Coefficient, ufl.core.expr.Expr]
"""Replacements of terminals in the forms of a system."""


class TimeScheme:
    """A scheme that turns a semi-discrete model into one time step.

    Subclasses give :meth:`prepare`, which a
    :class:`~sumfem.solvers.simulation.Simulation` calls once per
    stage, and :meth:`set_step`, which it calls before every step.
    """

    @property
    def order(self) -> int:
        """Order of accuracy of the scheme."""
        raise NotImplementedError

    @property
    def given_history(self) -> bool:
        """Whether the user sets the previous values before the first step.

        Otherwise a simulation sets them to the initial values.
        """
        return False

    def prepare(self, model: Residual, clock: Clock) -> tuple[Residual, Substitutions]:
        """The model of one step, and substitutions for its system.

        Pass both to :func:`~sumfem.symbolic.system.build_system`.
        """
        raise NotImplementedError

    def set_step(
        self,
        step: int,
        dt: float,
        discretization: Discretization,
        previous: Sequence[float] = (),
    ) -> None:
        """Set the scheme's parameters for a time step.

        Args:
            step: Number of the step, starting at 1.
            dt: Time step size.
            discretization: The discretization holding the parameter values.
            previous: Sizes of the steps before, most recent first. Missing
                ones are taken equal to ``dt``.
        """
        raise NotImplementedError


def _check_step(step: int, dt: float) -> None:
    if step < 1:
        raise ValueError(f"Steps are numbered from 1, got {step}.")
    if dt <= 0:
        raise ValueError(f"The time step must be positive, got {dt}.")


class BDF(TimeScheme):
    """Backward differentiation formula of order 1 (backward Euler), 2 or 3.

    The scheme replaces each time derivative ``field.dot`` by
    ``w0 * field + w1 * field.previous(1) + ...``, and the time derivative
    ``Dt(q)`` of an expression by ``w0 * q + w1 * q(previous(1)) + ...``,
    the expression of the fields' previous values. The weights are
    parameters, set at every step from the sizes of the current and the
    previous steps: they differentiate, at the new time, the polynomial
    through the values at the times of the steps. The steps may thus change
    in size, e.g. between the runs of a
    :class:`~sumfem.solvers.simulation.Simulation` or in an
    :class:`~sumfem.solvers.simulation.AdaptiveSimulation`; with equal steps
    the weights are the classical ones. The first steps of a higher order
    scheme use the highest order the available history allows, so one
    compiled system serves every step.

    A step that grows fast relative to the previous one can make the higher
    orders unstable: BDF2 stays zero-stable while each step is less than
    ``1 + sqrt(2)`` times the one before, and BDF3 needs smaller ratios.
    Shrinking steps are safe. BDF3 is not A-stable, so it suits damped
    problems better than waves.

    Without a given history, the first steps use lower orders. The first,
    backward Euler step then limits BDF3 to second order in general. To keep
    the full order, set the previous values ``field.previous(k)`` before
    running, e.g. to an exact solution or a finer start-up run, and pass
    ``given_history=True``. The given values are taken one step of the first
    step's size apart.

    Args:
        order: 1, 2 or 3.
        name: Prefix of the weight parameters, ``"<name>.<domain>.w<k>"``.
        blend: For order 2, the weight ``a`` of the generalized formula
            ``a * BDF2 + (1 - a) * BDF1``, e.g. of SWEMniCS; 1 is BDF2 and 0
            implicit Euler.
        given_history: Use the full order from the first step, with the
            previous values set by the user.
    """

    _order: int
    """Order of the scheme."""

    _name: str
    """Prefix of the weight parameters."""

    _weights: dict[str, tuple[Parameter, ...]]
    """The weight parameters ``w0, ..., w<order>``, by domain name."""

    _blend: float
    """Weight of BDF2 against BDF1 for order 2."""

    _given_history: bool
    """Whether the user sets the previous values."""

    def __init__(
        self,
        order: int = 1,
        name: str = "bdf",
        blend: float = 1.0,
        given_history: bool = False,
    ):
        if order not in _BDF_ORDERS:
            raise ValueError(f"BDF order must be one of {_BDF_ORDERS}, got {order}.")
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"The blend must lie in [0, 1], got {blend}.")
        if order != 2 and blend != 1.0:
            raise ValueError("A blend needs order 2.")
        self._order = order
        self._name = name
        self._weights = {}
        self._blend = blend
        self._given_history = given_history

    @property
    def order(self) -> int:
        """Order of the scheme, also the number of previous values it needs."""
        return self._order

    @property
    def given_history(self) -> bool:
        return self._given_history

    def _weights_on(self, domain: Domain) -> tuple[Parameter, ...]:
        weights = self._weights.get(domain.name)
        if weights is None:
            weights = tuple(
                Parameter(
                    f"{self._name}.{domain.name}.w{k}",
                    domain,
                    unit=1 / syu.second,
                    doc=f"Weight of the value {k} steps back in the BDF formula",
                )
                for k in range(self._order + 1)
            )
            self._weights[domain.name] = weights
        return weights

    def prepare(self, model: Residual, clock: Clock) -> tuple[Residual, Substitutions]:
        """The model itself, with its time derivatives replaced."""
        return model, self.substitutions(model)

    def substitutions(self, model: Residual) -> Substitutions:
        """Replacements of the time derivatives in a model.

        Pass them to :func:`~sumfem.symbolic.system.build_system`.
        """
        result: dict[ufl.Coefficient, ufl.core.expr.Expr] = {}
        for field in time_derivative_fields(model):
            w = self._weights_on(field.domain)
            terms = [w[k] * field.previous(k) for k in range(1, self._order + 1)]
            result[field.dot] = functools.reduce(
                lambda a, b: a + b, terms, w[0] * field
            )
        for dot in time_derivative_expressions(model):
            w = self._weights_on(dot.fields[0].domain)
            terms = [w[k] * dot.history(k) for k in range(1, self._order + 1)]
            result[dot] = functools.reduce(
                lambda a, b: a + b, terms, w[0] * dot.expression
            )
        return result

    def set_step(
        self,
        step: int,
        dt: float,
        discretization: Discretization,
        previous: Sequence[float] = (),
    ) -> None:
        """Set the weights for a time step.

        Args:
            step: Number of the step, starting at 1.
            dt: Time step size.
            discretization: The discretization holding the weight values.
            previous: Sizes of the steps before, most recent first. Missing
                ones, e.g. of a given history or of a loop that steps with
                one size, are taken equal to ``dt``.
        """
        _check_step(step, dt)
        if any(not size > 0 for size in previous):
            raise ValueError(f"Step sizes must be positive, got {list(previous)}.")
        order = self._order if self._given_history else min(self._order, step)
        steps = [dt, *previous[: order - 1]]
        steps += [dt] * (order - len(steps))
        values = [*_bdf_weights(steps), *[0.0] * (self._order - order)]
        if order == 2:
            first = [*_bdf_weights([dt]), *[0.0] * (self._order - 1)]
            values = [
                self._blend * a + (1 - self._blend) * b
                for a, b in zip(values, first, strict=True)
            ]
        for parameters in self._weights.values():
            # Only the weights of domains this discretization's systems use
            if parameters[0] not in discretization:
                continue
            discretization.set(dict(zip(parameters, values, strict=True)))


class Theta(TimeScheme):
    """The theta method, between forward (0) and backward (1) Euler.

    A model's residual ``R = M(dot) + F(u, t)``, with ``M`` linear in the
    time-derivative placeholders, becomes

    ``M((u - u_n) / dt) + theta F(u, t) + (1 - theta) F(u_n, t_n)``,

    where the old part replaces every field by its ``previous(1)`` and the
    clock's time by the previous time. Time derivatives ``Dt(q)`` of
    expressions become ``(q(u) - q(u_n)) / dt``. The method is of order 2
    for ``theta = 1/2``, and of order 1 otherwise.

    Data that changes in time other than through the clock, e.g. a known
    coefficient updated between steps, is taken at the new time in both
    parts. Hand-written Jacobians are not carried over: the Jacobian is
    derived from the residual.

    Args:
        theta: The weight ``theta`` of the new time, in [0, 1].
        name: Prefix of the scheme's parameters.
    """

    _theta: float
    """The weight of the new time."""

    _name: str
    """Prefix of the scheme's parameters."""

    _rates: dict[str, Parameter]
    """The parameter ``1 / dt``, by domain name."""

    _clock: Clock | None
    """The clock of the prepared models."""

    _previous_time: Parameter | None
    """The time of the previous step."""

    def __init__(self, theta: float, name: str = "theta"):
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must lie in [0, 1], got {theta}.")
        self._theta = theta
        self._name = name
        self._rates = {}
        self._clock = None
        self._previous_time = None

    @property
    def theta(self) -> float:
        """The weight of the new time."""
        return self._theta

    @property
    def order(self) -> int:
        return 2 if self._theta == 0.5 else 1

    def _rate_on(self, domain: Domain) -> Parameter:
        rate = self._rates.get(domain.name)
        if rate is None:
            rate = Parameter(
                f"{self._name}.{domain.name}.rate",
                domain,
                unit=1 / syu.second,
                doc="One over the time step",
            )
            self._rates[domain.name] = rate
        return rate

    def prepare(self, model: Residual, clock: Clock) -> tuple[Residual, Substitutions]:
        """Every summand of the model as its theta step."""
        if self._clock is not None and self._clock is not clock:
            raise ValueError("A theta scheme serves one clock.")
        self._clock = clock
        if self._previous_time is None:
            self._previous_time = Parameter(
                f"{self._name}.t_previous",
                domain_of(clock.t.ufl_function_space()),
                unit=syu.second,
                doc="Time of the previous step",
            )
        previous_time = self._previous_time
        steps = [_ThetaStep(m, self, clock.t, previous_time) for m in model.components]
        if len(steps) == 1:
            return steps[0], {}
        return ResidualSum(steps, name=model.name), {}

    def set_step(
        self,
        step: int,
        dt: float,
        discretization: Discretization,
        previous: Sequence[float] = (),
    ) -> None:
        """Set ``1 / dt`` and the previous time; a one-step method needs no
        previous step sizes."""
        _check_step(step, dt)
        for rate in self._rates.values():
            if rate in discretization:
                discretization.set({rate: 1 / dt})
        previous_time = self._previous_time
        # Only a step whose forms use the time holds the previous time
        if (
            self._clock is not None
            and previous_time is not None
            and previous_time in discretization
        ):
            t = float(discretization.value(self._clock.t))
            discretization.set({previous_time: t - dt})


class CrankNicolson(Theta):
    """The Crank-Nicolson method, the theta method of order 2 with theta 1/2.

    Args:
        name: Prefix of the scheme's parameters.
    """

    def __init__(self, name: str = "crank_nicolson"):
        super().__init__(0.5, name)


class ForwardEuler(Theta):
    """The forward Euler method, the theta method with theta 0.

    Each step solves with the time-derivative terms alone, e.g. a mass
    matrix, so it is cheap where that is easy to solve, as for
    discontinuous elements. It is stable only for small steps.

    Args:
        name: Prefix of the scheme's parameters.
    """

    def __init__(self, name: str = "forward_euler"):
        super().__init__(0.0, name)


class _ThetaStep(Residual):
    """A summand of a model as one step of a theta method."""

    _original: Residual
    """The semi-discrete summand."""

    _scheme: Theta
    """The theta scheme."""

    _time: Parameter
    """The clock's time."""

    _previous_time: Parameter
    """The time of the previous step."""

    def __init__(
        self,
        original: Residual,
        scheme: Theta,
        time: Parameter,
        previous_time: Parameter,
    ):
        super().__init__(original.name)
        self._original = original
        self._scheme = scheme
        self._time = time
        self._previous_time = previous_time

    def residual(self) -> FormOrZero:
        form = self._original.form
        quotients: Substitutions = {}
        zeros: Substitutions = {}
        for field in time_derivative_fields(self._original):
            rate = self._scheme._rate_on(field.domain)
            quotients[field.dot] = rate * (field - field.previous(1))
            zeros[field.dot] = ufl.zero(field.ufl_shape)
        for dot in time_derivative_expressions(self._original):
            rate = self._scheme._rate_on(dot.fields[0].domain)
            quotients[dot] = rate * (dot.expression - dot.history(1))
            zeros[dot] = ufl.zero(dot.ufl_shape)
        step = ufl.replace(form, quotients) if quotients else form
        theta = self._scheme.theta
        if theta == 1.0:
            return step
        # The steady part, at the new and at the old time
        steady = ufl.replace(form, zeros) if zeros else form
        old: Substitutions = {self._time: self._previous_time}
        for c in steady.coefficients():
            if isinstance(c, Field):
                old[c] = c.previous(1)
        return step + (1 - theta) * (ufl.replace(steady, old) - steady)


class AdaptiveTimeStep:
    """A time step adapted to the Newton iterations.

    After a solve with fewer Newton iterations than ``target_iterations``
    the step grows by ``growth_factor``, and after one with more it is cut
    by ``cutback_factor``, as in FESTIM. A step whose solve fails is
    retried, cut by ``cutback_factor``, up to ``max_retries`` times and not
    below ``min_step``. Every step is capped by ``max_step``, and shortened
    to end at the next milestone, e.g. an output time, and at the end time.
    An :class:`~sumfem.solvers.simulation.AdaptiveSimulation` steps by
    these rules.

    Args:
        initial: The first step, in s.
        growth_factor: Factor above one after easy solves.
        cutback_factor: Factor below one after hard or failed solves.
        target_iterations: Newton iterations of a solve that keeps the step.
        max_step: Largest step, a number or a callable of the time.
        min_step: Smallest step a failed solve is retried with.
        milestones: Times steps must end at.
        milestone_tolerance: Relative tolerance of reaching a milestone.
        max_retries: Retries of a step whose solve fails.

    Raises:
        ValueError: For a non-positive step, or retries without a cutback.
    """

    def __init__(
        self,
        initial: float,
        growth_factor: float | None = None,
        cutback_factor: float | None = None,
        target_iterations: int | None = None,
        max_step: float | typing.Callable[[float], float] | None = None,
        min_step: float = 0.0,
        milestones: typing.Sequence[float] = (),
        milestone_tolerance: float = 1e-5,
        max_retries: int = 0,
    ):
        if not initial > 0:
            raise ValueError(f"The time step must be positive, got {initial}.")
        if max_retries and cutback_factor is None:
            raise ValueError("Retrying a failed step needs a cutback_factor.")
        self.initial = float(initial)
        self.growth_factor = growth_factor
        self.cutback_factor = cutback_factor
        self.target_iterations = target_iterations
        self.max_step = max_step
        self.min_step = float(min_step)
        self.milestones = sorted(milestones)
        self.milestone_tolerance = milestone_tolerance
        self.max_retries = max_retries

    def limit(self, dt: float, t: float, t_end: float) -> float:
        """A step from ``t``, capped and ending at the next milestone or the end."""
        if self.max_step is not None:
            cap = self.max_step(t) if callable(self.max_step) else self.max_step
            dt = min(dt, cap)
        for milestone in [*self.milestones, t_end]:
            if milestone > t and not math.isclose(
                t, milestone, rel_tol=self.milestone_tolerance
            ):
                return min(dt, milestone - t)
        return dt

    def next(self, dt: float, iterations: int) -> float:
        """The step after a step ``dt`` solved in ``iterations`` Newton iterations."""
        if self.target_iterations is None:
            return dt
        if iterations < self.target_iterations and self.growth_factor:
            return dt * self.growth_factor
        if iterations > self.target_iterations and self.cutback_factor:
            return dt * self.cutback_factor
        return dt

    def retry(self, dt: float) -> float | None:
        """The step to retry a failed step ``dt`` with, or ``None`` to give up."""
        if self.cutback_factor is None:
            return None
        dt = dt * self.cutback_factor
        return dt if dt >= self.min_step else None
