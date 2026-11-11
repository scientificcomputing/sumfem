"""Solving the discrete counterparts of :mod:`sumfem.discrete`.

- :class:`Solver`: Newton's method for one discrete
  :class:`~sumfem.discrete.System`, with its Dirichlet conditions.
- Time-stepping schemes, which discretize time derivatives: :class:`BDF`,
  and the :class:`Theta` method with :class:`CrankNicolson` and
  :class:`ForwardEuler`, subclasses of :class:`TimeScheme`.
- :class:`Simulation`: steps a model in time with a scheme, and
  :class:`AdaptiveSimulation` with steps adapted to Newton's iterations by
  an :class:`AdaptiveTimeStep`; :class:`Output` sets what they write.
"""

from .simulation import AdaptiveSimulation, Output, Simulation
from .solver import Solver
from .time_stepping import (
    BDF,
    AdaptiveTimeStep,
    CrankNicolson,
    ForwardEuler,
    Theta,
    TimeScheme,
)

__all__ = [
    "AdaptiveSimulation",
    "AdaptiveTimeStep",
    "BDF",
    "CrankNicolson",
    "ForwardEuler",
    "Output",
    "Simulation",
    "Solver",
    "Theta",
    "TimeScheme",
]
