"""Solving the discrete counterparts of :mod:`sumfem.discrete`.

- :class:`Solver`: Newton's method for one discrete
  :class:`~sumfem.discrete.System`, with its Dirichlet conditions.
"""

from .solver import Solver

__all__ = ["Solver"]
