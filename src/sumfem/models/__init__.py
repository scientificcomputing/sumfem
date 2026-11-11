"""Models built on :mod:`sumfem`, to be summed and extended.

- Accumulation, diffusion and sources: :class:`Accumulation`,
  :class:`Diffusion`, :class:`Source` and :class:`Poisson`, and Ohmic
  :class:`Conduction`.
- Boundaries: :class:`BoundaryFlux`, :class:`Outflow`, and constraints
  with Lagrange multipliers, :class:`BoundaryConstraint` and
  :class:`DirichletMultiplier`.
- Interfaces between subdomains: :class:`InterfaceConstraint`, a condition
  across an interface with a Lagrange multiplier for the flux.
- Reactions between species in volumes, on surfaces and across membranes
  between two volumes: :class:`Reaction` by mass action and
  :class:`RateReaction` at any rate, with :func:`mass_action` and
  :func:`arrhenius`.
- Effective properties of porous media: the Bruggeman, Kozeny-Carman and
  Ergun relations.

Models of electrochemistry are in :mod:`sumfem.electrochemistry`.
"""

from .boundary import BoundaryConstraint, BoundaryFlux, DirichletMultiplier, Outflow
from .charge import Conduction
from .diffusion import Accumulation, Diffusion, Poisson, Source
from .interface import InterfaceConstraint
from .operators import apply_coefficient, normal_projection, tangential_projection
from .porous import (
    bruggeman,
    ergun_coefficient,
    kozeny_carman,
    kozeny_carman_permeability,
)
from .reactions import RateReaction, Reaction, arrhenius, mass_action

__all__ = [
    "mass_action",
    "arrhenius",
    "RateReaction",
    "Reaction",
    "Accumulation",
    "BoundaryConstraint",
    "BoundaryFlux",
    "Conduction",
    "Diffusion",
    "InterfaceConstraint",
    "DirichletMultiplier",
    "Outflow",
    "Poisson",
    "Source",
    "apply_coefficient",
    "normal_projection",
    "tangential_projection",
    "bruggeman",
    "ergun_coefficient",
    "kozeny_carman",
    "kozeny_carman_permeability",
]
