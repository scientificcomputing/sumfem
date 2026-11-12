"""Models built on :mod:`sumfem`, to be summed and extended.

- Accumulation, diffusion, advection and sources: :class:`Accumulation`,
  :class:`Diffusion`, :class:`ConvectiveAdvection` and
  :class:`ConservativeAdvection`, :class:`Source` and :class:`Poisson`, and
  Ohmic :class:`Conduction`.
- Boundaries: :class:`BoundaryFlux`, :class:`Outflow`, and constraints
  with Lagrange multipliers, :class:`BoundaryConstraint` and
  :class:`DirichletMultiplier`.
- Interfaces between subdomains: :class:`InterfaceLagrangeMultiplier`, a condition
  across an interface with a Lagrange multiplier for the flux, or penalized,
  by an exchange flux, :class:`InterfacePenalty`, or by an energy,
  :class:`InterfacePenaltyEnergy`.
- Reactions between species in volumes, on surfaces and across membranes
  between two volumes: :class:`Reaction` by mass action and
  :class:`RateReaction` at any rate, with :func:`mass_action`,
  :func:`arrhenius` and :func:`butler_volmer`.
- Flow: :class:`Darcy` through a porous medium, and :class:`Stokes` flow
  with the momentum terms :class:`Convection`, :class:`DarcyDrag` and
  :class:`ForchheimerDrag` and grad-div stabilization, :class:`GradDiv`,
  summed into :class:`NavierStokes`, :class:`Brinkman` and
  :class:`BrinkmanForchheimer`.
- Effective properties of porous media: the Bruggeman, Kozeny-Carman and
  Ergun relations.

Models of electrochemistry are in :mod:`sumfem.electrochemistry`.
"""

from .advection import Advection, ConservativeAdvection, ConvectiveAdvection
from .boundary import BoundaryConstraint, BoundaryFlux, DirichletMultiplier, Outflow
from .charge import Conduction
from .diffusion import Accumulation, Diffusion, Poisson, Source
from .flow import (
    Brinkman,
    BrinkmanForchheimer,
    Convection,
    Darcy,
    DarcyDrag,
    ForchheimerDrag,
    GradDiv,
    NavierStokes,
    Stokes,
)
from .interface import (
    InterfaceLagrangeMultiplier,
    InterfacePenalty,
    InterfacePenaltyEnergy,
)
from .operators import apply_coefficient, normal_projection, tangential_projection
from .porous import (
    bruggeman,
    ergun_coefficient,
    kozeny_carman,
    kozeny_carman_permeability,
)
from .reactions import RateReaction, Reaction, arrhenius, butler_volmer, mass_action

__all__ = [
    "mass_action",
    "arrhenius",
    "butler_volmer",
    "RateReaction",
    "Reaction",
    "Accumulation",
    "Advection",
    "ConservativeAdvection",
    "ConvectiveAdvection",
    "BoundaryConstraint",
    "BoundaryFlux",
    "Conduction",
    "Diffusion",
    "InterfaceLagrangeMultiplier",
    "InterfacePenalty",
    "InterfacePenaltyEnergy",
    "Brinkman",
    "BrinkmanForchheimer",
    "Darcy",
    "NavierStokes",
    "Stokes",
    "Convection",
    "DarcyDrag",
    "ForchheimerDrag",
    "GradDiv",
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
