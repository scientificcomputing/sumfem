r"""Models of electrochemistry, built on :mod:`sumfem.models`.

The package has three layers, each importing only from those before it:

- :mod:`~sumfem.electrochemistry.materials`, what the cells are made of:
  :class:`Electrolyte` of :class:`Ion`, :class:`RedoxCouple`,
  :class:`PorousElectrode` and :class:`Region`.
- :mod:`~sumfem.electrochemistry.models`, the equations: charge and
  transport, :class:`ElectrolyteConduction`, :class:`NernstPlanck` and
  :class:`TransientNernstPlanck`; kinetics, :class:`ButlerVolmer` and its
  variants on :class:`ElectrodeReaction`; membranes,
  :class:`ProtonExchangeMembrane`, :class:`SchloeglFlow` and
  :class:`MembraneProtonFlux`; and the flow through the electrodes,
  :class:`ElectrodeFlow` and :class:`DarcyFlow`.
- :mod:`~sumfem.electrochemistry.cells`, assembled from them:
  :class:`RedoxHalfCell` and :class:`FullCell`, with the specifications
  :class:`HalfCellSpec`, :class:`MembraneSpec` and :class:`FlowSpec`.

Every name is also imported from here.

The package imports from the core of :mod:`sumfem`, and never the other
way around.
"""

from .cells import FlowSpec, FullCell, HalfCellSpec, MembraneSpec, RedoxHalfCell
from .materials import Electrolyte, Ion, PorousElectrode, RedoxCouple, Region
from .models import (
    ButlerVolmer,
    DarcyFlow,
    ElectrodeFlow,
    ElectrodeReaction,
    ElectrolyteConduction,
    ExchangeCurrentButlerVolmer,
    FlowSetup,
    MassTransferButlerVolmer,
    MembraneProtonFlux,
    NernstPlanck,
    PistonVelocityButlerVolmer,
    ProtonExchangeMembrane,
    SchloeglFlow,
    TransientNernstPlanck,
)

__all__ = [
    "Electrolyte",
    "Ion",
    "PorousElectrode",
    "Region",
    "ElectrolyteConduction",
    "NernstPlanck",
    "TransientNernstPlanck",
    "ProtonExchangeMembrane",
    "SchloeglFlow",
    "MembraneProtonFlux",
    "RedoxCouple",
    "RedoxHalfCell",
    "FullCell",
    "HalfCellSpec",
    "MembraneSpec",
    "FlowSpec",
    "ElectrodeFlow",
    "DarcyFlow",
    "FlowSetup",
    "ButlerVolmer",
    "ElectrodeReaction",
    "ExchangeCurrentButlerVolmer",
    "MassTransferButlerVolmer",
    "PistonVelocityButlerVolmer",
]
