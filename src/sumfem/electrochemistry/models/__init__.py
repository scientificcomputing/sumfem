r"""The equations of electrochemistry: residuals of the materials' fields.

- Charge and transport in the electrolyte: :class:`ElectrolyteConduction`
  and :class:`NernstPlanck`, with storage :class:`TransientNernstPlanck`.
- Kinetics: :class:`ExchangeCurrentButlerVolmer`, :class:`ButlerVolmer`,
  :class:`MassTransferButlerVolmer` and :class:`PistonVelocityButlerVolmer`,
  on :class:`ElectrodeReaction`, with the Butler-Volmer equation
  :func:`~sumfem.models.butler_volmer`.
- Membranes: :class:`ProtonExchangeMembrane`, with the flow of water of
  :class:`SchloeglFlow`, whose current the protons of an electrode carry
  through its face, :class:`MembraneProtonFlux`.
- The flow of the electrolyte through the electrodes: any
  :class:`ElectrodeFlow`, e.g. :class:`DarcyFlow`, created from a
  :class:`FlowSetup`.

The models import from the materials, and nothing from the cells.
"""

from .charge import ElectrolyteConduction
from .electrode_flow import DarcyFlow, ElectrodeFlow, FlowSetup
from .kinetics import (
    ButlerVolmer,
    ElectrodeReaction,
    ExchangeCurrentButlerVolmer,
    MassTransferButlerVolmer,
    PistonVelocityButlerVolmer,
)
from .membrane import MembraneProtonFlux, ProtonExchangeMembrane, SchloeglFlow
from .transport import NernstPlanck, TransientNernstPlanck

__all__ = [
    "ButlerVolmer",
    "DarcyFlow",
    "ElectrodeFlow",
    "ElectrodeReaction",
    "ElectrolyteConduction",
    "ExchangeCurrentButlerVolmer",
    "FlowSetup",
    "MassTransferButlerVolmer",
    "MembraneProtonFlux",
    "NernstPlanck",
    "PistonVelocityButlerVolmer",
    "ProtonExchangeMembrane",
    "SchloeglFlow",
    "TransientNernstPlanck",
]
