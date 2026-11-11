"""Models of electrochemistry, built on :mod:`sumfem.models`.

- Electrolytes: :class:`Electrolyte`, a dilute solution of :class:`Ion`
  and the constants and properties its models share.
- Porous electrodes: :class:`PorousElectrode`, and the :class:`Region` of
  a cell filled with electrolyte.
- Charge and transport in the electrolyte: :class:`ElectrolyteConduction`
  and :class:`NernstPlanck`.
- Kinetics: :class:`ExchangeCurrentButlerVolmer`, :class:`ButlerVolmer`
  and :class:`MassTransferButlerVolmer`, on :class:`ElectrodeReaction`,
  with the Butler-Volmer equation :func:`~sumfem.models.butler_volmer`.

The package imports from the core of :mod:`sumfem`, and never the other
way around.
"""

from .charge import ElectrolyteConduction
from .electrolyte import Electrolyte, Ion
from .kinetics import (
    ButlerVolmer,
    ElectrodeReaction,
    ExchangeCurrentButlerVolmer,
    MassTransferButlerVolmer,
)
from .porous import PorousElectrode, Region
from .transport import NernstPlanck

__all__ = [
    "Electrolyte",
    "Ion",
    "PorousElectrode",
    "Region",
    "ElectrolyteConduction",
    "NernstPlanck",
    "ButlerVolmer",
    "ElectrodeReaction",
    "ExchangeCurrentButlerVolmer",
    "MassTransferButlerVolmer",
]
