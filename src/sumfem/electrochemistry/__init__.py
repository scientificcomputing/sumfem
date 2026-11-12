r"""Models of electrochemistry, built on :mod:`sumfem.models`.

- Electrolytes: :class:`Electrolyte`, a dilute solution of :class:`Ion`
  and the constants and properties its models share.
- Porous electrodes: :class:`PorousElectrode`, and the :class:`Region` of
  a cell filled with electrolyte.
- Charge and transport in the electrolyte: :class:`ElectrolyteConduction`
  and :class:`NernstPlanck`, with storage :class:`TransientNernstPlanck`.
- Kinetics: :class:`ExchangeCurrentButlerVolmer`, :class:`ButlerVolmer`,
  :class:`MassTransferButlerVolmer` and :class:`PistonVelocityButlerVolmer`,
  on :class:`ElectrodeReaction`, with the Butler-Volmer equation
  :func:`~sumfem.models.butler_volmer`.
- Half-cells of a redox flow battery: :class:`RedoxHalfCell`, negative or
  positive, of any :class:`RedoxCouple` and electrolyte.

The package imports from the core of :mod:`sumfem`, and never the other
way around.
"""

from .charge import ElectrolyteConduction
from .couple import RedoxCouple
from .electrolyte import Electrolyte, Ion
from .half_cell import RedoxHalfCell
from .kinetics import (
    ButlerVolmer,
    ElectrodeReaction,
    ExchangeCurrentButlerVolmer,
    MassTransferButlerVolmer,
    PistonVelocityButlerVolmer,
)
from .porous import PorousElectrode, Region
from .transport import NernstPlanck, TransientNernstPlanck

__all__ = [
    "Electrolyte",
    "Ion",
    "PorousElectrode",
    "Region",
    "ElectrolyteConduction",
    "NernstPlanck",
    "TransientNernstPlanck",
    "RedoxCouple",
    "RedoxHalfCell",
    "ButlerVolmer",
    "ElectrodeReaction",
    "ExchangeCurrentButlerVolmer",
    "MassTransferButlerVolmer",
    "PistonVelocityButlerVolmer",
]
