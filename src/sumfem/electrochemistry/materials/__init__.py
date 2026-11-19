"""What the cells are made of: ions, electrolytes, couples and electrodes.

These describe materials and their properties, as parameters and the
expressions of them that the models share, and hold no equations of their
own.

- Electrolytes: :class:`Electrolyte`, a dilute solution of :class:`Ion`.
- Redox couples: :class:`RedoxCouple`, the reaction of a half-cell.
- Porous electrodes: :class:`PorousElectrode`, and the :class:`Region` of
  a cell filled with electrolyte.

The materials import nothing from the models or the cells.
"""

from .couple import RedoxCouple
from .electrode import PorousElectrode, Region
from .electrolyte import Electrolyte, Ion

__all__ = ["Electrolyte", "Ion", "PorousElectrode", "RedoxCouple", "Region"]
