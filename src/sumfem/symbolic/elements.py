"""Specification of finite elements for fields and known coefficients."""

import basix.ufl
from ufl.finiteelement import AbstractFiniteElement

from ..error_handling import ComponentError
from .domains import Domain

__all__ = ["ElementSpec", "as_element"]

ElementSpec = AbstractFiniteElement | str | tuple
"""An element given either as a Basix/UFL element, or as a tuple
``(family, degree)`` or ``(family, degree, shape)``. The family ``"Real"``
gives a Real element, for which the degree is ignored."""


def as_element(spec: ElementSpec, domain: Domain) -> AbstractFiniteElement:
    """Convert an element specification to an element on a domain's cell.

    Args:
        spec: An element, see :data:`ElementSpec`.
        domain: The :class:`~sumfem.symbolic.domains.Domain` the element is
            used on.
    """
    cell_type = domain.ufl_coordinate_element().cell_type
    if isinstance(spec, AbstractFiniteElement):
        el_cell = getattr(spec, "cell_type", None)
        if el_cell is not None and el_cell != cell_type:
            raise ComponentError(
                f"Element is defined on {el_cell.name}, "
                f"but domain {domain.name!r} has cells of type {cell_type.name}."
            )
        return spec
    if isinstance(spec, str):
        spec = (spec,)
    if not isinstance(spec, tuple) or not (1 <= len(spec) <= 3):
        raise TypeError(
            "Element must be a UFL element or a tuple (family, degree[, shape]), "
            f"got {spec!r}."
        )
    family = spec[0]
    degree = spec[1] if len(spec) > 1 else None
    shape = tuple(spec[2]) if len(spec) > 2 else ()
    if family.lower() == "real":
        return basix.ufl.real_element(cell_type, shape)
    if degree is None:
        raise TypeError(f"Element {spec!r} needs a degree.")
    # Without a shape, so that vector-valued families such as RT stay unblocked
    return basix.ufl.element(family, cell_type, degree, shape=shape or None)
