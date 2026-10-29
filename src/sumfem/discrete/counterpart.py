"""The discrete counterpart of a symbolic object on a discretization."""

from __future__ import annotations

import functools
import typing

if typing.TYPE_CHECKING:
    from .discretization import Discretization

__all__ = ["counterpart"]


@functools.singledispatch
def counterpart(
    symbolic: typing.Any, discretization: Discretization, **options: typing.Any
) -> typing.Any:
    """The discrete counterpart of a symbolic object on a discretization.

    Dispatches on the type of the symbolic object. A discrete counterpart
    has the name of its symbolic one in :mod:`sumfem.discrete`, e.g. a
    symbolic :class:`~sumfem.symbolic.system.System` has a
    :class:`~sumfem.discrete.system.System`, and a symbolic
    :class:`~sumfem.symbolic.transfer.Interpolation` a
    :class:`~sumfem.discrete.transfer.Interpolation`. Expressions give
    expressions in the discrete functions, and functionals
    :class:`dolfinx.fem.Form`.

    Indexing a discretization, ``disc[symbolic]``, gives the counterpart
    with default options and keeps it. Call this function for other options,
    e.g. ``counterpart(interpolation, disc, padding=1e-10)``. Register the
    discrete counterpart of a new symbolic type, e.g. a transfer by
    projection, with ``@counterpart.register``::

        @counterpart.register
        def _(projection: Projection, discretization, **options):
            return DiscreteProjection(discretization, projection, **options)

    Args:
        symbolic: The symbolic object.
        discretization: The discretization holding its terminals.
        options: Discrete options of the counterpart, e.g. the point search
            of an interpolation.

    Raises:
        NotImplementedError: If no discrete counterpart is registered for
            the type.
    """
    raise NotImplementedError(
        f"No discrete counterpart is registered for {type(symbolic).__name__}. "
        "Register one with counterpart.register."
    )
