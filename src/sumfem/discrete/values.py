"""Assigning values to DOLFINx functions."""

from __future__ import annotations

from collections.abc import Callable

import dolfinx.fem
import numpy as np
import numpy.typing as npt

__all__ = ["Evaluation", "Value", "assign"]

Value = (
    float
    | npt.ArrayLike
    | dolfinx.fem.Function
    | dolfinx.fem.Expression
    | Callable[[npt.NDArray[np.floating]], npt.ArrayLike]
)
"""A value for a function: a number, a constant array, another function, a
compiled expression, or a callable of the coordinates ``x``, of shape
``(3, num_points)``."""


def assign(function: dolfinx.fem.Function, value: Value, factor: float = 1.0) -> None:
    """Set the values of a function.

    Args:
        function: The function to set.
        value: See :data:`Value`. A number sets every component. A constant
            array sets each component, and must have the function's value
            shape.
        factor: Factor applied after assigning, e.g. to convert the value to
            SI base units.

    Raises:
        ValueError: If a constant array has the wrong shape.
    """
    if isinstance(value, dolfinx.fem.Function | dolfinx.fem.Expression) or callable(
        value
    ):
        function.interpolate(value)
    else:
        array = np.asarray(value, dtype=function.x.array.dtype)
        shape = function.ufl_shape
        bs = function.function_space.dofmap.index_map_bs
        if array.ndim == 0:
            function.x.array[:] = array
        elif array.shape != shape:
            raise ValueError(
                f"Value for {function.name!r} has shape {array.shape}, "
                f"expected {shape}."
            )
        elif bs == array.size:
            # One value per component at every degree of freedom. Ghost
            # entries are set too, so no scatter is needed.
            function.x.array[:] = np.tile(array.ravel(), function.x.array.size // bs)
        else:
            function.interpolate(
                lambda x: np.broadcast_to(
                    array.reshape(-1, 1), (array.size, x.shape[1])
                )
            )
    if factor != 1.0:
        function.x.array[:] *= factor


class Evaluation:
    """A compiled expression kept to be evaluated into a function again.

    The values of the expression's inputs may change between evaluations,
    e.g. the time in a boundary value, or the temperature in an update of a
    liquid fraction.

    Args:
        function: The function written into.
        expression: The expression, compiled at the function's
            interpolation points.
    """

    _function: dolfinx.fem.Function
    """The function written into."""

    _expression: dolfinx.fem.Expression
    """The compiled expression."""

    def __init__(
        self, function: dolfinx.fem.Function, expression: dolfinx.fem.Expression
    ):
        self._function = function
        self._expression = expression

    @property
    def function(self) -> dolfinx.fem.Function:
        """The function written into."""
        return self._function

    @property
    def expression(self) -> dolfinx.fem.Expression:
        """The compiled expression."""
        return self._expression

    def __call__(self) -> None:
        """Evaluate the expression with the current inputs into the function."""
        self._function.interpolate(self._expression)
        self._function.x.scatter_forward()
