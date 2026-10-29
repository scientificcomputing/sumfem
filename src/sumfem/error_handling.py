"""Exceptions and name validation shared by the symbolic layer."""

import ufl

__all__ = [
    "ComponentError",
    "SystemBuildError",
    "check_name",
    "check_rank",
    "qualified_name",
]


def qualified_name(cls: type) -> str:
    """Module-qualified name of a class.

    For example ``sumfem.symbolic.domains.Domain``.

    Used in error messages, to tell our classes apart from UFL classes with
    similar names.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


class ComponentError(ValueError):
    """Raised when a component, or a sum of components, is defined inconsistently."""


class SystemBuildError(ValueError):
    """Raised when a residual cannot be turned into a system of equations."""


def check_name(name: object, kind: str, allow_dots: bool = True) -> str:
    """Validate a user-facing name and return it.

    Args:
        name: The candidate name.
        kind: What is being named, used in error messages.
        allow_dots: Whether ``.`` is allowed. Component names may not contain
            dots, since qualified parameter names are ``"<component>.<name>"``.
    """
    if not isinstance(name, str):
        raise TypeError(f"{kind} name must be a string, got {type(name).__name__}.")
    if name == "" or name != name.strip() or any(c.isspace() for c in name):
        raise ComponentError(
            f"{kind} name {name!r} must be non-empty and contain no whitespace."
        )
    if not allow_dots and "." in name:
        raise ComponentError(f"{kind} name {name!r} may not contain '.'.")
    return name


def check_rank(form: ufl.Form | ufl.ZeroBaseForm, rank: int, what: str) -> None:
    """Check that a form has exactly the given rank.

    A form of rank ``r`` has arguments numbered ``0, ..., r - 1``: a residual
    has rank 1 (test functions only), a Jacobian or preconditioner rank 2
    (test and trial functions). For a residual this rejects trial functions,
    and a form with no test function at all.

    Args:
        form: The form to check.
        rank: The expected rank.
        what: Description of the form, used in the error message.

    Raises:
        ComponentError: If the argument numbers of the form are not
            ``0, ..., rank - 1``.
    """
    numbers = {a.number() for a in form.arguments()}
    expected = set(range(rank))
    if numbers != expected:
        raise ComponentError(
            f"{what} must have rank {rank}, "
            f"but its arguments have numbers {sorted(numbers)}."
        )
