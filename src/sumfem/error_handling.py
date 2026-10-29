"""Exceptions and name validation shared by the symbolic layer."""

__all__ = [
    "ComponentError",
    "SystemBuildError",
    "check_name",
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
