"""Every public name of the package is documented on an API page."""

import importlib
import pathlib
import re

import pytest

API = pathlib.Path(__file__).parents[1] / "docs" / "api"
"""The pages of the Python API."""

DIRECTIVE = re.compile(r"^\.\. (auto\w+|currentmodule):: (\S+)", re.MULTILINE)


def documented(page: pathlib.Path) -> set[str]:
    """The full names an API page documents."""
    names, module = set(), ""
    for directive, target in DIRECTIVE.findall(page.read_text()):
        if directive in ("automodule", "currentmodule"):
            module = target
        else:
            names.add(target if "." in target else f"{module}.{target}")
    return names


def resolve(name: str) -> object:
    """The object a documented full name refers to, or ``None``."""
    module, _, attribute = name.rpartition(".")
    try:
        return getattr(importlib.import_module(module), attribute, None)
    except ImportError:
        return None


@pytest.mark.parametrize(
    "module, page",
    [
        ("sumfem.symbolic", "symbolic"),
        ("sumfem.discrete", "discrete"),
        ("sumfem.solvers", "solvers"),
        ("sumfem.solvers.solver", "solvers"),
        ("sumfem.solvers.time_stepping", "solvers"),
        ("sumfem.solvers.simulation", "solvers"),
        ("sumfem.post_processing", "post_processing"),
        ("sumfem.plotting", "plotting"),
        ("sumfem.models", "models"),
        ("sumfem.electrochemistry", "electrochemistry"),
    ],
)
def test_every_public_name_is_on_its_api_page(module, page):
    imported = importlib.import_module(module)
    names = documented(API / f"{page}.md")
    # A name may be documented under its module, a package that re-exports
    # it, or the module that defines it, as a constant must be, whose
    # docstring autodoc finds only in the source of the module that defines it
    parts = module.split(".")
    owners = [".".join(parts[: k + 1]) for k in range(1, len(parts))]
    objects = [resolve(name) for name in names]
    missing = sorted(
        n
        for n in imported.__all__
        if not any(f"{owner}.{n}" in names for owner in owners)
        and not any(o is getattr(imported, n) for o in objects if o is not None)
    )
    assert not missing, f"{missing} of {module} are missing from docs/api/{page}.md"
