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


@pytest.mark.parametrize(
    "module, page",
    [
        ("sumfem.symbolic", "symbolic"),
        ("sumfem.discrete", "discrete"),
        ("sumfem.solvers", "solvers"),
        ("sumfem.solvers.solver", "solvers"),
        ("sumfem.post_processing", "post_processing"),
        ("sumfem.plotting", "plotting"),
        ("sumfem.models", "models"),
    ],
)
def test_every_public_name_is_on_its_api_page(module, page):
    public = importlib.import_module(module).__all__
    names = documented(API / f"{page}.md")
    # A name may be documented under its module or a package that re-exports it
    parts = module.split(".")
    owners = [".".join(parts[: k + 1]) for k in range(1, len(parts))]
    missing = sorted(
        n for n in public if not any(f"{owner}.{n}" in names for owner in owners)
    )
    assert not missing, f"{missing} of {module} are missing from docs/api/{page}.md"
