from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.repository_imports import (
    absolute_import_modules,
    absolute_import_roots,
    dynamic_import_modules,
    imported_modules,
    python_files,
)

pytestmark = pytest.mark.unit


def _source(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_repository_import_index_preserves_static_and_dynamic_imports(
    tmp_path: Path,
) -> None:
    path = _source(
        tmp_path,
        """
import alpha.beta
import importlib as loader
from gamma.delta import value
from importlib import import_module as load

loader.import_module("dynamic.loader")
load("dynamic.alias")
__import__("dynamic.builtin")
""",
    )

    assert absolute_import_modules(path) == {
        "alpha.beta",
        "gamma.delta",
        "importlib",
    }
    assert absolute_import_roots(path) == {"alpha", "gamma", "importlib"}
    assert dynamic_import_modules(path) == {
        "dynamic.alias",
        "dynamic.builtin",
        "dynamic.loader",
    }
    assert imported_modules(path) == {
        "alpha.beta",
        "dynamic.alias",
        "dynamic.builtin",
        "dynamic.loader",
        "gamma.delta",
        "importlib",
    }


def test_repository_import_index_resolves_dynamic_aliases_independent_of_source_order(
    tmp_path: Path,
) -> None:
    path = _source(
        tmp_path,
        """
loader.import_module("dynamic.loader")
load("dynamic.alias")
__import__("dynamic.builtin")

import importlib as loader
from importlib import import_module as load
""",
    )

    assert dynamic_import_modules(path) == {
        "dynamic.alias",
        "dynamic.builtin",
        "dynamic.loader",
    }


def test_repository_import_index_excludes_relative_and_nonliteral_imports(
    tmp_path: Path,
) -> None:
    path = _source(
        tmp_path,
        """
import importlib
from .relative import value

module_name = "dynamic.nonliteral"
importlib.import_module(module_name)
""",
    )

    assert absolute_import_modules(path) == {"importlib"}
    assert absolute_import_roots(path) == {"importlib"}
    assert dynamic_import_modules(path) == set()


def test_repository_import_index_reaches_nested_statement_bodies(
    tmp_path: Path,
) -> None:
    path = _source(
        tmp_path,
        """
if enabled: import conditional.branch

def nested_function():
    import function.local

class NestedType:
    from class_local.module import value

try:
    while retrying:
        from loop.local import value
except RuntimeError:
    import exception.local

match payload:
    case {"kind": kind}:
        import match.local

import semicolon.one; import semicolon.two
""",
    )

    assert absolute_import_modules(path) == {
        "conditional.branch",
        "class_local.module",
        "exception.local",
        "function.local",
        "loop.local",
        "match.local",
        "semicolon.one",
        "semicolon.two",
    }
    assert absolute_import_roots(path) == {
        "class_local",
        "conditional",
        "exception",
        "function",
        "loop",
        "match",
        "semicolon",
    }


def test_repository_import_index_returns_caller_owned_sets(tmp_path: Path) -> None:
    path = _source(tmp_path, "import stable.module\n")

    roots = absolute_import_roots(path)
    roots.add("caller_mutation")

    assert absolute_import_roots(path) == {"stable"}


def test_repository_import_index_returns_sorted_immutable_source_inventory(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    cached = tmp_path / "__pycache__"
    nested.mkdir()
    cached.mkdir()
    later = tmp_path / "z.py"
    earlier = nested / "a.py"
    ignored = cached / "ignored.py"
    for path in (later, earlier, ignored):
        path.write_text("", encoding="utf-8")

    assert python_files(tmp_path) == (earlier, later)
