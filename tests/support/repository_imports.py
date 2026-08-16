"""Compact, read-only import index for repository boundary tests.

Boundary tests repeatedly inspect the same source files from different ownership
perspectives.  Cache only immutable import summaries for the duration of one
pytest process; do not retain source text or AST objects.  The repository is
expected to remain unchanged while a test session is running.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class _ImportSummary:
    absolute_modules: frozenset[str]
    absolute_roots: frozenset[str]
    dynamic_modules: frozenset[str]


@lru_cache(maxsize=None)
def python_files(root: Path) -> tuple[Path, ...]:
    """Return a stable, sorted Python source inventory below ``root``."""

    return tuple(
        sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    )


@lru_cache(maxsize=None)
def _import_summary(path: Path) -> _ImportSummary:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    absolute_modules: set[str] = set()
    importlib_names = {"importlib"}
    import_module_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute_modules.update(alias.name for alias in node.names)
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                absolute_modules.add(node.module)
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_names.add(alias.asname or alias.name)

    dynamic_modules: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_import_module = (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_names
            and function.attr == "import_module"
        ) or (
            isinstance(function, ast.Name)
            and function.id in import_module_names
        )
        is_builtin_import = (
            isinstance(function, ast.Name) and function.id == "__import__"
        )
        if not (is_import_module or is_builtin_import):
            continue
        module = node.args[0]
        if isinstance(module, ast.Constant) and isinstance(module.value, str):
            dynamic_modules.add(module.value)

    return _ImportSummary(
        absolute_modules=frozenset(absolute_modules),
        absolute_roots=frozenset(
            module.split(".", 1)[0] for module in absolute_modules
        ),
        dynamic_modules=frozenset(dynamic_modules),
    )


def absolute_import_modules(path: Path) -> set[str]:
    """Return absolute statically imported module names in ``path``."""

    return set(_import_summary(path).absolute_modules)


def absolute_import_roots(path: Path) -> set[str]:
    """Return top-level roots of absolute static imports in ``path``."""

    return set(_import_summary(path).absolute_roots)


def dynamic_import_modules(path: Path) -> set[str]:
    """Return literal modules loaded through supported dynamic import forms."""

    return set(_import_summary(path).dynamic_modules)


def imported_modules(path: Path) -> set[str]:
    """Return static absolute and supported literal dynamic imports."""

    summary = _import_summary(path)
    return set(summary.absolute_modules | summary.dynamic_modules)


__all__ = [
    "absolute_import_modules",
    "absolute_import_roots",
    "dynamic_import_modules",
    "imported_modules",
    "python_files",
]
