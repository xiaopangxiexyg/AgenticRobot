#!/usr/bin/env python3
"""Validate the source projection used by refinement-annotation machines."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_MANIFEST = Path("release/annotation_projection.toml")


class AnnotationProjectionError(ValueError):
    """Raised when the annotation projection is incomplete or unsafe."""


@dataclass(frozen=True)
class AnnotationProjectionManifest:
    schema: str
    entrypoint: str
    closure_entry_modules: tuple[str, ...]
    include: tuple[str, ...]
    deny_import_roots: frozenset[str]


@dataclass(frozen=True)
class AnnotationProjectionReport:
    manifest_path: Path
    included_paths: tuple[Path, ...]
    closure_modules: tuple[str, ...]
    closure_paths: tuple[Path, ...]

    def to_payload(self, *, repository_root: Path) -> dict[str, object]:
        return {
            "manifest": str(self.manifest_path.relative_to(repository_root)),
            "included_paths": [
                str(path.relative_to(repository_root))
                for path in self.included_paths
            ],
            "closure_modules": list(self.closure_modules),
            "closure_paths": [
                str(path.relative_to(repository_root))
                for path in self.closure_paths
            ],
        }


def _flat_string_manifest(path: Path) -> dict[str, str | list[str]]:
    """Read the intentionally flat, Python-compatible TOML subset.

    The projection manifest uses only top-level string and string-array
    assignments. Parsing that strict subset with ``ast`` keeps the checker
    usable on the project's supported Python 3.10 without adding a TOML
    parser to the annotation runtime environment.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise AnnotationProjectionError(
            f"Cannot parse annotation projection manifest {path}: {exc}"
        ) from exc
    values: dict[str, str | list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            raise AnnotationProjectionError(
                "Projection manifest permits only top-level assignments."
            )
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise AnnotationProjectionError(
                "Projection manifest keys must be simple names."
            )
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise AnnotationProjectionError(
                f"Projection manifest value for {target.id!r} is not literal."
            ) from exc
        if isinstance(value, str):
            values[target.id] = value
            continue
        if isinstance(value, list) and all(
            isinstance(item, str) for item in value
        ):
            values[target.id] = value
            continue
        raise AnnotationProjectionError(
            f"Projection manifest value for {target.id!r} must be a string "
            "or string array."
        )
    return values


def load_annotation_projection_manifest(
    path: str | Path,
) -> AnnotationProjectionManifest:
    manifest_path = Path(path).resolve()
    values = _flat_string_manifest(manifest_path)
    expected = {
        "schema",
        "entrypoint",
        "closure_entry_modules",
        "include",
        "deny_import_roots",
    }
    unknown = sorted(set(values) - expected)
    missing = sorted(expected - set(values))
    if unknown or missing:
        raise AnnotationProjectionError(
            f"Projection manifest keys mismatch: missing={missing}, "
            f"unknown={unknown}."
        )

    def required_string(name: str) -> str:
        value = values[name]
        if not isinstance(value, str) or not value.strip():
            raise AnnotationProjectionError(f"{name} must be non-empty.")
        return value.strip()

    def required_list(name: str) -> tuple[str, ...]:
        value = values[name]
        if not isinstance(value, list) or not value:
            raise AnnotationProjectionError(f"{name} must be non-empty.")
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise AnnotationProjectionError(f"{name} contains an empty value.")
        if len(normalized) != len(set(normalized)):
            raise AnnotationProjectionError(f"{name} contains duplicates.")
        return normalized

    schema = required_string("schema")
    if schema != "dexteleop.annotation-source-projection.v1":
        raise AnnotationProjectionError(
            f"Unsupported annotation projection schema: {schema!r}."
        )
    entrypoint = required_string("entrypoint")
    if entrypoint != "python -m control_station refinement-annotation":
        raise AnnotationProjectionError(
            f"Unsupported annotation projection entrypoint: {entrypoint!r}."
        )
    return AnnotationProjectionManifest(
        schema=schema,
        entrypoint=entrypoint,
        closure_entry_modules=required_list("closure_entry_modules"),
        include=required_list("include"),
        deny_import_roots=frozenset(required_list("deny_import_roots")),
    )


@dataclass(frozen=True)
class _ResolvedModule:
    name: str
    path: Path
    is_package: bool


def _resolve_local_module(
    repository_root: Path,
    module_name: str,
) -> _ResolvedModule | None:
    relative = Path(*module_name.split("."))
    module_path = repository_root / relative.with_suffix(".py")
    if module_path.is_file():
        return _ResolvedModule(module_name, module_path.resolve(), False)
    package_path = repository_root / relative / "__init__.py"
    if package_path.is_file():
        return _ResolvedModule(module_name, package_path.resolve(), True)
    return None


def _parent_module_names(module_name: str) -> Iterable[str]:
    parts = module_name.split(".")
    for length in range(1, len(parts)):
        yield ".".join(parts[:length])


def _absolute_from_module(
    *,
    current: _ResolvedModule,
    level: int,
    module: str | None,
) -> str:
    if level == 0:
        return module or ""
    package = current.name if current.is_package else current.name.rpartition(".")[0]
    package_parts = package.split(".") if package else []
    remove_count = level - 1
    if remove_count > len(package_parts):
        return ""
    base_parts = package_parts[: len(package_parts) - remove_count]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _import_candidates(
    module: _ResolvedModule,
) -> set[str]:
    try:
        tree = ast.parse(
            module.path.read_text(encoding="utf-8"),
            filename=str(module.path),
        )
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise AnnotationProjectionError(
            f"Cannot inspect imports in {module.path}: {exc}"
        ) from exc
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _absolute_from_module(
            current=module,
            level=node.level,
            module=node.module,
        )
        if base:
            candidates.add(base)
        if any(alias.name == "*" for alias in node.names):
            continue
        for alias in node.names:
            if base:
                candidates.add(f"{base}.{alias.name}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "importlib"
            and function.attr == "import_module"
        )
        imported_name = node.args[0]
        if (
            is_dynamic_import
            and isinstance(imported_name, ast.Constant)
            and isinstance(imported_name.value, str)
        ):
            candidates.add(imported_name.value)
    return candidates


def annotation_import_closure(
    repository_root: str | Path,
    entry_modules: Sequence[str],
) -> Mapping[str, Path]:
    root = Path(repository_root).resolve()
    pending = list(entry_modules)
    resolved: dict[str, Path] = {}
    while pending:
        module_name = pending.pop()
        if module_name in resolved:
            continue
        module = _resolve_local_module(root, module_name)
        if module is None:
            continue
        resolved[module_name] = module.path
        for parent_name in _parent_module_names(module_name):
            if parent_name not in resolved:
                pending.append(parent_name)
        for candidate in _import_candidates(module):
            if candidate not in resolved:
                pending.append(candidate)
    return resolved


def validate_annotation_projection(
    repository_root: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> AnnotationProjectionReport:
    root = Path(repository_root).resolve()
    unresolved_manifest = Path(manifest_path)
    if not unresolved_manifest.is_absolute():
        unresolved_manifest = root / unresolved_manifest
    manifest = load_annotation_projection_manifest(unresolved_manifest)

    included_paths: list[Path] = []
    for relative_value in manifest.include:
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise AnnotationProjectionError(
                f"Projection include must stay repository-relative: {relative}."
            )
        path = (root / relative).resolve()
        if root not in path.parents:
            raise AnnotationProjectionError(
                f"Projection include escaped repository root: {relative}."
            )
        if not path.is_file():
            raise AnnotationProjectionError(
                f"Projection include does not exist: {relative}."
            )
        included_paths.append(path)

    closure = annotation_import_closure(
        root,
        manifest.closure_entry_modules,
    )
    forbidden_modules = sorted(
        module_name
        for module_name in closure
        if module_name.split(".", 1)[0] in manifest.deny_import_roots
    )
    if forbidden_modules:
        raise AnnotationProjectionError(
            "Annotation import closure reached forbidden modules: "
            + ", ".join(forbidden_modules)
        )
    included_set = set(included_paths)
    missing_paths = sorted(
        path.relative_to(root)
        for path in closure.values()
        if path not in included_set
    )
    if missing_paths:
        raise AnnotationProjectionError(
            "Annotation import closure is not covered by the projection: "
            + ", ".join(str(path) for path in missing_paths)
        )

    return AnnotationProjectionReport(
        manifest_path=unresolved_manifest.resolve(),
        included_paths=tuple(sorted(included_paths)),
        closure_modules=tuple(sorted(closure)),
        closure_paths=tuple(sorted(set(closure.values()))),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the DexTeleop refinement-annotation projection."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_annotation_projection(
            args.repository_root,
            args.manifest,
        )
    except AnnotationProjectionError as exc:
        raise SystemExit(f"annotation projection invalid: {exc}") from exc
    payload = report.to_payload(repository_root=args.repository_root.resolve())
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "annotation projection ok "
            f"files={len(report.included_paths)} "
            f"modules={len(report.closure_modules)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
