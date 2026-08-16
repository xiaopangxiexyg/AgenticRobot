from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import LabError

FRAMEWORK_BOUNDARY_SCHEMA_ID = "lab.framework-boundary.v1"


@dataclass(frozen=True)
class FrameworkBoundaryPolicy:
    core_globs: tuple[str, ...]
    controlled_globs: tuple[str, ...]
    forbidden_core_literals: tuple[str, ...]
    max_controlled_file_bytes: int
    path: Path
    sha256: str


def _string_list(payload: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(value, str) and value for value in payload)
    ):
        raise LabError(f"{label} must be a non-empty string list")
    if len(payload) != len(set(payload)):
        raise LabError(f"{label} contains duplicates")
    return tuple(payload)


def _safe_glob(value: str) -> str:
    path = PurePosixPath(value)
    if value.startswith(("/", "~")) or "\\" in value or "\x00" in value:
        raise LabError(f"unsafe framework boundary glob: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(f"unsafe framework boundary glob: {value!r}")
    return value


def load_framework_boundary_policy(path: Path) -> FrameworkBoundaryPolicy:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read framework boundary policy {resolved}: {exc}") from exc
    expected = {
        "schema",
        "core_globs",
        "controlled_globs",
        "forbidden_core_literals",
        "max_controlled_file_bytes",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError(f"framework boundary policy has an unexpected shape: {resolved}")
    if payload.get("schema") != FRAMEWORK_BOUNDARY_SCHEMA_ID:
        raise LabError(f"unsupported framework boundary policy schema: {resolved}")
    core_globs = tuple(
        _safe_glob(value)
        for value in _string_list(payload.get("core_globs"), label="core globs")
    )
    controlled_globs = tuple(
        _safe_glob(value)
        for value in _string_list(
            payload.get("controlled_globs"), label="controlled globs"
        )
    )
    forbidden = _string_list(
        payload.get("forbidden_core_literals"),
        label="forbidden core literals",
    )
    if len({value.casefold() for value in forbidden}) != len(forbidden):
        raise LabError("forbidden core literals contain case-insensitive duplicates")
    maximum = payload.get("max_controlled_file_bytes")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 64 * 1024 * 1024:
        raise LabError("max controlled file bytes must be between 1 and 67108864")
    return FrameworkBoundaryPolicy(
        core_globs=core_globs,
        controlled_globs=controlled_globs,
        forbidden_core_literals=forbidden,
        max_controlled_file_bytes=maximum,
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _collect_matches(
    repository: Path,
    patterns: tuple[str, ...],
    *,
    violations: list[str],
) -> set[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        pattern_matches = sorted(repository.glob(pattern))
        if not pattern_matches:
            violations.append(f"framework boundary glob matched no files: {pattern}")
            continue
        for candidate in pattern_matches:
            if candidate.is_symlink():
                violations.append(
                    f"framework-controlled path must not be a symlink: "
                    f"{candidate.relative_to(repository)}"
                )
                continue
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(repository):
                violations.append(f"framework-controlled path escapes repository: {candidate}")
                continue
            matches.add(resolved)
    return matches


def inspect_framework_boundary(
    repository: Path,
    *,
    policy: FrameworkBoundaryPolicy,
) -> dict[str, object]:
    root = repository.expanduser().resolve()
    if not root.is_dir():
        raise LabError(f"framework boundary repository is not a directory: {root}")
    if not policy.path.is_relative_to(root):
        raise LabError("framework boundary policy must be inside the repository")

    violations: list[str] = []
    core_files = _collect_matches(root, policy.core_globs, violations=violations)
    controlled_files = core_files | _collect_matches(
        root, policy.controlled_globs, violations=violations
    )
    folded_literals = tuple(
        (literal, literal.casefold()) for literal in policy.forbidden_core_literals
    )
    for path in sorted(core_files):
        relative = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            violations.append(f"cannot read framework core source {relative}: {exc}")
            continue
        folded_text = text.casefold()
        for literal, folded_literal in folded_literals:
            if folded_literal in folded_text:
                violations.append(
                    f"project-specific literal {literal!r} found in framework core: {relative}"
                )

    largest_path: str | None = None
    largest_size = 0
    for path in sorted(controlled_files):
        try:
            size = path.stat().st_size
        except OSError as exc:
            violations.append(f"cannot stat framework-controlled file {path}: {exc}")
            continue
        if size > largest_size:
            largest_path = str(path.relative_to(root))
            largest_size = size
        if size > policy.max_controlled_file_bytes:
            violations.append(
                f"framework-controlled file exceeds {policy.max_controlled_file_bytes} bytes: "
                f"{path.relative_to(root)} ({size})"
            )

    return {
        "schema": "lab.framework-boundary-report.v1",
        "status": "PASSED" if not violations else "FAILED",
        "repository": str(root),
        "policy": {
            "path": str(policy.path),
            "sha256": policy.sha256,
        },
        "core_file_count": len(core_files),
        "controlled_file_count": len(controlled_files),
        "largest_controlled_file": {
            "path": largest_path,
            "size": largest_size,
        },
        "violations": sorted(set(violations)),
    }


def enforce_framework_boundary(
    repository: Path,
    *,
    policy: FrameworkBoundaryPolicy,
) -> dict[str, object]:
    report = inspect_framework_boundary(repository, policy=policy)
    violations = report["violations"]
    if isinstance(violations, list) and violations:
        raise LabError("framework boundary failed: " + "; ".join(violations))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lab_control.framework_boundary",
        description="validate project-independent framework source and file limits",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        policy = load_framework_boundary_policy(arguments.policy)
        report = inspect_framework_boundary(arguments.root, policy=policy)
    except LabError as exc:
        parser.error(str(exc))
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Status: {report['status']}")
        print(f"Core files: {report['core_file_count']}")
        print(f"Controlled files: {report['controlled_file_count']}")
        raw_violations = report["violations"]
        if isinstance(raw_violations, list):
            for violation in raw_violations:
                print(f"Violation: {violation}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
