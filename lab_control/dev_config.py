from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import LabError, validate_task_id

DEV_CONFIG_SCHEMA_ID = "lab.dev-config.v1"
MODULE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
PYTHON_VERSION_PATTERN = re.compile(r"([0-9]+)\.([0-9]+)")


@dataclass(frozen=True)
class DevConfig:
    python_major_minor: tuple[int, int]
    import_modules: tuple[str, ...]
    test_targets: dict[str, tuple[str, ...]]
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


def _validate_test_path(value: str) -> str:
    if value.startswith(("/", "~")) or "\\" in value or "\x00" in value:
        raise LabError(f"unsafe dev test path: {value!r}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(f"unsafe dev test path: {value!r}")
    return value


def load_dev_config(path: Path) -> DevConfig:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read dev config {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "python_major_minor",
        "import_modules",
        "test_targets",
    }:
        raise LabError(f"dev config has an unexpected shape: {resolved}")
    if payload.get("schema") != DEV_CONFIG_SCHEMA_ID:
        raise LabError(f"unsupported dev config schema: {resolved}")

    raw_python = payload.get("python_major_minor")
    match = (
        PYTHON_VERSION_PATTERN.fullmatch(raw_python)
        if isinstance(raw_python, str)
        else None
    )
    if match is None:
        raise LabError(f"dev config has invalid Python requirement: {resolved}")
    python_major_minor = (int(match.group(1)), int(match.group(2)))

    import_modules = _string_list(
        payload.get("import_modules"), label="dev import modules"
    )
    for module_name in import_modules:
        if not MODULE_PATTERN.fullmatch(module_name):
            raise LabError(f"invalid dev import module: {module_name!r}")

    raw_targets = payload.get("test_targets")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise LabError(f"dev config has no test targets: {resolved}")
    test_targets: dict[str, tuple[str, ...]] = {}
    for target_id, raw_paths in raw_targets.items():
        validate_task_id(target_id)
        paths = _string_list(raw_paths, label=f"dev test target {target_id}")
        test_targets[target_id] = tuple(_validate_test_path(value) for value in paths)

    return DevConfig(
        python_major_minor=python_major_minor,
        import_modules=import_modules,
        test_targets=dict(sorted(test_targets.items())),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
