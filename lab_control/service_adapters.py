from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .service_launch_plan import TrustedServiceAdapter, _validate_adapter
from .workspace import LabError, validate_task_id

SERVICE_ADAPTER_SCHEMA_ID = "lab.service-adapter.v1"
_MAX_MANIFEST_BYTES = 256 * 1024


def _safe_adapter_path(root: Path, launch_command_id: str) -> Path:
    validate_task_id(launch_command_id)
    resolved_root = root.expanduser().resolve()
    path = (resolved_root / f"{launch_command_id}.json").resolve()
    if path.parent != resolved_root or not path.is_file() or path.is_symlink():
        raise LabError(
            f"protected control plane has no service adapter: {launch_command_id}"
        )
    return path


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LabError(f"service adapter has invalid {label}")
    return tuple(value)


def load_trusted_service_adapter(
    adapter_root: Path,
    launch_command_id: str,
) -> TrustedServiceAdapter:
    """Load one strictly shaped adapter from an installed protected package.

    The caller supplies the trusted package root. A Task can select only the
    logical launch command ID already authorized by its protected Profile; it
    cannot provide this path, raw argv, projections, or environment entries.
    """

    path = _safe_adapter_path(adapter_root, launch_command_id)
    try:
        raw = path.read_bytes()
        if not 1 <= len(raw) <= _MAX_MANIFEST_BYTES:
            raise LabError("service adapter manifest has an invalid size")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read service adapter {path}: {exc}") from exc
    expected = {
        "schema",
        "adapter_id",
        "launch_command_id",
        "argv_template",
        "working_directory",
        "environment",
        "supported_workload_ids",
        "asset_projections",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError(f"service adapter has an unexpected shape: {path}")
    if (
        payload.get("schema") != SERVICE_ADAPTER_SCHEMA_ID
        or payload.get("launch_command_id") != launch_command_id
    ):
        raise LabError(f"service adapter identity mismatch: {path}")
    adapter_id = payload.get("adapter_id")
    working_directory = payload.get("working_directory")
    raw_environment = payload.get("environment")
    raw_projections = payload.get("asset_projections")
    if (
        not isinstance(adapter_id, str)
        or not isinstance(working_directory, str)
        or not isinstance(raw_environment, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in raw_environment.items())
        or not isinstance(raw_projections, list)
    ):
        raise LabError(f"service adapter has invalid fields: {path}")
    projections: list[tuple[str, str, str]] = []
    for projection in raw_projections:
        if not isinstance(projection, dict) or set(projection) != {
            "source_role",
            "destination_role",
            "relative_target",
        }:
            raise LabError(f"service adapter has an invalid asset projection: {path}")
        values = (
            projection.get("source_role"),
            projection.get("destination_role"),
            projection.get("relative_target"),
        )
        if not all(isinstance(value, str) for value in values):
            raise LabError(f"service adapter has an invalid asset projection: {path}")
        relative = PurePosixPath(str(values[2]))
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise LabError(f"service adapter has an unsafe asset projection: {path}")
        projections.append((str(values[0]), str(values[1]), str(values[2])))
    adapter = TrustedServiceAdapter(
        adapter_id=adapter_id,
        launch_command_id=launch_command_id,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        argv_template=_string_list(payload.get("argv_template"), label="argv template"),
        working_directory=working_directory,
        environment=tuple(sorted((str(key), str(value)) for key, value in raw_environment.items())),
        supported_workload_ids=_string_list(
            payload.get("supported_workload_ids"), label="supported workload IDs"
        ),
        asset_projections=tuple(projections),
    )
    _validate_adapter(adapter)
    return adapter
