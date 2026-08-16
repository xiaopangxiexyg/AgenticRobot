from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .remote_request import RemoteProfile
from .workspace import LabError, validate_task_id

COMMAND_REGISTRY_SCHEMA_ID = "lab.command-registry.v1"


@dataclass(frozen=True)
class TrustedCommandSpec:
    command_id: str
    adapter_id: str
    service_required: bool
    device_access: bool
    network_egress: str
    downloads_permitted: bool


@dataclass(frozen=True)
class TrustedCommandRegistry:
    registry_id: str
    commands: dict[str, TrustedCommandSpec]
    path: Path
    sha256: str


def load_command_registry(path: Path) -> TrustedCommandRegistry:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read command registry {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "registry_id",
        "commands",
    }:
        raise LabError(f"command registry has an unexpected shape: {resolved}")
    if payload.get("schema") != COMMAND_REGISTRY_SCHEMA_ID:
        raise LabError(f"unsupported command registry: {resolved}")
    registry_id = payload.get("registry_id")
    if not isinstance(registry_id, str):
        raise LabError(f"command registry has invalid ID: {resolved}")
    validate_task_id(registry_id)
    raw_commands = payload.get("commands")
    if not isinstance(raw_commands, dict) or not raw_commands:
        raise LabError(f"command registry has no commands: {resolved}")
    commands: dict[str, TrustedCommandSpec] = {}
    adapter_ids: set[str] = set()
    expected_fields = {
        "adapter_id",
        "service_required",
        "device_access",
        "network_egress",
        "downloads_permitted",
    }
    for command_id, raw_spec in raw_commands.items():
        validate_task_id(command_id)
        if not isinstance(raw_spec, dict) or set(raw_spec) != expected_fields:
            raise LabError(f"command registry has invalid command: {command_id}")
        adapter_id = raw_spec.get("adapter_id")
        if not isinstance(adapter_id, str):
            raise LabError(f"command registry has invalid adapter: {command_id}")
        validate_task_id(adapter_id)
        if adapter_id in adapter_ids:
            raise LabError(f"command registry repeats adapter ID: {adapter_id}")
        adapter_ids.add(adapter_id)
        if (
            raw_spec.get("service_required") is not True
            or raw_spec.get("device_access") is not False
            or raw_spec.get("network_egress") != "loopback-only"
            or raw_spec.get("downloads_permitted") is not False
        ):
            raise LabError(f"command registry has unsafe policy: {command_id}")
        commands[command_id] = TrustedCommandSpec(
            command_id=command_id,
            adapter_id=adapter_id,
            service_required=True,
            device_access=False,
            network_egress="loopback-only",
            downloads_permitted=False,
        )
    return TrustedCommandRegistry(
        registry_id=registry_id,
        commands=dict(sorted(commands.items())),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def resolve_trusted_command(
    registry: TrustedCommandRegistry,
    *,
    profile: RemoteProfile,
    command_id: str,
) -> TrustedCommandSpec:
    validate_task_id(command_id)
    if command_id not in profile.command_ids:
        raise LabError(f"command is not authorized by remote Profile: {command_id}")
    spec = registry.commands.get(command_id)
    if spec is None:
        raise LabError(f"command is not present in protected registry: {command_id}")
    if (
        spec.device_access != profile.device_access
        or spec.network_egress != profile.network_egress
        or spec.downloads_permitted != profile.downloads_permitted
    ):
        raise LabError(f"command registry policy differs from Profile: {command_id}")
    return spec
