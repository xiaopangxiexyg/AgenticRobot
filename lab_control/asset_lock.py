from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .workspace import LabError, validate_task_id

ASSET_LOCK_SCHEMA_ID = "lab.asset-lock.v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class AssetPin:
    role: str
    asset_id: str
    sha256: str


@dataclass(frozen=True)
class AssetLock:
    lock_id: str
    workload_id: str | None
    assets: Mapping[str, AssetPin]
    unresolved: tuple[str, ...]
    path: Path
    sha256: str


def load_asset_lock(path: Path) -> AssetLock:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read remote asset lock {resolved}: {exc}") from exc
    expected_keys = {"schema", "lock_id", "workload_id", "assets", "unresolved"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise LabError(f"remote asset lock has an unexpected shape: {resolved}")
    if payload.get("schema") != ASSET_LOCK_SCHEMA_ID:
        raise LabError(f"unsupported remote asset lock: {resolved}")
    lock_id = payload.get("lock_id")
    if not isinstance(lock_id, str):
        raise LabError(f"remote asset lock has invalid ID: {resolved}")
    validate_task_id(lock_id)
    workload_id = payload.get("workload_id")
    if workload_id is not None:
        if not isinstance(workload_id, str):
            raise LabError(f"remote asset lock has invalid workload ID: {resolved}")
        validate_task_id(workload_id)
    raw_assets = payload.get("assets")
    raw_unresolved = payload.get("unresolved")
    if not isinstance(raw_assets, dict) or not isinstance(raw_unresolved, list):
        raise LabError(f"remote asset lock has invalid assets: {resolved}")
    if not all(isinstance(role, str) for role in raw_unresolved):
        raise LabError(f"remote asset lock has invalid unresolved roles: {resolved}")
    unresolved = tuple(raw_unresolved)
    if len(unresolved) != len(set(unresolved)):
        raise LabError(f"remote asset lock repeats unresolved roles: {resolved}")
    for role in unresolved:
        validate_task_id(role)
    assets: dict[str, AssetPin] = {}
    for role, identity in raw_assets.items():
        validate_task_id(role)
        if not isinstance(identity, dict) or set(identity) != {"asset_id", "sha256"}:
            raise LabError(f"remote asset lock has invalid pin: {role}")
        asset_id = identity.get("asset_id")
        sha256 = identity.get("sha256")
        if not isinstance(asset_id, str):
            raise LabError(f"remote asset lock has invalid asset ID: {role}")
        validate_task_id(asset_id)
        if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
            raise LabError(f"remote asset lock has invalid SHA-256: {role}")
        assets[role] = AssetPin(role=role, asset_id=asset_id, sha256=sha256)
    overlap = set(assets) & set(unresolved)
    if overlap:
        raise LabError(
            f"remote asset roles are both pinned and unresolved: {sorted(overlap)}"
        )
    return AssetLock(
        lock_id=lock_id,
        workload_id=workload_id,
        assets=dict(sorted(assets.items())),
        unresolved=tuple(sorted(unresolved)),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def resolve_asset_identities(
    lock: AssetLock,
    roles: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    if not roles or len(roles) != len(set(roles)):
        raise LabError("remote Profile has invalid required asset roles")
    unresolved = [role for role in roles if role in lock.unresolved]
    missing = [
        role for role in roles if role not in lock.assets and role not in unresolved
    ]
    if unresolved or missing:
        details: list[str] = []
        if unresolved:
            details.append(f"unresolved={','.join(unresolved)}")
        if missing:
            details.append(f"missing={','.join(missing)}")
        raise LabError(
            f"remote asset lock {lock.lock_id} cannot satisfy Profile: "
            + "; ".join(details)
        )
    return {
        role: {
            "asset_id": lock.assets[role].asset_id,
            "sha256": lock.assets[role].sha256,
        }
        for role in roles
    }
