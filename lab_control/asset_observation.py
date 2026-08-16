from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from .remote_request import OID_PATTERN, SHA256_PATTERN
from .workspace import LabError, validate_task_id

ASSET_OBSERVATION_SCHEMA_ID = "lab.asset-observations.v2"
_REQUIRED_BLOCKERS = {"content-manifest", "explicit-selection"}


@dataclass(frozen=True)
class ObservedAssetPin:
    role: str
    asset_id: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class AssetCandidateObservation:
    candidate_id: str
    content_role: str
    content_relative_path: str
    companion_assets: dict[str, ObservedAssetPin]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class AssetObservationSet:
    observation_id: str
    host_id: str
    canonical_identity: str
    commit: str
    tree_oid: str
    registry_role: str
    registry_relative_path: str
    registry_sha256: str
    candidates: dict[str, AssetCandidateObservation]
    observed_at: str
    path: Path
    sha256: str


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise LabError(f"asset observation has invalid {label}")
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "~"))
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise LabError(f"asset observation has unsafe {label}")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise LabError(f"asset observation has invalid {label} SHA-256")
    return value


def _timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise LabError("asset observation has invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("asset observation has invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise LabError("asset observation timestamp requires a timezone")
    return value


def _pin(value: object, *, role: str) -> ObservedAssetPin:
    if not isinstance(value, dict) or set(value) != {
        "asset_id",
        "relative_path",
        "sha256",
    }:
        raise LabError(f"asset observation has invalid {role} pin")
    asset_id = value.get("asset_id")
    if not isinstance(asset_id, str):
        raise LabError(f"asset observation has invalid {role} asset ID")
    validate_task_id(role)
    validate_task_id(asset_id)
    return ObservedAssetPin(
        role=role,
        asset_id=asset_id,
        relative_path=_relative_path(value.get("relative_path"), label=f"{role} path"),
        sha256=_sha256(value.get("sha256"), label=role),
    )


def load_asset_observations(path: Path) -> AssetObservationSet:
    """Load non-authorizing observations with project-defined logical roles."""

    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read asset observations {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "observation_id",
        "host_id",
        "repository",
        "registry",
        "candidates",
        "observed_at",
    }:
        raise LabError(f"asset observations have an unexpected shape: {resolved}")
    if payload.get("schema") != ASSET_OBSERVATION_SCHEMA_ID:
        raise LabError(f"unsupported asset observations: {resolved}")
    observation_id = payload.get("observation_id")
    host_id = payload.get("host_id")
    if not isinstance(observation_id, str) or not isinstance(host_id, str):
        raise LabError("asset observations have invalid identity")
    validate_task_id(observation_id)
    validate_task_id(host_id)

    repository = payload.get("repository")
    if not isinstance(repository, dict) or set(repository) != {
        "canonical_identity",
        "commit",
        "tree_oid",
    }:
        raise LabError("asset observations have invalid repository identity")
    canonical_identity = repository.get("canonical_identity")
    commit = repository.get("commit")
    tree_oid = repository.get("tree_oid")
    if (
        not isinstance(canonical_identity, str)
        or canonical_identity.count("/") != 1
        or not OID_PATTERN.fullmatch(str(commit))
        or not OID_PATTERN.fullmatch(str(tree_oid))
    ):
        raise LabError("asset observations have invalid repository identity")

    registry = payload.get("registry")
    if not isinstance(registry, dict) or set(registry) != {
        "asset_role",
        "relative_path",
        "sha256",
    }:
        raise LabError("asset observations have invalid registry identity")
    registry_role = registry.get("asset_role")
    if not isinstance(registry_role, str):
        raise LabError("asset observations have invalid registry role")
    validate_task_id(registry_role)
    registry_path = _relative_path(registry.get("relative_path"), label="registry path")
    registry_sha = _sha256(registry.get("sha256"), label="registry")

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise LabError("asset observations require at least one candidate")
    candidates: dict[str, AssetCandidateObservation] = {}
    for candidate_id, raw_candidate in raw_candidates.items():
        if not isinstance(candidate_id, str):
            raise LabError("asset observations have invalid candidate ID")
        validate_task_id(candidate_id)
        if not isinstance(raw_candidate, dict) or set(raw_candidate) != {
            "candidate_id",
            "content",
            "companion_assets",
            "selection_authorized",
            "activation_ready",
            "blockers",
        }:
            raise LabError(f"asset observation candidate is invalid: {candidate_id}")
        if raw_candidate.get("candidate_id") != candidate_id:
            raise LabError(f"asset observation candidate identity mismatch: {candidate_id}")
        content = raw_candidate.get("content")
        if not isinstance(content, dict) or set(content) != {
            "asset_role",
            "relative_path",
            "exists",
            "content_manifest_sha256",
        }:
            raise LabError(f"asset observation content is invalid: {candidate_id}")
        content_role = content.get("asset_role")
        if not isinstance(content_role, str):
            raise LabError(f"asset observation content role is invalid: {candidate_id}")
        validate_task_id(content_role)
        content_path = _relative_path(
            content.get("relative_path"), label="content path"
        )
        if content.get("exists") is not True or content.get("content_manifest_sha256") is not None:
            raise LabError(
                f"asset observation cannot claim content identity: {candidate_id}"
            )
        raw_companions = raw_candidate.get("companion_assets")
        if not isinstance(raw_companions, dict) or not raw_companions:
            raise LabError(f"asset observation companions are invalid: {candidate_id}")
        companions = {
            role: _pin(pin, role=role)
            for role, pin in raw_companions.items()
            if isinstance(role, str)
        }
        if len(companions) != len(raw_companions) or content_role in companions:
            raise LabError(f"asset observation roles are invalid: {candidate_id}")
        blockers = raw_candidate.get("blockers")
        registry_pin = companions.get(registry_role)
        if (
            raw_candidate.get("selection_authorized") is not False
            or raw_candidate.get("activation_ready") is not False
            or not isinstance(blockers, list)
            or not all(isinstance(blocker, str) for blocker in blockers)
            or set(blockers) != _REQUIRED_BLOCKERS
            or len(blockers) != len(_REQUIRED_BLOCKERS)
            or registry_pin is None
            or registry_pin.relative_path != registry_path
            or registry_pin.sha256 != registry_sha
        ):
            raise LabError(
                f"asset observation candidate has an unsafe activation envelope: {candidate_id}"
            )
        candidates[candidate_id] = AssetCandidateObservation(
            candidate_id=candidate_id,
            content_role=content_role,
            content_relative_path=content_path,
            companion_assets=dict(sorted(companions.items())),
            blockers=tuple(sorted(blockers)),
        )
    return AssetObservationSet(
        observation_id=observation_id,
        host_id=host_id,
        canonical_identity=str(canonical_identity),
        commit=str(commit),
        tree_oid=str(tree_oid),
        registry_role=registry_role,
        registry_relative_path=registry_path,
        registry_sha256=registry_sha,
        candidates=dict(sorted(candidates.items())),
        observed_at=_timestamp(payload.get("observed_at")),
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
