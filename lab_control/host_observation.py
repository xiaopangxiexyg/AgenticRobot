from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .asset_selection import validate_runtime_asset_verification
from .gpu_lease import validate_gpu_lease_receipt
from .remote_preflight import (
    HOST_OBSERVATION_SCHEMA_ID,
    EnvironmentPlacement,
    HostProfile,
    RepositoryPlacement,
)
from .remote_request import SHA256_PATTERN
from .workspace import LabError, validate_task_id

RUNTIME_RECEIPTS_SCHEMA_ID = "lab.host-runtime-receipts.v1"

_ENVIRONMENT_PROBE = """
import importlib.util
import json
import pathlib
import sys

origins = {}
for module in sys.argv[1:]:
    spec = importlib.util.find_spec(module)
    origin = spec.origin if spec is not None else None
    if origin is None and spec is not None and spec.submodule_search_locations:
        origin = next(iter(spec.submodule_search_locations), None)
    origins[module] = str(pathlib.Path(origin).resolve()) if origin else None
print(json.dumps({
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "import_origins": origins,
}, sort_keys=True))
""".strip()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _validate_runtime_gpu(
    gpu: object,
    *,
    host_profile: HostProfile,
) -> dict[str, object]:
    expected = {
        "reserved_count",
        "lease_id",
        "gpu_id",
        "gpu_uuid",
        "lease_receipt_sha256",
        "lease_receipt",
    }
    if not isinstance(gpu, dict) or set(gpu) != expected:
        raise LabError("host runtime receipts have an invalid GPU observation")
    gpu_uuid = gpu.get("gpu_uuid")
    if gpu.get("reserved_count") != 1 or not isinstance(gpu_uuid, str):
        raise LabError("host runtime receipts require exactly one GPU lease")
    lease = validate_gpu_lease_receipt(
        gpu.get("lease_receipt"),
        host_id=host_profile.host_id,
        gpu_uuid=gpu_uuid,
        require_active=True,
        host_profile_sha256=host_profile.sha256,
    )
    if (
        gpu.get("lease_id") != lease.get("lease_id")
        or gpu.get("gpu_id") != lease.get("gpu_id")
        or gpu.get("lease_receipt_sha256") != lease.get("receipt_sha256")
    ):
        raise LabError("host runtime GPU observation does not match its lease")
    return gpu


def _validate_runtime_assets(
    assets: object,
    *,
    host_profile: HostProfile,
) -> dict[str, object]:
    if not isinstance(assets, dict):
        raise LabError("host runtime receipts assets must be an object")
    validated: dict[str, object] = {}
    for role, raw_asset in assets.items():
        if not isinstance(role, str):
            raise LabError("host runtime asset role must be a string")
        validate_task_id(role)
        if not isinstance(raw_asset, dict) or set(raw_asset) != {
            "asset_id",
            "sha256",
            "path",
        }:
            raise LabError(f"host runtime asset receipt is invalid: {role}")
        asset_id = raw_asset.get("asset_id")
        sha256 = raw_asset.get("sha256")
        raw_path = raw_asset.get("path")
        if not isinstance(asset_id, str):
            raise LabError(f"host runtime asset ID is invalid: {role}")
        validate_task_id(asset_id)
        if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
            raise LabError(f"host runtime asset SHA-256 is invalid: {role}")
        if not isinstance(raw_path, str):
            raise LabError(f"host runtime asset path is invalid: {role}")
        path = Path(raw_path)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path.is_symlink()
            or path.resolve() != path
            or not any(path == root or root in path.parents for root in host_profile.asset_roots)
        ):
            raise LabError(f"host runtime asset path is unsafe: {role}")
        validated[role] = raw_asset
    return validated


def build_runtime_receipts(
    host_profile: HostProfile,
    *,
    gpu: object,
    assets: object,
    asset_verification: object | None = None,
) -> dict[str, object]:
    """Build a hashed, non-authorizing host runtime receipt.

    Asset hashes must already be bound to a content-rehashed verification receipt
    supplied by the protected asset resolver. This function deliberately does not
    read large artifact contents a second time.
    """

    validated_gpu = _validate_runtime_gpu(gpu, host_profile=host_profile)
    validated_assets = _validate_runtime_assets(assets, host_profile=host_profile)
    validated_asset_verification: dict[str, object] | None = None
    if validated_assets:
        validated_asset_verification = validate_runtime_asset_verification(
            asset_verification
        )
        if validated_asset_verification.get("assets") != validated_assets:
            raise LabError(
                "runtime assets do not match their content verification receipt"
            )
    elif asset_verification is not None:
        raise LabError("empty runtime assets cannot carry a verification receipt")
    content: dict[str, object] = {
        "schema": RUNTIME_RECEIPTS_SCHEMA_ID,
        "host_id": host_profile.host_id,
        "host_profile_sha256": host_profile.sha256,
        "capabilities": list(host_profile.capabilities),
        "device_groups": [],
        "gpu": validated_gpu,
        "assets": validated_assets,
        "asset_verification": validated_asset_verification,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "execution_eligible": False,
        "device_access": False,
        "network_used": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def validate_runtime_receipts(
    payload: object,
    *,
    host_profile: HostProfile,
) -> dict[str, object]:
    expected = {
        "schema",
        "host_id",
        "host_profile_sha256",
        "capabilities",
        "device_groups",
        "gpu",
        "assets",
        "asset_verification",
        "issued_at",
        "execution_eligible",
        "device_access",
        "network_used",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError("host runtime receipts have an unexpected shape")
    capabilities = payload.get("capabilities")
    if (
        payload.get("schema") != RUNTIME_RECEIPTS_SCHEMA_ID
        or payload.get("host_id") != host_profile.host_id
        or payload.get("host_profile_sha256") != host_profile.sha256
        or capabilities != list(host_profile.capabilities)
        or payload.get("device_groups") != []
        or payload.get("execution_eligible") is not False
        or payload.get("device_access") is not False
        or payload.get("network_used") is not False
    ):
        raise LabError("host runtime receipts identity or authority is invalid")
    issued_at = payload.get("issued_at")
    if not isinstance(issued_at, str):
        raise LabError("host runtime receipts have an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("host runtime receipts have an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise LabError("host runtime receipts timestamp requires a timezone")
    _validate_runtime_gpu(payload.get("gpu"), host_profile=host_profile)
    assets = _validate_runtime_assets(
        payload.get("assets"),
        host_profile=host_profile,
    )
    raw_asset_verification = payload.get("asset_verification")
    if assets:
        asset_verification = validate_runtime_asset_verification(
            raw_asset_verification
        )
        if asset_verification.get("assets") != assets:
            raise LabError(
                "runtime assets do not match their content verification receipt"
            )
    elif raw_asset_verification is not None:
        raise LabError("empty runtime assets cannot carry a verification receipt")
    content = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != hashlib.sha256(
        _canonical_json(content)
    ).hexdigest():
        raise LabError("host runtime receipts hash mismatch")
    return payload


def _empty_repository_observation(placement: RepositoryPlacement) -> dict[str, object]:
    path = placement.path
    return {
        "canonical_identity": None,
        "path": str(path),
        "realpath": str(path.resolve()),
        "is_symlink": path.is_symlink(),
        "exists": path.exists(),
        "dirty": None,
        "commit": None,
        "tree_oid": None,
        "submodules": None,
        "submodules_initialized": None,
        "published_ref_contains": None,
    }


def collect_repository_observation(
    placement: RepositoryPlacement,
) -> dict[str, object]:
    """Read local Git facts without fetch, checkout, hooks, or other writes."""

    observation = _empty_repository_observation(placement)
    path = placement.path
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        return observation
    try:
        root = git.repository_root(path)
        if root != path:
            return {**observation, "observation_error": "repository root mismatch"}
        commit = git.current_commit(path)
        submodules, submodules_initialized = git.submodule_status(path)
        containing_refs = git.run_git(
            path,
            [
                "for-each-ref",
                f"--contains={commit}",
                "--format=%(refname)",
                "refs/remotes",
            ],
        ).stdout.splitlines()
        return {
            **observation,
            "canonical_identity": git.origin_identity(path),
            "dirty": bool(git.status_short(path)),
            "commit": commit,
            "tree_oid": git.tree_oid(path, commit),
            "submodules": submodules,
            "submodules_initialized": submodules_initialized,
            "published_ref_contains": any(
                reference and not reference.endswith("/HEAD")
                for reference in containing_refs
            ),
        }
    except git.GitError as exc:
        return {**observation, "observation_error": str(exc)}


def collect_environment_observation(
    placement: EnvironmentPlacement,
) -> dict[str, object]:
    """Probe Python identity/import origins using the configured interpreter."""

    path = placement.path
    python = path / "bin" / "python"
    observation: dict[str, object] = {
        "path": str(path),
        "realpath": str(path.resolve()),
        "is_symlink": path.is_symlink(),
        "exists": path.is_dir() and python.is_file(),
        "python_executable": str(python),
        "python_major_minor": None,
        "import_origins": {},
    }
    if not observation["exists"] or path.is_symlink() or path.resolve() != path:
        return observation
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                _ENVIRONMENT_PROBE,
                *placement.import_modules,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**observation, "observation_error": str(exc)}
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {**observation, "observation_error": detail}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {**observation, "observation_error": str(exc)}
    if not isinstance(payload, dict):
        return {**observation, "observation_error": "probe returned a non-object"}
    return {
        **observation,
        "python_major_minor": payload.get("python_major_minor"),
        "import_origins": payload.get("import_origins"),
    }


def load_runtime_receipts(
    path: Path,
    *,
    host_profile: HostProfile,
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read host runtime receipts {path}: {exc}") from exc
    return validate_runtime_receipts(payload, host_profile=host_profile)


def collect_host_observation(
    host_profile: HostProfile,
    runtime_receipts: Mapping[str, object],
) -> dict[str, object]:
    """Combine live read-only facts with separately issued runtime receipts."""

    validated_runtime = validate_runtime_receipts(
        dict(runtime_receipts),
        host_profile=host_profile,
    )
    raw_assets = validated_runtime.get("assets")
    if not isinstance(raw_assets, Mapping):
        raise LabError("runtime receipts assets must be an object")
    assets: dict[str, object] = {}
    for role, raw_asset in raw_assets.items():
        if not isinstance(role, str) or not isinstance(raw_asset, Mapping):
            raise LabError("invalid runtime asset receipt")
        raw_path = raw_asset.get("path")
        if not isinstance(raw_path, str):
            raise LabError(f"runtime asset receipt lacks path: {role}")
        path = Path(raw_path)
        if not path.is_absolute() or ".." in path.parts:
            raise LabError(f"runtime asset path is unsafe: {role}")
        assets[role] = {
            "asset_id": raw_asset.get("asset_id"),
            "sha256": raw_asset.get("sha256"),
            "exists": path.exists(),
            "readable": os.access(path, os.R_OK),
            "realpath": str(path.resolve()),
        }
    asset_verification = validated_runtime.get("asset_verification")
    return {
        "schema": HOST_OBSERVATION_SCHEMA_ID,
        "host_id": host_profile.host_id,
        "hostname": socket.gethostname(),
        "capabilities": validated_runtime.get("capabilities"),
        "device_groups": validated_runtime.get("device_groups"),
        "gpu": validated_runtime.get("gpu"),
        "repositories": {
            name: collect_repository_observation(placement)
            for name, placement in host_profile.repositories.items()
        },
        "environments": {
            environment_id: collect_environment_observation(placement)
            for environment_id, placement in host_profile.environments.items()
        },
        "assets": assets,
        "asset_verification_receipt_sha256": (
            asset_verification.get("receipt_sha256")
            if isinstance(asset_verification, Mapping)
            else None
        ),
    }
