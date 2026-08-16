from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .clean_slot import _verify_live, validate_clean_slot_receipt
from .remote_preflight import HostProfile, preflight_remote_request
from .remote_request import RemoteProfile, _canonical_json, validate_remote_request
from .workspace import LabError, validate_task_id

SLOT_READINESS_SCHEMA_ID = "lab.slot-readiness.v1"

_STAGED_IMPORT_PROBE = """
import importlib.util
import json
import pathlib
import sys

roots = json.loads(sys.argv[1])
sys.path[:0] = roots
origins = {}
for module in sys.argv[2:]:
    spec = importlib.util.find_spec(module)
    origin = spec.origin if spec is not None else None
    if origin is None and spec is not None and spec.submodule_search_locations:
        origin = next(iter(spec.submodule_search_locations), None)
    origins[module] = str(pathlib.Path(origin).resolve()) if origin else None
print(json.dumps({
    "python_executable": str(pathlib.Path(sys.executable).resolve()),
    "python_prefix": str(pathlib.Path(sys.prefix).resolve()),
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "import_origins": origins,
}, sort_keys=True))
""".strip()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def validate_slot_readiness_receipt(
    receipt: object,
    *,
    request_id: str,
    idempotency_key: str,
    slot_id: str,
) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise LabError("slot readiness receipt must be an object")
    expected_keys = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "snapshot_id",
        "slot_id",
        "host_profile_sha256",
        "remote_profile_sha256",
        "control_plane_version",
        "preflight_receipt_sha256",
        "source_receipt_sha256",
        "observation_sha256",
        "environments",
        "assets",
        "gpu_lease_id",
        "gpu_lease_receipt_sha256",
        "gpu_uuid",
        "source_prepared",
        "environment_prepared",
        "assets_prepared",
        "service_started",
        "execution_eligible",
        "device_access",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("slot readiness receipt has an unexpected shape")
    if (
        receipt.get("schema") != SLOT_READINESS_SCHEMA_ID
        or receipt.get("request_id") != request_id
        or receipt.get("idempotency_key") != idempotency_key
        or receipt.get("slot_id") != slot_id
    ):
        raise LabError("slot readiness receipt identity mismatch")
    for field in (
        "host_profile_sha256",
        "remote_profile_sha256",
        "preflight_receipt_sha256",
        "source_receipt_sha256",
        "observation_sha256",
        "gpu_lease_receipt_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LabError(f"slot readiness receipt has invalid {field}")
    if not isinstance(receipt.get("environments"), dict) or not isinstance(
        receipt.get("assets"), dict
    ):
        raise LabError("slot readiness receipt lacks environment/assets objects")
    environment_keys = {
        "environment_id",
        "path",
        "python_executable",
        "python_prefix",
        "python_major_minor",
        "repository",
        "source_roots",
        "import_origins",
        "first_party_from_staged_source",
    }
    for environment_id, environment in receipt["environments"].items():
        if (
            not isinstance(environment_id, str)
            or not isinstance(environment, dict)
            or set(environment) != environment_keys
            or environment.get("environment_id") != environment_id
            or environment.get("first_party_from_staged_source") is not True
        ):
            raise LabError(
                f"slot readiness environment receipt is invalid: {environment_id}"
            )
    asset_keys = {
        "asset_id",
        "sha256",
        "realpath",
        "identity_attested_by_runtime_receipt",
        "content_rehashed",
    }
    for role, asset in receipt["assets"].items():
        if (
            not isinstance(role, str)
            or not isinstance(asset, dict)
            or set(asset) != asset_keys
            or asset.get("identity_attested_by_runtime_receipt") is not True
            or asset.get("content_rehashed") is not False
        ):
            raise LabError(f"slot readiness asset receipt is invalid: {role}")
    for field in (
        "task_id",
        "snapshot_id",
        "control_plane_version",
        "gpu_lease_id",
        "gpu_uuid",
    ):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise LabError(f"slot readiness receipt has invalid {field}")
    if (
        receipt.get("source_prepared") is not True
        or receipt.get("environment_prepared") is not True
        or receipt.get("assets_prepared") is not True
        or receipt.get("service_started") is not False
        or receipt.get("execution_eligible") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("slot readiness receipt has an unsafe execution envelope")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _sha256(content) != receipt.get("receipt_sha256"):
        raise LabError("slot readiness receipt hash mismatch")
    return receipt


def _probe_environment(
    environment_id: str,
    *,
    host_profile: HostProfile,
    staged_repositories: Mapping[str, Path],
) -> dict[str, object]:
    placement = host_profile.environments.get(environment_id)
    if placement is None:
        raise LabError(f"host Profile lacks environment: {environment_id}")
    environment_root = placement.path
    python = environment_root / "bin" / "python"
    if (
        environment_root.is_symlink()
        or environment_root.resolve() != environment_root
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise LabError(f"environment is missing or unsafe: {environment_id}")
    repository = staged_repositories.get(placement.repository)
    if repository is None:
        raise LabError(f"environment repository was not staged: {environment_id}")
    roots: list[Path] = []
    for relative in placement.source_roots:
        raw_root = repository / relative
        root = raw_root.resolve()
        if (
            raw_root.is_symlink()
            or not root.is_relative_to(repository)
            or not root.is_dir()
        ):
            raise LabError(
                f"staged source root is missing or unsafe: {environment_id}/{relative}"
            )
        roots.append(root)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    try:
        completed = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                _STAGED_IMPORT_PROBE,
                json.dumps([str(root) for root in roots]),
                *placement.import_modules,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(
            f"staged environment probe failed: {environment_id}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"staged environment probe failed: {environment_id}: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LabError(
            f"staged environment probe returned invalid JSON: {environment_id}"
        ) from exc
    if (
        not isinstance(result, dict)
        or result.get("python_major_minor") != placement.python_major_minor
    ):
        raise LabError(f"staged environment Python identity mismatch: {environment_id}")
    origins = result.get("import_origins")
    if not isinstance(origins, dict) or set(origins) != set(placement.import_modules):
        raise LabError(f"staged environment import mapping mismatch: {environment_id}")
    for module, raw_origin in origins.items():
        if not isinstance(raw_origin, str):
            raise LabError(f"staged import is unresolved: {environment_id}/{module}")
        origin = Path(raw_origin)
        if not origin.is_absolute() or not origin.resolve().is_relative_to(repository):
            raise LabError(
                f"staged import escaped source checkout: {environment_id}/{module}"
            )
    return {
        "environment_id": environment_id,
        "path": str(environment_root),
        "python_executable": result.get("python_executable"),
        "python_prefix": result.get("python_prefix"),
        "python_major_minor": result.get("python_major_minor"),
        "repository": placement.repository,
        "source_roots": [str(root) for root in roots],
        "import_origins": origins,
        "first_party_from_staged_source": True,
    }


def prepare_slot_readiness(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    observation: object,
    source_receipt: object,
    control_plane_version: str,
    slot_id: str,
) -> dict[str, object]:
    validate_task_id(slot_id)
    validated_request = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    if not isinstance(observation, Mapping):
        raise LabError("host observation must be an object")
    preflight = preflight_remote_request(
        validated_request,
        remote_profile=remote_profile,
        host_profile=host_profile,
        observation=observation,
        control_plane_version=control_plane_version,
    )
    source = validate_clean_slot_receipt(
        source_receipt,
        request_id=str(validated_request["request_id"]),
        idempotency_key=str(validated_request["idempotency_key"]),
        slot_id=slot_id,
    )
    materialization_root = Path(str(source.get("materialization_root", ""))).resolve()
    _verify_live(
        materialization_root,
        validated_request,
        source,
        slot_id=slot_id,
    )
    raw_source_repositories = source.get("repositories")
    if not isinstance(raw_source_repositories, Mapping):
        raise LabError("source receipt has no repositories")
    staged_repositories = {
        name: Path(str(value.get("path", ""))).resolve()
        for name, value in raw_source_repositories.items()
        if isinstance(name, str) and isinstance(value, Mapping)
    }
    if set(staged_repositories) != set(remote_profile.repositories):
        raise LabError("staged source repository mapping mismatch")
    environments = {
        environment_id: _probe_environment(
            environment_id,
            host_profile=host_profile,
            staged_repositories=staged_repositories,
        )
        for environment_id in remote_profile.required_environments
    }
    raw_observation_assets = observation.get("assets")
    request_assets = validated_request.get("assets")
    if not isinstance(raw_observation_assets, Mapping) or not isinstance(
        request_assets, Mapping
    ):
        raise LabError("readiness requires observation/request assets")
    assets: dict[str, object] = {}
    allowed_asset_roots: list[Path] = []
    for root in host_profile.asset_roots:
        if root.is_symlink() or root.resolve() != root or not root.is_dir():
            raise LabError(f"host asset root is missing or unsafe: {root}")
        allowed_asset_roots.append(root)
    for role in remote_profile.required_assets:
        actual = raw_observation_assets.get(role)
        expected = request_assets.get(role)
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise LabError(f"readiness asset mapping is invalid: {role}")
        realpath = Path(str(actual.get("realpath", "")))
        if (
            not realpath.is_absolute()
            or realpath.resolve() != realpath
            or not realpath.exists()
            or not os.access(realpath, os.R_OK)
            or not any(realpath.is_relative_to(root) for root in allowed_asset_roots)
        ):
            raise LabError(f"readiness asset path is unavailable or unsafe: {role}")
        assets[role] = {
            "asset_id": expected.get("asset_id"),
            "sha256": expected.get("sha256"),
            "realpath": str(realpath),
            "identity_attested_by_runtime_receipt": True,
            "content_rehashed": False,
        }
    gpu = observation.get("gpu")
    if not isinstance(gpu, Mapping):
        raise LabError("readiness observation lacks GPU receipt")
    content: dict[str, object] = {
        "schema": SLOT_READINESS_SCHEMA_ID,
        "request_id": validated_request["request_id"],
        "idempotency_key": validated_request["idempotency_key"],
        "task_id": validated_request["task_id"],
        "snapshot_id": validated_request["snapshot_id"],
        "slot_id": slot_id,
        "host_profile_sha256": host_profile.sha256,
        "remote_profile_sha256": remote_profile.sha256,
        "control_plane_version": control_plane_version,
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "observation_sha256": _sha256(observation),
        "environments": environments,
        "assets": assets,
        "gpu_lease_id": gpu.get("lease_id"),
        "gpu_lease_receipt_sha256": gpu.get("lease_receipt_sha256"),
        "gpu_uuid": gpu.get("gpu_uuid"),
        "source_prepared": True,
        "environment_prepared": True,
        "assets_prepared": True,
        "service_started": False,
        "execution_eligible": False,
        "device_access": False,
    }
    receipt = {**content, "receipt_sha256": _sha256(content)}
    return validate_slot_readiness_receipt(
        receipt,
        request_id=str(validated_request["request_id"]),
        idempotency_key=str(validated_request["idempotency_key"]),
        slot_id=slot_id,
    )
