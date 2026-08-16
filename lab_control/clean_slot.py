from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from . import git
from .remote_preflight import HostProfile
from .remote_request import RemoteProfile, _canonical_json, validate_remote_request
from .slot_materializer import (
    _anchor_fingerprint,
    _clone_detached,
    _materialize_submodules,
    _real_repository_anchor,
    _run_git,
    _safe_slot_root,
    _slot_lock,
)
from .workspace import LabError, _atomic_write_text, validate_task_id

CLEAN_SLOT_SCHEMA_ID = "lab.clean-slot-source.v1"


def _repository_identity(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LabError(f"invalid clean request repository: {name}")
    commit = value.get("commit")
    tree_oid = value.get("tree_oid")
    raw_submodules = value.get("submodules")
    if not isinstance(commit, str) or not isinstance(tree_oid, str):
        raise LabError(f"invalid clean request source identity: {name}")
    if value.get("dirty") is not False or not isinstance(raw_submodules, Mapping):
        raise LabError(f"clean request source is not clean: {name}")
    return {
        "commit": commit,
        "tree_oid": tree_oid,
        "submodules": {str(path): str(oid) for path, oid in raw_submodules.items()},
    }


def _request_ref(request_id: str, repository: str) -> str:
    suffix = hashlib.sha256(repository.encode()).hexdigest()[:16]
    return f"refs/lab/requests/{request_id}/{suffix}"


def validate_clean_slot_receipt(
    receipt: object,
    *,
    request_id: str,
    idempotency_key: str,
    slot_id: str,
) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise LabError("clean slot receipt must be an object")
    expected_keys = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "snapshot_id",
        "slot_id",
        "materialization_root",
        "host_profile_sha256",
        "remote_profile_sha256",
        "control_plane_version",
        "repositories",
        "source_prepared",
        "source_anchor_mutated",
        "lfs_smudge_disabled",
        "environment_prepared",
        "assets_prepared",
        "service_started",
        "execution_eligible",
        "device_access",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("clean slot receipt has an unexpected shape")
    if (
        receipt.get("schema") != CLEAN_SLOT_SCHEMA_ID
        or receipt.get("request_id") != request_id
        or receipt.get("idempotency_key") != idempotency_key
        or receipt.get("slot_id") != slot_id
    ):
        raise LabError("clean slot receipt identity mismatch")
    for field in (
        "host_profile_sha256",
        "remote_profile_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LabError(f"clean slot receipt has invalid {field}")
    repositories = receipt.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise LabError("clean slot receipt has no repositories")
    repository_keys = {
        "path",
        "commit",
        "tree_oid",
        "submodules",
        "detached",
        "clean",
        "large_file_smudge",
        "depends_on_anchor_objects",
    }
    for name, repository in repositories.items():
        if not isinstance(name, str) or not isinstance(repository, dict):
            raise LabError("clean slot receipt has an invalid repository entry")
        if set(repository) != repository_keys:
            raise LabError(f"clean slot repository receipt shape mismatch: {name}")
        if (
            repository.get("detached") is not True
            or repository.get("clean") is not True
            or repository.get("large_file_smudge") is not False
            or repository.get("depends_on_anchor_objects") is not True
        ):
            raise LabError(f"clean slot repository receipt is unsafe: {name}")
    if (
        receipt.get("source_prepared") is not True
        or receipt.get("source_anchor_mutated") is not False
        or receipt.get("lfs_smudge_disabled") is not True
        or receipt.get("environment_prepared") is not False
        or receipt.get("assets_prepared") is not False
        or receipt.get("service_started") is not False
        or receipt.get("execution_eligible") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("clean slot receipt has an unsafe readiness envelope")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if hashlib.sha256(_canonical_json(content)).hexdigest() != receipt.get(
        "receipt_sha256"
    ):
        raise LabError("clean slot receipt hash mismatch")
    return receipt


def _verify_live(
    target: Path,
    request: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    slot_id: str,
) -> None:
    validated = validate_clean_slot_receipt(
        dict(receipt),
        request_id=str(request["request_id"]),
        idempotency_key=str(request["idempotency_key"]),
        slot_id=slot_id,
    )
    if validated.get("materialization_root") != str(target):
        raise LabError("clean slot materialization root mismatch")
    request_repositories = request.get("repositories")
    receipt_repositories = validated.get("repositories")
    if (
        not isinstance(request_repositories, Mapping)
        or not isinstance(receipt_repositories, Mapping)
        or set(request_repositories) != set(receipt_repositories)
    ):
        raise LabError("clean slot repository mapping mismatch")
    for name, raw_identity in request_repositories.items():
        if not isinstance(name, str):
            raise LabError("invalid clean slot repository name")
        identity = _repository_identity(raw_identity, name=name)
        raw_repository_receipt = receipt_repositories.get(name)
        checkout = (target / "repositories" / name).resolve()
        if not isinstance(raw_repository_receipt, Mapping) or (
            raw_repository_receipt.get("path") != str(checkout)
            or raw_repository_receipt.get("commit") != identity["commit"]
            or raw_repository_receipt.get("tree_oid") != identity["tree_oid"]
            or raw_repository_receipt.get("submodules") != identity["submodules"]
            or raw_repository_receipt.get("detached") is not True
            or raw_repository_receipt.get("clean") is not True
        ):
            raise LabError(f"clean slot repository receipt mismatch: {name}")
        if (
            git.repository_root(checkout) != checkout
            or git.current_branch(checkout) is not None
            or git.current_commit(checkout) != identity["commit"]
            or git.tree_oid(checkout) != identity["tree_oid"]
            or git.status_short(checkout)
            or _run_git(checkout, ["config", "--get", "lfs.fetchexclude"]) != "*"
        ):
            raise LabError(f"clean slot repository drifted: {name}")
        submodules, exact = git.submodule_status(checkout)
        if not exact or submodules != identity["submodules"]:
            raise LabError(f"clean slot submodules drifted: {name}")
        if (
            git.resolve_commit(checkout, _request_ref(str(request["request_id"]), name))
            != identity["commit"]
        ):
            raise LabError(f"clean slot source ref drifted: {name}")


def materialize_clean_request_sources(
    request: object,
    *,
    remote_profile: RemoteProfile,
    host_profile: HostProfile,
    control_plane_version: str,
    slot_id: str,
) -> dict[str, object]:
    """Prepare source only; it never resolves environments/assets or runs payloads."""

    validate_task_id(slot_id)
    validated = validate_remote_request(
        request,
        profile=remote_profile,
        control_plane_version=control_plane_version,
    )
    if remote_profile.target_host != host_profile.host_id:
        raise LabError("remote and host Profile identities do not match")
    if slot_id not in host_profile.slots:
        raise LabError(f"slot is not authorized by host Profile: {slot_id}")
    if host_profile.device_access or remote_profile.device_access:
        raise LabError("clean GPU slot source preparation refuses device access")
    request_repositories = validated.get("repositories")
    if not isinstance(request_repositories, Mapping):
        raise LabError("clean request has no repositories")
    anchors = {
        name: _real_repository_anchor(host_profile.repositories[name].path, name=name)
        for name in remote_profile.repositories
    }
    slot_root = _safe_slot_root(host_profile.slot_root, anchors)
    request_id = str(validated["request_id"])
    slot = slot_root / slot_id
    target = slot / "prepared" / request_id

    identities: dict[str, dict[str, object]] = {}
    for name, anchor in anchors.items():
        identity = _repository_identity(request_repositories[name], name=name)
        submodules, exact = git.submodule_status(anchor)
        if (
            git.current_commit(anchor) != identity["commit"]
            or git.tree_oid(anchor) != identity["tree_oid"]
            or git.status_short(anchor)
            or not exact
            or submodules != identity["submodules"]
        ):
            raise LabError(f"clean source anchor identity mismatch: {name}")
        identities[name] = identity

    with _slot_lock(slot_root, slot_id):
        if slot.is_symlink():
            raise LabError(f"clean slot path is a symlink: {slot}")
        prepared_root = target.parent
        if prepared_root.is_symlink() or target.is_symlink():
            raise LabError(f"clean slot path is a symlink: {target}")
        prepared_root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            receipt_path = target / "source-receipt.json"
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise LabError(f"clean slot receipt is missing or a symlink: {target}")
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LabError(f"cannot read clean slot receipt: {exc}") from exc
            if not isinstance(receipt, dict):
                raise LabError("clean slot receipt must contain an object")
            _verify_live(target, validated, receipt, slot_id=slot_id)
            return receipt

        fingerprints = {
            name: _anchor_fingerprint(anchor) for name, anchor in anchors.items()
        }
        temporary = Path(tempfile.mkdtemp(prefix=f".{request_id}.", dir=prepared_root))
        try:
            repositories_root = temporary / "repositories"
            repositories_root.mkdir()
            repository_receipts: dict[str, object] = {}
            for name, anchor in anchors.items():
                identity = identities[name]
                checkout = repositories_root / name
                _clone_detached(anchor, checkout, str(identity["commit"]))
                raw_submodules = identity["submodules"]
                assert isinstance(raw_submodules, Mapping)
                submodules = {
                    str(path): str(commit) for path, commit in raw_submodules.items()
                }
                _materialize_submodules(anchor, checkout, submodules)
                reference = _request_ref(request_id, name)
                _run_git(checkout, ["update-ref", reference, str(identity["commit"])])
                repository_receipts[name] = {
                    "path": str(target / "repositories" / name),
                    "commit": identity["commit"],
                    "tree_oid": identity["tree_oid"],
                    "submodules": submodules,
                    "detached": True,
                    "clean": True,
                    "large_file_smudge": False,
                    "depends_on_anchor_objects": True,
                }
            for name, anchor in anchors.items():
                if _anchor_fingerprint(anchor) != fingerprints[name]:
                    raise LabError(f"source anchor changed during preparation: {name}")
            content: dict[str, object] = {
                "schema": CLEAN_SLOT_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": validated["idempotency_key"],
                "task_id": validated["task_id"],
                "snapshot_id": validated["snapshot_id"],
                "slot_id": slot_id,
                "materialization_root": str(target),
                "host_profile_sha256": host_profile.sha256,
                "remote_profile_sha256": remote_profile.sha256,
                "control_plane_version": control_plane_version,
                "repositories": repository_receipts,
                "source_prepared": True,
                "source_anchor_mutated": False,
                "lfs_smudge_disabled": True,
                "environment_prepared": False,
                "assets_prepared": False,
                "service_started": False,
                "execution_eligible": False,
                "device_access": False,
            }
            receipt = {
                **content,
                "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
            }
            _atomic_write_text(
                temporary / "source-receipt.json",
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )
            temporary.rename(target)
            _verify_live(target, validated, receipt, slot_id=slot_id)
            return receipt
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
