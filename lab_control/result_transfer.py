from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .request_transport import AUTHENTICATION_SCHEME
from .run_requests import RunRequestStore, validate_run_request
from .source_transfer import (
    _canonical_json,
    _secret,
    _sha256_bytes,
    _sha256_file,
    _write_new_json,
)
from .workspace import LabError, validate_task_id

RESULT_PACKAGE_SCHEMA_ID = "lab.result-package.v1"
RESULT_EVIDENCE_SCHEMA_ID = "lab.result-evidence.v1"
RESULT_INGRESS_RECEIPT_SCHEMA_ID = "lab.result-ingress-receipt.v1"
RESULT_PACKAGE_MANIFEST = "result-package.json"
RESULT_EVIDENCE_FILE = "result-evidence.json"

_MAXIMUM_DOCUMENT_BYTES = 2 * 1024 * 1024
_DEFAULT_MAXIMUM_ARTIFACT_BYTES = 16 * 1024 * 1024
_DEFAULT_MAXIMUM_TOTAL_ARTIFACT_BYTES = 64 * 1024 * 1024
_DEFAULT_MAXIMUM_ARTIFACTS = 256
_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "QUARANTINED"})
_RESULT_ATTACHMENTS = (
    "runner-claim",
    "snapshot-stage-receipt",
    "command-execution-receipt",
    "artifact-receipt",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, maximum_bytes: int = _MAXIMUM_DOCUMENT_BYTES) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LabError(f"result package file is missing or unsafe: {path}")
    if path.stat().st_size > maximum_bytes:
        raise LabError(f"result package file exceeds {maximum_bytes} bytes: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read result package file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"result package document must be an object: {path}")
    return payload


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LabError(f"invalid {label}: {value!r}")
    path = PurePosixPath(value)
    if (
        value.startswith(("/", "~"))
        or "\\" in value
        or "\x00" in value
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise LabError(f"unsafe {label}: {value!r}")
    return value


def _safe_file(root: Path, relative_path: object, *, maximum_bytes: int) -> Path:
    relative = _safe_relative_path(relative_path, label="result package path")
    path = root.joinpath(*PurePosixPath(relative).parts)
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise LabError(f"result package path escapes its root: {relative!r}")
    if path.is_symlink() or not path.is_file():
        raise LabError(f"result package file is missing or unsafe: {relative}")
    if path.stat().st_size > maximum_bytes:
        raise LabError(f"result package file exceeds {maximum_bytes} bytes: {relative}")
    return path


def _validate_receipt_hash(payload: object, *, attachment_id: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise LabError(f"result attachment is invalid: {attachment_id}")
    request_id = payload.get("request_id")
    idempotency_key = payload.get("idempotency_key")
    receipt_sha256 = payload.get("receipt_sha256")
    if not all(isinstance(value, str) for value in (request_id, idempotency_key, receipt_sha256)):
        raise LabError(f"result attachment identity is invalid: {attachment_id}")
    excluded = {"receipt_sha256"}
    if attachment_id == "runner-claim":
        excluded.add("claim_id")
    content = {key: value for key, value in payload.items() if key not in excluded}
    if receipt_sha256 != _sha256_bytes(_canonical_json(content)):
        raise LabError(f"result attachment hash mismatch: {attachment_id}")
    if attachment_id == "runner-claim" and payload.get("claim_id") != (
        f"C-{receipt_sha256[:24]}"
    ):
        raise LabError("result runner claim identity mismatch")
    return payload


def _validate_events(
    events: object,
    *,
    request: Mapping[str, object],
    state: Mapping[str, object],
) -> list[dict[str, object]]:
    expected = {
        "schema",
        "request_id",
        "idempotency_key",
        "sequence",
        "previous_state",
        "state",
        "actor",
        "at",
        "slot_id",
        "outcome",
        "event_sha256",
    }
    history = state.get("history")
    if not isinstance(history, list) or not history:
        raise LabError("result state history is unavailable")
    if not isinstance(events, list) or len(events) != len(history):
        raise LabError("result event sequence differs from state history")
    validated: list[dict[str, object]] = []
    previous_state: object = None
    for sequence, (event, summary) in enumerate(zip(events, history, strict=True)):
        if (
            not isinstance(event, dict)
            or set(event) != expected
            or not isinstance(summary, dict)
        ):
            raise LabError("result event has an unexpected shape")
        if (
            event.get("request_id") != request.get("request_id")
            or event.get("idempotency_key") != request.get("idempotency_key")
            or event.get("sequence") != sequence
            or event.get("previous_state") != previous_state
            or event.get("state") != summary.get("state")
            or event.get("at") != summary.get("at")
            or event.get("event_sha256") != summary.get("event_sha256")
        ):
            raise LabError("result event identity mismatch")
        content = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
        if event.get("event_sha256") != _sha256_bytes(_canonical_json(content)):
            raise LabError("result event hash mismatch")
        previous_state = event.get("state")
        validated.append(event)
    final_event = validated[-1]
    if (
        final_event.get("state") != state.get("state")
        or final_event.get("outcome") != state.get("outcome")
        or final_event.get("slot_id") != state.get("slot_id")
    ):
        raise LabError("result final event differs from terminal state")
    return validated


def _validate_remote_state(
    state: object,
    *,
    request: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "snapshot_id",
        "profile_id",
        "target_id",
        "evidence_tier",
        "state",
        "slot_id",
        "outcome",
        "submitted_at",
        "updated_at",
        "history",
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise LabError("result state has an unexpected shape")
    snapshot = request.get("snapshot")
    profile = request.get("profile")
    if not isinstance(snapshot, dict) or not isinstance(profile, dict):
        raise LabError("result request has an invalid nested identity")
    identity = {
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "task_id": request["task_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "profile_id": profile["profile_id"],
        "target_id": request["target_id"],
        "evidence_tier": request["evidence_tier"],
    }
    if any(state.get(key) != value for key, value in identity.items()):
        raise LabError("result state identity mismatch")
    if state.get("state") not in _TERMINAL_STATES or state.get("outcome") is None:
        raise LabError("result package requires a terminal remote state")
    outcome = state.get("outcome")
    if (
        not isinstance(outcome, dict)
        or set(outcome) != {"classification", "detail"}
        or not isinstance(outcome.get("classification"), str)
        or not isinstance(outcome.get("detail"), str)
    ):
        raise LabError("result package terminal outcome is invalid")
    if not isinstance(state.get("slot_id"), str):
        raise LabError("result package terminal state lacks a slot")
    validate_task_id(str(state["slot_id"]))
    return state


def _copy_new_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            descriptor = -1
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class ResultPackageBuilder:
    """Export a terminal request and only its registered, bounded artifacts."""

    def __init__(
        self,
        store: RunRequestStore,
        *,
        slot_root: Path,
        maximum_artifact_bytes: int = _DEFAULT_MAXIMUM_ARTIFACT_BYTES,
        maximum_total_artifact_bytes: int = _DEFAULT_MAXIMUM_TOTAL_ARTIFACT_BYTES,
        maximum_artifacts: int = _DEFAULT_MAXIMUM_ARTIFACTS,
    ) -> None:
        self.store = store
        self.slot_root = slot_root.expanduser().resolve()
        self.maximum_artifact_bytes = maximum_artifact_bytes
        self.maximum_total_artifact_bytes = maximum_total_artifact_bytes
        self.maximum_artifacts = maximum_artifacts

    def build(
        self,
        request_id: str,
        destination: Path,
        *,
        sender_id: str,
        recipient_id: str,
        key_id: str,
        secret: bytes,
    ) -> dict[str, object]:
        request_id = validate_task_id(request_id)
        sender_id = validate_task_id(sender_id)
        recipient_id = validate_task_id(recipient_id)
        key_id = validate_task_id(key_id)
        secret = _secret(secret, key_id=key_id)
        request = self.store.load_request(request_id)
        state = _validate_remote_state(self.store.load(request_id), request=request)
        events = _validate_events(
            self.store.events(request_id), request=request, state=state
        )
        attachments: dict[str, object] = {}
        for attachment_id in _RESULT_ATTACHMENTS:
            if not self.store.has_attachment(request_id, attachment_id):
                continue
            attachment = _validate_receipt_hash(
                self.store.load_attachment(request_id, attachment_id),
                attachment_id=attachment_id,
            )
            if (
                attachment.get("request_id") != request_id
                or attachment.get("idempotency_key") != request.get("idempotency_key")
            ):
                raise LabError(f"result attachment identity mismatch: {attachment_id}")
            attachments[attachment_id] = attachment
        if "runner-claim" not in attachments:
            raise LabError("result package requires a registered runner claim")
        artifact_receipt = attachments.get("artifact-receipt")
        if state.get("state") == "COMPLETED" and artifact_receipt is None:
            raise LabError("completed result lacks its artifact receipt")
        raw_artifacts = (
            artifact_receipt.get("artifacts")
            if isinstance(artifact_receipt, dict)
            else []
        )
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) > self.maximum_artifacts:
            raise LabError("result package artifact count exceeds its limit")
        slot_id = validate_task_id(str(state["slot_id"]))
        request_root = self.slot_root / slot_id / request_id
        if raw_artifacts and (
            request_root.is_symlink()
            or not request_root.is_dir()
            or not request_root.resolve().is_relative_to(self.slot_root.resolve())
        ):
            raise LabError(f"result request root is missing or unsafe: {request_root}")

        destination = destination.expanduser().resolve()
        if destination.exists() or destination.is_symlink():
            raise LabError(f"result package destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink():
            raise LabError(f"result package parent must not be a symlink: {destination.parent}")
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            artifact_records: list[dict[str, object]] = []
            total_bytes = 0
            for raw_artifact in raw_artifacts:
                expected = {
                    "artifact_id",
                    "relative_path",
                    "media_type",
                    "byte_size",
                    "sha256",
                    "storage_id",
                }
                if not isinstance(raw_artifact, dict) or set(raw_artifact) != expected:
                    raise LabError("result artifact descriptor has an unexpected shape")
                artifact_id = validate_task_id(str(raw_artifact.get("artifact_id", "")))
                source_relative = _safe_relative_path(
                    raw_artifact.get("relative_path"), label="result artifact path"
                )
                source = request_root.joinpath(*PurePosixPath(source_relative).parts)
                if (
                    not source.resolve(strict=False).is_relative_to(request_root.resolve())
                    or source.is_symlink()
                    or not source.is_file()
                ):
                    raise LabError(f"result artifact is missing or unsafe: {source_relative}")
                actual_size = source.stat().st_size
                if (
                    actual_size != raw_artifact.get("byte_size")
                    or _sha256_file(source) != raw_artifact.get("sha256")
                ):
                    raise LabError(f"result artifact changed after registration: {artifact_id}")
                if actual_size > self.maximum_artifact_bytes:
                    raise LabError(
                        f"result artifact exceeds {self.maximum_artifact_bytes} bytes: "
                        f"{artifact_id}"
                    )
                total_bytes += actual_size
                if total_bytes > self.maximum_total_artifact_bytes:
                    raise LabError(
                        "result artifacts exceed "
                        f"{self.maximum_total_artifact_bytes} bytes"
                    )
                package_relative = f"artifacts/{artifact_id}"
                _copy_new_file(source, staging / package_relative)
                artifact_records.append(
                    {
                        **raw_artifact,
                        "source_relative_path": source_relative,
                        "package_path": package_relative,
                    }
                )

            evidence: dict[str, object] = {
                "schema": RESULT_EVIDENCE_SCHEMA_ID,
                "request": request,
                "remote_state": state,
                "events": events,
                "attachments": attachments,
            }
            evidence_path = staging / RESULT_EVIDENCE_FILE
            _write_new_json(evidence_path, evidence)
            if evidence_path.stat().st_size > _MAXIMUM_DOCUMENT_BYTES:
                raise LabError("result evidence exceeds its size limit")
            content: dict[str, object] = {
                "schema": RESULT_PACKAGE_SCHEMA_ID,
                "message_type": "result-package",
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "created_at": _utc_now(),
                "request_id": request_id,
                "idempotency_key": request["idempotency_key"],
                "evidence_path": RESULT_EVIDENCE_FILE,
                "evidence_sha256": _sha256_file(evidence_path),
                "artifacts": artifact_records,
            }
            package_sha256 = _sha256_bytes(_canonical_json(content))
            authenticated = {
                **content,
                "package_id": f"RP-{package_sha256[:24]}",
                "package_sha256": package_sha256,
                "authentication": {
                    "scheme": AUTHENTICATION_SCHEME,
                    "key_id": key_id,
                },
            }
            manifest = {
                **authenticated,
                "authentication_mac": hmac.new(
                    secret, _canonical_json(authenticated), hashlib.sha256
                ).hexdigest(),
            }
            _write_new_json(staging / RESULT_PACKAGE_MANIFEST, manifest)
            os.replace(staging, destination)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


class ResultPackageIngress:
    """Authenticate immutable runner evidence and import it beside controller state."""

    def __init__(
        self,
        store: RunRequestStore,
        *,
        result_root: Path,
        recipient_id: str,
        trusted_sender_keys: Mapping[str, Mapping[str, bytes]],
        maximum_artifact_bytes: int = _DEFAULT_MAXIMUM_ARTIFACT_BYTES,
        maximum_total_artifact_bytes: int = _DEFAULT_MAXIMUM_TOTAL_ARTIFACT_BYTES,
        maximum_artifacts: int = _DEFAULT_MAXIMUM_ARTIFACTS,
    ) -> None:
        self.store = store
        self.result_root = result_root.expanduser().resolve()
        self.recipient_id = validate_task_id(recipient_id)
        self.trusted_sender_keys = trusted_sender_keys
        self.maximum_artifact_bytes = maximum_artifact_bytes
        self.maximum_total_artifact_bytes = maximum_total_artifact_bytes
        self.maximum_artifacts = maximum_artifacts

    def _validate_manifest(self, payload: object) -> dict[str, object]:
        expected = {
            "schema",
            "message_type",
            "sender_id",
            "recipient_id",
            "created_at",
            "request_id",
            "idempotency_key",
            "evidence_path",
            "evidence_sha256",
            "artifacts",
            "package_id",
            "package_sha256",
            "authentication",
            "authentication_mac",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise LabError("result package manifest has an unexpected shape")
        sender_id = validate_task_id(str(payload.get("sender_id", "")))
        if (
            payload.get("schema") != RESULT_PACKAGE_SCHEMA_ID
            or payload.get("message_type") != "result-package"
            or payload.get("recipient_id") != self.recipient_id
        ):
            raise LabError("result package identity mismatch")
        authentication = payload.get("authentication")
        if not isinstance(authentication, dict) or set(authentication) != {
            "scheme",
            "key_id",
        }:
            raise LabError("result package authentication has an unexpected shape")
        key_id = validate_task_id(str(authentication.get("key_id", "")))
        if authentication.get("scheme") != AUTHENTICATION_SCHEME:
            raise LabError("unsupported result package authentication scheme")
        sender_keys = self.trusted_sender_keys.get(sender_id)
        if sender_keys is None or key_id not in sender_keys:
            raise LabError(f"result package sender/key is not trusted: {sender_id}/{key_id}")
        secret = _secret(sender_keys[key_id], key_id=key_id)
        content = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "package_id",
                "package_sha256",
                "authentication",
                "authentication_mac",
            }
        }
        package_sha256 = _sha256_bytes(_canonical_json(content))
        if (
            payload.get("package_sha256") != package_sha256
            or payload.get("package_id") != f"RP-{package_sha256[:24]}"
        ):
            raise LabError("result package manifest hash mismatch")
        authenticated = {
            **content,
            "package_id": payload["package_id"],
            "package_sha256": package_sha256,
            "authentication": authentication,
        }
        mac = payload.get("authentication_mac")
        if not isinstance(mac, str) or not hmac.compare_digest(
            mac,
            hmac.new(secret, _canonical_json(authenticated), hashlib.sha256).hexdigest(),
        ):
            raise LabError("result package authentication failed")
        return payload

    def _validate_imported_copy(
        self,
        target: Path,
        *,
        manifest: Mapping[str, object],
    ) -> None:
        if target.is_symlink() or not target.is_dir():
            raise LabError(f"imported result package is missing or unsafe: {target}")
        if _read_json(target / RESULT_PACKAGE_MANIFEST) != manifest:
            raise LabError("imported result package manifest changed")
        evidence = _safe_file(
            target,
            manifest.get("evidence_path"),
            maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
        )
        if _sha256_file(evidence) != manifest.get("evidence_sha256"):
            raise LabError("imported result evidence changed")
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise LabError("imported result artifact index is invalid")
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, dict):
                raise LabError("imported result artifact index is invalid")
            artifact = _safe_file(
                target,
                raw_artifact.get("package_path"),
                maximum_bytes=self.maximum_artifact_bytes,
            )
            if (
                artifact.stat().st_size != raw_artifact.get("byte_size")
                or _sha256_file(artifact) != raw_artifact.get("sha256")
            ):
                raise LabError("imported result artifact changed")

    def ingest(self, package_root: Path) -> dict[str, object]:
        package_root = package_root.expanduser().resolve()
        if package_root.is_symlink() or not package_root.is_dir():
            raise LabError(f"result package root is missing or unsafe: {package_root}")
        manifest = self._validate_manifest(
            _read_json(package_root / RESULT_PACKAGE_MANIFEST)
        )
        request_id = validate_task_id(str(manifest.get("request_id", "")))
        local_request = self.store.load_request(request_id)
        if local_request.get("idempotency_key") != manifest.get("idempotency_key"):
            raise LabError("result package differs from the local request")
        evidence_path = _safe_file(
            package_root,
            manifest.get("evidence_path"),
            maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
        )
        if _sha256_file(evidence_path) != manifest.get("evidence_sha256"):
            raise LabError("result package evidence hash mismatch")
        evidence = _read_json(evidence_path)
        if set(evidence) != {
            "schema",
            "request",
            "remote_state",
            "events",
            "attachments",
        } or evidence.get("schema") != RESULT_EVIDENCE_SCHEMA_ID:
            raise LabError("result evidence has an unexpected shape")
        remote_request = validate_run_request(evidence.get("request"))
        if remote_request != local_request:
            raise LabError("result package request differs from local immutable content")
        remote_state = _validate_remote_state(
            evidence.get("remote_state"), request=remote_request
        )
        _validate_events(
            evidence.get("events"), request=remote_request, state=remote_state
        )
        attachments = evidence.get("attachments")
        if not isinstance(attachments, dict) or any(
            key not in _RESULT_ATTACHMENTS for key in attachments
        ):
            raise LabError("result evidence attachments are invalid")
        for attachment_id, attachment in attachments.items():
            validated = _validate_receipt_hash(
                attachment, attachment_id=str(attachment_id)
            )
            if (
                validated.get("request_id") != request_id
                or validated.get("idempotency_key") != local_request.get("idempotency_key")
            ):
                raise LabError(f"result attachment identity mismatch: {attachment_id}")

        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) > self.maximum_artifacts:
            raise LabError("result package artifact count exceeds its limit")
        validated_artifacts: list[tuple[dict[str, object], Path]] = []
        total_bytes = 0
        for raw_artifact in raw_artifacts:
            expected = {
                "artifact_id",
                "relative_path",
                "media_type",
                "byte_size",
                "sha256",
                "storage_id",
                "source_relative_path",
                "package_path",
            }
            if not isinstance(raw_artifact, dict) or set(raw_artifact) != expected:
                raise LabError("result package artifact has an unexpected shape")
            artifact_id = validate_task_id(str(raw_artifact.get("artifact_id", "")))
            byte_size = raw_artifact.get("byte_size")
            if (
                not isinstance(byte_size, int)
                or isinstance(byte_size, bool)
                or not 0 <= byte_size <= self.maximum_artifact_bytes
            ):
                raise LabError(f"result artifact size is invalid: {artifact_id}")
            total_bytes += byte_size
            if total_bytes > self.maximum_total_artifact_bytes:
                raise LabError("result package total artifact size exceeds its limit")
            artifact_path = _safe_file(
                package_root,
                raw_artifact.get("package_path"),
                maximum_bytes=self.maximum_artifact_bytes,
            )
            if (
                artifact_path.stat().st_size != byte_size
                or _sha256_file(artifact_path) != raw_artifact.get("sha256")
            ):
                raise LabError(f"result package artifact hash mismatch: {artifact_id}")
            validated_artifacts.append((raw_artifact, artifact_path))
        artifact_receipt = attachments.get("artifact-receipt")
        expected_descriptors = (
            artifact_receipt.get("artifacts")
            if isinstance(artifact_receipt, dict)
            else []
        )
        observed_descriptors = [
            {
                key: artifact[key]
                for key in (
                    "artifact_id",
                    "relative_path",
                    "media_type",
                    "byte_size",
                    "sha256",
                    "storage_id",
                )
            }
            for artifact, _ in validated_artifacts
        ]
        if observed_descriptors != expected_descriptors:
            raise LabError("result package artifacts differ from their receipt")

        package_id = validate_task_id(str(manifest.get("package_id", "")))
        target = self.result_root / request_id / package_id
        attachment_id = f"result-{str(manifest['package_sha256'])[:24]}"
        if self.store.has_attachment(request_id, attachment_id):
            receipt = self.store.load_attachment(request_id, attachment_id)
            if (
                receipt.get("schema") != RESULT_INGRESS_RECEIPT_SCHEMA_ID
                or receipt.get("package_id") != package_id
                or receipt.get("package_sha256") != manifest.get("package_sha256")
            ):
                raise LabError("result package ingress receipt collision")
            self._validate_imported_copy(target, manifest=manifest)
            return {
                "schema": "lab.result-ingress-result.v1",
                "request_id": request_id,
                "created": False,
                "receipt": receipt,
            }
        self.result_root.mkdir(parents=True, exist_ok=True)
        if self.result_root.is_symlink():
            raise LabError(f"result import root must not be a symlink: {self.result_root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            self._validate_imported_copy(target, manifest=manifest)
        else:
            staging = Path(
                tempfile.mkdtemp(prefix=f".{package_id}.", dir=target.parent)
            )
            try:
                _copy_new_file(evidence_path, staging / RESULT_EVIDENCE_FILE)
                _copy_new_file(
                    package_root / RESULT_PACKAGE_MANIFEST,
                    staging / RESULT_PACKAGE_MANIFEST,
                )
                for artifact, source in validated_artifacts:
                    relative = _safe_relative_path(
                        artifact["package_path"],
                        label="result artifact package path",
                    )
                    _copy_new_file(
                        source, staging.joinpath(*PurePosixPath(relative).parts)
                    )
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            self._validate_imported_copy(target, manifest=manifest)
        remote_outcome = remote_state["outcome"]
        assert isinstance(remote_outcome, dict)
        receipt_content: dict[str, object] = {
            "schema": RESULT_INGRESS_RECEIPT_SCHEMA_ID,
            "request_id": request_id,
            "idempotency_key": local_request["idempotency_key"],
            "package_id": package_id,
            "package_sha256": manifest["package_sha256"],
            "sender_id": manifest["sender_id"],
            "recipient_id": manifest["recipient_id"],
            "remote_state": remote_state["state"],
            "classification": remote_outcome["classification"],
            "artifact_count": len(validated_artifacts),
            "import_path": str(target),
            "ingested_at": _utc_now(),
        }
        receipt = {
            **receipt_content,
            "receipt_sha256": _sha256_bytes(_canonical_json(receipt_content)),
        }
        stored, _ = self.store.put_attachment(request_id, attachment_id, receipt)
        return {
            "schema": "lab.result-ingress-result.v1",
            "request_id": request_id,
            "created": True,
            "receipt": stored,
        }
