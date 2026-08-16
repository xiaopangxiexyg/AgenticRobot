from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .remote_dispatch import validate_remote_outbox
from .remote_request import (
    SHA256_PATTERN,
    RemoteProfile,
    _sha256,
    validate_remote_request,
)
from .runner_state import RequestStateStore
from .workspace import LabError, _atomic_write_text, validate_task_id

REMOTE_ACCEPTANCE_SCHEMA_ID = "lab.remote-acceptance.v1"


def validate_remote_acceptance(
    receipt: object,
    *,
    request_id: str,
    idempotency_key: str,
    target_host: str,
) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise LabError("remote acceptance receipt must be an object")
    expected_keys = {
        "schema",
        "request_id",
        "idempotency_key",
        "target_host",
        "remote_profile_sha256",
        "control_plane_version",
        "outbox_envelope_sha256",
        "request_state_sha256",
        "accepted_state",
        "accepted_at",
        "execution_eligible",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise LabError("remote acceptance receipt has an unexpected shape")
    if (
        receipt.get("schema") != REMOTE_ACCEPTANCE_SCHEMA_ID
        or receipt.get("request_id") != request_id
        or receipt.get("idempotency_key") != idempotency_key
        or receipt.get("target_host") != target_host
    ):
        raise LabError("remote acceptance receipt identity mismatch")
    for field in (
        "remote_profile_sha256",
        "outbox_envelope_sha256",
        "request_state_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise LabError(f"remote acceptance receipt has invalid {field}")
    if (
        receipt.get("accepted_state") != "SUBMITTED"
        or receipt.get("execution_eligible") is not False
    ):
        raise LabError("remote acceptance receipt overclaims execution state")
    control_version = receipt.get("control_plane_version")
    if not isinstance(control_version, str):
        raise LabError("remote acceptance receipt has invalid control-plane version")
    validate_task_id(control_version)
    accepted_at = receipt.get("accepted_at")
    if not isinstance(accepted_at, str) or not accepted_at:
        raise LabError("remote acceptance receipt has invalid accepted_at")
    try:
        timestamp = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("remote acceptance receipt has invalid accepted_at") from exc
    if timestamp.tzinfo is None:
        raise LabError("remote acceptance receipt accepted_at requires a timezone")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("remote acceptance receipt hash mismatch")
    return receipt


class RemoteInbox:
    """Persist authenticated-transport payloads before any runner preparation."""

    def __init__(self, root: Path, state_store: RequestStateStore) -> None:
        self.root = root.expanduser().resolve()
        self.state_store = state_store

    @contextmanager
    def _lock(self, request_id: str) -> Iterator[None]:
        validate_task_id(request_id)
        lock_root = self.root / ".locks"
        if self.root.is_symlink() or lock_root.is_symlink():
            raise LabError(f"remote inbox root is unsafe: {self.root}")
        lock_root.mkdir(parents=True, exist_ok=True)
        path = lock_root / f"{request_id}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LabError(
                    f"remote inbox request lock is held: {request_id}"
                ) from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def accept(
        self,
        envelope: object,
        *,
        remote_profile: RemoteProfile,
        control_plane_version: str,
    ) -> dict[str, object]:
        validated_envelope = validate_remote_outbox(envelope)
        request = validate_remote_request(
            validated_envelope["request"],
            profile=remote_profile,
            control_plane_version=control_plane_version,
        )
        request_profile = request.get("profile")
        if not isinstance(request_profile, dict) or (
            remote_profile.target_host != request_profile.get("target_host")
        ):
            raise LabError("remote inbox target host does not match protected Profile")
        request_id = str(request["request_id"])
        target = self.root / request_id
        envelope_path = target / "envelope.json"
        request_path = target / "request.json"
        receipt_path = target / "acceptance.json"
        with self._lock(request_id):
            if target.is_symlink():
                raise LabError(f"remote inbox target is a symlink: {target}")
            target.mkdir(parents=True, exist_ok=True)
            if envelope_path.is_file():
                try:
                    existing_envelope = json.loads(
                        envelope_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise LabError(
                        f"cannot read existing remote inbox envelope: {envelope_path}"
                    ) from exc
                if existing_envelope != validated_envelope:
                    raise LabError(f"remote inbox envelope collision: {request_id}")
            else:
                _atomic_write_text(
                    envelope_path,
                    json.dumps(validated_envelope, indent=2, sort_keys=True) + "\n",
                )
            if request_path.is_file():
                try:
                    existing_request = json.loads(
                        request_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise LabError(
                        f"cannot read existing remote inbox request: {request_path}"
                    ) from exc
                if existing_request != request:
                    raise LabError(f"remote inbox request collision: {request_id}")
            else:
                _atomic_write_text(
                    request_path,
                    json.dumps(request, indent=2, sort_keys=True) + "\n",
                )
            state = self.state_store.submit(request)
            if state.get("state") != "SUBMITTED":
                raise LabError(
                    f"remote inbox request already advanced beyond acceptance: {request_id}"
                )
            if receipt_path.is_file():
                return self.load_acceptance(
                    request_id,
                    idempotency_key=str(request["idempotency_key"]),
                    target_host=remote_profile.target_host,
                )
            content: dict[str, object] = {
                "schema": REMOTE_ACCEPTANCE_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": request["idempotency_key"],
                "target_host": remote_profile.target_host,
                "remote_profile_sha256": remote_profile.sha256,
                "control_plane_version": control_plane_version,
                "outbox_envelope_sha256": validated_envelope["envelope_sha256"],
                "request_state_sha256": _sha256(state),
                "accepted_state": "SUBMITTED",
                "accepted_at": datetime.now(timezone.utc).isoformat(),
                "execution_eligible": False,
            }
            receipt = {**content, "receipt_sha256": _sha256(content)}
            validate_remote_acceptance(
                receipt,
                request_id=request_id,
                idempotency_key=str(request["idempotency_key"]),
                target_host=remote_profile.target_host,
            )
            _atomic_write_text(
                receipt_path,
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )
            return receipt

    def load_acceptance(
        self,
        request_id: str,
        *,
        idempotency_key: str,
        target_host: str,
    ) -> dict[str, object]:
        validate_task_id(request_id)
        path = self.root / request_id / "acceptance.json"
        if self.root.is_symlink() or path.is_symlink() or not path.is_file():
            raise LabError(f"remote acceptance receipt does not exist: {request_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(
                f"cannot read remote acceptance receipt {path}: {exc}"
            ) from exc
        return validate_remote_acceptance(
            payload,
            request_id=request_id,
            idempotency_key=idempotency_key,
            target_host=target_host,
        )
