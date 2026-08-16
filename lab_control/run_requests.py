from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .queue_profiles import QueueProfile, load_queue_profile
from .runs import (
    CANDIDATE_EVIDENCE_TIER,
    DEVELOPMENT_EVIDENCE_TIER,
    RunManager,
    control_plane_identity,
)
from .snapshot_resolution import resolve_clean_snapshot_identity
from .workspace import (
    LabError,
    WorkspaceManager,
    _atomic_write_text,
    validate_repository_name,
    validate_submodule_path,
    validate_task_id,
)

RUN_REQUEST_SCHEMA_ID = "lab.run-request.v1"
REQUEST_STATE_SCHEMA_ID = "lab.run-request-state.v1"
REQUEST_EVENT_SCHEMA_ID = "lab.run-request-event.v1"
REQUEST_SUBMISSION_SCHEMA_ID = "lab.run-request-submission.v1"
_OID_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ACTIVE_STATES = frozenset(
    {"RESERVED", "PREPARING", "RUNNING", "COLLECTING", "CANCEL_REQUESTED"}
)
_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "QUARANTINED"})
_TRANSITIONS = {
    "SUBMITTED": frozenset({"RESERVED", "CANCELLED", "FAILED"}),
    "RESERVED": frozenset({"PREPARING", "CANCEL_REQUESTED", "FAILED"}),
    "PREPARING": frozenset(
        {"RUNNING", "CANCEL_REQUESTED", "FAILED", "QUARANTINED"}
    ),
    "RUNNING": frozenset(
        {"COLLECTING", "CANCEL_REQUESTED", "FAILED", "QUARANTINED"}
    ),
    "COLLECTING": frozenset(
        {"COMPLETED", "CANCEL_REQUESTED", "FAILED", "QUARANTINED"}
    ),
    "CANCEL_REQUESTED": frozenset({"CANCELLED", "QUARANTINED"}),
}
_VALID_STATES = frozenset({"SUBMITTED", *_TRANSITIONS, *_TERMINAL_STATES})
_OUTCOME_CLASSIFICATIONS = frozenset(
    {
        "PASSED",
        "FAILED_ASSERTION",
        "BLOCKED_PRECONDITION",
        "ABORTED_SAFETY",
        "INFRASTRUCTURE_FAILURE",
        "INVALID_SOURCE",
        "INVALID_ENVIRONMENT",
        "INCONCLUSIVE_MANUAL",
        "CANCELLED",
        "QUARANTINED",
    }
)
_MAX_STATE_BYTES = 2 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_oid(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _OID_PATTERN.fullmatch(value):
        raise LabError(f"invalid {label}: {value!r}")
    return value


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise LabError(f"invalid {label}: {value!r}")
    return value


def _validate_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise LabError(f"invalid {label}: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LabError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LabError(f"{label} must include a UTC offset")
    return value


def _portable_snapshot(
    snapshot_id: str,
    identity: Mapping[str, object],
) -> dict[str, object]:
    validate_task_id(snapshot_id)
    raw_repositories = identity.get("repositories")
    if not isinstance(raw_repositories, dict) or not raw_repositories:
        raise LabError("Run request snapshot requires repositories")
    repositories: dict[str, object] = {}
    for raw_name, raw_repository in sorted(raw_repositories.items()):
        name = validate_repository_name(str(raw_name))
        if not isinstance(raw_repository, dict):
            raise LabError(f"invalid Run request repository identity: {name}")
        raw_submodules = raw_repository.get("submodules", {})
        if not isinstance(raw_submodules, dict):
            raise LabError(f"invalid Run request submodules: {name}")
        submodules = {
            validate_submodule_path(str(path)): _validate_oid(
                commit, label=f"submodule commit {name}/{path}"
            )
            for path, commit in sorted(raw_submodules.items())
        }
        repositories[name] = {
            "commit": _validate_oid(
                raw_repository.get("commit"), label=f"repository commit {name}"
            ),
            "tree_oid": _validate_oid(
                raw_repository.get("tree_oid"), label=f"repository tree {name}"
            ),
            "submodules": submodules,
        }

    raw_links = identity.get("repository_links", [])
    if not isinstance(raw_links, list):
        raise LabError("Run request repository links must be a list")
    links: list[dict[str, str]] = []
    for raw_link in raw_links:
        if not isinstance(raw_link, dict) or set(raw_link) != {
            "parent_repository",
            "submodule_path",
            "child_repository",
            "commit",
        }:
            raise LabError("Run request repository link has an unexpected shape")
        parent = validate_repository_name(str(raw_link["parent_repository"]))
        child = validate_repository_name(str(raw_link["child_repository"]))
        submodule_path = validate_submodule_path(str(raw_link["submodule_path"]))
        commit = _validate_oid(raw_link["commit"], label="repository link commit")
        parent_identity = repositories.get(parent)
        child_identity = repositories.get(child)
        if not isinstance(parent_identity, dict) or not isinstance(child_identity, dict):
            raise LabError("Run request repository link references an absent repository")
        parent_submodules = parent_identity.get("submodules")
        if (
            not isinstance(parent_submodules, dict)
            or parent_submodules.get(submodule_path) != commit
            or child_identity.get("commit") != commit
        ):
            raise LabError("Run request repository link differs from snapshot identities")
        links.append(
            {
                "parent_repository": parent,
                "submodule_path": submodule_path,
                "child_repository": child,
                "commit": commit,
            }
        )

    raw_baseline = identity.get("baseline")
    baseline: dict[str, str] | None = None
    if raw_baseline is not None:
        if not isinstance(raw_baseline, dict) or set(raw_baseline) != {
            "baseline_id",
            "sha256",
        }:
            raise LabError("Run request baseline has an unexpected shape")
        baseline = {
            "baseline_id": validate_task_id(str(raw_baseline["baseline_id"])),
            "sha256": _validate_sha256(
                raw_baseline["sha256"], label="baseline SHA-256"
            ),
        }
    return {
        "snapshot_id": snapshot_id,
        "repositories": repositories,
        "repository_links": links,
        "baseline": baseline,
    }


def _validated_control_plane(value: object) -> dict[str, object]:
    expected = {
        "version",
        "source_commit",
        "source_tree_oid",
        "source_dirty",
        "executor_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise LabError("Run request control plane has an unexpected shape")
    version = value.get("version")
    if not isinstance(version, str) or not version or len(version) > 64:
        raise LabError("Run request control-plane version is invalid")
    validate_task_id(version)
    source_dirty = value.get("source_dirty")
    if not isinstance(source_dirty, bool):
        raise LabError("Run request control-plane dirty state is invalid")
    normalized: dict[str, object] = {
        "version": version,
        "source_commit": _validate_oid(
            value.get("source_commit"), label="control-plane commit"
        ),
        "source_tree_oid": _validate_oid(
            value.get("source_tree_oid"), label="control-plane tree"
        ),
        "source_dirty": source_dirty,
        "executor_sha256": _validate_sha256(
            value.get("executor_sha256"), label="control-plane executor SHA-256"
        ),
    }
    if normalized != value:
        raise LabError("Run request control-plane identity is not canonical")
    return normalized


def build_run_request(
    *,
    task_id: str,
    snapshot_id: str,
    snapshot_identity: Mapping[str, object],
    profile: QueueProfile,
    evidence_tier: str,
    control_plane: Mapping[str, object],
) -> dict[str, object]:
    validate_task_id(task_id)
    if evidence_tier not in {DEVELOPMENT_EVIDENCE_TIER, CANDIDATE_EVIDENCE_TIER}:
        raise LabError(f"unsupported queued evidence tier: {evidence_tier}")
    snapshot = _portable_snapshot(snapshot_id, snapshot_identity)
    content: dict[str, object] = {
        "schema": RUN_REQUEST_SCHEMA_ID,
        "task_id": task_id,
        "snapshot": snapshot,
        "profile": {
            "profile_id": profile.profile_id,
            "sha256": profile.sha256,
            "risk_level": profile.risk_level,
            "command_ids": list(profile.command_ids),
        },
        "target_id": profile.target_id,
        "capabilities": list(profile.capabilities),
        "evidence_tier": evidence_tier,
        "control_plane": _validated_control_plane(dict(control_plane)),
    }
    idempotency_key = _sha256_json(content)
    request = {
        **content,
        "request_id": f"Q-{idempotency_key[:24]}",
        "idempotency_key": idempotency_key,
    }
    return validate_run_request(request)


def validate_run_request(payload: object) -> dict[str, object]:
    expected = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "snapshot",
        "profile",
        "target_id",
        "capabilities",
        "evidence_tier",
        "control_plane",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError("Run request has an unexpected shape")
    if payload.get("schema") != RUN_REQUEST_SCHEMA_ID:
        raise LabError("unsupported Run request schema")
    validate_task_id(str(payload.get("task_id", "")))
    validate_task_id(str(payload.get("target_id", "")))
    evidence_tier = payload.get("evidence_tier")
    if evidence_tier not in {DEVELOPMENT_EVIDENCE_TIER, CANDIDATE_EVIDENCE_TIER}:
        raise LabError("Run request has an invalid evidence tier")
    _validated_control_plane(payload.get("control_plane"))
    raw_profile = payload.get("profile")
    if not isinstance(raw_profile, dict) or set(raw_profile) != {
        "profile_id",
        "sha256",
        "risk_level",
        "command_ids",
    }:
        raise LabError("Run request Profile has an unexpected shape")
    validate_task_id(str(raw_profile.get("profile_id", "")))
    _validate_sha256(raw_profile.get("sha256"), label="Profile SHA-256")
    risk_level = raw_profile.get("risk_level")
    if risk_level not in {"R0", "R1", "R2", "R3", "R4", "R5"}:
        raise LabError("Run request Profile has an invalid risk level")
    command_ids = raw_profile.get("command_ids")
    if (
        not isinstance(command_ids, list)
        or not command_ids
        or len(command_ids) != len(set(command_ids))
    ):
        raise LabError("Run request command IDs are invalid")
    for command_id in command_ids:
        validate_task_id(str(command_id))
    capabilities = payload.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or len(capabilities) != len(set(capabilities))
    ):
        raise LabError("Run request capabilities are invalid")
    for capability in capabilities:
        validate_task_id(str(capability))
    raw_snapshot = payload.get("snapshot")
    if not isinstance(raw_snapshot, dict) or set(raw_snapshot) != {
        "snapshot_id",
        "repositories",
        "repository_links",
        "baseline",
    }:
        raise LabError("Run request snapshot has an unexpected shape")
    normalized_snapshot = _portable_snapshot(
        str(raw_snapshot.get("snapshot_id", "")), raw_snapshot
    )
    if normalized_snapshot != raw_snapshot:
        raise LabError("Run request snapshot is not canonical")
    supplied_key = _validate_sha256(
        payload.get("idempotency_key"), label="request idempotency key"
    )
    content = {key: value for key, value in payload.items() if key not in {"request_id", "idempotency_key"}}
    expected_key = _sha256_json(content)
    if supplied_key != expected_key:
        raise LabError("Run request idempotency hash mismatch")
    expected_id = f"Q-{expected_key[:24]}"
    if payload.get("request_id") != expected_id:
        raise LabError("Run request ID does not match its content")
    return payload


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise LabError(f"request evidence is missing or unsafe: {path}")
    try:
        if path.stat().st_size > _MAX_STATE_BYTES:
            raise LabError(f"request evidence exceeds its size limit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read request evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"request evidence must be an object: {path}")
    return payload


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise LabError(f"cannot create immutable request evidence {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _outcome(classification: str, detail: str) -> dict[str, str]:
    if classification not in _OUTCOME_CLASSIFICATIONS:
        raise LabError(f"invalid Run request outcome classification: {classification}")
    if not isinstance(detail, str) or not detail or len(detail) > 512:
        raise LabError("Run request outcome detail must contain 1-512 characters")
    return {"classification": classification, "detail": detail}


def _validate_outcome(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"classification", "detail"}:
        raise LabError("Run request outcome has an unexpected shape")
    classification = value.get("classification")
    detail = value.get("detail")
    if not isinstance(classification, str) or not isinstance(detail, str):
        raise LabError("Run request outcome fields must be strings")
    normalized = _outcome(classification, detail)
    if normalized != value:
        raise LabError("Run request outcome is not canonical")
    return normalized


def _validate_terminal_outcome(state: str, value: object) -> dict[str, str]:
    outcome = _validate_outcome(value)
    classification = outcome["classification"]
    if state == "COMPLETED" and classification != "PASSED":
        raise LabError("completed Run request must have a PASSED outcome")
    if state == "CANCELLED" and classification != "CANCELLED":
        raise LabError("cancelled Run request must have a CANCELLED outcome")
    if state == "QUARANTINED" and classification != "QUARANTINED":
        raise LabError("quarantined Run request must have a QUARANTINED outcome")
    if state == "FAILED" and classification in {"PASSED", "CANCELLED", "QUARANTINED"}:
        raise LabError("failed Run request has an invalid outcome classification")
    return outcome


class RunRequestStore:
    """Durable, transport-neutral request state; it never executes payload code."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.requests_root = self.root / "requests"
        self.locks_root = self.root / "locks"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.locks_root.mkdir(parents=True, exist_ok=True)
        path = self.locks_root / "request-store.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LabError("Run request store lock is held") from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _request_root(self, request_id: str) -> Path:
        return self.requests_root / validate_task_id(request_id)

    def _request_path(self, request_id: str) -> Path:
        return self._request_root(request_id) / "request.json"

    def _state_path(self, request_id: str) -> Path:
        return self._request_root(request_id) / "state.json"

    def _event_path(self, request_id: str, sequence: int, state: str) -> Path:
        return self._request_root(request_id) / "events" / f"{sequence:06d}-{state}.json"

    def _validate_request_directories(self, request_id: str) -> None:
        request_root = self._request_root(request_id)
        if request_root.is_symlink():
            raise LabError(f"Run request root must not be a symlink: {request_root}")
        event_root = request_root / "events"
        if event_root.is_symlink():
            raise LabError(f"Run request event root must not be a symlink: {event_root}")

    def _event(
        self,
        request: Mapping[str, object],
        *,
        sequence: int,
        previous_state: str | None,
        state: str,
        actor: str,
        slot_id: str | None,
        outcome: Mapping[str, str] | None,
        at: str | None = None,
    ) -> dict[str, object]:
        validate_task_id(actor)
        content: dict[str, object] = {
            "schema": REQUEST_EVENT_SCHEMA_ID,
            "request_id": request["request_id"],
            "idempotency_key": request["idempotency_key"],
            "sequence": sequence,
            "previous_state": previous_state,
            "state": state,
            "actor": actor,
            "at": at or _utc_now(),
            "slot_id": slot_id,
            "outcome": dict(outcome) if outcome is not None else None,
        }
        return {**content, "event_sha256": _sha256_json(content)}

    def _validate_event(
        self,
        event: object,
        *,
        request: Mapping[str, object],
        sequence: int,
    ) -> dict[str, object]:
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
        if not isinstance(event, dict) or set(event) != expected:
            raise LabError("Run request event has an unexpected shape")
        if (
            event.get("schema") != REQUEST_EVENT_SCHEMA_ID
            or event.get("request_id") != request.get("request_id")
            or event.get("idempotency_key") != request.get("idempotency_key")
            or event.get("sequence") != sequence
        ):
            raise LabError("Run request event identity mismatch")
        validate_task_id(str(event.get("actor", "")))
        state = event.get("state")
        if state not in _VALID_STATES:
            raise LabError("Run request event has an invalid state")
        _validate_timestamp(event.get("at"), label="Run request event timestamp")
        slot_id = event.get("slot_id")
        if slot_id is not None:
            validate_task_id(str(slot_id))
        outcome = event.get("outcome")
        if state in _TERMINAL_STATES:
            _validate_terminal_outcome(str(state), outcome)
        elif outcome is not None:
            raise LabError("non-terminal Run request event must not have an outcome")
        content = {key: value for key, value in event.items() if key != "event_sha256"}
        expected_hash = _sha256_json(content)
        if event.get("event_sha256") != expected_hash:
            raise LabError("Run request event hash mismatch")
        return event

    def submit(self, request: Mapping[str, object]) -> tuple[dict[str, object], bool]:
        validated = validate_run_request(dict(request))
        request_id = str(validated["request_id"])
        with self._lock():
            self._validate_request_directories(request_id)
            request_path = self._request_path(request_id)
            if request_path.exists():
                existing_request = validate_run_request(_read_json(request_path))
                if existing_request != validated:
                    raise LabError("Run request ID collision with different content")
            else:
                _write_new_json(request_path, validated)
            state_path = self._state_path(request_id)
            if state_path.exists():
                return self.load(request_id), False
            event = self._event(
                validated,
                sequence=0,
                previous_state=None,
                state="SUBMITTED",
                actor="control-plane",
                slot_id=None,
                outcome=None,
            )
            event_path = self._event_path(request_id, 0, "SUBMITTED")
            if event_path.exists():
                event = self._validate_event(
                    _read_json(event_path), request=validated, sequence=0
                )
            else:
                _write_new_json(event_path, event)
            validated_snapshot = validated["snapshot"]
            validated_profile = validated["profile"]
            if not isinstance(validated_snapshot, dict) or not isinstance(
                validated_profile, dict
            ):
                raise LabError("validated Run request lost its nested identity")
            state: dict[str, object] = {
                "schema": REQUEST_STATE_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": validated["idempotency_key"],
                "task_id": validated["task_id"],
                "snapshot_id": validated_snapshot["snapshot_id"],
                "profile_id": validated_profile["profile_id"],
                "target_id": validated["target_id"],
                "evidence_tier": validated["evidence_tier"],
                "state": "SUBMITTED",
                "slot_id": None,
                "outcome": None,
                "submitted_at": event["at"],
                "updated_at": event["at"],
                "history": [
                    {
                        "sequence": 0,
                        "state": "SUBMITTED",
                        "at": event["at"],
                        "event_sha256": event["event_sha256"],
                    }
                ],
            }
            _atomic_write_text(
                state_path, json.dumps(state, indent=2, sort_keys=True) + "\n"
            )
            return self.load(request_id), True

    def load(self, request_id: str) -> dict[str, object]:
        self._validate_request_directories(request_id)
        request = validate_run_request(_read_json(self._request_path(request_id)))
        state = _read_json(self._state_path(request_id))
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
        if set(state) != expected or state.get("schema") != REQUEST_STATE_SCHEMA_ID:
            raise LabError("Run request state has an unexpected shape")
        snapshot = request["snapshot"]
        profile = request["profile"]
        if not isinstance(snapshot, dict) or not isinstance(profile, dict):
            raise LabError("validated Run request lost its nested identity")
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
            raise LabError("Run request state identity mismatch")
        history = state.get("history")
        if not isinstance(history, list) or not history:
            raise LabError("Run request state history is empty")
        previous_state: str | None = None
        previous_at: str | None = None
        previous_timestamp: datetime | None = None
        effective_slot: str | None = None
        final_outcome: object = None
        for sequence, summary in enumerate(history):
            if not isinstance(summary, dict) or set(summary) != {
                "sequence",
                "state",
                "at",
                "event_sha256",
            }:
                raise LabError("Run request history has an invalid entry")
            if summary.get("sequence") != sequence:
                raise LabError("Run request history sequence is not contiguous")
            observed_state = str(summary.get("state", ""))
            if sequence == 0:
                if observed_state != "SUBMITTED":
                    raise LabError("Run request history must start at SUBMITTED")
            elif observed_state not in _TRANSITIONS.get(previous_state or "", frozenset()):
                raise LabError(
                    f"Run request history has invalid transition: "
                    f"{previous_state} -> {observed_state}"
                )
            event = self._validate_event(
                _read_json(self._event_path(request_id, sequence, observed_state)),
                request=request,
                sequence=sequence,
            )
            if (
                event.get("previous_state") != previous_state
                or event.get("state") != observed_state
                or event.get("at") != summary.get("at")
                or event.get("event_sha256") != summary.get("event_sha256")
            ):
                raise LabError("Run request event differs from state history")
            event_at = _validate_timestamp(
                event.get("at"), label="Run request event timestamp"
            )
            event_timestamp = datetime.fromisoformat(event_at)
            if previous_timestamp is not None and event_timestamp < previous_timestamp:
                raise LabError("Run request event timestamps are not monotonic")
            event_slot = event.get("slot_id")
            if observed_state == "SUBMITTED" and event_slot is not None:
                raise LabError("submitted Run request must not hold a slot")
            if observed_state == "RESERVED" and event_slot is None:
                raise LabError("reserved Run request must hold a slot")
            if effective_slot is not None and event_slot != effective_slot:
                raise LabError("Run request slot changed after reservation")
            if event_slot is not None:
                effective_slot = str(event_slot)
            previous_state = observed_state
            previous_at = event_at
            previous_timestamp = event_timestamp
            final_outcome = event.get("outcome")
        if state.get("state") != previous_state or state.get("updated_at") != previous_at:
            raise LabError("Run request state differs from its final event")
        if state.get("submitted_at") != history[0].get("at"):
            raise LabError("Run request submitted timestamp differs from its first event")
        if state.get("slot_id") != effective_slot:
            raise LabError("Run request state slot differs from its event history")
        if state.get("outcome") != final_outcome:
            raise LabError("Run request state outcome differs from its final event")
        if state.get("state") in _TERMINAL_STATES:
            _validate_terminal_outcome(str(state.get("state")), state.get("outcome"))
        elif state.get("outcome") is not None:
            raise LabError("active Run request state must not have an outcome")
        return state

    def list(self, task_id: str | None = None) -> list[dict[str, object]]:
        if task_id is not None:
            validate_task_id(task_id)
        if not self.requests_root.is_dir():
            return []
        states = []
        for path in sorted(self.requests_root.iterdir()):
            if not path.is_dir() or path.is_symlink():
                continue
            state = self.load(path.name)
            if task_id is None or state.get("task_id") == task_id:
                states.append(state)
        return sorted(states, key=lambda value: str(value.get("submitted_at", "")), reverse=True)

    def _advance(
        self,
        request_id: str,
        new_state: str,
        *,
        actor: str,
        slot_id: str | None = None,
        outcome: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        validate_task_id(actor)
        with self._lock():
            current = self.load(request_id)
            current_state = str(current["state"])
            if current_state == new_state:
                if slot_id is not None and current.get("slot_id") != slot_id:
                    raise LabError("repeated Run request transition changed its slot")
                if outcome is not None and current.get("outcome") != dict(outcome):
                    raise LabError("repeated Run request transition changed its outcome")
                return current
            if new_state not in _TRANSITIONS.get(current_state, frozenset()):
                raise LabError(
                    f"invalid Run request transition: {current_state} -> {new_state}"
                )
            effective_slot = current.get("slot_id")
            if new_state == "RESERVED":
                if slot_id is None:
                    raise LabError("Run request reservation requires a slot ID")
                effective_slot = validate_task_id(slot_id)
                for other in self.list():
                    if (
                        other.get("request_id") != request_id
                        and other.get("slot_id") == effective_slot
                        and other.get("state") in _ACTIVE_STATES
                    ):
                        raise LabError(
                            f"Run slot {effective_slot} is held by "
                            f"{other.get('request_id')}"
                        )
            elif slot_id is not None and slot_id != effective_slot:
                raise LabError("Run request slot cannot change after reservation")
            effective_outcome = dict(outcome) if outcome is not None else None
            if new_state in _TERMINAL_STATES and effective_outcome is None:
                raise LabError("terminal Run request transition requires an outcome")
            if new_state not in _TERMINAL_STATES and effective_outcome is not None:
                raise LabError("non-terminal Run request transition rejects an outcome")
            request = validate_run_request(_read_json(self._request_path(request_id)))
            history = current["history"]
            assert isinstance(history, list)
            sequence = len(history)
            event = self._event(
                request,
                sequence=sequence,
                previous_state=current_state,
                state=new_state,
                actor=actor,
                slot_id=str(effective_slot) if effective_slot is not None else None,
                outcome=effective_outcome,
            )
            event_path = self._event_path(request_id, sequence, new_state)
            if event_path.exists():
                existing = self._validate_event(
                    _read_json(event_path), request=request, sequence=sequence
                )
                comparable = {"previous_state", "state", "actor", "slot_id", "outcome"}
                if any(existing.get(key) != event.get(key) for key in comparable):
                    raise LabError("Run request event sequence collision")
                event = existing
            else:
                _write_new_json(event_path, event)
            updated = {
                **current,
                "state": new_state,
                "slot_id": effective_slot,
                "outcome": effective_outcome,
                "updated_at": event["at"],
                "history": [
                    *history,
                    {
                        "sequence": sequence,
                        "state": new_state,
                        "at": event["at"],
                        "event_sha256": event["event_sha256"],
                    },
                ],
            }
            _atomic_write_text(
                self._state_path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return self.load(request_id)

    def reserve(self, request_id: str, *, slot_id: str, actor: str) -> dict[str, object]:
        return self._advance(request_id, "RESERVED", actor=actor, slot_id=slot_id)

    def begin_preparing(self, request_id: str, *, actor: str) -> dict[str, object]:
        return self._advance(request_id, "PREPARING", actor=actor)

    def begin_running(self, request_id: str, *, actor: str) -> dict[str, object]:
        return self._advance(request_id, "RUNNING", actor=actor)

    def begin_collecting(self, request_id: str, *, actor: str) -> dict[str, object]:
        return self._advance(request_id, "COLLECTING", actor=actor)

    def complete(self, request_id: str, *, actor: str, detail: str) -> dict[str, object]:
        return self._advance(
            request_id,
            "COMPLETED",
            actor=actor,
            outcome=_outcome("PASSED", detail),
        )

    def fail(
        self,
        request_id: str,
        *,
        actor: str,
        classification: str,
        detail: str,
    ) -> dict[str, object]:
        if classification in {"PASSED", "CANCELLED", "QUARANTINED"}:
            raise LabError(f"invalid failure classification: {classification}")
        return self._advance(
            request_id,
            "FAILED",
            actor=actor,
            outcome=_outcome(classification, detail),
        )

    def quarantine(self, request_id: str, *, actor: str, detail: str) -> dict[str, object]:
        return self._advance(
            request_id,
            "QUARANTINED",
            actor=actor,
            outcome=_outcome("QUARANTINED", detail),
        )

    def cancel(self, request_id: str, *, actor: str, reason: str) -> dict[str, object]:
        if not reason or len(reason) > 512:
            raise LabError("cancel reason must contain 1-512 characters")
        current = self.load(request_id)
        state = str(current["state"])
        if state in _TERMINAL_STATES or state == "CANCEL_REQUESTED":
            return current
        if state == "SUBMITTED":
            return self._advance(
                request_id,
                "CANCELLED",
                actor=actor,
                outcome=_outcome("CANCELLED", reason),
            )
        return self._advance(request_id, "CANCEL_REQUESTED", actor=actor)

    def acknowledge_cancel(
        self, request_id: str, *, actor: str, detail: str
    ) -> dict[str, object]:
        return self._advance(
            request_id,
            "CANCELLED",
            actor=actor,
            outcome=_outcome("CANCELLED", detail),
        )

    def watch(
        self,
        request_id: str,
        *,
        after_sequence: int = -1,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.25,
    ) -> dict[str, object]:
        if after_sequence < -1:
            raise LabError("watch sequence must be at least -1")
        if not 0 <= timeout_seconds <= 60:
            raise LabError("watch timeout must be between 0 and 60 seconds")
        if not 0.01 <= poll_interval_seconds <= 5:
            raise LabError("watch poll interval must be between 0.01 and 5 seconds")
        deadline = time.monotonic() + timeout_seconds
        while True:
            current = self.load(request_id)
            history = current["history"]
            assert isinstance(history, list)
            observed_sequence = len(history) - 1
            terminal = current.get("state") in _TERMINAL_STATES
            changed = observed_sequence > after_sequence
            if changed or terminal or time.monotonic() >= deadline:
                return {
                    "schema": "lab.run-request-watch.v1",
                    "request_id": request_id,
                    "after_sequence": after_sequence,
                    "observed_sequence": observed_sequence,
                    "changed": changed,
                    "terminal": terminal,
                    "request": current,
                }
            time.sleep(min(poll_interval_seconds, max(deadline - time.monotonic(), 0)))


class RunRequestManager:
    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self.workspace_manager = workspace_manager
        configured_profiles = workspace_manager.environ.get("LAB_QUEUE_PROFILE_ROOT")
        self.profile_root = (
            Path(configured_profiles).expanduser().resolve()
            if configured_profiles
            else workspace_manager.control_repository / "integration" / "queue-profiles"
        )
        configured_baseline = workspace_manager.environ.get("LAB_BASELINE_LOCK")
        self.baseline_path = (
            Path(configured_baseline).expanduser().resolve()
            if configured_baseline
            else workspace_manager.control_repository / "stack" / "baseline.json"
        )
        configured_state = workspace_manager.environ.get("LAB_REQUEST_STATE_ROOT")
        state_root = (
            Path(configured_state).expanduser().resolve()
            if configured_state
            else workspace_manager.state_root / "run-requests"
        )
        self.store = RunRequestStore(state_root)

    def submit_clean_profile(
        self,
        task_id: str,
        profile_id: str,
        *,
        evidence_tier: str,
    ) -> dict[str, object]:
        RunManager(self.workspace_manager).validate_evidence_tier(evidence_tier)
        profile = load_queue_profile(self.profile_root, profile_id)
        snapshot_id, snapshot_identity = resolve_clean_snapshot_identity(
            self.workspace_manager,
            task_id=task_id,
            profile_id=profile.profile_id,
            profile_sha256=profile.sha256,
            repositories=profile.repositories,
            baseline_path=self.baseline_path,
        )
        request = build_run_request(
            task_id=task_id,
            snapshot_id=snapshot_id,
            snapshot_identity=snapshot_identity,
            profile=profile,
            evidence_tier=evidence_tier,
            control_plane=control_plane_identity(
                self.workspace_manager.control_repository
            ),
        )
        state, created = self.store.submit(request)
        return {
            "schema": REQUEST_SUBMISSION_SCHEMA_ID,
            "created": created,
            "request": request,
            "state": state,
        }
