from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .queue_profiles import ALLOWED_RISK_LEVELS, load_queue_profile
from .run_requests import (
    RunRequestStore,
    validate_control_plane_identity,
    validate_run_request,
)
from .runs import CANDIDATE_EVIDENCE_TIER, DEVELOPMENT_EVIDENCE_TIER
from .workspace import LabError, validate_task_id

RUNNER_PROFILE_SCHEMA_ID = "lab.runner-profile.v1"
RUNNER_CLAIM_SCHEMA_ID = "lab.runner-claim.v1"
RUNNER_INSPECTION_SCHEMA_ID = "lab.runner-inspection.v1"
ARTIFACT_RECEIPT_SCHEMA_ID = "lab.artifact-receipt.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MEDIA_TYPE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}"
)
_MAX_PROFILE_BYTES = 256 * 1024
_MAX_ARTIFACTS = 256
_MAX_ARTIFACT_BYTES = 2**63 - 1
_CLAIM_ACTIVE_STATES = frozenset(
    {"RESERVED", "PREPARING", "RUNNING", "COLLECTING", "CANCEL_REQUESTED"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


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


def _string_ids(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise LabError(f"{label} must be a unique non-empty string list")
    return tuple(validate_task_id(item) for item in value)


@dataclass(frozen=True)
class RunnerProfile:
    runner_id: str
    description: str
    target_ids: tuple[str, ...]
    slots: tuple[str, ...]
    capabilities: tuple[str, ...]
    accepted_risk_levels: tuple[str, ...]
    accepted_evidence_tiers: tuple[str, ...]
    path: Path
    sha256: str

    def public_identity(self) -> dict[str, object]:
        return {
            "runner_id": self.runner_id,
            "profile_sha256": self.sha256,
            "target_ids": list(self.target_ids),
            "slots": list(self.slots),
            "capabilities": list(self.capabilities),
            "accepted_risk_levels": list(self.accepted_risk_levels),
            "accepted_evidence_tiers": list(self.accepted_evidence_tiers),
        }


def load_runner_profile(profile_root: Path, runner_id: str) -> RunnerProfile:
    validate_task_id(runner_id)
    root = profile_root.expanduser().resolve()
    candidate = root / f"{runner_id}.json"
    if candidate.parent != root or candidate.is_symlink() or not candidate.is_file():
        raise LabError(f"unknown or unsafe runner Profile: {runner_id} ({candidate})")
    path = candidate.resolve()
    if path.parent != root:
        raise LabError(f"runner Profile path escapes trusted root: {runner_id}")
    try:
        if path.stat().st_size > _MAX_PROFILE_BYTES:
            raise LabError(f"runner Profile exceeds its size limit: {path}")
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read runner Profile {path}: {exc}") from exc
    expected = {
        "schema",
        "runner_id",
        "description",
        "target_ids",
        "slots",
        "capabilities",
        "accepted_risk_levels",
        "accepted_evidence_tiers",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LabError(f"runner Profile has an unexpected shape: {path}")
    if payload.get("schema") != RUNNER_PROFILE_SCHEMA_ID:
        raise LabError(f"unsupported runner Profile schema: {path}")
    if payload.get("runner_id") != runner_id:
        raise LabError(f"runner Profile identity mismatch: {path}")
    description = payload.get("description")
    if not isinstance(description, str) or not description or len(description) > 512:
        raise LabError(f"runner Profile has an invalid description: {path}")
    target_ids = _string_ids(payload.get("target_ids"), label="runner targets")
    slots = _string_ids(payload.get("slots"), label="runner slots")
    capabilities = _string_ids(
        payload.get("capabilities"), label="runner capabilities"
    )
    raw_risks = _string_ids(
        payload.get("accepted_risk_levels"), label="accepted risk levels"
    )
    if not set(raw_risks).issubset(ALLOWED_RISK_LEVELS):
        raise LabError(f"runner Profile has an invalid risk level: {path}")
    evidence_tiers = _string_ids(
        payload.get("accepted_evidence_tiers"),
        label="accepted evidence tiers",
    )
    if not set(evidence_tiers).issubset(
        {DEVELOPMENT_EVIDENCE_TIER, CANDIDATE_EVIDENCE_TIER}
    ):
        raise LabError(f"runner Profile has an invalid evidence tier: {path}")
    return RunnerProfile(
        runner_id=runner_id,
        description=description,
        target_ids=target_ids,
        slots=slots,
        capabilities=capabilities,
        accepted_risk_levels=raw_risks,
        accepted_evidence_tiers=evidence_tiers,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _request_compatibility(
    request: Mapping[str, object],
    *,
    runner: RunnerProfile,
    queue_profile_root: Path,
) -> list[str]:
    validated = validate_run_request(dict(request))
    raw_profile = validated["profile"]
    raw_snapshot = validated["snapshot"]
    if not isinstance(raw_profile, dict) or not isinstance(raw_snapshot, dict):
        raise LabError("validated Run request lost its nested identity")
    profile_id = str(raw_profile["profile_id"])
    try:
        profile = load_queue_profile(queue_profile_root, profile_id)
    except LabError as exc:
        return [f"trusted queue Profile unavailable: {exc}"]

    reasons: list[str] = []
    if raw_profile.get("sha256") != profile.sha256:
        reasons.append("queue Profile hash differs from the trusted runner copy")
    if validated.get("target_id") != profile.target_id:
        reasons.append("request target differs from the trusted queue Profile")
    if raw_profile.get("risk_level") != profile.risk_level:
        reasons.append("request risk differs from the trusted queue Profile")
    if raw_profile.get("command_ids") != list(profile.command_ids):
        reasons.append("request commands differ from the trusted queue Profile")
    if validated.get("capabilities") != list(profile.capabilities):
        reasons.append("request capabilities differ from the trusted queue Profile")
    raw_repositories = raw_snapshot.get("repositories")
    if not isinstance(raw_repositories, dict) or set(raw_repositories) != set(
        profile.repositories
    ):
        reasons.append("request repositories differ from the trusted queue Profile")
    if profile.target_id not in runner.target_ids:
        reasons.append("runner does not serve the requested target")
    missing_capabilities = sorted(set(profile.capabilities) - set(runner.capabilities))
    if missing_capabilities:
        reasons.append("runner lacks capabilities: " + ", ".join(missing_capabilities))
    if profile.risk_level not in runner.accepted_risk_levels:
        reasons.append(f"runner does not accept risk level {profile.risk_level}")
    evidence_tier = str(validated.get("evidence_tier", ""))
    if evidence_tier not in runner.accepted_evidence_tiers:
        reasons.append(f"runner does not accept {evidence_tier} evidence")
    return reasons


def _validate_relative_artifact_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or value.startswith(("/", "~"))
    ):
        raise LabError(f"invalid artifact relative path: {value!r}")
    path = PurePosixPath(value)
    if value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(f"invalid artifact relative path: {value!r}")
    return value


def _validated_artifacts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) > _MAX_ARTIFACTS:
        raise LabError(f"artifact receipt supports at most {_MAX_ARTIFACTS} entries")
    artifacts: list[dict[str, object]] = []
    artifact_ids: set[str] = set()
    relative_paths: set[str] = set()
    for raw_artifact in value:
        expected = {
            "artifact_id",
            "relative_path",
            "media_type",
            "byte_size",
            "sha256",
            "storage_id",
        }
        if not isinstance(raw_artifact, dict) or set(raw_artifact) != expected:
            raise LabError("artifact descriptor has an unexpected shape")
        artifact_id = validate_task_id(str(raw_artifact.get("artifact_id", "")))
        relative_path = _validate_relative_artifact_path(
            raw_artifact.get("relative_path")
        )
        media_type = raw_artifact.get("media_type")
        if not isinstance(media_type, str) or not _MEDIA_TYPE_PATTERN.fullmatch(media_type):
            raise LabError(f"invalid artifact media type: {media_type!r}")
        byte_size = raw_artifact.get("byte_size")
        if (
            not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or not 0 <= byte_size <= _MAX_ARTIFACT_BYTES
        ):
            raise LabError(f"invalid artifact byte size: {byte_size!r}")
        sha256 = raw_artifact.get("sha256")
        if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
            raise LabError(f"invalid artifact SHA-256: {sha256!r}")
        storage_id = validate_task_id(str(raw_artifact.get("storage_id", "")))
        if artifact_id in artifact_ids or relative_path in relative_paths:
            raise LabError("artifact receipt contains duplicate identities or paths")
        artifact_ids.add(artifact_id)
        relative_paths.add(relative_path)
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "relative_path": relative_path,
                "media_type": media_type,
                "byte_size": byte_size,
                "sha256": sha256,
                "storage_id": storage_id,
            }
        )
    return artifacts


class RunnerCoordinator:
    """Validate and claim requests without executing their project payload."""

    def __init__(
        self,
        store: RunRequestStore,
        *,
        runner_profile_root: Path,
        queue_profile_root: Path,
        control_plane: Mapping[str, object],
    ) -> None:
        self.store = store
        self.runner_profile_root = runner_profile_root.expanduser().resolve()
        self.queue_profile_root = queue_profile_root.expanduser().resolve()
        self.control_plane = validate_control_plane_identity(dict(control_plane))
        self.lock_root = store.root / "runner-locks"

    @contextmanager
    def _runner_lock(self, runner_id: str) -> Iterator[None]:
        validate_task_id(runner_id)
        self.lock_root.mkdir(parents=True, exist_ok=True)
        if self.lock_root.is_symlink():
            raise LabError(f"runner lock root must not be a symlink: {self.lock_root}")
        path = self.lock_root / f"{runner_id}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LabError(f"runner claim loop is already active: {runner_id}") from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def inspect(self, runner_id: str) -> dict[str, object]:
        runner = load_runner_profile(self.runner_profile_root, runner_id)
        pending: list[dict[str, object]] = []
        for state in self.store.list():
            if state.get("state") != "SUBMITTED":
                continue
            request_id = str(state["request_id"])
            request = self.store.load_request(request_id)
            reasons = _request_compatibility(
                request,
                runner=runner,
                queue_profile_root=self.queue_profile_root,
            )
            pending.append(
                {
                    "request_id": request_id,
                    "task_id": state["task_id"],
                    "profile_id": state["profile_id"],
                    "submitted_at": state["submitted_at"],
                    "eligible": not reasons,
                    "reasons": reasons,
                }
            )
        return {
            "schema": RUNNER_INSPECTION_SCHEMA_ID,
            "runner": runner.public_identity(),
            "pending": pending,
            "eligible_count": sum(item["eligible"] is True for item in pending),
        }

    def _claim_document(
        self,
        *,
        request: Mapping[str, object],
        state: Mapping[str, object],
        runner: RunnerProfile,
        slot_id: str,
    ) -> dict[str, object]:
        capabilities = request.get("capabilities")
        if not isinstance(capabilities, list):
            raise LabError("validated Run request lost its capabilities")
        content: dict[str, object] = {
            "schema": RUNNER_CLAIM_SCHEMA_ID,
            "request_id": request["request_id"],
            "idempotency_key": request["idempotency_key"],
            "runner_id": runner.runner_id,
            "runner_profile_sha256": runner.sha256,
            "runner_control_plane": self.control_plane,
            "slot_id": slot_id,
            "target_id": request["target_id"],
            "capabilities": list(capabilities),
            "claimed_at": state["updated_at"],
        }
        receipt_sha256 = _sha256_json(content)
        return {
            **content,
            "claim_id": f"C-{receipt_sha256[:24]}",
            "receipt_sha256": receipt_sha256,
        }

    def claim_next(self, runner_id: str, slot_id: str) -> dict[str, object] | None:
        runner = load_runner_profile(self.runner_profile_root, runner_id)
        validate_task_id(slot_id)
        if slot_id not in runner.slots:
            raise LabError(f"slot {slot_id} does not belong to runner {runner_id}")
        with self._runner_lock(runner_id):
            for state in self.store.list():
                if (
                    state.get("slot_id") != slot_id
                    or state.get("state") not in _CLAIM_ACTIVE_STATES
                ):
                    continue
                request_id = str(state["request_id"])
                if self.store.has_attachment(request_id, "runner-claim"):
                    raw_claim = self.store.load_attachment(
                        request_id, "runner-claim"
                    )
                    if raw_claim.get("runner_id") != runner_id:
                        raise LabError(
                            f"slot {slot_id} is held by another runner claim"
                        )
                    claim, _, _ = self._validated_claim(request_id, runner_id)
                    return claim
                latest = self.store.latest_event(request_id)
                request = self.store.load_request(request_id)
                reasons = _request_compatibility(
                    request,
                    runner=runner,
                    queue_profile_root=self.queue_profile_root,
                )
                if (
                    state.get("state") != "RESERVED"
                    or latest.get("actor") != runner_id
                    or reasons
                ):
                    raise LabError(
                        f"slot {slot_id} has an unrecoverable reservation without a claim"
                    )
                recovered = self._claim_document(
                    request=request,
                    state=state,
                    runner=runner,
                    slot_id=slot_id,
                )
                persisted, _ = self.store.put_attachment(
                    request_id, "runner-claim", recovered
                )
                return persisted
            inspection = self.inspect(runner_id)
            pending = inspection["pending"]
            assert isinstance(pending, list)
            candidates = sorted(
                (item for item in pending if item.get("eligible") is True),
                key=lambda item: (str(item.get("submitted_at", "")), str(item["request_id"])),
            )
            for candidate in candidates:
                request_id = str(candidate["request_id"])
                try:
                    state = self.store.reserve(
                        request_id,
                        slot_id=slot_id,
                        actor=runner.runner_id,
                    )
                except LabError:
                    if self.store.load(request_id).get("state") != "SUBMITTED":
                        continue
                    raise
                request = self.store.load_request(request_id)
                claim = self._claim_document(
                    request=request,
                    state=state,
                    runner=runner,
                    slot_id=slot_id,
                )
                persisted, _ = self.store.put_attachment(
                    request_id, "runner-claim", claim
                )
                return persisted
        return None

    def _validated_claim(
        self, request_id: str, runner_id: str
    ) -> tuple[dict[str, object], dict[str, object], RunnerProfile]:
        runner = load_runner_profile(self.runner_profile_root, runner_id)
        state = self.store.load(request_id)
        request = self.store.load_request(request_id)
        claim = self.store.load_attachment(request_id, "runner-claim")
        expected = {
            "schema",
            "request_id",
            "idempotency_key",
            "runner_id",
            "runner_profile_sha256",
            "runner_control_plane",
            "slot_id",
            "target_id",
            "capabilities",
            "claimed_at",
            "claim_id",
            "receipt_sha256",
        }
        if not isinstance(claim, dict) or set(claim) != expected:
            raise LabError("runner claim has an unexpected shape")
        if (
            claim.get("schema") != RUNNER_CLAIM_SCHEMA_ID
            or claim.get("request_id") != request_id
            or claim.get("runner_id") != runner_id
            or claim.get("runner_profile_sha256") != runner.sha256
            or claim.get("runner_control_plane") != self.control_plane
            or claim.get("slot_id") != state.get("slot_id")
            or claim.get("idempotency_key") != request.get("idempotency_key")
            or claim.get("target_id") != request.get("target_id")
            or claim.get("capabilities") != request.get("capabilities")
        ):
            raise LabError("runner claim identity mismatch")
        claimed_at = _validate_timestamp(
            claim.get("claimed_at"), label="runner claim timestamp"
        )
        history = state.get("history")
        if not isinstance(history, list):
            raise LabError("Run request state has no claim history")
        reservation = next(
            (
                entry
                for entry in history
                if isinstance(entry, dict) and entry.get("state") == "RESERVED"
            ),
            None,
        )
        if not isinstance(reservation, dict) or reservation.get("at") != claimed_at:
            raise LabError("runner claim timestamp differs from reservation")
        reasons = _request_compatibility(
            request,
            runner=runner,
            queue_profile_root=self.queue_profile_root,
        )
        if reasons:
            raise LabError("runner claim is no longer authorized: " + "; ".join(reasons))
        content = {
            key: value
            for key, value in claim.items()
            if key not in {"claim_id", "receipt_sha256"}
        }
        receipt_sha256 = _sha256_json(content)
        if (
            claim.get("receipt_sha256") != receipt_sha256
            or claim.get("claim_id") != f"C-{receipt_sha256[:24]}"
        ):
            raise LabError("runner claim hash mismatch")
        return claim, state, runner

    def validated_claim(
        self, request_id: str, runner_id: str
    ) -> tuple[dict[str, object], dict[str, object], RunnerProfile]:
        """Expose a revalidated claim to trusted runner adapters."""

        return self._validated_claim(request_id, runner_id)

    def _validated_artifact_receipt(
        self,
        request_id: str,
        claim: Mapping[str, object],
        runner: RunnerProfile,
    ) -> dict[str, object]:
        receipt = self.store.load_attachment(request_id, "artifact-receipt")
        expected = {
            "schema",
            "request_id",
            "idempotency_key",
            "claim_id",
            "runner_id",
            "runner_profile_sha256",
            "slot_id",
            "artifacts",
            "evidence_complete",
            "recorded_at",
            "receipt_sha256",
        }
        if set(receipt) != expected or receipt.get("schema") != ARTIFACT_RECEIPT_SCHEMA_ID:
            raise LabError("artifact receipt has an unexpected shape")
        if (
            receipt.get("request_id") != request_id
            or receipt.get("idempotency_key") != claim.get("idempotency_key")
            or receipt.get("claim_id") != claim.get("claim_id")
            or receipt.get("runner_id") != runner.runner_id
            or receipt.get("runner_profile_sha256") != runner.sha256
            or receipt.get("slot_id") != claim.get("slot_id")
            or not isinstance(receipt.get("evidence_complete"), bool)
        ):
            raise LabError("artifact receipt identity mismatch")
        normalized_artifacts = _validated_artifacts(receipt.get("artifacts"))
        if normalized_artifacts != receipt.get("artifacts"):
            raise LabError("artifact receipt is not canonical")
        _validate_timestamp(
            receipt.get("recorded_at"), label="artifact receipt timestamp"
        )
        content = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _sha256_json(content):
            raise LabError("artifact receipt hash mismatch")
        return receipt

    def begin_preparing(self, request_id: str, runner_id: str) -> dict[str, object]:
        self._validated_claim(request_id, runner_id)
        return self.store.begin_preparing(request_id, actor=runner_id)

    def begin_running(self, request_id: str, runner_id: str) -> dict[str, object]:
        self._validated_claim(request_id, runner_id)
        return self.store.begin_running(request_id, actor=runner_id)

    def begin_collecting(self, request_id: str, runner_id: str) -> dict[str, object]:
        self._validated_claim(request_id, runner_id)
        return self.store.begin_collecting(request_id, actor=runner_id)

    def record_artifact_receipt(
        self,
        request_id: str,
        runner_id: str,
        *,
        artifacts: Sequence[Mapping[str, object]],
        evidence_complete: bool,
    ) -> dict[str, object]:
        claim, state, runner = self._validated_claim(request_id, runner_id)
        if state.get("state") != "COLLECTING":
            raise LabError("artifact receipt requires a COLLECTING request")
        if not isinstance(evidence_complete, bool):
            raise LabError("artifact evidence_complete must be a boolean")
        normalized_artifacts = _validated_artifacts(list(artifacts))
        if self.store.has_attachment(request_id, "artifact-receipt"):
            existing = self._validated_artifact_receipt(
                request_id, claim=claim, runner=runner
            )
            if (
                existing.get("artifacts") != normalized_artifacts
                or existing.get("evidence_complete") != evidence_complete
            ):
                raise LabError("repeated artifact receipt changed its content")
            return existing
        content: dict[str, object] = {
            "schema": ARTIFACT_RECEIPT_SCHEMA_ID,
            "request_id": request_id,
            "idempotency_key": claim["idempotency_key"],
            "claim_id": claim["claim_id"],
            "runner_id": runner.runner_id,
            "runner_profile_sha256": runner.sha256,
            "slot_id": claim["slot_id"],
            "artifacts": normalized_artifacts,
            "evidence_complete": evidence_complete,
            "recorded_at": _utc_now(),
        }
        receipt = {**content, "receipt_sha256": _sha256_json(content)}
        self.store.put_attachment(request_id, "artifact-receipt", receipt)
        return self._validated_artifact_receipt(
            request_id, claim=claim, runner=runner
        )

    def complete(
        self, request_id: str, runner_id: str, *, detail: str
    ) -> dict[str, object]:
        claim, _, runner = self._validated_claim(request_id, runner_id)
        artifact_receipt = self._validated_artifact_receipt(
            request_id, claim=claim, runner=runner
        )
        if artifact_receipt.get("evidence_complete") is not True:
            raise LabError("Run request cannot complete with incomplete artifact evidence")
        return self.store.complete(request_id, actor=runner_id, detail=detail)

    def acknowledge_cancel(
        self, request_id: str, runner_id: str, *, detail: str
    ) -> dict[str, object]:
        self._validated_claim(request_id, runner_id)
        return self.store.acknowledge_cancel(
            request_id, actor=runner_id, detail=detail
        )
