"""Operator-facing lifecycle for passive Quest paired-pose calibration capture.

This concrete Control Station leaf translates typed operator intents into the
existing deploy-owned capture application.  It owns UI revision/admission
state and a pose-redacted status projection only.  Source production, capture
reduction, evidence construction, artifact publication, fitting, acceptance,
frame installation, rendering, and device lifecycle remain lower-owned.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.deploy.quest_paired_pose_capture import (
    QuestCaptureFrameFacts,
    QuestCaptureSelectionIntent,
    QuestPairedPoseCaptureCleanupError,
    QuestPairedPoseCaptureTerminalError,
    StaticWindowRejectedError,
)

QUEST_CALIBRATION_INTENT_SCHEMA_V1 = (
    "dexteleop.control_station.quest_calibration_intent.v1"
)
QUEST_CALIBRATION_INTENT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.quest_calibration_intent.v1"
)
QUEST_CALIBRATION_STATUS_SCHEMA_V1 = (
    "dexteleop.control_station.quest_calibration_status.v1"
)
QUEST_CALIBRATION_INTENT_KINDS_V1 = frozenset(
    {
        "select_source",
        "begin_window",
        "complete_window",
        "seal",
        "cancel",
    }
)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _optional_text(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class QuestCalibrationIntent:
    """One immutable operator request without pose or matrix payloads."""

    client_id: str
    client_sequence: int
    expected_revision: int
    intent_kind: str
    candidate_hash: str | None = None
    robot_side: str | None = None
    reference_frame_id: str | None = None
    moving_frame_id: str | None = None
    xr_frame_clock_domain_id: str | None = None
    window_id: str | None = None
    reason: str = "operator_request"

    def __post_init__(self) -> None:
        client_id = _required_text(self.client_id, name="client_id")
        client_sequence = _non_negative_int(
            self.client_sequence,
            name="client_sequence",
        )
        expected_revision = _non_negative_int(
            self.expected_revision,
            name="expected_revision",
        )
        intent_kind = _required_text(self.intent_kind, name="intent_kind")
        if intent_kind not in QUEST_CALIBRATION_INTENT_KINDS_V1:
            raise ValueError(f"unsupported Quest calibration intent: {intent_kind}")

        candidate_hash = _optional_text(
            self.candidate_hash,
            name="candidate_hash",
        )
        robot_side = _optional_text(self.robot_side, name="robot_side")
        reference_frame_id = _optional_text(
            self.reference_frame_id,
            name="reference_frame_id",
        )
        moving_frame_id = _optional_text(
            self.moving_frame_id,
            name="moving_frame_id",
        )
        xr_frame_clock_domain_id = _optional_text(
            self.xr_frame_clock_domain_id,
            name="xr_frame_clock_domain_id",
        )
        window_id = _optional_text(self.window_id, name="window_id")

        selection_fields = {
            "candidate_hash": candidate_hash,
            "robot_side": robot_side,
            "reference_frame_id": reference_frame_id,
            "moving_frame_id": moving_frame_id,
            "xr_frame_clock_domain_id": xr_frame_clock_domain_id,
        }
        if intent_kind == "select_source":
            missing = sorted(
                name for name, value in selection_fields.items() if value is None
            )
            if missing:
                raise ValueError(f"select_source requires fields: {missing}")
            if robot_side not in {"left", "right"}:
                raise ValueError("robot_side must be 'left' or 'right'")
        elif any(value is not None for value in selection_fields.values()):
            raise ValueError(f"{intent_kind} must not carry source-selection fields")

        if intent_kind == "begin_window":
            if window_id is None:
                raise ValueError("begin_window requires window_id")
        elif window_id is not None:
            raise ValueError(f"{intent_kind} must not carry window_id")

        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "client_sequence", client_sequence)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "intent_kind", intent_kind)
        object.__setattr__(self, "candidate_hash", candidate_hash)
        object.__setattr__(self, "robot_side", robot_side)
        object.__setattr__(self, "reference_frame_id", reference_frame_id)
        object.__setattr__(self, "moving_frame_id", moving_frame_id)
        object.__setattr__(
            self,
            "xr_frame_clock_domain_id",
            xr_frame_clock_domain_id,
        )
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(
            self,
            "reason",
            _required_text(self.reason, name="reason"),
        )

    @property
    def intent_id(self) -> str:
        payload = {
            "schema": QUEST_CALIBRATION_INTENT_SCHEMA_V1,
            "client_id": self.client_id,
            "client_sequence": self.client_sequence,
            "expected_revision": self.expected_revision,
            "intent_kind": self.intent_kind,
            "candidate_hash": self.candidate_hash,
            "robot_side": self.robot_side,
            "reference_frame_id": self.reference_frame_id,
            "moving_frame_id": self.moving_frame_id,
            "xr_frame_clock_domain_id": self.xr_frame_clock_domain_id,
            "window_id": self.window_id,
            "reason": self.reason,
        }
        return "quest-calibration-intent:" + canonical_content_hash(
            payload,
            domain=QUEST_CALIBRATION_INTENT_HASH_DOMAIN_V1,
        )


def quest_calibration_intent_from_json(
    payload: Mapping[str, Any],
) -> QuestCalibrationIntent:
    """Validate an exact future UI payload before constructing an intent."""

    if not isinstance(payload, Mapping):
        raise TypeError("Quest calibration intent payload must be an object")
    allowed = {
        "client_id",
        "client_sequence",
        "expected_revision",
        "intent_kind",
        "candidate_hash",
        "robot_side",
        "reference_frame_id",
        "moving_frame_id",
        "xr_frame_clock_domain_id",
        "window_id",
        "reason",
    }
    required = {
        "client_id",
        "client_sequence",
        "expected_revision",
        "intent_kind",
    }
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError(f"unknown Quest calibration intent fields: {unknown}")
    if missing:
        raise ValueError(f"missing Quest calibration intent fields: {missing}")
    return QuestCalibrationIntent(
        client_id=payload["client_id"],
        client_sequence=payload["client_sequence"],
        expected_revision=payload["expected_revision"],
        intent_kind=payload["intent_kind"],
        candidate_hash=payload.get("candidate_hash"),
        robot_side=payload.get("robot_side"),
        reference_frame_id=payload.get("reference_frame_id"),
        moving_frame_id=payload.get("moving_frame_id"),
        xr_frame_clock_domain_id=payload.get("xr_frame_clock_domain_id"),
        window_id=payload.get("window_id"),
        reason=payload.get("reason", "operator_request"),
    )


@dataclass(frozen=True)
class QuestCalibrationSubmissionReceipt:
    """Admission and synchronous operation result for one operator intent."""

    intent_id: str
    accepted: bool
    revision: int
    outcome: str | None = None
    rejection_code: str | None = None
    message: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "accepted": self.accepted,
            "revision": self.revision,
            "outcome": self.outcome,
            "rejection_code": self.rejection_code,
            "message": self.message,
        }


class QuestPairedPoseOperatorPort(Protocol):
    """Exact lower lifecycle used by the Station without source authority."""

    def select(
        self,
        *,
        intent: QuestCaptureSelectionIntent,
        frame_facts: QuestCaptureFrameFacts,
    ) -> object: ...

    def begin_window(self, *, window_id: str) -> None: ...

    def complete_window(self) -> object | None: ...

    def request_seal(self) -> object: ...

    def close(self) -> None: ...

    def snapshot_status(self) -> Mapping[str, Any]: ...


class QuestCalibrationOperatorController:
    """Serialize operator actions and expose only pose-redacted UI status."""

    def __init__(self, application: QuestPairedPoseOperatorPort) -> None:
        required_methods = (
            "select",
            "begin_window",
            "complete_window",
            "request_seal",
            "close",
            "snapshot_status",
        )
        missing = tuple(
            name
            for name in required_methods
            if not callable(getattr(application, name, None))
        )
        if missing:
            raise TypeError(f"Quest capture application is missing methods: {missing}")
        self._application = application
        self._lock = threading.RLock()
        self._revision = 0
        self._last_sequence_by_client: dict[str, int] = {}
        self._active_intent_id: str | None = None
        self._active_intent_kind: str | None = None
        self._last_completion_revision = -1
        self._last_intent_id: str | None = None
        self._last_intent_kind: str | None = None
        self._last_outcome: str | None = None
        self._last_error_code: str | None = None

    def _reject(
        self,
        intent: QuestCalibrationIntent,
        *,
        code: str,
        message: str,
    ) -> QuestCalibrationSubmissionReceipt:
        return QuestCalibrationSubmissionReceipt(
            intent_id=intent.intent_id,
            accepted=False,
            revision=self._revision,
            rejection_code=code,
            message=message,
        )

    @staticmethod
    def _operation_error_code(error: Exception) -> str:
        if isinstance(error, QuestPairedPoseCaptureCleanupError):
            return "cleanup_failed"
        if isinstance(error, QuestPairedPoseCaptureTerminalError):
            return "capture_terminal"
        if isinstance(error, StaticWindowRejectedError):
            return "window_rejected"
        if isinstance(error, KeyError):
            return "unknown_source"
        if isinstance(error, (TypeError, ValueError, RuntimeError)):
            return "invalid_state"
        return "operation_failed"

    def _execute(self, intent: QuestCalibrationIntent) -> str:
        if intent.intent_kind == "select_source":
            assert intent.candidate_hash is not None
            assert intent.robot_side is not None
            assert intent.reference_frame_id is not None
            assert intent.moving_frame_id is not None
            assert intent.xr_frame_clock_domain_id is not None
            self._application.select(
                intent=QuestCaptureSelectionIntent(
                    candidate_hash=intent.candidate_hash,
                    robot_side=intent.robot_side,
                ),
                frame_facts=QuestCaptureFrameFacts(
                    candidate_hash=intent.candidate_hash,
                    reference_frame_id=intent.reference_frame_id,
                    moving_frame_id=intent.moving_frame_id,
                    xr_frame_clock_domain_id=intent.xr_frame_clock_domain_id,
                ),
            )
            return "source_selected"
        if intent.intent_kind == "begin_window":
            assert intent.window_id is not None
            self._application.begin_window(window_id=intent.window_id)
            return "window_started"
        if intent.intent_kind == "complete_window":
            receipt = self._application.complete_window()
            return "window_drain_pending" if receipt is None else "window_completed"
        if intent.intent_kind == "seal":
            self._application.request_seal()
            return "sealed"
        if intent.intent_kind == "cancel":
            self._application.close()
            return "cancelled"
        raise AssertionError(f"unhandled intent kind: {intent.intent_kind}")

    def _record_completion(
        self,
        *,
        intent: QuestCalibrationIntent,
        revision: int,
        outcome: str | None,
        error_code: str | None,
    ) -> None:
        with self._lock:
            if self._active_intent_id == intent.intent_id:
                self._active_intent_id = None
                self._active_intent_kind = None
            if revision >= self._last_completion_revision:
                self._last_completion_revision = revision
                self._last_intent_id = intent.intent_id
                self._last_intent_kind = intent.intent_kind
                self._last_outcome = outcome
                self._last_error_code = error_code

    def submit(
        self,
        intent: QuestCalibrationIntent,
    ) -> QuestCalibrationSubmissionReceipt:
        """Admit and execute one bounded lower lifecycle action synchronously.

        An admitted operation consumes the client sequence and advances the
        revision even when the lower operation fails.  This prevents an
        ambiguous retry from reusing the same intent identity.  ``cancel`` is
        the only stale-revision exception so it remains reachable while an
        earlier lower action is in progress.
        """

        if not isinstance(intent, QuestCalibrationIntent):
            raise TypeError("intent must be QuestCalibrationIntent")
        with self._lock:
            previous = self._last_sequence_by_client.get(intent.client_id, -1)
            if intent.client_sequence <= previous:
                return self._reject(
                    intent,
                    code="non_monotonic_client_sequence",
                    message=f"last admitted sequence is {previous}",
                )
            is_cancel = intent.intent_kind == "cancel"
            if not is_cancel and intent.expected_revision != self._revision:
                return self._reject(
                    intent,
                    code="stale_revision",
                    message=(
                        f"expected {intent.expected_revision}, current {self._revision}"
                    ),
                )
            if not is_cancel and self._active_intent_id is not None:
                return self._reject(
                    intent,
                    code="operation_in_progress",
                    message="wait for the current Quest calibration operation",
                )
            self._last_sequence_by_client[intent.client_id] = intent.client_sequence
            self._revision += 1
            admitted_revision = self._revision
            if not is_cancel:
                self._active_intent_id = intent.intent_id
                self._active_intent_kind = intent.intent_kind

        try:
            outcome = self._execute(intent)
        except Exception as error:  # noqa: BLE001 - project a bounded error code.
            error_code = self._operation_error_code(error)
            self._record_completion(
                intent=intent,
                revision=admitted_revision,
                outcome=None,
                error_code=error_code,
            )
            return QuestCalibrationSubmissionReceipt(
                intent_id=intent.intent_id,
                accepted=False,
                revision=admitted_revision,
                rejection_code=error_code,
                message=f"Quest calibration operation failed: {type(error).__name__}",
            )

        self._record_completion(
            intent=intent,
            revision=admitted_revision,
            outcome=outcome,
            error_code=None,
        )
        return QuestCalibrationSubmissionReceipt(
            intent_id=intent.intent_id,
            accepted=True,
            revision=admitted_revision,
            outcome=outcome,
            message="Quest calibration operation completed",
        )

    def submit_json(self, payload: Mapping[str, Any]) -> dict[str, object]:
        return self.submit(quest_calibration_intent_from_json(payload)).to_payload()

    @staticmethod
    def _candidate_projection(value: object) -> dict[str, object] | None:
        if not isinstance(value, Mapping):
            return None
        candidate_hash = value.get("candidate_hash")
        handedness = value.get("handedness")
        reference_space_type = value.get("reference_space_type")
        generation = value.get("reference_space_generation")
        profiles = value.get("profiles")
        if not all(
            isinstance(item, str) and item
            for item in (candidate_hash, handedness, reference_space_type)
        ):
            return None
        if isinstance(generation, bool) or not isinstance(generation, int):
            return None
        if not isinstance(profiles, (list, tuple)) or not all(
            isinstance(profile, str) and profile for profile in profiles
        ):
            return None
        return {
            "candidate_hash": candidate_hash,
            "handedness": handedness,
            "reference_space_type": reference_space_type,
            "reference_space_generation": generation,
            "profiles": list(profiles),
        }

    def status(self) -> dict[str, object]:
        """Return rebuildable, pose-redacted operator state only."""

        lower = self._application.snapshot_status()
        if not isinstance(lower, Mapping):
            raise TypeError("Quest capture status must be a mapping")
        raw_candidates = lower.get("candidates", ())
        candidates = []
        if isinstance(raw_candidates, (list, tuple)):
            candidates = [
                projected
                for candidate in raw_candidates
                if (projected := self._candidate_projection(candidate)) is not None
            ]
        capture = lower.get("capture")
        capture = capture if isinstance(capture, Mapping) else {}
        lower_state = lower.get("state")
        state = lower_state if isinstance(lower_state, str) else "unknown"
        active_window_id = capture.get("active_window_id")
        if not isinstance(active_window_id, str):
            active_window_id = None
        completed_window_count = capture.get("completed_window_count", 0)
        if (
            isinstance(completed_window_count, bool)
            or not isinstance(completed_window_count, int)
            or completed_window_count < 0
        ):
            completed_window_count = 0
        draining = capture.get("draining") is True
        finalizing = capture.get("finalizing") is True
        closed = lower.get("closed") is True
        evidence_hash = lower.get("evidence_hash")
        if not isinstance(evidence_hash, str):
            evidence_hash = None
        selected_candidate_hash = lower.get("selected_candidate_hash")
        if not isinstance(selected_candidate_hash, str):
            selected_candidate_hash = None
        selected_robot_side = lower.get("selected_robot_side")
        if selected_robot_side not in {"left", "right"}:
            selected_robot_side = None

        with self._lock:
            revision = self._revision
            active_intent_id = self._active_intent_id
            active_intent_kind = self._active_intent_kind
            last_intent_id = self._last_intent_id
            last_intent_kind = self._last_intent_kind
            last_outcome = self._last_outcome
            last_error_code = self._last_error_code
        idle = active_intent_id is None
        bound = state == "bound" and not closed
        return {
            "schema": QUEST_CALIBRATION_STATUS_SCHEMA_V1,
            "revision": revision,
            "state": state,
            "capture_id": lower.get("capture_id"),
            "profile_id": lower.get("profile_id"),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "selected_candidate_hash": selected_candidate_hash,
            "selected_robot_side": selected_robot_side,
            "active_window_id": active_window_id,
            "completed_window_count": completed_window_count,
            "draining": draining,
            "finalizing": finalizing,
            "evidence_hash": evidence_hash,
            "closed": closed,
            "active_intent_id": active_intent_id,
            "active_intent_kind": active_intent_kind,
            "last_intent_id": last_intent_id,
            "last_intent_kind": last_intent_kind,
            "last_outcome": last_outcome,
            "last_error_code": last_error_code,
            "actions": {
                "can_select_source": (
                    idle and state == "waiting_for_selection" and bool(candidates)
                ),
                "can_begin_window": (
                    idle and bound and active_window_id is None and not draining
                ),
                "can_complete_window": (
                    idle and bound and active_window_id is not None
                ),
                "can_seal": (
                    idle
                    and bound
                    and active_window_id is None
                    and completed_window_count > 0
                    and not draining
                ),
                "can_cancel": not closed,
            },
        }
