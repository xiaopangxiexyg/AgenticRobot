from __future__ import annotations

import json
import threading
from collections.abc import Mapping

import pytest

from control_station.quest_calibration import (
    QuestCalibrationIntent,
    QuestCalibrationOperatorController,
    quest_calibration_intent_from_json,
)
from teleop_runtime.deploy.quest_paired_pose_capture import (
    QuestCaptureFrameFacts,
    QuestCaptureSelectionIntent,
    StaticWindowRejectedError,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _FakeCaptureApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.complete_results: list[object | None] = [object()]
        self.failure_by_method: dict[str, Exception] = {}
        self.seal_started = threading.Event()
        self.seal_release = threading.Event()
        self.block_seal = False
        self.closed = False
        self.status_payload: dict[str, object] = {
            "state": "waiting_for_selection",
            "capture_id": "capture-1",
            "profile_id": "static-profile-1",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_hash": "candidate-hash-1",
                    "handedness": "right",
                    "reference_space_type": "local-floor",
                    "reference_space_generation": 0,
                    "profiles": ["oculus-touch-v3"],
                    "client_id": 17,
                    "client_instance_id": "secret-client-instance",
                    "xr_session_id": "secret-xr-session",
                    "controller_source_session_id": "secret-controller-session",
                    "matrix": [1.0] * 16,
                    "position": [0.1, 0.2, 0.3],
                }
            ],
            "selected_candidate_hash": None,
            "selected_robot_side": None,
            "capture": None,
            "terminal_reason": "secret-terminal-pose-details",
            "evidence_hash": None,
            "closed": False,
        }

    def _fail(self, method: str) -> None:
        error = self.failure_by_method.get(method)
        if error is not None:
            raise error

    def select(
        self,
        *,
        intent: QuestCaptureSelectionIntent,
        frame_facts: QuestCaptureFrameFacts,
    ) -> object:
        self._fail("select")
        self.calls.append(("select", (intent, frame_facts)))
        self.status_payload.update(
            {
                "state": "bound",
                "selected_candidate_hash": intent.candidate_hash,
                "selected_robot_side": intent.robot_side,
                "capture": {
                    "active_window_id": None,
                    "completed_window_count": 0,
                    "draining": False,
                    "finalizing": False,
                    "robot_next_event_sequence": 91,
                    "terminal_reason": "secret-reducer-reason",
                },
            }
        )
        return object()

    def begin_window(self, *, window_id: str) -> None:
        self._fail("begin_window")
        self.calls.append(("begin_window", window_id))
        capture = dict(self.status_payload["capture"])
        capture["active_window_id"] = window_id
        self.status_payload["capture"] = capture

    def complete_window(self) -> object | None:
        self._fail("complete_window")
        self.calls.append(("complete_window", None))
        result = self.complete_results.pop(0)
        if result is not None:
            capture = dict(self.status_payload["capture"])
            capture["active_window_id"] = None
            capture["completed_window_count"] = (
                int(capture["completed_window_count"]) + 1
            )
            self.status_payload["capture"] = capture
        return result

    def request_seal(self) -> object:
        self._fail("request_seal")
        self.calls.append(("request_seal", None))
        self.seal_started.set()
        if self.block_seal:
            assert self.seal_release.wait(timeout=2.0)
            if self.closed:
                raise RuntimeError("secret pose data from cancelled seal")
        self.status_payload.update(
            {
                "state": "sealed",
                "evidence_hash": "evidence-hash-1",
            }
        )
        return object()

    def close(self) -> None:
        self._fail("close")
        self.calls.append(("close", None))
        self.closed = True
        self.seal_release.set()
        self.status_payload.update({"state": "closed", "closed": True})

    def snapshot_status(self) -> Mapping[str, object]:
        return self.status_payload


def _intent(
    *,
    sequence: int,
    revision: int,
    kind: str,
    **kwargs: object,
) -> QuestCalibrationIntent:
    return QuestCalibrationIntent(
        client_id="operator-ui-1",
        client_sequence=sequence,
        expected_revision=revision,
        intent_kind=kind,
        **kwargs,
    )


def _select_intent(*, sequence: int = 1, revision: int = 0):
    return _intent(
        sequence=sequence,
        revision=revision,
        kind="select_source",
        candidate_hash="candidate-hash-1",
        robot_side="right",
        reference_frame_id="quest-local-floor-generation-0",
        moving_frame_id="quest-right-controller-grip",
        xr_frame_clock_domain_id="webxr-frame-clock:session-1",
    )


def test_intent_validation_is_exact_and_contains_no_pose_payload() -> None:
    with pytest.raises(ValueError, match="requires fields"):
        _intent(sequence=1, revision=0, kind="select_source")
    with pytest.raises(ValueError, match="must not carry window_id"):
        _intent(
            sequence=1,
            revision=0,
            kind="seal",
            window_id="not-allowed",
        )
    with pytest.raises(ValueError, match="unknown"):
        quest_calibration_intent_from_json(
            {
                "client_id": "ui",
                "client_sequence": 1,
                "expected_revision": 0,
                "intent_kind": "cancel",
                "matrix": [1.0] * 16,
            }
        )


def test_json_parser_and_intent_identity_are_deterministic() -> None:
    payload = {
        "client_id": "operator-ui-1",
        "client_sequence": 1,
        "expected_revision": 0,
        "intent_kind": "begin_window",
        "window_id": "stationary-window-001",
    }

    first = quest_calibration_intent_from_json(payload)
    second = quest_calibration_intent_from_json(dict(reversed(tuple(payload.items()))))

    assert first == second
    assert first.intent_id == second.intent_id
    prefix, digest = first.intent_id.split(":", maxsplit=1)
    assert prefix == "quest-calibration-intent"
    assert len(digest) == 64
    assert int(digest, 16) >= 0


def test_select_source_builds_exact_lower_owned_contracts() -> None:
    application = _FakeCaptureApplication()
    controller = QuestCalibrationOperatorController(application)

    receipt = controller.submit(_select_intent())

    assert receipt.accepted is True
    assert receipt.outcome == "source_selected"
    method, payload = application.calls[-1]
    selection, frame_facts = payload
    assert method == "select"
    assert selection == QuestCaptureSelectionIntent(
        candidate_hash="candidate-hash-1",
        robot_side="right",
    )
    assert frame_facts == QuestCaptureFrameFacts(
        candidate_hash="candidate-hash-1",
        reference_frame_id="quest-local-floor-generation-0",
        moving_frame_id="quest-right-controller-grip",
        xr_frame_clock_domain_id="webxr-frame-clock:session-1",
    )


def test_full_lifecycle_forwards_without_writing_or_fitting_artifacts() -> None:
    application = _FakeCaptureApplication()
    application.complete_results = [None, object()]
    controller = QuestCalibrationOperatorController(application)

    assert controller.submit(_select_intent()).outcome == "source_selected"
    assert (
        controller.submit(
            _intent(
                sequence=2,
                revision=1,
                kind="begin_window",
                window_id="stationary-window-001",
            )
        ).outcome
        == "window_started"
    )
    assert (
        controller.submit(
            _intent(sequence=3, revision=2, kind="complete_window")
        ).outcome
        == "window_drain_pending"
    )
    assert (
        controller.submit(
            _intent(sequence=4, revision=3, kind="complete_window")
        ).outcome
        == "window_completed"
    )
    assert (
        controller.submit(_intent(sequence=5, revision=4, kind="seal")).outcome
        == "sealed"
    )

    status = controller.status()
    assert status["revision"] == 5
    assert status["state"] == "sealed"
    assert status["evidence_hash"] == "evidence-hash-1"
    assert status["actions"]["can_cancel"] is True
    assert [name for name, _ in application.calls] == [
        "select",
        "begin_window",
        "complete_window",
        "complete_window",
        "request_seal",
    ]


def test_status_is_pose_redacted_and_exposes_bounded_actions_only() -> None:
    application = _FakeCaptureApplication()
    controller = QuestCalibrationOperatorController(application)

    status = controller.status()
    encoded = json.dumps(status)

    assert status["candidate_count"] == 1
    assert status["candidates"] == [
        {
            "candidate_hash": "candidate-hash-1",
            "handedness": "right",
            "reference_space_type": "local-floor",
            "reference_space_generation": 0,
            "profiles": ["oculus-touch-v3"],
        }
    ]
    assert status["actions"] == {
        "can_select_source": True,
        "can_begin_window": False,
        "can_complete_window": False,
        "can_seal": False,
        "can_cancel": True,
    }
    for forbidden in (
        "matrix",
        "position",
        "secret-client-instance",
        "secret-xr-session",
        "secret-controller-session",
        "secret-terminal-pose-details",
        "robot_next_event_sequence",
        "terminal_reason",
    ):
        assert forbidden not in encoded


def test_revision_and_client_sequence_reject_replay_without_lower_call() -> None:
    application = _FakeCaptureApplication()
    controller = QuestCalibrationOperatorController(application)

    stale = controller.submit(_intent(sequence=1, revision=1, kind="complete_window"))
    assert stale.accepted is False
    assert stale.rejection_code == "stale_revision"
    assert application.calls == []

    accepted = controller.submit(_select_intent(sequence=1, revision=0))
    replay = controller.submit(_intent(sequence=1, revision=1, kind="complete_window"))
    assert accepted.accepted is True
    assert replay.accepted is False
    assert replay.rejection_code == "non_monotonic_client_sequence"
    assert [name for name, _ in application.calls] == ["select"]


def test_failed_operation_consumes_sequence_and_redacts_exception_detail() -> None:
    application = _FakeCaptureApplication()
    application.failure_by_method["select"] = ValueError("secret matrix [1, 0, 0, 1]")
    controller = QuestCalibrationOperatorController(application)

    receipt = controller.submit(_select_intent())
    replay = controller.submit(_select_intent())

    assert receipt.accepted is False
    assert receipt.revision == 1
    assert receipt.rejection_code == "invalid_state"
    assert "secret" not in receipt.message
    assert "matrix" not in receipt.message
    assert replay.rejection_code == "non_monotonic_client_sequence"
    status = controller.status()
    assert status["last_error_code"] == "invalid_state"
    assert status["revision"] == 1


def test_window_rejection_is_projected_without_lower_detail() -> None:
    application = _FakeCaptureApplication()
    application.status_payload.update(
        {
            "state": "bound",
            "capture": {
                "active_window_id": "stationary-window-001",
                "completed_window_count": 0,
                "draining": False,
                "finalizing": False,
            },
        }
    )
    application.failure_by_method["complete_window"] = StaticWindowRejectedError(
        "secret pose excursion details"
    )
    controller = QuestCalibrationOperatorController(application)

    receipt = controller.submit(_intent(sequence=1, revision=0, kind="complete_window"))

    assert receipt.accepted is False
    assert receipt.rejection_code == "window_rejected"
    assert "secret" not in receipt.message
    assert "excursion" not in receipt.message


def test_cancel_bypasses_stale_revision_and_wins_completion_projection() -> None:
    application = _FakeCaptureApplication()
    application.status_payload.update(
        {
            "state": "bound",
            "capture": {
                "active_window_id": None,
                "completed_window_count": 1,
                "draining": False,
                "finalizing": False,
            },
        }
    )
    application.block_seal = True
    controller = QuestCalibrationOperatorController(application)
    seal_receipt: list[object] = []

    thread = threading.Thread(
        target=lambda: seal_receipt.append(
            controller.submit(_intent(sequence=1, revision=0, kind="seal"))
        )
    )
    thread.start()
    assert application.seal_started.wait(timeout=1.0)

    cancel = controller.submit(
        _intent(sequence=2, revision=0, kind="cancel", reason="operator_abort")
    )
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert cancel.accepted is True
    assert cancel.revision == 2
    assert cancel.outcome == "cancelled"
    assert seal_receipt[0].accepted is False
    status = controller.status()
    assert status["revision"] == 2
    assert status["state"] == "closed"
    assert status["last_intent_kind"] == "cancel"
    assert status["last_outcome"] == "cancelled"
    assert status["last_error_code"] is None
    assert status["actions"]["can_cancel"] is False


def test_controller_rejects_an_incomplete_lower_application() -> None:
    with pytest.raises(TypeError, match="missing methods"):
        QuestCalibrationOperatorController(object())
