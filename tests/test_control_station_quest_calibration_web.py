from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import pytest

from control_station.quest_calibration import QuestCalibrationWebCoordinator
from teleop_runtime.deploy.quest_paired_pose_capture import (
    QuestCaptureFrameFacts,
    QuestCaptureSelectionIntent,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _FakeApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.close_calls = 0
        self.status: dict[str, object] = {
            "state": "waiting_for_selection",
            "capture_id": "capture-web-1",
            "profile_id": "profile-web-1",
            "candidates": [
                {
                    "candidate_hash": "candidate-web-1",
                    "handedness": "right",
                    "reference_space_type": "local-floor",
                    "reference_space_generation": 0,
                    "profiles": ["oculus-touch"],
                    "matrix": [1.0] * 16,
                    "client_instance_id": "secret-client",
                }
            ],
            "selected_candidate_hash": None,
            "selected_robot_side": None,
            "capture": None,
            "evidence_hash": None,
            "terminal_reason": "secret-terminal-detail",
            "closed": False,
        }

    def select(
        self,
        *,
        intent: QuestCaptureSelectionIntent,
        frame_facts: QuestCaptureFrameFacts,
    ) -> object:
        self.calls.append(("select", (intent, frame_facts)))
        self.status.update(
            {
                "state": "bound",
                "selected_candidate_hash": intent.candidate_hash,
                "selected_robot_side": intent.robot_side,
                "capture": {
                    "active_window_id": None,
                    "completed_window_count": 0,
                    "draining": False,
                    "finalizing": False,
                },
            }
        )
        return object()

    def begin_window(self, *, window_id: str) -> None:
        self.calls.append(("begin_window", window_id))

    def complete_window(self) -> object:
        self.calls.append(("complete_window", None))
        return object()

    def request_seal(self) -> object:
        self.calls.append(("request_seal", None))
        return object()

    def close(self) -> None:
        self.calls.append(("close", None))
        self.close_calls += 1
        self.status.update({"state": "closed", "closed": True})

    def snapshot_status(self) -> Mapping[str, object]:
        return self.status


class _FakeFactory:
    def __init__(self) -> None:
        from threading import Event

        self._ready = Event()
        self.application: object | None = None
        self.error: Exception | None = None
        self.close_calls = 0

    def publish(self, application: object) -> None:
        self.application = application
        self._ready.set()

    def fail(self, error: Exception) -> None:
        self.error = error
        self._ready.set()

    def wait_for_application(self, *, timeout_s: float) -> object:
        if not self._ready.wait(timeout=timeout_s):
            raise TimeoutError("not ready")
        if self.error is not None:
            raise self.error
        assert self.application is not None
        return self.application

    def close(self) -> None:
        self.close_calls += 1


def _wait_until(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return (
            response.status,
            response.headers.get_content_type(),
            response.read().decode("utf-8"),
        )


def _post_json(url: str, payload: object) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=2.0)
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read().decode("utf-8"))
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_assets_are_packaged_and_describe_the_no_motion_boundary() -> None:
    package_root = resources.files("control_station.quest_calibration")
    html = package_root.joinpath("static/index.html").read_text(encoding="utf-8")
    script = package_root.joinpath("static/app.js").read_text(encoding="utf-8")
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert "不发送机械臂动作" in html
    assert 'fetch("/status.json"' in script
    assert 'fetch("/intent"' in script
    assert '"quest_calibration/static/*.html"' in pyproject
    assert '"quest_calibration/static/*.js"' in pyproject


def test_web_server_waits_then_projects_pose_redacted_status() -> None:
    factory = _FakeFactory()
    application = _FakeApplication()
    coordinator = QuestCalibrationWebCoordinator(
        factory,
        open_browser=False,
        application_timeout_s=1.0,
        attach_poll_s=0.02,
    )
    coordinator.start()
    try:
        status, content_type, html = _get(coordinator.url)
        assert (status, content_type) == (200, "text/html")
        assert "Quest × 机械臂配对标定" in html
        waiting = json.loads(_get(coordinator.url + "status.json")[2])
        assert waiting["binding_state"] == "waiting_for_application"
        assert waiting["operator"] is None

        factory.publish(application)
        _wait_until(lambda: coordinator.binding_state == "ready")
        ready = json.loads(_get(coordinator.url + "status.json")[2])
        encoded = json.dumps(ready)
        assert ready["binding_state"] == "ready"
        assert ready["operator"]["candidate_count"] == 1
        assert ready["operator"]["candidates"][0]["candidate_hash"] == (
            "candidate-web-1"
        )
        for forbidden in (
            "matrix",
            "secret-client",
            "secret-terminal-detail",
            "terminal_reason",
        ):
            assert forbidden not in encoded
    finally:
        coordinator.close()

    assert application.close_calls == 0
    assert factory.close_calls == 0


def test_intent_endpoint_uses_typed_controller_and_cancel_is_explicit() -> None:
    factory = _FakeFactory()
    application = _FakeApplication()
    factory.publish(application)
    coordinator = QuestCalibrationWebCoordinator(
        factory,
        open_browser=False,
        application_timeout_s=1.0,
        attach_poll_s=0.02,
    )
    coordinator.start()
    try:
        _wait_until(lambda: coordinator.binding_state == "ready")
        status, receipt = _post_json(
            coordinator.url + "intent",
            {
                "client_id": "browser-1",
                "client_sequence": 1,
                "expected_revision": 0,
                "intent_kind": "select_source",
                "candidate_hash": "candidate-web-1",
                "robot_side": "right",
                "reference_frame_id": "quest-local-floor-generation-0",
                "moving_frame_id": "quest-controller-grip",
                "xr_frame_clock_domain_id": "webxr-frame-clock",
            },
        )
        assert status == 200
        assert receipt["accepted"] is True
        method, values = application.calls[-1]
        selection, frame_facts = values
        assert method == "select"
        assert selection == QuestCaptureSelectionIntent(
            candidate_hash="candidate-web-1",
            robot_side="right",
        )
        assert frame_facts.reference_frame_id == "quest-local-floor-generation-0"

        status, cancel = _post_json(
            coordinator.url + "intent",
            {
                "client_id": "browser-1",
                "client_sequence": 2,
                "expected_revision": 0,
                "intent_kind": "cancel",
                "reason": "operator_web_cancel",
            },
        )
        assert status == 200
        assert cancel["outcome"] == "cancelled"
        assert application.close_calls == 1
    finally:
        coordinator.close()

    assert application.close_calls == 1
    assert factory.close_calls == 0


def test_invalid_or_early_intents_fail_without_raw_details() -> None:
    factory = _FakeFactory()
    coordinator = QuestCalibrationWebCoordinator(
        factory,
        open_browser=False,
        application_timeout_s=1.0,
        attach_poll_s=0.02,
    )
    coordinator.start()
    try:
        status, early = _post_json(
            coordinator.url + "intent",
            {
                "client_id": "browser-1",
                "client_sequence": 1,
                "expected_revision": 0,
                "intent_kind": "seal",
            },
        )
        assert status == 409
        assert early["rejection_code"] == "application_not_ready"

        factory.publish(_FakeApplication())
        _wait_until(lambda: coordinator.binding_state == "ready")
        status, invalid = _post_json(
            coordinator.url + "intent",
            {
                "client_id": "browser-1",
                "client_sequence": 2,
                "expected_revision": 0,
                "intent_kind": "unknown",
                "matrix": [1.0] * 16,
            },
        )
        assert status == 400
        assert invalid == {
            "accepted": False,
            "message": "ValueError",
            "rejection_code": "invalid_intent",
        }
    finally:
        coordinator.close()


def test_attach_failure_exposes_type_only_and_close_is_bounded() -> None:
    factory = _FakeFactory()
    factory.fail(RuntimeError("secret factory address and pose"))
    coordinator = QuestCalibrationWebCoordinator(
        factory,
        open_browser=False,
        application_timeout_s=1.0,
        attach_poll_s=0.02,
    )
    coordinator.start()
    _wait_until(lambda: coordinator.binding_state == "attach_failed")

    status = coordinator.status_payload()
    encoded = json.dumps(status)
    coordinator.close()

    assert status["attach_error_code"] == "RuntimeError"
    assert status["operator"] is None
    assert "secret" not in encoded
    assert "address" not in encoded
    assert coordinator.binding_state == "closed"
    assert factory.close_calls == 0


def test_constructor_rejects_missing_factory_port() -> None:
    with pytest.raises(TypeError, match="wait_for_application"):
        QuestCalibrationWebCoordinator(object())
