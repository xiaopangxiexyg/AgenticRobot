from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.deploy.quest_paired_pose_capture import (
    QuestPairedPoseCaptureApplication,
    QuestTerminalCaptureController,
)


def test_terminal_r_and_c_reuse_the_capture_application_without_owning_stdin() -> None:
    application = object.__new__(QuestPairedPoseCaptureApplication)
    state = {
        "selected_candidate_hash": None,
        "candidate_count": 1,
        "state": "waiting_for_selection",
        "capture": {
            "active_window_id": None,
            "completed_window_count": 3,
            "draining": False,
            "sealed": False,
        },
        "evidence_hash": None,
        "terminal_reason": None,
    }
    calls: list[tuple[str, object]] = []
    status_messages: list[str | None] = []
    preview_calls: list[object] = []
    now_ns = [0]

    def snapshot_status():
        return {
            **state,
            "capture": dict(state["capture"]),
        }

    def select_unique_candidate(**_kwargs):
        state["selected_candidate_hash"] = "candidate-hash"
        state["state"] = "bound"
        return SimpleNamespace(
            quest=SimpleNamespace(
                controller_source_session_id="input-1",
                handedness="left",
            )
        )

    def begin_window(*, window_id: str):
        calls.append(("begin_window", window_id))
        state["capture"]["active_window_id"] = window_id

    def complete_window():
        calls.append(("complete_window", None))
        window_id = state["capture"]["active_window_id"]
        state["capture"]["active_window_id"] = None
        return SimpleNamespace(
            window_id=window_id,
            metrics={"robot_sample_count": 5, "quest_sample_count": 10},
        )

    def request_seal():
        calls.append(("request_seal", None))
        state["capture"]["sealed"] = True
        state["state"] = "sealed"
        state["evidence_hash"] = "evidence-hash"
        return object()

    application.snapshot_status = snapshot_status  # type: ignore[method-assign]
    application.poll = lambda: snapshot_status()  # type: ignore[method-assign]
    application.select_unique_candidate = select_unique_candidate  # type: ignore[method-assign]
    application.begin_window = begin_window  # type: ignore[method-assign]
    application.complete_window = complete_window  # type: ignore[method-assign]
    application.request_seal = request_seal  # type: ignore[method-assign]
    application.publish_capture_status = (  # type: ignore[method-assign]
        lambda *, terminal_message=None: status_messages.append(terminal_message)
    )
    application.publish_diagnostic_preview = (  # type: ignore[method-assign]
        lambda diagnostic_fit: preview_calls.append(diagnostic_fit) or {}
    )

    diagnostic_fit = SimpleNamespace(
        fit=SimpleNamespace(
            sample_count=3,
            translation_rms_m=0.001,
            rotation_rms_deg=0.2,
        )
    )
    output: list[str] = []
    controller = QuestTerminalCaptureController(
        application=application,
        robot_side="left",
        preferred_handedness="left",
        monotonic_ns_clock=lambda: now_ns[0],
        emit=output.append,
        fit_function=lambda _evidence: diagnostic_fit,
    )

    assert controller.handle_key("r") is True
    assert calls[0][0] == "begin_window"
    now_ns[0] = 2_000_000_000
    assert controller.handle_key("r") is True
    assert [name for name, _value in calls] == ["begin_window"]
    assert any("still recording" in line for line in output)

    # The fixed three-second boundary is owned by the existing deployment
    # poll, not by a second stdin/timer thread.
    now_ns[0] = 3_000_000_000
    controller.poll()
    assert calls[1][0] == "complete_window"

    # Once the automatic completion is visible, the next r starts the next
    # window; it no longer finishes the previous one.
    assert controller.handle_key("r") is True
    assert calls[2] == ("begin_window", "static-window-002")
    assert state["capture"]["active_window_id"] == "static-window-002"

    # Finish this test window through the same automatic path before c.
    now_ns[0] = 6_000_000_000
    controller.poll()
    assert calls[3][0] == "complete_window"

    # c completes/seals the already-recorded three-window set and publishes
    # only a diagnostic preview.  No stdin reader is created by the controller.
    assert controller.handle_key("c") is True
    assert ("request_seal", None) in calls
    assert preview_calls == [diagnostic_fit]
    assert any("preview only" in line for line in output)
    assert status_messages
