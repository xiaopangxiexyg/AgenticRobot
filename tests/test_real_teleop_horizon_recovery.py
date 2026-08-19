from __future__ import annotations

from robot_motion.runtime.dispatch import RigDispatchError
from teleop_runtime.real_teleop.control_loop import (
    _consume_joint_horizon_certification_dispatch,
    _joint_horizon_certification_dispatch_context,
)


class _Clutch:
    def __init__(self) -> None:
        self.hold_reasons: list[str] = []

    def request_hold(self, *, reason: str) -> None:
        self.hold_reasons.append(reason)


class _ControlUi:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def append_message(self, tag: str, message: str) -> None:
        self.messages.append((tag, message))


def _raise_writer_error(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("writer already closed")


def _raise_countdown_error(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("countdown already cancelled")


def _wrapped_horizon_error() -> BaseException:
    try:
        raise RuntimeError(
            "reused batched joint horizon candidate failed certification for "
            "joint 3: position_below=0rad; position_above=1.217e-06rad"
        )
    except RuntimeError as inner:
        try:
            raise RigDispatchError(
                rig_name="bimanual_rig",
                failures={"right_arm": inner},
            ) from inner
        except RigDispatchError as wrapped:
            return wrapped


def test_horizon_certification_context_is_narrowly_classified() -> None:
    context = _joint_horizon_certification_dispatch_context(
        _wrapped_horizon_error()
    )

    assert context is not None
    assert context["marker"] == (
        "reused batched joint horizon candidate failed certification"
    )
    assert _joint_horizon_certification_dispatch_context(
        RuntimeError("right arm transport unavailable")
    ) is None
    projection_context = _joint_horizon_certification_dispatch_context(
        RuntimeError(
            "joint horizon position output exceeds numeric projection budget: "
            "correction=3e-6rad"
        )
    )
    assert projection_context is not None


def test_horizon_certification_dispatch_closes_recording_and_retains_runtime() -> None:
    clutch = _Clutch()
    control_ui = _ControlUi()
    cancelled: list[tuple[float, str]] = []
    stopped: list[tuple[float, str, dict[str, object]]] = []
    logs: list[tuple[str, str]] = []

    consumed = _consume_joint_horizon_certification_dispatch(
        _wrapped_horizon_error(),
        clutch_input=clutch,
        recording_active=lambda: True,
        stop_recording=lambda now, *, end_reason, summary: stopped.append(
            (now, end_reason, dict(summary))
        ),
        cancel_recording_countdown=lambda now, *, reason: cancelled.append(
            (now, reason)
        ),
        now=12.5,
        step_index=41,
        last_state="RUNNING",
        log=lambda tag, message, *, now: logs.append((tag, message)),
        control_ui=control_ui,
    )

    assert consumed is True
    assert clutch.hold_reasons == ["joint_horizon_certification"]
    assert cancelled == [(12.5, "joint_horizon_certification")]
    assert stopped[0][0:2] == (12.5, "joint_horizon_certification")
    assert stopped[0][2]["joint_horizon_certification_detected"] is True
    assert control_ui.messages[0][0] == "JointHorizon"
    assert any(tag == "JointHorizon" for tag, _ in logs)


def test_horizon_certification_cleanup_failures_do_not_escape() -> None:
    clutch = _Clutch()
    consumed = _consume_joint_horizon_certification_dispatch(
        _wrapped_horizon_error(),
        clutch_input=clutch,
        recording_active=lambda: True,
        stop_recording=_raise_writer_error,
        cancel_recording_countdown=_raise_countdown_error,
        now=1.0,
        step_index=0,
        last_state=None,
        log=lambda *_args, **_kwargs: None,
        control_ui=None,
    )

    assert consumed is True
    assert clutch.hold_reasons == ["joint_horizon_certification"]
