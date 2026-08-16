from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.deploy import run_policy, status_view


class _CameraSource:
    def __init__(self, info: dict[str, object]) -> None:
        self.info = dict(info)

    def device_info(self) -> dict[str, object]:
        return dict(self.info)


class _BrokenCameraSource:
    def device_info(self) -> dict[str, object]:
        raise RuntimeError("camera offline")


class _HandBackend:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = dict(state)

    def get_state(self) -> dict[str, object]:
        return dict(self.state)


class _HistoryPoller:
    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def latest_by_side(self, timestamp: float) -> dict[str, dict[str, object]]:
        self.timestamps.append(timestamp)
        return {"left": {"history": True}}


def test_run_policy_preserves_status_view_compatibility_exports() -> None:
    assert run_policy.compact_log_value is status_view.compact_log_value
    assert run_policy.camera_status_text is status_view.camera_status_text
    assert run_policy.hand_state_snapshot is status_view.hand_state_snapshot
    assert run_policy._compact_faults is status_view.compact_faults
    assert run_policy._policy_status_text is status_view.policy_status_text
    assert run_policy.print_controls is status_view.print_controls


def test_camera_status_text_preserves_deploy_format_and_error_isolation() -> None:
    sources = {
        "wrist": _BrokenCameraSource(),
        "front": _CameraSource({"frame_count": 3, "last_error": None}),
    }
    previewer = SimpleNamespace(disabled_reason="no display")

    assert status_view.camera_status_text({}, None) == "camera=off"
    assert status_view.camera_status_text(sources, previewer) == (
        "camera=preview=disabled:no display;front=frames:3;"
        "wrist=status_error:RuntimeError"
    )


def test_hand_state_snapshot_prefers_timestamped_history() -> None:
    backends = {"left": _HandBackend({"position": 0.25})}

    assert status_view.hand_state_snapshot(backends) == {
        "left": {"position": 0.25}
    }

    poller = _HistoryPoller()
    assert status_view.hand_state_snapshot(
        backends,
        history_poller=poller,
        timestamp=12.5,
    ) == {"left": {"history": True}}
    assert poller.timestamps == [12.5]


def test_fault_and_policy_status_formatting_is_unchanged() -> None:
    assert status_view.compact_log_value("abcdef", max_chars=4) == "a..."
    assert status_view.compact_faults([]) == "none"
    assert status_view.compact_faults(["a", "b", "c"], limit=2) == "a;b,+1"
    assert status_view.policy_status_text(
        SimpleNamespace(policy=SimpleNamespace(status_text="ready"))
    ) == "ready"
    assert status_view.policy_status_text(
        SimpleNamespace(policy=SimpleNamespace(status_text=lambda: "warming"))
    ) == "warming"
    assert status_view.policy_status_text(SimpleNamespace(policy=None)) == ""


def test_print_controls_preserves_fixed_init_auto_start_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status_view.print_controls(
        send_actions=False,
        auto_start=True,
        wait_for_ready_confirm=True,
        fixed_init_enabled=True,
    )

    output = capsys.readouterr().out
    assert "send_actions=off(dry-run)" in output
    assert "initial_policy_state=paused(auto-start after init)" in output
    assert "ready_confirm=on(A/a required after startup init)" in output
    assert "fixed_init=on(H first-frame qpos, H execute T, then startup init)" in output
    assert (
        "Controls: H=pause policy or fixed-init step, g=start after fixed-init, "
        "r=manual safety re-arm, A=confirm, q=quit"
    ) in output
