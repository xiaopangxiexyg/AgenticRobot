from __future__ import annotations

import numpy as np
import pytest

from teleop_runtime.arm_frames import TOOL_TCP_FRAME
from teleop_runtime.debug.rm65_real_debug import run_rm65_real_state_viz as state_viz


def test_rm65_real_state_visualizer_disables_startup_sdk_writes() -> None:
    backend = state_viz._build_backend(
        "left",
        "192.0.2.10",
        TOOL_TCP_FRAME,
        "/tmp/nonexistent-rm65.urdf",
        connect_hardware=False,
        initial_qpos=(0.0,) * 6,
    )

    assert not backend.config.configure_singularity_thresholds
    assert not backend.config.force_position_control
    assert not backend.config.background_streaming_enabled


def test_save_once_requires_an_output_path(monkeypatch) -> None:
    monkeypatch.setattr(
        state_viz.sys,
        "argv",
        ["run_rm65_real_state_viz", "--save-once", "--no-motion"],
    )

    with pytest.raises(SystemExit) as exc_info:
        state_viz.parse_args()

    assert exc_info.value.code == 2


def test_rm65_real_state_visualizer_reports_existing_sdk_error_fields(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        state_viz,
        "log_event",
        lambda _stage, message, **_kwargs: messages.append(message),
    )

    state_viz._log_side_status(
        "left",
        {
            "current_qpos": np.zeros(6, dtype=float),
            "hardware_result_code": 0,
            "hardware_result_text": "0",
            "hardware_arm_err": 12,
            "hardware_sys_err": 34,
        },
        np.eye(4, dtype=float),
        gate=state_viz.ChangeGate(),
        heartbeat_sec=None,
        now=1.0,
    )

    assert len(messages) == 1
    assert "result=0 arm_err=12 sys_err=34" in messages[0]
