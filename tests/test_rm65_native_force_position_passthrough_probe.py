from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "hardware/rm65/utils/scripts/rm65_native_force_position_passthrough_probe.py"
)


def _load_probe_module(monkeypatch: pytest.MonkeyPatch):
    adapter = types.ModuleType("hardware.rm65.rm_api2_adapter")
    adapter.Arm = object
    adapter.LEFT_ARM_IP = "192.0.2.1"
    adapter.RIGHT_ARM_IP = "192.0.2.2"
    adapter.RM65 = object()
    monkeypatch.setitem(sys.modules, "hardware", types.ModuleType("hardware"))
    monkeypatch.setitem(sys.modules, "hardware.rm65", types.ModuleType("hardware.rm65"))
    monkeypatch.setitem(sys.modules, "hardware.rm65.rm_api2_adapter", adapter)
    spec = importlib.util.spec_from_file_location("rm65_native_force_probe_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_wrench_delta_metrics_separates_static_preload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module(monkeypatch)
    metrics = probe._wrench_delta_metrics(
        [-1.0, 0.8, -1.5, 0.02, -0.03, 0.01],
        [-1.0, -0.2, -1.5, 0.02, -0.03, 0.01],
        axis_index=1,
    )

    assert metrics["force_delta_norm_n"] == pytest.approx(1.0)
    assert metrics["torque_delta_norm_nm"] == pytest.approx(0.0)
    assert metrics["selected_axis_delta"] == pytest.approx(1.0)
    assert metrics["selected_axis_ratio"] == pytest.approx(1.0)


def test_wrench_delta_metrics_detects_off_axis_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module(monkeypatch)
    metrics = probe._wrench_delta_metrics(
        [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0] * 6,
        axis_index=1,
    )

    assert metrics["force_delta_norm_n"] == pytest.approx(1.0198039)
    assert metrics["selected_axis_ratio"] == pytest.approx(0.1961161)


def test_selected_axis_ratio_validation_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module(monkeypatch)
    args = probe.build_parser().parse_args(
        [
            "--arm",
            "left",
            "--probe-mode",
            "float-axis",
            "--calibration-profile",
            "profile.json",
            "--min-selected-axis-force-ratio",
            "1.1",
            "--execute",
            "--confirm",
            probe.FLOAT_AXIS_CONFIRM_TEXT,
        ]
    )

    with pytest.raises(RuntimeError, match="must be in"):
        probe._validate_args(args)


def test_linear_velocity_is_passed_to_vendor_sdk_in_metres_per_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module(monkeypatch)
    args = probe.build_parser().parse_args(
        [
            "--arm",
            "left",
            "--probe-mode",
            "float-axis",
            "--force-axis",
            "y",
            "--max-force-axis-linear-speed-m-s",
            "0.004",
            "--calibration-profile",
            "profile.json",
            "--execute",
            "--confirm",
            probe.FLOAT_AXIS_CONFIRM_TEXT,
        ]
    )

    _, _, limit_velocity = probe._control_vectors(args)

    assert limit_velocity == pytest.approx([0.0, 0.004, 0.0, 0.0, 0.0, 0.0])


def test_measured_pose_rate_uses_se3_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module(monkeypatch)

    linear_speed, angular_speed = probe._measured_pose_rate(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.003, 0.004, 0.0, 0.0, 0.0, 0.0],
        elapsed_sec=0.5,
    )

    assert linear_speed == pytest.approx(0.010)
    assert angular_speed == pytest.approx(0.0)


def test_measured_speed_guard_limits_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module(monkeypatch)
    args = probe.build_parser().parse_args(
        [
            "--arm",
            "left",
            "--probe-mode",
            "float-axis",
            "--calibration-profile",
            "profile.json",
            "--max-measured-linear-speed-m-s",
            "0.101",
            "--execute",
            "--confirm",
            probe.FLOAT_AXIS_CONFIRM_TEXT,
        ]
    )

    with pytest.raises(RuntimeError, match="must be in"):
        probe._validate_args(args)
