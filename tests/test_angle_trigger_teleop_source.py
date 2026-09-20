from __future__ import annotations

import argparse
import inspect
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from teleop_runtime.action import HandActionSpec
from teleop_runtime.retargeters import ScalarToGripperRetargeter, ScalarToGripperRetargeterConfig
from teleop_runtime.schemas import ObservationChannel, TeleopObservation
from teleop_runtime.sources.hand import AngleTriggerSource
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.real_teleop import hand_cli
from teleop_runtime.real_teleop import cli_runtime_config
from teleop_runtime.real_teleop.device_retry_runtime import (
    build_device_retry_runtime,
)


class _FakeReader:
    def __init__(self) -> None:
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def read(self) -> dict[str, float]:
        assert self.opened
        return {"left": 0.25, "right": 0.75}


class _FakeQ6Source(real_teleop.HandQ6Source):
    def __init__(self, *, valid: bool = True) -> None:
        self.started = False
        self.closed = False
        self.valid = bool(valid)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        now = time.time()
        return {
            f"{side}_hand_q6": ObservationChannel(
                channel_name=f"{side}_hand_q6",
                capability="joint_vector",
                side=side,
                value=np.full(6, 0.25, dtype=np.float32),
                frame_name=None,
                timestamp=now,
                source_name="fake_q6",
                valid=self.valid,
            )
            for side in ("left", "right")
        }


def _gripper_retargeter(side: str, *, q6: bool) -> ScalarToGripperRetargeter:
    return ScalarToGripperRetargeter(
        ScalarToGripperRetargeterConfig(
            side=side,
            input_channel_name=(
                f"{side}_hand_q6" if q6 else f"{side}_angle_trigger"
            ),
            source_dim=2 if q6 else None,
            alpha=1.0,
        )
    )


def test_angle_trigger_source_publishes_per_side_scalar_channels() -> None:
    source = AngleTriggerSource(_FakeReader(), rate_hz=100.0)
    source.start()
    try:
        deadline = time.monotonic() + 0.5
        channels = source.get_latest_channels()
        while not channels and time.monotonic() < deadline:
            time.sleep(0.01)
            channels = source.get_latest_channels()
    finally:
        source.close()

    assert channels["left_angle_trigger"].capability == "scalar"
    assert channels["left_angle_trigger"].valid is True
    assert channels["left_angle_trigger"].value == 0.25
    assert channels["right_angle_trigger"].value == 0.75


def test_angle_trigger_source_publishes_only_declared_single_side() -> None:
    source = AngleTriggerSource(
        _FakeReader(),
        rate_hz=100.0,
        active_sides=("right",),
    )
    source.start()
    try:
        deadline = time.monotonic() + 0.5
        channels = source.get_latest_channels()
        while not channels and time.monotonic() < deadline:
            time.sleep(0.01)
            channels = source.get_latest_channels()
    finally:
        source.close()

    assert source.active_sides == ("right",)
    assert set(channels) == {"right_angle_trigger"}
    assert channels["right_angle_trigger"].value == 0.75


def test_single_side_missing_reader_value_publishes_invalid_without_deadlock() -> None:
    class MissingRightReader(_FakeReader):
        def read(self) -> dict[str, float]:
            assert self.opened
            return {"left": 0.25}

    source = AngleTriggerSource(
        MissingRightReader(),
        rate_hz=100.0,
        active_sides=("right",),
    )
    source.start()
    try:
        deadline = time.monotonic() + 0.5
        channel = None
        while channel is None and time.monotonic() < deadline:
            time.sleep(0.01)
            channel = source.get_latest_channels().get(
                "right_angle_trigger"
            )
        assert channel is not None
        assert channel.valid is False
        assert "missing right" in channel.metadata["read_error"]
    finally:
        source.close()


def test_transient_read_failure_retries_open_connection_before_rediscovery() -> None:
    class FlakyReader(_FakeReader):
        def __init__(self) -> None:
            super().__init__()
            self.open_count = 0
            self.close_count = 0
            self.read_count = 0

        def open(self) -> None:
            super().open()
            self.open_count += 1

        def close(self) -> None:
            super().close()
            self.close_count += 1

        def read(self) -> dict[str, float]:
            assert self.opened
            self.read_count += 1
            if self.read_count == 1:
                raise RuntimeError("one delayed response")
            return {"right": 0.75}

    reader = FlakyReader()
    source = AngleTriggerSource(
        reader,
        rate_hz=100.0,
        active_sides=("right",),
        reconnect_after_consecutive_errors=3,
    )
    source.start()
    try:
        deadline = time.monotonic() + 0.5
        channel = None
        while time.monotonic() < deadline:
            channel = source.get_latest_channels().get("right_angle_trigger")
            if channel is not None and channel.valid:
                break
            time.sleep(0.005)
        assert channel is not None and channel.valid
        assert channel.value == 0.75
        assert reader.read_count >= 2
        assert reader.open_count == 1
        assert reader.close_count == 0
    finally:
        source.close()

    assert reader.close_count == 1


def test_scalar_retargeter_accepts_angle_trigger_scalar() -> None:
    retargeter = ScalarToGripperRetargeter(
        ScalarToGripperRetargeterConfig(
            side="left",
            input_channel_name="left_angle_trigger",
            source_dim=None,
            alpha=1.0,
            input_min=0.0,
            input_max=1.0,
        )
    )
    observation = TeleopObservation(
        timestamp=1.0,
        channels={
            "left_angle_trigger": ObservationChannel(
                channel_name="left_angle_trigger",
                capability="scalar",
                side="left",
                value=0.25,
                frame_name=None,
                timestamp=1.0,
                source_name="angle_trigger",
                valid=True,
            )
        },
    )

    command = retargeter.retarget(observation)
    gripper = command.commands["left_hand"]
    np.testing.assert_allclose(gripper.value, [0.25, 0, 0, 0, 0, 0])
    assert gripper.metadata["source_dim"] is None
    assert gripper.metadata["hand_action_spec"] == HandActionSpec.dmtac_gripper("left").to_metadata()


def test_gripper_input_selection_prefers_q6_and_falls_back_to_angle_trigger() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=False,
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": []},
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))
    assert args.gripper_input_source == "angle_trigger"
    assert "detected only 0 q6" in args.q6_startup_error.lower()

    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": [{}, {}]},
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))
    assert args.gripper_input_source == "q6"
    assert args.q6_startup_error == ""

    angle_args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=True,
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        side_effect=pytest.fail,
    ):
        real_teleop._resolve_gripper_input_source(angle_args, ("left", "right"))
    assert angle_args.gripper_input_source == "angle_trigger"


def test_gripper_input_source_cli_selector_preserves_legacy_flag_and_default() -> None:
    parser = argparse.ArgumentParser()
    hand_cli.add_hand_backend_arguments(parser)

    default_args = parser.parse_args([])
    assert default_args.gripper_input_source is None
    assert default_args.gripper_angle_trigger is False

    angle_args = parser.parse_args(
        ["--gripper-input-source", "angle_trigger"]
    )
    assert angle_args.gripper_input_source == "angle_trigger"
    assert angle_args.gripper_angle_trigger is False

    q6_args = parser.parse_args(["--gripper-input-source", "q6"])
    assert q6_args.gripper_input_source == "q6"
    assert q6_args.gripper_angle_trigger is False

    none_args = parser.parse_args(["--gripper-input-source", "none"])
    assert none_args.gripper_input_source == "none"
    assert none_args.gripper_angle_trigger is False

    legacy_args = parser.parse_args(["--gripper-angle-trigger"])
    assert legacy_args.gripper_input_source == "angle_trigger"
    assert legacy_args.gripper_angle_trigger is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--gripper-input-source"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--gripper-input-source", "q6", "--gripper-angle-trigger"]
        )


def test_explicit_angle_trigger_selection_skips_q6_probe() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="angle_trigger",
        gripper_angle_trigger=False,
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        side_effect=pytest.fail,
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))

    assert args.gripper_input_source == "angle_trigger"
    assert args._gripper_input_source_request == "angle_trigger"


def test_explicit_no_input_selection_skips_q6_probe_and_assignment_load() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="none",
        gripper_angle_trigger=False,
        active_sides=("left", "right"),
        q6_default_config=True,
        q6_config="latest",
        q6_config_required=True,
        left_hand_port="",
        right_hand_port="",
    )
    with (
        patch(
            "hardware.teleop_hand.q6_assignment.scan_q6_devices",
            side_effect=pytest.fail,
        ),
        patch.object(
            cli_runtime_config,
            "load_latest_q6_ports",
            side_effect=pytest.fail,
        ),
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))
        real_teleop._apply_default_q6_ports(args)

    assert args.gripper_input_source == "none"
    assert args._gripper_input_source_request == "none"
    assert args.q6_startup_error == ""
    assert args.q6_config_loaded_ports == {}


def test_no_input_source_has_no_gripper_input_channel() -> None:
    components = SimpleNamespace(gripper_input_source="none")

    assert real_teleop._gripper_input_channel_name(components, "right") is None


def test_no_input_source_does_not_require_q6_in_gui_preflight() -> None:
    args = SimpleNamespace(
        gui_requested=True,
        device_preflight=True,
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="none",
        visualize_tactile=False,
        dmtac_tactile=False,
        no_camera=True,
    )
    captured: dict[str, object] = {}

    def prepare(options):
        captured["require_q6"] = options.require_q6
        captured["require_dmtac_gripper"] = options.require_dmtac_gripper
        return SimpleNamespace(ok=True, messages=(), q6_ports={}, dmtac_assignment=None)

    parser = SimpleNamespace(error=lambda message: pytest.fail(message))
    with patch.object(cli_runtime_config, "prepare_devices_for_teleop", prepare):
        real_teleop._run_gui_device_preflight(
            args,
            active_sides=("left", "right"),
            parser=parser,
        )

    assert captured == {"require_q6": False, "require_dmtac_gripper": True}


def test_explicit_q6_selection_fails_closed_when_q6_is_missing() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="q6",
        gripper_angle_trigger=False,
        gui_requested=False,
        hardware_admission_mode="strict",
    )
    with (
        patch(
            "hardware.teleop_hand.q6_assignment.scan_q6_devices",
            return_value={"q6_devices": []},
        ),
        pytest.raises(ValueError, match="Detected only 0 Q6"),
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))


def test_explicit_q6_selection_stays_q6_for_gui_retry() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="q6",
        gripper_angle_trigger=False,
        gui_requested=True,
        hardware_admission_mode="capability",
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": []},
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))

    assert args.gripper_input_source == "q6"
    assert "detected only 0 q6" in args.q6_startup_error.lower()


def test_partial_q6_isolated_for_gui_capability_admission() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=False,
        gui_requested=True,
        hardware_admission_mode="capability",
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": [{}]},
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))

    assert args.gripper_input_source == "angle_trigger"
    assert "detected only 1 q6" in args.q6_startup_error.lower()


def test_partial_q6_remains_fail_closed_outside_gui_capability_mode() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=False,
        gui_requested=False,
        hardware_admission_mode="strict",
    )
    with (
        patch(
            "hardware.teleop_hand.q6_assignment.scan_q6_devices",
            return_value={"q6_devices": [{}]},
        ),
        pytest.raises(ValueError, match="Detected only 1 Q6"),
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))


def test_q6_scanner_value_error_isolated_only_for_gui_capability_mode() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=False,
        gui_requested=True,
        hardware_admission_mode="capability",
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        side_effect=ValueError("malformed Q6 scan"),
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))

    assert args.gripper_input_source == "angle_trigger"
    assert "malformed Q6 scan" in args.q6_startup_error


def test_inactive_arm_ip_is_accepted_when_active_sides_select_one_arm() -> None:
    args = SimpleNamespace(
        swap_arm_sides=False,
        left_arm_ip="192.168.101.21",
        right_arm_ip="192.168.1.20",
    )

    real_teleop._validate_required_arm_ips(args, ("left",))


def test_gui_q6_preflight_failure_isolated_for_runtime_retry() -> None:
    args = SimpleNamespace(
        gui_requested=True,
        hardware_admission_mode="capability",
        device_preflight=True,
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="q6",
        visualize_tactile=False,
        dmtac_tactile=None,
        no_camera=True,
    )
    parser = SimpleNamespace(error=lambda message: pytest.fail(message))
    with patch.object(
        cli_runtime_config,
        "prepare_devices_for_teleop",
        side_effect=real_teleop.DevicePreflightError(
            "Q6 count is insufficient: detected 1, expected 2."
        ),
    ):
        real_teleop._run_gui_device_preflight(
            args,
            active_sides=("left", "right"),
            parser=parser,
        )

    assert args.gripper_input_source == "angle_trigger"
    assert "Q6 count is insufficient" in args.q6_startup_error


def test_gui_dmtac_preflight_failure_does_not_exit_capability_startup() -> None:
    args = SimpleNamespace(
        gui_requested=True,
        hardware_admission_mode="capability",
        device_preflight=True,
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="none",
        visualize_tactile=False,
        dmtac_tactile=False,
        no_camera=True,
    )
    parser = SimpleNamespace(error=lambda message: pytest.fail(message))
    with patch.object(
        cli_runtime_config,
        "prepare_devices_for_teleop",
        side_effect=real_teleop.DevicePreflightError(
            "DMTAC right gripper override '/dev/ttyUSB0' did not produce a successful read-only probe"
        ),
    ):
        real_teleop._run_gui_device_preflight(
            args,
            active_sides=("left", "right"),
            parser=parser,
        )

    assert "DMTAC right gripper override" in args.device_preflight_startup_error


def test_gui_required_q6_assignment_failure_isolated_for_runtime_retry() -> None:
    args = SimpleNamespace(
        active_sides=("left", "right"),
        gui_requested=True,
        hardware_admission_mode="capability",
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="q6",
        q6_default_config=True,
        q6_config="latest",
        q6_config_required=True,
        left_hand_port="",
        right_hand_port="",
    )
    parser = SimpleNamespace(error=lambda message: pytest.fail(message))
    with patch.object(
        cli_runtime_config,
        "load_latest_q6_ports",
        side_effect=FileNotFoundError("missing latest assignment"),
    ):
        real_teleop._apply_default_q6_ports(args, parser=parser)

    assert args.gripper_input_source == "angle_trigger"
    assert "missing latest assignment" in args.q6_startup_error


def test_default_q6_assignment_merges_preflight_provenance_for_missing_side() -> None:
    args = SimpleNamespace(
        active_sides=("left", "right"),
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_input_source="q6",
        q6_default_config=True,
        q6_config="latest",
        q6_config_required=True,
        q6_config_loaded_ports={"left": "/dev/q6-preflight-left"},
        left_hand_port="/dev/q6-preflight-left",
        right_hand_port="",
    )
    with patch.object(
        cli_runtime_config,
        "load_latest_q6_ports",
        return_value={"right": "/dev/q6-default-right"},
    ) as load_latest:
        real_teleop._apply_default_q6_ports(args)

    load_latest.assert_called_once_with(
        active_sides=("right",),
        config="latest",
        allow_missing=False,
    )
    assert args.q6_config_loaded_ports == {
        "left": "/dev/q6-preflight-left",
        "right": "/dev/q6-default-right",
    }
    assert args.left_hand_port == "/dev/q6-preflight-left"
    assert args.right_hand_port == "/dev/q6-default-right"


def test_runtime_reconnect_promotes_angle_trigger_to_q6_atomically() -> None:
    angle_source = AngleTriggerSource(_FakeReader(), rate_hz=100.0)
    old_left = _gripper_retargeter("left", q6=False)
    old_right = _gripper_retargeter("right", q6=False)
    session = SimpleNamespace(
        sources=[SimpleNamespace(name="other"), angle_source],
        retargeters=[old_left, old_right],
    )
    components = SimpleNamespace(
        session=session,
        active_sides=("left", "right"),
        gripper_input_source="angle_trigger",
        left_gripper_retargeter=old_left,
        right_gripper_retargeter=old_right,
    )
    args = SimpleNamespace(
        gripper_input_source="angle_trigger",
        q6_config_loaded_ports={},
        left_hand_port="",
        right_hand_port="",
    )
    q6_source = _FakeQ6Source()

    with (
        patch.object(
            real_teleop,
            "_discover_runtime_q6_ports",
            return_value={"left": "/dev/q6-left", "right": "/dev/q6-right"},
        ),
        patch.object(
            real_teleop.HandQ6Source,
            "from_bimanual",
            return_value=q6_source,
        ),
        patch.object(
            real_teleop,
            "_q6_gripper_retargeter_for_side",
            side_effect=lambda _args, side: _gripper_retargeter(side, q6=True),
        ),
    ):
        promoted, message = (
            real_teleop._promote_angle_trigger_gripper_input_to_q6(
                args,
                components,
                ready_timeout_sec=0.0,
            )
        )

    assert promoted is True
    assert "替换 angle trigger" in message
    assert q6_source.started is True
    assert q6_source.closed is False
    assert angle_source not in session.sources
    assert q6_source in session.sources
    assert components.gripper_input_source == "q6"
    assert components.left_gripper_retargeter.config.input_channel_name == "left_hand_q6"
    assert components.right_gripper_retargeter.config.input_channel_name == "right_hand_q6"
    assert session.retargeters == [
        components.left_gripper_retargeter,
        components.right_gripper_retargeter,
    ]
    assert args.gripper_input_source == "q6"
    assert args.left_hand_port == "/dev/q6-left"
    assert args.right_hand_port == "/dev/q6-right"


def test_runtime_q6_promotion_failure_keeps_angle_trigger_fallback() -> None:
    angle_source = AngleTriggerSource(_FakeReader(), rate_hz=100.0)
    old_left = _gripper_retargeter("left", q6=False)
    old_right = _gripper_retargeter("right", q6=False)
    original_sources = [angle_source]
    original_retargeters = [old_left, old_right]
    session = SimpleNamespace(
        sources=list(original_sources),
        retargeters=list(original_retargeters),
    )
    components = SimpleNamespace(
        session=session,
        active_sides=("left", "right"),
        gripper_input_source="angle_trigger",
        left_gripper_retargeter=old_left,
        right_gripper_retargeter=old_right,
    )
    args = SimpleNamespace(
        gripper_input_source="angle_trigger",
        q6_config_loaded_ports={},
        left_hand_port="",
        right_hand_port="",
    )
    q6_source = _FakeQ6Source(valid=False)

    with (
        patch.object(
            real_teleop,
            "_discover_runtime_q6_ports",
            return_value={"left": "/dev/q6-left", "right": "/dev/q6-right"},
        ),
        patch.object(
            real_teleop.HandQ6Source,
            "from_bimanual",
            return_value=q6_source,
        ),
        pytest.raises(RuntimeError, match="did not become ready"),
    ):
        real_teleop._promote_angle_trigger_gripper_input_to_q6(
            args,
            components,
            ready_timeout_sec=0.0,
        )

    assert q6_source.closed is True
    assert session.sources == original_sources
    assert session.retargeters == original_retargeters
    assert components.gripper_input_source == "angle_trigger"
    assert components.left_gripper_retargeter is old_left
    assert components.right_gripper_retargeter is old_right
    assert args.gripper_input_source == "angle_trigger"


def test_runtime_q6_promotion_recovers_when_angle_trigger_source_was_missing() -> None:
    old_left = _gripper_retargeter("left", q6=False)
    old_right = _gripper_retargeter("right", q6=False)
    session = SimpleNamespace(
        # Capability-admission startup can leave the configured fallback
        # contract alive while omitting a source whose hardware init failed.
        sources=[],
        retargeters=[old_left, old_right],
    )
    components = SimpleNamespace(
        session=session,
        active_sides=("left", "right"),
        gripper_input_source="angle_trigger",
        left_gripper_retargeter=old_left,
        right_gripper_retargeter=old_right,
    )
    args = SimpleNamespace(
        gripper_input_source="angle_trigger",
        q6_config_loaded_ports={},
        left_hand_port="",
        right_hand_port="",
    )
    q6_source = _FakeQ6Source()

    with (
        patch.object(
            real_teleop,
            "_discover_runtime_q6_ports",
            return_value={"left": "/dev/q6-left", "right": "/dev/q6-right"},
        ),
        patch.object(
            real_teleop.HandQ6Source,
            "from_bimanual",
            return_value=q6_source,
        ),
        patch.object(
            real_teleop,
            "_q6_gripper_retargeter_for_side",
            side_effect=lambda _args, side: _gripper_retargeter(side, q6=True),
        ),
    ):
        promoted, message = real_teleop._promote_angle_trigger_gripper_input_to_q6(
            args,
            components,
            ready_timeout_sec=0.0,
        )

    assert promoted is True
    assert "fallback 已补齐" in message
    assert q6_source.started is True
    assert q6_source in session.sources
    assert components.gripper_input_source == "q6"
    assert components.left_gripper_retargeter.config.input_channel_name == "left_hand_q6"
    assert components.right_gripper_retargeter.config.input_channel_name == "right_hand_q6"
    assert args.gripper_input_source == "q6"


def test_runtime_q6_promotion_keeps_dmtac_retargeters_out_of_session() -> None:
    """DMTAC grippers remain on the direct dispatch path after promotion."""

    old_left = _gripper_retargeter("left", q6=False)
    old_right = _gripper_retargeter("right", q6=False)
    arm_retargeter = object()
    session = SimpleNamespace(
        # This mirrors the real DMTAC topology: session retargeters contain
        # arm/dual-arm producers, while ScalarToGripperRetargeters are held
        # by components and dispatched directly by gripper_control.py.
        sources=[],
        retargeters=[arm_retargeter],
    )
    components = SimpleNamespace(
        session=session,
        active_sides=("left", "right"),
        gripper_input_source="angle_trigger",
        left_gripper_retargeter=old_left,
        right_gripper_retargeter=old_right,
    )
    args = SimpleNamespace(
        gripper_input_source="angle_trigger",
        q6_config_loaded_ports={},
        left_hand_port="",
        right_hand_port="",
    )
    q6_source = _FakeQ6Source()

    with (
        patch.object(
            real_teleop,
            "_discover_runtime_q6_ports",
            return_value={"left": "/dev/q6-left", "right": "/dev/q6-right"},
        ),
        patch.object(
            real_teleop.HandQ6Source,
            "from_bimanual",
            return_value=q6_source,
        ),
        patch.object(
            real_teleop,
            "_q6_gripper_retargeter_for_side",
            side_effect=lambda _args, side: _gripper_retargeter(side, q6=True),
        ),
    ):
        promoted, _message = real_teleop._promote_angle_trigger_gripper_input_to_q6(
            args,
            components,
            ready_timeout_sec=0.0,
        )

    assert promoted is True
    assert session.retargeters == [arm_retargeter]
    assert components.left_gripper_retargeter not in session.retargeters
    assert components.right_gripper_retargeter not in session.retargeters


def test_device_retry_wires_angle_trigger_fallback_to_q6_promotion() -> None:
    source = inspect.getsource(build_device_retry_runtime)
    retry_start = source.index("def retry_failed_devices()")
    retry_end = source.index(
        "return DeviceRetryRuntime(",
        retry_start,
    )
    retry_source = source[retry_start:retry_end]

    assert "_promote_angle_trigger_gripper_input_to_q6(" in retry_source
    assert "clutch_input._episode_busy()" in retry_source
    assert "recording_active()" not in retry_source
    assert "episode_start_countdown_deadline" not in retry_source
    assert "clutch_input.hold_active()" in retry_source


def test_successful_q6_retry_clears_recoverable_input_failures() -> None:
    startup_errors = {
        "gripper.input": "Q6UnavailableError: missing",
        "angle_trigger": "AngleTriggerError: missing fallback",
    }
    published: list[dict[str, object]] = []
    messages: list[str] = []
    clutch = SimpleNamespace(
        _teleop_started=False,
        _gripper_control_enabled=True,
        _episode_busy=lambda: False,
        hold_active=lambda: False,
        device_retry_admission=lambda: (True, ""),
        _current_active_controls=lambda: ["device_retry"],
        publish_gripper_status=lambda **_kwargs: None,
    )
    components = SimpleNamespace(
        gripper_input_source="angle_trigger",
        left_state_backend=None,
        right_state_backend=None,
        left_hand_backend=object(),
        right_hand_backend=object(),
    )
    args = SimpleNamespace(
        q6_config_required=True,
        q6_startup_error=startup_errors["gripper.input"],
        angle_trigger_startup_error=startup_errors["angle_trigger"],
        foot_pad_requested=False,
        ignore_camera=(),
    )
    ui = SimpleNamespace(
        publish=lambda **updates: published.append(updates),
        append_message=lambda _kind, message: messages.append(str(message)),
    )
    runtime = build_device_retry_runtime(
        args=args,
        components=components,
        control_ui=ui,
        clutch_input=clutch,
        camera_sources={},
        vive_source=None,
        startup_device_errors=startup_errors,
        position_force_guard_requested=False,
        dataset_recorder_provider=lambda: None,
        set_dataset_recorder=lambda _value: None,
        foot_pad_input_provider=lambda: None,
        set_foot_pad_input=lambda _value: None,
        dataset_recorder_type=object,
        dmtac_gripper_backend_type=object,
        foot_pad_input_type=object,
        foot_pad_control_mode="foot-pad",
        has_dmtac_gripper_backend=lambda _components: True,
        promote_angle_trigger_gripper_input_to_q6=lambda *_args, **_kwargs: (
            True,
            "Q6 reconnected",
        ),
        log=lambda *_args, **_kwargs: None,
    )

    runtime.retry_failed_devices()

    assert startup_errors == {}
    assert args.q6_startup_error == ""
    assert args.angle_trigger_startup_error == ""
    assert clutch._gripper_control_enabled is False
    assert any("Q6 reconnected" in message for message in messages)
    dataset_writer_status = published[-1]["hardware_status"]["dataset.writer"]
    assert dataset_writer_status["online"] is False
    assert dataset_writer_status["health"] == "deferred"


def test_manual_foot_pad_retry_updates_status_provider_atomically() -> None:
    """A successful pedal retry must survive the next status synchronization."""

    class FootPad:
        def __init__(self, *, online: bool) -> None:
            self.is_online = online
            self.closed = False
            self.foot_pad_count = 2

        def close(self) -> None:
            self.closed = True
            self.is_online = False

    old_input = FootPad(online=False)
    replacement = FootPad(online=True)

    class FootPadType:
        @classmethod
        def open_online(cls):
            return replacement

    current = {"value": old_input}
    startup_errors = {"foot_pad": "FootPadUnavailableError: missing"}
    published: list[dict[str, object]] = []
    foot_pad_test_messages: list[str] = []

    class Ui:
        def publish(self, **updates: object) -> None:
            published.append(updates)

        def append_message(self, *_args: object) -> None:
            return None

    clutch = SimpleNamespace(
        _teleop_started=False,
        _last_foot_pad_online=False,
        _gripper_control_enabled=False,
        foot_pad_input=old_input,
        _foot_pad_test_active=True,
        _foot_pad_test_completed=True,
        _foot_pad_test_seen_events={"fix_arm_left"},
        _foot_pad_test_last_event="旧连接事件",
        _foot_pad_test_status="旧连接已完成",
        _episode_busy=lambda: False,
        hold_active=lambda: False,
        device_retry_admission=lambda: (True, ""),
        _current_active_controls=lambda: ["device_retry"],
        publish_gripper_status=lambda **_kwargs: None,
        _foot_pad_test_ui_state=lambda: {
            "foot_pad_available": bool(
                getattr(clutch.foot_pad_input, "is_online", False)
            ),
            "foot_pad_test_completed": bool(clutch._foot_pad_test_completed),
            "foot_pad_test_status": clutch._foot_pad_test_status,
        },
        _publish_foot_pad_test_state=lambda **kwargs: foot_pad_test_messages.append(
            str(kwargs.get("message", ""))
        ),
    )
    components = SimpleNamespace(
        gripper_input_source="",
        left_state_backend=None,
        right_state_backend=None,
        left_hand_backend=None,
        right_hand_backend=None,
    )
    args = SimpleNamespace(
        foot_pad_requested=True,
        foot_pad_startup_error=startup_errors["foot_pad"],
        special_arm_control_mode="manual",
        ignore_camera=(),
    )

    runtime = build_device_retry_runtime(
        args=args,
        components=components,
        control_ui=Ui(),
        clutch_input=clutch,
        camera_sources={},
        vive_source=None,
        startup_device_errors=startup_errors,
        position_force_guard_requested=False,
        dataset_recorder_provider=lambda: None,
        set_dataset_recorder=lambda _value: None,
        foot_pad_input_provider=lambda: current["value"],
        set_foot_pad_input=lambda value: current.__setitem__("value", value),
        dataset_recorder_type=object,
        dmtac_gripper_backend_type=object,
        foot_pad_input_type=FootPadType,
        foot_pad_control_mode="foot-pad",
        has_dmtac_gripper_backend=lambda _components: False,
        promote_angle_trigger_gripper_input_to_q6=lambda *_args, **_kwargs: (
            False,
            "not used",
        ),
        log=lambda *_args, **_kwargs: None,
    )

    runtime.retry_failed_devices()
    assert old_input.closed is True
    assert current["value"] is replacement
    assert clutch.foot_pad_input is replacement
    assert clutch._foot_pad_test_completed is False
    assert clutch._foot_pad_test_seen_events == set()
    assert clutch._foot_pad_test_last_event == ""
    assert "脚踏板安全测试" in clutch._foot_pad_test_status
    assert any("脚踏板安全测试" in message for message in foot_pad_test_messages)

    # The periodic synchronizer must observe the replacement, not the stale
    # offline placeholder captured before the manual retry.
    runtime.sync_foot_pad_runtime_status()
    assert "foot_pad" not in startup_errors
    assert args.foot_pad_startup_error == ""
    assert published
