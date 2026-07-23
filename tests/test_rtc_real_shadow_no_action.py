from __future__ import annotations

import argparse
import json
import threading

import numpy as np
import pytest

from deployments.openpi.client import real as openpi_real
from robot_motion.runtime.contracts import EffectorCommand
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_impedance import (
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from teleop_runtime.backends.grippers.dmtac import DMTacGripperBackend, DMTacGripperBackendConfig
from teleop_runtime.backends.hands.inspire_tac import (
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from teleop_runtime.deploy import run_policy


RAW_PROFILE = "raw_rtc_final_chunk_15_30_60_120_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser


def _raw_args(*extra: str) -> tuple[argparse.Namespace, tuple[str, ...]]:
    argv = (
        "--rtc-profile",
        RAW_PROFILE,
        "--policy-artifact-json",
        "/tmp/rtc-real-shadow-artifact.json",
        *extra,
    )
    args = _parser().parse_args(list(argv))
    return args, argv


def test_real_shadow_is_explicit_and_rejects_live_endpoints_without_guard() -> None:
    args, argv = _raw_args("--left-arm-ip", "192.0.2.10")
    with pytest.raises(ValueError, match="requires explicit --rtc-real-shadow"):
        run_policy.validate_args(args, raw_argv=argv)

    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--left-arm-ip",
        "192.0.2.10",
        "--active-sides",
        "left",
        "--end-effector-sides",
        "none",
        "--arm-execution-profile",
        "impedance_40hz",
    )
    run_policy.validate_args(args, raw_argv=argv)
    assert args.rtc_real_shadow is True
    assert args.send_actions is False
    assert args.position_chunk_lookahead is False
    # RTC's position clock is irrelevant to impedance output; the existing
    # 40 Hz impedance execution profile remains valid for observation shadow.
    assert args.arm_control_mode == "impedance"
    assert args.arm_control_hz == 40.0


def test_rtc_validity_and_bounded_close_overrides_are_explicit_only() -> None:
    args, argv = _raw_args(
        "--rtc-request-validity-steps",
        "50",
        "--rtc-worker-stop-timeout-sec",
        "2.0",
    )
    run_policy.validate_args(args, raw_argv=argv)
    assert args.rtc_request_validity_steps == 50
    assert args.rtc_worker_stop_timeout_sec == 2.0

    ordinary = _parser().parse_args(
        [
            "--rtc-request-validity-steps",
            "50",
            "--rtc-worker-stop-timeout-sec",
            "2.0",
        ]
    )
    with pytest.raises(ValueError, match="require an explicit --rtc-profile"):
        run_policy.validate_args(ordinary)


def test_openpi_real_shadow_profile_drops_qpos_init_authority() -> None:
    ordinary = openpi_real._rm65_bimanual_dmtac_args()
    shadow = openpi_real._rm65_bimanual_dmtac_args(rtc_real_shadow=True)
    assert "--policy-qpos-init-dataset" in ordinary
    assert "--policy-qpos-init-dataset" not in shadow
    assert "--policy-qpos-init-collision-check" not in shadow
    assert "--wait-for-ready-confirm" in ordinary
    assert "--auto-start" in shadow
    assert "--no-wait-for-ready-confirm" in shadow
    assert "--dmtac-no-tactile" in shadow


def test_debug_logger_serializes_worker_exchange_events_without_interleaving(tmp_path) -> None:
    logger = run_policy.DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path,
        active_sides=("left",),
    )
    assert logger.path is not None

    def write_events(worker_index: int) -> None:
        for event_index in range(12):
            logger.log_event(
                "rtc_transport_exchange",
                {
                    "worker_index": worker_index,
                    "event_index": event_index,
                    "non_authoritative": True,
                },
            )

    threads = [
        threading.Thread(target=write_events, args=(index,))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    path = logger.path
    logger.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 48
    assert all(record["event"] == "rtc_transport_exchange" for record in records)
    assert all(record["non_authoritative"] is True for record in records)


def test_real_shadow_arm_backend_configs_disable_all_output_side_effects() -> None:
    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--left-arm-ip",
        "192.0.2.10",
        "--active-sides",
        "left",
        "--end-effector-sides",
        "none",
        "--arm-execution-profile",
        "impedance_40hz",
    )
    run_policy.validate_args(args, raw_argv=argv)
    backends = run_policy.build_arm_backends(
        args,
        ("left",),
        control_hz=args.arm_control_hz,
        action_hz=args.action_hz,
    )
    config = backends["left"].config
    assert config.observation_only is True
    assert config.impedance_overrides["canfd_follow"] is False

    position_args, position_argv = _raw_args(
        "--rtc-real-shadow",
        "--left-arm-ip",
        "192.0.2.10",
        "--active-sides",
        "left",
        "--end-effector-sides",
        "none",
        "--arm-execution-profile",
        "position_15_60_120hz",
    )
    run_policy.validate_args(position_args, raw_argv=position_argv)
    position_backend = run_policy.build_arm_backends(
        position_args,
        ("left",),
        control_hz=position_args.arm_control_hz,
        action_hz=position_args.action_hz,
    )["left"]
    position_config = position_backend.config
    assert position_config.observation_only is True
    assert position_config.background_streaming_enabled is False
    assert position_config.force_position_control is False
    assert position_config.configure_singularity_thresholds is False
    assert position_config.canfd_follow is False


@pytest.mark.parametrize(
    "backend",
    [
        RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.0.2.10",
                side="left",
                connect_hardware=False,
                observation_only=True,
            )
        ),
        DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=False,
                observation_only=True,
                send_commands=False,
            )
        ),
        InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.11",
                side="left",
                connect_hardware=False,
                observation_only=True,
            )
        ),
    ],
)
def test_real_shadow_effector_backends_reject_command_dispatch(backend) -> None:
    command = EffectorCommand(
        effector_name=backend.name,
        side="left",
        command_type="joint_vector",
        value=np.zeros(6, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="observation-only"):
        backend.send_command(command)


def test_position_backend_observation_only_rejects_command_before_transport() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.10",
            side="left",
            connect_hardware=False,
            observation_only=True,
        )
    )
    command = EffectorCommand(
        effector_name=backend.name,
        side="left",
        command_type="joint_vector",
        value=np.zeros(6, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="observation-only"):
        backend.send_command(command)


def test_impedance_observation_only_uses_direct_feedback_without_controller_loop() -> None:
    class FakeController:
        _last_raw_observed_joint_deg = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
        _last_observed_joint_normalization = {"status": "exact"}

        def _read_state(self):
            return {
                "joint_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "pose": [0.1, 0.2, 0.3, 0.01, 0.02, 0.03],
            }

    backend = RM65ImpedanceArmBackend(
        RM65ImpedanceArmBackendConfig(
            arm_ip="192.0.2.10",
            side="left",
            connect_hardware=True,
            observation_only=True,
        )
    )
    backend.arm = object()
    backend._controller = FakeController()

    state = backend.refresh_state_from_hardware()

    assert state["observation_only"] is True
    assert state["hardware_read_success"] is True
    assert state["hardware_measurement_valid"] is True
    assert state["arm_state_source"] == "hardware_observation_only"
    np.testing.assert_allclose(
        state["hardware_qpos"],
        np.deg2rad(np.asarray([1, 2, 3, 4, 5, 6], dtype=float)),
    )
    assert state["impedance_enabled"] is False
    assert state["impedance_loop_state"] == "not_started"


def test_dmtac_observation_only_reconnect_skips_startup_register_writes() -> None:
    class FakeDevice:
        gripper = object()

        def clear_fault(self):
            raise AssertionError("clear_fault must not run in observation shadow")

        def set_speed(self, _value):
            raise AssertionError("set_speed must not run in observation shadow")

        def set_torque(self, _value):
            raise AssertionError("set_torque must not run in observation shadow")

    backend = DMTacGripperBackend(
        DMTacGripperBackendConfig(
            side="left",
            connect_hardware=True,
            connect_tactile=False,
            observation_only=True,
        )
    )
    backend.device = FakeDevice()
    backend._apply_gripper_start_config_locked(open_on_start=True)
