from __future__ import annotations

import argparse
import inspect
import json
import threading
from types import SimpleNamespace
from unittest.mock import call, patch

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
from teleop_runtime.deploy import camera_runtime as deploy_camera_runtime
from teleop_runtime.deploy import run_policy
from teleop_runtime.real_teleop import camera_runtime


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


def test_500ms_rtc_latency_bound_is_explicit_diagnostic_only() -> None:
    args, argv = _raw_args("--rtc-active-max-inference-latency-sec", "0.5")
    run_policy.validate_args(args, raw_argv=argv)
    assert args.rtc_active_max_inference_latency_sec == pytest.approx(0.5)

    ordinary = _parser().parse_args([])
    assert ordinary.rtc_active_max_inference_latency_sec == pytest.approx(0.4)


def test_bounded_rtc_acceptance_requires_explicit_profile_and_limit() -> None:
    args, argv = _raw_args(
        "--max-steps",
        "100",
        "--min-rtc-emissions-before-exit",
        "1",
    )
    run_policy.validate_args(args, raw_argv=argv)
    assert args.min_rtc_emissions_before_exit == 1

    unbounded, unbounded_argv = _raw_args(
        "--min-rtc-emissions-before-exit",
        "1",
    )
    with pytest.raises(ValueError, match="requires bounded --max-steps"):
        run_policy.validate_args(unbounded, raw_argv=unbounded_argv)

    ordinary = _parser().parse_args(
        [
            "--max-steps",
            "100",
            "--min-rtc-emissions-before-exit",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="requires an explicit --rtc-profile"):
        run_policy.validate_args(ordinary)


def test_retired_prediction_invalidation_is_not_called_from_deploy_loop() -> None:
    source = inspect.getsource(run_policy.main)
    assert "invalidate_in_flight_policy_prediction" not in source


def test_active_rtc_gap_uses_measured_qpos_hold_and_clears_position_plan() -> None:
    class _Arm:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def request_policy_inference_soft_hold(self, reason: str):
            self.reasons.append(reason)
            return {"active": True, "status": "activated"}

    class _Executor:
        def __init__(self) -> None:
            self.clear_count = 0

        def snapshot(self):
            return {"active": True}

        def clear(self) -> None:
            self.clear_count += 1

    left = _Arm()
    right = _Arm()
    executor = _Executor()

    receipts = run_policy._hold_active_rtc_gap(
        reason="rtc_plan_underrun",
        send_actions=True,
        arm_backends={"left": left, "right": right},
        position_deploy_executor=executor,
    )

    assert executor.clear_count == 1
    assert left.reasons == ["rtc_plan_underrun"]
    assert right.reasons == ["rtc_plan_underrun"]
    assert set(receipts) == {"left", "right"}


def test_active_rtc_gap_rejects_backend_without_measured_qpos_hold() -> None:
    class _UnsupportedArm:
        def __init__(self) -> None:
            self.hold_reasons: list[str] = []

        def hold_motion(self, reason: str) -> None:
            self.hold_reasons.append(reason)

    backend = _UnsupportedArm()
    with pytest.raises(RuntimeError, match="unsupported for sides: left"):
        run_policy._hold_active_rtc_gap(
            reason="rtc_producer_not_eligible",
            send_actions=True,
            arm_backends={"left": backend},
            position_deploy_executor=None,
        )
    assert backend.hold_reasons == [
        "rtc_producer_not_eligible:measured_qpos_hold_failed"
    ]


@pytest.mark.parametrize("receipt", ({}, {"active": False, "status": "skipped"}, {"active": True, "status": "unknown"}))
def test_active_rtc_gap_rejects_inactive_or_malformed_hold_receipt(receipt) -> None:
    class _Arm:
        def __init__(self):
            self.holds = []

        def request_policy_inference_soft_hold(self, reason):
            return receipt

        def hold_motion(self, reason):
            self.holds.append(reason)

    arm = _Arm()
    with pytest.raises(RuntimeError, match="Hold was not established"):
        run_policy._hold_active_rtc_gap(
            reason="rtc_plan_underrun", send_actions=True,
            arm_backends={"left": arm}, position_deploy_executor=None,
        )
    assert arm.holds == ["rtc_plan_underrun:measured_qpos_hold_failed"]


def test_active_rtc_gap_clear_failure_still_holds_both_arms() -> None:
    class _Executor:
        def snapshot(self):
            return {"active": True}

        def clear(self):
            raise RuntimeError("clear failed")

    class _Arm:
        def __init__(self):
            self.holds = []

        def hold_motion(self, reason):
            self.holds.append(reason)

    left, right = _Arm(), _Arm()
    with pytest.raises(RuntimeError, match="clear failed"):
        run_policy._hold_active_rtc_gap(
            reason="rtc_plan_underrun", send_actions=True,
            arm_backends={"left": left, "right": right},
            position_deploy_executor=_Executor(),
        )
    assert left.holds == right.holds == ["rtc_plan_underrun:measured_qpos_hold_failed"]


def test_active_rtc_gap_without_reason_does_not_touch_backends() -> None:
    assert run_policy._hold_active_rtc_gap(
        reason=None,
        send_actions=False,
        arm_backends={"left": object()}, position_deploy_executor=object(),
    ) == {}


def test_active_rtc_gap_dry_run_clears_position_queue_without_backend_hold() -> None:
    class _Executor:
        def __init__(self) -> None:
            self.clear_count = 0

        def snapshot(self):
            return {"active": True}

        def clear(self) -> None:
            self.clear_count += 1

    executor = _Executor()
    assert run_policy._hold_active_rtc_gap(
        reason="rtc_producer_not_eligible",
        send_actions=False,
        arm_backends={"left": object()},
        position_deploy_executor=executor,
    ) == {}
    assert executor.clear_count == 1


def test_active_rtc_gap_dry_run_clear_failure_does_not_hold_backend() -> None:
    class _Executor:
        def snapshot(self):
            return {"active": True}

        def clear(self) -> None:
            raise RuntimeError("dry-run clear failed")

    class _Arm:
        def __init__(self) -> None:
            self.holds = []

        def hold_motion(self, reason):
            self.holds.append(reason)

    arm = _Arm()
    with pytest.raises(RuntimeError, match="dry-run clear failed"):
        run_policy._hold_active_rtc_gap(
            reason="rtc_plan_underrun",
            send_actions=False,
            arm_backends={"left": arm},
            position_deploy_executor=_Executor(),
        )
    assert arm.holds == []


def test_active_rtc_latency_gate_discards_cold_sample_and_bounds_warm_max() -> None:
    pending = run_policy._rtc_active_latency_gate_result(
        [8.0, 0.20, 0.30],
        discard_completions=1,
        required_completions=3,
        maximum_latency_sec=0.4,
    )
    assert pending == (False, False, None)

    accepted = run_policy._rtc_active_latency_gate_result(
        [8.0, 0.20, 0.30, 0.399999],
        discard_completions=1,
        required_completions=3,
        maximum_latency_sec=0.4,
    )
    assert accepted == (True, True, 0.399999)

    boundary_rejected = run_policy._rtc_active_latency_gate_result(
        [8.0, 0.20, 0.30, 0.40],
        discard_completions=1,
        required_completions=3,
        maximum_latency_sec=0.4,
    )
    assert boundary_rejected == (True, False, 0.40)

    rejected = run_policy._rtc_active_latency_gate_result(
        [8.0, 0.20, 0.41, 0.30],
        discard_completions=1,
        required_completions=3,
        maximum_latency_sec=0.4,
    )
    assert rejected == (True, False, 0.41)


@pytest.mark.parametrize(
    "reason",
    ("rtc_preflight_timeout", "rtc_result_timeout", "request_deadline_expired"),
)
def test_recoverable_timeout_classifier_accepts_only_timeout_markers(reason: str) -> None:
    step = SimpleNamespace(
        lifecycle_transition=(
            SimpleNamespace(
                producer_transition=SimpleNamespace(
                    state=SimpleNamespace(last_transition_reason=reason),
                ),
            )
            if reason != "request_deadline_expired"
            else None
        ),
        result_transition=(
            SimpleNamespace(
                receipt=SimpleNamespace(reasons=(reason,)),
                merge_receipt=SimpleNamespace(reasons=(reason,)),
            )
            if reason == "request_deadline_expired"
            else None
        ),
        completion=None,
    )
    assert run_policy._rtc_recoverable_timeout_reason(step) == reason

    non_timeout = SimpleNamespace(
        lifecycle_transition=SimpleNamespace(
            producer_transition=SimpleNamespace(
                state=SimpleNamespace(last_transition_reason="policy_worker_fault"),
            ),
        ),
        result_transition=None,
        completion=None,
    )
    assert run_policy._rtc_recoverable_timeout_reason(non_timeout) is None


def test_rtc_position_executor_recovery_classifier_is_narrow() -> None:
    stale = {
        "fault_latched": True,
        "fault_stage": "reference_worker",
        "fault_type": "TimeoutError",
        "fault": (
            "15 Hz position keyframe stream is stale: "
            "age=0.251713s limit=0.250000s"
        ),
    }
    assert (
        run_policy._rtc_recoverable_position_executor_fault_reason(stale)
        == "rtc_position_reference_stale"
    )

    for rejected in (
        {**stale, "fault": "reference deadline burst exceeded"},
        {**stale, "fault_stage": "safety_gate"},
        {**stale, "fault_type": "RuntimeError"},
        {**stale, "fault_latched": False},
    ):
        assert run_policy._rtc_recoverable_position_executor_fault_reason(rejected) is None


@pytest.mark.parametrize(
    ("request_admitted_at", "result_admitted_at"),
    (
        (float("nan"), 10.0),
        (10.0, float("nan")),
        (float("inf"), 10.0),
        (10.0, float("inf")),
    ),
)
def test_active_rtc_completion_latency_rejects_nonfinite_timestamps(
    request_admitted_at: float,
    result_admitted_at: float,
) -> None:
    with pytest.raises(ValueError, match="timestamps must be finite"):
        run_policy._rtc_active_completion_latency_sec(
            request_admitted_at=request_admitted_at,
            result_admitted_at=result_admitted_at,
        )


def test_active_rtc_completion_latency_preserves_valid_elapsed_time() -> None:
    assert run_policy._rtc_active_completion_latency_sec(
        request_admitted_at=10.0,
        result_admitted_at=10.399999,
    ) == pytest.approx(0.399999)
    assert (
        run_policy._rtc_active_completion_latency_sec(
            request_admitted_at=10.0,
            result_admitted_at=10.0,
        )
        == 0.0
    )


def test_active_rtc_completion_latency_rejects_reversed_timestamps() -> None:
    with pytest.raises(ValueError, match="result admission predates request admission"):
        run_policy._rtc_active_completion_latency_sec(
            request_admitted_at=10.0,
            result_admitted_at=9.999999,
        )


def test_openpi_real_shadow_profile_preserves_qpos_init_authority() -> None:
    ordinary = openpi_real._rm65_bimanual_dmtac_args()
    shadow = openpi_real._rm65_bimanual_dmtac_args(rtc_real_shadow=True)
    active = openpi_real._rm65_bimanual_dmtac_args(
        rtc_real_shadow=True,
        send_actions=True,
    )
    assert "--policy-qpos-init-dataset" in ordinary
    position_index = ordinary.index("--arm-execution-profile")
    assert ordinary[position_index + 1] == "position_15_60_120hz"
    assert "--policy-qpos-init-dataset" in shadow
    assert "--policy-qpos-init-collision-check" in shadow
    assert "--wait-for-ready-confirm" in ordinary
    assert "--dmtac-no-tactile" in ordinary
    assert "--auto-start" in shadow
    assert "--no-wait-for-ready-confirm" in shadow
    assert "--wait-for-ready-confirm" in active
    assert "--no-wait-for-ready-confirm" not in active
    assert "--dmtac-no-tactile" in shadow
    assert "--camera-startup-hardware-reset" not in ordinary
    assert "--camera-start-stagger-sec" not in ordinary
    assert "--camera-startup-hardware-reset" not in shadow
    stagger_index = shadow.index("--camera-start-stagger-sec")
    assert shadow[stagger_index + 1] == "0.5"
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    tactile_opt_in = parser.parse_args([*ordinary, "--dmtac-with-tactile"])
    assert tactile_opt_in.dmtac_tactile is True


def test_default_openpi_active_profile_builds_rtc_capable_60hz_position_backends() -> None:
    args, argv = _raw_args(
        *openpi_real._rm65_bimanual_dmtac_args(rtc_real_shadow=True, send_actions=True),
        "--rtc-real-shadow", "--send-actions",
        "--policy-qpos-init-dataset", "dataset",
    )
    run_policy.validate_args(args, raw_argv=argv)
    assert args.arm_control_mode == "position"
    assert args.position_reference_hz == 60.0
    assert args.arm_control_hz == 120.0
    backends = run_policy.build_arm_backends(
        args, ("left", "right"), control_hz=args.arm_control_hz,
        action_hz=args.action_hz,
    )
    for backend in backends.values():
        assert isinstance(backend, RM65ArmBackend)
        assert backend.config.background_streaming_enabled
        assert backend.config.background_streaming_hz == 120.0
        assert callable(backend.request_policy_inference_soft_hold)
        assert backend.arm is None  # Constructor is not device startup.
        backend.close()


def test_active_real_model_rtc_rejects_no_ready_confirmation() -> None:
    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--send-actions",
        "--no-wait-for-ready-confirm",
        "--left-arm-ip",
        "192.0.2.10",
        "--active-sides",
        "left",
        "--end-effector-sides",
        "none",
        "--arm-execution-profile",
        "impedance_40hz",
        "--policy-qpos-init-dataset",
        "dataset",
    )
    with pytest.raises(ValueError, match="requires --wait-for-ready-confirm"):
        run_policy.validate_args(args, raw_argv=argv)


def test_real_shadow_manual_dmtac_ports_bypass_stale_assignment_without_command_authority() -> None:
    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--active-sides",
        "both",
        "--end-effector-sides",
        "both",
        "--enable-hands",
        "--hand-backend",
        "dmtac-gripper",
        "--dmtac-no-tactile",
        "--no-dmtac-use-config",
        "--require-dmtac-config",
        "--left-gripper-port",
        "/dev/ttyUSB0",
        "--right-gripper-port",
        "/dev/ttyUSB1",
    )
    run_policy.validate_args(args, raw_argv=argv)

    configs = run_policy._dmtac_backend_configs(args, ("left", "right"))

    assert set(configs) == {"left", "right"}
    assert configs["left"].gripper_port == "/dev/ttyUSB0"
    assert configs["right"].gripper_port == "/dev/ttyUSB1"
    assert all(config.observation_only for config in configs.values())
    assert all(config.send_commands is False for config in configs.values())
    assert all(config.clear_fault_on_start is False for config in configs.values())
    assert all(config.open_on_start is False for config in configs.values())
    assert all(config.connect_tactile is False for config in configs.values())


def test_deploy_camera_sources_use_collection_priority_order_with_stagger() -> None:
    started: list[str] = []

    class _Config:
        def __init__(self, serial: str, priority: int) -> None:
            self.serial = serial
            self.startup_hardware_reset = True
            self.startup_warmup_policy = "bounded_first_frame"
            self.metadata = {"startup_priority": priority}

    class _Source:
        startup_timing: dict[str, float] = {}

        def __init__(self, name: str, serial: str, priority: int) -> None:
            self.name = name
            self.config = _Config(serial, priority)

        def start(self) -> None:
            started.append(self.name)

    sources = {
        "front": _Source("front", "FRONT", 0),
        "left_wrist": _Source("left_wrist", "LEFT", 20),
        "right_wrist": _Source("right_wrist", "RIGHT", 10),
    }

    with patch.object(camera_runtime.time, "sleep") as sleep_mock:
        run_policy.start_camera_sources(sources, stagger_sec=0.5)

    assert started == ["front", "right_wrist", "left_wrist"]
    assert sleep_mock.call_args_list == [call(0.5), call(0.5)]


def test_deploy_camera_startup_reuses_collection_deferred_retry() -> None:
    calls: list[str] = []

    class _Source:
        startup_timing: dict[str, float] = {}

        def __init__(self, name: str, priority: int, *, fail_first: bool = False) -> None:
            self.name = name
            self.config = argparse.Namespace(
                metadata={"startup_priority": priority},
                startup_warmup_policy="bounded_first_frame",
            )
            self.fail_first = fail_first
            self.start_count = 0

        def start(self) -> None:
            self.start_count += 1
            calls.append(f"start:{self.name}:{self.start_count}")
            if self.fail_first and self.start_count == 1:
                raise RuntimeError(
                    "startup_warmup_failed:policy=bounded_first_frame:"
                    "no_frames_within_sec=5.000:attempts=5"
                )

        def close(self) -> None:
            calls.append(f"close:{self.name}")

    sources = {
        "front": _Source("front", 0, fail_first=True),
        "left_wrist": _Source("left_wrist", 20),
        "right_wrist": _Source("right_wrist", 10),
    }

    with patch.object(camera_runtime.time, "sleep") as sleep_mock:
        run_policy.start_camera_sources(sources, stagger_sec=0.5)

    assert calls == [
        "start:front:1",
        "start:right_wrist:1",
        "start:left_wrist:1",
        "close:front",
        "start:front:2",
    ]
    assert sleep_mock.call_args_list == [call(0.5), call(0.5), call(1.0)]


def test_deploy_camera_startup_fails_closed_after_collection_retry() -> None:
    class _Source:
        config = argparse.Namespace(
            metadata={"startup_priority": 0},
            startup_warmup_policy="bounded_first_frame",
        )
        startup_timing: dict[str, float] = {}

        def start(self) -> None:
            raise RuntimeError(
                "startup_warmup_failed:policy=bounded_first_frame:"
                "no_frames_within_sec=5.000:attempts=5"
            )

        def close(self) -> None:
            return None

    with patch.object(camera_runtime.time, "sleep"):
        with pytest.raises(
            RuntimeError,
            match="failed after the formal real-teleop deferred retry lifecycle",
        ):
            run_policy.start_camera_sources({"front": _Source()})


def test_deploy_camera_defaults_match_formal_collection_startup() -> None:
    args = _parser().parse_args([])

    assert args.camera_color_format == "bgr8"
    assert args.camera_startup_warmup_policy == "bounded_first_frame"
    assert args.camera_start_stagger_sec == camera_runtime.CAMERA_START_STAGGER_SEC


@pytest.mark.parametrize(
    (
        "warmup_policy",
        "require_complete",
        "restart_incomplete",
        "restart_limit",
        "reset_on_failure",
    ),
    [
        ("bounded_first_frame", True, True, 3, False),
        ("legacy", False, False, 0, True),
    ],
)
def test_deploy_single_camera_uses_reversible_stream_lifecycle_profile(
    warmup_policy: str,
    require_complete: bool,
    restart_incomplete: bool,
    restart_limit: int,
    reset_on_failure: bool,
) -> None:
    args = _parser().parse_args(
        [
            "--camera-mode",
            "single",
            "--camera-serial",
            "SERIAL",
            "--camera-startup-warmup-policy",
            warmup_policy,
        ]
    )

    source = run_policy.build_camera_sources(args)[args.camera_key]

    assert source.config.require_complete_frameset is require_complete
    assert source.config.restart_on_incomplete_frameset is restart_incomplete
    assert (
        source.config.restart_max_attempts_without_complete_frame
        == restart_limit
    )
    assert source.config.startup_hardware_reset_on_failure is reset_on_failure


def test_deploy_camera_self_check_uses_collection_freshness_gate() -> None:
    class _Source:
        name = "front"
        config = argparse.Namespace(
            serial="FRONT",
            stale_after_sec=2.0,
            enable_depth=False,
        )

        def __init__(self) -> None:
            self.frame_count = 0

        def device_info(self) -> dict[str, object]:
            self.frame_count += 1
            return {
                "frame_count": self.frame_count,
                "last_frame_age_sec": 0.01,
                "stale_after_sec": 2.0,
            }

    source = _Source()
    with patch.object(run_policy.time, "sleep") as sleep_mock:
        run_policy.run_camera_startup_self_check({"front": source})

    assert source.frame_count == 3
    assert sleep_mock.call_count == 2


def test_deploy_camera_reset_override_is_explicit_and_applies_to_each_config() -> None:
    args = _parser().parse_args(
        [
            "--camera-mode",
            "multi",
            "--camera-startup-hardware-reset",
        ]
    )
    configs = {
        "front": argparse.Namespace(startup_hardware_reset=False),
        "left_wrist": argparse.Namespace(startup_hardware_reset=False),
        "right_wrist": argparse.Namespace(startup_hardware_reset=False),
    }

    with patch.object(
        deploy_camera_runtime,
        "load_realsense_camera_config_map",
        return_value=configs,
    ), patch.object(
        deploy_camera_runtime,
        "RealSenseRGBSource",
        side_effect=lambda config: config,
    ):
        sources = run_policy.build_camera_sources(args)

    assert list(sources) == ["front", "left_wrist", "right_wrist"]
    assert all(config.startup_hardware_reset for config in configs.values())


def test_deploy_camera_defaults_preserve_config_owned_reset_behavior() -> None:
    args = _parser().parse_args(["--camera-mode", "multi"])
    configs = {
        "front": argparse.Namespace(startup_hardware_reset=True),
        "left_wrist": argparse.Namespace(startup_hardware_reset=False),
    }

    with patch.object(
        deploy_camera_runtime,
        "load_realsense_camera_config_map",
        return_value=configs,
    ), patch.object(
        deploy_camera_runtime,
        "RealSenseRGBSource",
        side_effect=lambda config: config,
    ):
        run_policy.build_camera_sources(args)

    assert configs["front"].startup_hardware_reset is True
    assert configs["left_wrist"].startup_hardware_reset is False


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


def test_real_shadow_explicit_send_actions_restores_command_capable_arm_configs() -> None:
    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--send-actions",
        "--left-arm-ip",
        "192.0.2.10",
        "--right-arm-ip",
        "192.0.2.11",
        "--active-sides",
        "left",
        "right",
        "--end-effector-sides",
        "none",
        "--no-hands",
        "--arm-execution-profile",
        "impedance_40hz",
        "--policy-qpos-init-dataset",
        "dataset",
    )
    run_policy.validate_args(args, raw_argv=argv)
    backends = run_policy.build_arm_backends(
        args,
        ("left", "right"),
        control_hz=args.arm_control_hz,
        action_hz=args.action_hz,
    )

    assert all(backend.config.observation_only is False for backend in backends.values())
    assert all(
        backend.config.impedance_overrides["canfd_follow"] is False
        for backend in backends.values()
    )

    position_args, position_argv = _raw_args(
        "--rtc-real-shadow",
        "--send-actions",
        "--left-arm-ip",
        "192.0.2.10",
        "--right-arm-ip",
        "192.0.2.11",
        "--active-sides",
        "left",
        "right",
        "--end-effector-sides",
        "none",
        "--no-hands",
        "--arm-execution-profile",
        "position_15_60_120hz",
        "--policy-qpos-init-dataset",
        "dataset",
    )
    run_policy.validate_args(position_args, raw_argv=position_argv)
    position_backends = run_policy.build_arm_backends(
        position_args,
        ("left", "right"),
        control_hz=position_args.arm_control_hz,
        action_hz=position_args.action_hz,
    )

    assert all(
        backend.config.observation_only is False
        for backend in position_backends.values()
    )
    assert all(backend.config.background_streaming_enabled is True for backend in position_backends.values())
    assert all(backend.config.configure_singularity_thresholds is True for backend in position_backends.values())
    assert all(backend.config.canfd_follow is True for backend in position_backends.values())


def test_real_shadow_explicit_send_actions_restores_dmtac_command_start_flags() -> None:
    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--send-actions",
        "--active-sides",
        "both",
        "--end-effector-sides",
        "both",
        "--enable-hands",
        "--hand-backend",
        "dmtac-gripper",
        "--dmtac-no-tactile",
        "--no-dmtac-use-config",
        "--require-dmtac-config",
        "--left-gripper-port",
        "/dev/ttyUSB0",
        "--right-gripper-port",
        "/dev/ttyUSB1",
        "--left-arm-ip",
        "192.0.2.10",
        "--right-arm-ip",
        "192.0.2.11",
        "--arm-execution-profile",
        "impedance_40hz",
        "--policy-qpos-init-dataset",
        "dataset",
    )
    run_policy.validate_args(args, raw_argv=argv)
    configs = run_policy._dmtac_backend_configs(args, ("left", "right"))

    assert set(configs) == {"left", "right"}
    assert all(config.observation_only is False for config in configs.values())
    assert all(config.send_commands is True for config in configs.values())
    assert all(config.clear_fault_on_start is True for config in configs.values())
    # The reviewed manual-port config keeps the normal default of not opening
    # the gripper at startup; the active RTC change must not force an override.
    assert all(config.open_on_start is False for config in configs.values())
    assert all(config.connect_tactile is False for config in configs.values())


def test_real_shadow_explicit_send_actions_restores_inspire_command_config() -> None:
    args, argv = _raw_args(
        "--rtc-real-shadow",
        "--send-actions",
        "--active-sides",
        "left",
        "--end-effector-sides",
        "left",
        "--enable-hands",
        "--hand-backend",
        "inspire",
        "--left-hand-ip",
        "192.0.2.12",
        "--left-arm-ip",
        "192.0.2.10",
        "--arm-execution-profile",
        "impedance_40hz",
        "--policy-qpos-init-dataset",
        "dataset",
    )
    run_policy.validate_args(args, raw_argv=argv)
    backends = run_policy.build_hand_backends(args, ("left",))

    config = backends["left"].config
    assert config.observation_only is False
    assert config.connect_hardware is True
    assert config.enable_telemetry is True


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
