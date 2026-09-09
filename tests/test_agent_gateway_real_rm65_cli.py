from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real import gateway_cli
from agent_gateway.profiles.real.session_authorization import (
    REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _argv(*, confirmation: str, socket_path: Path | None = None) -> list[str]:
    existing = str(REPO_ROOT / "pyproject.toml")
    return [
        "--socket-path",
        str(
            REPO_ROOT / "temp/test-real-rm65-gateway.sock"
            if socket_path is None
            else socket_path
        ),
        "--arm-ip-assignment",
        existing,
        "--left-tool-profile",
        existing,
        "--right-tool-profile",
        existing,
        "--envelope-profile",
        existing,
        "--authorized-side",
        "right",
        "--authorized-action",
        "forward",
        "--confirm",
        confirmation,
        "--runtime-session-id",
        "real-rm65-cli:test",
        "--embodiment-instance-id",
        "dual-rm65:test",
    ]


def test_cli_rejects_confirmation_before_profile_or_hardware_resolution(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        nonlocal called
        called = True
        raise AssertionError("profile resolution must remain default-off")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(_argv(confirmation="wrong"))

    assert called is False


def test_cli_rejects_retired_one_shot_confirmation_before_hardware_resolution(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        del profile
        nonlocal called
        called = True
        raise AssertionError("retired confirmation must remain default-off")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(_argv(confirmation="AUTHORIZE_ONE_REAL_TCP_PRIMITIVE"))

    assert called is False


def test_cli_accepts_a_list_of_reviewed_actions_for_persistent_session() -> None:
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    action_index = argv.index("--authorized-action")
    del argv[action_index : action_index + 2]
    argv.extend(
        [
            "--authorized-actions",
            "forward",
            "backward",
            "left",
            "right",
            "up",
            "down",
            "level_forward",
            "level_forward_left_30",
            "level_forward_right_30",
        ]
    )
    parsed = gateway_cli.build_parser().parse_args(argv)
    assert parsed.authorized_action is None
    assert tuple(parsed.authorized_actions) == (
        "forward",
        "backward",
        "left",
        "right",
        "up",
        "down",
        "level_forward",
        "level_forward_left_30",
        "level_forward_right_30",
    )
    assert not hasattr(parsed, "calibration_profile")


def test_cli_defaults_to_the_detailed_v4_camera_mount_urdf() -> None:
    parsed = gateway_cli.build_parser().parse_args(
        [
            *_argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN),
            "--enable-point-grasp",
        ]
    )
    effective = gateway_cli._effective_camera_mount_urdf(parsed)

    assert effective == gateway_cli.DEFAULT_RM65_CAMERA_MOUNT_URDF
    assert effective.name == (
        "rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf"
    )


def test_point_grasp_defaults_to_v4_registry_without_changing_camera_only() -> None:
    parser = gateway_cli.build_parser()
    camera_only = parser.parse_args(
        _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    )
    point_grasp = parser.parse_args(
        [
            *_argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN),
            "--enable-point-grasp",
        ]
    )

    assert gateway_cli._effective_camera_registry(camera_only) is None
    assert (
        gateway_cli._effective_camera_registry(point_grasp)
        == gateway_cli.DEFAULT_RM65_CAMERA_REGISTRY_V4
    )


def test_object_coordinate_render_debug_is_explicit_and_default_off() -> None:
    parser = gateway_cli.build_parser()
    default = parser.parse_args(
        _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    )
    enabled = parser.parse_args(
        [
            *_argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN),
            "--enable-point-grasp",
            "--debug-render-object-coordinates",
        ]
    )

    assert default.debug_render_object_coordinates is False
    assert enabled.debug_render_object_coordinates is True


def test_motion_trace_is_explicit_and_preallocated_duration_is_bounded() -> None:
    parser = gateway_cli.build_parser()
    default = parser.parse_args(
        _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    )
    enabled = parser.parse_args(
        [
            *_argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN),
            "--motion-trace-root",
            "/private/trace-root",
            "--motion-trace-max-duration-sec",
            "45",
        ]
    )

    assert default.motion_trace_root is None
    assert default.motion_trace_max_duration_sec == 120.0
    assert enabled.motion_trace_root == Path("/private/trace-root")
    assert enabled.motion_trace_max_duration_sec == 45.0


def test_motion_trace_attaches_and_detaches_all_three_control_rates() -> None:
    class Owner:
        def __init__(self, method):
            self.values = []
            setattr(self, method, self.values.append)

    recorder = SimpleNamespace(
        record_control_cycle=lambda *_args: None,
        record_ik_shadow=lambda *_args: None,
        record_actuator_command=lambda *_args: None,
    )
    control = Owner("set_motion_trace_sink")
    target = Owner("set_ik_shadow_trace_sink")
    left = Owner("set_actuator_command_trace_sink")
    right = Owner("set_actuator_command_trace_sink")
    composition = SimpleNamespace(
        deployment=SimpleNamespace(control_runtime=control),
        lower=SimpleNamespace(
            target_builder=target,
            arm_backends={"left": left, "right": right},
        ),
    )

    gateway_cli._attach_motion_trace(composition, recorder)
    errors = gateway_cli._detach_motion_trace(composition)

    assert errors == []
    assert control.values == [recorder.record_control_cycle, None]
    assert target.values == [recorder.record_ik_shadow, None]
    assert left.values == [recorder.record_actuator_command, None]
    assert right.values == [recorder.record_actuator_command, None]


def test_motion_trace_partial_attach_failure_rolls_back_existing_hooks() -> None:
    class Owner:
        def __init__(self, method):
            self.values = []
            setattr(self, method, self.values.append)

    recorder = SimpleNamespace(
        record_control_cycle=lambda *_args: None,
        record_ik_shadow=lambda *_args: None,
        record_actuator_command=lambda *_args: None,
    )
    control = Owner("set_motion_trace_sink")
    target = Owner("set_ik_shadow_trace_sink")
    left = Owner("set_actuator_command_trace_sink")
    composition = SimpleNamespace(
        deployment=SimpleNamespace(control_runtime=control),
        lower=SimpleNamespace(
            target_builder=target,
            arm_backends={"left": left, "right": object()},
        ),
    )

    with pytest.raises(TypeError, match="right RM65 backend"):
        gateway_cli._attach_motion_trace(composition, recorder)

    assert control.values == [recorder.record_control_cycle, None]
    assert target.values == [recorder.record_ik_shadow, None]
    assert left.values == [recorder.record_actuator_command, None]


def test_cli_rejects_object_coordinate_debug_without_point_grasp(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        del profile
        nonlocal called
        called = True
        raise AssertionError("debug flag validation must precede profile resolution")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    argv.append("--debug-render-object-coordinates")

    with pytest.raises(SystemExit) as captured:
        gateway_cli.main(argv)

    assert captured.value.code == 2
    assert called is False


def test_cli_accepts_distinct_repeatable_ssh_read_only_identities() -> None:
    parsed = gateway_cli.build_parser().parse_args(
        [
            *_argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN),
            "--ssh-read-only-client",
            "rpent.remote.alice",
            "1101",
            "--ssh-read-only-client",
            "rpent.remote.bob",
            "1102",
        ]
    )

    assert parsed.ssh_read_only_client == [
        ["rpent.remote.alice", "1101"],
        ["rpent.remote.bob", "1102"],
    ]


def test_cli_rejects_partial_perception_attachment_before_profile_resolution(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        nonlocal called
        called = True
        raise AssertionError("partial perception flags must fail first")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    argv.extend(
        [
            "--camera-config",
            str(
                REPO_ROOT
                / "teleop_runtime/configs/devices/agent_gateway_realsense.json"
            ),
        ]
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(argv)

    assert called is False


def test_cli_rejects_point_grasp_bundle_without_opt_in_before_profile_resolution(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        nonlocal called
        called = True
        raise AssertionError("point-grasp flags must fail before profile resolution")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    argv.extend(
        [
            "--point-grasp-asset-bundle",
            str(REPO_ROOT),
        ]
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(argv)

    assert called is False


def test_cli_rejects_camera_mount_urdf_without_point_grasp_opt_in(
    monkeypatch,
) -> None:
    called = False

    def should_not_resolve(profile):
        nonlocal called
        called = True
        raise AssertionError("camera mount must fail before profile resolution")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    argv.extend(
        [
            "--camera-mount-urdf",
            str(gateway_cli.DEFAULT_RM65_CAMERA_MOUNT_URDF),
        ]
    )

    with pytest.raises(SystemExit):
        gateway_cli.main(argv)

    assert called is False


def test_cli_rejects_point_grasp_when_selected_tool_tcp_is_not_reviewed_dmtac(
    monkeypatch,
    tmp_path: Path,
    short_unix_socket_path_factory,
) -> None:
    resolved = SimpleNamespace(
        semantic_content_hash="a" * 64,
        tool_tcp_pose_flange_m_rad_by_side={
            "left": [0.0, 0.0, 0.14, 0.0, 0.0, 0.0],
            "right": [0.0, 0.0, 0.12, 0.0, 0.0, 0.0],
        },
    )
    lower_assembled = False

    def should_not_assemble(*, profile):
        del profile
        nonlocal lower_assembled
        lower_assembled = True
        raise AssertionError(
            "mismatched tool geometry must fail before hardware assembly"
        )

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        lambda _profile: resolved,
    )
    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_skill_lower",
        should_not_assemble,
    )
    socket_path = short_unix_socket_path_factory("tool-geometry")
    socket_path.parent.chmod(0o2750)
    argv = _argv(
        confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        socket_path=socket_path,
    )
    argv.extend(
        [
            "--camera-config",
            str(REPO_ROOT / "pyproject.toml"),
            "--artifact-root",
            str(tmp_path),
            "--enable-point-grasp",
            "--object-localization-asset-bundle",
            str(REPO_ROOT),
            "--point-grasp-asset-bundle",
            str(REPO_ROOT),
        ]
    )

    with pytest.raises(SystemExit) as captured:
        gateway_cli.main(argv)

    assert captured.value.code == 2
    assert lower_assembled is False


def test_cli_rejects_overlong_socket_before_profile_or_hardware_assembly(
    monkeypatch,
    capsys,
) -> None:
    profile_resolved = False
    lower_assembled = False

    def should_not_resolve(_profile):
        nonlocal profile_resolved
        profile_resolved = True
        raise AssertionError("socket preflight must precede profile resolution")

    def should_not_assemble(*, profile):
        del profile
        nonlocal lower_assembled
        lower_assembled = True
        raise AssertionError("socket preflight must precede hardware assembly")

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        should_not_resolve,
    )
    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_skill_lower",
        should_not_assemble,
    )
    argv = _argv(confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    argv[1] = "/" + "a" * 108

    with pytest.raises(SystemExit) as captured:
        gateway_cli.main(argv)

    assert captured.value.code == 2
    assert profile_resolved is False
    assert lower_assembled is False
    error = capsys.readouterr().err
    assert "--socket-path preflight failed" in error
    assert "uses 109 encoded bytes" in error
    assert "will not relocate it automatically" in error
    assert "Traceback" not in error


def test_cli_assembles_then_reports_ready_and_lower_hold(
    monkeypatch,
    capsys,
    short_unix_socket_path_factory,
) -> None:
    resolved = SimpleNamespace(semantic_content_hash="a" * 64)
    lower = object()
    captured = {}

    class _Application:
        state = gateway_cli.ExclusiveRealSkillApplicationState.CREATED
        control_fault = None

        def serve_forever(self, *, ready_callback=None):
            assert ready_callback is not None
            self.state = gateway_cli.ExclusiveRealSkillApplicationState.RUNNING
            ready_callback()
            time.sleep(0.05)
            self.state = gateway_cli.ExclusiveRealSkillApplicationState.CLOSED

        def request_shutdown(self):
            return False

    route = SimpleNamespace(
        skill_id="motion.step_tcp_cardinal",
        skill_version="1",
        route_id="real.rm65.agent_skill_tcp_cardinal_step",
        route_version="1",
    )
    entry = SimpleNamespace(route=route, semantic_content_hash="route-entry:test")
    manifest = SimpleNamespace(
        target_kind="real",
        current_mode="real_active",
        runtime_session_id="real-rm65-cli:test",
        control_domain_id="exclusive-rm65-agent-skill",
        target_binding_id="real-rm65-exclusive:test",
        semantic_content_hash="manifest:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        supported_transport_features=("own_domain_emergency_stop_v1",),
        skill_routes=(entry,),
    )
    hold = SimpleNamespace(succeeded=True, evidence_refs=("hold:test",))
    motion_port = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(dispatch_count=0),
    )
    camera_executor = SimpleNamespace(
        failure_diagnostics=(
            {
                "executor_id": "camera-reader:test",
                "operation_id": "operation:rgb:test",
                "exception_type": "RuntimeError",
                "message": "rgb_capture_inter_camera_skew_exceeded",
                "failed_at": 10.25,
            },
        )
    )
    point_grasp_executor = SimpleNamespace(
        failure_diagnostics=(
            {
                "executor_id": "point-grasp-planner:test",
                "operation_id": "operation:point-grasp:test",
                "exception_type": "RuntimeError",
                "message": "point-grasp inference failed",
                "failed_at": 10.25,
            },
        )
    )
    runtime = SimpleNamespace(
        motion_port=motion_port,
        last_hold_receipt=hold,
        control_cycle_snapshot=lambda: {
            "cycle_count": 9,
            "latest_cycle_sec": 0.01,
            "max_cycle_sec": 0.02,
            "expired_reference_skip_count": 0,
            "consecutive_expired_reference_skips": 0,
            "latest_expired_measurement_age_sec": None,
            "max_measurement_age_sec": 0.2,
            "last_published_source_sequence_by_side": {
                "left": 10,
                "right": 11,
            },
        },
    )
    socket_path = short_unix_socket_path_factory("real-cli")
    socket_path.parent.chmod(0o2750)
    composition = SimpleNamespace(
        socket_path=socket_path,
        client_id="rpent.dexteleop",
        authorized_side="right",
        authorized_action="forward",
        lower=SimpleNamespace(
            profile=resolved,
            ports=SimpleNamespace(
                last_emergency_stop_receipt=None,
                arm_backend_diagnostics_snapshot=lambda: {
                    "left": {"background_fault": None},
                    "right": {
                        "background_fault": (
                            "Movej_CANFD returned non-zero/unknown result: "
                            "error code 7"
                        ),
                        "background_fault_stop_attempted": True,
                        "background_fault_stop_success": True,
                    },
                },
                measurement_sampler_snapshot=lambda: {
                    "enabled": True,
                    "thread_alive": False,
                    "sample_count": 7,
                    "fault": None,
                    "latest_measurement_age_sec": 0.01,
                    "latest_available_age_sec": 0.005,
                    "source_sequence_by_side": {"left": 10, "right": 11},
                }
            ),
        ),
        deployment=SimpleNamespace(
            gateway_session=SimpleNamespace(
                contract_bundle=SimpleNamespace(route_manifest=manifest),
                emergency_stop_port=object(),
                service=SimpleNamespace(
                    emergency_stop_latched=False,
                    emergency_stop_receipt=None,
                ),
                executors_by_skill_id={
                    "motion.step_tcp_cardinal": SimpleNamespace()
                },
                passive_read_executors_by_skill_id={
                    "perception.capture_rgb": camera_executor,
                    "perception.plan_point_grasp": point_grasp_executor,
                },
            ),
            control_runtime=runtime,
        ),
        application=_Application(),
    )

    monkeypatch.setattr(
        gateway_cli,
        "resolve_exclusive_rm65_skill_deployment_profile",
        lambda profile: resolved,
    )
    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_skill_lower",
        lambda *, profile: lower,
    )

    def assemble(**kwargs):
        captured.update(kwargs)
        return composition

    monkeypatch.setattr(
        gateway_cli,
        "assemble_exclusive_rm65_agent_primitive_composition",
        assemble,
    )

    argv = _argv(
        confirmation=REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        socket_path=socket_path,
    )
    argv.extend(
        [
            "--ssh-read-only-client",
            "rpent.remote.alice",
            "1101",
        ]
    )
    assert gateway_cli.main(argv) == 0

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "gateway_ready",
        "gateway_stopped",
    ]
    assert events[0]["authorization_scope"] == "persistent_session_matching_action"
    assert events[0]["domain_emergency_stop_available"] is True
    assert events[1]["lower_hold_proved"] is True
    assert events[1]["emergency_stop_latched"] is False
    assert events[1]["emergency_stop_confirmed"] is False
    assert events[1]["control_fault"] is None
    assert events[1]["measurement_sampler"]["sample_count"] == 7
    assert events[1]["arm_backend_diagnostics"]["right"] == {
        "background_fault": (
            "Movej_CANFD returned non-zero/unknown result: error code 7"
        ),
        "background_fault_stop_attempted": True,
        "background_fault_stop_success": True,
    }
    assert events[1]["control_cycle"]["cycle_count"] == 9
    assert events[1]["passive_read_failures"] == [
        {
            "skill_id": "perception.capture_rgb",
            "executor_id": "camera-reader:test",
            "operation_id": "operation:rgb:test",
            "exception_type": "RuntimeError",
            "message": "rgb_capture_inter_camera_skew_exceeded",
            "failed_at": 10.25,
        },
        {
            "skill_id": "perception.plan_point_grasp",
            "executor_id": "point-grasp-planner:test",
            "operation_id": "operation:point-grasp:test",
            "exception_type": "RuntimeError",
            "message": "point-grasp inference failed",
            "failed_at": 10.25,
        }
    ]
    assert captured["lower"] is lower
    assert captured["authorized_side"] == "right"
    assert captured["authorized_action"] == "forward"
    assert captured["ssh_read_only_client_id_by_uid"] == {1101: "rpent.remote.alice"}

    composition.lower.ports.last_emergency_stop_receipt = SimpleNamespace(
        succeeded=True,
        status_by_side={"left": "confirmed", "right": "confirmed"},
        evidence_refs=("estop:left", "estop:right"),
    )
    composition.deployment.gateway_session.service.emergency_stop_latched = True
    composition.deployment.gateway_session.service.emergency_stop_receipt = (
        SimpleNamespace(succeeded=True, proves_motion_or_hold=True)
    )
    stopped_after_emergency_stop = gateway_cli._stopped_payload(composition)
    assert stopped_after_emergency_stop["emergency_stop_confirmed"] is True
    assert stopped_after_emergency_stop["safe_shutdown"] is True
    assert stopped_after_emergency_stop["safe_shutdown_basis"] == (
        "lower_emergency_stop_receipt"
    )
