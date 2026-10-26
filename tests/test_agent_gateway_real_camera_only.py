from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import perception_cli, realsense_perception
from agent_gateway.profiles.real.perception_application import (
    CameraOnlyRealPerceptionApplicationState,
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
)
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend_args", [[], ["--object-localization-backend", "sam3-service"]])
@pytest.mark.parametrize("enable_localization", [False, True])
def test_camera_only_sam3_binding_never_constructs_sam2_or_starts_devices(
    monkeypatch, tmp_path, backend_args, enable_localization,
):
    from deployments.perception import sam3_service

    captured = []
    original = sam3_service.Sam3ServiceLocalizer

    def construct(**kwargs):
        value = original(**kwargs)
        captured.append(value)
        return value

    def reject_sam2(*args, **kwargs):
        raise AssertionError("SAM3 selected but SAM2 was constructed")

    def stop_before_sources(*args, **kwargs):
        raise RuntimeError("stop before camera construction")

    monkeypatch.setattr(sam3_service, "Sam3ServiceLocalizer", construct)
    monkeypatch.setattr(perception_cli.Sam2GroundingDinoModelSpec, "from_asset_bundle", reject_sam2)
    monkeypatch.setattr(perception_cli, "build_rm65_dual_arm_scene_profile", stop_before_sources)
    existing = str(REPO_ROOT / "pyproject.toml")
    with pytest.raises(RuntimeError, match="stop before camera construction"):
        argv = [
            "--socket-path", str(tmp_path / "unused.sock"), "--camera-config", existing,
            "--camera-registry", existing, "--artifact-root", str(tmp_path),
            "--sam3-endpoint", "http://127.0.0.1:18114", *backend_args,
        ]
        if enable_localization:
            argv.append("--enable-object-localization")
        perception_cli.main(argv)
    assert len(captured) == int(enable_localization)
    if enable_localization:
        assert not captured[0]._started
        assert captured[0].model_spec.endpoint == "http://127.0.0.1:18114"


@pytest.mark.parametrize("backend_args, bundle, message", [
    ([], True, "invalid with sam3-service"),
    (["--object-localization-backend", "sam2-grounding-dino"], False,
     "requires --object-localization-asset-bundle"),
])
def test_camera_only_rejects_ambiguous_or_missing_localization_bundle_before_resources(
    monkeypatch, tmp_path, capsys, backend_args, bundle, message,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid localization options must fail before resources")

    monkeypatch.setattr(perception_cli.Sam2GroundingDinoModelSpec, "from_asset_bundle", forbidden)
    monkeypatch.setattr(perception_cli, "build_rm65_dual_arm_scene_profile", forbidden)
    existing = str(REPO_ROOT / "pyproject.toml")
    argv = [
        "--socket-path", str(tmp_path / "unused.sock"), "--camera-config", existing,
        "--camera-registry", existing, "--artifact-root", str(tmp_path),
        "--enable-object-localization", *backend_args,
    ]
    if bundle:
        argv += ["--object-localization-asset-bundle", str(tmp_path)]
    with pytest.raises(SystemExit) as error:
        perception_cli.main(argv)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_perception_default_does_not_import_optional_sam3_client():
    result = subprocess.run([
        sys.executable, "-c",
        ("import sys; from agent_gateway.profiles.real import perception_cli; "
        "assert 'deployments.perception.sam3_service' not in sys.modules; "
        "p=perception_cli.build_parser(); "
        "a=p.parse_args(['--socket-path','/unused.sock','--camera-config','/unused.json',"
        "'--camera-registry','/unused.json','--artifact-root','/unused']); "
        "assert a.object_localization_backend=='sam3-service'; "
        "assert not a.enable_object_localization; assert not a.enable_point_grasp; "
        "from agent_gateway.profiles.real import gateway_cli; "
        "g=gateway_cli.build_parser(); "
        "assert g.get_default('object_localization_backend')=='sam3-service'; "
        "assert not g.get_default('enable_point_grasp'); "
        "assert 'deployments.perception.sam3_service' not in sys.modules"),
    ], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_camera_only_explicit_sam2_preserves_localization_settings(monkeypatch, tmp_path):
    captured = {}
    model_spec = object()

    def spec(bundle):
        assert bundle == tmp_path
        return model_spec

    def construct(**kwargs):
        captured.update(kwargs)
        return object()

    def stop_before_sources(*args, **kwargs):
        raise RuntimeError("stop before camera construction")

    monkeypatch.setattr(perception_cli.Sam2GroundingDinoModelSpec, "from_asset_bundle", spec)
    monkeypatch.setattr(perception_cli, "Sam2GroundingDinoLocalizer", construct)
    monkeypatch.setattr(perception_cli, "build_rm65_dual_arm_scene_profile", stop_before_sources)
    existing = str(REPO_ROOT / "pyproject.toml")
    with pytest.raises(RuntimeError, match="stop before camera construction"):
        perception_cli.main([
            "--socket-path", str(tmp_path / "unused.sock"), "--camera-config", existing,
            "--camera-registry", existing, "--artifact-root", str(tmp_path),
            "--enable-object-localization",
            "--object-localization-backend", "sam2-grounding-dino",
            "--object-localization-asset-bundle", str(tmp_path),
            "--object-localization-device", "cpu",
            "--object-localization-box-threshold", "0.4",
            "--object-localization-text-threshold", "0.3",
            "--object-localization-automatic-score-threshold", "0.85",
            "--object-localization-nms-iou-threshold", "0.6",
        ])
    assert captured == {
        "model_spec": model_spec, "device": "cpu", "box_threshold": 0.4,
        "text_threshold": 0.3, "automatic_score_threshold": 0.85,
        "nms_iou_threshold": 0.6,
    }


class _CameraSource:
    def __init__(self, source_id: str, *, enable_depth: bool) -> None:
        self.source_id = source_id
        self.config = SimpleNamespace(
            camera_key=source_id,
            width=4,
            height=3,
            depth_width=4,
            depth_height=3,
            align_depth_to_color=True,
            enable_depth=enable_depth,
            color_space="rgb",
            serial=source_id + "-serial",
        )

    def device_info(self):
        return {
            "running": True,
            "generation": 1,
            "serial_number": self.config.serial,
        }

    def close(self) -> None:
        return None


def _source_binding() -> RealPerceptionSourceBinding:
    return RealPerceptionSourceBinding(
        camera_sources={
            "front": _CameraSource("front", enable_depth=True),
            "left_wrist": _CameraSource("left_wrist", enable_depth=False),
            "right_wrist": _CameraSource("right_wrist", enable_depth=False),
        },
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:camera-only-test",
    )


def _frame_binding():
    station_from_source = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return build_real_perception_station_frame_binding(
        collision_scene_profile="camera-only-test",
        scene_mount_hash="scene-mounts:camera-only-test",
        scene_world_from_canonical_station=np.linalg.inv(station_from_source),
    )


def _composition(tmp_path: Path):
    return assemble_camera_only_real_perception_composition(
        source_binding=_source_binding(),
        socket_path=tmp_path / "gateway.sock",
        artifact_root=tmp_path,
        client_bindings=compile_agent_gateway_unix_client_bindings(
            primary_client_id="rpent:test",
            primary_client_uid=1000,
        ),
        runtime_session_id="real-perception:test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        clock=lambda: 10.0,
    )


def test_camera_only_composition_exposes_only_real_observe_perception_routes(
    tmp_path,
) -> None:
    composition = _composition(tmp_path)
    try:
        manifest = composition.contract_bundle.route_manifest
        assert manifest.current_mode == "real_observe"
        assert [entry.route.skill_id for entry in manifest.skill_routes] == [
            "perception.back_project",
            "perception.capture_depth",
            "perception.capture_rgb",
        ]
        assert all(
            entry.route.supported_modes == ("real_observe",)
            for entry in manifest.skill_routes
        )
        assert all(
            entry.route.control_effect_class == "read_only"
            for entry in manifest.skill_routes
        )
        assert all(
            entry.route.may_move_robot is False for entry in manifest.skill_routes
        )
        assert composition.registration.profile.rgb_roles == (
            "main",
            "wrist_left",
            "wrist_right",
        )
        assert composition.registration.profile.depth_roles == ("main",)
        assert composition.registration.rgb.routes[0].route_version == "2"
        assert composition.registration.rgb.definition.skill_version == "2"
        assert composition.registration.depth.routes[0].route_version == "2"
        assert composition.registration.frame_binding is composition.frame_binding
        assert composition.contract_bundle.frame_graph_hash == (
            composition.frame_binding.frame_graph.semantic_content_hash
        )
        assert not hasattr(composition.resources, "lower")
        assert not hasattr(composition, "lower")
        assert composition.resources.started is False
        expected_target_hash = canonical_content_hash(
            {
                "source_binding_hash": composition.source_binding.semantic_content_hash,
                "frame_binding_hash": composition.frame_binding.semantic_content_hash,
                "resource_kind": "camera_bundle",
                "object_localizer_hash": None,
            },
            domain="dexteleop.real_perception.target_binding.v1",
        )
        assert (
            manifest.target_binding_id == "real-realsense-only:" + expected_target_hash
        )
        expected_execution_hash = canonical_content_hash(
            {
                "resource_owner": "camera_only_real_perception_application",
                "source_binding_hash": composition.source_binding.semantic_content_hash,
                "robot_backend": "absent",
                "command_owner_count": 0,
                "downstream_control_effects": "forbidden",
                "object_localizer_hash": None,
            },
            domain="dexteleop.real_perception.execution_manifest.v1",
        )
        assert manifest.execution_manifest_hash == expected_execution_hash
        assert composition.deployment_profile_hash == canonical_content_hash(
            {
                "source_binding_hash": composition.source_binding.semantic_content_hash,
                "frame_binding_hash": composition.frame_binding.semantic_content_hash,
                "frame_graph_hash": composition.frame_binding.frame_graph.semantic_content_hash,
                "route_manifest_hash": manifest.semantic_content_hash,
                "artifact_max_bytes": 16 * 1024 * 1024,
                "object_localizer_hash": None,
                "local_path_test_input": False,
                "robot_backend": "absent",
            },
            domain="dexteleop.real_perception.camera_only_profile.v1",
        )
    finally:
        composition.application.close()


def test_camera_only_application_starts_cameras_before_socket_and_closes_cleanly(
    monkeypatch,
    tmp_path,
) -> None:
    events = []
    monkeypatch.setattr(
        realsense_perception,
        "start_realsense_sources",
        lambda sources: events.append(("start-cameras", tuple(sorted(sources)))),
    )
    monkeypatch.setattr(
        realsense_perception,
        "admit_realsense_sources",
        lambda sources: events.append(("admit-cameras", tuple(sorted(sources)))),
    )
    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        lambda sources: events.append(("close-cameras", tuple(sorted(sources)))),
    )
    composition = _composition(tmp_path)

    class _Server:
        def serve_forever(self, poll_interval=0.5):
            return None

        def shutdown(self):
            events.append(("shutdown",))

        def server_close(self):
            events.append(("close-gateway",))
            composition.session.close()

    composition.application.server_factory = lambda: (
        events.append(("create-socket",)) or _Server()
    )
    composition.application.start()
    assert (
        composition.application.state
        == CameraOnlyRealPerceptionApplicationState.RUNNING
    )
    assert [event[0] for event in events[:3]] == [
        "start-cameras",
        "admit-cameras",
        "create-socket",
    ]
    composition.application.close()
    assert (
        composition.application.state == CameraOnlyRealPerceptionApplicationState.CLOSED
    )
    assert [event[0] for event in events[-2:]] == [
        "close-gateway",
        "close-cameras",
    ]


def _trace_resource_close(composition, monkeypatch):
    calls = []
    original_close = composition.resources.close

    def close():
        calls.append("close-resources")
        original_close()

    monkeypatch.setattr(composition.resources, "close", close)
    return calls


@pytest.mark.parametrize("with_server", [False, True])
def test_camera_only_close_retains_dependencies_until_reads_quiesce(
    monkeypatch,
    tmp_path,
    with_server,
) -> None:
    composition = _composition(tmp_path)
    settled = [False]
    composition.session.segmentation_executor = SimpleNamespace(
        wait_all=lambda **kwargs: settled[0],
    )
    calls = _trace_resource_close(composition, monkeypatch)
    server = None
    if with_server:
        server = composition.application.server_factory()
        composition.application._server = server
    try:
        with pytest.raises(RuntimeError, match="segmentation did not quiesce"):
            composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.FAULT
        )
        assert calls == []
        if server is not None:
            assert isinstance(server.gateway_close_error, RuntimeError)
            assert not Path(server.socket_path).exists()

        settled[0] = True
        composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.CLOSED
        )
        assert calls == ["close-resources"]
        if server is not None:
            assert server.gateway_close_error is None
        composition.application.close()
        assert calls == ["close-resources"]
    finally:
        settled[0] = True
        composition.application.close()


@pytest.mark.parametrize("with_server", [False, True])
def test_camera_only_rejected_gateway_close_is_retryable(
    monkeypatch,
    tmp_path,
    with_server,
) -> None:
    composition = _composition(tmp_path)
    accepted = [False]
    original_close = composition.session.service.close

    def close_gateway(**kwargs):
        if not accepted[0]:
            return SimpleNamespace(
                accepted=False, reason_code="injected_close_rejection"
            )
        return original_close(**kwargs)

    monkeypatch.setattr(composition.session.service, "close", close_gateway)
    calls = _trace_resource_close(composition, monkeypatch)
    if with_server:
        composition.application._server = composition.application.server_factory()
    try:
        with pytest.raises(RuntimeError, match="injected_close_rejection"):
            composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.FAULT
        )
        assert calls == []

        accepted[0] = True
        composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.CLOSED
        )
        assert calls == ["close-resources"]
    finally:
        accepted[0] = True
        composition.application.close()


def test_camera_only_startup_cleanup_defers_dependencies_until_reads_quiesce(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        realsense_perception, "start_realsense_sources", lambda sources: None
    )
    monkeypatch.setattr(
        realsense_perception, "admit_realsense_sources", lambda sources: None
    )
    composition = _composition(tmp_path)
    settled = [False]
    composition.session.segmentation_executor = SimpleNamespace(
        wait_all=lambda **kwargs: settled[0],
    )
    calls = _trace_resource_close(composition, monkeypatch)

    def fail_server_factory():
        raise RuntimeError("socket allocation failed")

    composition.application.server_factory = fail_server_factory
    try:
        with pytest.raises(RuntimeError, match="startup cleanup failed") as error:
            composition.application.start()
        assert str(error.value.__cause__) == "socket allocation failed"
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.FAULT
        )
        assert calls == []
        settled[0] = True
        composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.CLOSED
        )
        assert calls == ["close-resources"]
    finally:
        settled[0] = True
        composition.application.close()


def test_camera_only_resource_cleanup_failure_remains_retryable(
    monkeypatch, tmp_path
) -> None:
    composition = _composition(tmp_path)
    calls = []
    original_close = composition.resources.close

    def fail_once():
        calls.append("close-resources")
        if len(calls) == 1:
            raise RuntimeError("store close failed")
        original_close()

    monkeypatch.setattr(composition.resources, "close", fail_once)
    try:
        with pytest.raises(RuntimeError, match="store close failed"):
            composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.FAULT
        )
        composition.application.close()
        assert (
            composition.application.state
            == CameraOnlyRealPerceptionApplicationState.CLOSED
        )
        composition.application.close()
        assert calls == ["close-resources", "close-resources"]
    finally:
        composition.application.close()


def test_real_perception_cli_is_discoverable_without_importing_hardware() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agent_gateway", "real-perception", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "never constructs or connects an RM65 backend" in result.stdout
    assert "--camera-config" in result.stdout
    assert "--artifact-root" in result.stdout
    assert "--enable-object-localization" in result.stdout
    assert "--enable-point-grasp" in result.stdout
    assert "--point-grasp-backend" in result.stdout
    assert "graspgenx" in result.stdout
    assert "--graspgenx-mode" in result.stdout
    assert "--graspgenx-source-root" in result.stdout
    assert "--allow-local-image-root" not in result.stdout


def test_real_perception_defaults_to_graspgenx_inprocess() -> None:
    argv = [
        "--socket-path",
        "/tmp/agenticrobot-real-perception-test.sock",
        "--camera-config",
        str(REPO_ROOT / "pyproject.toml"),
        "--camera-registry",
        str(REPO_ROOT / "pyproject.toml"),
        "--artifact-root",
        "/tmp",
    ]
    parsed = perception_cli.build_parser().parse_args(argv)

    assert parsed.point_grasp_backend == "graspgenx"
    assert parsed.graspgenx_mode == "in-process"
    assert perception_cli._graspgenx_mode_was_explicit(argv) is False
    assert perception_cli._graspgenx_mode_was_explicit(
        [*argv, "--graspgenx-mode=service"]
    ) is True
