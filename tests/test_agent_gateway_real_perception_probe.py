from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import perception_probe, realsense_perception
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
)
from teleop_runtime.video_frame import VideoFrame


class _ProbeCameraSource:
    def __init__(self, source_id: str, *, enable_depth: bool) -> None:
        self.source_id = source_id
        self.config = SimpleNamespace(
            camera_key=source_id,
            width=640,
            height=480,
            depth_width=640,
            depth_height=480,
            depth_fps=15,
            align_depth_to_color=True,
            enable_depth=enable_depth,
            color_space="rgb",
            serial=source_id + "-serial",
        )
        self.closed = 0
        self.generation = 1

    def get_latest_frame(self) -> VideoFrame:
        # Simulate a fresh continuously published frame that is unambiguously
        # earlier than the provider's already-pinned capture cut. Deriving a
        # floor from a fresh wall-clock read can cross a 20 Hz boundary between
        # camera reads and incorrectly manufacture a future frame in this fake.
        captured_at = time.time() - 0.1
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        if self.source_id != "front":
            image[..., 1] = 64
        return VideoFrame(
            image=image,
            source_timestamp=captured_at,
            receive_timestamp=captured_at,
            available_timestamp=captured_at,
            frame_index=1,
            metadata={
                "camera_key": self.source_id,
                "serial": self.config.serial,
                "source_generation": self.generation,
            },
            depth_image=(
                np.arange(640 * 480, dtype=np.uint16).reshape(480, 640)
                if self.config.enable_depth
                else None
            ),
            depth_metadata=(
                {
                    "aligned_to": "color",
                    "depth_scale_m_per_unit": 0.001,
                }
                if self.config.enable_depth
                else {}
            ),
        )

    def device_info(self):
        return {
            "running": True,
            "lifecycle_state": "running",
            "serial_number": self.config.serial,
            "generation": self.generation,
            "worker_alive": True,
            "frame_count": 1,
            "timeout_count": 0,
            "black_frame_count": 1 if self.source_id == "front" else 0,
            "consecutive_black_frame_count": (1 if self.source_id == "front" else 0),
            "restart_count": 0,
            "last_error": None,
            "last_runtime_error": None,
            "intrinsics": {
                "width": 640,
                "height": 480,
                "fx": 600.0,
                "fy": 600.0,
                "ppx": 320.0,
                "ppy": 240.0,
            },
        }

    def close(self) -> None:
        self.closed += 1


def _source_binding():
    sources = {
        "front": _ProbeCameraSource("front", enable_depth=True),
        "left_wrist": _ProbeCameraSource("left_wrist", enable_depth=False),
        "right_wrist": _ProbeCameraSource("right_wrist", enable_depth=False),
    }
    binding = RealPerceptionSourceBinding(
        camera_sources=sources,
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:real-perception-probe-test",
    )
    return binding, sources


def _patch_probe_hardware(monkeypatch, binding, sources):
    station_from_source = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile="probe-test",
        scene_mount_hash="scene-mounts:probe-test",
        scene_world_from_canonical_station=np.linalg.inv(station_from_source),
    )
    monkeypatch.setattr(
        perception_probe,
        "_scene_geometry",
        lambda: ({"left": np.eye(4), "right": np.eye(4)}, frame_binding),
    )
    monkeypatch.setattr(
        perception_probe,
        "_build_source_binding",
        lambda **_kwargs: binding,
    )
    monkeypatch.setattr(
        realsense_perception,
        "start_realsense_sources",
        lambda _sources: None,
    )
    monkeypatch.setattr(
        realsense_perception,
        "admit_realsense_sources",
        lambda _sources: None,
    )
    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        lambda _sources: [source.close() for source in sources.values()] and None,
    )
    return frame_binding


def test_real_perception_probe_cli_exposes_no_robot_or_control_arguments() -> None:
    parser = perception_probe.build_parser()
    help_text = parser.format_help()
    normalized_help = " ".join(help_text.split())

    assert "--run-root" in help_text
    assert "no robot backend" in normalized_help
    for forbidden in (
        "--arm-ip-assignment",
        "--authorized-side",
        "--authorized-action",
        "--confirm",
        "--camera-registry",
        "--camera-config",
        "--collision-scene-profile",
    ):
        assert forbidden not in help_text

    with pytest.raises(SystemExit):
        parser.parse_args(["--run-root", "/tmp/probe", "--confirm", "YES"])


def test_camera_only_probe_loads_complete_typed_station_frame_graph() -> None:
    world_from_base, frame_binding = perception_probe._scene_geometry()

    assert set(world_from_base) == {"left", "right"}
    assert frame_binding.source_frame_id == "dual_arm_solver_world"
    assert frame_binding.reference_frame_id == "canonical_station"
    assert set(frame_binding.frame_graph.frame_by_id) == {
        "dual_arm_solver_world",
        "canonical_station",
    }
    assert frame_binding.frame_graph.connected(
        frame_binding.source_frame_id,
        frame_binding.reference_frame_id,
    )
    assert len(frame_binding.frame_graph.semantic_content_hash) == 64


@pytest.mark.parametrize("violation", ["unexpected_route", "back_projection_motion"])
def test_real_perception_probe_rejects_unexpected_or_motion_routes(violation) -> None:
    routes = [
        SimpleNamespace(
            skill_id=skill_id,
            route_id=skill_id + ":test",
            control_effect_class="read_only",
            privilege_tier="t0_read_only",
            may_move_robot=False,
            may_open_publication=False,
            requires_operator_confirmation=False,
            requires_deadman=False,
        )
        for skill_id in (
            "perception.capture_rgb",
            "perception.capture_depth",
            "perception.back_project",
        )
    ]
    entries = [SimpleNamespace(route=route) for route in routes]
    composition = SimpleNamespace(
        contract_bundle=SimpleNamespace(
            route_manifest=SimpleNamespace(skill_routes=entries)
        ),
        resources=SimpleNamespace(),
    )
    perception_probe._assert_no_motion_surface(composition)
    if violation == "unexpected_route":
        entries.append(SimpleNamespace(route=SimpleNamespace(skill_id="perception.extra")))
        message = "probe route surface is not perception-only"
    else:
        routes[-1].control_effect_class = "motion"
        routes[-1].may_move_robot = True
        message = "probe route is not strictly passive"
    with pytest.raises(RuntimeError, match=message):
        perception_probe._assert_no_motion_surface(composition)


def test_real_perception_probe_runs_both_skills_and_validates_artifacts(
    monkeypatch,
    tmp_path,
    short_unix_socket_path_factory,
) -> None:
    binding, sources = _source_binding()
    _patch_probe_hardware(monkeypatch, binding, sources)
    socket_path = short_unix_socket_path_factory("perception-probe")

    result = perception_probe.run_probe(
        run_root=tmp_path / "probe",
        client_id="rpent:perception-probe-test",
        operation_timeout_sec=2.0,
        socket_path=socket_path,
    )

    assert result["status"] == "passed"
    assert result["robot_backend"] == "absent"
    assert result["control_effects"] == "forbidden"
    assert result["hardware_reset"] == "forbidden"
    assert result["application_state"] == "closed"
    assert [item["role"] for item in result["rgb"]] == [
        "main",
        "wrist_left",
        "wrist_right",
    ]
    assert result["rgb"][0]["pixel_max"] == 0
    assert result["depth"]["role"] == "main"
    assert result["depth"]["depth_max_m"] > 0.0
    assert result["depth"]["reference_frame_id"] == "canonical_station"
    assert len(result["depth"]["frame_graph_hash"]) == 64
    assert result["depth"]["camera_pose_reference_xyz_quat_xyzw"] == pytest.approx(
        [0.4, -0.2, 0.1, 0.0, 0.0, 1.0, 0.0],
        abs=1e-6,
    )
    assert result["camera_diagnostics"]["front"]["black_frame_count"] == 1
    evidence_path = Path(result["evidence_path"])
    assert json.loads(evidence_path.read_text(encoding="utf-8"))["status"] == ("passed")
    assert not socket_path.exists()
    assert [source.closed for source in sources.values()] == [1, 1, 1]


def test_real_perception_probe_closes_resources_when_skill_execution_fails(
    monkeypatch,
    tmp_path,
    short_unix_socket_path_factory,
) -> None:
    binding, sources = _source_binding()
    _patch_probe_hardware(monkeypatch, binding, sources)
    socket_path = short_unix_socket_path_factory("perception-probe-fail")
    monkeypatch.setattr(
        perception_probe,
        "_run_skill",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("probe read failed")),
    )

    with pytest.raises(RuntimeError, match="probe read failed"):
        perception_probe.run_probe(
            run_root=tmp_path / "probe-fail",
            client_id="rpent:perception-probe-test",
            operation_timeout_sec=2.0,
            socket_path=socket_path,
        )

    assert not socket_path.exists()
    assert [source.closed for source in sources.values()] == [1, 1, 1]


def test_depth_v2_fails_closed_if_station_changes_before_artifact_commit(
    monkeypatch,
    tmp_path,
) -> None:
    binding, sources = _source_binding()
    frame_binding = _patch_probe_hardware(monkeypatch, binding, sources)
    client_id = "rpent:perception-revision-test"
    composition = perception_probe._assemble_composition(
        source_binding=binding,
        frame_binding=frame_binding,
        socket_path=tmp_path / "unused.sock",
        artifact_root=tmp_path,
        client_bindings=(
            perception_probe.compile_agent_gateway_unix_client_bindings(
                primary_client_id=client_id,
                primary_client_uid=os.getuid(),
            )
        ),
        runtime_session_id="real-perception:revision-test",
        control_domain_id="real-perception:revision-test",
    )
    try:
        adapter = composition.session.wire_adapter(
            authenticated_client_id=client_id
        )
        composition.session.observation_producer.capture_and_publish(
            composition.session.service
        )
        original_capture = composition.session.provider.capture_depth

        def capture_then_change_station(*, role, expected_station_revision):
            frame = original_capture(
                role=role,
                expected_station_revision=expected_station_revision,
            )
            sources["front"].generation += 1
            return frame

        monkeypatch.setattr(
            composition.session.provider,
            "capture_depth",
            capture_then_change_station,
        )
        errors = perception_probe._capture_executor_errors(composition)
        with pytest.raises(
            RuntimeError,
            match=(
                "did not complete: failed:"
                "passive_read_executor_exception:RuntimeError"
            ),
        ):
            perception_probe._run_skill(
                composition=composition,
                adapter=adapter,
                client_id=client_id,
                client_sequence=1,
                skill_id="perception.capture_depth",
                parameters={"role": "main"},
                timeout_sec=2.0,
                executor_errors=errors,
            )
        assert any(
            "Station revision changed after passive depth capture" in value
            for value in errors.values()
        )
    finally:
        composition.application.close()


def test_real_perception_probe_allocates_short_endpoint_outside_long_ambient_tmpdir(
    monkeypatch,
    tmp_path,
) -> None:
    binding, sources = _source_binding()
    _patch_probe_hardware(monkeypatch, binding, sources)
    long_temp_root = tmp_path / ("harness-runner-slot-" + "x" * 140)
    long_temp_root.mkdir()
    monkeypatch.setattr(perception_probe.tempfile, "tempdir", str(long_temp_root))
    captured: dict[str, Path] = {}

    def reject_after_endpoint_allocation(**kwargs):
        captured["socket_path"] = Path(kwargs["socket_path"])
        raise RuntimeError("stop after endpoint allocation")

    monkeypatch.setattr(
        perception_probe,
        "_assemble_composition",
        reject_after_endpoint_allocation,
    )

    with pytest.raises(RuntimeError, match="stop after endpoint allocation"):
        perception_probe.run_probe(
            run_root=tmp_path / "probe-long-runner-path",
            client_id="rpent:perception-probe-test",
            operation_timeout_sec=2.0,
        )

    socket_path = captured["socket_path"]
    assert len(os.fsencode(socket_path)) <= 107
    assert not socket_path.parent.exists()
    assert [source.closed for source in sources.values()] == [1, 1, 1]
