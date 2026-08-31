from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.real import realsense_perception
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

REPO_ROOT = Path(__file__).resolve().parents[1]


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
