from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import perception_probe, realsense_perception
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
                "source_generation": 1,
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
            "generation": 1,
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


def _patch_probe_hardware(monkeypatch, binding, sources) -> None:
    monkeypatch.setattr(
        perception_probe,
        "_scene_geometry",
        lambda: ({"left": np.eye(4), "right": np.eye(4)}, "scene:test"),
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
