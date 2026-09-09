from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from deployments.perception import localize_image_cli, point_grasp_image_cli
from deployments.perception.sam2_grounding_dino import GroundedObjectDetection

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install_perception_models.sh"
POINT_GRASP_INSTALLER = REPO_ROOT / "scripts" / "install_point_grasp_models.sh"
POINT_GRASP_ENV_INSTALLER = REPO_ROOT / "scripts" / "install_point_grasp_env.sh"
GRASPGENX_INSTALLER = REPO_ROOT / "scripts" / "install_graspgenx_env.sh"


def test_perception_model_installer_is_executable_and_documents_fixed_inputs() -> None:
    assert INSTALLER.stat().st_mode & 0o111
    completed = subprocess.run(
        ["bash", str(INSTALLER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--backend auto|cuda|cpu" in completed.stdout
    assert "--verify-only" in completed.stdout
    source = INSTALLER.read_text(encoding="utf-8")
    assert "aa9b8722d0585b661ded4b3dff1bd103540554ae" in source
    assert "856dde20aee659246248e20734ef9ba5214f5e44" in source
    assert "grounding_dino_torch_2_10_compat.h" in source
    assert "DEXTELEOP_ENV_NAME:-dexteleop310" in source


def test_point_grasp_installer_targets_canonical_environment() -> None:
    completed = subprocess.run(
        ["bash", str(POINT_GRASP_INSTALLER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "default: dexteleop310" in completed.stdout
    source = POINT_GRASP_INSTALLER.read_text(encoding="utf-8")
    assert "DEXTELEOP_ENV_NAME:-dexteleop310" in source
    assert '"scipy==1.13.1"' in source
    assert "dexteleop_graspnet_baseline.pth" in source


def test_point_grasp_environment_installer_composes_base_and_model_installers() -> None:
    assert POINT_GRASP_ENV_INSTALLER.stat().st_mode & 0o111
    completed = subprocess.run(
        ["bash", str(POINT_GRASP_ENV_INSTALLER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--update" in completed.stdout
    assert "--skip-vendor-sdk" in completed.stdout
    assert "checkpoints are not downloaded" in completed.stdout.lower()
    source = POINT_GRASP_ENV_INSTALLER.read_text(encoding="utf-8")
    assert "scripts/install_conda_env.sh" in source
    assert "scripts/install_point_grasp_models.sh" in source
    assert "model/perception/graspnet-baseline" in source


def test_graspgenx_installer_uses_isolated_asset_environment() -> None:
    assert GRASPGENX_INSTALLER.stat().st_mode & 0o111
    completed = subprocess.run(
        ["bash", str(GRASPGENX_INSTALLER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "never modifies dexteleop310" in completed.stdout
    assert "--verify-only" in completed.stdout
    source = GRASPGENX_INSTALLER.read_text(encoding="utf-8")
    assert "assets/perception/graspgenx-point-grasp" not in source
    assert 'BUNDLE_ROOT="$ASSET_ROOT/perception/graspgenx-point-grasp"' in source
    assert 'VENV_ROOT="$BUNDLE_ROOT/runtime/.venv"' in source
    assert "adithyamurali/GraspGenXModel" in source
    assert "--torch-backend" in completed.stdout
    assert '"$SOURCE_ROOT[serve]"' not in source
    assert "PyOpenGL" in source
    assert 'PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in source


def test_point_grasp_image_cli_is_offline_and_has_no_gateway_dependency() -> None:
    source = Path(point_grasp_image_cli.__file__).read_text(encoding="utf-8")
    assert "agent_gateway" not in source
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "deployments.perception.point_grasp_image_cli",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--calibration-json" in completed.stdout
    assert "--output-image" in completed.stdout


def _point_grasp_cli_arguments(tmp_path: Path) -> SimpleNamespace:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rgb = tmp_path / "frame.png"
    Image.new("RGB", (16, 12), color=(32, 64, 96)).save(rgb)
    depth = tmp_path / "depth.npy"
    np.save(depth, np.zeros((12, 16), dtype=np.float32), allow_pickle=False)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "camera_frame_id": "camera:test",
                "base_frame_id": "base:test",
                "camera_intrinsics": {
                    "width": 16,
                    "height": 12,
                    "fx": 10.0,
                    "fy": 10.0,
                    "cx": 8.0,
                    "cy": 6.0,
                },
                "T_base_from_camera": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        sam2_asset_bundle=bundle,
        point_grasp_asset_bundle=bundle,
        point_grasp_backend="vgn",
        input_rgb=rgb,
        input_depth=depth,
        calibration_json=calibration,
        target_u=8,
        target_v=6,
        max_candidates=5,
        output_json=tmp_path / "result.json",
        output_image=tmp_path / "annotated.jpg",
        device="cpu",
        overwrite=True,
    )


@pytest.mark.parametrize(
    "aliased_output", ["input_rgb", "input_depth", "calibration_json"]
)
def test_point_grasp_image_cli_never_overwrites_an_input(
    tmp_path: Path,
    aliased_output: str,
) -> None:
    args = _point_grasp_cli_arguments(tmp_path)
    args.output_image = getattr(args, aliased_output)
    if args.output_image.suffix.lower() not in {".jpg", ".jpeg"}:
        args.output_json = args.output_image
        args.output_image = tmp_path / "annotated.jpg"

    with pytest.raises(ValueError, match="paths must be distinct"):
        point_grasp_image_cli.run(args)


def test_point_grasp_image_cli_requires_distinct_output_paths(tmp_path: Path) -> None:
    args = _point_grasp_cli_arguments(tmp_path)
    args.output_json = args.output_image

    with pytest.raises(ValueError, match="paths must be distinct"):
        point_grasp_image_cli.run(args)


def test_point_grasp_image_cli_bounds_depth_before_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _point_grasp_cli_arguments(tmp_path)
    monkeypatch.setattr(point_grasp_image_cli, "MAX_DEPTH_BYTES", 1)

    with pytest.raises(ValueError, match="at most 64 MiB"):
        point_grasp_image_cli.run(args)


def test_point_grasp_cli_annotation_renders_only_selected_candidate(
    monkeypatch,
) -> None:
    calls = []

    def _render(image, *, candidates, **kwargs):
        calls.append((candidates, kwargs))
        return image.copy(), ()

    monkeypatch.setattr(point_grasp_image_cli, "render_dmtac_grasp_overlays", _render)
    pose = tuple(np.eye(4).reshape(-1))
    candidate = SimpleNamespace(
        score=0.9,
        gripper_width_m=0.04,
        camera_from_tcp=pose,
    )
    planned = SimpleNamespace(
        candidates=(candidate, candidate),
        target_mask_bbox_xyxy=(2, 2, 10, 9),
    )

    encoded = point_grasp_image_cli._annotated_jpeg(
        Image.new("RGB", (16, 12)),
        (8, 6),
        planned,
        {"width": 16, "height": 12, "fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
    )

    assert Image.open(io.BytesIO(encoded)).size == (16, 12)
    assert len(calls) == 1
    assert len(calls[0][0]) == 1
    assert calls[0][0][0].selected


class _FakeSpec:
    model_profile_id = "test-sam2-grounding-dino"
    semantic_content_hash = "model-spec:test"

    @classmethod
    def from_asset_bundle(cls, _bundle: Path) -> _FakeSpec:
        return cls()


class _FakeLocalizer:
    instances: ClassVar[list[_FakeLocalizer]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self.calls = []
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def localize(self, image, *, language_prompt, max_detections):
        self.calls.append((image.size, language_prompt, max_detections))
        return (
            GroundedObjectDetection(
                box_xyxy=(2, 3, 11, 9),
                score=0.875,
                label=language_prompt,
            ),
        )


def test_local_image_cli_writes_text_bbox_json_and_annotated_jpeg(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    input_image = tmp_path / "frame.png"
    Image.new("RGB", (16, 12), color=(32, 64, 96)).save(input_image)
    output_json = tmp_path / "result.json"
    output_image = tmp_path / "annotated.jpg"
    _FakeLocalizer.instances.clear()
    monkeypatch.setattr(localize_image_cli, "Sam2GroundingDinoModelSpec", _FakeSpec)
    monkeypatch.setattr(
        localize_image_cli, "Sam2GroundingDinoLocalizer", _FakeLocalizer
    )

    assert (
        localize_image_cli.main(
            [
                "--asset-bundle",
                str(bundle),
                "--input-image",
                str(input_image),
                "--output-json",
                str(output_json),
                "--output-image",
                str(output_image),
                "--language-prompt",
                "apple",
                "--max-detections",
                "7",
            ]
        )
        == 0
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["mode"] == "text"
    assert result["language_prompt"] == "apple"
    assert result["coordinate_format"] == "xyxy_pixel_0based_half_open"
    assert result["detections"] == [
        {
            "box_xyxy": [2, 3, 11, 9],
            "label": "apple",
            "ordinal": 0,
            "score": 0.875,
        }
    ]
    assert Image.open(output_image).size == (16, 12)
    instance = _FakeLocalizer.instances[0]
    assert instance.started and instance.closed
    assert instance.calls == [((16, 12), "apple", 7)]
    assert json.loads(capsys.readouterr().out)["detections"] == 1


def test_local_image_cli_automatic_mode_and_overwrite_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    input_image = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 12), color=(32, 64, 96)).save(input_image)
    output_json = tmp_path / "result.json"
    output_image = tmp_path / "annotated.jpg"
    _FakeLocalizer.instances.clear()
    monkeypatch.setattr(localize_image_cli, "Sam2GroundingDinoModelSpec", _FakeSpec)
    monkeypatch.setattr(
        localize_image_cli, "Sam2GroundingDinoLocalizer", _FakeLocalizer
    )

    arguments = [
        "--asset-bundle",
        str(bundle),
        "--input-image",
        str(input_image),
        "--output-json",
        str(output_json),
        "--output-image",
        str(output_image),
    ]
    assert localize_image_cli.main(arguments) == 0
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["mode"] == "automatic"
    assert result["language_prompt"] is None
    assert _FakeLocalizer.instances[0].calls == [((16, 12), None, 50)]

    try:
        localize_image_cli.main(arguments)
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - argparse must terminate on an unsafe overwrite
        raise AssertionError("CLI replaced existing outputs without --overwrite")
