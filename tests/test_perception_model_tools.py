from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

from deployments.perception import localize_image_cli
from deployments.perception.sam2_grounding_dino import GroundedObjectDetection


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install_perception_models.sh"


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


class _FakeSpec:
    model_profile_id = "test-sam2-grounding-dino"
    semantic_content_hash = "model-spec:test"

    @classmethod
    def from_asset_bundle(cls, _bundle: Path) -> "_FakeSpec":
        return cls()


class _FakeLocalizer:
    instances: list["_FakeLocalizer"] = []

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
    monkeypatch.setattr(localize_image_cli, "Sam2GroundingDinoLocalizer", _FakeLocalizer)

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
    monkeypatch.setattr(localize_image_cli, "Sam2GroundingDinoLocalizer", _FakeLocalizer)

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
