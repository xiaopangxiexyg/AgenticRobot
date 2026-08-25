from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from teleop_runtime.deploy import camera_cli_validation, run_policy


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "camera_mode": "none",
        "camera_serial": "",
        "camera_config": "/does/not/exist.json",
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "camera_keep_last": 4,
        "camera_frame_max_age_sec": 0.2,
        "camera_wait_timeout_ms": 100,
        "camera_stale_after_sec": 0.5,
        "camera_start_stagger_sec": 0.0,
        "vis_cam": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_run_policy_uses_the_extracted_validator() -> None:
    assert (
        run_policy.validate_camera_cli_args
        is camera_cli_validation.validate_camera_cli_args
    )


def test_none_mode_without_preview_is_valid() -> None:
    camera_cli_validation.validate_camera_cli_args(_args())


def test_serial_single_camera_does_not_require_a_config_file() -> None:
    camera_cli_validation.validate_camera_cli_args(
        _args(camera_mode="single", camera_serial="123")
    )


def test_multi_camera_requires_config_and_rejects_serial() -> None:
    with pytest.raises(ValueError, match="uses --camera-config"):
        camera_cli_validation.validate_camera_cli_args(
            _args(camera_mode="multi", camera_serial="123")
        )

    with pytest.raises(ValueError, match="requires an existing --camera-config"):
        camera_cli_validation.validate_camera_cli_args(_args(camera_mode="multi"))


def test_existing_multi_camera_config_is_valid(tmp_path: Path) -> None:
    config = tmp_path / "cameras.json"
    config.write_text("{}", encoding="utf-8")

    camera_cli_validation.validate_camera_cli_args(
        _args(camera_mode="multi", camera_config=config)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("camera_width", 0, "camera-width/--camera-height must be positive"),
        ("camera_height", 0, "camera-width/--camera-height must be positive"),
        ("camera_fps", 0, "camera-fps must be positive"),
        ("camera_keep_last", 0, "camera-keep-last must be positive"),
        (
            "camera_frame_max_age_sec",
            0.0,
            "camera-frame-max-age-sec must be positive",
        ),
        ("camera_wait_timeout_ms", 0, "camera-wait-timeout-ms must be positive"),
        (
            "camera_stale_after_sec",
            0.0,
            "camera-stale-after-sec must be positive",
        ),
        ("camera_start_stagger_sec", -0.1, "camera-start-stagger-sec must be >= 0"),
    ),
)
def test_camera_numeric_constraints_are_preserved(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        camera_cli_validation.validate_camera_cli_args(
            _args(camera_mode="single", camera_serial="123", **{field: value})
        )


def test_preview_requires_a_camera_mode() -> None:
    with pytest.raises(ValueError, match="--vis_cam requires"):
        camera_cli_validation.validate_camera_cli_args(_args(vis_cam=True))
