from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from teleop_runtime.collection import calibration_profile, run_collect


def test_run_collect_preserves_calibration_utility_aliases() -> None:
    assert (
        run_collect._calibration_profile_timestamp
        is calibration_profile.calibration_profile_timestamp
    )
    assert (
        run_collect._latest_calibration_profile_path
        is calibration_profile.latest_calibration_profile_path
    )
    assert run_collect._qpos6_deg_to_rad is calibration_profile.qpos6_deg_to_rad
    assert (
        run_collect._qpos6_radians_look_suspicious
        is calibration_profile.qpos6_radians_look_suspicious
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("20260825_153000.json", True),
        ("20260825_153000_backup.json", True),
        ("20260825.json", False),
        ("20261301_010000.json", False),
        ("profile.json", False),
    ),
)
def test_calibration_profile_timestamp(name: str, expected: bool) -> None:
    value = calibration_profile.calibration_profile_timestamp(
        Path(name)
    )
    assert (value is not None) is expected


def test_latest_profile_prefers_newest_timestamped_name(tmp_path) -> None:
    older = tmp_path / "20260824_235959.json"
    newer = tmp_path / "20260825_000001.json"
    fallback = tmp_path / "profile.json"
    for path in (older, newer, fallback):
        path.write_text("{}", encoding="utf-8")
    os.utime(newer, (1, 1))
    os.utime(fallback, (999999999, 999999999))

    assert calibration_profile.latest_calibration_profile_path(tmp_path) == newer


def test_latest_profile_uses_mtime_when_names_have_no_timestamp(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    os.utime(first, (10, 10))
    os.utime(second, (20, 20))

    assert calibration_profile.latest_calibration_profile_path(tmp_path) == second


def test_qpos_conversion_and_suspicious_value_guard() -> None:
    radians = calibration_profile.qpos6_deg_to_rad((0, 90, -90, 180, 0, 45))

    assert radians[0] == pytest.approx(0.0)
    assert radians[1] == pytest.approx(math.pi / 2.0)
    assert radians[2] == pytest.approx(-math.pi / 2.0)
    assert radians[3] == pytest.approx(math.pi)
    assert calibration_profile.qpos6_radians_look_suspicious(radians) is False
    assert calibration_profile.qpos6_radians_look_suspicious((0, 0, 0, 0, 0, 7)) is True


def test_qpos_helpers_require_six_values() -> None:
    with pytest.raises(ValueError):
        calibration_profile.qpos6_deg_to_rad((0, 1, 2))
    with pytest.raises(ValueError):
        calibration_profile.qpos6_radians_look_suspicious((0, 1, 2))
