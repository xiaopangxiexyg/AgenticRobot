from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from teleop_runtime.hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    VIVE_TRACKER_ALIGNMENT_SCHEMA_VERSION,
    add_vive_tracker_height_correction_args,
    load_vive_tracker_alignment_config,
    vive_tracker_height_correction_m_from_args,
    vive_tracker_local_translation_offset_from_args,
)
from teleop_runtime.real_teleop.cli_runtime_config import _resolve_gripper_input_source


def test_default_alignment_config_selects_q6_and_angle_trigger_offsets() -> None:
    config = DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG

    assert config.source_path is not None
    assert config.height_correction_for_gripper_input_source("q6") == pytest.approx(
        0.04
    )
    assert config.height_correction_for_gripper_input_source(
        "angle_trigger"
    ) == pytest.approx(
        0.08,
    )
    np.testing.assert_allclose(
        config.local_translation_offset(gripper_input_source="q6"),
        [0.0, 0.0, -0.04],
    )
    np.testing.assert_allclose(
        config.local_translation_offset(gripper_input_source="angle_trigger"),
        [0.0, 0.0, -0.08],
    )


@pytest.mark.parametrize(
    ("input_source", "expected_height_m"),
    (("q6", 0.04), ("angle_trigger", 0.08)),
)
def test_args_select_configured_offset_after_gripper_input_resolution(
    input_source: str,
    expected_height_m: float,
) -> None:
    args = argparse.Namespace(
        gripper_input_source=input_source,
        vive_tracker_height_correction=True,
        vive_tracker_height_correction_m=None,
    )

    assert vive_tracker_height_correction_m_from_args(args) == pytest.approx(
        expected_height_m
    )
    np.testing.assert_allclose(
        vive_tracker_local_translation_offset_from_args(args),
        [0.0, 0.0, -expected_height_m],
    )


def test_explicit_cli_height_overrides_input_source_config() -> None:
    parser = argparse.ArgumentParser()
    add_vive_tracker_height_correction_args(parser)
    args = parser.parse_args(["--vive-tracker-height-correction-m", "0.06"])
    args.gripper_input_source = "angle_trigger"

    assert vive_tracker_height_correction_m_from_args(args) == pytest.approx(0.06)
    np.testing.assert_allclose(
        vive_tracker_local_translation_offset_from_args(args),
        [0.0, 0.0, -0.06],
    )


def test_automatic_q6_fallback_selects_angle_trigger_alignment() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=False,
        vive_tracker_height_correction=True,
        vive_tracker_height_correction_m=None,
        gui_requested=True,
        hardware_admission_mode="capability",
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": []},
    ):
        _resolve_gripper_input_source(args, ("left", "right"))

    assert args.gripper_input_source == "angle_trigger"
    assert vive_tracker_height_correction_m_from_args(args) == pytest.approx(0.08)
    np.testing.assert_allclose(
        vive_tracker_local_translation_offset_from_args(args),
        [0.0, 0.0, -0.08],
    )


def test_config_loader_rejects_missing_required_input_source(tmp_path) -> None:
    path = tmp_path / "alignment.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": VIVE_TRACKER_ALIGNMENT_SCHEMA_VERSION,
                "height_correction_enabled": True,
                "height_correction_local_axis": [0.0, 0.0, -1.0],
                "default_gripper_input_source": "q6",
                "height_correction_m_by_gripper_input_source": {"q6": 0.04},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="angle_trigger"):
        load_vive_tracker_alignment_config(path)
