from __future__ import annotations

import json
import math

from teleop_runtime.debug.analyze_position_root_x import analyze_root_x, render_analysis


def _pose(x: float) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def test_root_x_analysis_joins_accepted_qcmd_to_its_reference(tmp_path) -> None:
    reference_row = {
        "epoch": 9,
        "reference_sequence": 42,
        "published_wall_time": 123.25,
        "references": {
            "left": {
                "hold": True,
                "frame_metadata": {
                    "source_target_pose_urdf_root": _pose(0.30),
                    "q_ref_pose_urdf_root": _pose(0.20),
                    "target_activity": {"state": "INACTIVE"},
                    "collision_avoidance_status": "tracking",
                },
            }
        },
    }
    command_rows = [
        {
            "side": "left",
            "sdk_accepted": True,
            "reference_epoch": 9,
            "reference_sequence": 42,
            "metadata": {"sdk_accepted_qcmd_pose_urdf_root": _pose(0.15)},
        },
        {
            "side": "left",
            "sdk_accepted": False,
            "reference_epoch": 9,
            "reference_sequence": 42,
            "metadata": {"sdk_accepted_qcmd_pose_urdf_root": _pose(0.10)},
        },
    ]
    (tmp_path / "position_reference.jsonl").write_text(
        json.dumps(reference_row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "position_command_150.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in command_rows),
        encoding="utf-8",
    )

    analysis = analyze_root_x(tmp_path)

    assert analysis.references_with_root_poses == 1
    assert analysis.accepted_qcmd_rows == 1
    assert len(analysis.samples) == 1
    sample = analysis.samples[0]
    assert math.isclose(sample.target_minus_qref_x_m, 0.10)
    assert math.isclose(sample.qref_minus_qcmd_x_m, 0.05)
    assert math.isclose(sample.target_minus_qcmd_x_m, 0.15)
    text = render_analysis(analysis, top=1, threshold_m=0.01)
    assert "matched=1" in text
    assert "target-qcmd=+0.1500" in text
