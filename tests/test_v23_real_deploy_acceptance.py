from __future__ import annotations

import json
from pathlib import Path

from teleop_runtime.deploy.acceptance import compare_real_deploy_debug_logs


def _pose(x: float, y: float) -> list[float]:
    return [x, y, 0.3, 0.0, 0.0, 0.0, 1.0]


def _events(mode: str, *, target_offset: float = 0.0) -> list[dict]:
    startup = {
        "event": "startup",
        "send_actions": True,
        "args": {
            "policy_action_pose_mode": mode,
            "debug_replay_output_pose_mode": mode,
            "debug_replay_dataset": "/dataset",
            "debug_replay_parquet": None,
            "debug_replay_episode": 0,
            "debug_replay_start_index": 0,
            "debug_replay_end_index": 3,
            "debug_replay_chunk_size": 2,
            "arm_control_mode": "impedance",
            "active_sides": ["left", "right"],
            "end_effector_sides": [],
        },
    }
    events = [startup]
    for source_index in (1, 2):
        poses = {
            "left": _pose(0.1 * source_index + target_offset, -0.2),
            "right": _pose(0.1 * source_index + target_offset, 0.2),
        }
        events.append(
            {
                "event": "action_step",
                "dispatch_error": None,
                "faults": [],
                "raw_action": {
                    "metadata": {
                        "debug_replay_source_action_index": source_index,
                    },
                    "arm_pose_by_side": poses,
                },
                "command": {"arm_pose_by_side": poses},
                "pre_arm_state_by_side": {
                    side: {
                        "current_tcp_xyzquat": pose,
                        "hardware_qpos": [0.01 * source_index] * 6,
                    }
                    for side, pose in poses.items()
                },
            }
        )
    return events


def _write(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_real_deploy_acceptance_passes_equal_decoded_and_measured_paths(tmp_path) -> None:
    absolute = tmp_path / "absolute.jsonl"
    relative = tmp_path / "relative.jsonl"
    _write(absolute, _events("absolute"))
    _write(relative, _events("relative"))

    report = compare_real_deploy_debug_logs(
        absolute,
        relative,
        expected_actions=2,
    )

    assert report["passed"] is True
    assert report["source_action_indices"] == (1, 2)


def test_real_deploy_acceptance_detects_wrong_relative_target(tmp_path) -> None:
    absolute = tmp_path / "absolute.jsonl"
    relative = tmp_path / "relative.jsonl"
    _write(absolute, _events("absolute"))
    _write(relative, _events("relative", target_offset=0.01))

    report = compare_real_deploy_debug_logs(absolute, relative)

    assert report["passed"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "decoded_runtime_absolute_target:left" in failed
    assert "decoded_runtime_absolute_target:right" in failed


def test_real_deploy_acceptance_requires_stable_replay_lineage(tmp_path) -> None:
    absolute = tmp_path / "absolute.jsonl"
    relative = tmp_path / "relative.jsonl"
    absolute_events = _events("absolute")
    del absolute_events[1]["raw_action"]["metadata"][
        "debug_replay_source_action_index"
    ]
    _write(absolute, absolute_events)
    _write(relative, _events("relative"))

    try:
        compare_real_deploy_debug_logs(absolute, relative)
    except ValueError as exc:
        assert "regenerate the log" in str(exc)
    else:
        raise AssertionError("missing replay lineage must fail closed")
