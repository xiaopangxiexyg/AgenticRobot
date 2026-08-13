from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_interface.control_trace import ActualDecisionRow, EpisodeControlTrace
from robot_motion.action import HandActionSpec
from sim.demonstration_editing import preflight_zero_edit_replay
from sim.mujoco.adapters import (
    SAFE_ACTION_DEMONSTRATION_STAGE_V1,
    CommittedTraceDemonstrationSelection,
    ControlTraceDemonstrationError,
    DemonstrationSuccessEvidence,
    demonstration_from_committed_control_trace,
)
from sim.mujoco.deploy.measurement import MUJOCO_BACKEND_MEASUREMENT_VERSION
import sim.mujoco.adapters.demonstration_trace as demonstration_trace_module


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = REPO_ROOT / "sim/mujoco/adapters/demonstration_trace.py"


def _action(value: float, *, selected: bool = False) -> dict[str, object]:
    return {
        "action_space": "bimanual_ee_pose_v1",
        "timestamp": 10.0 + value,
        "arm_pose_by_side": {
            "right": [99.0 if selected else value, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        },
        "arm_qpos_by_side": {},
        "hand_qpos_by_side": {"right": [value] * 6},
        "end_effector_targets_by_side": {},
        "active_effectors": ("right_arm", "right_hand"),
        "frame_by_effector": {"right_arm": "right_arm_base"},
        "controlled_frame_by_effector": {"right_arm": "tool_tcp"},
        "source": "fixture",
        "execution_metadata": {
            "hand_action_specs_by_side": {
                "right": HandActionSpec.dmtac_gripper("right").to_metadata(),
            },
        },
    }


def _row(index: int) -> ActualDecisionRow:
    decision_id = f"decision-{index}"
    return ActualDecisionRow(
        episode_uuid="episode-1",
        trace_session_id="session-1",
        decision_index=index,
        decision_id=decision_id,
        decision_journal_sequence=index + 1,
        decision_cutoff_at=10.0 + 0.1 * index,
        decided_at=10.0 + 0.1 * index,
        published_at=10.0 + 0.1 * index,
        effective_interval_start=10.0 + 0.1 * index,
        effective_interval_end=10.1 + 0.1 * index,
        dt_sec=0.1,
        selected_action=_action(float(index), selected=True),
        selected_action_hash=f"selected-{index}",
        selected_by_group={"right": f"proposal-{index}"},
        execution_directive_id=f"directive-{index}",
        execution_status="normal",
        safe_action=_action(0.1 * index),
        safe_action_hash=f"safe-{index}",
        safety_status="safe",
        command_id=f"command-{index}",
        command={"commands": {}},
        dispatch_receipt_ids=(f"dispatch-{index}",),
        measurement_alignment_ids=(f"measurement-{index}",),
        source_decision_ids=(decision_id,),
    )


def _fact(fact_id: str, event_type: str, payload: dict[str, object]):
    return SimpleNamespace(
        fact=SimpleNamespace(
            fact_id=fact_id,
            event_type=event_type,
            payload=payload,
        )
    )


def _trace(*, all_accepted: bool = True, committed: bool = True) -> EpisodeControlTrace:
    records = []
    for index in range(3):
        records.extend(
            (
                _fact(
                    f"dispatch-{index}",
                    "dispatch_receipt",
                    {
                        "command_id": f"command-{index}",
                        "all_accepted": all_accepted,
                        "dispatched_at": 10.0 + 0.1 * index,
                        "receipts_by_effector": {
                            "right_arm": {"accepted": all_accepted},
                            "right_hand": {"accepted": all_accepted},
                        },
                    },
                ),
                _fact(
                    f"measurement-{index}",
                    "measurement_alignment",
                    {
                        "decision_id": f"decision-{index}",
                        "command_id": f"command-{index}",
                        "dispatch_receipt_id": f"dispatch-{index}",
                        "measurement_schema_id": "dexteleop.joint_measurement.v1",
                        "measured_at": 10.01 + 0.1 * index,
                        "measurement": {
                            "schema_version": MUJOCO_BACKEND_MEASUREMENT_VERSION,
                            "measured_qpos_by_side": {"right": [0.0] * 6},
                        },
                    },
                ),
            )
        )
    return EpisodeControlTrace(
        dataset_root=Path("/unused"),
        episode_uuid="episode-1",
        trace_session_id="session-1",
        profile_id="sim-hil-fixture-v1",
        committed=committed,
        capabilities=("control_trace_core",),
        seal_payload={},
        commit_payload=(
            {"semantic_content_hash": "a" * 64} if committed else None
        ),
        episode_records=tuple(records),
        dependency_records=(),
        wal_relative_path="unused",
        wal_tail_status="clean_eof",
    )


def _selection(**overrides) -> CommittedTraceDemonstrationSelection:
    values = {
        "trajectory_id": "successful-right-grasp-1",
        "arm_side": "right",
        "hand_side": "right",
        "hand_component_id": "right_gripper",
        "hand_payload_field": "hand_qpos_by_side",
        "hand_target_representation": "normalized_gripper_close",
        "hand_target_names": ("gripper_close",),
        "hand_target_units": ("1",),
        "execution_hand_effector_id": "right_hand",
        "hand_payload_indices": (0,),
        "decision_start_index": 0,
        "decision_end_index": 2,
        "hand_edit_start_index": 1,
        "hand_edit_end_index": 2,
    }
    values.update(overrides)
    return CommittedTraceDemonstrationSelection(**values)


def _success(**overrides) -> DemonstrationSuccessEvidence:
    values = {
        "evidence_id": "success-evidence-1",
        "kind": "task_terminal_success",
        "semantic_content_hash": "b" * 64,
        "episode_uuid": "episode-1",
        "first_decision_id": "decision-0",
        "last_decision_id": "decision-2",
    }
    values.update(overrides)
    return DemonstrationSuccessEvidence(**values)


def test_committed_trace_exports_safe_pose_and_keeps_execution_evidence_distinct(
    monkeypatch,
) -> None:
    rows = tuple(_row(index) for index in range(3))
    monkeypatch.setattr(
        demonstration_trace_module,
        "materialize_actual_decision_view",
        lambda _trace: rows,
    )

    trajectory = demonstration_from_committed_control_trace(
        _trace(),
        selection=_selection(),
        success_evidence=_success(),
    )

    np.testing.assert_allclose(trajectory.timestamps_sec, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(trajectory.tool_pose_xyzquat_xyzw[:, 0], [0.0, 0.1, 0.2])
    assert not np.any(trajectory.tool_pose_xyzquat_xyzw[:, 0] == 99.0)
    np.testing.assert_allclose(
        trajectory.hand_targets[:, 0],
        [0.0, 0.1, 0.2],
    )
    assert trajectory.reference_frame == "right_arm_base"
    assert trajectory.controlled_frame == "tool_tcp"
    assert trajectory.hand_component_id == "right_gripper"
    assert trajectory.hand_target_representation == "normalized_gripper_close"
    assert trajectory.hand_target_names == ("gripper_close",)
    assert trajectory.source_content_hash == "a" * 64
    assert (
        trajectory.metadata["nominal_action_stage"]
        == SAFE_ACTION_DEMONSTRATION_STAGE_V1
    )
    assert trajectory.metadata["success_evidence_hash"] == "b" * 64
    assert trajectory.metadata["execution_hand_effector_id"] == "right_hand"
    assert trajectory.metadata["hand_payload_indices"] == "0"

    preflight = preflight_zero_edit_replay(trajectory)
    assert preflight.accepted
    assert preflight.max_position_error_m == pytest.approx(0.0)
    assert preflight.max_rotation_error_rad == pytest.approx(0.0)
    assert preflight.max_hand_error == pytest.approx(0.0)


def test_export_rejects_uncommitted_trace_before_materialization(monkeypatch) -> None:
    monkeypatch.setattr(
        demonstration_trace_module,
        "materialize_actual_decision_view",
        lambda _trace: pytest.fail("uncommitted trace must not be materialized"),
    )
    with pytest.raises(ControlTraceDemonstrationError, match="requires a committed"):
        demonstration_from_committed_control_trace(
            _trace(committed=False),
            selection=_selection(),
            success_evidence=_success(),
        )


def test_export_rejects_dispatch_that_was_not_fully_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        demonstration_trace_module,
        "materialize_actual_decision_view",
        lambda _trace: tuple(_row(index) for index in range(3)),
    )
    with pytest.raises(ControlTraceDemonstrationError, match="not fully accepted"):
        demonstration_from_committed_control_trace(
            _trace(all_accepted=False),
            selection=_selection(),
            success_evidence=_success(),
        )


def test_export_rejects_success_evidence_for_another_decision_span(monkeypatch) -> None:
    monkeypatch.setattr(
        demonstration_trace_module,
        "materialize_actual_decision_view",
        lambda _trace: tuple(_row(index) for index in range(3)),
    )
    with pytest.raises(ControlTraceDemonstrationError, match="span does not match"):
        demonstration_from_committed_control_trace(
            _trace(),
            selection=_selection(),
            success_evidence=_success(last_decision_id="decision-1"),
        )


def test_export_rejects_hand_slot_without_matching_execution_spec(monkeypatch) -> None:
    rows = list(_row(index) for index in range(3))
    invalid = dict(rows[0].safe_action)
    invalid["execution_metadata"] = {
        "hand_action_specs_by_side": {
            "right": HandActionSpec.dmtac_gripper(
                "right",
                gripper_slot=1,
            ).to_metadata(),
        },
    }
    rows[0] = replace(rows[0], safe_action=invalid)
    monkeypatch.setattr(
        demonstration_trace_module,
        "materialize_actual_decision_view",
        lambda _trace: tuple(rows),
    )

    with pytest.raises(ControlTraceDemonstrationError, match="is not bound"):
        demonstration_from_committed_control_trace(
            _trace(),
            selection=_selection(),
            success_evidence=_success(),
        )


def test_adapter_does_not_import_application_or_temporary_pipeline() -> None:
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"), filename=str(ADAPTER_PATH))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        name
        for name in imported
        if name.split(".", 1)[0] in {"control_station", "teleop_runtime", "torch"}
        or name == "sim.mujoco_temp_pipeline"
        or name.startswith("sim.mujoco_temp_pipeline.")
    }
    assert forbidden == set()
