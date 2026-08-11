from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    AllocationTransitionContinuityPolicy,
    AllocationTransitionContinuityRuntime,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schema() -> ComponentActionSchema:
    payloads = {
        "arm": ActionPayloadSchema(
            schema_id="continuity-pose",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="world",
            controlled_frame="tool",
        ),
        "gripper": ActionPayloadSchema(
            schema_id="continuity-qpos",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    return ComponentActionSchema(
        schema_id="continuity-action",
        schema_version="development",
        component_catalog=ActionComponentCatalogV2(
            catalog_id="continuity-action",
            catalog_version="development",
            components=tuple(
                ControlComponentSpecV2(
                    component_id=component_id,
                    controlled_entity_id=component_id,
                    command_channel_id=component_id,
                    selection_group_id=component_id,
                    resource_claims=(f"resource:{component_id}",),
                    accepted_payload_schema_hashes=(
                        payloads[component_id].semantic_content_hash,
                    ),
                )
                for component_id in ("arm", "gripper")
            ),
        ),
        payload_schema_by_component=payloads,
    )


def _action(
    *,
    position: tuple[float, float, float],
    yaw_deg: float,
    qpos: tuple[float, float],
) -> ComponentAction:
    quaternion = Rotation.from_euler("z", yaw_deg, degrees=True).as_quat()
    return ComponentAction(
        action_schema=_schema(),
        values_by_component={
            "arm": (*position, *quaternion),
            "gripper": qpos,
        },
    )


def _runtime() -> AllocationTransitionContinuityRuntime:
    runtime = AllocationTransitionContinuityRuntime(
        AllocationTransitionContinuityPolicy(
            policy_id="test-three-state-continuity",
            component_ids=("arm", "gripper"),
            handoff_duration_sec=0.2,
        )
    )
    runtime.activate(
        composition_activation_id="activation:1",
        composition_activation_hash="activation-hash:1",
    )
    return runtime


def _values(action: ComponentAction) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(action.value("arm").values, dtype=np.float64),
        np.asarray(action.value("gripper").values, dtype=np.float64),
    )


def test_absolute_entry_aligns_to_baseline_then_preserves_human_delta() -> None:
    runtime = _runtime()
    baseline = _action(position=(0.0, 0.0, 0.0), yaw_deg=0.0, qpos=(0.0, 0.0))
    first_raw = _action(position=(10.0, 0.0, 0.0), yaw_deg=90.0, qpos=(5.0, -5.0))
    first = runtime.evaluate(
        raw_candidate=first_raw,
        baseline_candidate=baseline,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
            "gripper": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        },
        target_effective_at=1.0,
        invocation_id="cut:1",
        invocation_hash="cut-hash:1",
    )
    first_pose, first_qpos = _values(first.output)
    assert first_pose[:3] == pytest.approx((0.0, 0.0, 0.0))
    assert Rotation.from_quat(first_pose[3:7]).magnitude() == pytest.approx(0.0)
    assert first_qpos == pytest.approx((0.0, 0.0))

    moved_raw = _action(position=(11.0, -2.0, 0.5), yaw_deg=120.0, qpos=(6.0, -3.0))
    moved = runtime.evaluate(
        raw_candidate=moved_raw,
        baseline_candidate=baseline,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
            "gripper": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        },
        target_effective_at=1.1,
        invocation_id="cut:2",
        invocation_hash="cut-hash:2",
    )
    moved_pose, moved_qpos = _values(moved.output)
    assert moved_pose[:3] == pytest.approx((1.0, -2.0, 0.5))
    assert Rotation.from_quat(moved_pose[3:7]).magnitude() == pytest.approx(
        math.radians(30.0)
    )
    assert moved_qpos == pytest.approx((1.0, 2.0))
    assert set(moved.receipt.correction_kind_by_component.values()) == {
        "persistent_absolute_alignment"
    }


def test_absolute_handback_is_continuous_then_converges_to_raw_target() -> None:
    runtime = _runtime()
    baseline = _action(position=(0.0, 0.0, 0.0), yaw_deg=0.0, qpos=(0.0, 0.0))
    absolute = runtime.evaluate(
        raw_candidate=_action(
            position=(10.0, 0.0, 0.0), yaw_deg=90.0, qpos=(5.0, -5.0)
        ),
        baseline_candidate=baseline,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
            "gripper": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        },
        target_effective_at=1.0,
        invocation_id="cut:1",
        invocation_hash="cut-hash:1",
    )
    moved = runtime.evaluate(
        raw_candidate=_action(
            position=(11.0, 0.0, 0.0), yaw_deg=120.0, qpos=(6.0, -3.0)
        ),
        baseline_candidate=baseline,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
            "gripper": COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        },
        target_effective_at=1.1,
        invocation_id="cut:2",
        invocation_hash="cut-hash:2",
    )
    raw_handback = _action(
        position=(2.0, 0.0, 0.0), yaw_deg=0.0, qpos=(2.0, 2.0)
    )
    transition = runtime.evaluate(
        raw_candidate=raw_handback,
        baseline_candidate=raw_handback,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            "gripper": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        },
        target_effective_at=1.2,
        invocation_id="cut:3",
        invocation_hash="cut-hash:3",
    )
    assert _values(transition.output)[0] == pytest.approx(_values(moved.output)[0])
    assert _values(transition.output)[1] == pytest.approx(_values(moved.output)[1])

    halfway = runtime.evaluate(
        raw_candidate=raw_handback,
        baseline_candidate=raw_handback,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            "gripper": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        },
        target_effective_at=1.3,
        invocation_id="cut:4",
        invocation_hash="cut-hash:4",
    )
    halfway_pose, halfway_qpos = _values(halfway.output)
    moved_pose, moved_qpos = _values(moved.output)
    raw_pose, raw_qpos = _values(raw_handback)
    assert halfway_pose[:3] == pytest.approx((moved_pose[:3] + raw_pose[:3]) / 2.0)
    assert halfway_qpos == pytest.approx((moved_qpos + raw_qpos) / 2.0)
    assert Rotation.from_quat(halfway_pose[3:7]).magnitude() == pytest.approx(
        math.radians(15.0)
    )

    completed = runtime.evaluate(
        raw_candidate=raw_handback,
        baseline_candidate=raw_handback,
        selected_mode_by_component={
            "arm": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            "gripper": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        },
        target_effective_at=1.4,
        invocation_id="cut:5",
        invocation_hash="cut-hash:5",
    )
    assert _values(completed.output)[0] == pytest.approx(raw_pose)
    assert _values(completed.output)[1] == pytest.approx(raw_qpos)
    assert set(completed.receipt.correction_kind_by_component.values()) == {"none"}


def test_runtime_is_duplicate_idempotent_and_rejects_time_reordering() -> None:
    runtime = _runtime()
    baseline = _action(position=(0.0, 0.0, 0.0), yaw_deg=0.0, qpos=(0.0, 0.0))
    kwargs = {
        "raw_candidate": baseline,
        "baseline_candidate": baseline,
        "selected_mode_by_component": {
            "arm": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            "gripper": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        },
        "target_effective_at": 1.0,
        "invocation_id": "cut:1",
        "invocation_hash": "cut-hash:1",
    }
    first = runtime.evaluate(**kwargs)
    duplicate = runtime.evaluate(**kwargs)
    assert duplicate is first
    assert runtime.state is not None and runtime.state.revision == 1

    with pytest.raises(ValueError, match="increase strictly"):
        runtime.evaluate(
            **{
                **kwargs,
                "target_effective_at": 0.9,
                "invocation_id": "cut:2",
                "invocation_hash": "cut-hash:2",
            }
        )


def test_prepare_is_non_committing_and_only_current_candidate_can_commit() -> None:
    runtime = _runtime()
    baseline = _action(position=(0.0, 0.0, 0.0), yaw_deg=0.0, qpos=(0.0, 0.0))
    common = {
        "raw_candidate": baseline,
        "baseline_candidate": baseline,
        "selected_mode_by_component": {
            "arm": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            "gripper": COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        },
    }
    first = runtime.prepare(
        **common,
        target_effective_at=1.0,
        invocation_id="cut:1",
        invocation_hash="cut-hash:1",
    )
    assert runtime.state is not None and runtime.state.revision == 0

    replacement = runtime.prepare(
        **common,
        target_effective_at=1.1,
        invocation_id="cut:2",
        invocation_hash="cut-hash:2",
    )
    assert runtime.state is not None and runtime.state.revision == 0
    with pytest.raises(ValueError, match="not the current preparation"):
        runtime.validate_prepared(
            receipt_hash=first.receipt.semantic_content_hash,
        )

    runtime.validate_prepared(
        receipt_hash=replacement.receipt.semantic_content_hash,
    )
    committed = runtime.commit_prepared(
        receipt_hash=replacement.receipt.semantic_content_hash,
    )
    assert committed is replacement
    assert runtime.state is not None and runtime.state.revision == 1
    assert runtime.commit_prepared(
        receipt_hash=replacement.receipt.semantic_content_hash,
    ) is replacement
