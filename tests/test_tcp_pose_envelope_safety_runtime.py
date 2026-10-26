from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.target_safety import (
    RobotActionTCPPoseEnvelopeFilter,
    TCPPoseEnvelopeCatalogEntry,
    TCPPoseEnvelopeLifecycleError,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSelection,
    TCPPoseEnvelopeSideConfig,
)


ROUTE_ID = "sim.mujoco.test.motion"


def _pose(x: float) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[0, 3] = x
    return result


def _profile(profile_id: str, delta: float) -> TCPPoseEnvelopeProfile:
    return TCPPoseEnvelopeProfile(
        profile_id=profile_id,
        target_frame_binding_hash="target-binding",
        tool_frame_binding_hash_by_side={"right": "right-tool-binding"},
        reference_frame_by_side={"right": "right_arm_base"},
        side_config_by_side={
            "right": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-1.0, -1.0, -1.0),
                absolute_max_xyz_m=(1.0, 1.0, 1.0),
                relative_min_xyz_m=(-delta, -delta, -delta),
                relative_max_xyz_m=(delta, delta, delta),
                max_rotation_rad=0.5,
            )
        },
    )


def _entry(envelope_id: str, delta: float) -> TCPPoseEnvelopeCatalogEntry:
    return TCPPoseEnvelopeCatalogEntry(
        envelope_id=envelope_id,
        profile=_profile(envelope_id, delta),
        constraint_set_hash=f"constraint:{envelope_id}",
        allowed_route_ids=(ROUTE_ID,),
    )


def _selection(
    entry: TCPPoseEnvelopeCatalogEntry,
    *,
    safety_profile_id: str,
) -> TCPPoseEnvelopeSelection:
    return TCPPoseEnvelopeSelection(
        safety_profile_id=safety_profile_id,
        safety_profile_hash=f"profile-hash:{safety_profile_id}",
        deployment_envelope_id=entry.envelope_id,
        deployment_envelope_hash=entry.semantic_content_hash,
        constraint_set_hash=entry.constraint_set_hash,
    )


def _runtime(*entries: TCPPoseEnvelopeCatalogEntry) -> TCPPoseEnvelopeSafetyRuntime:
    runtime = TCPPoseEnvelopeSafetyRuntime(
        catalog_entries=entries,
        target_frame_binding_hash="target-binding",
        tool_frame_binding_hash_by_side={"right": "right-tool-binding"},
        reference_frame_by_side={"right": "right_arm_base"},
        clock_domain_id="test-monotonic",
    )
    runtime.commit_reference(
        reference_id="qpos-init-reference:0",
        tcp_pose_by_side={"right": _pose(0.0)},
        captured_at_by_side={"right": 1.0},
        source_event_ids=("qpos-initialization-completed:0",),
        lower_state="configured",
    )
    return runtime


def test_readonly_preview_uses_committed_reference_without_activation_or_mutation():
    entry = _entry("envelope.standard", .2)
    runtime = _runtime(entry)
    before = runtime.snapshot()
    preview = runtime.preview_for_envelope({"right": _pose(.4)},
        deployment_envelope_hash=entry.semantic_content_hash)
    assert preview.modified_sides == ("right",)
    assert runtime.snapshot() == before
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="absent"):
        runtime.preview_for_envelope({"right": _pose(0)}, deployment_envelope_hash="unknown")
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="no operation-local"):
        runtime.project({"right": _pose(0)})
    runtime.activate(selection=_selection(entry, safety_profile_id="test"),
        operation_id="operation:1", route_id=ROUTE_ID, execution_epoch=3,
        source_event_ids=("operation:1",), lower_state="hold")
    active_before = runtime.snapshot()
    runtime.preview_for_envelope({"right": _pose(.1)}, deployment_envelope_hash=entry.semantic_content_hash)
    assert runtime.snapshot() == active_before
    normal = runtime.project({"right": _pose(.4)})
    np.testing.assert_array_equal(normal.effective_tcp_pose_by_side["right"], preview.effective_tcp_pose_by_side["right"])


def test_hold_rebinds_epoch_without_moving_stable_reference() -> None:
    standard = _entry("envelope.standard", 0.20)
    strict = _entry("envelope.strict", 0.05)
    runtime = _runtime(standard, strict)
    reference_hash = runtime.snapshot().safety_reference_hash

    first = runtime.activate(
        selection=_selection(
            standard,
            safety_profile_id="safety.standard@1",
        ),
        operation_id="operation:1",
        route_id=ROUTE_ID,
        execution_epoch=3,
        source_event_ids=("operation:1", "execution-epoch:3"),
        lower_state="hold",
    )
    projected = runtime.project({"right": _pose(0.15)})
    assert projected.effective_tcp_pose_by_side["right"][0, 3] == pytest.approx(0.15)

    runtime.deactivate_after_hold(
        operation_id="operation:1",
        next_execution_epoch=4,
        hold_evidence_ids=("hold:operation:1",),
    )
    after_hold = runtime.snapshot()
    assert after_hold.safety_reference_hash == reference_hash
    assert after_hold.active_operation_id is None

    second = runtime.activate(
        selection=_selection(strict, safety_profile_id="safety.strict@1"),
        operation_id="operation:2",
        route_id=ROUTE_ID,
        execution_epoch=4,
        source_event_ids=("operation:2", "execution-epoch:4"),
        lower_state="hold",
    )
    limited = runtime.project({"right": _pose(0.15)})
    assert limited.effective_tcp_pose_by_side["right"][0, 3] == pytest.approx(0.05)
    assert second.safety_reference_hash == first.safety_reference_hash
    assert second.operation_anchor_hash != first.operation_anchor_hash

    runtime.deactivate_after_hold(
        operation_id="operation:2",
        next_execution_epoch=5,
        hold_evidence_ids=("hold:operation:2",),
    )
    runtime.activate(
        selection=_selection(
            standard,
            safety_profile_id="safety.standard@1",
        ),
        operation_id="operation:3",
        route_id=ROUTE_ID,
        execution_epoch=5,
        source_event_ids=("operation:3", "execution-epoch:5"),
        lower_state="hold",
    )
    anti_ratchet = runtime.project({"right": _pose(0.25)})
    assert anti_ratchet.effective_tcp_pose_by_side["right"][0, 3] == pytest.approx(0.20)


def test_reset_invalidates_reference_until_fresh_measured_commit() -> None:
    entry = _entry("envelope.standard", 0.10)
    runtime = _runtime(entry)
    original_hash = runtime.snapshot().safety_reference_hash
    epoch = runtime.invalidate_reference(
        reason="scene_reset",
        source_event_ids=("scene-reset:1",),
        lower_state="configured",
    )
    assert epoch == 1
    assert runtime.snapshot().safety_reference_hash is None
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="no committed"):
        runtime.activate(
            selection=_selection(entry, safety_profile_id="safety.standard@1"),
            operation_id="operation:after-reset",
            route_id=ROUTE_ID,
            execution_epoch=7,
            source_event_ids=("operation:after-reset",),
            lower_state="hold",
        )

    fresh = runtime.commit_reference(
        reference_id="qpos-init-reference:1",
        tcp_pose_by_side={"right": _pose(0.4)},
        captured_at_by_side={"right": 2.0},
        source_event_ids=("qpos-initialization-completed:1",),
        lower_state="hold",
    )
    assert fresh.safety_reference_epoch == 1
    assert fresh.semantic_content_hash != original_hash
    runtime.activate(
        selection=_selection(entry, safety_profile_id="safety.standard@1"),
        operation_id="operation:after-reset",
        route_id=ROUTE_ID,
        execution_epoch=7,
        source_event_ids=("operation:after-reset",),
        lower_state="hold",
    )
    projected = runtime.project({"right": _pose(0.8)})
    assert projected.effective_tcp_pose_by_side["right"][0, 3] == pytest.approx(0.5)


def test_unknown_or_cross_route_selection_fails_before_activation() -> None:
    entry = _entry("envelope.standard", 0.10)
    runtime = _runtime(entry)
    invalid = TCPPoseEnvelopeSelection(
        safety_profile_id="safety.standard@1",
        safety_profile_hash="profile-hash",
        deployment_envelope_id=entry.envelope_id,
        deployment_envelope_hash="unknown-envelope-hash",
        constraint_set_hash=entry.constraint_set_hash,
    )
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="not in the lower catalog"):
        runtime.activate(
            selection=invalid,
            operation_id="operation:unknown",
            route_id=ROUTE_ID,
            execution_epoch=1,
            source_event_ids=("operation:unknown",),
            lower_state="hold",
        )
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="does not allow"):
        runtime.activate(
            selection=_selection(entry, safety_profile_id="safety.standard@1"),
            operation_id="operation:wrong-route",
            route_id="sim.mujoco.other.motion",
            execution_epoch=1,
            source_event_ids=("operation:wrong-route",),
            lower_state="hold",
        )


def test_robot_action_filter_projects_only_the_configured_side() -> None:
    entry = _entry("envelope.standard", 0.10)
    runtime = _runtime(entry)
    runtime.activate(
        selection=_selection(entry, safety_profile_id="safety.standard@1"),
        operation_id="operation:filter",
        route_id=ROUTE_ID,
        execution_epoch=1,
        source_event_ids=("operation:filter",),
        lower_state="hold",
    )
    target = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "left": (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "right": (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
        frame_by_effector={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
    )
    effective = RobotActionTCPPoseEnvelopeFilter(runtime)(target)
    assert effective.arm_pose_by_side["left"][0] == pytest.approx(0.3)
    assert effective.arm_pose_by_side["right"][0] == pytest.approx(0.1)


def test_absolute_only_envelope_does_not_claim_an_anchor() -> None:
    profile = TCPPoseEnvelopeProfile(
        profile_id="envelope.absolute-only",
        target_frame_binding_hash="target-binding",
        tool_frame_binding_hash_by_side={"right": "right-tool-binding"},
        reference_frame_by_side={"right": "right_arm_base"},
        side_config_by_side={
            "right": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-0.1, -0.1, -0.1),
                absolute_max_xyz_m=(0.1, 0.1, 0.1),
            )
        },
    )
    entry = TCPPoseEnvelopeCatalogEntry(
        envelope_id="envelope.absolute-only",
        profile=profile,
        constraint_set_hash="constraint:absolute-only",
        allowed_route_ids=(ROUTE_ID,),
    )
    runtime = _runtime(entry)

    receipt = runtime.activate(
        selection=_selection(entry, safety_profile_id="safety.standard@1"),
        operation_id="operation:absolute-only",
        route_id=ROUTE_ID,
        execution_epoch=1,
        source_event_ids=("operation:absolute-only",),
        lower_state="hold",
    )

    assert receipt.safety_reference_hash is None
    assert receipt.operation_anchor_hash is None


def test_robot_action_filter_rejects_wrong_reference_frame() -> None:
    entry = _entry("envelope.standard", 0.10)
    runtime = _runtime(entry)
    runtime.activate(
        selection=_selection(entry, safety_profile_id="safety.standard@1"),
        operation_id="operation:wrong-frame",
        route_id=ROUTE_ID,
        execution_epoch=1,
        source_event_ids=("operation:wrong-frame",),
        lower_state="hold",
    )
    target = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "right": (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
        frame_by_effector={"right_arm": "world"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
    )

    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="reference-frame"):
        RobotActionTCPPoseEnvelopeFilter(runtime)(target)


def test_robot_action_filter_reconstructs_float32_quaternion_in_float64() -> None:
    entry = _entry("envelope.standard", 0.10)
    runtime = _runtime(entry)
    runtime.activate(
        selection=_selection(entry, safety_profile_id="safety.standard@1"),
        operation_id="operation:float32-quaternion",
        route_id=ROUTE_ID,
        execution_epoch=1,
        source_event_ids=("operation:float32-quaternion",),
        lower_state="hold",
    )
    target = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "right": np.asarray(
                (
                    0.05,
                    0.0,
                    0.0,
                    -0.5064188241958618,
                    0.4796024262905121,
                    0.5148372054100037,
                    0.4984617531299591,
                ),
                dtype=np.float32,
            ),
        },
        frame_by_effector={"right_arm": "right_arm_base"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
    )

    effective = RobotActionTCPPoseEnvelopeFilter(runtime)(target)

    assert effective.arm_pose_by_side["right"][0] == pytest.approx(0.05)
    assert runtime.snapshot().last_projection_hash is not None


def test_planning_preview_matches_active_projection_without_activation_or_state_update():
    entry = _entry("envelope.preview", .10)
    runtime = _runtime(entry)
    before = runtime.snapshot()
    projector = runtime.planning_projector_for_route(route_id=ROUTE_ID)
    preview = projector({"right": _pose(.15)})
    assert runtime.snapshot() == before
    assert preview.modified_sides == ("right",)
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="no operation-local"):
        runtime.project({"right": _pose(.15)})
    runtime.activate(selection=_selection(entry, safety_profile_id="test-preview"),
        operation_id="operation:preview-check", route_id=ROUTE_ID, execution_epoch=3,
        source_event_ids=("operation:preview-check",), lower_state="hold")
    active = runtime.project({"right": _pose(.15)})
    np.testing.assert_array_equal(active.effective_tcp_pose_by_side["right"],
                                  preview.effective_tcp_pose_by_side["right"])
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="closed motion"):
        projector({"right": _pose(.15)})



def test_planning_preview_rejects_missing_ambiguous_and_invalidated_bindings():
    first, second = _entry("first", .10), _entry("second", .20)
    runtime = _runtime(first, second)
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="one exact"):
        runtime.planning_projector_for_route(route_id=ROUTE_ID)
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="one exact"):
        runtime.planning_projector_for_route(route_id="unknown-route")
    runtime = _runtime(first)
    projector = runtime.planning_projector_for_route(route_id=ROUTE_ID)
    runtime.invalidate_reference(reason="reset", source_event_ids=("reset:1",), lower_state="configured")
    before = runtime.snapshot()
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="no committed"):
        runtime.planning_projector_for_route(route_id=ROUTE_ID)
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="reference changed"):
        projector({"right": _pose(.1)})
    assert runtime.snapshot() == before
