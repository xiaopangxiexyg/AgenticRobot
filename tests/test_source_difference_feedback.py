from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from control_composition.composition import (
    CompositionInputDraft,
    CompositionInputStore,
    CompositionSpec,
    DeterministicComposer,
    build_composition_input_snapshot,
    project_composition_inputs,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.observability import (
    DifferenceReceiptReference,
    GuidanceFeedbackSpec,
    NoIoGuidanceFeedbackAdapter,
    SourceDifferenceImmediateObserver,
    SourceDifferenceSpec,
    SourceToComposedDifferenceObserver,
    build_guidance_feedback_command,
    evaluate_composition_source_differences,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_HAND16_V1,
    HandActionSpec,
    RobotAction,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from dataset_interface.composition_control_trace import (
    project_guidance_feedback_command,
    project_guidance_feedback_receipt,
    project_source_difference_observation,
    project_source_difference_spec,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return float(self.value)


def _pose(
    x: float,
    yaw_degrees: float,
    *,
    frame: str = "left_base",
) -> RobotAction:
    quaternion = Rotation.from_euler(
        "z",
        yaw_degrees,
        degrees=True,
    ).as_quat()
    return RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "left": np.concatenate(
                ([x, -0.1, 0.3], quaternion)
            ).astype(np.float32)
        },
        active_effectors=("left_arm",),
        frame_by_effector={"left_arm": frame},
        source="fixture",
    )


def _evaluate(
    action_by_role,
    *,
    spec: CompositionSpec,
):
    canonical_action_by_role = {}
    action_schema = None
    for role in spec.input_roles:
        canonical = project_robot_action_v1_to_component_action(
            action_by_role[role],
            action_schema=action_schema,
        )
        if action_schema is None:
            action_schema = canonical.action_schema
        canonical_action_by_role[role] = canonical
    assert action_schema is not None

    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            "control.monotonic",
            "session-0",
            "boot-0",
            0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    store = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=TimeWindowRetentionSpec(
            spec_id="difference-test-window-v1",
            history_window_sec=0.0,
            max_future_lead_sec=2.0,
            max_effective_duration_sec=2.0,
        ),
        action_schema=action_schema,
    )
    for sequence, role in enumerate(spec.input_roles):
        producer = f"{role}-producer"
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=1.0,
            produced_clock_domain_id=f"{producer}.clock",
            reason="fixture",
        )
        receipt = store.admit(
            CompositionInputDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                producer_id=producer,
                producer_epoch=1,
                sequence=sequence,
                input_role=role,
                action=canonical_action_by_role[role],
                produced_at=9.0 + 0.1 * sequence,
                produced_clock_domain_id=f"{producer}.clock",
                effective_from=10.0 + 0.01 * sequence,
                effective_until=11.0,
                effective_clock_domain_id="control.monotonic",
            )
        )
        assert receipt.accepted
    cut = coordinator.capture_cut(cutoff_at=10.5).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    result = DeterministicComposer().compose(snapshot, spec=spec)
    return snapshot, result


def _pose_spec() -> CompositionSpec:
    return CompositionSpec(
        spec_id="single-arm-human-policy-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "policy_pose"),
        required_roles=("human_pose", "policy_pose"),
        weight_by_role={"human_pose": 0.5, "policy_pose": 0.5},
        max_translation_disagreement_m=1.0,
        max_rotation_disagreement_rad=math.radians(120.0),
        pose_component_dependencies=("left_arm",),
    )


def _difference_spec(spec: CompositionSpec) -> SourceDifferenceSpec:
    return SourceDifferenceSpec(
        spec_id="human-policy-source-difference-v1",
        composition_spec_hash=spec.semantic_content_hash,
        compared_reference_role_pairs=(("human_pose", "policy_pose"),),
        output_residual_roles=("human_pose", "policy_pose"),
    )


def test_source_difference_is_bound_to_one_cut_and_has_signed_pose_error() -> None:
    spec = _pose_spec()
    snapshot, result = _evaluate(
        {
            "human_pose": _pose(0.2, 0.0),
            "policy_pose": _pose(0.5, 90.0),
        },
        spec=spec,
    )
    observation = evaluate_composition_source_differences(
        snapshot,
        result,
        spec=spec,
        difference_spec=_difference_spec(spec),
        evaluated_at=10.5,
        clock_domain_id="control.monotonic",
    )

    assert observation.composition_input_snapshot_id == (
        snapshot.composition_input_snapshot_id
    )
    assert observation.target_effective_at == snapshot.target_effective_at
    assert len(observation.pairwise_differences) == 1
    pair = observation.pairwise_differences[0]
    assert pair.compared_role == "human_pose"
    assert pair.reference_role == "policy_pose"
    metric = pair.component_metrics[0]
    assert metric.component_id == "left_arm"
    np.testing.assert_allclose(
        metric.translation_delta_xyz_m,
        [0.3, 0.0, 0.0],
        atol=1e-7,
    )
    assert metric.rotation_angle_rad == pytest.approx(
        math.radians(90.0),
        abs=1e-6,
    )
    assert len(observation.output_residuals) == 2
    assert observation.valid_until == 11.0


def test_immediate_difference_observer_returns_only_source_pair_payload() -> None:
    spec = _pose_spec()
    snapshot, result = _evaluate(
        {
            "human_pose": _pose(0.2, 0.0),
            "policy_pose": _pose(0.5, 90.0),
        },
        spec=spec,
    )
    observer = SourceDifferenceImmediateObserver(
        composition_spec=spec,
        difference_spec=_difference_spec(spec),
        capability_id="fixture.immediate.pose",
    )
    envelope = observer.observe(
        snapshot,
        observed_at=10.5,
        clock_domain_id="control.monotonic",
    )

    assert result.composition_evaluation_id not in envelope.payload
    assert envelope.status == "available"
    assert envelope.payload["difference_scope"] == "source_to_source"
    assert len(envelope.payload["pairwise_differences"]) == 1
    assert envelope.composition_evaluation_id is None
    assert envelope.valid_until == pytest.approx(11.0)


def test_post_mix_observer_returns_result_lineage_and_receipt_references() -> None:
    spec = _pose_spec()
    snapshot, result = _evaluate(
        {
            "human_pose": _pose(0.2, 0.0),
            "policy_pose": _pose(0.5, 90.0),
        },
        spec=spec,
    )
    observer = SourceToComposedDifferenceObserver(
        composition_spec=spec,
        difference_spec=_difference_spec(spec),
        capability_id="fixture.post-mix.pose",
    )
    reference = DifferenceReceiptReference(
        stage="dispatch",
        receipt_id="dispatch:fixture",
        receipt_hash="dispatch-hash:fixture",
        observed_at=10.6,
        clock_domain_id="control.monotonic",
    )
    envelope = observer.observe(
        snapshot,
        result,
        observed_at=10.5,
        clock_domain_id="control.monotonic",
        receipt_references=(reference,),
    )

    assert envelope.status == "available"
    assert envelope.payload["difference_scope"] == "source_to_composed"
    assert len(envelope.payload["output_residuals"]) == 2
    assert envelope.composition_evaluation_id == result.composition_evaluation_id
    assert envelope.receipt_references == (reference,)


def test_source_difference_qpos_uses_declared_active_slots() -> None:
    spec = _pose_spec()
    active_indices = (0, 4, 5)
    hand_spec = HandActionSpec(
        side="left",
        hand_model="fixture-hand16",
        slot_count=16,
        active_dof=len(active_indices),
        active_mask=tuple(index in active_indices for index in range(16)),
        joint_names=("q0", "q4", "q5"),
    )
    human = np.full(16, np.nan, dtype=np.float32)
    policy = np.full(16, np.nan, dtype=np.float32)
    human[list(active_indices)] = [0.0, 0.4, 0.5]
    policy[list(active_indices)] = [0.3, 0.1, 0.5]
    metadata = {
        "hand_action_specs_by_side": {"left": hand_spec.to_metadata()}
    }
    snapshot, result = _evaluate(
        {
            "human_pose": RobotAction(
                action_space=BIMANUAL_QPOS_HAND16_V1,
                hand_qpos_by_side={"left": human},
                active_effectors=("left_hand",),
                metadata=metadata,
            ),
            "policy_pose": RobotAction(
                action_space=BIMANUAL_QPOS_HAND16_V1,
                hand_qpos_by_side={"left": policy},
                active_effectors=("left_hand",),
                metadata=metadata,
            ),
        },
        spec=spec,
    )
    observation = evaluate_composition_source_differences(
        snapshot,
        result,
        spec=spec,
        difference_spec=_difference_spec(spec),
        evaluated_at=10.5,
        clock_domain_id="control.monotonic",
    )
    metric = observation.pairwise_differences[0].component_metrics[0]

    assert result.output_status == "invalid"
    assert metric.qpos_active_indices == (0, 4, 5)
    np.testing.assert_allclose(metric.qpos_delta, [0.3, -0.3, 0.0])
    assert observation.output_residuals == ()


def test_composition_input_schema_rejects_implicit_frame_conversion() -> None:
    spec = _pose_spec()
    with pytest.raises(ValueError, match="does not match the runtime payload schema"):
        _evaluate(
            {
                "human_pose": _pose(0.2, 0.0, frame="left_base"),
                "policy_pose": _pose(0.3, 0.0, frame="canonical_desk"),
            },
            spec=spec,
        )


def test_virtual_guidance_is_bounded_ttl_checked_and_non_authoritative() -> None:
    spec = _pose_spec()
    snapshot, result = _evaluate(
        {
            "human_pose": _pose(0.2, 0.0),
            "policy_pose": _pose(0.5, 90.0),
        },
        spec=spec,
    )
    observation = evaluate_composition_source_differences(
        snapshot,
        result,
        spec=spec,
        difference_spec=_difference_spec(spec),
        evaluated_at=10.5,
        clock_domain_id="control.monotonic",
    )
    feedback_spec = GuidanceFeedbackSpec(
        spec_id="virtual-guidance-v1",
        compared_role="human_pose",
        reference_role="policy_pose",
        translation_full_scale_m=0.1,
        rotation_full_scale_rad=math.radians(30.0),
        max_intensity=0.7,
        command_ttl_sec=0.1,
    )
    command = build_guidance_feedback_command(
        observation,
        spec=feedback_spec,
        sequence=0,
        produced_at=10.5,
        clock_domain_id="control.monotonic",
    )
    cue = command.cues[0]

    assert command.feedback_kind == "guidance_haptic_non_force"
    assert command.authoritative is False
    assert cue.translation_intensity == 0.7
    assert cue.rotation_intensity == 0.7
    np.testing.assert_allclose(cue.translation_direction_xyz, [1.0, 0.0, 0.0])
    adapter = NoIoGuidanceFeedbackAdapter()
    accepted = adapter.submit(
        command,
        observed_at=10.55,
        clock_domain_id="control.monotonic",
    )
    stale = adapter.submit(
        command,
        observed_at=10.6,
        clock_domain_id="control.monotonic",
    )
    assert accepted.accepted
    assert accepted.status == "validated_no_io"
    assert not stale.accepted
    assert stale.status == "stale"

    clear = build_guidance_feedback_command(
        observation,
        spec=feedback_spec,
        sequence=1,
        produced_at=11.1,
        clock_domain_id="control.monotonic",
        clear=True,
        reason="execution_gate_hold",
    )
    assert clear.status == "clear"
    assert clear.cues == ()
    assert NoIoGuidanceFeedbackAdapter().submit(
        clear,
        observed_at=11.15,
        clock_domain_id="control.monotonic",
    ).accepted

    context = AuthoritativeTraceContext(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="virtual-0",
        clock_domain_id="control.monotonic",
    )
    observation_fact = project_source_difference_observation(
        context,
        observation,
    )
    spec_fact = project_source_difference_spec(
        context,
        feedback_source_spec := _difference_spec(spec),
        occurred_at=observation.evaluated_at,
    )
    command_fact = project_guidance_feedback_command(context, command)
    receipt_fact = project_guidance_feedback_receipt(context, accepted)
    assert observation_fact.event_type == "source_difference_observation"
    assert spec_fact.event_type == "source_difference_spec"
    assert spec_fact.payload["semantic_content_hash"] == (
        feedback_source_spec.semantic_content_hash
    )
    assert command_fact.event_type == "guidance_feedback_command"
    assert receipt_fact.event_type == "guidance_feedback_receipt"
