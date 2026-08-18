from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from control_composition.composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionSpec,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    hand_action_spec_semantic_hash,
)
from dataset_contracts.completeness import (
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from robot_motion.action import (
    ActionResolutionError,
    ActionResolutionFailure,
    BIMANUAL_EE_POSE_V1,
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    ConservativeCompoundAuthorityProfile,
    ActionProposalDraft,
    HandActionSpec,
    HandQposSafetyLimits,
    IMMEDIATE_QREF_SIM_V1,
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    SharedCanonicalSafetyResult,
    apply_canonical_keyframe_safety,
    project_robot_action_v1_to_component_action,
)
from robot_motion.position_control import PositionKeyframeSafetyConfig
from robot_motion.runtime import CommandDispatchReceipt
from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.actuation import (
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    MujocoCompositionRuntimeConfig,
    MujocoCoupledPoseToQposResolver,
    MujocoDeployConfig,
    MujocoDeployRuntime,
)
from sim.mujoco.ik import RM65CanonicalIK
from sim.mujoco.scene import MujocoScene


pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.slow_sim]


CONTRACT_PATH = "sim/configs/rm65_bimanual_physics_v1.json"


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class Planner:
    def __init__(self, scene: MujocoScene) -> None:
        self.scene = scene

    def reset(self, _qpos_by_side):
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        identity = np.eye(4, dtype=float)
        return SimpleNamespace(
            valid=True,
            hold=False,
            status="tracking",
            q_ref_by_side={
                side: self.scene.get_arm_qpos(side) for side in pose_by_side
            },
            target_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            realized_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            modification_reasons=(),
            metadata={"planner": "formal-compound-authority-fixture"},
        )


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
        parameter_source="formal_compound_authority_fixture",
    )


def _contracts():
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    rules = (
        *(
            PoseComponentCompositionRule(
                component_id=f"{side}_arm",
                contributing_roles=("human", "policy"),
                weight_by_role={"human": 0.75, "policy": 0.25},
                max_translation_disagreement_m=0.10,
                max_rotation_disagreement_rad=0.50,
            )
            for side in ("left", "right")
        ),
        *(
            QposComponentCompositionRule(
                component_id=f"{side}_hand",
                contributing_roles=("human", "policy"),
                weight_by_role={"human": 0.5, "policy": 0.5},
                value_contract_hash=hand_action_spec_semantic_hash(
                    hand_specs[side]
                ),
            )
            for side in ("left", "right")
        ),
    )
    rule_set = ComponentCompositionRuleSet(
        rule_set_id="formal-bimanual-dmtac-authority-rules-v1",
        action_space=BIMANUAL_EE_POSE_V1,
        rules=rules,
    )
    composition_spec = CompositionSpec(
        spec_id="formal-bimanual-dmtac-authority-composition-v2",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=("human", "policy"),
        required_roles=("human", "policy"),
        component_rule_set_hash=rule_set.semantic_content_hash,
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="formal-bimanual-dmtac-authority-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash=composition_spec.semantic_content_hash,
        implementation_contract_hashes=(rule_set.semantic_content_hash,),
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        qpos_value_spec_by_component={
            f"{side}_hand": hand_specs[side]
            for side in ("left", "right")
        },
    )
    authority_profile = ConservativeCompoundAuthorityProfile(
        profile_id="mujoco-formal-bimanual-dmtac-authority-v1",
        producer_id="formal-composer",
        component_coverage=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        selection_groups=("arms", "left_hand", "right_hand"),
        composition_spec_hash=composition_spec.semantic_content_hash,
        component_rule_set_hash=rule_set.semantic_content_hash,
        formal_strategy_spec_hash=strategy_spec.semantic_content_hash,
    )
    return hand_specs, composition_spec, strategy, authority_profile


def _runtime(
    *,
    omit_authority_profile: bool = False,
    component_execution_handoff: bool = False,
    reference_activation_profile_id: str = SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    target_admission_margin_sec: float | None = None,
):
    pytest.importorskip("mujoco")
    clock = ManualClock()
    contract = SimulationContract.load(CONTRACT_PATH)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    scene = MujocoScene.from_contract(
        contract,
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
    )
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    hand_specs, composition_spec, strategy, authority_profile = _contracts()
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="formal-compound-authority-anchor-0",
        anchor_kind="measured_state",
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            side: np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for side in ("left", "right")
        },
        hand_qpos_by_side={
            side: np.zeros(6, dtype=np.float32)
            for side in ("left", "right")
        },
        hand_action_specs_by_side=hand_specs,
        frame_by_side={
            side: f"{side}_arm_base" for side in ("left", "right")
        },
        observed_at=clock.value,
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="formal-compound-authority-safety-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=False),
        hand_limits_by_side={
            side: HandQposSafetyLimits.uniform(
                hand_spec,
                lower=0.0,
                upper=1.0,
            )
            for side, hand_spec in hand_specs.items()
        },
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="formal-compound-authority-session",
            control_domain_id="formal-compound-authority-domain",
            embodiment_instance_id="formal-compound-authority-robot",
            composer_producer_id="formal-composer",
            enable_primary_policy_authority=False,
            enable_composer_authority=True,
            enable_replay_authority=False,
            reference_activation_profile_id=reference_activation_profile_id,
            pose_frame_contract_hash="formal-compound-arm-base-frame-v1",
            compound_hand_action_specs_by_side=hand_specs,
            compound_composer_authority_profile=(
                None if omit_authority_profile else authority_profile
            ),
            enable_component_execution_handoff=(
                component_execution_handoff
            ),
            target_admission_margin_sec=target_admission_margin_sec,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(
            RM65CanonicalIK(scene, Planner(scene))
        ),
        shared_safety=lambda action, input_id: apply_canonical_keyframe_safety(
            action,
            input_action_id=input_id,
            anchor=anchor,
            profile=safety_profile,
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(),
        composition_strategy=strategy,
        clock=clock,
    )
    return (
        clock,
        scene,
        runtime,
        hand_specs,
        composition_spec,
        authority_profile,
    )


def _input(
    runtime: MujocoDeployRuntime,
    *,
    role: str,
    hand_specs: dict[str, HandActionSpec],
    hand_close: tuple[float, float],
) -> CompositionInputDraft:
    hand_by_side = {}
    for side, close in zip(("left", "right"), hand_close, strict=True):
        values = np.zeros(6, dtype=np.float32)
        values[0] = close
        hand_by_side[side] = values
    return CompositionInputDraft(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        producer_id=f"{role}-source",
        producer_epoch=1,
        sequence=0,
        input_role=role,
        action=project_robot_action_v1_to_component_action(
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                arm_pose_by_side={
                    side: np.asarray(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        dtype=np.float32,
                    )
                    for side in ("left", "right")
                },
                hand_qpos_by_side=hand_by_side,
                active_effectors=(
                    "left_arm",
                    "right_arm",
                    "left_hand",
                    "right_hand",
                ),
                frame_by_effector={
                    f"{side}_arm": f"{side}_arm_base"
                    for side in ("left", "right")
                },
                controlled_frame_by_effector={
                    f"{side}_arm": "tool_tcp" for side in ("left", "right")
                },
                source=f"formal-compound-{role}",
                metadata={
                    "hand_action_specs_by_side": {
                        side: spec.to_metadata()
                        for side, spec in hand_specs.items()
                    }
                },
            ),
            action_schema=runtime.action_schema,
        ),
        produced_at=10.0,
        produced_clock_domain_id=runtime.config.clock_domain_id,
        effective_from=10.0,
        effective_until=10.2,
        effective_clock_domain_id=runtime.config.clock_domain_id,
    )


def _admit_ready_composition(
    runtime: MujocoDeployRuntime,
    *,
    composition_spec: CompositionSpec,
    hand_specs: dict[str, HandActionSpec],
    target_effective_at: float = 10.05,
):
    runtime.start()
    for role, hand_close in (
        ("human", (0.2, 0.8)),
        ("policy", (0.6, 0.4)),
    ):
        runtime.activate_policy_node_producer(f"{role}-source")
        receipt = runtime.submit_composition_input(
            _input(
                runtime,
                role=role,
                hand_specs=hand_specs,
                hand_close=hand_close,
            )
        )
        assert receipt.accepted
    submission = runtime.compose_and_submit(
        spec=composition_spec,
        target_effective_at=target_effective_at,
    )
    assert submission.proposal_admission is not None
    assert submission.proposal_admission.accepted
    return submission


def test_formal_compound_authority_dispatches_one_atomic_composer_proposal() -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        authority_profile,
    ) = _runtime()
    submission = _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )

    step = runtime.step(now=10.0)
    assert step.status == "dispatched"
    assert step.decision is not None
    assert set(step.decision.selected_by_group) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    assert len(set(step.decision.selected_by_group.values())) == 1
    assert next(iter(step.decision.selected_by_group.values())) == (
        submission.proposal_admission.proposal_id
    )
    assert set(runtime.core.pending_execution_segment_by_group) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    manifest = runtime.last_compound_dispatch_manifest
    assert manifest is not None
    assert set(manifest.group_binding_by_id) == set(
        authority_profile.selection_groups
    )
    profile_facts = [
        fact
        for fact in runtime.trace_facts
        if fact.event_type == "conservative_compound_authority_profile"
    ]
    assert len(profile_facts) == 1
    assert profile_facts[0].payload["semantic_content_hash"] == (
        authority_profile.semantic_content_hash
    )
    runtime.close()
    issues = evaluate_control_trace_lineage(
        runtime.trace_facts,
        profile=trace_requirement_profile("deployable_sim_v1"),
    )
    assert not {
        issue.code
        for issue in issues
        if issue.code.startswith("compound_authority_")
    }


def test_component_target_handoff_closes_and_admits_exact_sim_target() -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(component_execution_handoff=True)
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )

    step = runtime.step(now=10.0)

    assert step.status == "dispatched"
    assert step.target_execution_request is not None
    assert (
        step.target_submission_receipt is not None
        and step.target_submission_receipt.accepted_for_processing
    )
    assert (
        step.target_completion_receipt is not None
        and step.target_completion_receipt.status == "resolved"
    )
    assert step.target_dispatch_result is not None
    target = step.target_dispatch_result.target
    assert target.component_dependencies == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    assert target.target_execution_request_id == (
        step.target_execution_request.request_id
    )
    assert target.temporal_semantics == "point_target"
    assert target.scheduled_until is None
    assert not hasattr(target, "safe_action")
    assert step.target_dispatch_result.target_admission.accepted
    trace_count_before_status = len(runtime.trace_facts)
    pending_status = runtime.capture_target_execution_status_snapshot(now=10.0)
    assert pending_status is not None
    assert pending_status.request_id == step.target_execution_request.request_id
    assert pending_status.submission.status == "accepted_for_processing"
    assert pending_status.completion.status == "resolved"
    assert pending_status.admission.status == "accepted"
    assert pending_status.activation.status == "pending"
    assert len(runtime.trace_facts) == trace_count_before_status
    assert runtime.reference_receiver is not None
    assert runtime.reference_receiver.component_targets == (target,)
    assert set(runtime.core.pending_execution_segment_by_group) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    assert runtime.last_compound_dispatch_manifest is None
    for _ in range(18):
        runtime.rig.step_physics()
    assert set(runtime.core.active_execution_segment_by_group) == {
        "arms",
        "left_hand",
        "right_hand",
    }
    activations = runtime.reference_receiver.component_target_activations
    assert len(activations) == 1
    assert activations[0].target_id == target.target_id
    assert activations[0].activation_status == "activated"
    assert activations[0].effective_from < target.scheduled_for
    assert activations[0].effective_until == pytest.approx(
        target.scheduled_for
    )
    activated_status = runtime.capture_target_execution_status_snapshot(now=10.0)
    assert activated_status is not None
    assert activated_status.request_id == pending_status.request_id
    assert activated_status.activation.status == "activated"
    assert activated_status.activation.record_id == activations[0].activation_id
    runtime.close()
    assert {
        "target_execution_request",
        "target_execution_submission",
        "target_execution_completion",
        "component_execution_target",
        "component_execution_target_admission",
        "component_execution_target_activation",
    }.issubset({fact.event_type for fact in runtime.trace_facts})


def test_component_target_handoff_closes_rejected_private_safety_work() -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(component_execution_handoff=True)
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )
    runtime.core.shared_safety = lambda _action, input_id: (
        SharedCanonicalSafetyResult(
            input_action_id=input_id,
            safe_action=None,
            status="hold",
            reasons=("fixture_target_private_hold",),
        )
    )

    step = runtime.step(now=10.0)

    assert step.status == "safety_hold"
    assert (
        step.target_submission_receipt is not None
        and step.target_submission_receipt.accepted_for_processing
    )
    assert step.target_completion_receipt is not None
    assert step.target_completion_receipt.status == "rejected"
    assert step.target_completion_receipt.outcome_code == (
        "target_private_safety_hold"
    )
    assert step.target_dispatch_result is None
    assert runtime.reference_receiver is not None
    assert runtime.reference_receiver.component_targets == ()
    runtime.close()
    completions = [
        fact
        for fact in runtime.trace_facts
        if fact.event_type == "target_execution_completion"
    ]
    assert len(completions) == 1
    assert completions[0].payload["status"] == "rejected"


def test_component_target_short_support_margin_skips_only_current_cycle() -> None:
    (
        clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(
        component_execution_handoff=True,
        target_admission_margin_sec=0.20,
    )
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )

    step = runtime.step(now=10.0)

    assert step.status == "target_lifecycle_rejected_keep_existing"
    assert step.target_execution_request is not None
    assert step.target_submission_receipt is not None
    assert not step.target_submission_receipt.accepted_for_processing
    assert step.target_submission_receipt.rejection_code == (
        "insufficient_support_margin"
    )
    assert step.target_completion_receipt is None
    assert step.target_dispatch_result is None
    assert step.cancellation is None
    assert runtime.capture_execution_gate_snapshot(now=clock.value).status == (
        "ready"
    )
    lifecycle = runtime.capture_target_execution_status_snapshot(
        now=clock.value
    )
    assert lifecycle is not None
    assert lifecycle.submission.status == "rejected"
    assert lifecycle.completion.status == "not_applicable"
    assert lifecycle.admission.status == "not_applicable"
    assert runtime.reference_receiver is not None
    assert runtime.reference_receiver.component_targets == ()
    runtime.close()


def test_immediate_component_target_expiry_after_resolution_is_nonfatal() -> None:
    (
        clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(
        component_execution_handoff=True,
        reference_activation_profile_id=IMMEDIATE_QREF_SIM_V1,
    )
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
        target_effective_at=10.0,
    )
    base_resolver = runtime.core.resolver

    class DelayedResolver:
        def resolve(self, action, *, observation=None, context=None):
            command = base_resolver.resolve(
                action,
                observation=observation,
                context=context,
            )
            clock.value += 0.20
            return command

    runtime.core.resolver = DelayedResolver()

    step = runtime.step(now=10.0)

    assert step.status == "target_lifecycle_rejected_keep_existing"
    assert step.target_completion_receipt is not None
    assert step.target_completion_receipt.status == "resolved"
    assert step.target_dispatch_result is not None
    assert not step.target_dispatch_result.target_admission.accepted
    assert step.target_dispatch_result.target_admission.rejection_code == (
        "target_support_expired_before_admission"
    )
    assert step.cancellation is None
    assert runtime.capture_execution_gate_snapshot(now=clock.value).status == (
        "ready"
    )
    lifecycle = runtime.capture_target_execution_status_snapshot(
        now=clock.value
    )
    assert lifecycle is not None
    assert lifecycle.completion.status == "resolved"
    assert lifecycle.admission.status == "rejected"
    assert step.dispatch_receipts is not None
    assert not any(
        receipt.accepted for receipt in step.dispatch_receipts.values()
    )
    assert runtime._immediate_target_by_reference_sequence == {}
    runtime.close()


def test_component_target_pending_activation_is_terminal_on_cancel() -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(component_execution_handoff=True)
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )
    step = runtime.step(now=10.0)
    assert step.status == "dispatched"
    assert runtime.reference_receiver is not None

    runtime.reference_receiver.cancel_all(
        reason="fixture_hold",
        effective_at=10.01,
    )

    activations = runtime.reference_receiver.component_target_activations
    assert len(activations) == 1
    assert activations[0].activation_status == "rejected"
    assert activations[0].rejection_code == (
        "cancelled_before_activation:fixture_hold"
    )
    runtime.close()


def test_component_target_backend_schedule_failure_keeps_phase_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(component_execution_handoff=True)
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )

    def fail_schedule(*_args, **_kwargs):
        raise RuntimeError("fixture_schedule_failure")

    monkeypatch.setattr(
        runtime.rig,
        "send_scheduled_commands",
        fail_schedule,
    )
    step = runtime.step(now=10.0)

    assert step.status == "compound_downstream_failure_hold"
    assert (
        step.target_completion_receipt is not None
        and step.target_completion_receipt.status == "resolved"
    )
    assert step.target_dispatch_result is not None
    activation = step.target_dispatch_result.target_activation
    assert activation is not None
    assert activation.activation_status == "rejected"
    assert activation.rejection_code == (
        "scheduled_backend_exception:RuntimeError"
    )
    assert not any(
        receipt.accepted
        for receipt in (step.dispatch_receipts or {}).values()
    )
    runtime.close()


def test_compound_composer_authority_is_rejected_without_typed_profile() -> None:
    with pytest.raises(
        ValueError,
        match="requires an explicit ConservativeCompoundAuthorityProfile",
    ):
        _runtime(omit_authority_profile=True)


def test_compound_profile_rejects_direct_partial_composer_proposal() -> None:
    (
        clock,
        _scene,
        runtime,
        _hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime()
    runtime.start()
    partial = ActionProposalDraft(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        producer_kind="composer",
        producer_id=runtime.config.composer_producer_id,
        producer_epoch=1,
        sequence=0,
        action=project_robot_action_v1_to_component_action(
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                arm_pose_by_side={
                    side: np.asarray(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        dtype=np.float32,
                    )
                    for side in ("left", "right")
                },
                active_effectors=("left_arm", "right_arm"),
                frame_by_effector={
                    f"{side}_arm": f"{side}_arm_base"
                    for side in ("left", "right")
                },
                controlled_frame_by_effector={
                    f"{side}_arm": "tool_tcp" for side in ("left", "right")
                },
            ),
            action_schema=runtime.action_schema,
        ),
        produced_at=clock.value,
        produced_clock_domain_id=runtime.config.clock_domain_id,
        effective_from=10.05,
        effective_until=10.08,
        target_effective_at=10.05,
        effective_clock_domain_id=runtime.config.clock_domain_id,
        policy_output_id="composition-evaluation:partial-bypass",
        composition_input_ids=("composition-input:partial-bypass",),
        composition_input_hashes=("partial-bypass-hash",),
        composition_spec_hash=composition_spec.semantic_content_hash,
    )
    receipt = runtime.core.admit_proposal(
        partial,
        source_event_id="partial-bypass-source",
    )
    assert not receipt.accepted
    assert receipt.rejection_code == "compound_authority_scope_mismatch"
    assert "component_coverage_mismatch" in receipt.message
    assert len(runtime.core.proposals) == 0
    runtime.close()


def test_compound_downstream_partial_receipt_cancels_every_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime()
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )

    def reject_one(command, **_kwargs):
        return {
            effector_name: CommandDispatchReceipt(
                effector_name=effector_name,
                accepted=effector_name != "right_hand",
                operation="formal_compound_fixture",
                status=(
                    "accepted"
                    if effector_name != "right_hand"
                    else "rejected"
                ),
                timestamp=clock.value,
            )
            for effector_name in command.commands
        }

    monkeypatch.setattr(runtime.rig, "send_scheduled_commands", reject_one)
    step = runtime.step(now=10.0)
    assert step.status == "compound_downstream_failure_hold"
    assert step.cancellation is not None
    assert step.cancellation.effective_scope == "control_domain"
    assert not runtime.core.pending_execution_segment_by_group
    assert not runtime.core.active_execution_segment_by_group
    assert runtime.reference_receiver is not None
    assert runtime.reference_receiver.terminals
    assert runtime.reference_receiver.terminals[-1].status == "canceled"
    failures = [
        fact
        for fact in runtime.trace_facts
        if fact.event_type == "compound_authority_failure"
    ]
    assert len(failures) == 1
    assert failures[0].payload["stage"] == "downstream_admission_failure"
    runtime.close()


def test_compound_profile_is_default_off_and_missing_proposal_holds() -> None:
    (
        _clock,
        _scene,
        runtime,
        _hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime()
    runtime.start()
    evaluation = runtime.compose_and_submit(
        spec=composition_spec,
        target_effective_at=10.05,
    )
    assert evaluation.composition_evaluation.output_status != "ready"
    assert evaluation.proposal_admission is None

    step = runtime.step(now=10.0)
    assert step.decision is None
    assert step.cancellation is not None
    assert set(step.computation.status_by_group.values()) == {"hold"}
    assert not runtime.core.pending_execution_segment_by_group
    runtime.close()


def test_compound_proposal_expires_after_one_decision_period() -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime()
    submission = _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )
    proposal = submission.proposal_admission.proposal
    assert proposal is not None
    assert proposal.effective_until - proposal.effective_from == pytest.approx(
        1.0 / runtime.config.decision_hz
    )
    assert runtime.step(now=10.0).status == "dispatched"
    assert runtime.core.pending_execution_segment_by_group

    expired = runtime.step(now=10.04)
    assert expired.decision is None
    assert expired.cancellation is not None
    assert set(expired.computation.status_by_group.values()) == {"hold"}
    assert not runtime.core.pending_execution_segment_by_group
    runtime.close()


@pytest.mark.parametrize(
    ("failure_kind", "expected_status", "expected_stage"),
    (
        (
            "safety_exception",
            "compound_safety_exception_hold",
            "shared_safety_exception",
        ),
        (
            "safety_scope",
            "compound_safety_scope_hold",
            "safe_action_scope_mismatch",
        ),
        (
            "resolution",
            "compound_resolution_failure_hold",
            "resolution_failure",
        ),
    ),
)
def test_compound_pre_dispatch_failures_hold_without_partial_pending(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_status: str,
    expected_stage: str,
) -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime()
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )
    if failure_kind == "safety_exception":
        def fail_safety(_action, _input_id):
            raise RuntimeError("virtual safety failure")

        runtime.core.shared_safety = fail_safety
    elif failure_kind == "safety_scope":
        def shrink_safety(action, input_id):
            arms_only = RobotAction(
                action_space=action.action_space,
                arm_pose_by_side=action.arm_pose_by_side,
                active_effectors=("left_arm", "right_arm"),
                frame_by_effector={
                    key: value
                    for key, value in action.frame_by_effector.items()
                    if key.endswith("_arm")
                },
                controlled_frame_by_effector={
                    key: value
                    for key, value in action.controlled_frame_by_effector.items()
                    if key.endswith("_arm")
                },
            )
            return SharedCanonicalSafetyResult(
                input_action_id=input_id,
                safe_action=arms_only,
                status="safe",
            )

        runtime.core.shared_safety = shrink_safety
    else:
        def fail_resolution(_action):
            raise RuntimeError("virtual resolution failure")

        monkeypatch.setattr(runtime.core.resolver, "resolve", fail_resolution)

    step = runtime.step(now=10.0)
    assert step.status == expected_status
    assert step.cancellation is not None
    assert step.cancellation.effective_scope == "control_domain"
    assert not runtime.core.pending_execution_segment_by_group
    assert not runtime.core.active_execution_segment_by_group
    failures = [
        fact
        for fact in runtime.trace_facts
        if fact.event_type == "compound_authority_failure"
    ]
    assert len(failures) == 1
    assert failures[0].payload["stage"] == expected_stage
    runtime.close()


def test_typed_resolution_failure_is_traced_and_existing_hold_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _clock,
        _scene,
        runtime,
        hand_specs,
        composition_spec,
        _authority_profile,
    ) = _runtime(component_execution_handoff=True)
    _admit_ready_composition(
        runtime,
        composition_spec=composition_spec,
        hand_specs=hand_specs,
    )
    runtime.core._backend_hold_active = True

    failure = ActionResolutionFailure(
        reason_code="collision_margin_low",
        resolver_status="hold_last_safe",
        message="blended target is below the collision margin",
        diagnostics={"best_candidate_distance_m": 0.029},
    )

    def fail_resolution(_action):
        raise ActionResolutionError(failure)

    monkeypatch.setattr(runtime.core.resolver, "resolve", fail_resolution)
    cancellations_before = len(
        [
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "execution_cancellation"
        ]
    )

    step = runtime.step(now=10.0)

    assert step.status == "compound_resolution_failure_hold"
    assert step.resolution_failure is failure
    assert step.target_completion_receipt is not None
    assert step.target_completion_receipt.outcome_code == (
        "target_private_resolution_failure.collision_margin_low"
    )
    resolution_failures = [
        fact
        for fact in runtime.trace_facts
        if fact.event_type == "target_resolution_failure"
    ]
    assert len(resolution_failures) == 1
    assert resolution_failures[0].payload["reason_code"] == (
        "collision_margin_low"
    )
    assert resolution_failures[0].payload["diagnostics"] == {
        "best_candidate_distance_m": 0.029
    }
    assert resolution_failures[0].payload["completion_receipt_id"] == (
        step.target_completion_receipt.receipt_id
    )
    assert step.cancellation is None
    assert len(
        [
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "execution_cancellation"
        ]
    ) == cancellations_before
    runtime.close()
