from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.completeness import (
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentCatalogV2,
    ActionComponentSpec,
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    ComponentActionSchema,
    ControlComponentSpecV2,
    EndEffectorTarget,
    HandActionSpec,
    ProposalBuffer,
    RobotAction,
    build_robot_action_v1_runtime_schema,
    project_robot_action_v1_to_component_action,
)
from robot_motion.runtime import (
    EffectorCommand,
    ObservationChannel,
    ObservationSampleRef,
    ReceiverObservationCutBuilder,
    TeleopCommand,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from control_station.sources import (
    admit_human_command_emission,
    admit_human_observation_cut,
)
from robot_motion.runtime.dispatch import CommandDispatchReceipt
from teleop_runtime.action import (
    HumanCommandEmission,
    HumanLegacyCommandObservation,
    HumanProposalAdapter,
    HumanProposalAdapterConfig,
    TeleopCommandActionAdapterConfig,
)
from teleop_runtime.session import TeleopSession, TeleopSessionConfig
from teleop_runtime.control_trace.projection import project_human_proposal_adapter_result
from teleop_runtime.control_trace.projection import (
    make_runtime_trace_terminal,
    project_legacy_execution,
    project_shadow_startup,
    project_shadow_step,
)
from teleop_runtime.control_trace import ControlTraceRuntime, RuntimeTraceSession
from teleop_runtime.deploy.arbiter_shadow import (
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)
from teleop_runtime.action.human_shadow import HumanProposalShadowRuntime


def _pose(x: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = float(x)
    return value


def _arm_command(
    x: float,
    *,
    frame: str | None = "left_arm_base",
    metadata: dict | None = None,
) -> TeleopCommand:
    spec = ArmActionSpec.tool_tcp("left", reference_frame="left_arm_base")
    command = TeleopCommand(timestamp=4.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="cartesian_pose",
            value=_pose(x),
            frame_name=frame,
            metadata={
                "arm_action_spec": spec.to_metadata(),
                **dict(metadata or {}),
            },
        )
    )
    return command


def _emission(*commands: tuple[str, TeleopCommand]) -> HumanCommandEmission:
    merged = TeleopCommand(timestamp=4.0)
    for _, command in commands:
        for name, effector in command.commands.items():
            merged.commands[name] = effector.copy()
        merged.metadata.update(command.metadata)
    return HumanCommandEmission.capture(
        emission_sequence=7,
        emitted_at=10.0,
        contributions=commands,
        merged_command=merged,
    )


def _adapter(
    *,
    action_space: str = BIMANUAL_EE_POSE_V1,
    effector_name: str = "left_arm",
    component_id: str = "left_arm_pose",
    action_adapter: TeleopCommandActionAdapterConfig | None = None,
    action_schema: ComponentActionSchema | None = None,
) -> HumanProposalAdapter:
    catalog = ActionComponentCatalog(
        catalog_id="human-test-v1",
        action_space=action_space,
        components=(
            ActionComponentSpec(
                component_id=component_id,
                selection_group_id=effector_name,
                effector_name=effector_name,
            ),
        ),
    )
    resolved_adapter = (
        action_adapter
        if action_adapter is not None
        else TeleopCommandActionAdapterConfig(
            include_command_metadata=False,
            source="human:vive",
        )
    )
    hand_specs = dict(resolved_adapter.hand_specs or {})
    for component in catalog.components:
        side, kind = component.effector_name.split("_", 1)
        if kind == "hand" and side not in hand_specs:
            hand_specs[side] = HandActionSpec.inspire_tac(side)
    action_schema = action_schema or build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
        arm_action_specs_by_side=dict(resolved_adapter.arm_specs or {}),
        hand_action_specs_by_side=hand_specs,
    )
    return HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id="runtime-1",
            control_domain_id="robot-1",
            embodiment_instance_id="rm65-1",
            producer_id="human-vive",
            producer_epoch=0,
            produced_clock_domain_id="control-monotonic",
            effective_clock_domain_id="control-monotonic",
            component_catalog=catalog,
            action_schema=action_schema,
            action_adapter=resolved_adapter,
        )
    )


def _native_mode_action_schema() -> ComponentActionSchema:
    template = RobotAction(
        action_space="dexteleop_policy_io_v2.2",
        end_effector_targets_by_side={
            "right": EndEffectorTarget(
                value_type="discrete",
                values=np.asarray([0], dtype=np.int64),
                adapter="two_state_gripper_v1",
                names=("mode",),
            )
        },
        active_effectors=("right_hand",),
        source="native-schema-template",
    )
    local = project_robot_action_v1_to_component_action(template).action_schema
    payload_schema = local.payload_schema_by_component["right_hand"]
    catalog = ActionComponentCatalogV2(
        catalog_id="native-mode-human-test-v1",
        catalog_version="1",
        components=(
            ControlComponentSpecV2(
                component_id="right_gripper_mode",
                controlled_entity_id="right",
                command_channel_id="right_hand",
                selection_group_id="right_hand",
                resource_claims=("right_hand",),
                accepted_payload_schema_hashes=(
                    payload_schema.semantic_content_hash,
                ),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="native-mode-human-test-v1",
        schema_version="1",
        component_catalog=catalog,
        payload_schema_by_component={
            "right_gripper_mode": payload_schema,
        },
        semantic_parameters=local.semantic_parameters,
    )


class _AdmissionCore:
    def __init__(self, producer_id: str, producer_epoch: int = 0) -> None:
        self.clock = lambda: 10.0
        self.coordinator = ControlAdmissionCoordinator(
            "robot-1",
            clock_domain=ClockDomainDescriptor(
                clock_domain_id="control-monotonic",
                runtime_session_id="runtime-1",
                boot_id="test-boot",
                reset_epoch=0,
            ),
            clock=self.clock,
        )
        self.epochs = ProducerEpochRegistry(
            runtime_session_id="runtime-1",
            coordinator=self.coordinator,
        )
        self.epochs.activate(
            producer_id=producer_id,
            producer_epoch=producer_epoch,
            produced_at=9.0,
            produced_clock_domain_id="control-monotonic",
            reason="test",
        )
        self.proposals = ProposalBuffer(
            runtime_session_id="runtime-1",
            coordinator=self.coordinator,
            producer_epochs=self.epochs,
            capacity=8,
        )
        self.external_facts = []
        self.observation_cuts = []

    def record_external_facts(self, facts):
        self.external_facts.extend(facts)

    def record_observation_cut(self, record):
        self.observation_cuts.append(record)
        return record

    def admit_proposal(self, draft, *, source_event_id):
        assert source_event_id
        return self.proposals.admit(draft)


def test_human_adapter_preserves_pre_merge_winner_and_absolute_pose() -> None:
    emission = _emission(
        ("first", _arm_command(0.1)),
        ("second", _arm_command(0.2)),
    )

    result = _adapter().adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert result.source_validation.accepted
    assert result.source_adapter_receipt.authoritative_eligible
    assert result.source_adapter_receipt.lossiness == "lossless"
    assert result.proposal_draft is not None
    assert result.proposal_draft.validity == "valid"
    assert "source_observation_cut_id" not in result.proposal_draft.audit_metadata
    assert result.proposal_draft.action.component_coverage == (
        "left_arm_pose",
    )
    assert result.proposal_draft.selection_groups == ("left_arm",)
    np.testing.assert_allclose(result.robot_action.arm_pose_by_side["left"][:3], [0.2, 0, 0])
    mapping = result.source_adapter_receipt.component_mappings[0]
    assert mapping.retargeter_name == "second"
    assert mapping.contribution_index == 1
    assert mapping.overwritten_contribution_indices == (0,)
    assert result.source_adapter_receipt.source_emission_semantic_hash == (
        emission.semantic_content_hash
    )

    emission_fact, receipt_fact, validation_fact = project_human_proposal_adapter_result(
        emission,
        result,
        runtime_session_id="runtime-1",
        control_domain_id="robot-1",
        embodiment_instance_id="rm65-1",
        occurred_at=10.0,
        clock_domain_id="control-monotonic",
    )
    assert emission_fact.event_type == "human_command_emission"
    assert emission_fact.payload["source_emission_semantic_hash"] == (
        emission.semantic_content_hash
    )
    assert receipt_fact.event_type == "source_adapter_receipt"
    assert receipt_fact.payload["component_mappings"][0]["retargeter_name"] == "second"
    assert receipt_fact.payload["semantic_content_hash"] == (
        result.source_adapter_receipt.semantic_content_hash
    )
    assert validation_fact.event_type == "source_validation"
    assert validation_fact.payload["source_event_id"] == (
        result.source_adapter_receipt.source_emission_id
    )
    assert evaluate_control_trace_lineage(
        (emission_fact, receipt_fact, validation_fact),
        profile=trace_requirement_profile("plain_dry_run_v1"),
    ) == ()


def test_human_adapter_marks_backend_control_metadata_shadow_only() -> None:
    emission = _emission(("arm", _arm_command(0.2, metadata={"backend_mode": "position"})))

    result = _adapter().adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert result.source_validation.accepted
    assert not result.source_adapter_receipt.authoritative_eligible
    assert result.source_adapter_receipt.lossiness == "lossy_but_explicit"
    assert result.proposal_draft is not None
    assert result.proposal_draft.validity == "shadow_only"
    assert result.source_adapter_receipt.component_mappings[0].reasons == (
        "noncanonical_control_metadata:backend_mode",
    )


def test_human_adapter_rejects_missing_pose_frame_without_guessing() -> None:
    emission = _emission(("arm", _arm_command(0.2, frame=None)))

    result = _adapter().adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert not result.source_validation.accepted
    assert result.robot_action is None
    assert result.proposal_draft is None
    assert result.source_adapter_receipt.lossiness == "unsupported"
    assert "arm_pose_requires_explicit_reference_frame" in result.source_validation.reasons


def test_human_adapter_rejects_hand_q6_without_explicit_hand_spec() -> None:
    command = TeleopCommand(timestamp=4.0)
    command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=np.arange(6, dtype=np.float32),
        )
    )
    emission = _emission(("hand", command))
    adapter = _adapter(
        action_space="bimanual_qpos_v1",
        effector_name="right_hand",
        component_id="right_hand_qpos",
    )

    result = adapter.adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert not result.source_validation.accepted
    assert "hand_joint_vector_requires_explicit_hand_action_spec" in (
        result.source_validation.reasons
    )


def test_human_adapter_accepts_explicit_hand_spec() -> None:
    command = TeleopCommand(timestamp=4.0)
    command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=np.arange(6, dtype=np.float32),
            metadata={"hand_action_spec": HandActionSpec.inspire_tac("right").to_metadata()},
        )
    )
    emission = _emission(("hand", command))
    result = _adapter(
        action_space="bimanual_qpos_v1",
        effector_name="right_hand",
        component_id="right_hand_qpos",
    ).adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert result.source_validation.accepted
    assert result.source_adapter_receipt.authoritative_eligible
    np.testing.assert_allclose(result.robot_action.hand_qpos_by_side["right"], np.arange(6))


def test_hand_only_emission_uses_bound_pose_hand_catalog_action_space() -> None:
    command = TeleopCommand(timestamp=4.0)
    command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=np.arange(6, dtype=np.float32),
        )
    )
    emission = _emission(("hand", command))
    result = _adapter(
        action_space=BIMANUAL_EE_POSE_V1,
        effector_name="right_hand",
        component_id="right_hand_qpos",
        action_adapter=TeleopCommandActionAdapterConfig(
            hand_specs={"right": HandActionSpec.inspire_tac("right")},
            include_command_metadata=False,
            source="human:virtual-hand",
        ),
    ).adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert result.source_validation.accepted
    assert result.robot_action.action_space == BIMANUAL_EE_POSE_V1
    assert result.robot_action.active_effectors == ("right_hand",)
    assert result.proposal_draft is not None
    assert result.proposal_draft.action.component_coverage == (
        "right_hand_qpos",
    )


def test_pose_and_hand_share_one_catalog_bound_proposal() -> None:
    hand = TeleopCommand(timestamp=4.0)
    hand.add_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=np.linspace(0.0, 0.5, 6, dtype=np.float32),
        )
    )
    catalog = ActionComponentCatalog(
        catalog_id="pose-hand-human-test-v1",
        action_space=BIMANUAL_EE_POSE_V1,
        components=(
            ActionComponentSpec(
                component_id="arms",
                selection_group_id="arms",
                effector_name="left_arm",
                coupled=False,
            ),
            ActionComponentSpec(
                component_id="left_hand_qpos",
                selection_group_id="left_hand",
                effector_name="left_hand",
            ),
        ),
    )
    adapter = HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id="runtime-1",
            control_domain_id="robot-1",
            embodiment_instance_id="generic-compatible-v1",
            producer_id="human-virtual-device",
            producer_epoch=0,
            produced_clock_domain_id="control-monotonic",
            effective_clock_domain_id="control-monotonic",
            component_catalog=catalog,
            action_schema=build_robot_action_v1_runtime_schema(
                component_catalog=catalog,
                arm_action_specs_by_side={
                    "left": ArmActionSpec.tool_tcp(
                        "left",
                        reference_frame="left_arm_base",
                    )
                },
                hand_action_specs_by_side={
                    "left": HandActionSpec.inspire_tac("left")
                },
            ),
            action_adapter=TeleopCommandActionAdapterConfig(
                arm_specs={
                    "left": ArmActionSpec.tool_tcp(
                        "left",
                        reference_frame="left_arm_base",
                    )
                },
                hand_specs={"left": HandActionSpec.inspire_tac("left")},
                include_command_metadata=False,
                source="human:virtual-device",
            ),
        )
    )

    emission = _emission(("arm", _arm_command(0.2)), ("hand", hand))
    core = _AdmissionCore("human-virtual-device")
    result, admission = admit_human_command_emission(
        core=core,
        adapter=adapter,
        emission=emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        occurred_at=10.0,
        clock_domain_id="control-monotonic",
    )

    assert result.source_validation.accepted
    assert result.robot_action.action_space == BIMANUAL_EE_POSE_V1
    assert result.proposal_draft is not None
    assert result.proposal_draft.action.component_coverage == (
        "arms",
        "left_hand_qpos",
    )
    assert result.proposal_draft.selection_groups == ("arms", "left_hand")
    assert admission is not None and admission.accepted
    assert len(core.proposals) == 1
    assert len(core.external_facts) == 3

    _, duplicate = admit_human_command_emission(
        core=core,
        adapter=adapter,
        emission=emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        occurred_at=10.0,
        clock_domain_id="control-monotonic",
        adaptation=result,
    )
    assert duplicate is not None and not duplicate.accepted
    assert duplicate.rejection_code == "duplicate"
    assert len(core.proposals) == 1


def test_human_proposal_binds_receiver_admitted_observation_cut() -> None:
    command = _arm_command(0.2)
    source_sample = ObservationSampleRef(
        stream_id="vive.left_wrist_pose",
        sample_id="vive-pose-7",
        sequence=7,
        measured_at=4.0,
        measured_clock_domain_id="vive.device_time",
        available_at=8.0,
        available_clock_domain_id="teleop_session.monotonic",
        payload_object_id="vive-pose-payload-7",
    )
    emission = HumanCommandEmission.capture(
        emission_sequence=7,
        emitted_at=10.0,
        contributions=(("vive", command),),
        merged_command=command,
        observation_sample_refs={source_sample.stream_id: source_sample},
    )
    adapter = _adapter()
    core = _AdmissionCore("human-vive")
    builder = ReceiverObservationCutBuilder(
        producer_id="human-vive",
        producer_epoch=0,
        observation_contract_hash="vive-input-contract-v1",
        io_layout_hash="human-pose-layout-v1",
    )
    cut = admit_human_observation_cut(
        core=core,
        builder=builder,
        emissions=(emission,),
        payload_handles_by_stream={
            source_sample.stream_id: np.asarray(_pose(0.2), dtype=np.float64)
        },
        built_at=10.0,
    )

    _result, admission = admit_human_command_emission(
        core=core,
        adapter=adapter,
        emission=emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        occurred_at=10.0,
        clock_domain_id="control-monotonic",
        observation_cut=cut.record,
    )

    assert len(core.observation_cuts) == 1
    assert admission is not None and admission.accepted
    assert admission.proposal is not None
    assert admission.proposal.observation_cut_id == (
        cut.record.observation_cut_id
    )
    assert admission.proposal.audit_metadata["source_observation_cut_id"] == (
        cut.record.observation_cut_id
    )
    admitted_sample = cut.record.selected_sample_by_stream[
        source_sample.stream_id
    ]
    assert admitted_sample.sample_id == source_sample.sample_id
    assert admitted_sample.available_clock_domain_id == "control-monotonic"
    assert len(core.external_facts) == 3


def test_catalog_binding_rejects_arm_pose_with_qpos_action_space() -> None:
    result = _adapter(action_space="bimanual_qpos_v1").adapt(
        _emission(("arm", _arm_command(0.2))),
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert not result.source_validation.accepted
    assert result.robot_action is None
    assert any(
        "Arm pose commands are incompatible with output_action_space" in reason
        for reason in result.source_validation.reasons
    )


def test_human_adapter_accepts_explicit_native_end_effector_target() -> None:
    command = TeleopCommand(timestamp=4.0)
    command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="end_effector_discrete",
            value=np.asarray([1], dtype=np.int64),
            metadata={
                "end_effector_adapter": "two_state_gripper_v1",
                "end_effector_target_names": ("mode",),
            },
        )
    )
    emission = _emission(("gripper", command))
    result = _adapter(
        action_space="dexteleop_policy_io_v2.2",
        effector_name="right_hand",
        component_id="right_gripper_mode",
        action_schema=_native_mode_action_schema(),
    ).adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )

    assert result.source_validation.accepted
    target = result.robot_action.end_effector_targets_by_side["right"]
    assert target.adapter == "two_state_gripper_v1"
    np.testing.assert_array_equal(target.values, [1])


def test_human_draft_uses_shared_passive_arbiter_path_without_dispatch() -> None:
    emission = _emission(("arm", _arm_command(0.25)))
    catalog = ActionComponentCatalog(
        catalog_id="human-shadow-components-v1",
        action_space=BIMANUAL_EE_POSE_V1,
        components=(
            ActionComponentSpec(
                component_id="left_arm",
                selection_group_id="left_arm",
                effector_name="left_arm",
            ),
        ),
    )
    adapter = HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id="human-shadow-runtime",
            control_domain_id="robot-1",
            embodiment_instance_id="rm65-1",
            producer_id="human-vive",
            producer_epoch=1,
            produced_clock_domain_id="local_monotonic",
            effective_clock_domain_id="local_monotonic",
            component_catalog=catalog,
            action_schema=build_robot_action_v1_runtime_schema(
                component_catalog=catalog,
            ),
            action_adapter=TeleopCommandActionAdapterConfig(
                include_command_metadata=False,
                source="human:vive",
            ),
        )
    )
    adapted = adapter.adapt(
        emission,
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
    )
    assert adapted.proposal_draft is not None
    shadow = PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id="human-shadow-runtime",
            control_domain_id="robot-1",
            embodiment_instance_id="rm65-1",
            producer_id="human-vive",
            active_effectors=("left_arm",),
            action_space=BIMANUAL_EE_POSE_V1,
            decision_hz=60.0,
            component_catalog=catalog,
            action_schema=build_robot_action_v1_runtime_schema(
                component_catalog=catalog,
            ),
            producer_epoch=1,
        ),
        clock=lambda: 10.0,
    )

    step = shadow.evaluate_proposal_draft(
        adapted.proposal_draft,
        source_event_id=adapted.source_adapter_receipt.source_emission_id,
    )

    assert step.status == "evaluated", (
        step.to_debug_payload(),
        None if step.admission_receipt is None else step.admission_receipt.message,
    )
    assert step.admission_receipt is not None and step.admission_receipt.accepted
    assert step.evaluation is not None
    assert step.evaluation.non_authoritative
    assert step.evaluation.would_select_by_group == {
        "left_arm": step.admission_receipt.proposal_id
    }
    np.testing.assert_allclose(
        step.evaluation.would_select_action.value("left_arm").values[:3],
        [0.25, 0.0, 0.0],
    )
    legacy_observation = HumanLegacyCommandObservation.capture(
        emission=emission,
        observed_at=10.0,
        outcome="dispatched",
        command_dispatched=True,
        run_state="RUNNING",
        command=emission.to_merged_command(),
    )
    legacy_action = adapter.adapt_legacy_observation(legacy_observation)
    assert legacy_action.source_validation.accepted
    legacy_event = shadow.observe_legacy_action(
        source_event_id=adapted.source_adapter_receipt.source_emission_id,
        shadow_step=step,
        observed_action=legacy_action.robot_action,
        command_present=bool(legacy_observation.final_commands),
        execution_accepted=legacy_observation.command_dispatched,
        execution_mode=legacy_observation.outcome,
        dispatch_attempted=legacy_observation.command_dispatched,
        dispatch_completed=legacy_observation.command_dispatched,
        observed_at=legacy_observation.observed_at,
    )
    assert legacy_event.component_parity is not None
    assert legacy_event.component_parity.matched

    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="human-shadow-trace",
            runtime_session_id="human-shadow-runtime",
            control_domain_id="robot-1",
            profile_id="human_shadow_dry_run_v1",
            started_at=10.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=64,
    )
    facts = (
        *project_shadow_startup(shadow),
        *project_human_proposal_adapter_result(
            emission,
            adapted,
            runtime_session_id="human-shadow-runtime",
            control_domain_id="robot-1",
            embodiment_instance_id="rm65-1",
            occurred_at=10.0,
            clock_domain_id="local_monotonic",
        ),
        *project_shadow_step(shadow, step),
        project_legacy_execution(shadow, legacy_event),
        make_runtime_trace_terminal(
            shadow,
            terminal_status="completed",
            reason="human_shadow_fixture_complete",
            occurred_at=10.0,
        ),
    )
    for fact in facts:
        assert trace.issue(fact).accepted
    report = trace.completeness()
    assert report.in_memory_complete
    assert report.lineage_complete
    assert report.lineage_issues == ()
    assert {
        record.fact.event_type for record in trace.records()
    } >= {
        "human_command_emission",
        "source_adapter_receipt",
        "action_proposal",
        "arbitration_evaluation",
    }


def test_human_emission_is_detached_from_mutable_command() -> None:
    command = _arm_command(0.1)
    emission = _emission(("arm", command))
    command.commands["left_arm"].value[0, 3] = 9.0

    snapshot = emission.merged_commands[0]
    assert float(snapshot.value[0, 3]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        snapshot.value[0, 3] = 8.0


class _Source:
    name = "human-test-source"

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def expected_channel_names(self) -> tuple[str, ...]:
        return ("left_wrist_pose",)

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {
            "left_wrist_pose": ObservationChannel(
                channel_name="left_wrist_pose",
                capability="pose6d",
                side="left",
                value=_pose(0.1),
                frame_name="desk",
                timestamp=1.0,
                source_name=self.name,
            )
        }


class _Retargeter:
    name = "human-test-retargeter"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def retarget(self, observation, calibration) -> TeleopCommand:
        del observation, calibration
        self.events.append("retarget")
        return _arm_command(0.3)


class _Rig:
    name = "human-test-rig"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sent: list[TeleopCommand] = []
        self.return_receipts = None

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def send_commands(self, command: TeleopCommand, phase_recorder=None):
        del phase_recorder
        self.events.append("dispatch")
        self.sent.append(command)
        return self.return_receipts


def _armed_session(
    *,
    events: list[str],
    observer,
    filter_observer=None,
    execution_observer=None,
) -> tuple[TeleopSession, _Rig]:
    rig = _Rig(events)

    def command_filter(command, observation, now):
        del observation, now
        events.append("filter")
        return command

    def command_guard(command, observation, now):
        del command, observation, now
        events.append("guard")
        return None

    session = TeleopSession(
        TeleopSessionConfig(
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        sources=[_Source()],
        retargeters=[_Retargeter(events)],
        rig=rig,
        command_filter=command_filter,
        command_guard=command_guard,
        human_command_observer=observer,
        human_command_filter_observer=filter_observer,
        human_execution_observer=execution_observer,
    )
    session.start()
    session.set_calibration_ready()
    session.mark_debug_passed()
    session.arm_hardware()
    return session, rig


def test_session_human_observer_is_passive_and_runs_before_filter_guard_dispatch() -> None:
    events: list[str] = []
    emissions: list[HumanCommandEmission] = []

    def observer(emission: HumanCommandEmission) -> None:
        events.append("observer")
        emissions.append(emission)

    session, rig = _armed_session(events=events, observer=observer)
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()

    assert events == ["retarget", "observer", "filter", "guard", "dispatch"]
    assert result.command_dispatched
    assert len(rig.sent) == 1
    assert len(emissions) == 1
    assert emissions[0].contributions[0].retargeter_name == "human-test-retargeter"
    assert session.human_command_observer_error_count == 0


def test_session_filtered_command_observer_is_correlated_and_pre_guard() -> None:
    events: list[str] = []
    observed: list[tuple[int, float]] = []

    def source_observer(_emission: HumanCommandEmission) -> None:
        events.append("observer")

    def filter_observer(emission, command, observed_at) -> None:
        events.append("filter_observer")
        observed.append(
            (emission.emission_sequence, float(command.commands["left_arm"].value[0, 3]))
        )
        assert observed_at > 0.0

    session, rig = _armed_session(
        events=events,
        observer=source_observer,
        filter_observer=filter_observer,
    )
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()

    assert result.command_dispatched
    assert len(rig.sent) == 1
    assert observed == [(0, pytest.approx(0.3))]
    assert events == [
        "retarget",
        "observer",
        "filter",
        "filter_observer",
        "guard",
        "dispatch",
    ]


def test_session_filtered_command_observer_failure_is_passive() -> None:
    events: list[str] = []

    def filter_observer(_emission, _command, _observed_at) -> None:
        events.append("filter_observer")
        raise ValueError("observer failure")

    session, rig = _armed_session(
        events=events,
        observer=lambda _emission: events.append("observer"),
        filter_observer=filter_observer,
    )
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()

    assert result.command_dispatched
    assert len(rig.sent) == 1
    assert session.human_command_filter_observer_error_count == 1
    assert isinstance(session.last_human_command_filter_observer_error, ValueError)


def test_session_human_observer_failure_cannot_block_or_mutate_dispatch() -> None:
    events: list[str] = []

    def observer(emission: HumanCommandEmission) -> None:
        events.append("observer")
        emission.merged_commands[0].value[0, 3] = 99.0

    session, rig = _armed_session(events=events, observer=observer)
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()

    assert result.command_dispatched
    assert len(rig.sent) == 1
    assert float(rig.sent[0].commands["left_arm"].value[0, 3]) == pytest.approx(0.3)
    assert session.human_command_observer_error_count == 1
    assert isinstance(session.last_human_command_observer_error, ValueError)
    assert events == ["retarget", "observer", "filter", "guard", "dispatch"]


def test_session_post_dispatch_observer_reports_legacy_owner_result() -> None:
    events: list[str] = []
    observations: list[HumanLegacyCommandObservation] = []

    def source_observer(_emission: HumanCommandEmission) -> None:
        events.append("observer")

    def execution_observer(observation: HumanLegacyCommandObservation) -> None:
        events.append("execution_observer")
        observations.append(observation)

    session, rig = _armed_session(
        events=events,
        observer=source_observer,
        execution_observer=execution_observer,
    )
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()

    assert result.command_dispatched
    assert len(rig.sent) == 1
    assert events == [
        "retarget",
        "observer",
        "filter",
        "guard",
        "dispatch",
        "execution_observer",
    ]
    assert len(observations) == 1
    observed = observations[0]
    assert observed.emission_sequence == 0
    assert observed.outcome == "dispatched"
    assert observed.command_dispatched
    assert observed.run_state == "RUNNING"
    assert observed.guard_fault_code is None
    assert float(observed.final_commands[0].value[0, 3]) == pytest.approx(0.3)


def test_session_post_dispatch_observer_preserves_typed_backend_receipt() -> None:
    events: list[str] = []
    observations: list[HumanLegacyCommandObservation] = []
    session, rig = _armed_session(
        events=events,
        observer=lambda _emission: None,
        execution_observer=observations.append,
    )
    rig.return_receipts = {
        "left_arm": CommandDispatchReceipt(
            effector_name="left_arm",
            accepted=True,
            operation="queue_background_joint_target",
            status="queued",
            timestamp=2.0,
            metadata={"position_transport_target_sequence": 7},
        )
    }
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()

    assert result.command_dispatched
    receipt = observations[0].dispatch_receipts[0]
    assert receipt.receipt_kind == "typed_command_dispatch_receipt"
    assert receipt.accepted is True
    assert receipt.operation == "queue_background_joint_target"
    assert receipt.metadata["position_transport_target_sequence"] == 7


def test_session_hooks_feed_default_off_human_shadow_runtime_without_dispatch_ownership(
    monkeypatch,
) -> None:
    monkeypatch.setattr("teleop_runtime.session.time.monotonic", lambda: 1.0)
    events: list[str] = []
    catalog = ActionComponentCatalog(
        catalog_id="session-human-shadow-v1",
        action_space=BIMANUAL_EE_POSE_V1,
        components=(
            ActionComponentSpec(
                component_id="left_arm",
                selection_group_id="left_arm",
                effector_name="left_arm",
            ),
        ),
    )
    adapter = HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id="session-human-runtime",
            control_domain_id="robot-1",
            embodiment_instance_id="rm65-1",
            producer_id="human-vive",
            producer_epoch=1,
            produced_clock_domain_id="local_monotonic",
            effective_clock_domain_id="local_monotonic",
            component_catalog=catalog,
            action_schema=build_robot_action_v1_runtime_schema(
                component_catalog=catalog,
            ),
            action_adapter=TeleopCommandActionAdapterConfig(
                include_command_metadata=False,
                source="human:vive",
            ),
        )
    )
    shadow = PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id="session-human-runtime",
            control_domain_id="robot-1",
            embodiment_instance_id="rm65-1",
            producer_id="human-vive",
            active_effectors=("left_arm",),
            action_space=BIMANUAL_EE_POSE_V1,
            decision_hz=60.0,
            component_catalog=catalog,
            action_schema=build_robot_action_v1_runtime_schema(
                component_catalog=catalog,
            ),
            producer_epoch=1,
        ),
        clock=lambda: 1.0,
    )
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="session-human-trace",
            runtime_session_id="session-human-runtime",
            control_domain_id="robot-1",
            profile_id="human_shadow_dry_run_v1",
            started_at=1.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=64,
    )
    runtime = HumanProposalShadowRuntime(
        adapter=adapter,
        shadow=shadow,
        trace_runtime=trace,
    )
    session, rig = _armed_session(
        events=events,
        observer=runtime.observe_emission,
        execution_observer=runtime.observe_legacy_execution,
    )
    try:
        result = session.run_step(now=1.0)
    finally:
        session.close()
        runtime.close(reason="fixture_complete")

    assert result.command_dispatched
    assert len(rig.sent) == 1
    assert runtime.last_source_observation is not None
    assert runtime.last_source_observation.shadow_step.evaluation is not None
    assert runtime.last_legacy_observation is not None
    assert runtime.last_legacy_observation.legacy_event.component_parity is not None
    assert runtime.last_legacy_observation.legacy_event.component_parity.matched
    assert runtime.trace_rejection_count == 0
    report = trace.completeness()
    assert report.in_memory_complete
    assert report.lineage_complete
    assert report.lineage_issues == ()
