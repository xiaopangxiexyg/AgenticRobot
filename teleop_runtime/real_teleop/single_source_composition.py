"""Identity composition for one already-selected exclusive policy source."""

from __future__ import annotations

from dataclasses import dataclass

from control_composition.composition import (
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionSpec,
    DeterministicComposer,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ComponentActionSchema,
    RobotAction,
    materialize_robot_action_v1,
    project_robot_action_v1_to_component_action,
)


EXCLUSIVE_SINGLE_SOURCE_INPUT_HASH_DOMAIN_V1 = (
    "dexteleop.teleop_runtime.exclusive_single_source_input.v1"
)
EXCLUSIVE_SINGLE_SOURCE_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.teleop_runtime.exclusive_single_source_snapshot.v1"
)


@dataclass(frozen=True)
class ExclusiveSingleSourceCompositionResult:
    action: RobotAction
    composition_input_id: str
    composition_input_hash: str
    composition_evaluation_id: str
    composition_evaluation_hash: str
    composition_spec_hash: str
    composed_candidate_hash: str


class ExclusiveSingleSourceComposer:
    """Run a one-role ``component_union`` without owning authority or safety."""

    def __init__(self) -> None:
        self._action_schema: ComponentActionSchema | None = None
        self._composer = DeterministicComposer()
        self._sequence = 0

    @property
    def action_schema(self) -> ComponentActionSchema | None:
        return self._action_schema

    def reset_activation(self) -> None:
        self._action_schema = None
        self._sequence = 0

    def compose(
        self,
        action: RobotAction,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        producer_id: str,
        producer_epoch: int,
        policy_output_id: str,
        produced_at: float,
        produced_clock_domain_id: str,
        available_at: float,
        available_clock_domain_id: str,
        effective_from: float,
        effective_until: float,
        effective_clock_domain_id: str,
        target_effective_at: float,
        context_ids: tuple[str, ...] = (),
    ) -> ExclusiveSingleSourceCompositionResult:
        if not isinstance(action, RobotAction):
            raise TypeError("action must be RobotAction.")
        if self._action_schema is None:
            self._action_schema = project_robot_action_v1_to_component_action(
                action
            ).action_schema
        component_action = project_robot_action_v1_to_component_action(
            action,
            action_schema=self._action_schema,
        )
        spec = CompositionSpec(
            spec_id="exclusive-hitl-single-selected-source-v1",
            algebra="component_union",
            input_roles=("selected",),
            required_roles=("selected",),
            action_schema_hash=self._action_schema.semantic_content_hash,
        )
        sequence = self._sequence
        self._sequence += 1
        input_payload = {
            "runtime_session_id": str(runtime_session_id),
            "control_domain_id": str(control_domain_id),
            "producer_id": str(producer_id),
            "producer_epoch": int(producer_epoch),
            "sequence": sequence,
            "input_role": "selected",
            "action_hash": component_action.semantic_content_hash,
            "produced_at": float(produced_at),
            "produced_clock_domain_id": str(produced_clock_domain_id),
            "available_at": float(available_at),
            "available_clock_domain_id": str(available_clock_domain_id),
            "effective_from": float(effective_from),
            "effective_until": float(effective_until),
            "effective_clock_domain_id": str(effective_clock_domain_id),
            "policy_output_id": str(policy_output_id),
            "context_ids": tuple(context_ids),
        }
        input_hash = canonical_content_hash(
            input_payload,
            domain=EXCLUSIVE_SINGLE_SOURCE_INPUT_HASH_DOMAIN_V1,
        )
        input_id = f"exclusive-hitl-composition-input:{input_hash}"
        record = CompositionInputRecord(
            composition_input_id=input_id,
            runtime_session_id=str(runtime_session_id),
            control_domain_id=str(control_domain_id),
            producer_id=str(producer_id),
            producer_epoch=int(producer_epoch),
            sequence=sequence,
            input_role="selected",
            action=component_action,
            produced_at=float(produced_at),
            produced_clock_domain_id=str(produced_clock_domain_id),
            available_at=float(available_at),
            available_clock_domain_id=str(available_clock_domain_id),
            control_admission_sequence=sequence,
            effective_from=float(effective_from),
            effective_until=float(effective_until),
            effective_clock_domain_id=str(effective_clock_domain_id),
            context_ids=tuple(context_ids),
            observation_cut_id=None,
            policy_output_id=str(policy_output_id),
            source_fact_ids=(),
            source_fact_hashes=(),
            semantic_binding_id_by_kind={},
            semantic_binding_hash_by_kind={},
            composition_activation_id=None,
            composition_activation_hash=None,
            authority_candidate=False,
            semantic_content_hash=input_hash,
            audit_metadata={"exclusive_source_selection": True},
        )
        snapshot_payload = {
            "control_domain_id": str(control_domain_id),
            "target_effective_at": float(target_effective_at),
            "composition_spec_hash": spec.semantic_content_hash,
            "action_schema_hash": self._action_schema.semantic_content_hash,
            "input_id": input_id,
            "input_hash": input_hash,
        }
        snapshot_hash = canonical_content_hash(
            snapshot_payload,
            domain=EXCLUSIVE_SINGLE_SOURCE_SNAPSHOT_HASH_DOMAIN_V1,
        )
        snapshot = CompositionInputSnapshot(
            composition_input_snapshot_id=(
                f"exclusive-hitl-composition-snapshot:{snapshot_hash}"
            ),
            control_domain_id=str(control_domain_id),
            cutoff_at=float(available_at),
            cutoff_admission_sequence=sequence,
            target_effective_at=float(target_effective_at),
            composition_spec_hash=spec.semantic_content_hash,
            action_schema_hash=self._action_schema.semantic_content_hash,
            target_clock_domain_id=str(effective_clock_domain_id),
            selected_input_by_role={"selected": record},
            missing_roles=(),
            producer_epoch_snapshot_hash=f"exclusive-epoch:{int(producer_epoch)}",
            semantic_content_hash=snapshot_hash,
            selected_input_before_dependency_by_role={"selected": record},
            required_roles=("selected",),
        )
        evaluation = self._composer.compose(snapshot, spec=spec)
        if evaluation.output_status != "ready" or evaluation.composed_candidate is None:
            raise RuntimeError(
                "Exclusive single-source composition did not produce a candidate: "
                f"{evaluation.output_status}:{evaluation.reason}"
            )
        materialized = materialize_robot_action_v1(
            evaluation.composed_candidate,
            timestamp=action.timestamp,
            source=action.source,
        )
        return ExclusiveSingleSourceCompositionResult(
            action=materialized,
            composition_input_id=input_id,
            composition_input_hash=input_hash,
            composition_evaluation_id=evaluation.composition_evaluation_id,
            composition_evaluation_hash=evaluation.semantic_content_hash,
            composition_spec_hash=spec.semantic_content_hash,
            composed_candidate_hash=evaluation.composed_candidate.semantic_content_hash,
        )


__all__ = [
    "EXCLUSIVE_SINGLE_SOURCE_INPUT_HASH_DOMAIN_V1",
    "EXCLUSIVE_SINGLE_SOURCE_SNAPSHOT_HASH_DOMAIN_V1",
    "ExclusiveSingleSourceComposer",
    "ExclusiveSingleSourceCompositionResult",
]
