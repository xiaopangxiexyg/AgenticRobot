"""Scoped replay-only Control Station composition for MuJoCo.

The application consumes an existing immutable ``ReplaySourceFrameEmission``.
It does not open a dataset, own a cursor, resample a schedule, or create a
second replay action.  The current authoritative MuJoCo target is explicitly
``bimanual_qpos_v1`` or the explicit CS-2b-b arm-base pose execution profile.
Pose remains proposal/selected/safe until a downstream coupled IK resolver
creates a separate qpos execution target.  This module never converts pose in
the source adapter.  Real targets and production entrypoints remain excluded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.replay import (
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
)
from robot_motion.replay import (
    ReplayProposalSource,
    ReplayProposalSourceConfig,
    ReplaySourceFrameEmission,
    build_replay_binding_assessment,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1, BIMANUAL_QPOS_V1
from robot_motion.action.execution_manifest import (
    ControlExecutionManifest,
    ExecutionProfileExclusion,
    build_control_execution_manifest,
)
from sim.mujoco.deploy import (
    MujocoDeployRuntime,
    MujocoReplaySubmission,
)
from teleop_runtime.replay.source_binding import ReplayGuiSourceBinding

from ..adapters import StationOperationAdapter
from ..contracts import (
    ControlStationRunManifest,
    ExecutionDomainBinding,
    SourceSlotSpec,
    StationIntent,
    StationOperationResultDraft,
)
from ..publication_barrier import (
    StationPublicationBarrier,
    StationPublicationBarrierSnapshot,
)
from ..runtime import ControlStationRuntime, StationOperationExecution
from ..sources.replay import ReplaySourceObserverBinding
from ..state_machine import ControlStationTransition
from ..targets.authoritative import (
    StationAuthoritativeTargetAdapterBase,
    authoritative_authority_epoch_source_id,
    authoritative_execution_epoch_source_id,
    authoritative_owner_registry_id,
    capture_authoritative_execution_gate_snapshot,
)


CS2B_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1 = (
    "control_station_cs2b_qpos_replay_mujoco_station_v1"
)
CS2B_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1 = (
    "control_station_cs2b_arm_base_pose_replay_mujoco_station_v1"
)
CS2B_CANONICAL_DESK_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1 = (
    "control_station_cs2b_canonical_desk_pose_replay_mujoco_station_v1"
)
CS2BC_REPLAY_GUI_EXECUTION_MANIFEST_ID_V1 = (
    "control_station_cs2bc_dataset_replay_pose_to_qpos_sim_v1"
)
CS2B_REPLAY_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2b_mujoco_replay_binding_receipt.v1"
)
CS2B_REPLAY_PRODUCER_EPOCH_POLICY_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2b_replay_fixed_producer_epoch.v1"
)


def _station_trace_profile_id(runtime: MujocoDeployRuntime) -> str:
    mode = runtime.reference_activation_spec.activation_mode
    if mode == "immediate_target":
        return "deployable_sim_immediate_qref_station_v1"
    if mode == "scheduled_segment":
        if runtime.catalog.action_space == BIMANUAL_EE_POSE_V1:
            return "deployable_sim_scheduled_pose_to_qpos_station_v1"
        return "deployable_sim_scheduled_qpos_station_v1"
    raise ValueError(f"Unsupported MuJoCo activation mode: {mode!r}.")


def build_cs2b_mujoco_replay_station_manifest(
    runtime: MujocoDeployRuntime,
    source: ReplayProposalSource,
    observer_binding: ReplaySourceObserverBinding,
) -> ControlStationRunManifest:
    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(source, ReplayProposalSource):
        raise TypeError("source must be a ReplayProposalSource.")
    if not isinstance(observer_binding, ReplaySourceObserverBinding):
        raise TypeError("observer_binding must be a ReplaySourceObserverBinding.")
    config = source.config
    trace_profile_id = _station_trace_profile_id(runtime)
    producer_epoch_policy_hash = canonical_content_hash(
        {
            "producer_id": config.producer_id,
            "producer_epoch": config.producer_epoch,
            "activation": "fixed_on_mujoco_runtime_start",
        },
        domain=CS2B_REPLAY_PRODUCER_EPOCH_POLICY_HASH_DOMAIN_V1,
    )
    target = ExecutionDomainBinding(
        binding_id=f"mujoco-target:{runtime.config.control_domain_id}",
        target_kind="sim",
        control_domain_id=runtime.config.control_domain_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        target_runtime_action_contract_hash=(
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        execution_capability_profile_hash=(
            runtime.execution_capability.semantic_content_hash
        ),
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        selection_group_ids=runtime.core.config.active_selection_groups,
        owner_registry_id=authoritative_owner_registry_id(runtime),
        authority_epoch_source_id=authoritative_authority_epoch_source_id(runtime),
        execution_epoch_source_id=authoritative_execution_epoch_source_id(runtime),
        trace_requirement_profile_id=trace_profile_id,
        explicit_exclusions=("hardware_device_initialization", "real_backend"),
    )
    pose_profile = runtime.catalog.action_space == BIMANUAL_EE_POSE_V1
    pose_transformer = (
        None
        if not pose_profile or runtime.pose_resolver is None
        else runtime.pose_resolver.pose_transformer
    )
    return ControlStationRunManifest(
        manifest_id=(
            CS2B_CANONICAL_DESK_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1
            if pose_transformer is not None
            else (
                CS2B_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1
                if pose_profile
                else CS2B_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1
            )
        ),
        manifest_version="1",
        execution_binding=target,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=observer_binding.source_slot_id,
                source_kind="replay",
                producer_id=config.producer_id,
                source_action_contract_hash=(
                    config.source_action_contract.semantic_content_hash
                ),
                producer_epoch_policy_hash=producer_epoch_policy_hash,
                supported_uses=("candidate", "authority"),
                expected_emission_seam=(
                    "robot_motion.replay.proposal_source.ReplaySourceFrameEmission"
                ),
                required_binding_refs=(
                    observer_binding.semantic_content_hash,
                    config.source_contract.semantic_content_hash,
                    config.schedule.semantic_content_hash,
                    config.target_binding_spec.semantic_content_hash,
                    config.binding_assessment.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "replay_source_frame_emission",
                    "replay_source_adapter_receipt",
                    "replay_binding_assessment",
                    "candidate_binding_eligibility",
                    "action_proposal",
                    "arbitration_decision",
                    "station_publication_barrier_transition",
                ),
                explicit_exclusions=(
                    "control_station_dataset_reader",
                    "control_station_replay_cursor",
                    "control_station_replay_resampler",
                    "source_side_pose_to_qpos_conversion",
                ),
            ),
        ),
        authority_policy_hash=runtime.core.config.authority_policy.semantic_content_hash,
        decision_schedule_hash=runtime.core.config.decision_schedule.semantic_content_hash,
        trace_requirement_profile_id=trace_profile_id,
        capture_policy_id="capture_not_owned_by_cs2b_v1",
        allowed_intent_kinds=(
            "acquire_control_lease",
            "release_control_lease",
            "connect_source",
            "disconnect_source",
            "connect_target",
            "disconnect_target",
            "run_preflight",
            "request_candidate",
            "release_candidate",
            "request_motion_authority",
            "release_motion_authority",
            "hold",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "browser_raw_action_submission",
            "formal_episode_lifecycle",
            "human_motion_source",
            *(() if pose_profile else ("legacy_gui_pose_replay_authority",)),
            *(
                ("canonical_desk_pose_transform",)
                if pose_profile and pose_transformer is None
                else ()
            ),
            "mixed_source_authority",
            "policy_model",
            "production_entrypoint",
            "real_motion",
            "replay_cursor_ownership",
            "replay_dataset_reading",
            "replay_resampling",
        ),
    )


@dataclass(frozen=True)
class MujocoReplayOnlyBindingReceipt:
    manifest_hash: str
    source_contract_hash: str
    source_action_contract_hash: str
    source_observer_binding_hash: str
    target_binding_assessment_hash: str
    target_runtime_action_contract_hash: str
    execution_capability_profile_hash: str
    component_catalog_hash: str
    authority_policy_hash: str
    decision_schedule_hash: str
    owner_token_id: str
    owner_token_hash: str
    execution_epoch_snapshot_hash: str
    source_emission_identity_preserved: bool
    cursor_owned_by_control_station: bool
    resampler_owned_by_control_station: bool
    authoritative_data_plane_bound: bool
    station_authority_lifecycle_bound: bool
    pose_transform_contract_hash: str | None
    semantic_content_hash: str


class MujocoReplayOnlyComposition:
    """Exact replay-emission → existing MuJoCo authority binding."""

    def __init__(
        self,
        *,
        manifest: ControlStationRunManifest,
        runtime: MujocoDeployRuntime,
        source: ReplayProposalSource,
        observer_binding: ReplaySourceObserverBinding,
    ) -> None:
        self.manifest = manifest
        self.runtime = runtime
        self.source = source
        self.observer_binding = observer_binding
        self._started = False
        self.binding_receipt = self._validate_exact_binding()

    @property
    def started(self) -> bool:
        return self._started

    def _validate_exact_binding(self) -> MujocoReplayOnlyBindingReceipt:
        expected = build_cs2b_mujoco_replay_station_manifest(
            self.runtime,
            self.source,
            self.observer_binding,
        )
        if self.manifest.semantic_content_hash != expected.semantic_content_hash:
            raise ValueError("Control Station replay manifest binding drifted.")
        config = self.source.config
        if not self.runtime.config.enable_replay_authority:
            raise ValueError("CS-2b requires explicit MuJoCo replay authority.")
        if self.runtime.config.enable_composer_authority:
            raise ValueError("CS-2b forbids composer authority.")
        if self.runtime.core.config.initial_control_mode != "replay":
            raise ValueError("CS-2b requires exact replay control mode.")
        if self.runtime.config.decision_hz > 60.0:
            raise ValueError("CS-2b ActionArbiter rate cannot exceed 60 Hz.")
        if config.requested_use != REPLAY_BINDING_USE_EXECUTION_CANDIDATE:
            raise ValueError("CS-2b requires execution-candidate replay assessment.")
        if config.binding_assessment.verdict != REPLAY_BINDING_VERIFIED_COMPATIBLE:
            raise ValueError("CS-2b requires verified-compatible replay binding.")
        source_action_space = config.source_action_contract.action_space
        target_action_space = self.runtime.catalog.action_space
        if source_action_space != target_action_space:
            raise ValueError(
                "Control Station replay source/target action spaces are not exact."
            )
        if target_action_space == BIMANUAL_QPOS_V1:
            if getattr(self.runtime, "_pose_execution_enabled", False):
                raise ValueError("Qpos replay cannot bind a pose execution runtime.")
        elif target_action_space == BIMANUAL_EE_POSE_V1:
            if not getattr(self.runtime, "_pose_execution_enabled", False):
                raise ValueError(
                    "Pose replay requires the explicit shared-safe pose execution profile."
                )
            source_frame_hash = config.source_action_contract.pose_frame_contract_hash
            target_frame_hash = (
                self.runtime.target_runtime_action_contract.pose_frame_contract_hash
            )
            transformer = self.runtime.pose_resolver.pose_transformer
            if source_frame_hash != target_frame_hash:
                if transformer is None:
                    raise ValueError(
                        "Pose-frame mismatch requires an explicit downstream transform."
                    )
                receipt = transformer.binding_receipt
                if (
                    receipt.source_pose_frame_contract_hash != source_frame_hash
                    or receipt.target_pose_frame_contract_hash != target_frame_hash
                    or config.target_binding_spec.transform_receipt_id
                    != receipt.receipt_id
                    or config.target_binding_spec.transform_receipt_hash
                    != receipt.semantic_content_hash
                    or self.runtime.target_runtime_action_contract.supported_transform_hash_by_source_frame.get(
                        str(source_frame_hash)
                    )
                    != receipt.semantic_content_hash
                ):
                    raise ValueError(
                        "Replay pose transform binding does not match source and target contracts."
                    )
            elif transformer is not None:
                raise ValueError(
                    "Exact pose-frame replay must not bind an unnecessary transform."
                )
        else:
            raise ValueError("Unsupported Control Station replay action profile.")
        if (
            config.runtime_session_id != self.runtime.config.runtime_session_id
            or config.control_domain_id != self.runtime.config.control_domain_id
            or config.embodiment_instance_id
            != self.runtime.config.embodiment_instance_id
            or config.producer_id != self.runtime.config.replay_producer_id
            or config.producer_epoch != self.runtime.config.producer_epoch
            or config.component_schema_hash
            != self.runtime.catalog.semantic_content_hash
        ):
            raise ValueError("Replay source is not bound to the exact MuJoCo runtime.")
        if config.target_runtime_contract.semantic_content_hash != (
            self.runtime.target_runtime_action_contract.semantic_content_hash
        ):
            raise ValueError("Replay assessment targets another runtime contract.")
        if self.observer_binding.source_locator_id != config.source_contract.source_dataset_id:
            raise ValueError("Replay observer locator/source dataset identity disagree.")
        if self.observer_binding.source_start_index != config.schedule.source_start_index:
            raise ValueError("Replay observer/schedule start indices disagree.")
        if self.observer_binding.clock_domain_id != self.runtime.config.clock_domain_id:
            raise ValueError("Replay observer and MuJoCo runtime clock domains disagree.")
        owner_snapshot = self.runtime.core.dispatch_owners.snapshot()
        token = owner_snapshot.current_token
        if token is None or token.owner_kind != "shared_arbitration_runtime":
            raise ValueError("MuJoCo target lacks shared authoritative owner token.")
        execution_snapshot = self.runtime.core.execution_epochs.snapshot()
        payload = {
            "manifest_hash": self.manifest.semantic_content_hash,
            "source_contract_hash": config.source_contract.semantic_content_hash,
            "source_action_contract_hash": (
                config.source_action_contract.semantic_content_hash
            ),
            "source_observer_binding_hash": self.observer_binding.semantic_content_hash,
            "target_binding_assessment_hash": (
                config.binding_assessment.semantic_content_hash
            ),
            "target_runtime_action_contract_hash": (
                self.runtime.target_runtime_action_contract.semantic_content_hash
            ),
            "execution_capability_profile_hash": (
                self.runtime.execution_capability.semantic_content_hash
            ),
            "component_catalog_hash": self.runtime.catalog.semantic_content_hash,
            "authority_policy_hash": (
                self.runtime.core.config.authority_policy.semantic_content_hash
            ),
            "decision_schedule_hash": (
                self.runtime.core.config.decision_schedule.semantic_content_hash
            ),
            "owner_token_id": token.token_id,
            "owner_token_hash": token.semantic_content_hash,
            "execution_epoch_snapshot_hash": execution_snapshot.semantic_content_hash,
            "source_emission_identity_preserved": True,
            "cursor_owned_by_control_station": False,
            "resampler_owned_by_control_station": False,
            "authoritative_data_plane_bound": True,
            "station_authority_lifecycle_bound": True,
            "pose_transform_contract_hash": (
                None
                if self.runtime.pose_resolver is None
                or self.runtime.pose_resolver.pose_transformer is None
                else self.runtime.pose_resolver.pose_transformer.binding_receipt.semantic_content_hash
            ),
        }
        return MujocoReplayOnlyBindingReceipt(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=CS2B_REPLAY_BINDING_RECEIPT_HASH_DOMAIN_V1,
            ),
        )

    def start(self) -> MujocoReplayOnlyBindingReceipt:
        if not self._started:
            self.runtime.start()
            self._started = True
        return self.binding_receipt

    def submit_source_frame_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> MujocoReplaySubmission:
        if not self._started:
            raise RuntimeError("MujocoReplayOnlyComposition.start() is required.")
        self.observer_binding.validate_emission(emission)
        submission = self.runtime.submit_replay_source_frame_emission(
            self.source,
            emission,
        )
        if submission.source_emission is not emission:
            raise RuntimeError("Replay source-frame identity was rebuilt unexpectedly.")
        return submission

    def step(self, *, now: float | None = None, physics_ticks_after_dispatch: int = 0):
        if not self._started:
            raise RuntimeError("MujocoReplayOnlyComposition.start() is required.")
        return self.runtime.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        if self._started:
            self.runtime.close()
            self._started = False


class MujocoReplayStationTargetAdapter(StationAuthoritativeTargetAdapterBase):
    def __init__(self, *, composition, clock, clock_domain_id) -> None:
        super().__init__(
            adapter_id="cs2b-mujoco-replay-target-adapter",
            subject_id=composition.manifest.execution_binding.binding_id,
            supported_operations=(
                "connect_target",
                "disconnect_target",
                "run_preflight",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="replay",
        )

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        if intent.intent_kind == "connect_target":
            self.composition.start()
            status = "mujoco_replay_target_started_without_station_authority"
        elif intent.intent_kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "mujoco_replay_target_binding_preflighted"
        else:
            self.composition.close()
            status = "mujoco_replay_target_closed"
        return self._build_result(
            intent,
            publication_enabled=False,
            evidence_status=status,
        )


class MujocoReplayStationSourceAdapter(StationAuthoritativeTargetAdapterBase):
    def __init__(self, *, composition, clock, clock_domain_id) -> None:
        super().__init__(
            adapter_id="cs2b-existing-replay-emission-source-adapter",
            subject_id=composition.observer_binding.source_slot_id,
            supported_operations=(
                "connect_source",
                "disconnect_source",
                "run_preflight",
                "request_candidate",
                "release_candidate",
                "request_motion_authority",
                "release_motion_authority",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="replay",
        )

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        kind = intent.intent_kind
        lower_event_ids: tuple[str, ...] = ()
        if kind == "connect_source":
            status = "existing_replay_emission_source_connected_without_cursor"
        elif kind == "disconnect_source":
            status = "existing_replay_emission_source_disconnected"
        elif kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "verified_replay_emission_binding_preflighted"
        elif kind == "request_candidate":
            self._validate_live_binding(require_started=True)
            status = "verified_replay_candidate_bound"
        elif kind == "release_candidate":
            status = "verified_replay_candidate_released"
        elif kind == "request_motion_authority":
            self._validate_live_binding(require_started=True)
            gate = capture_authoritative_execution_gate_snapshot(self.composition)
            if gate.status != "ready":
                return self._build_failure(
                    intent,
                    status=f"execution_gate_not_ready:{gate.status}",
                )
            status = "existing_single_replay_source_authority_bound"
        else:
            outcome = self.composition.runtime.transition_execution_gate(
                "hold",
                reason="control_station_release_replay_motion_authority",
                source_event_ids=(intent.intent_id,),
            )
            lower_event_ids = outcome.lower_event_ids
            status = "station_replay_publication_closed_and_execution_held"
        return self._build_result(
            intent,
            publication_enabled=False,
            lower_event_ids=lower_event_ids,
            evidence_status=status,
        )


class ScopedMujocoReplayStationComposition:
    """Station lifecycle and atomic publication gate for CS-2b replay."""

    def __init__(self, *, station, data_plane, clock, clock_domain_id) -> None:
        self._station = station
        self._data_plane = data_plane
        self.target_adapter: StationOperationAdapter = MujocoReplayStationTargetAdapter(
            composition=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.source_adapter = MujocoReplayStationSourceAdapter(
            composition=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.publication_barrier = StationPublicationBarrier(
            source_slot_id=self.source_adapter.subject_id
        )
        self._barrier_transition_observer: Callable[..., object] | None = None
        self._barrier_observer_failure_count = 0
        self._barrier_observer_lock = threading.RLock()
        self._observed_barrier_transition_keys: set[tuple[str, str]] = set()

    @property
    def station(self) -> ControlStationRuntime:
        return self._station

    @property
    def data_plane(self) -> MujocoReplayOnlyComposition:
        return self._data_plane

    def authority_status_payload(self) -> dict[str, object]:
        state = self._station.state
        barrier = self.publication_barrier.snapshot()
        receiver = self._data_plane.runtime.core.capture_receiver_state()
        source_state = state.source_state_by_slot[self.source_adapter.subject_id]
        active = bool(
            barrier.enabled
            and state.lifecycle_state == "running"
            and source_state == "active"
        )
        return {
            "status_kind": "scoped_sim_station_replay_publication",
            "motion_authorized": active,
            "publication_enabled": barrier.enabled,
            "publication_generation": barrier.generation,
            "publication_snapshot_hash": barrier.semantic_content_hash,
            "authority_operation_result_id": barrier.authority_operation_result_id,
            "station_lifecycle_state": state.lifecycle_state,
            "source_slot_id": self.source_adapter.subject_id,
            "source_runtime_state": source_state,
            "owner_token_id": receiver.current_owner_token_id,
            "authority_epoch": receiver.current_authority_epoch,
            "execution_epoch": receiver.current_execution_epoch,
            "authoritative": False,
        }

    def bind_publication_barrier_observer(self, observer: Callable[..., object]) -> None:
        if not callable(observer):
            raise TypeError("publication barrier observer must be callable.")
        if self._barrier_transition_observer is not None:
            raise RuntimeError("Publication barrier observer is already bound.")
        if self._station.state.state_revision != 0:
            raise RuntimeError("Publication barrier observer must bind before use.")
        self._barrier_transition_observer = observer

    def _observe_barrier_transition(
        self,
        snapshot: StationPublicationBarrierSnapshot,
        *,
        causal_station_fact_id: str,
        occurred_at: float,
        clock_domain_id: str,
    ) -> None:
        observer = self._barrier_transition_observer
        if observer is None:
            return
        key = (snapshot.semantic_content_hash, str(causal_station_fact_id))
        with self._barrier_observer_lock:
            if key in self._observed_barrier_transition_keys:
                return
            try:
                observer(
                    snapshot,
                    causal_station_fact_id=causal_station_fact_id,
                    occurred_at=occurred_at,
                    clock_domain_id=clock_domain_id,
                )
                self._observed_barrier_transition_keys.add(key)
            except Exception:
                self._barrier_observer_failure_count += 1

    def prepare_pending_operation(self, transition: ControlStationTransition) -> None:
        if transition.receipt.status != "operation_requested":
            return
        if transition.intent.intent_kind in {
            "release_motion_authority",
            "release_candidate",
            "disconnect_source",
            "disconnect_target",
        }:
            snapshot = self.publication_barrier.disable(
                reason=f"station_operation_admitted:{transition.intent.intent_kind}"
            )
            self._observe_barrier_transition(
                snapshot,
                causal_station_fact_id=transition.intent.intent_id,
                occurred_at=transition.intent.available_at,
                clock_domain_id=transition.intent.available_clock_domain_id,
            )

    def complete_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> StationOperationExecution:
        self.prepare_pending_operation(transition)
        subject = transition.intent.subject_id
        if subject == self._station.manifest.execution_binding.binding_id:
            adapter = self.target_adapter
        elif subject == self.source_adapter.subject_id:
            adapter = self.source_adapter
        else:
            raise ValueError("Pending operation has no scoped replay adapter.")
        execution = self._station.execute_pending_operation(transition, adapter)
        completion = execution.completion_transition
        if transition.intent.intent_kind == "request_motion_authority":
            if (
                completion.accepted
                and completion.result.succeeded
                and completion.state_after.lifecycle_state == "running"
                and completion.state_after.source_state_by_slot[
                    self.source_adapter.subject_id
                ]
                == "active"
            ):
                snapshot = self.publication_barrier.enable_from_authority_result(
                    completion.result
                )
            else:
                snapshot = self.publication_barrier.disable(
                    reason="authority_completion_rejected_or_failed"
                )
            self._observe_barrier_transition(
                snapshot,
                causal_station_fact_id=completion.result.operation_result_id,
                occurred_at=completion.result.available_at,
                clock_domain_id=completion.result.available_clock_domain_id,
            )
        return execution

    def submit_source_frame_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> MujocoReplaySubmission:
        initial = self._station.state
        if (
            initial.lifecycle_state != "running"
            or initial.source_state_by_slot[self.source_adapter.subject_id] != "active"
        ):
            raise RuntimeError("Station replay authority is not active.")

        def submit() -> MujocoReplaySubmission:
            state = self._station.state
            if (
                state.lifecycle_state != "running"
                or state.source_state_by_slot[self.source_adapter.subject_id]
                != "active"
            ):
                raise RuntimeError("Station replay authority is not active.")
            return self._data_plane.submit_source_frame_emission(emission)

        return self.publication_barrier.execute_if_enabled(submit)

    def step(self, *, now: float | None = None, physics_ticks_after_dispatch: int = 0):
        return self._data_plane.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        self.publication_barrier.disable(reason="composition_closed")
        self._data_plane.close()


def build_scoped_cs2b_mujoco_replay_station(
    runtime: MujocoDeployRuntime,
    source: ReplayProposalSource,
    observer_binding: ReplaySourceObserverBinding,
    *,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoReplayStationComposition:
    manifest = build_cs2b_mujoco_replay_station_manifest(
        runtime,
        source,
        observer_binding,
    )
    data_plane = MujocoReplayOnlyComposition(
        manifest=manifest,
        runtime=runtime,
        source=source,
        observer_binding=observer_binding,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoReplayStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
    )


def build_cs2bc_replay_source_from_gui_binding(
    runtime: MujocoDeployRuntime,
    source_binding: ReplayGuiSourceBinding,
) -> tuple[ReplayProposalSource, ReplaySourceObserverBinding]:
    """Bind checked GUI source evidence to one exact MuJoCo target.

    This application-leaf helper does not read the dataset or create a replay
    emission.  It only relates the lower-owned preflight binding to the target
    contract already owned by ``MujocoDeployRuntime``.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(source_binding, ReplayGuiSourceBinding):
        raise TypeError("source_binding must be a ReplayGuiSourceBinding.")
    if source_binding.clock_domain_id != runtime.config.clock_domain_id:
        raise ValueError("GUI replay and MuJoCo runtime clock domains disagree.")
    source_action = source_binding.source_action_contract
    source_components = tuple(source_action.component_ids)
    if set(source_components) != set(runtime.catalog.component_by_id):
        raise ValueError(
            "CS-2b-c requires exact source/target component coverage."
        )
    transform_receipt_id: str | None = None
    transform_receipt_hash: str | None = None
    source_frame_hash = source_action.pose_frame_contract_hash
    target_frame_hash = runtime.target_runtime_action_contract.pose_frame_contract_hash
    if source_frame_hash != target_frame_hash:
        resolver = runtime.pose_resolver
        transformer = None if resolver is None else resolver.pose_transformer
        if transformer is None:
            raise ValueError(
                "GUI replay pose-frame mismatch requires the target's verified transform."
            )
        receipt = transformer.binding_receipt
        if (
            receipt.source_pose_frame_contract_hash != source_frame_hash
            or receipt.target_pose_frame_contract_hash != target_frame_hash
        ):
            raise ValueError("GUI replay pose transform does not bind these contracts.")
        transform_receipt_id = receipt.receipt_id
        transform_receipt_hash = receipt.semantic_content_hash
    binding_spec, assessment = build_replay_binding_assessment(
        source_action,
        runtime.target_runtime_action_contract,
        binding_id=(
            "control-station-gui-replay:"
            f"{source_binding.semantic_content_hash}:"
            f"{runtime.target_runtime_action_contract.semantic_content_hash}"
        ),
        transform_receipt_id=transform_receipt_id,
        transform_receipt_hash=transform_receipt_hash,
    )
    source = ReplayProposalSource(
        ReplayProposalSourceConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=runtime.config.replay_producer_id,
            producer_epoch=runtime.config.producer_epoch,
            component_schema_hash=runtime.catalog.semantic_content_hash,
            component_coverage=source_components,
            selection_groups=tuple(
                dict.fromkeys(
                    source_action.selection_group_by_component[component]
                    for component in source_components
                )
            ),
            source_contract=source_binding.source_contract,
            source_action_contract=source_action,
            target_runtime_contract=runtime.target_runtime_action_contract,
            derivation_receipt=source_binding.derivation_receipt,
            schedule=source_binding.schedule,
            target_binding_spec=binding_spec,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )
    observer = ReplaySourceObserverBinding.create(
        source_slot_id=runtime.config.replay_producer_id,
        source_locator_id=source_binding.source_contract.source_dataset_id,
        expected_source_path=source_binding.expected_source_path,
        source_episode_index=source_binding.source_episode_index,
        group_key=source_binding.group_key,
        source_start_index=source_binding.source_start_index,
        clock_domain_id=source_binding.clock_domain_id,
    )
    return source, observer


def build_scoped_cs2bc_mujoco_replay_gui_station(
    runtime: MujocoDeployRuntime,
    source_binding: ReplayGuiSourceBinding,
    *,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoReplayStationComposition:
    """Create Station wiring from a checked dataset-backed GUI source."""

    source, observer = build_cs2bc_replay_source_from_gui_binding(
        runtime,
        source_binding,
    )
    return build_scoped_cs2b_mujoco_replay_station(
        runtime,
        source,
        observer,
        station_clock=station_clock,
        station_clock_domain_id=station_clock_domain_id,
    )


def build_cs2bc_replay_gui_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the dataset-backed GUI replay → scheduled pose/qpos sim run.

    This is an immutable capture dependency, not motion authorization.  The
    existing GUI replay cursor remains the sole source-emission owner and the
    Control Station publication barrier remains application authority only.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if station_manifest.manifest_id not in {
        CS2B_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1,
        CS2B_CANONICAL_DESK_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1,
    }:
        raise ValueError(
            "CS2bc formal replay capture requires a pose-to-qpos Station manifest."
        )
    if runtime.catalog.action_space != BIMANUAL_EE_POSE_V1:
        raise ValueError("CS2bc formal replay capture requires pose proposals.")
    if runtime.reference_activation_spec.activation_mode != "scheduled_segment":
        raise ValueError("CS2bc formal replay capture requires scheduled activation.")
    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="station_dataset_replay_sim_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "hardware_device_initialization",
            "human_motion_proposal",
            "policy_model",
            "production_station_entrypoint",
            "real_backend",
            "replay_real_authority",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS2BC_REPLAY_GUI_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        catalog=runtime.catalog,
        motion_authority_sources=(
            "replay:existing_gui_source_via_control_station",
        ),
        human_motion_proposal_status="not_applicable",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("control_station_fixture",),
        lifecycle_control_requests=("start", "stop"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="not_applicable",
        exclusions=exclusions,
    )


__all__ = [
    "CS2B_CANONICAL_DESK_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1",
    "CS2B_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1",
    "CS2B_POSE_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1",
    "CS2BC_REPLAY_GUI_EXECUTION_MANIFEST_ID_V1",
    "MujocoReplayOnlyBindingReceipt",
    "MujocoReplayOnlyComposition",
    "MujocoReplayStationSourceAdapter",
    "MujocoReplayStationTargetAdapter",
    "ReplaySourceObserverBinding",
    "ScopedMujocoReplayStationComposition",
    "build_cs2b_mujoco_replay_station_manifest",
    "build_cs2bc_replay_gui_execution_manifest",
    "build_cs2bc_replay_source_from_gui_binding",
    "build_scoped_cs2b_mujoco_replay_station",
    "build_scoped_cs2bc_mujoco_replay_gui_station",
]
