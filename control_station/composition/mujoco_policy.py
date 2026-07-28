"""Scoped sync-policy emission → MuJoCo Control Station composition.

CS-2c consumes an already-created ``PolicyActionEmission``.  Prediction,
action-chunk ownership, queue mutation, and the only queue pop remain in
``PolicyDeployRuntime``.  This application-leaf module binds that immutable
fact to the existing authoritative MuJoCo runtime; it never calls ``predict``
or ``prepare_next_action`` itself.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any, Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ControlExecutionManifest,
    ExecutionProfileExclusion,
    build_control_execution_manifest,
)
from robot_motion.deploy import (
    CanonicalDeployActionPipeline,
    PolicyActionEmission,
    PolicyDeployRuntime,
)
from sim.mujoco.deploy import (
    MujocoDeployRuntime,
    MujocoPolicyEmissionSubmission,
)

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
from ..sources.policy import (
    SYNC_POLICY_BINDING_HASH_DOMAIN_V1,
    SYNC_POLICY_RUNTIME_CONFIG_HASH_DOMAIN_V1,
    SyncPolicyEmissionBinding,
)
from ..state_machine import ControlStationTransition
from ..targets.authoritative import (
    StationAuthoritativeTargetAdapterBase,
    authoritative_authority_epoch_source_id,
    authoritative_execution_epoch_source_id,
    authoritative_owner_registry_id,
    capture_authoritative_execution_gate_snapshot,
)


CS2C_SYNC_POLICY_MUJOCO_MANIFEST_ID_V1 = (
    "control_station_cs2c_sync_policy_mujoco_v1"
)
CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1 = (
    "control_station_cs2c_sync_policy_mujoco_station_v1"
)
CS2C_SYNC_POLICY_EXECUTION_MANIFEST_ID_V1 = (
    "cs2c_sync_policy_station_scheduled_qpos_sim_v1"
)
CS2C_POLICY_PRODUCER_EPOCH_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2c_policy_producer_epoch.v1"
)
CS2C_POLICY_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2c_mujoco_policy_binding_receipt.v1"
)


def build_sync_policy_emission_binding(
    *,
    policy_runtime: PolicyDeployRuntime,
    pipeline: CanonicalDeployActionPipeline,
    target_runtime: MujocoDeployRuntime,
) -> SyncPolicyEmissionBinding:
    """Bind a generic policy source to this exact MuJoCo target profile."""

    if not isinstance(policy_runtime, PolicyDeployRuntime):
        raise TypeError("policy_runtime must be a PolicyDeployRuntime.")
    if not isinstance(pipeline, CanonicalDeployActionPipeline):
        raise TypeError("pipeline must be a CanonicalDeployActionPipeline.")
    if not isinstance(target_runtime, MujocoDeployRuntime):
        raise TypeError("target_runtime must be a MujocoDeployRuntime.")
    config = policy_runtime.config
    if tuple(config.active_sides) != ("left", "right"):
        raise ValueError("CS-2c currently requires exact left/right policy coverage.")
    if not config.arms_enabled or config.hands_enabled:
        raise ValueError("CS-2c v1 is arms-only.")
    if config.policy_node_id != target_runtime.config.policy_node_id:
        raise ValueError("Policy source and MuJoCo target policy-node IDs disagree.")
    if config.policy_clock_domain_id != target_runtime.config.clock_domain_id:
        raise ValueError("Policy source and MuJoCo target clock domains disagree.")
    runtime_payload = {
        "active_sides": config.active_sides,
        "arms_enabled": config.arms_enabled,
        "hands_enabled": config.hands_enabled,
        "policy_action_pose_mode": config.policy_action_pose_mode,
        "policy_absolute_frame": config.policy_absolute_frame,
        "policy_node_id": config.policy_node_id,
        "policy_clock_domain_id": config.policy_clock_domain_id,
        "action_keyframe_period_sec": config.action_keyframe_period_sec,
        "action_maximum_hold_sec": config.action_maximum_hold_sec,
        "action_emission_admission_mode": config.action_emission_admission_mode,
        "action_chunk_execution_steps": config.action_chunk_execution_steps,
        "policy_observation_contract_hash": (
            config.policy_observation_contract.semantic_content_hash
        ),
    }
    runtime_hash = canonical_content_hash(
        runtime_payload,
        domain=SYNC_POLICY_RUNTIME_CONFIG_HASH_DOMAIN_V1,
    )
    stable = {
        "source_slot_id": target_runtime.config.producer_id,
        "producer_id": target_runtime.config.producer_id,
        "producer_epoch": target_runtime.config.producer_epoch,
        "policy_node_id": config.policy_node_id,
        "policy_clock_domain_id": config.policy_clock_domain_id,
        "policy_runtime_config_hash": runtime_hash,
        "action_adapter_contract_hash": pipeline.action_adapter.contract_hash,
    }
    return SyncPolicyEmissionBinding(
        **stable,
        semantic_content_hash=canonical_content_hash(
            stable,
            domain=SYNC_POLICY_BINDING_HASH_DOMAIN_V1,
        ),
    )


def _trace_profile_id(runtime: MujocoDeployRuntime) -> str:
    if runtime.reference_activation_spec.activation_mode == "immediate_target":
        return "deployable_sim_immediate_qref_v1"
    if runtime.reference_activation_spec.activation_mode == "scheduled_segment":
        return "deployable_sim_scheduled_qpos_v1"
    raise ValueError("Unsupported MuJoCo activation mode.")


def build_cs2c_sync_policy_mujoco_manifest(
    runtime: MujocoDeployRuntime,
    binding: SyncPolicyEmissionBinding,
) -> ControlStationRunManifest:
    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(binding, SyncPolicyEmissionBinding):
        raise TypeError("binding must be a SyncPolicyEmissionBinding.")
    trace_profile_id = _trace_profile_id(runtime)
    producer_epoch_policy_hash = canonical_content_hash(
        {
            "producer_id": binding.producer_id,
            "producer_epoch": binding.producer_epoch,
            "activation": "fixed_on_mujoco_runtime_start",
        },
        domain=CS2C_POLICY_PRODUCER_EPOCH_HASH_DOMAIN_V1,
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
    return ControlStationRunManifest(
        manifest_id=CS2C_SYNC_POLICY_MUJOCO_MANIFEST_ID_V1,
        manifest_version="1",
        execution_binding=target,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=binding.source_slot_id,
                source_kind="policy",
                producer_id=binding.producer_id,
                source_action_contract_hash=binding.action_adapter_contract_hash,
                producer_epoch_policy_hash=producer_epoch_policy_hash,
                supported_uses=("authority",),
                expected_emission_seam=(
                    "robot_motion.deploy.runtime.PolicyActionEmission"
                ),
                required_binding_refs=(
                    binding.semantic_content_hash,
                    binding.policy_runtime_config_hash,
                    runtime.catalog.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "policy_action_emission",
                    "source_validation",
                    "action_proposal",
                    "arbitration_decision",
                ),
                explicit_exclusions=(
                    "control_station_action_chunk_queue",
                    "control_station_policy_merge",
                    "control_station_policy_pop",
                    "control_station_policy_predict",
                    "rtc_timed_plan",
                ),
            ),
        ),
        authority_policy_hash=runtime.core.config.authority_policy.semantic_content_hash,
        decision_schedule_hash=runtime.core.config.decision_schedule.semantic_content_hash,
        trace_requirement_profile_id=trace_profile_id,
        capture_policy_id="capture_not_owned_by_cs2c_v1",
        allowed_intent_kinds=(
            "connect_source",
            "connect_target",
            "run_preflight",
            "hold",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "browser_raw_action_submission",
            "formal_episode_lifecycle",
            "human_motion_source",
            "mixed_source_authority",
            "policy_artifact_preflight",
            "policy_inference_ownership",
            "production_entrypoint",
            "real_motion",
            "replay_source",
            "rtc_async_policy",
            "station_motion_authority_lifecycle",
        ),
    )


def build_cs2c_sync_policy_mujoco_station_manifest(
    runtime: MujocoDeployRuntime,
    binding: SyncPolicyEmissionBinding,
) -> ControlStationRunManifest:
    """Add scoped Station lifecycle without changing the policy data plane."""

    base = build_cs2c_sync_policy_mujoco_manifest(runtime, binding)
    trace_profile_id = (
        "deployable_sim_immediate_qref_station_v1"
        if runtime.reference_activation_spec.activation_mode == "immediate_target"
        else "deployable_sim_scheduled_qpos_policy_station_v1"
    )
    source_slot = replace(
        base.source_slots[0],
        supported_uses=("candidate", "authority"),
        trace_requirement_refs=tuple(
            sorted(
                {
                    *base.source_slots[0].trace_requirement_refs,
                    "station_run_manifest",
                    "station_intent_transition",
                    "station_operation_transition",
                    "station_publication_barrier_transition",
                }
            )
        ),
    )
    exclusions = tuple(
        value
        for value in base.explicit_exclusions
        if value != "station_motion_authority_lifecycle"
    )
    return replace(
        base,
        manifest_id=CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1,
        execution_binding=replace(
            base.execution_binding,
            trace_requirement_profile_id=trace_profile_id,
        ),
        source_slots=(source_slot,),
        trace_requirement_profile_id=trace_profile_id,
        capture_policy_id="capture_not_owned_by_cs2c_station_v1",
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
        explicit_exclusions=exclusions,
    )


def build_cs2c_sync_policy_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the existing sync-policy stream used by formal sim capture.

    The fact is a capture dependency, not artifact preflight, inference
    ownership, a production entrypoint, or real-motion authorization.
    """

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if (
        station_manifest.manifest_id
        != CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1
    ):
        raise ValueError(
            "Policy execution manifest requires the exact CS2c Station manifest."
        )
    if runtime.catalog.action_space != BIMANUAL_QPOS_V1:
        raise ValueError("Formal CS2c policy capture requires bimanual qpos.")
    if runtime.reference_activation_spec.activation_mode != "scheduled_segment":
        raise ValueError("Formal CS2c policy capture requires scheduled activation.")
    if (
        station_manifest.trace_requirement_profile_id
        != "deployable_sim_scheduled_qpos_policy_station_v1"
    ):
        raise ValueError("CS2c Station manifest lacks the policy formal profile.")

    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="station_sync_policy_sim_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "dataset_replay",
            "hardware_device_initialization",
            "human_motion_proposal",
            "policy_artifact_preflight",
            "production_station_entrypoint",
            "real_backend",
            "rtc_async_policy",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS2C_SYNC_POLICY_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=(
            "policy:existing_sync_emission_via_control_station",
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


@dataclass(frozen=True)
class MujocoPolicyOnlyBindingReceipt:
    manifest_hash: str
    source_binding_hash: str
    policy_runtime_config_hash: str
    action_adapter_contract_hash: str
    target_runtime_action_contract_hash: str
    execution_capability_profile_hash: str
    component_catalog_hash: str
    authority_policy_hash: str
    decision_schedule_hash: str
    owner_token_id: str
    owner_token_hash: str
    execution_epoch_snapshot_hash: str
    source_emission_identity_preserved: bool
    policy_queue_owned_by_control_station: bool
    prediction_owned_by_control_station: bool
    authoritative_data_plane_bound: bool
    station_authority_lifecycle_bound: bool
    semantic_content_hash: str


def validate_policy_data_plane_binding(
    *,
    manifest: ControlStationRunManifest,
    expected_manifest: ControlStationRunManifest,
    runtime: MujocoDeployRuntime,
    binding: Any,
    station_authority_lifecycle_bound: bool,
    receipt_hash_domain: str,
    scope_label: str,
) -> MujocoPolicyOnlyBindingReceipt:
    """Validate the common exact-emission/MuJoCo authority boundary."""

    if manifest.semantic_content_hash != expected_manifest.semantic_content_hash:
        raise ValueError(f"Control Station {scope_label} manifest binding drifted.")
    if runtime.catalog.action_space != BIMANUAL_QPOS_V1:
        raise ValueError(f"{scope_label} requires exact bimanual qpos execution.")
    if runtime.config.enable_replay_authority:
        raise ValueError(f"{scope_label} forbids replay authority.")
    if runtime.config.enable_composer_authority:
        raise ValueError(f"{scope_label} forbids composer authority.")
    if runtime.core.config.initial_control_mode != "policy":
        raise ValueError(f"{scope_label} requires exact policy control mode.")
    if (
        binding.producer_id != runtime.config.producer_id
        or binding.producer_epoch != runtime.config.producer_epoch
        or binding.policy_node_id != runtime.config.policy_node_id
        or binding.policy_clock_domain_id != runtime.config.clock_domain_id
    ):
        raise ValueError("Policy emission binding targets another runtime.")
    owner_snapshot = runtime.core.dispatch_owners.snapshot()
    token = owner_snapshot.current_token
    if token is None or token.owner_kind != "shared_arbitration_runtime":
        raise ValueError("MuJoCo target lacks shared authoritative owner token.")
    execution_snapshot = runtime.core.execution_epochs.snapshot()
    payload = {
        "manifest_hash": manifest.semantic_content_hash,
        "source_binding_hash": binding.semantic_content_hash,
        "policy_runtime_config_hash": binding.policy_runtime_config_hash,
        "action_adapter_contract_hash": binding.action_adapter_contract_hash,
        "target_runtime_action_contract_hash": (
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        "execution_capability_profile_hash": (
            runtime.execution_capability.semantic_content_hash
        ),
        "component_catalog_hash": runtime.catalog.semantic_content_hash,
        "authority_policy_hash": (
            runtime.core.config.authority_policy.semantic_content_hash
        ),
        "decision_schedule_hash": (
            runtime.core.config.decision_schedule.semantic_content_hash
        ),
        "owner_token_id": token.token_id,
        "owner_token_hash": token.semantic_content_hash,
        "execution_epoch_snapshot_hash": execution_snapshot.semantic_content_hash,
        "source_emission_identity_preserved": True,
        "policy_queue_owned_by_control_station": False,
        "prediction_owned_by_control_station": False,
        "authoritative_data_plane_bound": True,
        "station_authority_lifecycle_bound": station_authority_lifecycle_bound,
    }
    return MujocoPolicyOnlyBindingReceipt(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=receipt_hash_domain,
        ),
    )


class MujocoPolicyOnlyComposition:
    """Exact lower-owned policy emission → existing MuJoCo authority binding."""

    def __init__(
        self,
        *,
        manifest: ControlStationRunManifest,
        runtime: MujocoDeployRuntime,
        binding: SyncPolicyEmissionBinding,
    ) -> None:
        self.manifest = manifest
        self.runtime = runtime
        self.binding = binding
        self._started = False
        self.binding_receipt = self._validate_exact_binding()

    @property
    def started(self) -> bool:
        return self._started

    def _validate_exact_binding(self) -> MujocoPolicyOnlyBindingReceipt:
        if self.manifest.manifest_id == CS2C_SYNC_POLICY_MUJOCO_MANIFEST_ID_V1:
            expected = build_cs2c_sync_policy_mujoco_manifest(
                self.runtime,
                self.binding,
            )
            station_authority_lifecycle_bound = False
        elif (
            self.manifest.manifest_id
            == CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1
        ):
            expected = build_cs2c_sync_policy_mujoco_station_manifest(
                self.runtime,
                self.binding,
            )
            station_authority_lifecycle_bound = True
        else:
            raise ValueError("Unsupported sync-policy MuJoCo Station manifest.")
        return validate_policy_data_plane_binding(
            manifest=self.manifest,
            expected_manifest=expected,
            runtime=self.runtime,
            binding=self.binding,
            station_authority_lifecycle_bound=station_authority_lifecycle_bound,
            receipt_hash_domain=CS2C_POLICY_BINDING_RECEIPT_HASH_DOMAIN_V1,
            scope_label="CS-2c sync-policy",
        )

    def start(self) -> MujocoPolicyOnlyBindingReceipt:
        if not self._started:
            self.runtime.start()
            self._started = True
        return self.binding_receipt

    def submit_emission(
        self,
        emission: PolicyActionEmission,
    ) -> MujocoPolicyEmissionSubmission:
        if not self._started:
            raise RuntimeError("MujocoPolicyOnlyComposition.start() is required.")
        self.binding.validate_emission(emission)
        return self.runtime.submit_policy_action_emission(emission)

    def step(self, *, now: float | None = None, physics_ticks_after_dispatch: int = 0):
        if not self._started:
            raise RuntimeError("MujocoPolicyOnlyComposition.start() is required.")
        return self.runtime.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        if self._started:
            self.runtime.close()
            self._started = False


class MujocoPolicyStationTargetAdapter(StationAuthoritativeTargetAdapterBase):
    """Connect/preflight the existing MuJoCo target without policy work."""

    def __init__(self, *, composition, clock, clock_domain_id) -> None:
        super().__init__(
            adapter_id="cs2c-mujoco-target-adapter",
            subject_id=composition.manifest.execution_binding.binding_id,
            supported_operations=(
                "connect_target",
                "disconnect_target",
                "run_preflight",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="policy",
        )

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        if intent.intent_kind == "connect_target":
            self.composition.start()
            status = "mujoco_policy_target_connected"
        elif intent.intent_kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "mujoco_policy_target_binding_preflighted"
        else:
            self.composition.close()
            status = "mujoco_policy_target_closed"
        return self._build_result(
            intent,
            publication_enabled=False,
            evidence_status=status,
        )


class MujocoPolicyStationSourceAdapter(StationAuthoritativeTargetAdapterBase):
    """Manage source lifecycle without owning inference or the action queue."""

    def __init__(self, *, composition, clock, clock_domain_id) -> None:
        super().__init__(
            adapter_id="cs2c-existing-policy-emission-source-adapter",
            subject_id=composition.binding.source_slot_id,
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
            expected_control_mode="policy",
        )

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        kind = intent.intent_kind
        lower_event_ids: tuple[str, ...] = ()
        if kind == "connect_source":
            status = "existing_policy_emission_source_connected_without_queue"
        elif kind == "disconnect_source":
            status = "existing_policy_emission_source_disconnected"
        elif kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "policy_emission_binding_preflighted"
        elif kind == "request_candidate":
            self._validate_live_binding(require_started=True)
            status = "policy_emission_candidate_bound"
        elif kind == "release_candidate":
            status = "policy_emission_candidate_released"
        elif kind == "request_motion_authority":
            self._validate_live_binding(require_started=True)
            gate = capture_authoritative_execution_gate_snapshot(self.composition)
            if gate.status != "ready":
                return self._build_failure(
                    intent,
                    status=f"execution_gate_not_ready:{gate.status}",
                )
            status = "existing_single_policy_source_authority_bound"
        else:
            outcome = self.composition.runtime.transition_execution_gate(
                "hold",
                reason="control_station_release_policy_motion_authority",
                source_event_ids=(intent.intent_id,),
            )
            lower_event_ids = outcome.lower_event_ids
            status = "station_policy_publication_closed_and_execution_held"
        return self._build_result(
            intent,
            publication_enabled=False,
            lower_event_ids=lower_event_ids,
            evidence_status=status,
        )


class ScopedMujocoPolicyStationComposition:
    """Station lifecycle and atomic publication gate for CS-2c sync policy."""

    def __init__(self, *, station, data_plane, clock, clock_domain_id) -> None:
        self._station = station
        self._data_plane = data_plane
        self.target_adapter: StationOperationAdapter = MujocoPolicyStationTargetAdapter(
            composition=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.source_adapter = MujocoPolicyStationSourceAdapter(
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
    def data_plane(self) -> MujocoPolicyOnlyComposition:
        return self._data_plane

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
            raise ValueError("Pending operation has no scoped policy adapter.")
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

    def submit_emission(
        self,
        emission: PolicyActionEmission,
    ) -> MujocoPolicyEmissionSubmission:
        initial = self._station.state
        if (
            initial.lifecycle_state != "running"
            or initial.source_state_by_slot[self.source_adapter.subject_id] != "active"
        ):
            raise RuntimeError("Station policy authority is not active.")

        def submit() -> MujocoPolicyEmissionSubmission:
            state = self._station.state
            if (
                state.lifecycle_state != "running"
                or state.source_state_by_slot[self.source_adapter.subject_id]
                != "active"
            ):
                raise RuntimeError("Station policy authority is not active.")
            return self._data_plane.submit_emission(emission)

        return self.publication_barrier.execute_if_enabled(submit)

    def step(self, *, now: float | None = None, physics_ticks_after_dispatch: int = 0):
        return self._data_plane.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        self.publication_barrier.disable(reason="composition_closed")
        self._data_plane.close()


def build_scoped_cs2c_sync_policy_mujoco(
    runtime: MujocoDeployRuntime,
    *,
    policy_runtime: PolicyDeployRuntime,
    pipeline: CanonicalDeployActionPipeline,
) -> MujocoPolicyOnlyComposition:
    """Build the CS-2c data-plane fixture without consuming policy work."""

    binding = build_sync_policy_emission_binding(
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        target_runtime=runtime,
    )
    manifest = build_cs2c_sync_policy_mujoco_manifest(runtime, binding)
    return MujocoPolicyOnlyComposition(
        manifest=manifest,
        runtime=runtime,
        binding=binding,
    )


def build_scoped_cs2c_sync_policy_mujoco_station(
    runtime: MujocoDeployRuntime,
    *,
    policy_runtime: PolicyDeployRuntime,
    pipeline: CanonicalDeployActionPipeline,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoPolicyStationComposition:
    """Build scoped policy Station lifecycle without owning policy work."""

    binding = build_sync_policy_emission_binding(
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        target_runtime=runtime,
    )
    manifest = build_cs2c_sync_policy_mujoco_station_manifest(runtime, binding)
    data_plane = MujocoPolicyOnlyComposition(
        manifest=manifest,
        runtime=runtime,
        binding=binding,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoPolicyStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
    )


__all__ = [
    "CS2C_SYNC_POLICY_EXECUTION_MANIFEST_ID_V1",
    "CS2C_SYNC_POLICY_MUJOCO_MANIFEST_ID_V1",
    "CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1",
    "MujocoPolicyOnlyBindingReceipt",
    "MujocoPolicyOnlyComposition",
    "MujocoPolicyStationSourceAdapter",
    "MujocoPolicyStationTargetAdapter",
    "ScopedMujocoPolicyStationComposition",
    "SyncPolicyEmissionBinding",
    "build_cs2c_sync_policy_mujoco_manifest",
    "build_cs2c_sync_policy_mujoco_station_manifest",
    "build_cs2c_sync_policy_execution_manifest",
    "build_scoped_cs2c_sync_policy_mujoco",
    "build_scoped_cs2c_sync_policy_mujoco_station",
    "build_sync_policy_emission_binding",
    "validate_policy_data_plane_binding",
]
