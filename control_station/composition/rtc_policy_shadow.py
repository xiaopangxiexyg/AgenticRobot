"""CS3 null-target Station lifecycle for passive RTC policy observation.

This application composition owns only Station intent admission and a passive
subscriber fence.  It has no execution target, publication permit, dispatcher,
policy worker, timed plan, queue pop, emission commit, or episode lifecycle.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.deploy.rtc_frontend import RtcFrontendStep
from teleop_runtime.deploy.rtc_step_fanout import (
    RtcFrontendStepFanout,
    RtcFrontendStepSourceBinding,
)

from ..adapters import NullExecutionTargetAdapter, StationOperationAdapter
from ..contracts import (
    ControlStationRunManifest,
    ExecutionDomainBinding,
    SourceSlotSpec,
    StationIntent,
    StationOperationResultDraft,
)
from ..runtime import ControlStationRuntime, StationOperationExecution
from ..sources.rtc_policy import RtcPolicyStationConsumer
from ..state_machine import ControlStationTransition


CS3_RTC_POLICY_PASSIVE_SHADOW_MANIFEST_ID_V1 = (
    "control_station_cs3_rtc_policy_passive_shadow_v1"
)
CS3_RTC_POLICY_PASSIVE_OPERATION_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs3_rtc_policy_passive_operation.v1"
)
CS3_RTC_POLICY_PRODUCER_EPOCH_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs3_rtc_policy_producer_epoch.v1"
)


@dataclass(frozen=True)
class RtcPolicyPassiveStationOperationEvidence:
    source_intent_id: str
    operation_kind: str
    subject_id: str
    source_binding_hash: str
    shadow_enabled: bool
    fence_sequence: int
    evidence_status: str
    semantic_content_hash: str


def build_cs3_rtc_policy_passive_shadow_manifest(
    binding: RtcFrontendStepSourceBinding,
) -> ControlStationRunManifest:
    """Freeze a target-neutral, shadow-only Station run."""

    if not isinstance(binding, RtcFrontendStepSourceBinding):
        raise TypeError("binding must be an RtcFrontendStepSourceBinding.")
    producer_epoch_policy_hash = canonical_content_hash(
        {
            "producer_id": binding.producer_id,
            "producer_epoch": binding.producer_epoch,
            "activation": "owned_by_existing_rtc_frontend",
        },
        domain=CS3_RTC_POLICY_PRODUCER_EPOCH_HASH_DOMAIN_V1,
    )
    target = ExecutionDomainBinding(
        binding_id=f"null-target:rtc-shadow:{binding.control_domain_id}",
        target_kind="null",
        control_domain_id=binding.control_domain_id,
        embodiment_instance_id="not-applicable:passive-rtc-observer",
        target_runtime_action_contract_hash=(
            "not-applicable:cs3-passive-rtc-shadow-no-target-v1"
        ),
        execution_capability_profile_hash=(
            "no-execution-capability:cs3-passive-rtc-shadow-v1"
        ),
        component_catalog_hash="not-applicable:cs3-passive-source-only-v1",
        selection_group_ids=("not_applicable_passive_shadow",),
        owner_registry_id="no-owner-registry:cs3-passive-rtc-shadow-v1",
        authority_epoch_source_id="no-authority-epoch:cs3-passive-rtc-shadow-v1",
        execution_epoch_source_id="no-execution-epoch:cs3-passive-rtc-shadow-v1",
        trace_requirement_profile_id="control_station_passive_rtc_shadow_v1",
        explicit_exclusions=(
            "backend_dispatch",
            "execution_target",
            "motion_authority",
            "real_backend",
            "sim_backend",
        ),
    )
    return ControlStationRunManifest(
        manifest_id=CS3_RTC_POLICY_PASSIVE_SHADOW_MANIFEST_ID_V1,
        manifest_version="1",
        execution_binding=target,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=binding.source_slot_id,
                source_kind="policy",
                producer_id=binding.producer_id,
                source_action_contract_hash=binding.action_adapter_contract_hash,
                producer_epoch_policy_hash=producer_epoch_policy_hash,
                supported_uses=("shadow",),
                expected_emission_seam=(
                    "teleop_runtime.deploy.rtc_frontend.RtcFrontendStep"
                ),
                required_binding_refs=(
                    binding.semantic_content_hash,
                    binding.policy_runtime_config_hash,
                    binding.rtc_frontend_config_hash,
                    binding.rtc_profile_hash,
                ),
                explicit_exclusions=(
                    "action_candidate",
                    "control_station_policy_merge",
                    "control_station_policy_pop",
                    "control_station_policy_predict",
                    "control_station_rtc_producer_lifecycle",
                    "motion_authority",
                ),
            ),
        ),
        authority_policy_hash="no-motion-authority:cs3-passive-rtc-shadow-v1",
        decision_schedule_hash="not-applicable:no-station-decision-loop-v1",
        trace_requirement_profile_id="control_station_passive_rtc_shadow_v1",
        capture_policy_id="capture-not-owned:cs3-passive-rtc-shadow-v1",
        allowed_intent_kinds=(
            "acquire_control_lease",
            "release_control_lease",
            "connect_source",
            "disconnect_source",
            "connect_target",
            "disconnect_target",
            "run_preflight",
            "start_shadow",
            "stop_shadow",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "action_candidate",
            "backend_dispatch",
            "capture_episode_lifecycle",
            "control_station_rtc_producer_lifecycle",
            "motion",
            "motion_authority",
            "policy_artifact_loading",
            "production_entrypoint",
            "real_motion",
            "sim_authority",
        ),
    )


class RtcPolicyPassiveStationSourceAdapter:
    """Apply Station shadow fences without operating the RTC producer."""

    def __init__(
        self,
        *,
        consumer: RtcPolicyStationConsumer,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        if not isinstance(consumer, RtcPolicyStationConsumer):
            raise TypeError("consumer must be an RtcPolicyStationConsumer.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.consumer = consumer
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        if not self.clock_domain_id:
            raise ValueError("clock_domain_id must be non-empty.")
        self._results: dict[str, StationOperationResultDraft] = {}
        self._evidence: dict[str, RtcPolicyPassiveStationOperationEvidence] = {}

    @property
    def adapter_id(self) -> str:
        return "cs3-existing-rtc-step-passive-source-adapter"

    @property
    def subject_id(self) -> str:
        return self.consumer.binding.source_slot_id

    @property
    def supported_operations(self) -> tuple[str, ...]:
        return (
            "connect_source",
            "disconnect_source",
            "run_preflight",
            "start_shadow",
            "stop_shadow",
        )

    @property
    def evidence_by_intent_id(
        self,
    ) -> dict[str, RtcPolicyPassiveStationOperationEvidence]:
        return dict(self._evidence)

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        if not isinstance(intent, StationIntent):
            raise TypeError("intent must be a StationIntent.")
        if intent.subject_id != self.subject_id:
            raise ValueError("Station operation belongs to another source slot.")
        if intent.intent_kind not in self.supported_operations:
            raise ValueError("Station operation is not supported by this source.")
        cached = self._results.get(intent.intent_id)
        if cached is not None:
            return cached

        kind = intent.intent_kind
        if kind == "connect_source":
            self.consumer.start()
            status = "existing_rtc_step_fanout_source_connected"
            fence_sequence = self.consumer.status().fence_sequence
        elif kind == "run_preflight":
            snapshot = self.consumer.status()
            if (
                not snapshot.started
                or self.consumer.authoritative
                or self.consumer.producer_lifecycle_owned
                or self.consumer.motion_capable
            ):
                return self._result(
                    intent,
                    succeeded=False,
                    status="passive_rtc_source_preflight_failed",
                    fence_sequence=snapshot.fence_sequence,
                )
            status = "passive_rtc_step_source_binding_preflighted"
            fence_sequence = snapshot.fence_sequence
        elif kind == "start_shadow":
            fence_sequence = self.consumer.set_shadow_enabled(True)
            status = "passive_rtc_step_observation_enabled"
        elif kind == "stop_shadow":
            fence_sequence = self.consumer.set_shadow_enabled(False)
            status = "passive_rtc_step_observation_disabled"
        else:
            snapshot = self.consumer.status()
            if snapshot.shadow_enabled:
                return self._result(
                    intent,
                    succeeded=False,
                    status="passive_rtc_shadow_must_stop_before_disconnect",
                    fence_sequence=snapshot.fence_sequence,
                )
            status = "passive_rtc_step_source_disconnected"
            fence_sequence = snapshot.fence_sequence
        return self._result(
            intent,
            succeeded=True,
            status=status,
            fence_sequence=fence_sequence,
        )

    def _result(
        self,
        intent: StationIntent,
        *,
        succeeded: bool,
        status: str,
        fence_sequence: int,
    ) -> StationOperationResultDraft:
        payload = {
            "source_intent_id": intent.intent_id,
            "operation_kind": intent.intent_kind,
            "subject_id": self.subject_id,
            "source_binding_hash": self.consumer.binding.semantic_content_hash,
            "shadow_enabled": self.consumer.status().shadow_enabled,
            "fence_sequence": int(fence_sequence),
            "evidence_status": str(status),
        }
        evidence = RtcPolicyPassiveStationOperationEvidence(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=CS3_RTC_POLICY_PASSIVE_OPERATION_HASH_DOMAIN_V1,
            ),
        )
        self._evidence[intent.intent_id] = evidence
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=self.subject_id,
            succeeded=bool(succeeded),
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=evidence.evidence_status,
            evidence_refs=(
                evidence.semantic_content_hash,
                evidence.source_binding_hash,
            ),
        )
        self._results[intent.intent_id] = result
        return result


class ScopedRtcPolicyPassiveStationComposition:
    """Entry-point-free CS3 Station fixture around one passive consumer."""

    def __init__(
        self,
        *,
        station: ControlStationRuntime,
        consumer: RtcPolicyStationConsumer,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        expected = build_cs3_rtc_policy_passive_shadow_manifest(consumer.binding)
        if station.manifest.semantic_content_hash != expected.semantic_content_hash:
            raise ValueError("Station manifest does not match RTC source binding.")
        self.station = station
        self.consumer = consumer
        self.target_adapter: StationOperationAdapter = NullExecutionTargetAdapter(
            adapter_id="cs3-null-target-adapter",
            target_binding_id=station.manifest.execution_binding.binding_id,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.source_adapter: StationOperationAdapter = (
            RtcPolicyPassiveStationSourceAdapter(
                consumer=consumer,
                clock=clock,
                clock_domain_id=clock_domain_id,
            )
        )

    @property
    def adapters(self) -> tuple[StationOperationAdapter, ...]:
        return (self.target_adapter, self.source_adapter)

    def complete_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> StationOperationExecution:
        subject_id = transition.intent.subject_id
        adapter = next(
            (item for item in self.adapters if item.subject_id == subject_id),
            None,
        )
        if adapter is None:
            raise ValueError("No Station adapter owns the pending subject.")
        return self.station.execute_pending_operation(transition, adapter)

    def prepare_pending_operation(self, transition: ControlStationTransition) -> None:
        """Reserve no lower resource; fences are applied on the worker only."""

        if not isinstance(transition, ControlStationTransition):
            raise TypeError("transition must be a ControlStationTransition.")

    def passive_status_payload(self) -> dict[str, object]:
        status = self.consumer.status()
        return {
            "status_kind": "cs3a_passive_rtc_policy_source",
            "motion_authorized": False,
            "authoritative": False,
            "source_binding_hash": self.consumer.binding.semantic_content_hash,
            "shadow_enabled": status.shadow_enabled,
            "producer_lifecycle_owned": False,
            "queue_size": status.queue_size,
            "queue_capacity": status.queue_capacity,
            "observed_step_count": status.observed_step_count,
            "observed_emission_count": status.observed_emission_count,
            "inactive_step_count": status.inactive_step_count,
            "rejected_step_count": status.rejected_step_count,
            "error_count": status.error_count,
        }

    def close(self) -> None:
        self.consumer.close()


class RtcPolicyPassiveStationBootstrap:
    """Prepare root wiring before the lower deploy frontend exists.

    A future application root passes ``fanout`` into ``run_policy.main``.  The
    lower runtime publishes its source binding exactly once; only then can the
    root construct the null-target Station around the already-supplied queue.
    This object neither starts deploy nor waits on policy inference.
    """

    def __init__(self, *, queue_capacity: int = 256) -> None:
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("RTC Station bootstrap queue_capacity must be positive.")
        self.step_queue: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.fanout = RtcFrontendStepFanout(
            {"control_station:rtc-policy-passive": self.step_queue}
        )
        self._lock = threading.RLock()
        self._station: ScopedRtcPolicyPassiveStationComposition | None = None

    @property
    def station(self) -> ScopedRtcPolicyPassiveStationComposition | None:
        with self._lock:
            return self._station

    def build_station_after_source_binding(
        self,
        observe_step: Callable[[RtcFrontendStep], object],
        *,
        station_clock: Callable[[], float],
        timeout_sec: float,
        station_clock_domain_id: str = "control-station-rtc-passive-monotonic",
    ) -> ScopedRtcPolicyPassiveStationComposition:
        binding = self.fanout.wait_for_source_binding(timeout_sec=timeout_sec)
        if binding is None:
            raise TimeoutError("RTC deploy source binding was not published in time.")
        with self._lock:
            if self._station is not None:
                raise RuntimeError("RTC passive Station was already constructed.")
            station = build_scoped_cs3_rtc_policy_passive_station(
                binding,
                observe_step,
                station_clock=station_clock,
                station_clock_domain_id=station_clock_domain_id,
                step_queue=self.step_queue,
            )
            self._station = station
            return station


def build_scoped_cs3_rtc_policy_passive_station(
    binding: RtcFrontendStepSourceBinding,
    observe_step: Callable[[RtcFrontendStep], object],
    *,
    station_clock: Callable[[], float],
    station_clock_domain_id: str = "control-station-rtc-passive-monotonic",
    step_queue: queue.Queue[object] | None = None,
    queue_capacity: int = 256,
) -> ScopedRtcPolicyPassiveStationComposition:
    """Build a passive Station fixture without receiving an RTC frontend."""

    manifest = build_cs3_rtc_policy_passive_shadow_manifest(binding)
    consumer = RtcPolicyStationConsumer(
        binding,
        observe_step,
        step_queue=step_queue,
        queue_capacity=queue_capacity,
    )
    station = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id=f"control-station:{binding.runtime_session_id}",
        clock_domain_id=station_clock_domain_id,
        clock=station_clock,
    )
    return ScopedRtcPolicyPassiveStationComposition(
        station=station,
        consumer=consumer,
        clock=station_clock,
        clock_domain_id=station_clock_domain_id,
    )


__all__ = [
    "CS3_RTC_POLICY_PASSIVE_SHADOW_MANIFEST_ID_V1",
    "RtcPolicyPassiveStationOperationEvidence",
    "RtcPolicyPassiveStationBootstrap",
    "RtcPolicyPassiveStationSourceAdapter",
    "ScopedRtcPolicyPassiveStationComposition",
    "build_cs3_rtc_policy_passive_shadow_manifest",
    "build_scoped_cs3_rtc_policy_passive_station",
]
