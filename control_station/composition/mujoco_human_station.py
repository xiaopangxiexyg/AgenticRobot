"""Scoped CS2a2b Station lifecycle over the synthetic-human MuJoCo slice.

This is a test/application composition, not a production entrypoint.  It does
not create another Arbiter, owner registry, execution epoch registry, queue,
dispatcher, or episode lifecycle.  Station authority only opens a publication
gate in front of the already-authoritative single-human MuJoCo data plane.
"""

from __future__ import annotations

import threading
from typing import Callable

from sim.mujoco.deploy import MujocoDeployRuntime
from teleop_runtime.action.human_proposal import HumanCommandEmission

from ..adapters import StationOperationAdapter
from ..contracts import StationIntent, StationOperationResultDraft
from ..publication_barrier import StationPublicationBarrier
from ..publication_barrier import StationPublicationBarrierSnapshot
from ..runtime import ControlStationRuntime, StationOperationExecution
from ..state_machine import ControlStationTransition
from ..sources.synthetic_human import SyntheticHumanQposSourceSpec
from ..targets.authoritative import (
    StationAuthoritativeTargetAdapterBase,
    StationTargetOperationEvidence,
    capture_authoritative_execution_gate_snapshot,
)
from .mujoco_human import (
    CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1,
    MujocoHumanOnlyComposition,
    MujocoHumanSubmission,
    build_cs2a2b_mujoco_human_station_manifest,
    build_mujoco_human_proposal_adapter,
)


# Stable public compatibility name for existing ControlStation consumers.
MujocoStationOperationEvidence = StationTargetOperationEvidence


class MujocoHumanStationTargetAdapter(StationAuthoritativeTargetAdapterBase):
    """Start/preflight the existing MuJoCo runtime without granting authority."""

    def __init__(
        self,
        *,
        composition: MujocoHumanOnlyComposition,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        super().__init__(
            adapter_id="cs2a2b-mujoco-target-adapter",
            subject_id=composition.manifest.execution_binding.binding_id,
            supported_operations=(
                "connect_target",
                "disconnect_target",
                "run_preflight",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        if intent.intent_kind == "connect_target":
            self.composition.start()
            status = "mujoco_target_started_without_station_authority"
        elif intent.intent_kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "mujoco_target_binding_preflighted"
        else:
            self.composition.close()
            status = "mujoco_target_closed"
        return self._build_result(
            intent,
            publication_enabled=False,
            evidence_status=status,
        )


class MujocoHumanStationSourceAdapter(StationAuthoritativeTargetAdapterBase):
    """Bind Station source lifecycle to an explicit publication gate."""

    def __init__(
        self,
        *,
        composition: MujocoHumanOnlyComposition,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        super().__init__(
            adapter_id="cs2a2b-synthetic-human-source-adapter",
            subject_id=composition.source_spec.source_id,
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
        )
    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        self._validate_intent(intent)
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached

        kind = intent.intent_kind
        lower_event_ids: tuple[str, ...] = ()
        if kind == "connect_source":
            status = "synthetic_human_source_connected"
        elif kind == "disconnect_source":
            status = "synthetic_human_source_disconnected"
        elif kind == "run_preflight":
            self._validate_live_binding(require_started=True)
            status = "synthetic_human_source_binding_preflighted"
        elif kind == "request_candidate":
            self._validate_live_binding(require_started=True)
            status = "synthetic_human_candidate_bound"
        elif kind == "release_candidate":
            status = "synthetic_human_candidate_released"
        elif kind == "request_motion_authority":
            self._validate_live_binding(require_started=True)
            gate = capture_authoritative_execution_gate_snapshot(self.composition)
            if gate.status != "ready":
                return self._build_failure(
                    intent,
                    status=f"execution_gate_not_ready:{gate.status}",
                )
            # This validates binding to the existing human-only core.  The
            # publication barrier opens only after receiver admission of this
            # typed completion.
            status = "existing_single_source_authority_bound"
        else:
            outcome = self.composition.runtime.transition_execution_gate(
                "hold",
                reason="control_station_release_motion_authority",
                source_event_ids=(intent.intent_id,),
            )
            lower_event_ids = outcome.lower_event_ids
            status = "station_publication_closed_and_execution_held"
        return self._build_result(
            intent,
            publication_enabled=False,
            lower_event_ids=lower_event_ids,
            evidence_status=status,
        )


class ScopedMujocoHumanStationComposition:
    """CS2a2b test composition with typed Station-gated publication.

    The synchronous operation seam is intentionally scoped to deterministic
    tests.  A production web binding needs a side-effect-aware asynchronous
    coordinator and an atomic publication barrier.
    """

    def __init__(
        self,
        *,
        station: ControlStationRuntime,
        data_plane: MujocoHumanOnlyComposition,
        clock: Callable[[], float],
        clock_domain_id: str = "control-station-mujoco-monotonic",
    ) -> None:
        if not isinstance(station, ControlStationRuntime):
            raise TypeError("station must be a ControlStationRuntime.")
        if not isinstance(data_plane, MujocoHumanOnlyComposition):
            raise TypeError("data_plane must be a MujocoHumanOnlyComposition.")
        if station.manifest.semantic_content_hash != data_plane.manifest.semantic_content_hash:
            raise ValueError("Station and MuJoCo data plane must share one manifest.")
        if (
            station.state.runtime_session_id
            != data_plane.runtime.config.runtime_session_id
            or station.state.control_domain_id
            != data_plane.runtime.config.control_domain_id
        ):
            raise ValueError("Station and MuJoCo data plane runtime scope differs.")
        if not data_plane.binding_receipt.station_authority_lifecycle_bound:
            raise ValueError("Data plane lacks the CS2a2b Station lifecycle binding.")
        self._station = station
        self._data_plane = data_plane
        self.target_adapter: StationOperationAdapter = MujocoHumanStationTargetAdapter(
            composition=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.source_adapter = MujocoHumanStationSourceAdapter(
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
    def data_plane(self) -> MujocoHumanOnlyComposition:
        return self._data_plane

    def authority_status_payload(self) -> dict[str, object]:
        """Return a rebuildable UI projection, never an authority fact."""

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
            "status_kind": "scoped_sim_station_publication",
            "motion_authorized": active,
            "publication_enabled": barrier.enabled,
            "publication_generation": barrier.generation,
            "publication_snapshot_hash": barrier.semantic_content_hash,
            "authority_operation_result_id": (
                barrier.authority_operation_result_id
            ),
            "station_lifecycle_state": state.lifecycle_state,
            "source_slot_id": self.source_adapter.subject_id,
            "source_runtime_state": source_state,
            "owner_token_id": receiver.current_owner_token_id,
            "authority_epoch": receiver.current_authority_epoch,
            "execution_epoch": receiver.current_execution_epoch,
            "authoritative": False,
        }

    @property
    def barrier_observer_failure_count(self) -> int:
        return self._barrier_observer_failure_count

    def bind_publication_barrier_observer(
        self,
        observer: Callable[..., object],
    ) -> None:
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

    def prepare_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> None:
        """Close publication before release work becomes worker-visible.

        This hook is no-fail and idempotent.  An asynchronous coordinator calls
        it after receiver admission of the request and before queue publication.
        """

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
        subject = transition.intent.subject_id
        adapter: StationOperationAdapter
        self.prepare_pending_operation(transition)
        if subject == self._station.manifest.execution_binding.binding_id:
            adapter = self.target_adapter
        elif subject == self.source_adapter.subject_id:
            adapter = self.source_adapter
        else:
            raise ValueError("Pending operation has no scoped MuJoCo adapter.")
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
                self._observe_barrier_transition(
                    snapshot,
                    causal_station_fact_id=completion.result.operation_result_id,
                    occurred_at=completion.result.available_at,
                    clock_domain_id=completion.result.available_clock_domain_id,
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
        emission: HumanCommandEmission,
        *,
        effective_from: float,
        effective_until: float,
        target_effective_at: float,
    ) -> MujocoHumanSubmission:
        initial_state = self._station.state
        if (
            initial_state.lifecycle_state != "running"
            or initial_state.source_state_by_slot[self.source_adapter.subject_id]
            != "active"
        ):
            raise RuntimeError(
                "Station motion authority is not active for this source."
            )

        def submit() -> MujocoHumanSubmission:
            state = self._station.state
            source_state = state.source_state_by_slot[self.source_adapter.subject_id]
            if state.lifecycle_state != "running" or source_state != "active":
                raise RuntimeError(
                    "Station motion authority is not active for this source."
                )
            return self._data_plane.submit_emission(
                emission,
                effective_from=effective_from,
                effective_until=effective_until,
                target_effective_at=target_effective_at,
            )

        return self.publication_barrier.execute_if_enabled(submit)

    def step(self, *, now: float | None = None, physics_ticks_after_dispatch: int = 0):
        return self._data_plane.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        self.publication_barrier.disable(reason="composition_closed")
        self._data_plane.close()


def build_scoped_cs2a2b_mujoco_human_station(
    runtime: MujocoDeployRuntime,
    source_spec: SyntheticHumanQposSourceSpec,
    *,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoHumanStationComposition:
    """Wire the real contracts into a deterministic, entrypoint-free fixture."""

    adapter = build_mujoco_human_proposal_adapter(runtime)
    manifest = build_cs2a2b_mujoco_human_station_manifest(
        runtime,
        adapter,
        source_spec,
    )
    data_plane = MujocoHumanOnlyComposition(
        manifest=manifest,
        runtime=runtime,
        adapter=adapter,
        source_spec=source_spec,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoHumanStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
    )


__all__ = [
    "MujocoHumanStationSourceAdapter",
    "MujocoHumanStationTargetAdapter",
    "MujocoStationOperationEvidence",
    "ScopedMujocoHumanStationComposition",
    "build_scoped_cs2a2b_mujoco_human_station",
]
