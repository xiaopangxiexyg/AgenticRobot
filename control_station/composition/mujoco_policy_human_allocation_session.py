"""Application lifecycle adapter for allocation-aware MuJoCo Sim-HIL.

The adapter starts one existing lower runtime and creates activation-local
composition identity.  It never manufactures Station authority: callers must
inject typed start/rearm operation-result providers, and publication opens only
after those results carry the exact activation, preflight, and producer
evidence assembled here.

Source startup/close remains owned by :class:`MujocoSimHILApplication`; source
polling and proposal publication remain owned by the selected control-cycle
binding.  This object owns no policy worker, device, composition algorithm,
canonical safety, target resolution, IK, dispatch, or episode lifecycle.
"""

from __future__ import annotations

import threading
from typing import Any

from control_composition.activation import (
    CompositionActivationScope,
    build_composition_activation_scope,
)
from control_composition.lifecycle import CompositionSessionOperationSpec
from control_station.publication_barrier import StationPublicationBarrier
from robot_motion.runtime import (
    LifecycleStartupError,
    collect_lifecycle_failures,
    raise_lifecycle_failures,
)

from .mujoco_policy_human_allocation import (
    MujocoPolicyHumanAllocationProfileBundle,
)
from .mujoco_policy_human_allocation_lifecycle import (
    AllocationSessionResultProvider,
    MujocoPolicyHumanAllocationActivationEvidence,
    validate_allocation_session_operation_result,
)
from .mujoco_policy_human_allocation_preflight import (
    allocation_required_readiness_fields,
    build_mujoco_policy_human_allocation_preflight,
)
from .mujoco_sim_hil import (
    MujocoSimHILPreflightReport,
    MujocoSimHILReadinessEvidence,
)


class MujocoPolicyHumanAllocationSession:
    """Compatibility surface consumed by ``MujocoSimHILApplication``."""

    def __init__(
        self,
        *,
        runtime: Any,
        bundle: MujocoPolicyHumanAllocationProfileBundle,
        binding_factory: Any,
        operation_spec: CompositionSessionOperationSpec,
        readiness_provider: Any,
        activation_observer: Any,
        start_result_provider: AllocationSessionResultProvider,
        rearm_result_provider: AllocationSessionResultProvider,
        first_session_epoch: int = 1,
    ) -> None:
        if not isinstance(bundle, MujocoPolicyHumanAllocationProfileBundle):
            raise TypeError("bundle must be a typed allocation profile bundle.")
        for name in (
            "runtime",
            "bundle",
            "publication_barrier",
            "policy_input_adapter",
            "human_source",
        ):
            if not hasattr(binding_factory, name):
                raise TypeError(f"binding_factory must expose {name}.")
        if not callable(getattr(binding_factory, "required_producer_epochs", None)):
            raise TypeError("binding_factory must expose required_producer_epochs().")
        if binding_factory.runtime is not runtime or binding_factory.bundle is not bundle:
            raise ValueError("Binding factory belongs to another runtime/profile.")
        if not isinstance(operation_spec, CompositionSessionOperationSpec):
            raise TypeError("operation_spec must be typed.")
        if not callable(getattr(readiness_provider, "snapshot", None)):
            raise TypeError("readiness_provider must expose snapshot(now=...).")
        if not callable(activation_observer):
            raise TypeError("activation_observer must be callable.")
        if not callable(start_result_provider) or not callable(rearm_result_provider):
            raise TypeError("Session operation-result providers must be callable.")
        epoch = int(first_session_epoch)
        if (
            isinstance(first_session_epoch, bool)
            or epoch != first_session_epoch
            or epoch < 1
        ):
            raise ValueError("first_session_epoch must be a positive integer.")
        contracts = bundle.composition_contracts
        barrier = binding_factory.publication_barrier
        if not isinstance(barrier, StationPublicationBarrier):
            raise TypeError("Binding factory publication barrier must be typed.")
        if barrier.source_slot_id != operation_spec.composer_source_slot_id:
            raise ValueError("Session composer and publication barrier differ.")
        if operation_spec.composition_spec_hash != (
            contracts.composition_spec.semantic_content_hash
        ):
            raise ValueError("Session operation spec binds another composition.")
        policy_slot = binding_factory.policy_input_adapter.binding.source_slot_id
        if policy_slot not in operation_spec.contributor_source_slot_ids:
            raise ValueError("Session contributors omit the policy source slot.")
        required_runtime_methods = (
            "start",
            "close",
            "activate_policy_node_producer",
            "capture_execution_gate_snapshot",
            "transition_execution_gate",
        )
        if any(not callable(getattr(runtime, name, None)) for name in required_runtime_methods):
            raise TypeError("runtime lacks the allocation session lifecycle port.")
        self.runtime = runtime
        self.bundle = bundle
        # Application status and target diagnostics consume a read-only
        # contract facade; the bundle forwards to the generic contracts.
        self.contracts = bundle
        self.binding_factory = binding_factory
        self.operation_spec = operation_spec
        self.readiness_provider = readiness_provider
        self.activation_observer = activation_observer
        self.start_result_provider = start_result_provider
        self.rearm_result_provider = rearm_result_provider
        self.publication_barrier = barrier
        self._next_session_epoch = epoch
        self._activation_scope: CompositionActivationScope | None = None
        self._started = False
        self._runtime_close_pending = False
        self._closing = False
        self._rearm_failed = False
        self._last_preflight: MujocoSimHILPreflightReport | None = None
        self._last_operation_result: Any | None = None
        self._lock = threading.RLock()

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def activation_scope(self) -> CompositionActivationScope | None:
        with self._lock:
            return self._activation_scope

    def required_readiness_checks(
        self,
        *,
        target_effective_at: float | None = None,
    ) -> tuple[str, ...]:
        del target_effective_at
        return allocation_required_readiness_fields(self.bundle)

    def readiness_failures(
        self,
        readiness: MujocoSimHILReadinessEvidence,
        *,
        target_effective_at: float | None = None,
    ) -> tuple[str, ...]:
        del target_effective_at
        if not isinstance(readiness, MujocoSimHILReadinessEvidence):
            raise TypeError("readiness must be typed.")
        required = set(self.required_readiness_checks())
        return tuple(
            name for name in readiness.failed_checks if name in required
        )

    def preflight(self, *, now: float | None = None) -> MujocoSimHILPreflightReport:
        timestamp = float(self.runtime.clock() if now is None else now)
        return build_mujoco_policy_human_allocation_preflight(
            runtime=self.runtime,
            bundle=self.bundle,
            binding_factory=self.binding_factory,
            readiness_provider=self.readiness_provider,
            now=timestamp,
        )

    def _build_activation(self, *, activated_at: float) -> CompositionActivationScope:
        return build_composition_activation_scope(
            operation_spec_id=self.operation_spec.operation_spec_id,
            operation_spec_hash=self.operation_spec.semantic_content_hash,
            session_epoch=self._next_session_epoch,
            activated_at=float(activated_at),
            clock_domain_id=self.runtime.config.clock_domain_id,
            input_barrier_at=float(activated_at),
            required_roles=(
                self.bundle.composition_contracts.composition_spec.required_roles
            ),
        )

    def _activate_input_producers(
        self,
        scope: CompositionActivationScope,
    ) -> tuple[Any, ...]:
        return tuple(
            self.runtime.activate_policy_node_producer(
                producer_id,
                producer_epoch=producer_epoch,
                reason=(
                    "control-station allocation composition activation "
                    f"{scope.session_epoch}"
                ),
            )
            for producer_id, producer_epoch in sorted(
                self.binding_factory.required_producer_epochs(scope).items()
            )
        )

    def _validate_operation_result(self, result: Any, evidence: Any) -> None:
        config = self.runtime.config
        validate_allocation_session_operation_result(
            result,
            evidence,
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            clock_domain_id=config.clock_domain_id,
            composer_source_slot_id=self.operation_spec.composer_source_slot_id,
        )

    def _enable_publication(
        self,
        result: Any,
        *,
        rearm: bool,
    ) -> None:
        if rearm:
            self.publication_barrier.enable_from_composition_session_rearm_result(
                result
            )
        else:
            self.publication_barrier.enable_from_composition_session_result(result)

    def start(self) -> MujocoSimHILPreflightReport:
        with self._lock:
            if self._closing:
                raise RuntimeError(
                    "Allocation composition session cleanup is pending."
                )
            if self._started:
                assert self._last_preflight is not None
                return self._last_preflight
            report = self.preflight()
            if not report.passed:
                raise RuntimeError(
                    "MuJoCo allocation Sim-HIL preflight failed: "
                    + ";".join(report.reasons)
                )
            try:
                # A failed lower start may itself retain retryable cleanup.
                # Record ownership before entering it so close() remains a
                # valid recovery operation in either case.
                self._runtime_close_pending = True
                self.runtime.start()
                activated_at = float(self.runtime.clock())
                scope = self._build_activation(activated_at=activated_at)
                producer_activations = self._activate_input_producers(scope)
                gate_snapshot = self.runtime.capture_execution_gate_snapshot(
                    now=float(self.runtime.clock())
                )
                if gate_snapshot.status != "ready":
                    raise RuntimeError(
                        "Allocation session start requires lower execution READY."
                    )
                self.activation_observer(scope)
                evidence = MujocoPolicyHumanAllocationActivationEvidence(
                    operation_kind="start_composition_session",
                    operation_spec=self.operation_spec,
                    activation_scope=scope,
                    preflight_report=report,
                    producer_activations=producer_activations,
                    execution_gate_snapshot=gate_snapshot,
                )
                result = self.start_result_provider(evidence)
                self._validate_operation_result(result, evidence)
                self._enable_publication(result, rearm=False)
                self._activation_scope = scope
                self._next_session_epoch += 1
                self._last_preflight = report
                self._last_operation_result = result
                self._started = True
                return report
            except BaseException as start_error:  # noqa: BLE001 - preserve rollback.
                self.publication_barrier.disable(reason="allocation_session_start_failed")
                self._activation_scope = None
                rollback_failures = collect_lifecycle_failures(
                    (("mujoco_runtime.close", self._close_runtime_once),)
                )
                self._closing = bool(self._runtime_close_pending)
                if rollback_failures:
                    raise LifecycleStartupError(
                        "MujocoPolicyHumanAllocationSession.start",
                        start_error,
                        rollback_failures,
                    ) from start_error
                raise

    def request_hold(
        self,
        *,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        force_cancellation: bool = False,
    ) -> Any | None:
        with self._lock:
            if not self._started or self._closing:
                raise RuntimeError("Allocation composition session is not started.")
            normalized_reason = str(reason).strip()
            if not normalized_reason:
                raise ValueError("reason must be non-empty.")
            self.publication_barrier.disable(reason=normalized_reason)
            now = float(self.runtime.clock())
            gate = self.runtime.capture_execution_gate_snapshot(now=now)
            if gate.status == "fault":
                return None
            if gate.status == "hold" and not force_cancellation:
                return None
            return self.runtime.transition_execution_gate(
                "hold",
                reason=normalized_reason,
                source_event_ids=tuple(source_event_ids),
                force_cancellation=bool(force_cancellation),
            )

    def rearm(self, *, now: float | None = None) -> MujocoSimHILPreflightReport:
        with self._lock:
            if (
                not self._started
                or self._closing
                or self._activation_scope is None
            ):
                raise RuntimeError("Allocation composition session is not started.")
            if self._rearm_failed:
                raise RuntimeError(
                    "A failed allocation rearm requires application rebuild."
                )
            timestamp = float(self.runtime.clock() if now is None else now)
            report = self.preflight(now=timestamp)
            if not report.passed:
                self.request_hold(
                    reason="allocation_session:rearm_readiness_not_ready"
                )
                raise RuntimeError(
                    "MuJoCo allocation Sim-HIL rearm preflight failed: "
                    + ";".join(report.reasons)
                )
            gate = self.runtime.capture_execution_gate_snapshot(now=timestamp)
            if gate.status != "hold":
                raise RuntimeError(
                    "Allocation rearm requires the lower execution gate in HOLD."
                )
            prior_scope = self._activation_scope
            scope = self._build_activation(activated_at=timestamp)
            try:
                producer_activations = self._activate_input_producers(scope)
                gate_transition = self.runtime.transition_execution_gate(
                    "ready",
                    reason="allocation_session:explicit_rearm",
                    source_event_ids=(scope.activation_id,),
                )
                self.activation_observer(scope)
                evidence = MujocoPolicyHumanAllocationActivationEvidence(
                    operation_kind="rearm_composition_session",
                    operation_spec=self.operation_spec,
                    activation_scope=scope,
                    prior_activation_scope=prior_scope,
                    preflight_report=report,
                    producer_activations=producer_activations,
                    execution_gate_snapshot=gate_transition.snapshot,
                    execution_gate_transition=gate_transition,
                )
                result = self.rearm_result_provider(evidence)
                self._validate_operation_result(result, evidence)
                self._enable_publication(result, rearm=True)
            except Exception:
                self.publication_barrier.disable(reason="allocation_session_rearm_failed")
                try:
                    self.runtime.transition_execution_gate(
                        "hold",
                        reason="allocation_session:rearm_failed",
                        source_event_ids=(scope.activation_id,),
                        force_cancellation=True,
                    )
                finally:
                    self._rearm_failed = True
                raise
            self._activation_scope = scope
            self._next_session_epoch += 1
            self._last_preflight = report
            self._last_operation_result = result
            return report

    def _close_runtime_once(self) -> None:
        if not self._runtime_close_pending:
            return
        self.runtime.close()
        self._runtime_close_pending = False

    def close(self) -> None:
        with self._lock:
            if not self._started and not self._runtime_close_pending:
                return
            self._closing = True
            self.publication_barrier.disable(reason="allocation_session_closed")
            failures = collect_lifecycle_failures(
                (("mujoco_runtime.close", self._close_runtime_once),)
            )
            if not self._runtime_close_pending:
                self._started = False
                self._activation_scope = None
                self._closing = False
            raise_lifecycle_failures(
                "MujocoPolicyHumanAllocationSession.close",
                failures,
            )


__all__ = [
    "MujocoPolicyHumanAllocationSession",
]
