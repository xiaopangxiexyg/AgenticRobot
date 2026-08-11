"""Activation-scoped Sim-HIL cycle port for allocation-aware composition.

The port is deliberately thinner than a source or composition adapter.  A
profile-owned binding supplies the already-configured source admission,
allocation-context publication, and exact proposal publication operations for
one composition activation.  This port only sequences that publication before
the one lower execution step and fences the binding on application lifecycle
changes.

In particular, a composition ``no_action`` result is still passed to the lower
runtime by calling ``execute_lower``.  Station does not reinterpret it as HOLD,
fallback, safety, or direct human authority.
"""

from __future__ import annotations

import threading
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from control_composition.activation import CompositionActivationScope
from robot_motion.action import AuthoritativeControlStep
from robot_motion.deploy import PolicyActionEmission

from .control_cycle import (
    SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT,
    SIM_HIL_POLICY_SOURCE_ALWAYS,
    SimHILControlCyclePortSpec,
)
from .validation import _required


SIM_HIL_CONTROL_CYCLE_ALLOCATION_PROPOSAL_EXECUTION = "allocation_proposal_execution"


@dataclass(frozen=True)
class SimHILAllocationCyclePublication:
    """Facts produced before lower execution for one application cycle.

    The fields intentionally retain their owner-specific typed values.  This
    Station envelope does not copy their payloads or introduce a second
    identity domain.
    """

    proposal_publication: Any
    observed_at: float
    human_step: Any | None = None
    human_submission: Any | None = None
    policy_submission: Any | None = None
    input_publication: Any | None = None
    allocation_publication: Any | None = None

    def __post_init__(self) -> None:
        if self.proposal_publication is None:
            raise ValueError("proposal_publication must not be None.")
        status = str(getattr(self.proposal_publication, "status", "")).strip()
        if not status:
            raise TypeError("proposal_publication must expose a non-empty status.")
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be finite.")
        object.__setattr__(self, "observed_at", observed_at)


@dataclass(frozen=True)
class SimHILAllocationActivationPrime:
    """Source/context facts prepared after the activation barrier."""

    observed_at: float
    human_step: Any | None = None
    human_submission: Any | None = None
    policy_submission: Any | None = None
    input_publication: Any | None = None
    allocation_publication: Any | None = None

    def __post_init__(self) -> None:
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be finite.")
        object.__setattr__(self, "observed_at", observed_at)


@runtime_checkable
class SimHILAllocationActivationBinding(Protocol):
    """Profile-owned operations for exactly one composition activation."""

    @property
    def activation_scope(self) -> CompositionActivationScope: ...

    def publish_cycle(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILAllocationCyclePublication: ...

    def prime_sources(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        human_emission: Any,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILAllocationActivationPrime: ...

    def execute_lower(self, *, now: float) -> AuthoritativeControlStep: ...

    def fence(self, *, reason: str) -> None: ...

    def join(self, *, timeout_sec: float) -> bool: ...


SimHILAllocationBindingFactory = Callable[
    [CompositionActivationScope],
    SimHILAllocationActivationBinding,
]


@dataclass(frozen=True)
class SimHILAllocationControlCycleResult:
    """Application-compatible result without duplicating lower-owned facts."""

    cycle_index: int
    publication: SimHILAllocationCyclePublication
    execution: AuthoritativeControlStep

    def __post_init__(self) -> None:
        cycle_index = self.cycle_index
        if (
            isinstance(cycle_index, bool)
            or not isinstance(cycle_index, int)
            or cycle_index < 0
        ):
            raise ValueError("cycle_index must be a non-negative integer.")
        if not isinstance(
            self.publication,
            SimHILAllocationCyclePublication,
        ):
            raise TypeError("publication must be typed.")
        if not isinstance(self.execution, AuthoritativeControlStep):
            raise TypeError("execution must be AuthoritativeControlStep.")

    @property
    def human_submission(self) -> Any | None:
        return self.publication.human_submission

    @property
    def human_step(self) -> Any | None:
        return self.publication.human_step

    @property
    def policy_submission(self) -> Any | None:
        return self.publication.policy_submission

    @property
    def composition_step(self) -> Any:
        return self.publication.proposal_publication


class AllocationProposalExecutionControlCyclePort:
    """Select allocation-aware publication for one Sim-HIL activation.

    The binding factory is lazy: it is not evaluated until the application has
    completed composition activation. The port then owns profile-specific
    fresh-input priming. Deactivation fences the exact binding before the
    application requests lower HOLD; terminal cleanup joins it afterwards.
    """

    def __init__(
        self,
        *,
        port_id: str,
        binding_factory: SimHILAllocationBindingFactory,
        activation_human_emission_required: bool = True,
    ) -> None:
        if not callable(binding_factory):
            raise TypeError("binding_factory must be callable.")
        if not isinstance(activation_human_emission_required, bool):
            raise TypeError(
                "activation_human_emission_required must be bool."
            )
        self._spec = SimHILControlCyclePortSpec(
            port_id=_required(port_id, name="port_id"),
            cycle_kind=SIM_HIL_CONTROL_CYCLE_ALLOCATION_PROPOSAL_EXECUTION,
            policy_source_requirement=SIM_HIL_POLICY_SOURCE_ALWAYS,
            application_blend_failure_fallback_enabled=False,
            application_human_timeout_fallback_enabled=False,
            activation_priming_owner=(SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT),
            activation_human_emission_required=(
                activation_human_emission_required
            ),
            application_human_blend_control_enabled=False,
            application_owns_human_source_close=True,
        )
        self._binding_factory = binding_factory
        self._lock = threading.RLock()
        self._binding: SimHILAllocationActivationBinding | None = None
        self._retired_bindings: list[SimHILAllocationActivationBinding] = []
        self._active = False
        self._closed = False
        self._cycle_index = 0
        self._last_prime: SimHILAllocationActivationPrime | None = None

    @property
    def spec(self) -> SimHILControlCyclePortSpec:
        return self._spec

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active and not self._closed

    def activate(
        self,
        *,
        activation_scope: CompositionActivationScope | None,
    ) -> None:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError(
                "Allocation control cycle requires a typed activation scope."
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("Control-cycle port is closed.")
            if self._active or self._binding is not None:
                raise RuntimeError(
                    "Control-cycle port must deactivate before activation."
                )
            binding = self._binding_factory(activation_scope)
            if not isinstance(binding, SimHILAllocationActivationBinding):
                raise TypeError(
                    "binding_factory must return an allocation activation binding."
                )
            bound_scope = binding.activation_scope
            if (
                bound_scope.activation_id != activation_scope.activation_id
                or bound_scope.semantic_content_hash
                != activation_scope.semantic_content_hash
            ):
                try:
                    binding.fence(reason="activation_scope_mismatch")
                finally:
                    binding.join(timeout_sec=2.0)
                raise ValueError("Allocation binding belongs to another activation.")
            self._binding = binding
            self._cycle_index = 0
            self._last_prime = None
            self._active = True

    def prime_activation(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        human_emission: Any,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILAllocationActivationPrime:
        with self._lock:
            if self._closed:
                raise RuntimeError("Control-cycle port is closed.")
            if not self._active or self._binding is None:
                raise RuntimeError("Control-cycle port is not active.")
            if self._last_prime is not None:
                raise RuntimeError("Control-cycle activation is already primed.")
            prime = self._binding.prime_sources(
                now=now,
                policy_emission=policy_emission,
                human_emission=human_emission,
                poll_human=bool(poll_human),
                target_effective_at=target_effective_at,
            )
            if not isinstance(prime, SimHILAllocationActivationPrime):
                raise TypeError("Allocation binding returned an invalid prime.")
            self._last_prime = prime
            return prime

    def deactivate(self, *, reason: str) -> None:
        normalized_reason = _required(reason, name="reason")
        with self._lock:
            binding = self._binding
            self._binding = None
            self._active = False
            self._last_prime = None
            if binding is not None:
                # Fence publication immediately; joining an in-flight context
                # callback must not delay the application's lower HOLD.
                try:
                    binding.fence(reason=normalized_reason)
                finally:
                    self._retired_bindings.append(binding)

    def run_cycle(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILAllocationControlCycleResult:
        with self._lock:
            if self._closed:
                raise RuntimeError("Control-cycle port is closed.")
            if not self._active or self._binding is None:
                raise RuntimeError("Control-cycle port is not active.")
            if self._last_prime is None:
                raise RuntimeError("Control-cycle activation is not primed.")
            publication = self._binding.publish_cycle(
                now=now,
                policy_emission=policy_emission,
                poll_human=bool(poll_human),
                target_effective_at=target_effective_at,
            )
            if not isinstance(publication, SimHILAllocationCyclePublication):
                raise TypeError("Allocation binding returned an invalid publication.")
            # Always advance the one lower owner.  Its active proposal/target
            # persistence and execution gate decide the physical response to
            # no_action, expiry, rejection, or a newly admitted proposal.
            execution = self._binding.execute_lower(
                now=max(float(now), publication.observed_at)
            )
            result = SimHILAllocationControlCycleResult(
                cycle_index=self._cycle_index,
                publication=publication,
                execution=execution,
            )
            self._cycle_index += 1
            return result

    def submit_operation_trigger(self, **kwargs: Any) -> Any:
        """Route one admitted application event to the active profile owner."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Control-cycle port is closed.")
            if not self._active or self._binding is None:
                raise RuntimeError("Control-cycle port is not active.")
            binding = self._binding
        submit = getattr(binding, "submit_operation_trigger", None)
        if not callable(submit):
            raise RuntimeError("Selected allocation profile has no operation port.")
        return submit(**kwargs)

    def operation_status(self) -> dict[str, object]:
        with self._lock:
            binding = self._binding
            active = self._active and not self._closed
        if not active or binding is None:
            return {"supported_signals": [], "active": False}
        status = getattr(binding, "operation_status", None)
        if not callable(status):
            return {"supported_signals": [], "active": True}
        return {"active": True, **dict(status())}

    def close(self) -> None:
        retired: tuple[SimHILAllocationActivationBinding, ...]
        fence_failure: str | None = None
        with self._lock:
            if self._closed:
                return
            binding = self._binding
            self._binding = None
            self._active = False
            self._last_prime = None
            self._closed = True
            if binding is not None:
                self._retired_bindings.append(binding)
                try:
                    binding.fence(reason="sim_hil_control_cycle_closed")
                except Exception as error:  # noqa: BLE001 - still join resources.
                    fence_failure = f"{type(error).__name__}:{error}"
            retired = tuple(self._retired_bindings)
            self._retired_bindings.clear()
        # Application close calls this after lower HOLD. Joining here preserves
        # resource ownership without delaying the safety-relevant transition.
        join_failures: list[str] = []
        if fence_failure is not None:
            join_failures.append(f"allocation_binding_fence_failed:{fence_failure}")
        for retired_binding in retired:
            try:
                joined = retired_binding.join(timeout_sec=2.0)
            except Exception as error:  # noqa: BLE001 - join every binding.
                join_failures.append(f"{type(error).__name__}:{error}")
            else:
                if not joined:
                    join_failures.append("allocation_binding_join_timeout")
        if join_failures:
            raise RuntimeError(";".join(join_failures))


__all__ = [
    "SIM_HIL_CONTROL_CYCLE_ALLOCATION_PROPOSAL_EXECUTION",
    "AllocationProposalExecutionControlCyclePort",
    "SimHILAllocationActivationPrime",
    "SimHILAllocationActivationBinding",
    "SimHILAllocationBindingFactory",
    "SimHILAllocationControlCycleResult",
    "SimHILAllocationCyclePublication",
]
