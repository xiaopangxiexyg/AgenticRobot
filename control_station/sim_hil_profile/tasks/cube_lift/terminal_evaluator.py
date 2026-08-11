"""Measured CubeLift terminal evaluation under the existing lower HOLD.

The evaluator advances only the already-owned deterministic MuJoCo rig after
Station has fenced publication and the lower runtime has acknowledged a forced
cancellation/HOLD.  It never submits a proposal, enters the Arbiter, resolves a
target, dispatches a new target, or calls the task kernel's private success
counter.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from control_station.agent_gateway.protocol.contracts import (
    AgentSkillOperation,
    AgentSkillRequest,
)
from control_station.agent_gateway.protocol.evaluation_contracts import (
    SkillAggregateEvaluationReceipt,
    SkillEvaluationPlan,
    SkillEvaluationReceipt,
)
from control_station.agent_gateway.protocol.skill_catalog import SkillExecutionRoute
from control_station.tasks.cube_lift.definition import (
    CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1,
)
from control_station.tasks.cube_lift.sim_route import (
    CUBE_LIFT_AGGREGATION_POLICY_HASH_V1,
    CUBE_LIFT_AGGREGATION_POLICY_ID_V1,
    CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1,
    CUBE_LIFT_EVALUATOR_EVIDENCE_SCHEMA_V1,
    CubeLiftSimDeploymentBinding,
    CubeLiftTerminalEvaluatorConfig,
    cube_lift_evaluator_input_schema_hash_v1,
    validate_cube_lift_sim_route_v1,
    validate_cube_lift_task_evaluation_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

CUBE_LIFT_LOWER_HOLD_PROOF_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_lower_hold_proof.v1"
)
CUBE_LIFT_TERMINAL_SAMPLE_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_terminal_sample.v1"
)
CUBE_LIFT_TERMINAL_EVIDENCE_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_terminal_evidence.v1"
)
CUBE_LIFT_TERMINAL_CUT_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.cube_lift_terminal_cut.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _refs(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(
        sorted({_required(value, field_name=f"{field_name} item") for value in values})
    )
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


@dataclass(frozen=True)
class CubeLiftLowerHoldProof:
    """Exact lower gate/cancellation evidence required before physics sampling."""

    runtime_session_id: str
    control_domain_id: str
    hold_receipt_ref: str
    gate_snapshot_hash: str
    gate_epoch: int
    cancellation_receipt_ref: str
    confirmed_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "hold_receipt_ref",
            "gate_snapshot_hash",
            "cancellation_receipt_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "gate_epoch",
            _nonnegative_integer(self.gate_epoch, field_name="gate_epoch"),
        )
        object.__setattr__(
            self,
            "confirmed_at",
            _finite(self.confirmed_at, field_name="confirmed_at"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _refs(self.evidence_refs, field_name="evidence_refs"),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=CUBE_LIFT_LOWER_HOLD_PROOF_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "hold_receipt_ref": self.hold_receipt_ref,
            "gate_snapshot_hash": self.gate_snapshot_hash,
            "gate_epoch": self.gate_epoch,
            "cancellation_receipt_ref": self.cancellation_receipt_ref,
            "confirmed_at": self.confirmed_at,
            "evidence_refs": self.evidence_refs,
        }
        if include_hash:
            payload["lower_hold_proof_hash"] = self.semantic_content_hash
        return payload


def cube_lift_lower_hold_proof_from_transition(
    transition: Any,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    confirmed_at: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CubeLiftLowerHoldProof:
    """Project one forced lower HOLD transition without inferring evidence."""

    event = getattr(transition, "event", None)
    snapshot = getattr(transition, "snapshot", None)
    cancellation = getattr(transition, "cancellation", None)
    receipt = getattr(transition, "cancellation_receipt", None)
    if event is None or snapshot is None or cancellation is None or receipt is None:
        raise ValueError(
            "Terminal HOLD requires an explicit gate event and cancellation receipt."
        )
    session_id = _required(runtime_session_id, field_name="runtime_session_id")
    domain_id = _required(control_domain_id, field_name="control_domain_id")
    if getattr(event, "runtime_session_id", None) != session_id:
        raise ValueError("HOLD event runtime session mismatch.")
    if getattr(event, "control_domain_id", None) != domain_id:
        raise ValueError("HOLD event control domain mismatch.")
    if getattr(snapshot, "control_domain_id", None) != domain_id:
        raise ValueError("HOLD snapshot control domain mismatch.")
    if getattr(snapshot, "status", None) != "hold":
        raise ValueError("Terminal evidence requires lower gate status HOLD.")
    if (
        not bool(getattr(receipt, "complete", False))
        or bool(getattr(receipt, "timed_out", False))
        or bool(getattr(receipt, "requires_fault_escalation", False))
        or tuple(getattr(receipt, "pending_levels", ()))
        or tuple(getattr(receipt, "rejected_levels", ()))
    ):
        raise ValueError("Lower cancellation receipt does not prove completed HOLD.")
    cancellation_id = _required(
        getattr(cancellation, "cancellation_id", None),
        field_name="cancellation_id",
    )
    if getattr(receipt, "cancellation_id", None) != cancellation_id:
        raise ValueError("Cancellation receipt identity mismatch.")
    event_id = _required(getattr(event, "event_id", None), field_name="event_id")
    gate_hash = _required(
        getattr(snapshot, "semantic_content_hash", None),
        field_name="gate_snapshot_hash",
    )
    trace_refs = tuple(
        _required(getattr(fact, "fact_id", None), field_name="trace fact_id")
        for fact in tuple(getattr(transition, "trace_facts", ()))
    )
    receipt_payload = {
        "cancellation_id": cancellation_id,
        "required_ack_levels": tuple(getattr(receipt, "required_ack_levels", ())),
        "acknowledged_levels": tuple(getattr(receipt, "acknowledged_levels", ())),
        "pending_levels": tuple(getattr(receipt, "pending_levels", ())),
        "rejected_levels": tuple(getattr(receipt, "rejected_levels", ())),
        "acknowledgement_deadline_at": _finite(
            getattr(receipt, "acknowledgement_deadline_at", None),
            field_name="acknowledgement_deadline_at",
        ),
        "observed_at": _finite(
            getattr(receipt, "observed_at", None),
            field_name="receipt.observed_at",
        ),
        "complete": True,
        "timed_out": False,
        "requires_fault_escalation": False,
        "late_levels": tuple(getattr(receipt, "late_levels", ())),
    }
    receipt_hash = canonical_content_hash(
        receipt_payload,
        domain="dexteleop.sim_hil.lower_cancellation_receipt_ref.v1",
    )
    timestamp = _finite(
        clock() if confirmed_at is None else confirmed_at,
        field_name="confirmed_at",
    )
    proof_payload = {
        "runtime_session_id": session_id,
        "control_domain_id": domain_id,
        "event_id": event_id,
        "gate_snapshot_hash": gate_hash,
        "gate_epoch": _nonnegative_integer(
            getattr(snapshot, "gate_epoch", None),
            field_name="gate_epoch",
        ),
        "cancellation_id": cancellation_id,
        "cancellation_receipt_hash": receipt_hash,
        "confirmed_at": timestamp,
        "trace_refs": trace_refs,
    }
    hold_hash = canonical_content_hash(
        proof_payload,
        domain="dexteleop.sim_hil.lower_hold_receipt_ref.v1",
    )
    return CubeLiftLowerHoldProof(
        runtime_session_id=session_id,
        control_domain_id=domain_id,
        hold_receipt_ref=f"mujoco-lower-hold:{hold_hash}",
        gate_snapshot_hash=gate_hash,
        gate_epoch=proof_payload["gate_epoch"],
        cancellation_receipt_ref=(
            f"execution-cancellation-receipt:{receipt_hash}"
        ),
        confirmed_at=timestamp,
        evidence_refs=tuple(
            sorted(
                {
                    event_id,
                    cancellation_id,
                    gate_hash,
                    *trace_refs,
                    f"execution-cancellation-receipt:{receipt_hash}",
                }
            )
        ),
    )


@dataclass(frozen=True)
class CubeLiftTerminalHoldSample:
    physics_tick_sequence: int
    measured_at: float
    cube_center_z_m: float
    left_finger_cube_contact: bool
    right_finger_cube_contact: bool
    table_cube_contact: bool
    finite_state: bool
    source_measurement_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physics_tick_sequence",
            _nonnegative_integer(
                self.physics_tick_sequence,
                field_name="physics_tick_sequence",
            ),
        )
        object.__setattr__(
            self,
            "measured_at",
            _finite(self.measured_at, field_name="measured_at"),
        )
        object.__setattr__(
            self,
            "cube_center_z_m",
            _finite(self.cube_center_z_m, field_name="cube_center_z_m"),
        )
        for field_name in (
            "left_finger_cube_contact",
            "right_finger_cube_contact",
            "table_cube_contact",
            "finite_state",
        ):
            object.__setattr__(
                self,
                field_name,
                _bool(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "source_measurement_ref",
            _required(
                self.source_measurement_ref,
                field_name="source_measurement_ref",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=CUBE_LIFT_TERMINAL_SAMPLE_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "physics_tick_sequence": self.physics_tick_sequence,
            "measured_at": self.measured_at,
            "cube_center_z_m": self.cube_center_z_m,
            "left_finger_cube_contact": self.left_finger_cube_contact,
            "right_finger_cube_contact": self.right_finger_cube_contact,
            "table_cube_contact": self.table_cube_contact,
            "finite_state": self.finite_state,
            "source_measurement_ref": self.source_measurement_ref,
        }


@runtime_checkable
class CubeLiftTerminalHoldSampler(Protocol):
    runtime_session_id: str
    control_domain_id: str
    task_config_hash: str
    realtime_worker: bool

    def validate_hold(self, proof: CubeLiftLowerHoldProof) -> None: ...

    def step_and_capture(
        self,
        *,
        proof: CubeLiftLowerHoldProof,
    ) -> CubeLiftTerminalHoldSample: ...


class MujocoCubeLiftTerminalHoldSampler:
    """Use the existing deterministic rig's held physics step and scene lock."""

    def __init__(
        self,
        *,
        application: Any,
        task_config: Any,
        task_config_hash: str,
        evaluator_config: CubeLiftTerminalEvaluatorConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(evaluator_config, CubeLiftTerminalEvaluatorConfig):
            raise TypeError(
                "evaluator_config must be a CubeLiftTerminalEvaluatorConfig."
            )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        runtime = getattr(application, "runtime", None)
        rig = getattr(runtime, "rig", None)
        if runtime is None or rig is None:
            raise TypeError("application must expose the existing runtime and rig.")
        for method_name in ("step_physics", "capture_scene", "snapshot"):
            if not callable(getattr(rig, method_name, None)):
                raise TypeError(f"MuJoCo rig must expose {method_name}().")
        config = getattr(rig, "config", None)
        realtime_worker = getattr(config, "realtime_worker", None)
        if realtime_worker is not False:
            raise ValueError(
                "Terminal HOLD sampling requires deterministic realtime_worker=false."
            )
        runtime_config = getattr(runtime, "config", None)
        self.runtime_session_id = _required(
            getattr(runtime_config, "runtime_session_id", None),
            field_name="runtime_session_id",
        )
        self.control_domain_id = _required(
            getattr(runtime_config, "control_domain_id", None),
            field_name="control_domain_id",
        )
        self.task_config_hash = _required(
            task_config_hash,
            field_name="task_config_hash",
        )
        if self.task_config_hash != evaluator_config.task_config_hash:
            raise ValueError("Sampler and evaluator task config hashes differ.")
        expected_task_fields = {
            "object_body_name": evaluator_config.object_body_name,
            "grasp_latch_left_finger_body_name": (
                evaluator_config.left_finger_body_name
            ),
            "grasp_latch_right_finger_body_name": (
                evaluator_config.right_finger_body_name
            ),
            "table_body_name": evaluator_config.table_body_name,
        }
        for field_name, expected in expected_task_fields.items():
            if getattr(task_config, field_name, None) != expected:
                raise ValueError(
                    f"Task config {field_name} does not match evaluator config."
                )
        physics_hz = _finite(
            getattr(task_config, "physics_hz", None),
            field_name="task_config.physics_hz",
        )
        if abs(physics_hz - evaluator_config.physics_hz) > 1e-12:
            raise ValueError("Task and evaluator physics_hz differ.")
        self.realtime_worker = False
        self.application = application
        self.task_config = task_config
        self.evaluator_config = evaluator_config
        self.runtime = runtime
        self.rig = rig
        self.clock = clock
        self._lock = threading.Lock()
        self._last_physics_tick: int | None = None
        self.rig.capture_scene(self._validate_scene)

    def _validate_scene(self, scene: Any) -> None:
        for body_name in (
            self.evaluator_config.object_body_name,
            self.evaluator_config.left_finger_body_name,
            self.evaluator_config.right_finger_body_name,
            self.evaluator_config.table_body_name,
        ):
            if not bool(scene.has_body(body_name)):
                raise ValueError(f"MuJoCo scene is missing body {body_name!r}.")

    def validate_hold(self, proof: CubeLiftLowerHoldProof) -> None:
        if not isinstance(proof, CubeLiftLowerHoldProof):
            raise TypeError("proof must be a CubeLiftLowerHoldProof.")
        if proof.runtime_session_id != self.runtime_session_id:
            raise ValueError("Terminal HOLD proof runtime session mismatch.")
        if proof.control_domain_id != self.control_domain_id:
            raise ValueError("Terminal HOLD proof control domain mismatch.")
        if not bool(getattr(self.application, "started", False)):
            raise RuntimeError("Terminal HOLD sampling requires a started application.")
        if not bool(getattr(self.application, "held", False)):
            raise RuntimeError("Terminal HOLD sampling requires application HOLD.")
        if bool(getattr(self.application, "closed", False)):
            raise RuntimeError("Terminal HOLD sampling cannot use a closed application.")
        snapshot = self.runtime.capture_execution_gate_snapshot(now=self.clock())
        if getattr(snapshot, "status", None) != "hold":
            raise RuntimeError("Authoritative lower execution gate left HOLD.")
        if getattr(snapshot, "control_domain_id", None) != self.control_domain_id:
            raise RuntimeError("Lower gate snapshot changed control domain.")
        if int(getattr(snapshot, "gate_epoch", -1)) < proof.gate_epoch:
            raise RuntimeError("Lower gate snapshot predates the HOLD proof.")

    def step_and_capture(
        self,
        *,
        proof: CubeLiftLowerHoldProof,
    ) -> CubeLiftTerminalHoldSample:
        with self._lock:
            self.validate_hold(proof)
            before = self.rig.snapshot()
            before_tick = _nonnegative_integer(
                before.get("physics_ticks"),
                field_name="rig.physics_ticks",
            )
            if self._last_physics_tick is not None and before_tick != self._last_physics_tick:
                raise RuntimeError(
                    "Another physics owner advanced the deterministic HOLD rig."
                )
            self.rig.step_physics()
            after = self.rig.snapshot()
            physics_tick = _nonnegative_integer(
                after.get("physics_ticks"),
                field_name="rig.physics_ticks",
            )
            if physics_tick != before_tick + 1:
                raise RuntimeError("Held evaluator must advance exactly one physics tick.")
            measured_at = _finite(self.clock(), field_name="measured_at")
            captured = self.rig.capture_scene(self._capture_scene)
            self.validate_hold(proof)
            self._last_physics_tick = physics_tick
            payload = {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "physics_tick_sequence": physics_tick,
                "measured_at": measured_at,
                **captured,
                "lower_hold_proof_hash": proof.semantic_content_hash,
            }
            measurement_hash = canonical_content_hash(
                payload,
                domain="dexteleop.sim_hil.cube_lift_scene_measurement.v1",
            )
            return CubeLiftTerminalHoldSample(
                physics_tick_sequence=physics_tick,
                measured_at=measured_at,
                cube_center_z_m=captured["cube_center_z_m"],
                left_finger_cube_contact=captured["left_finger_cube_contact"],
                right_finger_cube_contact=captured["right_finger_cube_contact"],
                table_cube_contact=captured["table_cube_contact"],
                finite_state=captured["finite_state"],
                source_measurement_ref=(
                    f"mujoco-terminal-hold-measurement:{measurement_hash}"
                ),
            )

    def _capture_scene(self, scene: Any) -> dict[str, object]:
        finite_state = bool(scene.state_is_finite())
        pose = scene.body_pose(self.evaluator_config.object_body_name)
        cube_center_z_m = float(pose[2, 3])
        if not math.isfinite(cube_center_z_m):
            raise RuntimeError("Cube center became non-finite under terminal HOLD.")
        object_name = self.evaluator_config.object_body_name
        touching_bodies: set[str] = set()
        for contact in scene.contacts():
            first = str(contact.body1_name)
            second = str(contact.body2_name)
            if first == object_name:
                touching_bodies.add(second)
            elif second == object_name:
                touching_bodies.add(first)
        return {
            "cube_center_z_m": cube_center_z_m,
            "left_finger_cube_contact": (
                self.evaluator_config.left_finger_body_name in touching_bodies
            ),
            "right_finger_cube_contact": (
                self.evaluator_config.right_finger_body_name in touching_bodies
            ),
            "table_cube_contact": (
                self.evaluator_config.table_body_name in touching_bodies
            ),
            "finite_state": finite_state,
        }


@dataclass(frozen=True)
class CubeLiftTerminalHoldEvidence:
    runtime_session_id: str
    control_domain_id: str
    evaluated_cut_id: str
    baseline_resting_center_z_m: float
    terminal_hold_receipt_ref: str
    execution_evidence_chain_ref: str
    execution_lineage_complete: bool
    samples: tuple[CubeLiftTerminalHoldSample, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "evaluated_cut_id",
            "terminal_hold_receipt_ref",
            "execution_evidence_chain_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "baseline_resting_center_z_m",
            _finite(
                self.baseline_resting_center_z_m,
                field_name="baseline_resting_center_z_m",
            ),
        )
        object.__setattr__(
            self,
            "execution_lineage_complete",
            _bool(
                self.execution_lineage_complete,
                field_name="execution_lineage_complete",
            ),
        )
        samples = tuple(self.samples)
        if not samples or any(
            not isinstance(sample, CubeLiftTerminalHoldSample) for sample in samples
        ):
            raise TypeError("samples must contain terminal HOLD samples.")
        object.__setattr__(self, "samples", samples)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=CUBE_LIFT_TERMINAL_EVIDENCE_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": CUBE_LIFT_EVALUATOR_EVIDENCE_SCHEMA_V1,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "evaluated_cut_id": self.evaluated_cut_id,
            "baseline_resting_center_z_m": self.baseline_resting_center_z_m,
            "terminal_hold_receipt_ref": self.terminal_hold_receipt_ref,
            "execution_evidence_chain_ref": self.execution_evidence_chain_ref,
            "execution_lineage_complete": self.execution_lineage_complete,
            "samples": [sample.to_wire() for sample in self.samples],
        }


@dataclass(frozen=True)
class CubeLiftTerminalEvaluationBundle:
    evidence: CubeLiftTerminalHoldEvidence
    evaluation_receipt: SkillEvaluationReceipt
    aggregate_receipt: SkillAggregateEvaluationReceipt
    task_evaluation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, CubeLiftTerminalHoldEvidence):
            raise TypeError("evidence must be CubeLiftTerminalHoldEvidence.")
        if not isinstance(self.evaluation_receipt, SkillEvaluationReceipt):
            raise TypeError("evaluation_receipt must be SkillEvaluationReceipt.")
        if not isinstance(
            self.aggregate_receipt,
            SkillAggregateEvaluationReceipt,
        ):
            raise TypeError(
                "aggregate_receipt must be SkillAggregateEvaluationReceipt."
            )
        object.__setattr__(
            self,
            "task_evaluation",
            MappingProxyType(dict(self.task_evaluation)),
        )


class CubeLiftTerminalStableHoldEvaluator:
    """Collect and evaluate exactly one Route-bound held physics window."""

    def __init__(
        self,
        *,
        config: CubeLiftTerminalEvaluatorConfig,
        sampler: CubeLiftTerminalHoldSampler,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, CubeLiftTerminalEvaluatorConfig):
            raise TypeError("config must be a CubeLiftTerminalEvaluatorConfig.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        for method_name in ("validate_hold", "step_and_capture"):
            if not callable(getattr(sampler, method_name, None)):
                raise TypeError(f"sampler must expose {method_name}().")
        if bool(getattr(sampler, "realtime_worker", True)):
            raise ValueError("Terminal evaluator rejects a realtime physics worker.")
        if getattr(sampler, "task_config_hash", None) != config.task_config_hash:
            raise ValueError("Sampler and evaluator task config hashes differ.")
        self.config = config
        self.sampler = sampler
        self.clock = clock
        self._lock = threading.Lock()
        self._used = False

    def evaluate(
        self,
        *,
        operation: AgentSkillOperation,
        request: AgentSkillRequest,
        route: SkillExecutionRoute,
        deployment_binding: CubeLiftSimDeploymentBinding,
        evaluation_plan: SkillEvaluationPlan,
        lower_hold_proof: CubeLiftLowerHoldProof,
        baseline_resting_center_z_m: float,
        execution_evidence_chain_ref: str,
        terminal_state_hash: str,
        execution_lineage_complete: bool,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> CubeLiftTerminalEvaluationBundle:
        with self._lock:
            if self._used:
                raise RuntimeError("Terminal evaluator is single-use.")
            self._used = True
        self._validate_bindings(
            operation=operation,
            request=request,
            route=route,
            deployment_binding=deployment_binding,
            evaluation_plan=evaluation_plan,
            lower_hold_proof=lower_hold_proof,
        )
        baseline = _finite(
            baseline_resting_center_z_m,
            field_name="baseline_resting_center_z_m",
        )
        execution_chain = _required(
            execution_evidence_chain_ref,
            field_name="execution_evidence_chain_ref",
        )
        terminal_hash = _required(
            terminal_state_hash,
            field_name="terminal_state_hash",
        )
        lineage_complete = _bool(
            execution_lineage_complete,
            field_name="execution_lineage_complete",
        )
        self.sampler.validate_hold(lower_hold_proof)
        samples: list[CubeLiftTerminalHoldSample] = []
        previous_tick: int | None = None
        for _ in range(self.config.stable_hold_physics_ticks):
            if cancel_requested is not None and bool(cancel_requested()):
                raise InterruptedError("Terminal HOLD evaluation was cancelled.")
            sample = self.sampler.step_and_capture(proof=lower_hold_proof)
            if not isinstance(sample, CubeLiftTerminalHoldSample):
                raise TypeError("Sampler returned an invalid terminal sample.")
            if not sample.finite_state:
                raise RuntimeError("MuJoCo state became non-finite under lower HOLD.")
            if previous_tick is not None and sample.physics_tick_sequence != previous_tick + 1:
                raise RuntimeError("Terminal samples are not consecutive physics ticks.")
            previous_tick = sample.physics_tick_sequence
            samples.append(sample)
        sample_values = tuple(samples)
        cut_hash = canonical_content_hash(
            {
                "runtime_session_id": request.runtime_session_id,
                "control_domain_id": lower_hold_proof.control_domain_id,
                "lower_hold_proof_hash": lower_hold_proof.semantic_content_hash,
                "sample_hashes": tuple(
                    sample.semantic_content_hash for sample in sample_values
                ),
            },
            domain=CUBE_LIFT_TERMINAL_CUT_HASH_DOMAIN_V1,
        )
        evaluated_cut_id = f"cube-lift-terminal-cut:{cut_hash}"
        evidence = CubeLiftTerminalHoldEvidence(
            runtime_session_id=request.runtime_session_id,
            control_domain_id=lower_hold_proof.control_domain_id,
            evaluated_cut_id=evaluated_cut_id,
            baseline_resting_center_z_m=baseline,
            terminal_hold_receipt_ref=lower_hold_proof.hold_receipt_ref,
            execution_evidence_chain_ref=execution_chain,
            execution_lineage_complete=lineage_complete,
            samples=sample_values,
        )
        return self._build_bundle(
            operation=operation,
            request=request,
            route=route,
            deployment_binding=deployment_binding,
            evaluation_plan=evaluation_plan,
            evidence=evidence,
            terminal_state_hash=terminal_hash,
        )

    def _validate_bindings(
        self,
        *,
        operation: AgentSkillOperation,
        request: AgentSkillRequest,
        route: SkillExecutionRoute,
        deployment_binding: CubeLiftSimDeploymentBinding,
        evaluation_plan: SkillEvaluationPlan,
        lower_hold_proof: CubeLiftLowerHoldProof,
    ) -> None:
        if not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation.")
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if operation.status != "running":
            raise ValueError("Terminal evaluation requires a running operation.")
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("Operation and request hashes differ.")
        if operation.runtime_session_id != request.runtime_session_id:
            raise ValueError("Operation and request runtime sessions differ.")
        validate_cube_lift_sim_route_v1(route=route)
        request_route_fields = (
            (request.skill_id, route.skill_id, "skill ID"),
            (request.skill_version, route.skill_version, "skill version"),
            (
                request.skill_definition_hash,
                route.skill_definition_hash,
                "skill definition hash",
            ),
            (request.route_id, route.route_id, "Route ID"),
            (request.route_version, route.route_version, "Route version"),
        )
        for actual, expected, label in request_route_fields:
            if actual != expected:
                raise ValueError(f"Request {label} does not match the CubeLift Route.")
        if request.route_hash != route.semantic_content_hash:
            raise ValueError("Request does not bind the exact CubeLift Route.")
        if not isinstance(deployment_binding, CubeLiftSimDeploymentBinding):
            raise TypeError(
                "deployment_binding must be a CubeLiftSimDeploymentBinding."
            )
        if deployment_binding.route_hash != route.semantic_content_hash:
            raise ValueError("Deployment binding does not bind the Route.")
        if request.deployment_binding_hash != deployment_binding.semantic_content_hash:
            raise ValueError("Request deployment binding hash mismatch.")
        if deployment_binding.evaluator_configuration_hash != self.config.semantic_content_hash:
            raise ValueError("Deployment binds another evaluator configuration.")
        if deployment_binding.task_config_hash != self.config.task_config_hash:
            raise ValueError("Deployment binds another task configuration.")
        if not isinstance(evaluation_plan, SkillEvaluationPlan):
            raise TypeError("evaluation_plan must be a SkillEvaluationPlan.")
        if evaluation_plan.skill_definition_hash != route.skill_definition_hash:
            raise ValueError("Evaluation plan skill definition hash mismatch.")
        if evaluation_plan.route_hash != route.semantic_content_hash:
            raise ValueError("Evaluation plan Route hash mismatch.")
        if evaluation_plan.deployment_binding_hash != deployment_binding.semantic_content_hash:
            raise ValueError("Evaluation plan deployment hash mismatch.")
        spec = evaluation_plan.evaluator(
            CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1,
            "1",
        )
        if spec.configuration_hash != self.config.semantic_content_hash:
            raise ValueError("Evaluation plan uses another evaluator config.")
        if tuple(evaluation_plan.evaluator_specs) != (spec,):
            raise ValueError("Exact CubeLift evaluation permits one evaluator only.")
        if spec.input_evidence_schema_hash != cube_lift_evaluator_input_schema_hash_v1(
            self.config
        ):
            raise ValueError("CubeLift evaluator input evidence schema mismatch.")
        if spec.output_schema_hash != route.task_evaluation_schema_hash:
            raise ValueError("CubeLift evaluator output schema mismatch.")
        if evaluation_plan.task_outcome_schema_hash != route.task_evaluation_schema_hash:
            raise ValueError("Evaluation plan task outcome schema mismatch.")
        if (
            evaluation_plan.aggregation_policy_id
            != CUBE_LIFT_AGGREGATION_POLICY_ID_V1
            or evaluation_plan.aggregation_policy_hash
            != CUBE_LIFT_AGGREGATION_POLICY_HASH_V1
        ):
            raise ValueError("CubeLift aggregation policy mismatch.")
        if request.runtime_session_id != deployment_binding.runtime_session_id:
            raise ValueError("Request and deployment runtime sessions differ.")
        if lower_hold_proof.runtime_session_id != request.runtime_session_id:
            raise ValueError("Lower HOLD proof runtime session mismatch.")
        if lower_hold_proof.control_domain_id != deployment_binding.control_domain_id:
            raise ValueError("Lower HOLD proof control domain mismatch.")
        if getattr(self.sampler, "runtime_session_id", None) != request.runtime_session_id:
            raise ValueError("Sampler runtime session mismatch.")
        if getattr(self.sampler, "control_domain_id", None) != lower_hold_proof.control_domain_id:
            raise ValueError("Sampler control domain mismatch.")

    def _build_bundle(
        self,
        *,
        operation: AgentSkillOperation,
        request: AgentSkillRequest,
        route: SkillExecutionRoute,
        deployment_binding: CubeLiftSimDeploymentBinding,
        evaluation_plan: SkillEvaluationPlan,
        evidence: CubeLiftTerminalHoldEvidence,
        terminal_state_hash: str,
    ) -> CubeLiftTerminalEvaluationBundle:
        samples = evidence.samples
        minimum_z = min(sample.cube_center_z_m for sample in samples)
        final_z = samples[-1].cube_center_z_m
        finite_all = all(sample.finite_state for sample in samples)
        bilateral_all = all(
            sample.left_finger_cube_contact
            and sample.right_finger_cube_contact
            for sample in samples
        )
        table_any = any(sample.table_cube_contact for sample in samples)
        failures: list[str] = []
        if minimum_z < evidence.baseline_resting_center_z_m + self.config.lift_delta_m:
            failures.append("cube_below_required_lift")
        if self.config.success_requires_bilateral_contact and not bilateral_all:
            failures.append("bilateral_contact_not_stable")
        if self.config.requires_no_table_contact and table_any:
            failures.append("table_contact_observed")
        if not finite_all:
            failures.append("non_finite_state_observed")
        if not evidence.execution_lineage_complete:
            verdict = "inconclusive"
            reason_codes = ("execution_lineage_incomplete", *failures)
        elif failures:
            verdict = "failed"
            reason_codes = tuple(failures)
        else:
            verdict = "succeeded"
            reason_codes = ("cube_lift_terminal_stable_hold_satisfied",)
        task_evaluation = {
            "schema_version": CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1,
            "verdict": verdict,
            "reason_codes": list(reason_codes),
            "evaluated_cut_id": evidence.evaluated_cut_id,
            "environment_evidence_hash": evidence.semantic_content_hash,
            "evaluator_configuration_hash": self.config.semantic_content_hash,
            "terminal_state_hash": terminal_state_hash,
            "baseline_resting_center_z_m": evidence.baseline_resting_center_z_m,
            "required_lift_delta_m": self.config.lift_delta_m,
            "minimum_verified_cube_center_z_m": minimum_z,
            "final_cube_center_z_m": final_z,
            "stable_hold_required_physics_ticks": (
                self.config.stable_hold_physics_ticks
            ),
            "stable_hold_observed_physics_ticks": len(samples),
            "finite_state_all_ticks": finite_all,
            "bilateral_contact_all_ticks": bilateral_all,
            "table_contact_any_tick": table_any,
            "terminal_hold_confirmed": True,
            "execution_lineage_complete": evidence.execution_lineage_complete,
        }
        validate_cube_lift_task_evaluation_v1(
            evaluation=task_evaluation,
            config=self.config,
        )
        evaluator_spec = evaluation_plan.evaluator(
            CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1,
            "1",
        )
        evaluated_at = _finite(self.clock(), field_name="evaluated_at")
        evidence_ref = (
            f"cube-lift-terminal-evidence:{evidence.semantic_content_hash}"
        )
        evaluator_receipt = SkillEvaluationReceipt(
            operation_id=operation.operation_id,
            request_hash=request.semantic_content_hash,
            skill_id=request.skill_id,
            skill_version=request.skill_version,
            skill_definition_hash=request.skill_definition_hash,
            route_id=request.route_id,
            route_version=request.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=deployment_binding.semantic_content_hash,
            evaluation_plan_hash=evaluation_plan.semantic_content_hash,
            evaluator_id=evaluator_spec.evaluator_id,
            evaluator_version=evaluator_spec.evaluator_version,
            evaluator_hash=evaluator_spec.semantic_content_hash,
            evaluator_kind=evaluator_spec.evaluator_kind,
            evidence_domain_id=evaluator_spec.evidence_domain_id,
            input_evidence_hash=evidence.semantic_content_hash,
            evaluated_cut_id=evidence.evaluated_cut_id,
            evaluated_at=evaluated_at,
            verdict=verdict,
            reason_codes=tuple(reason_codes),
            evidence_refs=(
                evidence.execution_evidence_chain_ref,
                evidence.terminal_hold_receipt_ref,
                evidence_ref,
            ),
        )
        aggregate = SkillAggregateEvaluationReceipt(
            operation_id=operation.operation_id,
            request_hash=request.semantic_content_hash,
            evaluation_plan_hash=evaluation_plan.semantic_content_hash,
            aggregation_policy_id=CUBE_LIFT_AGGREGATION_POLICY_ID_V1,
            aggregation_policy_hash=CUBE_LIFT_AGGREGATION_POLICY_HASH_V1,
            consumed_evaluation_receipt_hashes=(
                evaluator_receipt.semantic_content_hash,
            ),
            ignored_evaluation_receipt_hashes=(),
            missing_evaluator_ids=(),
            verdict=verdict,
            reason_codes=tuple(reason_codes),
            generated_at=evaluated_at,
        )
        return CubeLiftTerminalEvaluationBundle(
            evidence=evidence,
            evaluation_receipt=evaluator_receipt,
            aggregate_receipt=aggregate,
            task_evaluation=task_evaluation,
        )


__all__ = [
    "CUBE_LIFT_LOWER_HOLD_PROOF_HASH_DOMAIN_V1",
    "CUBE_LIFT_TERMINAL_CUT_HASH_DOMAIN_V1",
    "CUBE_LIFT_TERMINAL_EVIDENCE_HASH_DOMAIN_V1",
    "CUBE_LIFT_TERMINAL_SAMPLE_HASH_DOMAIN_V1",
    "CubeLiftLowerHoldProof",
    "CubeLiftTerminalEvaluationBundle",
    "CubeLiftTerminalHoldEvidence",
    "CubeLiftTerminalHoldSample",
    "CubeLiftTerminalHoldSampler",
    "CubeLiftTerminalStableHoldEvaluator",
    "MujocoCubeLiftTerminalHoldSampler",
    "cube_lift_lower_hold_proof_from_transition",
]
