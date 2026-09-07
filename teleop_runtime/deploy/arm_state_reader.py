"""Causal arm-state sampling for the real policy deployment frontend."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
    ExecutionObservationBinding,
    ExecutionObservationCapabilitySpec,
    validate_execution_observation_binding,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.runtime.causal_history import CausalSnapshotHistory
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import ObservationSampleRef
from robot_motion.runtime.pose_observation import (
    PassivePoseMeasurementJournal,
    PoseCoordinateBinding,
    PoseMeasurementJournalCut,
    PoseMeasurementJournalEvent,
    PoseMeasurementPublication,
)

from ..interfaces import EffectorBackend

REAL_ARM_QPOS_MEASUREMENT_HASH_DOMAIN_V1 = (
    "dexteleop.real_arm.hardware_qpos_measurement.v1"
)
REAL_ARM_TCP_POSE_MEASUREMENT_ID_HASH_DOMAIN_V1 = (
    "dexteleop.real_arm.hardware_qpos_fk_pose_measurement_id.v1"
)
HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY = "hardware_pose_coordinate_binding"
ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY = (
    "actual_tcp_pose_coordinate_binding_hash"
)


@dataclass(frozen=True)
class ArmPoseStateReadResult:
    """One already-refreshed state plus an application-owned pose binding.

    The binding travels separately from the backend mapping so a concrete lower
    backend cannot impersonate the application-owned canonical binding.  A
    missing binding is a passive-observation failure and must carry a diagnostic
    reason; it does not make the legacy arm state unavailable.
    """

    state: Mapping[str, Any]
    coordinate_binding: PoseCoordinateBinding | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, Mapping):
            raise TypeError("state must be a mapping.")
        if self.coordinate_binding is not None and not isinstance(
            self.coordinate_binding,
            PoseCoordinateBinding,
        ):
            raise TypeError(
                "coordinate_binding must be a PoseCoordinateBinding or None."
            )
        error = self.error
        if error is not None:
            if not isinstance(error, str) or not error.strip():
                raise ValueError("error must be a non-empty string when supplied.")
            error = error.strip()
            object.__setattr__(self, "error", error)
        if self.coordinate_binding is None and error is None:
            raise ValueError("A missing coordinate_binding requires an error reason.")
        if self.coordinate_binding is not None and error is not None:
            raise ValueError("A successful coordinate_binding cannot carry an error.")


ArmPoseStateReader = Callable[[EffectorBackend], ArmPoseStateReadResult]


def _read_backend_state(
    method: Callable[..., Any],
    *,
    raw_hardware_qpos_only: bool,
) -> Any:
    """Call one backend read without materializing unused derived state.

    The generic deployment reader historically called backend methods without
    arguments, so non-RM65 backends and older test doubles remain valid. The
    real RM65 backend exposes explicit switches for FK, timing-percentile, and
    pose-attestation work. A qpos-only owner can disable those switches before
    the call rather than paying their cost and discarding the result.
    """

    if not raw_hardware_qpos_only:
        return method()
    optional_arguments = {
        "include_derived_pose": False,
        "include_timing_percentiles": False,
        "include_pose_source_attestation": False,
    }
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method()
    accepts_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported_arguments = {
        name: value
        for name, value in optional_arguments.items()
        if accepts_arbitrary_keywords
        or (
            name in parameters
            and parameters[name].kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
    }
    return method(**supported_arguments)


@dataclass(frozen=True)
class ArmPoseObservationProjection:
    """Exact contracts required to publish one side's measured TCP pose."""

    coordinate_binding: PoseCoordinateBinding
    capability: ExecutionObservationCapabilitySpec
    observation_binding: ExecutionObservationBinding

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_binding, PoseCoordinateBinding):
            raise TypeError("coordinate_binding must be a PoseCoordinateBinding.")
        if not isinstance(self.capability, ExecutionObservationCapabilitySpec):
            raise TypeError("capability must be an ExecutionObservationCapabilitySpec.")
        if not isinstance(self.observation_binding, ExecutionObservationBinding):
            raise TypeError(
                "observation_binding must be an ExecutionObservationBinding."
            )
        coordinate = self.coordinate_binding
        capability = self.capability
        binding = self.observation_binding
        if (
            binding.binding_id != coordinate.observation_binding_id
            or binding.semantic_content_hash != coordinate.observation_binding_hash
            or binding.plan_hash != coordinate.observation_plan_hash
        ):
            raise ValueError("Coordinate and execution observation bindings differ.")
        if (
            capability.semantic_content_hash != binding.capability_hash
            or capability.capability_id != binding.capability_id
            or capability.clock_domain_id != coordinate.measurement_clock_domain_id
            or binding.clock_domain_id != coordinate.measurement_clock_domain_id
        ):
            raise ValueError("Pose capability/binding clock or identity mismatch.")
        component_id = coordinate.component_id
        if (
            component_id not in binding.component_ids
            or capability.representation_by_component.get(component_id)
            != "cartesian_pose"
            or capability.schema_hash_by_component.get(component_id)
            != coordinate.pose_schema_hash
        ):
            raise ValueError("Pose component is not covered by the accepted binding.")


def _fact_reference(
    *,
    stage: str,
    fact_id: str,
    fact_hash: str,
    observed_at: float,
    clock_domain_id: str,
    status: str,
) -> ExecutionFactReference:
    return ExecutionFactReference(
        stage=stage,
        fact_id=fact_id,
        fact_hash=fact_hash,
        observed_at=observed_at,
        clock_domain_id=clock_domain_id,
        status=status,
    )


def project_arm_tcp_pose_measurement(
    selected_state: Mapping[str, Any],
    *,
    projection: ArmPoseObservationProjection,
    received_monotonic_ns: int,
    receiver_clock: ClockDomainDescriptor,
) -> PoseMeasurementPublication:
    """Project an already-read hardware-qpos snapshot without device I/O."""

    if not isinstance(selected_state, Mapping):
        raise TypeError("selected_state must be a mapping.")
    if not isinstance(projection, ArmPoseObservationProjection):
        raise TypeError("projection must be an ArmPoseObservationProjection.")
    if not isinstance(receiver_clock, ClockDomainDescriptor):
        raise TypeError("receiver_clock must be a ClockDomainDescriptor.")
    if (
        selected_state.get("causal_snapshot_missing") is True
        or selected_state.get("hardware_measurement_valid") is not True
        or selected_state.get("hardware_read_success") is not True
        or selected_state.get("actual_tcp_pose_source") != "hardware_qpos_fk"
    ):
        raise ValueError("state does not contain a valid hardware-qpos TCP pose.")
    raw_sequence = selected_state.get("hardware_measurement_sequence")
    if (
        isinstance(raw_sequence, bool)
        or not isinstance(raw_sequence, (int, np.integer))
        or int(raw_sequence) <= 0
    ):
        raise ValueError("hardware measurement sequence must be a positive integer.")
    source_sequence = int(raw_sequence)
    measured_at = float(selected_state.get("hardware_measurement_timestamp"))
    available_at = float(selected_state.get("available_timestamp"))
    if (
        not math.isfinite(measured_at)
        or not math.isfinite(available_at)
        or available_at + 1e-12 < measured_at
    ):
        raise ValueError("hardware measurement timing is invalid.")
    coordinate = projection.coordinate_binding
    source_coordinate_binding = selected_state.get(
        HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY
    )
    if not isinstance(source_coordinate_binding, PoseCoordinateBinding):
        raise TypeError("hardware state has no typed pose-coordinate attestation.")
    if source_coordinate_binding != coordinate:
        raise ValueError(
            "hardware pose-coordinate attestation differs from projection."
        )
    if (
        selected_state.get(ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY)
        != coordinate.semantic_content_hash
    ):
        raise ValueError("FK output is not attested to the pose-coordinate binding.")
    if (
        receiver_clock.monotonic is not True
        or receiver_clock.clock_domain_id != coordinate.receipt_clock_domain_id
    ):
        raise ValueError("receiver clock does not match the coordinate binding.")
    source_clock = str(selected_state.get("source_clock") or "").strip()
    available_clock = str(selected_state.get("available_clock") or "").strip()
    if (
        source_clock != coordinate.measurement_clock_domain_id
        or available_clock != coordinate.measurement_clock_domain_id
    ):
        raise ValueError("hardware measurement clocks do not match the binding.")
    if float(selected_state.get("source_timestamp")) != measured_at:
        raise ValueError("causal source timestamp differs from hardware measurement.")
    if float(selected_state.get("actual_tcp_pose_timestamp")) != measured_at:
        raise ValueError("actual TCP pose timestamp differs from hardware measurement.")
    try:
        qpos = np.asarray(selected_state.get("hardware_qpos"), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("hardware_qpos must be a finite six-vector.") from exc
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise ValueError("hardware_qpos must be a finite six-vector.")
    try:
        pose = np.asarray(
            selected_state.get("actual_tcp_pose"),
            dtype=float,
        ).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual_tcp_pose must be a finite rigid 4x4 matrix.") from exc
    if (
        not np.all(np.isfinite(pose))
        or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError("actual_tcp_pose must be a finite rigid 4x4 matrix.")
    pose_xyzquat = pose_matrix_to_xyzquat(pose)
    if pose_xyzquat is None or pose_xyzquat.shape != (7,):
        raise ValueError("actual_tcp_pose cannot be represented as xyz+quat_xyzw.")
    if (
        isinstance(received_monotonic_ns, bool)
        or not isinstance(received_monotonic_ns, (int, np.integer))
        or int(received_monotonic_ns) < 0
    ):
        raise ValueError("received_monotonic_ns must be a non-negative integer.")
    received_at = int(received_monotonic_ns) / 1_000_000_000.0
    source_measurement_hash = canonical_content_hash(
        {
            "source_instance_id": coordinate.source_instance_id,
            "source_epoch_id": coordinate.source_epoch_id,
            "source_binding_hash": coordinate.source_binding_hash,
            "source_sequence": source_sequence,
            "measured_at": measured_at,
            "clock_domain_id": coordinate.measurement_clock_domain_id,
            "hardware_qpos": qpos,
            "hardware_read_success": True,
            "hardware_measurement_valid": True,
        },
        domain=REAL_ARM_QPOS_MEASUREMENT_HASH_DOMAIN_V1,
    )
    measurement_identity_hash = canonical_content_hash(
        {
            "source_measurement_hash": source_measurement_hash,
            "coordinate_binding_hash": coordinate.semantic_content_hash,
            "pose_xyz_quat_xyzw": tuple(float(value) for value in pose_xyzquat),
        },
        domain=REAL_ARM_TCP_POSE_MEASUREMENT_ID_HASH_DOMAIN_V1,
    )
    measurement_id = f"real-arm-tcp-pose:{measurement_identity_hash}"
    parent_facts = (
        _fact_reference(
            stage="real_robot.source_binding",
            fact_id=(f"{coordinate.source_instance_id}:{coordinate.source_epoch_id}"),
            fact_hash=coordinate.source_binding_hash,
            observed_at=available_at,
            clock_domain_id=coordinate.measurement_clock_domain_id,
            status="bound",
        ),
        _fact_reference(
            stage="real_robot.hardware_qpos",
            fact_id=(
                f"{coordinate.source_instance_id}:"
                f"{coordinate.source_epoch_id}:{source_sequence}"
            ),
            fact_hash=source_measurement_hash,
            observed_at=measured_at,
            clock_domain_id=coordinate.measurement_clock_domain_id,
            status="observed",
        ),
        _fact_reference(
            stage="real_robot.fk_contract",
            fact_id=coordinate.fk_contract_id,
            fact_hash=coordinate.fk_contract_hash,
            observed_at=available_at,
            clock_domain_id=coordinate.measurement_clock_domain_id,
            status="applied",
        ),
        _fact_reference(
            stage="real_robot.pose_coordinate_binding",
            fact_id=coordinate.binding_id,
            fact_hash=coordinate.semantic_content_hash,
            observed_at=available_at,
            clock_domain_id=coordinate.measurement_clock_domain_id,
            status="applied",
        ),
    )
    measurement = ExecutionMeasurementObservation(
        measurement_id=measurement_id,
        source_measurement_hash=source_measurement_hash,
        control_domain_id=coordinate.control_domain_id,
        measured_at=measured_at,
        available_at=available_at,
        clock_domain_id=coordinate.measurement_clock_domain_id,
        value_by_component={
            coordinate.component_id: tuple(float(value) for value in pose_xyzquat),
        },
        representation_by_component={coordinate.component_id: "cartesian_pose"},
        schema_hash_by_component={
            coordinate.component_id: coordinate.pose_schema_hash,
        },
        observation_binding_id=projection.observation_binding.binding_id,
        observation_plan_hash=projection.observation_binding.plan_hash,
        parent_facts=parent_facts,
    )
    validate_execution_observation_binding(
        measurement,
        projection.observation_binding,
        projection.capability,
    )
    sample = ObservationSampleRef(
        stream_id=coordinate.stream_id,
        sample_id=measurement.measurement_id,
        sequence=source_sequence,
        measured_at=measurement.measured_at,
        measured_clock_domain_id=measurement.clock_domain_id,
        available_at=received_at,
        available_clock_domain_id=coordinate.receipt_clock_domain_id,
        payload_object_id=measurement.semantic_content_hash,
        valid=True,
        stale=False,
        content_metadata={
            "source_instance_id": coordinate.source_instance_id,
            "source_epoch_id": coordinate.source_epoch_id,
            "source_kind": coordinate.source_kind,
            "source_sequence": source_sequence,
            "coordinate_binding_hash": coordinate.semantic_content_hash,
            "reference_frame_id": coordinate.reference_frame_id,
            "reference_frame_spec_hash": coordinate.reference_frame_spec_hash,
            "moving_frame_id": coordinate.moving_frame_id,
            "tool_profile_hash": coordinate.tool_profile_hash,
            "measurement_available_at": measurement.available_at,
            "measurement_available_clock_domain_id": measurement.clock_domain_id,
            "received_monotonic_ns": int(received_monotonic_ns),
        },
    )
    return PoseMeasurementPublication(
        coordinate_binding=coordinate,
        source_sequence=source_sequence,
        measurement=measurement,
        sample_ref=sample,
        received_monotonic_ns=int(received_monotonic_ns),
        receiver_clock=receiver_clock,
    )


class ArmStateReader:
    """Refresh backend state and select the latest snapshot admitted at a cut.

    The reader owns only refresh cadence and bounded causal history.  Backends
    continue to own device state and FK, while deployment retains control,
    safety, authority, and dispatch decisions.
    """

    def __init__(
        self,
        backends: Mapping[str, EffectorBackend],
        *,
        refresh_hz: float,
        compute_tcp_pose: bool = True,
        clock: Any = time.time,
        pose_projection_by_side: Mapping[
            str,
            ArmPoseObservationProjection,
        ]
        | None = None,
        pose_state_read_by_side: Mapping[str, ArmPoseStateReader] | None = None,
        concurrent_refresh_executor: Executor | None = None,
        receipt_clock: ClockDomainDescriptor | None = None,
        receipt_monotonic_ns_clock: Any = time.monotonic_ns,
    ):
        self.backends = dict(backends)
        self.refresh_hz = float(refresh_hz)
        self.compute_tcp_pose = bool(compute_tcp_pose)
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(receipt_monotonic_ns_clock):
            raise TypeError("receipt_monotonic_ns_clock must be callable")
        self.clock = clock
        self.receipt_monotonic_ns_clock = receipt_monotonic_ns_clock
        self.pose_projection_by_side = dict(pose_projection_by_side or {})
        unknown_projection_sides = set(self.pose_projection_by_side) - set(
            self.backends
        )
        if unknown_projection_sides:
            raise ValueError(
                "pose_projection_by_side contains unknown backends: "
                f"{sorted(unknown_projection_sides)}"
            )
        if any(
            not isinstance(value, ArmPoseObservationProjection)
            for value in self.pose_projection_by_side.values()
        ):
            raise TypeError(
                "pose_projection_by_side values must be ArmPoseObservationProjection."
            )
        self.pose_state_read_by_side = dict(pose_state_read_by_side or {})
        if concurrent_refresh_executor is not None and not callable(
            getattr(concurrent_refresh_executor, "submit", None)
        ):
            raise TypeError("concurrent_refresh_executor must expose submit().")
        self.concurrent_refresh_executor = concurrent_refresh_executor
        unknown_state_read_sides = set(self.pose_state_read_by_side) - set(
            self.pose_projection_by_side
        )
        if unknown_state_read_sides:
            raise ValueError(
                "pose_state_read_by_side requires matching pose projections: "
                f"{sorted(unknown_state_read_sides)}"
            )
        if any(not callable(value) for value in self.pose_state_read_by_side.values()):
            raise TypeError("pose_state_read_by_side values must be callable.")
        if self.pose_projection_by_side:
            if not isinstance(receipt_clock, ClockDomainDescriptor):
                raise TypeError(
                    "receipt_clock must be a ClockDomainDescriptor when pose "
                    "projection is enabled."
                )
            if receipt_clock.monotonic is not True:
                raise ValueError("receipt_clock.monotonic must be exactly True.")
            mismatched_clock_sides = sorted(
                side
                for side, projection in self.pose_projection_by_side.items()
                if projection.coordinate_binding.receipt_clock_domain_id
                != receipt_clock.clock_domain_id
            )
            if mismatched_clock_sides:
                raise ValueError(
                    "pose projection receipt clocks differ from receipt_clock: "
                    f"{mismatched_clock_sides}"
                )
            self.pose_measurement_journal: PassivePoseMeasurementJournal | None = (
                PassivePoseMeasurementJournal(receiver_clock=receipt_clock)
            )
        else:
            if receipt_clock is not None and not isinstance(
                receipt_clock,
                ClockDomainDescriptor,
            ):
                raise TypeError(
                    "receipt_clock must be a ClockDomainDescriptor or None."
                )
            self.pose_measurement_journal = None
        self.receipt_clock = receipt_clock
        self._pose_observation_error_by_side: dict[str, str | None] = {
            side: None for side in self.backends
        }
        self._last_refresh_at = {side: 0.0 for side in self.backends}
        self._history_by_side = {
            side: CausalSnapshotHistory(
                maxlen=256,
                name=f"deploy:arm:{side}",
                clock=self.clock,
            )
            for side in self.backends
        }

    def _invalidate_pose_projection(
        self,
        *,
        side: str,
        projection: ArmPoseObservationProjection,
        reason: str,
        error: Exception,
    ) -> None:
        """Clear a passive latest value after a source/projection failure."""

        self._pose_observation_error_by_side[side] = f"{type(error).__name__}: {error}"
        journal = self.pose_measurement_journal
        assert journal is not None
        try:
            journal.invalidate(
                coordinate_binding=projection.coordinate_binding,
                reason=reason,
                received_monotonic_ns=int(self.receipt_monotonic_ns_clock()),
                next_coordinate_binding=projection.coordinate_binding,
            )
        except Exception as invalidation_exc:  # noqa: BLE001
            self._pose_observation_error_by_side[side] += (
                "; invalidation_failed:"
                f"{type(invalidation_exc).__name__}:"
                f"{invalidation_exc}"
            )

    def snapshot(
        self,
        now: float,
        *,
        cut_after_refresh: bool = False,
        concurrent_refresh_sides: tuple[str, ...] = (),
    ) -> dict[str, dict[str, Any]]:
        """Refresh and select arm state at an explicit causal cut.

        The default preserves the historical/replay contract: ``now`` is the
        fixed cut, so a read completed after that cut is deliberately not
        visible.  ``cut_after_refresh`` is for an explicit live, on-demand
        observation (for example a startup Skill admission): it captures one
        shared cut after all requested refreshes and selects the newly
        published state without weakening historical queries.

        ``concurrent_refresh_sides`` is for an application owner that needs a
        coherent multi-arm live cut.  The named independent backends are
        force-refreshed concurrently, then every side is selected at one shared
        causal cutoff.  This avoids making the first serial hardware sample
        stale while waiting for the second controller.
        """

        requested_at = float(now)
        if not math.isfinite(requested_at):
            raise ValueError("now must be finite")
        concurrent_sides = frozenset(concurrent_refresh_sides)
        unknown_concurrent_sides = concurrent_sides - set(self.backends)
        if unknown_concurrent_sides:
            raise ValueError(
                "concurrent_refresh_sides contains unknown backends: "
                f"{sorted(unknown_concurrent_sides)}"
            )
        concurrent_ready: dict[str, bool] = {}
        concurrent_refresh_errors: dict[str, str] = {}
        if concurrent_sides:
            for side in concurrent_sides:
                backend = self.backends[side]
                projection = self.pose_projection_by_side.get(side)
                try:
                    concurrent_ready[side] = backend.is_ready() is True
                    if not concurrent_ready[side] and projection is not None:
                        self._invalidate_pose_projection(
                            side=side,
                            projection=projection,
                            reason="backend_not_ready",
                            error=RuntimeError("backend is not ready"),
                        )
                except Exception as exc:
                    if projection is None:
                        raise
                    concurrent_ready[side] = False
                    self._invalidate_pose_projection(
                        side=side,
                        projection=projection,
                        reason="backend_readiness_unavailable",
                        error=exc,
                    )

            def refresh_one(side: str) -> str | None:
                try:
                    refresh = getattr(
                        self.backends[side],
                        "refresh_state_from_hardware",
                        None,
                    )
                    if callable(refresh):
                        _read_backend_state(
                            refresh,
                            raw_hardware_qpos_only=not self.compute_tcp_pose,
                        )
                except Exception as exc:  # noqa: BLE001
                    return f"refresh_failed:{type(exc).__name__}:{exc}"
                return None

            ready_sides = tuple(
                side
                for side in self.backends
                if side in concurrent_sides and concurrent_ready[side]
            )
            if ready_sides:
                executor = self.concurrent_refresh_executor
                owns_executor = executor is None
                if executor is None:
                    executor = ThreadPoolExecutor(
                        max_workers=len(ready_sides),
                        thread_name_prefix="arm-state-refresh",
                    )
                try:
                    futures = {
                        side: executor.submit(refresh_one, side)
                        for side in ready_sides
                    }
                    for side, future in futures.items():
                        refresh_error = future.result()
                        if refresh_error is not None:
                            concurrent_refresh_errors[side] = refresh_error
                        self._last_refresh_at[side] = now
                finally:
                    if owns_executor:
                        executor.shutdown(wait=True)
        states: dict[str, dict[str, Any]] = {}
        refresh_period = 0.0 if self.refresh_hz <= 0.0 else 1.0 / self.refresh_hz
        for side, backend in self.backends.items():
            projection = self.pose_projection_by_side.get(side)
            projection_source_ready = True
            projection_binding_ready = True
            if projection is not None:
                if side in concurrent_sides:
                    projection_source_ready = concurrent_ready[side]
                else:
                    try:
                        projection_source_ready = backend.is_ready() is True
                        if not projection_source_ready:
                            self._invalidate_pose_projection(
                                side=side,
                                projection=projection,
                                reason="backend_not_ready",
                                error=RuntimeError("backend is not ready"),
                            )
                    except Exception as exc:  # noqa: BLE001
                        projection_source_ready = False
                        self._invalidate_pose_projection(
                            side=side,
                            projection=projection,
                            reason="backend_readiness_unavailable",
                            error=exc,
                        )
            refresh_error = concurrent_refresh_errors.get(side)
            if (
                side not in concurrent_sides
                and refresh_period > 0.0
                and now - self._last_refresh_at.get(side, 0.0) >= refresh_period
                and (
                    projection_source_ready
                    if projection is not None
                    else backend.is_ready()
                )
            ):
                try:
                    refresh = getattr(backend, "refresh_state_from_hardware", None)
                    if callable(refresh):
                        _read_backend_state(
                            refresh,
                            raw_hardware_qpos_only=not self.compute_tcp_pose,
                        )
                except Exception as exc:  # noqa: BLE001
                    refresh_error = f"refresh_failed:{type(exc).__name__}:{exc}"
                    if projection is not None:
                        self._invalidate_pose_projection(
                            side=side,
                            projection=projection,
                            reason="hardware_refresh_failed",
                            error=RuntimeError(refresh_error),
                        )
                self._last_refresh_at[side] = now
            elif refresh_error is not None and projection is not None:
                self._invalidate_pose_projection(
                    side=side,
                    projection=projection,
                    reason="hardware_refresh_failed",
                    error=RuntimeError(refresh_error),
                )
            try:
                pose_state_read = self.pose_state_read_by_side.get(side)
                if pose_state_read is None:
                    state = dict(
                        _read_backend_state(
                            backend.get_state,
                            raw_hardware_qpos_only=not self.compute_tcp_pose,
                        )
                    )
                else:
                    state_read = pose_state_read(backend)
                    if not isinstance(state_read, ArmPoseStateReadResult):
                        raise TypeError(
                            "pose state reader must return ArmPoseStateReadResult."
                        )
                    state = dict(state_read.state)
                    state.pop(HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY, None)
                    source_binding = state_read.coordinate_binding
                    assert projection is not None
                    if source_binding is None:
                        projection_binding_ready = False
                        self._invalidate_pose_projection(
                            side=side,
                            projection=projection,
                            reason="pose_coordinate_binding_unavailable",
                            error=RuntimeError(str(state_read.error)),
                        )
                    elif source_binding != projection.coordinate_binding:
                        projection_binding_ready = False
                        self._invalidate_pose_projection(
                            side=side,
                            projection=projection,
                            reason="pose_coordinate_binding_mismatch",
                            error=ValueError(
                                "pose state reader returned a coordinate binding "
                                "that differs from the configured projection"
                            ),
                        )
                    else:
                        state[HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY] = (
                            source_binding
                        )
            except Exception as exc:
                if projection is not None:
                    self._invalidate_pose_projection(
                        side=side,
                        projection=projection,
                        reason="backend_state_unavailable",
                        error=exc,
                    )
                raise
            if refresh_error is not None:
                state["state_error"] = refresh_error
            state["actual_tcp_pose"] = None
            state["actual_tcp_pose_source"] = None
            state["actual_tcp_pose_timestamp"] = None
            if projection is not None:
                state[ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY] = None
            if self.compute_tcp_pose:
                try:
                    hardware_qpos = state.get("hardware_qpos")
                    compute_pose = getattr(backend, "compute_tcp_pose", None)
                    if not callable(compute_pose):
                        raise TypeError("backend has no compute_tcp_pose")
                    if hardware_qpos is not None:
                        measured_pose = compute_pose(qpos=hardware_qpos)
                        state["current_tcp_pose"] = measured_pose
                        state["tcp_pose_source"] = "hardware_qpos_fk"
                        try:
                            hardware_qpos_vector = np.asarray(
                                hardware_qpos, dtype=float
                            ).reshape(-1)
                            measurement_timestamp = float(
                                state.get("hardware_measurement_timestamp")
                            )
                            measurement_sequence = state.get(
                                "hardware_measurement_sequence"
                            )
                            measurement_valid = state.get("hardware_measurement_valid")
                            read_success = state.get("hardware_read_success")
                            sequence_valid = (
                                measurement_sequence is None
                                or int(measurement_sequence) > 0
                            )
                            actual_valid = bool(
                                hardware_qpos_vector.shape == (6,)
                                and np.all(np.isfinite(hardware_qpos_vector))
                                and np.isfinite(measurement_timestamp)
                                and isinstance(measurement_valid, (bool, np.bool_))
                                and bool(measurement_valid)
                                and not (
                                    isinstance(read_success, (bool, np.bool_))
                                    and not bool(read_success)
                                )
                                and sequence_valid
                            )
                        except (TypeError, ValueError, OverflowError):
                            actual_valid = False
                            measurement_timestamp = None
                        if actual_valid:
                            state["actual_tcp_pose"] = np.asarray(
                                measured_pose, dtype=float
                            ).copy()
                            state["actual_tcp_pose_source"] = "hardware_qpos_fk"
                            state["actual_tcp_pose_timestamp"] = float(
                                measurement_timestamp
                            )
                            source_binding = state.get(
                                HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY
                            )
                            if (
                                projection is not None
                                and isinstance(source_binding, PoseCoordinateBinding)
                                and source_binding == projection.coordinate_binding
                            ):
                                state[
                                    ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY
                                ] = source_binding.semantic_content_hash
                    else:
                        state["current_tcp_pose"] = compute_pose()
                        state["tcp_pose_source"] = "current_qpos_fk"
                except Exception as exc:  # noqa: BLE001
                    state.setdefault(
                        "pose_error",
                        f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}",
                    )
            source_timestamp = state.get(
                "hardware_measurement_timestamp",
                state.get("hardware_refresh_time"),
            )
            published_state = self._history_by_side[side].publish(
                state,
                source_timestamp=source_timestamp,
                available_timestamp=None,
                source_clock=str(
                    state.get(
                        "hardware_measurement_clock_domain_id",
                        "local_wall_time",
                    )
                ),
                available_clock=str(
                    state.get(
                        "hardware_availability_clock_domain_id",
                        "local_wall_time",
                    )
                ),
            )
            if projection is not None:
                if (
                    not projection_source_ready
                    or not projection_binding_ready
                    or refresh_error is not None
                ):
                    continue
                journal = self.pose_measurement_journal
                receiver_clock = self.receipt_clock
                assert journal is not None
                assert isinstance(receiver_clock, ClockDomainDescriptor)
                try:
                    projection_state = published_state
                    received_monotonic_ns = int(self.receipt_monotonic_ns_clock())
                    latest = journal.latest(
                        stream_id=projection.coordinate_binding.stream_id
                    )
                    raw_source_sequence = published_state.get(
                        "hardware_measurement_sequence"
                    )
                    if (
                        latest is not None
                        and latest.coordinate_binding == projection.coordinate_binding
                        and not isinstance(raw_source_sequence, bool)
                        and isinstance(raw_source_sequence, (int, np.integer))
                        and int(raw_source_sequence) == latest.source_sequence
                    ):
                        projection_state = dict(published_state)
                        projection_state["available_timestamp"] = (
                            latest.measurement.available_at
                        )
                        received_monotonic_ns = latest.received_monotonic_ns
                    publication = project_arm_tcp_pose_measurement(
                        projection_state,
                        projection=projection,
                        received_monotonic_ns=received_monotonic_ns,
                        receiver_clock=receiver_clock,
                    )
                    journal.publish(publication)
                    self._pose_observation_error_by_side[side] = None
                except Exception as exc:  # noqa: BLE001 - passive feed is fail-closed.
                    self._invalidate_pose_projection(
                        side=side,
                        projection=projection,
                        reason="pose_projection_unavailable",
                        error=exc,
                    )

        cutoff = (
            max(requested_at, float(self.clock()))
            if cut_after_refresh
            else requested_at
        )
        if not math.isfinite(cutoff):
            raise ValueError("causal snapshot cutoff must be finite")
        for side in self.backends:
            selected = self._history_by_side[side].latest_at_or_before(cutoff)
            if selected is None:
                states[side] = {
                    "causal_snapshot_missing": True,
                    "causal_snapshot_cutoff_timestamp": cutoff,
                    "causal_snapshot_contract": (
                        "source_timestamp<=T0_and_available_timestamp<=T0"
                    ),
                }
            else:
                selected_state, selection = selected
                selected_state.update(
                    {
                        "causal_snapshot_missing": False,
                        "causal_snapshot_sequence": selection.sequence,
                        "causal_snapshot_source_timestamp": selection.source_timestamp,
                        "causal_snapshot_available_timestamp": (
                            selection.available_timestamp
                        ),
                        "causal_snapshot_age_sec": selection.age_sec,
                        "causal_snapshot_cutoff_timestamp": cutoff,
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                    }
                )
                states[side] = selected_state
        return states

    def pose_measurement_journal_cut(self) -> PoseMeasurementJournalCut | None:
        """Return one atomic passive cursor/latest cut without backend I/O."""

        journal = self.pose_measurement_journal
        return None if journal is None else journal.atomic_cut()

    def pose_measurement_events_after(
        self,
        event_sequence: int,
    ) -> tuple[PoseMeasurementJournalEvent, ...]:
        """Read retained passive events; never refresh or query a backend."""

        journal = self.pose_measurement_journal
        if journal is None:
            return ()
        return journal.events_after(event_sequence)

    def latest_pose_measurement(
        self,
        *,
        side: str,
    ) -> PoseMeasurementPublication | None:
        """Return the passive latest fact; never refresh a backend."""

        projection = self.pose_projection_by_side.get(str(side))
        journal = self.pose_measurement_journal
        if projection is None or journal is None:
            return None
        return journal.latest(
            stream_id=projection.coordinate_binding.stream_id,
        )

    def invalidate_pose_measurements(
        self,
        *,
        reason: str,
        terminal: bool = True,
    ) -> tuple[PoseMeasurementJournalEvent, ...]:
        """Publish passive lifecycle invalidations without device or motion I/O."""

        journal = self.pose_measurement_journal
        if journal is None:
            return ()
        events: list[PoseMeasurementJournalEvent] = []
        for side, projection in self.pose_projection_by_side.items():
            try:
                event = journal.invalidate(
                    coordinate_binding=projection.coordinate_binding,
                    reason=reason,
                    received_monotonic_ns=int(self.receipt_monotonic_ns_clock()),
                    next_coordinate_binding=(
                        None if terminal else projection.coordinate_binding
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - status remains fail-closed.
                self._pose_observation_error_by_side[side] = (
                    f"invalidation_failed:{type(exc).__name__}:{exc}"
                )
                continue
            if event is not None:
                events.append(event)
        return tuple(events)

    def pose_observation_status(self) -> dict[str, Any]:
        journal = self.pose_measurement_journal
        return {
            **(
                {"mode": "disabled", "event_sequence": 0}
                if journal is None
                else journal.snapshot_status()
            ),
            "error_by_side": dict(self._pose_observation_error_by_side),
        }


__all__ = [
    "ACTUAL_TCP_POSE_COORDINATE_BINDING_HASH_STATE_KEY",
    "HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY",
    "REAL_ARM_QPOS_MEASUREMENT_HASH_DOMAIN_V1",
    "REAL_ARM_TCP_POSE_MEASUREMENT_ID_HASH_DOMAIN_V1",
    "ArmPoseObservationProjection",
    "ArmPoseStateReadResult",
    "ArmPoseStateReader",
    "ArmStateReader",
    "project_arm_tcp_pose_measurement",
]
