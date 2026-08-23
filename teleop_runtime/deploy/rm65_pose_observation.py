"""Application-owned RM65 pose observation binding for policy deployment.

The concrete RM65 backend owns source, attachment, frame, tool, FK, and source
clock facts.  This application adapter validates those facts and joins them
with the deployment-owned component, stream, observation-plan, and receiver
clock identity.  It does not start, refresh, close, or command a backend.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
    bind_execution_observation_plan,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.pose_observation import PoseCoordinateBinding

from ..backends.arms.rm65 import RM65ArmBackend
from ..backends.arms.rm65_pose_source import (
    HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY,
    RM65PoseSourceAttestation,
)
from ..interfaces import EffectorBackend
from .arm_state_reader import (
    ArmPoseObservationProjection,
    ArmPoseStateReader,
    ArmPoseStateReadResult,
)

LINUX_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
RM65_TCP_POSE_SCHEMA_HASH_DOMAIN_V1 = "dexteleop.rm65_tcp_pose_schema.v1"
RM65_TCP_POSE_SCHEMA_HASH_V1 = "sha256:" + canonical_content_hash(
    {
        "schema": "dexteleop.rm65_tcp_pose.xyz_m_quat_xyzw.v1",
        "representation": "cartesian_pose",
        "value_order": ("x", "y", "z", "qx", "qy", "qz", "qw"),
        "translation_unit": "m",
        "quaternion_order": "xyzw",
        "quaternion_norm": "unit",
    },
    domain=RM65_TCP_POSE_SCHEMA_HASH_DOMAIN_V1,
)
RM65_POSE_OBSERVATION_MEASUREMENT_STAGE_V1 = "real_robot.hardware_qpos_fk_pose"


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def build_policy_deploy_receiver_clock(
    *,
    runtime_session_id: str | None = None,
    boot_id_path: Path = LINUX_BOOT_ID_PATH,
) -> ClockDomainDescriptor:
    """Build the exact local monotonic descriptor shared by later consumers."""

    try:
        raw_boot_id = Path(boot_id_path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read the Linux boot ID from {Path(boot_id_path)!s}."
        ) from exc
    try:
        boot_id = str(UUID(raw_boot_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Linux boot ID must be a canonical UUID.") from exc
    session_id = (
        f"policy-deploy:{uuid4().hex}"
        if runtime_session_id is None
        else _required_text(runtime_session_id, name="runtime_session_id")
    )
    monotonic_info = time.get_clock_info("monotonic")
    if monotonic_info.monotonic is not True:
        raise RuntimeError("Python monotonic clock is not monotonic on this host.")
    return ClockDomainDescriptor(
        clock_domain_id=f"linux-monotonic:{boot_id}",
        runtime_session_id=session_id,
        boot_id=boot_id,
        reset_epoch=0,
        kind="python.monotonic",
        resolution_sec=float(monotonic_info.resolution),
        monotonic=True,
    )


@dataclass(frozen=True)
class RM65PoseObservationAssembly:
    """Immutable application bundle consumed by the single arm-state reader."""

    projection_by_side: Mapping[str, ArmPoseObservationProjection]
    state_read_by_side: Mapping[str, ArmPoseStateReader]
    receipt_clock: ClockDomainDescriptor
    receipt_monotonic_ns_clock: Callable[[], int]

    def __post_init__(self) -> None:
        projections = dict(self.projection_by_side)
        state_readers = dict(self.state_read_by_side)
        if not projections or set(projections) != set(state_readers):
            raise ValueError(
                "projection_by_side and state_read_by_side must cover the same "
                "non-empty side set."
            )
        if any(
            not isinstance(value, ArmPoseObservationProjection)
            for value in projections.values()
        ):
            raise TypeError(
                "projection_by_side values must be ArmPoseObservationProjection."
            )
        if any(not callable(value) for value in state_readers.values()):
            raise TypeError("state_read_by_side values must be callable.")
        if not isinstance(self.receipt_clock, ClockDomainDescriptor):
            raise TypeError("receipt_clock must be a ClockDomainDescriptor.")
        if self.receipt_clock.monotonic is not True:
            raise ValueError("receipt_clock.monotonic must be exactly True.")
        if not callable(self.receipt_monotonic_ns_clock):
            raise TypeError("receipt_monotonic_ns_clock must be callable.")
        object.__setattr__(
            self,
            "projection_by_side",
            MappingProxyType(projections),
        )
        object.__setattr__(
            self,
            "state_read_by_side",
            MappingProxyType(state_readers),
        )


def _read_attested_state(backend: EffectorBackend) -> dict[str, Any]:
    get_state = getattr(backend, "get_state", None)
    if not callable(get_state):
        raise TypeError("RM65 pose observation backend has no get_state method.")
    return dict(
        get_state(
            include_derived_pose=False,
            include_timing_percentiles=False,
            include_pose_source_attestation=True,
        )
    )


def _validate_attested_state(
    state: Mapping[str, Any],
    *,
    side: str,
    effector_name: str,
    expected_attestation: RM65PoseSourceAttestation | None,
    require_hardware_measurement: bool,
) -> RM65PoseSourceAttestation:
    attestation = state.get(HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY)
    if not isinstance(attestation, RM65PoseSourceAttestation):
        raise TypeError("RM65 state has no typed pose-source attestation.")
    if expected_attestation is not None and attestation != expected_attestation:
        raise ValueError("RM65 pose-source attestation changed after binding.")
    if attestation.side != side or attestation.effector_name != effector_name:
        raise ValueError("RM65 pose-source side or effector identity is inconsistent.")
    identity_checks = {
        "pose_source_instance_id": attestation.source_instance_id,
        "pose_source_epoch_id": attestation.source_epoch_id,
        "hardware_pose_source_binding_hash": attestation.source_binding_hash,
        "hardware_measurement_clock_domain_id": (
            attestation.measurement_clock_domain_id
        ),
        "hardware_pose_source_attestation_status": "attested",
    }
    mismatches = {
        name: {"expected": expected, "actual": state.get(name)}
        for name, expected in identity_checks.items()
        if state.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "RM65 state identity differs from its pose-source attestation: "
            f"{mismatches!r}."
        )
    if not require_hardware_measurement:
        return attestation
    raw_sequence = state.get("hardware_measurement_sequence")
    if (
        isinstance(raw_sequence, bool)
        or not isinstance(raw_sequence, (int, np.integer))
        or int(raw_sequence) <= 0
    ):
        raise ValueError("RM65 startup measurement sequence must be positive.")
    if state.get("hardware_measurement_valid") is not True:
        raise ValueError("RM65 startup hardware measurement is not valid.")
    if state.get("hardware_read_success") is not True:
        raise ValueError("RM65 startup hardware read did not succeed.")
    try:
        hardware_qpos = np.asarray(state.get("hardware_qpos"), dtype=float).reshape(-1)
        measured_at = float(state.get("hardware_measurement_timestamp"))
    except (TypeError, ValueError) as exc:
        raise ValueError("RM65 startup hardware measurement is malformed.") from exc
    if (
        hardware_qpos.shape != (6,)
        or not np.all(np.isfinite(hardware_qpos))
        or not math.isfinite(measured_at)
    ):
        raise ValueError("RM65 startup hardware measurement is malformed.")
    return attestation


def _build_projection(
    *,
    attestation: RM65PoseSourceAttestation,
    receipt_clock: ClockDomainDescriptor,
    control_domain_id: str,
    bound_at: float,
) -> ArmPoseObservationProjection:
    side = attestation.side
    component_id = f"{side}_arm"
    plan = ExecutionObservationPlan(
        plan_id=(
            "policy-deploy:rm65-pose-observation:"
            f"{attestation.source_instance_id}:{attestation.source_epoch_id}"
        ),
        reference_mode="off",
        measurement_mode="full_frames",
        component_ids=(component_id,),
        required_representation_by_component={
            component_id: "cartesian_pose",
        },
    )
    capability = ExecutionObservationCapabilitySpec(
        capability_id=(
            f"rm65-pose-observation-capability:{attestation.semantic_content_hash}"
        ),
        target_handler_id=attestation.effector_name,
        clock_domain_id=attestation.measurement_clock_domain_id,
        supported_reference_modes=("off",),
        supported_measurement_modes=("off", "full_frames"),
        representation_by_component={component_id: "cartesian_pose"},
        schema_hash_by_component={component_id: RM65_TCP_POSE_SCHEMA_HASH_V1},
        measurement_stage_id=RM65_POSE_OBSERVATION_MEASUREMENT_STAGE_V1,
    )
    observation_binding = bind_execution_observation_plan(
        plan,
        capability,
        bound_at=bound_at,
    )
    coordinate_binding = PoseCoordinateBinding(
        binding_id=(
            f"policy-deploy:rm65-pose-coordinate:{observation_binding.binding_id}"
        ),
        source_instance_id=attestation.source_instance_id,
        source_epoch_id=attestation.source_epoch_id,
        source_binding_hash=attestation.source_binding_hash,
        control_domain_id=control_domain_id,
        component_id=component_id,
        stream_id=f"robot_state.{component_id}.tool_tcp_pose",
        reference_frame_id=attestation.reference_frame_id,
        reference_frame_spec_hash=attestation.reference_frame_spec_hash,
        moving_frame_id=attestation.moving_frame_id,
        tool_profile_hash=attestation.tool_profile_hash,
        pose_schema_hash=RM65_TCP_POSE_SCHEMA_HASH_V1,
        fk_contract_id=attestation.fk_contract_id,
        fk_contract_hash=attestation.fk_contract_hash,
        measurement_clock_domain_id=attestation.measurement_clock_domain_id,
        receipt_clock_domain_id=receipt_clock.clock_domain_id,
        observation_binding_id=observation_binding.binding_id,
        observation_binding_hash=observation_binding.semantic_content_hash,
        observation_plan_hash=observation_binding.plan_hash,
        source_kind=attestation.source_kind,
    )
    return ArmPoseObservationProjection(
        coordinate_binding=coordinate_binding,
        capability=capability,
        observation_binding=observation_binding,
    )


def _state_reader_for_binding(
    *,
    side: str,
    backend: EffectorBackend,
    attestation: RM65PoseSourceAttestation,
    coordinate_binding: PoseCoordinateBinding,
) -> ArmPoseStateReader:
    effector_name = attestation.effector_name

    def read_state(candidate_backend: EffectorBackend) -> ArmPoseStateReadResult:
        if candidate_backend is not backend:
            raise RuntimeError(
                f"{side} RM65 pose state reader received a different backend object."
            )
        state = _read_attested_state(candidate_backend)
        try:
            _validate_attested_state(
                state,
                side=side,
                effector_name=effector_name,
                expected_attestation=attestation,
                require_hardware_measurement=False,
            )
        except (TypeError, ValueError) as exc:
            return ArmPoseStateReadResult(
                state=state,
                coordinate_binding=None,
                error=(
                    f"rm65_pose_source_attestation_invalid:{type(exc).__name__}:{exc}"
                ),
            )
        return ArmPoseStateReadResult(
            state=state,
            coordinate_binding=coordinate_binding,
        )

    return read_state


def assemble_rm65_pose_observation(
    backends: Mapping[str, EffectorBackend],
    *,
    receipt_clock: ClockDomainDescriptor,
    control_domain_id: str,
    bound_at: float,
    receipt_monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
) -> RM65PoseObservationAssembly:
    """Validate all active sides and atomically return one reader configuration."""

    normalized_backends = dict(backends)
    if not normalized_backends:
        raise ValueError("RM65 pose observation requires at least one arm backend.")
    unknown_sides = sorted(set(normalized_backends) - {"left", "right"})
    if unknown_sides:
        raise ValueError(f"Unsupported RM65 pose observation sides: {unknown_sides}.")
    if not isinstance(receipt_clock, ClockDomainDescriptor):
        raise TypeError("receipt_clock must be a ClockDomainDescriptor.")
    if receipt_clock.monotonic is not True:
        raise ValueError("receipt_clock.monotonic must be exactly True.")
    control_domain = _required_text(control_domain_id, name="control_domain_id")
    normalized_bound_at = float(bound_at)
    if not math.isfinite(normalized_bound_at):
        raise ValueError("bound_at must be finite.")
    if not callable(receipt_monotonic_ns_clock):
        raise TypeError("receipt_monotonic_ns_clock must be callable.")

    projections: dict[str, ArmPoseObservationProjection] = {}
    state_readers: dict[str, ArmPoseStateReader] = {}
    for side, backend in normalized_backends.items():
        if not isinstance(backend, RM65ArmBackend):
            raise TypeError(
                f"{side} pose observation requires an RM65ArmBackend instance."
            )
        if backend.is_ready() is not True:
            raise RuntimeError(f"{side} RM65 backend is not ready for pose binding.")
        effector_name = _required_text(backend.name, name=f"{side} backend.name")
        expected_effector_name = f"{side}_arm"
        if effector_name != expected_effector_name:
            raise ValueError(f"{side} backend.name must be {expected_effector_name!r}.")
        initial_state = _read_attested_state(backend)
        attestation = _validate_attested_state(
            initial_state,
            side=side,
            effector_name=effector_name,
            expected_attestation=None,
            require_hardware_measurement=True,
        )
        projection = _build_projection(
            attestation=attestation,
            receipt_clock=receipt_clock,
            control_domain_id=control_domain,
            bound_at=normalized_bound_at,
        )
        projections[side] = projection
        state_readers[side] = _state_reader_for_binding(
            side=side,
            backend=backend,
            attestation=attestation,
            coordinate_binding=projection.coordinate_binding,
        )
    return RM65PoseObservationAssembly(
        projection_by_side=projections,
        state_read_by_side=state_readers,
        receipt_clock=receipt_clock,
        receipt_monotonic_ns_clock=receipt_monotonic_ns_clock,
    )


__all__ = [
    "LINUX_BOOT_ID_PATH",
    "RM65_POSE_OBSERVATION_MEASUREMENT_STAGE_V1",
    "RM65_TCP_POSE_SCHEMA_HASH_DOMAIN_V1",
    "RM65_TCP_POSE_SCHEMA_HASH_V1",
    "RM65PoseObservationAssembly",
    "assemble_rm65_pose_observation",
    "build_policy_deploy_receiver_clock",
]
