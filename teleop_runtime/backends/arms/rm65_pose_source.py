"""Immutable RM65-owned facts for one hardware-qpos FK pose source.

This module deliberately stops below :class:`PoseCoordinateBinding`.  The
backend can attest the exact source activation, model, solver frame, and TCP
semantics that it owns, while an application composition root remains
responsible for stream, observation-plan, and receiver-clock bindings.

Neither an RM65 address nor a local URDF path is retained.  The former enters
only a domain-separated endpoint hash and the latter contributes only its raw
file bytes.  File names, mtimes, profile notes, and arbitrary profile payloads
therefore cannot silently become coordinate identity.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.frames import (
    ARM_ENDPOINT_TO_HAND_BASE_Z_M,
    hand_base_to_arm_endpoint_transform,
)
from robot_motion.pose import pose6_to_matrix

HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY = "hardware_pose_source_attestation"

RM65_CONFIGURED_HARDWARE_ENDPOINT_HASH_DOMAIN_V1 = (
    "dexteleop.rm65.configured_hardware_endpoint.v1"
)
RM65_TOOL_TCP_CONTRACT_HASH_DOMAIN_V1 = "dexteleop.rm65.tool_tcp_contract.v1"
RM65_FORWARD_KINEMATICS_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.rm65.forward_kinematics_contract.v1"
)
RM65_SOLVER_MODEL_KINEMATICS_HASH_DOMAIN_V1 = (
    "dexteleop.rm65.solver_model_kinematics.v1"
)
RM65_POSE_SOURCE_BINDING_HASH_DOMAIN_V1 = "dexteleop.rm65.pose_source_binding.v1"
RM65_POSE_SOURCE_ATTESTATION_HASH_DOMAIN_V1 = (
    "dexteleop.rm65.pose_source_attestation.v1"
)

RM65_POSE_SOURCE_KIND_V1 = "real_robot_hardware_qpos_fk"
RM65_TOOL_TCP_FRAME_ID = "tool_tcp"
RM65_TCP_POSE_ENCODING_V1 = "flange_xyz_m_rpy_rad"
RM65_FK_CONTRACT_ID_V1 = "rm65.pinocchio.hardware_qpos_fk.v1"
RM65_TOOL_TCP_SOLVER_EEF_NAME_V1 = "hand_base"
RM65_HARDWARE_QPOS_SLOT_ORDER_V1 = tuple(
    f"rm65_hardware_joint_{index}" for index in range(1, 7)
)
RM65_HARDWARE_QPOS_UNIT_V1 = "deg"
RM65_FK_INPUT_UNIT_V1 = "rad"
RM65_SOLVER_JOINT_MOTION_V1 = "six_1d_revolute_joints"
RM65_HARDWARE_TO_SOLVER_CONVERSION_V1 = "q_rad=q_deg*pi/180;no_offset"
RM65_FK_OUTPUT_CONVENTION_TEMPLATE_V1 = "T_{side}_arm_base_tool_tcp"

_SHA256_PREFIX = "sha256:"
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_SOLVER_URDF_BINDING_ATTRIBUTE = "_rm65_pose_source_urdf_binding_v1"


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional_tool_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = _required_text(value, name="tool_name")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("tool_name must be a semantic name, not a filesystem path.")
    return normalized


def _side(value: object) -> str:
    normalized = _required_text(value, name="side").lower()
    if normalized not in {"left", "right"}:
        raise ValueError("side must be exactly 'left' or 'right'.")
    return normalized


def _finite_pose6(value: object, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a six-value numeric sequence.")
    try:
        normalized = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a six-value numeric sequence.") from exc
    if len(normalized) != 6:
        raise ValueError(f"{name} must contain exactly six values.")
    if any(not math.isfinite(item) for item in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


def _finite_matrix4(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite numeric 4x4 matrix.") from exc
    if matrix.size != 16:
        raise ValueError(f"{name} must contain exactly sixteen values.")
    matrix = matrix.reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{name} must be a homogeneous transform matrix.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _sha256_reference(value: object, *, name: str) -> str:
    normalized = _required_text(value, name=name)
    if not normalized.startswith(_SHA256_PREFIX):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex chars>.")
    digest = normalized.removeprefix(_SHA256_PREFIX)
    if len(digest) != 64 or any(
        character not in _LOWERCASE_HEX for character in digest
    ):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex chars>.")
    return normalized


def _prefixed_sha256(raw: bytes) -> str:
    return f"{_SHA256_PREFIX}{hashlib.sha256(raw).hexdigest()}"


def _prefixed_canonical_hash(payload: object, *, domain: str) -> str:
    return f"{_SHA256_PREFIX}{canonical_content_hash(payload, domain=domain)}"


@dataclass(frozen=True)
class RM65ToolTcpContract:
    """Exact flange-to-tool semantics used by the backend's pose projection."""

    side: str
    tool_name: str | None
    tcp_evidence: str
    tcp_pose_flange_m_rad: tuple[float, ...]
    configured_tcp_enabled: bool
    applied_solver_eef_to_tool_tcp_matrix: tuple[float, ...]
    moving_frame_id: str = field(default=RM65_TOOL_TCP_FRAME_ID, init=False)
    pose_encoding: str = field(default=RM65_TCP_POSE_ENCODING_V1, init=False)
    arm_endpoint_to_hand_base_z_m: float = field(
        default=ARM_ENDPOINT_TO_HAND_BASE_Z_M,
        init=False,
    )
    semantic_content_hash: str = field(init=False)
    tool_profile_hash: str = field(init=False)

    def __post_init__(self) -> None:
        side = _side(self.side)
        tool_name = _optional_tool_name(self.tool_name)
        tcp_evidence = _required_text(self.tcp_evidence, name="tcp_evidence")
        tcp_pose = _finite_pose6(
            self.tcp_pose_flange_m_rad,
            name="tcp_pose_flange_m_rad",
        )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tcp_evidence", tcp_evidence)
        object.__setattr__(self, "tcp_pose_flange_m_rad", tcp_pose)
        if not isinstance(self.configured_tcp_enabled, bool):
            raise TypeError("configured_tcp_enabled must be a bool.")
        applied_matrix = _finite_matrix4(
            self.applied_solver_eef_to_tool_tcp_matrix,
            name="applied_solver_eef_to_tool_tcp_matrix",
        )
        # The contract hashes the backend's actual matrix snapshot.  The
        # independent shared derivation is only a fail-closed consistency
        # check, so a future implementation drift cannot receive a plausible
        # but incorrect tool hash.
        expected_matrix = hand_base_to_arm_endpoint_transform() @ pose6_to_matrix(
            tcp_pose
        )
        expected_applied_matrix = (
            expected_matrix if self.configured_tcp_enabled else np.eye(4)
        )
        if not self.configured_tcp_enabled and not np.allclose(
            expected_matrix,
            np.eye(4),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "configured_tcp_enabled=False is inconsistent with the "
                "configured TCP pose."
            )
        if not np.allclose(
            applied_matrix,
            expected_applied_matrix,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "applied_solver_eef_to_tool_tcp_matrix does not match the "
                "backend's effective configured-TCP projection semantics."
            )
        object.__setattr__(
            self,
            "applied_solver_eef_to_tool_tcp_matrix",
            tuple(float(item) for item in applied_matrix.reshape(-1)),
        )

        semantic_hash = canonical_content_hash(
            self._semantic_payload(),
            domain=RM65_TOOL_TCP_CONTRACT_HASH_DOMAIN_V1,
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(self, "tool_profile_hash", f"sha256:{semantic_hash}")

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "tool_name": self.tool_name,
            "tcp_evidence": self.tcp_evidence,
            "tcp_pose_flange_m_rad": self.tcp_pose_flange_m_rad,
            "configured_tcp_enabled": self.configured_tcp_enabled,
            "moving_frame_id": self.moving_frame_id,
            "pose_encoding": self.pose_encoding,
            "arm_endpoint_to_hand_base_z_m": (self.arm_endpoint_to_hand_base_z_m),
            "applied_solver_eef_to_tool_tcp_matrix": (
                self.applied_solver_eef_to_tool_tcp_matrix
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
            "tool_profile_hash": self.tool_profile_hash,
        }


@dataclass(frozen=True)
class RM65ForwardKinematicsContract:
    """Exact Pinocchio/URDF facts used before the optional TCP transform."""

    urdf_raw_bytes_hash: str
    urdf_root_link: str
    non_fixed_joint_order: tuple[str, ...]
    solver_eef_name: str
    pinocchio_version: str
    solver_model_kinematics_hash: str
    tool_profile_hash: str
    output_convention: str
    fk_contract_id: str = field(default=RM65_FK_CONTRACT_ID_V1, init=False)
    hardware_qpos_slot_order: tuple[str, ...] = field(
        default=RM65_HARDWARE_QPOS_SLOT_ORDER_V1,
        init=False,
    )
    hardware_feedback_joint_unit: str = field(
        default=RM65_HARDWARE_QPOS_UNIT_V1,
        init=False,
    )
    input_joint_unit: str = field(default=RM65_FK_INPUT_UNIT_V1, init=False)
    solver_joint_motion: str = field(
        default=RM65_SOLVER_JOINT_MOTION_V1,
        init=False,
    )
    hardware_to_solver_conversion: str = field(
        default=RM65_HARDWARE_TO_SOLVER_CONVERSION_V1,
        init=False,
    )
    hardware_to_solver_joint_mapping: tuple[tuple[str, str], ...] = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        urdf_hash = _sha256_reference(
            self.urdf_raw_bytes_hash,
            name="urdf_raw_bytes_hash",
        )
        root_link = _required_text(self.urdf_root_link, name="urdf_root_link")
        if isinstance(self.non_fixed_joint_order, (str, bytes)):
            raise TypeError("non_fixed_joint_order must be a sequence of joint names.")
        try:
            joint_order = tuple(
                _required_text(name, name="non_fixed_joint_order item")
                for name in self.non_fixed_joint_order
            )
        except TypeError as exc:
            raise TypeError(
                "non_fixed_joint_order must be a sequence of joint names."
            ) from exc
        if len(joint_order) != 6:
            raise ValueError(
                "non_fixed_joint_order must contain exactly six RM65 solver joints."
            )
        if len(set(joint_order)) != len(joint_order):
            raise ValueError("non_fixed_joint_order must not contain duplicate names.")
        solver_eef_name = _required_text(
            self.solver_eef_name,
            name="solver_eef_name",
        )
        if solver_eef_name != RM65_TOOL_TCP_SOLVER_EEF_NAME_V1:
            raise ValueError(
                "solver_eef_name must be the canonical RM65 tool_tcp FK frame "
                f"{RM65_TOOL_TCP_SOLVER_EEF_NAME_V1!r}."
            )
        pinocchio_version = _required_text(
            self.pinocchio_version,
            name="pinocchio_version",
        )
        solver_model_kinematics_hash = _sha256_reference(
            self.solver_model_kinematics_hash,
            name="solver_model_kinematics_hash",
        )
        tool_profile_hash = _sha256_reference(
            self.tool_profile_hash,
            name="tool_profile_hash",
        )
        output_convention = _required_text(
            self.output_convention,
            name="output_convention",
        )

        object.__setattr__(self, "urdf_raw_bytes_hash", urdf_hash)
        object.__setattr__(self, "urdf_root_link", root_link)
        object.__setattr__(self, "non_fixed_joint_order", joint_order)
        object.__setattr__(self, "solver_eef_name", solver_eef_name)
        object.__setattr__(self, "pinocchio_version", pinocchio_version)
        object.__setattr__(
            self,
            "solver_model_kinematics_hash",
            solver_model_kinematics_hash,
        )
        object.__setattr__(self, "tool_profile_hash", tool_profile_hash)
        object.__setattr__(self, "output_convention", output_convention)
        object.__setattr__(
            self,
            "hardware_to_solver_joint_mapping",
            tuple(zip(self.hardware_qpos_slot_order, joint_order, strict=True)),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self._semantic_payload(),
                domain=RM65_FORWARD_KINEMATICS_CONTRACT_HASH_DOMAIN_V1,
            ),
        )

    @property
    def fk_contract_hash(self) -> str:
        return self.semantic_content_hash

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "fk_contract_id": self.fk_contract_id,
            "urdf_raw_bytes_hash": self.urdf_raw_bytes_hash,
            "urdf_root_link": self.urdf_root_link,
            "non_fixed_joint_order": self.non_fixed_joint_order,
            "solver_eef_name": self.solver_eef_name,
            "pinocchio_version": self.pinocchio_version,
            "solver_model_kinematics_hash": self.solver_model_kinematics_hash,
            # The backend's published FK pose is the solver result followed by
            # the exact hand_base->configured_tcp transform attested here.
            "tool_profile_hash": self.tool_profile_hash,
            "hardware_qpos_slot_order": self.hardware_qpos_slot_order,
            "hardware_feedback_joint_unit": self.hardware_feedback_joint_unit,
            "input_joint_unit": self.input_joint_unit,
            "solver_joint_motion": self.solver_joint_motion,
            "hardware_to_solver_conversion": self.hardware_to_solver_conversion,
            "hardware_to_solver_joint_mapping": (self.hardware_to_solver_joint_mapping),
            "output_convention": self.output_convention,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "semantic_content_hash": self.semantic_content_hash,
            "fk_contract_hash": self.fk_contract_hash,
        }


def _rm65_reference_frame_spec(
    *,
    side: str,
    urdf_root_link: str,
    urdf_raw_bytes_hash: str,
) -> CoordinateFrameSpec:
    normalized_side = _side(side)
    normalized_root = _required_text(urdf_root_link, name="urdf_root_link")
    normalized_urdf_hash = _sha256_reference(
        urdf_raw_bytes_hash,
        name="urdf_raw_bytes_hash",
    )
    reference_frame_id = f"{normalized_side}_arm_base"
    return CoordinateFrameSpec(
        frame_id=reference_frame_id,
        frame_version="1",
        semantic_parameters={
            "owner": "rm65_position_backend",
            "coordinate_source": "fixed_base_pinocchio_model_world",
            "urdf_root_link": normalized_root,
            "urdf_raw_bytes_hash": normalized_urdf_hash,
            "matrix_convention": "T_reference_frame_from_pinocchio_world",
            "reference_frame_from_pinocchio_world_matrix": tuple(
                float(item) for item in np.eye(4, dtype=np.float64).reshape(-1)
            ),
            "axis_contract": "urdf_root_xyz_right_handed",
            "translation_unit": "m",
            "rotation_unit": "rad",
        },
    )


@dataclass(frozen=True)
class RM65PoseSourceAttestation:
    """Lower-owned activation, frame, tool, FK, and source-clock binding."""

    side: str
    effector_name: str
    source_instance_id: str
    source_epoch_id: str
    configured_hardware_endpoint_hash: str
    reference_frame_spec: CoordinateFrameSpec
    tool_tcp_contract: RM65ToolTcpContract
    forward_kinematics_contract: RM65ForwardKinematicsContract
    measurement_clock_domain_id: str
    source_kind: str = field(default=RM65_POSE_SOURCE_KIND_V1, init=False)
    source_binding_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        side = _side(self.side)
        effector_name = _required_text(self.effector_name, name="effector_name")
        source_instance_id = _required_text(
            self.source_instance_id,
            name="source_instance_id",
        )
        source_epoch_id = _required_text(self.source_epoch_id, name="source_epoch_id")
        endpoint_hash = _sha256_reference(
            self.configured_hardware_endpoint_hash,
            name="configured_hardware_endpoint_hash",
        )
        if not isinstance(self.reference_frame_spec, CoordinateFrameSpec):
            raise TypeError("reference_frame_spec must be a CoordinateFrameSpec.")
        reference_frame_id = self.reference_frame_spec.frame_id
        expected_reference_frame_id = f"{side}_arm_base"
        if reference_frame_id != expected_reference_frame_id:
            raise ValueError(
                "reference_frame_id must match the canonical side-specific RM65 "
                f"base frame {expected_reference_frame_id!r}."
            )
        if not isinstance(self.tool_tcp_contract, RM65ToolTcpContract):
            raise TypeError("tool_tcp_contract must be an RM65ToolTcpContract.")
        if not isinstance(
            self.forward_kinematics_contract,
            RM65ForwardKinematicsContract,
        ):
            raise TypeError(
                "forward_kinematics_contract must be an RM65ForwardKinematicsContract."
            )
        expected_reference_frame_spec = _rm65_reference_frame_spec(
            side=side,
            urdf_root_link=self.forward_kinematics_contract.urdf_root_link,
            urdf_raw_bytes_hash=(self.forward_kinematics_contract.urdf_raw_bytes_hash),
        )
        if self.reference_frame_spec != expected_reference_frame_spec:
            raise ValueError(
                "reference_frame_spec must be the canonical fixed-base frame "
                "for the exact FK URDF contract."
            )
        if self.tool_tcp_contract.side != side:
            raise ValueError("tool_tcp_contract.side must match attestation side.")
        if (
            self.forward_kinematics_contract.tool_profile_hash
            != self.tool_tcp_contract.tool_profile_hash
        ):
            raise ValueError(
                "forward_kinematics_contract must bind the exact tool TCP contract."
            )
        expected_output_convention = RM65_FK_OUTPUT_CONVENTION_TEMPLATE_V1.format(
            side=side
        )
        if (
            self.forward_kinematics_contract.output_convention
            != expected_output_convention
        ):
            raise ValueError(
                "forward_kinematics_contract.output_convention must describe "
                f"the final {expected_output_convention}."
            )
        measurement_clock = _required_text(
            self.measurement_clock_domain_id,
            name="measurement_clock_domain_id",
        )

        object.__setattr__(self, "side", side)
        object.__setattr__(self, "effector_name", effector_name)
        object.__setattr__(self, "source_instance_id", source_instance_id)
        object.__setattr__(self, "source_epoch_id", source_epoch_id)
        object.__setattr__(self, "configured_hardware_endpoint_hash", endpoint_hash)
        object.__setattr__(
            self,
            "measurement_clock_domain_id",
            measurement_clock,
        )

        source_binding_hash = canonical_content_hash(
            self._source_binding_payload(),
            domain=RM65_POSE_SOURCE_BINDING_HASH_DOMAIN_V1,
        )
        object.__setattr__(self, "source_binding_hash", source_binding_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self._semantic_payload(),
                domain=RM65_POSE_SOURCE_ATTESTATION_HASH_DOMAIN_V1,
            ),
        )

    @property
    def reference_frame_id(self) -> str:
        return self.reference_frame_spec.frame_id

    @property
    def reference_frame_spec_hash(self) -> str:
        return self.reference_frame_spec.semantic_content_hash

    @property
    def moving_frame_id(self) -> str:
        return self.tool_tcp_contract.moving_frame_id

    @property
    def tool_profile_hash(self) -> str:
        return self.tool_tcp_contract.tool_profile_hash

    @property
    def fk_contract_id(self) -> str:
        return self.forward_kinematics_contract.fk_contract_id

    @property
    def fk_contract_hash(self) -> str:
        return self.forward_kinematics_contract.fk_contract_hash

    def _source_binding_payload(self) -> dict[str, Any]:
        return {
            "source_instance_id": self.source_instance_id,
            "source_epoch_id": self.source_epoch_id,
            "side": self.side,
            "effector_name": self.effector_name,
            "configured_hardware_endpoint_hash": (
                self.configured_hardware_endpoint_hash
            ),
            "source_kind": self.source_kind,
        }

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            **self._source_binding_payload(),
            "source_binding_hash": self.source_binding_hash,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "moving_frame_id": self.moving_frame_id,
            "tool_profile_hash": self.tool_profile_hash,
            "fk_contract_id": self.fk_contract_id,
            "fk_contract_hash": self.fk_contract_hash,
            "measurement_clock_domain_id": self.measurement_clock_domain_id,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._semantic_payload(),
            "tool_tcp_contract": self.tool_tcp_contract.as_dict(),
            "forward_kinematics_contract": (self.forward_kinematics_contract.as_dict()),
            "reference_frame_spec": self.reference_frame_spec.to_payload(),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class _UrdfFacts:
    raw_bytes_hash: str
    root_link: str
    non_fixed_joint_order: tuple[str, ...]
    non_fixed_joint_types: tuple[str, ...]
    link_names: frozenset[str]


@dataclass(frozen=True)
class _SolverModelFacts:
    model_identity: int
    joint_order: tuple[str, ...]
    frame_names: tuple[str, ...]
    eef_frame_id: int
    nq: int
    nv: int
    joint_model_types: tuple[str, ...]
    joint_idx_qs: tuple[int, ...]
    joint_nqs: tuple[int, ...]
    joint_idx_vs: tuple[int, ...]
    joint_nvs: tuple[int, ...]
    kinematic_model_hash: str


@dataclass(frozen=True)
class _SolverUrdfBinding:
    urdf: _UrdfFacts
    model: _SolverModelFacts
    solver_eef_name: str
    pinocchio_version: str
    compute_pose_token: tuple[int | None, int]


def _placement_matrix(value: object, *, name: str) -> tuple[float, ...]:
    matrix_value = getattr(value, "homogeneous", None)
    if matrix_value is None:
        raise TypeError(f"{name}.homogeneous is required for exact kinematics.")
    matrix = _finite_matrix4(matrix_value, name=f"{name}.homogeneous")
    return tuple(float(item) for item in matrix.reshape(-1))


def _solver_model_kinematic_payload(
    model: object,
    *,
    names: tuple[str, ...],
    frame_names: tuple[str, ...],
) -> dict[str, Any]:
    try:
        parents = tuple(int(value) for value in model.parents)  # type: ignore[attr-defined]
        idx_qs = tuple(int(value) for value in model.idx_qs)  # type: ignore[attr-defined]
        nqs = tuple(int(value) for value in model.nqs)  # type: ignore[attr-defined]
        idx_vs = tuple(int(value) for value in model.idx_vs)  # type: ignore[attr-defined]
        nvs = tuple(int(value) for value in model.nvs)  # type: ignore[attr-defined]
        joints = tuple(model.joints)  # type: ignore[attr-defined]
        joint_placements = tuple(model.jointPlacements)  # type: ignore[attr-defined]
        frames = tuple(model.frames)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "solver.model lacks exact Pinocchio joint/frame kinematic metadata."
        ) from exc
    joint_count = len(names)
    if any(
        len(values) != joint_count
        for values in (
            parents,
            idx_qs,
            nqs,
            idx_vs,
            nvs,
            joints,
            joint_placements,
        )
    ):
        raise ValueError("solver.model joint metadata lengths are inconsistent.")
    if len(frames) != len(frame_names):
        raise ValueError("solver.model frame metadata lengths are inconsistent.")
    joint_payload: list[dict[str, Any]] = []
    for index, joint in enumerate(joints):
        shortname = getattr(joint, "shortname", None)
        if not callable(shortname):
            raise TypeError("solver.model joint must provide shortname().")
        joint_payload.append(
            {
                "index": index,
                "name": names[index],
                "parent": parents[index],
                "joint_model": _required_text(
                    str(shortname()),
                    name="solver joint shortname",
                ),
                "idx_q": idx_qs[index],
                "nq": nqs[index],
                "idx_v": idx_vs[index],
                "nv": nvs[index],
                "placement": _placement_matrix(
                    joint_placements[index],
                    name=f"solver.model.jointPlacements[{index}]",
                ),
            }
        )
    frame_payload: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        try:
            parent_joint = int(frame.parentJoint)
            parent_frame = int(frame.parentFrame)
            frame_type = _required_text(str(frame.type), name="solver frame type")
            placement = _placement_matrix(
                frame.placement,
                name=f"solver.model.frames[{index}].placement",
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "solver.model frame lacks exact Pinocchio kinematic metadata."
            ) from exc
        frame_payload.append(
            {
                "index": index,
                "name": frame_names[index],
                "parent_joint": parent_joint,
                "parent_frame": parent_frame,
                "frame_type": frame_type,
                "placement": placement,
            }
        )
    return {
        "nq": int(model.nq),  # type: ignore[attr-defined]
        "nv": int(model.nv),  # type: ignore[attr-defined]
        "joints": tuple(joint_payload),
        "frames": tuple(frame_payload),
    }


def _solver_model_facts(solver: object) -> _SolverModelFacts:
    model = getattr(solver, "model", None)
    if model is None:
        raise TypeError("solver.model is required for exact URDF provenance.")
    try:
        raw_model_names = tuple(str(name) for name in model.names)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "solver.model.names is required for exact joint-order provenance."
        ) from exc
    if not raw_model_names or raw_model_names[0] != "universe":
        raise ValueError("solver.model.names must begin with Pinocchio 'universe'.")
    joint_order = raw_model_names[1:]
    try:
        frame_names = tuple(str(frame.name) for frame in model.frames)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "solver.model.frames is required for exact frame provenance."
        ) from exc
    eef_name = _required_text(
        getattr(solver, "eef_name", None),
        name="solver.eef_name",
    )
    eef_frame_id = getattr(solver, "eef_frame_id", None)
    if (
        isinstance(eef_frame_id, bool)
        or not isinstance(eef_frame_id, int)
        or eef_frame_id < 0
        or eef_frame_id >= len(frame_names)
        or frame_names[eef_frame_id] != eef_name
    ):
        raise ValueError(
            "solver.eef_frame_id must identify solver.eef_name in the actual model."
        )
    try:
        nq = int(model.nq)
        nv = int(model.nv)
        joint_model_types = tuple(str(joint.shortname()) for joint in model.joints)
        joint_idx_qs = tuple(int(value) for value in model.idx_qs)
        joint_nqs = tuple(int(value) for value in model.nqs)
        joint_idx_vs = tuple(int(value) for value in model.idx_vs)
        joint_nvs = tuple(int(value) for value in model.nvs)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "solver.model.nq/nv are required for exact qpos semantics."
        ) from exc
    kinematic_payload = _solver_model_kinematic_payload(
        model,
        names=raw_model_names,
        frame_names=frame_names,
    )
    return _SolverModelFacts(
        model_identity=id(model),
        joint_order=joint_order,
        frame_names=frame_names,
        eef_frame_id=eef_frame_id,
        nq=nq,
        nv=nv,
        joint_model_types=joint_model_types,
        joint_idx_qs=joint_idx_qs,
        joint_nqs=joint_nqs,
        joint_idx_vs=joint_idx_vs,
        joint_nvs=joint_nvs,
        kinematic_model_hash=_prefixed_canonical_hash(
            kinematic_payload,
            domain=RM65_SOLVER_MODEL_KINEMATICS_HASH_DOMAIN_V1,
        ),
    )


def _bind_rm65_solver_urdf_provenance(
    *,
    solver: object,
    urdf_path: str | Path,
    expected_raw_bytes: bytes,
) -> None:
    """Bind one canonical solver instance to the bytes used at construction.

    The backend captures ``expected_raw_bytes`` before calling the Pinocchio
    constructor.  This function rereads after construction and refuses a
    changed source, then binds only immutable semantic facts to the solver.  No
    filesystem path or raw payload is retained.
    """

    if not isinstance(expected_raw_bytes, bytes):
        raise TypeError("expected_raw_bytes must be construction-time bytes.")
    urdf = _urdf_facts(urdf_path)
    if urdf.raw_bytes_hash != _prefixed_sha256(expected_raw_bytes):
        raise ValueError("RM65 URDF changed while the solver was being constructed.")
    model = _solver_model_facts(solver)
    if model.nq != 6 or model.nv != 6:
        raise ValueError("RM65 pose solver model must have nq=nv=6.")
    if len(urdf.non_fixed_joint_order) != 6:
        raise ValueError(
            "RM65 pose-source URDF must have exactly six non-fixed joints."
        )
    if any(joint_type != "revolute" for joint_type in urdf.non_fixed_joint_types):
        raise ValueError("RM65 pose-source URDF joints must all be revolute.")
    if model.joint_order != urdf.non_fixed_joint_order:
        raise ValueError(
            "solver.model joint order does not match the exact source URDF order."
        )
    supported_revolute_models = {
        "JointModelRX",
        "JointModelRY",
        "JointModelRZ",
    }
    if (
        len(model.joint_model_types) != 7
        or any(
            joint_type not in supported_revolute_models
            for joint_type in model.joint_model_types[1:]
        )
        or model.joint_idx_qs[1:] != tuple(range(6))
        or model.joint_nqs[1:] != (1,) * 6
        or model.joint_idx_vs[1:] != tuple(range(6))
        or model.joint_nvs[1:] != (1,) * 6
    ):
        raise ValueError(
            "RM65 pose solver must expose six one-DoF revolute Pinocchio "
            "joints with contiguous q/v indices."
        )
    solver_eef_name = _required_text(
        getattr(solver, "eef_name", None),
        name="solver.eef_name",
    )
    if solver_eef_name not in urdf.link_names:
        raise ValueError(
            "solver.eef_name must identify a link in the exact source URDF."
        )
    if model.frame_names.count(solver_eef_name) != 1:
        raise ValueError("solver.model must expose exactly one solver.eef_name frame.")
    root_frame_indices = tuple(
        index
        for index, frame_name in enumerate(model.frame_names)
        if frame_name == urdf.root_link
    )
    if len(root_frame_indices) != 1:
        raise ValueError(
            "solver.model must expose exactly one exact URDF root link frame."
        )
    root_frame = tuple(solver.model.frames)[root_frame_indices[0]]
    if int(root_frame.parentJoint) != 0 or not np.array_equal(
        np.asarray(root_frame.placement.homogeneous, dtype=np.float64),
        np.eye(4, dtype=np.float64),
    ):
        raise ValueError(
            "solver.model URDF root frame must be fixed at Pinocchio world identity."
        )
    binding = _SolverUrdfBinding(
        urdf=urdf,
        model=model,
        solver_eef_name=solver_eef_name,
        pinocchio_version=_solver_pinocchio_version(solver, None),
        compute_pose_token=_solver_compute_pose_token(solver),
    )
    existing = getattr(solver, _SOLVER_URDF_BINDING_ATTRIBUTE, None)
    if existing is not None and existing != binding:
        raise ValueError("solver already carries a different RM65 URDF binding.")
    setattr(solver, _SOLVER_URDF_BINDING_ATTRIBUTE, binding)


def _bound_solver_urdf_provenance(solver: object) -> _SolverUrdfBinding:
    binding = getattr(solver, _SOLVER_URDF_BINDING_ATTRIBUTE, None)
    if not isinstance(binding, _SolverUrdfBinding):
        raise TypeError(
            "solver lacks construction-time RM65 URDF provenance; refusing attestation."
        )
    actual_model = _solver_model_facts(solver)
    if actual_model != binding.model:
        raise ValueError("solver model/frame identity changed after URDF binding.")
    actual_version = _solver_pinocchio_version(solver, None)
    if actual_version != binding.pinocchio_version:
        raise ValueError("solver Pinocchio version changed after URDF binding.")
    if getattr(solver, "eef_name", None) != binding.solver_eef_name:
        raise ValueError("solver.eef_name changed after URDF binding.")
    if _solver_compute_pose_token(solver) != binding.compute_pose_token:
        raise ValueError("solver.compute_pose changed after URDF binding.")
    return binding


def _solver_compute_pose_token(solver: object) -> tuple[int | None, int]:
    """Return process-local callable identity for construction-time drift checks."""

    compute_pose = getattr(solver, "compute_pose", None)
    if not callable(compute_pose):
        raise TypeError("solver.compute_pose must be callable for exact FK provenance.")
    bound_owner = getattr(compute_pose, "__self__", None)
    implementation = getattr(compute_pose, "__func__", compute_pose)
    return (
        None if bound_owner is None else id(bound_owner),
        id(implementation),
    )


def _urdf_facts(urdf_path: str | Path) -> _UrdfFacts:
    if not isinstance(urdf_path, (str, Path)):
        raise TypeError("urdf_path must be a string or pathlib.Path.")
    path = Path(urdf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"RM65 URDF does not exist: {path}")
    raw = path.read_bytes()
    try:
        robot = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("RM65 URDF is not well-formed XML.") from exc
    if _local_name(robot.tag) != "robot":
        raise ValueError("RM65 URDF root element must be <robot>.")

    link_names: list[str] = []
    joint_names: list[str] = []
    non_fixed_joint_order: list[str] = []
    non_fixed_joint_types: list[str] = []
    child_links: list[str] = []
    for element in robot:
        kind = _local_name(element.tag)
        if kind == "link":
            link_names.append(_required_xml_attribute(element, "name", owner="link"))
            continue
        if kind != "joint":
            continue
        joint_name = _required_xml_attribute(element, "name", owner="joint")
        joint_type = _required_xml_attribute(
            element, "type", owner=f"joint {joint_name!r}"
        )
        parent = _one_child_element(element, "parent", owner=f"joint {joint_name!r}")
        child = _one_child_element(element, "child", owner=f"joint {joint_name!r}")
        parent_link = _required_xml_attribute(
            parent,
            "link",
            owner=f"joint {joint_name!r} parent",
        )
        child_link = _required_xml_attribute(
            child,
            "link",
            owner=f"joint {joint_name!r} child",
        )
        if parent_link == child_link:
            raise ValueError(f"RM65 URDF joint {joint_name!r} cannot parent itself.")
        joint_names.append(joint_name)
        child_links.append(child_link)
        if joint_type != "fixed":
            non_fixed_joint_order.append(joint_name)
            non_fixed_joint_types.append(joint_type)

    if not link_names:
        raise ValueError("RM65 URDF must declare at least one link.")
    if len(set(link_names)) != len(link_names):
        raise ValueError("RM65 URDF link names must be unique.")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("RM65 URDF joint names must be unique.")
    if len(set(child_links)) != len(child_links):
        raise ValueError("RM65 URDF links must not have multiple parent joints.")

    link_set = frozenset(link_names)
    unknown_children = set(child_links) - link_set
    parent_links = {
        _required_xml_attribute(
            _one_child_element(joint, "parent", owner="joint"),
            "link",
            owner="joint parent",
        )
        for joint in robot
        if _local_name(joint.tag) == "joint"
    }
    unknown_parents = parent_links - link_set
    if unknown_parents or unknown_children:
        raise ValueError(
            "RM65 URDF joints must reference declared links; "
            f"unknown_parents={sorted(unknown_parents)!r}, "
            f"unknown_children={sorted(unknown_children)!r}."
        )

    roots = [link for link in link_names if link not in set(child_links)]
    if len(roots) != 1:
        raise ValueError(
            f"RM65 URDF must have exactly one root link; found={tuple(roots)!r}."
        )
    if not non_fixed_joint_order:
        raise ValueError("RM65 URDF must declare at least one non-fixed joint.")
    return _UrdfFacts(
        raw_bytes_hash=_prefixed_sha256(raw),
        root_link=roots[0],
        non_fixed_joint_order=tuple(non_fixed_joint_order),
        non_fixed_joint_types=tuple(non_fixed_joint_types),
        link_names=link_set,
    )


def _local_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _required_xml_attribute(element: ET.Element, name: str, *, owner: str) -> str:
    value = element.attrib.get(name)
    try:
        return _required_text(value, name=f"{owner} {name}")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"RM65 URDF {owner} requires a non-empty {name!r} attribute."
        ) from exc


def _one_child_element(element: ET.Element, name: str, *, owner: str) -> ET.Element:
    matches = [child for child in element if _local_name(child.tag) == name]
    if len(matches) != 1:
        raise ValueError(
            f"RM65 URDF {owner} must contain exactly one <{name}> element."
        )
    return matches[0]


def _solver_pinocchio_version(
    solver: object,
    supplied_version: str | None,
) -> str:
    solver_pin = getattr(solver, "pin", None)
    actual_value = getattr(solver_pin, "__version__", None)
    actual = (
        None
        if actual_value is None
        else _required_text(
            str(actual_value),
            name="solver.pin.__version__",
        )
    )
    if supplied_version is None:
        if actual is None:
            raise ValueError(
                "pinocchio_version is required when solver.pin.__version__ is unavailable."
            )
        return actual
    supplied = _required_text(supplied_version, name="pinocchio_version")
    if actual is not None and supplied != actual:
        raise ValueError(
            "pinocchio_version must match the actual solver.pin.__version__."
        )
    return actual if actual is not None else supplied


def build_rm65_pose_source_attestation(
    *,
    side: str,
    effector_name: str,
    source_instance_id: str,
    source_epoch_id: str,
    arm_ip: str,
    tcp_pose_flange_m_rad: Sequence[float],
    tcp_evidence: str,
    tool_name: str | None,
    configured_tcp_enabled: bool,
    applied_solver_eef_to_tool_tcp_matrix: Sequence[float] | np.ndarray,
    solver: object,
    measurement_clock_domain_id: str = "local_wall_time",
) -> RM65PoseSourceAttestation:
    """Build one lower attestation from the backend's resolved runtime facts.

    The solver must already carry the construction-time provenance installed by
    the backend's private construction-time provenance binder.  No current filesystem path is
    reread here, so a cached solver cannot be mislabeled after a config/path
    change.  Tool semantics hash the backend's effective applied matrix rather
    than a separately recomputed replacement.
    """

    normalized_side = _side(side)
    normalized_effector = _required_text(effector_name, name="effector_name")
    normalized_instance = _required_text(
        source_instance_id,
        name="source_instance_id",
    )
    normalized_epoch = _required_text(source_epoch_id, name="source_epoch_id")
    normalized_arm_ip = _required_text(arm_ip, name="arm_ip")
    solver_binding = _bound_solver_urdf_provenance(solver)
    solver_eef_name = solver_binding.solver_eef_name
    if solver_eef_name != RM65_TOOL_TCP_SOLVER_EEF_NAME_V1:
        raise ValueError(
            "tool_tcp attestation requires the actual solver eef_name to be "
            f"{RM65_TOOL_TCP_SOLVER_EEF_NAME_V1!r}; got {solver_eef_name!r}."
        )

    endpoint_hash = _prefixed_canonical_hash(
        {
            "transport": "rm65_direct",
            "side": normalized_side,
            "arm_ip": normalized_arm_ip,
        },
        domain=RM65_CONFIGURED_HARDWARE_ENDPOINT_HASH_DOMAIN_V1,
    )
    tool_tcp = RM65ToolTcpContract(
        side=normalized_side,
        tool_name=tool_name,
        tcp_evidence=tcp_evidence,
        tcp_pose_flange_m_rad=tuple(tcp_pose_flange_m_rad),
        configured_tcp_enabled=configured_tcp_enabled,
        applied_solver_eef_to_tool_tcp_matrix=tuple(
            np.asarray(
                applied_solver_eef_to_tool_tcp_matrix,
                dtype=np.float64,
            ).reshape(-1)
        ),
    )
    fk = RM65ForwardKinematicsContract(
        urdf_raw_bytes_hash=solver_binding.urdf.raw_bytes_hash,
        urdf_root_link=solver_binding.urdf.root_link,
        non_fixed_joint_order=solver_binding.urdf.non_fixed_joint_order,
        solver_eef_name=solver_eef_name,
        pinocchio_version=solver_binding.pinocchio_version,
        solver_model_kinematics_hash=(solver_binding.model.kinematic_model_hash),
        tool_profile_hash=tool_tcp.tool_profile_hash,
        output_convention=RM65_FK_OUTPUT_CONVENTION_TEMPLATE_V1.format(
            side=normalized_side
        ),
    )
    reference_frame_spec = _rm65_reference_frame_spec(
        side=normalized_side,
        urdf_root_link=fk.urdf_root_link,
        urdf_raw_bytes_hash=fk.urdf_raw_bytes_hash,
    )
    return RM65PoseSourceAttestation(
        side=normalized_side,
        effector_name=normalized_effector,
        source_instance_id=normalized_instance,
        source_epoch_id=normalized_epoch,
        configured_hardware_endpoint_hash=endpoint_hash,
        reference_frame_spec=reference_frame_spec,
        tool_tcp_contract=tool_tcp,
        forward_kinematics_contract=fk,
        measurement_clock_domain_id=measurement_clock_domain_id,
    )


__all__ = [
    "HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY",
    "RM65_CONFIGURED_HARDWARE_ENDPOINT_HASH_DOMAIN_V1",
    "RM65_FK_CONTRACT_ID_V1",
    "RM65_FORWARD_KINEMATICS_CONTRACT_HASH_DOMAIN_V1",
    "RM65_POSE_SOURCE_ATTESTATION_HASH_DOMAIN_V1",
    "RM65_POSE_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RM65_POSE_SOURCE_KIND_V1",
    "RM65_SOLVER_JOINT_MOTION_V1",
    "RM65_SOLVER_MODEL_KINEMATICS_HASH_DOMAIN_V1",
    "RM65_TOOL_TCP_CONTRACT_HASH_DOMAIN_V1",
    "RM65_TOOL_TCP_SOLVER_EEF_NAME_V1",
    "RM65ForwardKinematicsContract",
    "RM65PoseSourceAttestation",
    "RM65ToolTcpContract",
    "build_rm65_pose_source_attestation",
]
