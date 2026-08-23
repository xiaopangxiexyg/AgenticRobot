from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.frames import hand_base_to_arm_endpoint_transform
from robot_motion.pose import pose6_to_matrix
from teleop_runtime.backends.arms import rm65 as rm65_module
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_pose_source import (
    HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY,
    RM65_CONFIGURED_HARDWARE_ENDPOINT_HASH_DOMAIN_V1,
    RM65_FORWARD_KINEMATICS_CONTRACT_HASH_DOMAIN_V1,
    RM65_POSE_SOURCE_ATTESTATION_HASH_DOMAIN_V1,
    RM65_POSE_SOURCE_BINDING_HASH_DOMAIN_V1,
    RM65_SOLVER_MODEL_KINEMATICS_HASH_DOMAIN_V1,
    RM65_TOOL_TCP_CONTRACT_HASH_DOMAIN_V1,
    RM65ForwardKinematicsContract,
    RM65PoseSourceAttestation,
    RM65ToolTcpContract,
    _bind_rm65_solver_urdf_provenance,
    build_rm65_pose_source_attestation,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_URDF = b"""<?xml version="1.0"?>
<robot name="rm65-test">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="link3"/>
  <link name="link4"/>
  <link name="link5"/>
  <link name="hand_base"/>
  <link name="camera_mount"/>
  <link name="actual_tip"/>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
  </joint>
  <joint name="joint_2" type="revolute">
    <parent link="link1"/><child link="link2"/>
  </joint>
  <joint name="joint_3" type="revolute">
    <parent link="link2"/><child link="link3"/>
  </joint>
  <joint name="joint_4" type="revolute">
    <parent link="link3"/><child link="link4"/>
  </joint>
  <joint name="joint_5" type="revolute">
    <parent link="link4"/><child link="link5"/>
  </joint>
  <joint name="joint_6" type="revolute">
    <parent link="link5"/><child link="hand_base"/>
  </joint>
  <joint name="camera_fixed" type="fixed">
    <parent link="hand_base"/><child link="camera_mount"/>
  </joint>
  <joint name="tip_fixed" type="fixed">
    <parent link="hand_base"/><child link="actual_tip"/>
  </joint>
</robot>
"""

_JOINT_ORDER = tuple(f"joint_{index}" for index in range(1, 7))
_FRAME_NAMES = (
    "universe",
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "hand_base",
    "camera_mount",
    "actual_tip",
)


def _write_urdf(path: Path, payload: bytes = _URDF) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class _FakeJoint:
    def __init__(self, name: str) -> None:
        self._name = name

    def shortname(self) -> str:
        return self._name


def _placement(*, z: float = 0.0) -> SimpleNamespace:
    matrix = np.eye(4, dtype=float)
    matrix[2, 3] = z
    return SimpleNamespace(homogeneous=matrix)


def _solver(
    *,
    eef_name: str = "hand_base",
    version: str = "3.2.0",
    joint_order: tuple[str, ...] = _JOINT_ORDER,
) -> object:
    frames = tuple(
        SimpleNamespace(
            name=name,
            parentJoint=0 if name in {"universe", "base_link"} else min(index, 6),
            parentFrame=max(0, index - 1),
            type="BODY",
            placement=_placement(
                z=0.0 if name in {"universe", "base_link"} else 0.001 * index
            ),
        )
        for index, name in enumerate(_FRAME_NAMES)
    )
    model_names = ("universe", *joint_order)
    model = SimpleNamespace(
        names=model_names,
        frames=frames,
        nq=6,
        nv=6,
        parents=(0, 0, 1, 2, 3, 4, 5),
        idx_qs=(0, 0, 1, 2, 3, 4, 5),
        nqs=(0, 1, 1, 1, 1, 1, 1),
        idx_vs=(0, 0, 1, 2, 3, 4, 5),
        nvs=(0, 1, 1, 1, 1, 1, 1),
        joints=tuple(
            _FakeJoint("JointModelRX" if index == 0 else "JointModelRZ")
            for index in range(7)
        ),
        jointPlacements=tuple(_placement(z=0.01 * index) for index in range(7)),
    )
    return SimpleNamespace(
        eef_name=eef_name,
        eef_frame_id=_FRAME_NAMES.index(eef_name) if eef_name in _FRAME_NAMES else 999,
        pin=SimpleNamespace(__version__=version),
        model=model,
        compute_pose=lambda _qpos: np.eye(4, dtype=float),
    )


def _applied_tcp_matrix(
    tcp_pose_flange_m_rad: tuple[float, ...],
    *,
    configured_tcp_enabled: bool = True,
) -> tuple[float, ...]:
    if not configured_tcp_enabled:
        return tuple(float(item) for item in np.eye(4).reshape(-1))
    matrix = hand_base_to_arm_endpoint_transform() @ pose6_to_matrix(
        tcp_pose_flange_m_rad
    )
    return tuple(float(item) for item in matrix.reshape(-1))


def _build(urdf_path: Path, **overrides: object) -> RM65PoseSourceAttestation:
    values: dict[str, object] = {
        "side": "right",
        "effector_name": "right_arm",
        "source_instance_id": "rm65-right-runtime-1",
        "source_epoch_id": "rm65-right-epoch-4",
        "arm_ip": "192.168.1.18",
        "tcp_pose_flange_m_rad": (0.01, -0.02, 0.30, 0.1, -0.2, 0.3),
        "tcp_evidence": "active_tool_profile",
        "tool_name": "inspire_right_v1",
        "solver": _solver(),
        "configured_tcp_enabled": True,
        "measurement_clock_domain_id": "local_wall_time",
    }
    values.update(overrides)
    solver = values["solver"]
    if not hasattr(solver, "_rm65_pose_source_urdf_binding_v1"):
        _bind_rm65_solver_urdf_provenance(
            solver=solver,
            urdf_path=urdf_path,
            expected_raw_bytes=urdf_path.read_bytes(),
        )
    if "applied_solver_eef_to_tool_tcp_matrix" not in values:
        tcp_pose = tuple(values["tcp_pose_flange_m_rad"])  # type: ignore[arg-type]
        values["applied_solver_eef_to_tool_tcp_matrix"] = (
            _applied_tcp_matrix(
                tcp_pose,  # type: ignore[arg-type]
                configured_tcp_enabled=bool(values["configured_tcp_enabled"]),
            )
            if len(tcp_pose) == 6 and all(np.isfinite(float(item)) for item in tcp_pose)
            else tuple(float(item) for item in np.eye(4).reshape(-1))
        )
    return build_rm65_pose_source_attestation(**values)  # type: ignore[arg-type]


def _semantic_tool_payload(contract: RM65ToolTcpContract) -> dict[str, object]:
    return {
        "side": contract.side,
        "tool_name": contract.tool_name,
        "tcp_evidence": contract.tcp_evidence,
        "tcp_pose_flange_m_rad": contract.tcp_pose_flange_m_rad,
        "configured_tcp_enabled": contract.configured_tcp_enabled,
        "moving_frame_id": contract.moving_frame_id,
        "pose_encoding": contract.pose_encoding,
        "arm_endpoint_to_hand_base_z_m": contract.arm_endpoint_to_hand_base_z_m,
        "applied_solver_eef_to_tool_tcp_matrix": (
            contract.applied_solver_eef_to_tool_tcp_matrix
        ),
    }


def _semantic_fk_payload(contract: RM65ForwardKinematicsContract) -> dict[str, object]:
    return {
        "fk_contract_id": contract.fk_contract_id,
        "urdf_raw_bytes_hash": contract.urdf_raw_bytes_hash,
        "urdf_root_link": contract.urdf_root_link,
        "non_fixed_joint_order": contract.non_fixed_joint_order,
        "solver_eef_name": contract.solver_eef_name,
        "pinocchio_version": contract.pinocchio_version,
        "solver_model_kinematics_hash": contract.solver_model_kinematics_hash,
        "tool_profile_hash": contract.tool_profile_hash,
        "hardware_qpos_slot_order": contract.hardware_qpos_slot_order,
        "hardware_feedback_joint_unit": contract.hardware_feedback_joint_unit,
        "input_joint_unit": contract.input_joint_unit,
        "solver_joint_motion": contract.solver_joint_motion,
        "hardware_to_solver_conversion": contract.hardware_to_solver_conversion,
        "hardware_to_solver_joint_mapping": (contract.hardware_to_solver_joint_mapping),
        "output_convention": contract.output_convention,
    }


def test_builder_emits_narrow_typed_lower_attestation(tmp_path: Path) -> None:
    attestation = _build(_write_urdf(tmp_path / "model.urdf"))

    assert HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY == (
        "hardware_pose_source_attestation"
    )
    assert attestation.side == "right"
    assert attestation.reference_frame_id == "right_arm_base"
    assert attestation.moving_frame_id == "tool_tcp"
    assert attestation.source_kind == "real_robot_hardware_qpos_fk"
    assert attestation.measurement_clock_domain_id == "local_wall_time"
    assert attestation.forward_kinematics_contract.urdf_root_link == "base_link"
    assert attestation.forward_kinematics_contract.non_fixed_joint_order == (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    )
    assert attestation.forward_kinematics_contract.solver_eef_name == "hand_base"
    assert attestation.forward_kinematics_contract.pinocchio_version == "3.2.0"
    assert attestation.forward_kinematics_contract.hardware_feedback_joint_unit == (
        "deg"
    )
    assert attestation.forward_kinematics_contract.input_joint_unit == "rad"
    assert (
        len(attestation.forward_kinematics_contract.hardware_to_solver_joint_mapping)
        == 6
    )
    assert attestation.forward_kinematics_contract.output_convention == (
        "T_right_arm_base_tool_tcp"
    )
    assert attestation.tool_profile_hash.startswith("sha256:")
    assert len(attestation.tool_profile_hash) == len("sha256:") + 64
    assert attestation.fk_contract_hash == (
        attestation.forward_kinematics_contract.semantic_content_hash
    )
    assert attestation.forward_kinematics_contract.tool_profile_hash == (
        attestation.tool_profile_hash
    )

    applied_tcp = attestation.tool_tcp_contract.applied_solver_eef_to_tool_tcp_matrix
    assert applied_tcp[3] == pytest.approx(0.01)
    assert applied_tcp[7] == pytest.approx(-0.02)
    # 0.30m flange TCP minus the shared +0.009m flange->hand_base offset,
    # applied exactly once by the backend projection.
    assert applied_tcp[11] == pytest.approx(0.291)
    assert isinstance(attestation.reference_frame_spec, CoordinateFrameSpec)
    assert attestation.reference_frame_spec_hash == (
        attestation.reference_frame_spec.semantic_content_hash
    )
    assert (
        attestation.reference_frame_spec.semantic_parameters["matrix_convention"]
        == "T_reference_frame_from_pinocchio_world"
    )

    encoded = json.dumps(attestation.as_dict(), sort_keys=True)
    assert "192.168.1.18" not in encoded
    assert str(tmp_path) not in encoded
    forbidden_keys = {
        "arm_ip",
        "urdf_path",
        "path",
        "mtime",
        "notes",
        "payload",
    }
    assert forbidden_keys.isdisjoint(attestation.as_dict())
    assert forbidden_keys.isdisjoint(attestation.tool_tcp_contract.as_dict())
    assert forbidden_keys.isdisjoint(attestation.forward_kinematics_contract.as_dict())


def test_domain_separated_hashes_cover_exact_semantic_payloads(tmp_path: Path) -> None:
    attestation = _build(_write_urdf(tmp_path / "model.urdf"))
    tool = attestation.tool_tcp_contract
    fk = attestation.forward_kinematics_contract

    expected_tool_hash = canonical_content_hash(
        _semantic_tool_payload(tool),
        domain=RM65_TOOL_TCP_CONTRACT_HASH_DOMAIN_V1,
    )
    assert tool.semantic_content_hash == expected_tool_hash
    assert tool.tool_profile_hash == f"sha256:{expected_tool_hash}"
    assert fk.semantic_content_hash == canonical_content_hash(
        _semantic_fk_payload(fk),
        domain=RM65_FORWARD_KINEMATICS_CONTRACT_HASH_DOMAIN_V1,
    )

    source_payload = {
        "source_instance_id": attestation.source_instance_id,
        "source_epoch_id": attestation.source_epoch_id,
        "side": attestation.side,
        "effector_name": attestation.effector_name,
        "configured_hardware_endpoint_hash": (
            attestation.configured_hardware_endpoint_hash
        ),
        "source_kind": attestation.source_kind,
    }
    assert attestation.source_binding_hash == canonical_content_hash(
        source_payload,
        domain=RM65_POSE_SOURCE_BINDING_HASH_DOMAIN_V1,
    )
    aggregate_payload = {
        **source_payload,
        "source_binding_hash": attestation.source_binding_hash,
        "reference_frame_id": attestation.reference_frame_id,
        "reference_frame_spec_hash": attestation.reference_frame_spec_hash,
        "moving_frame_id": attestation.moving_frame_id,
        "tool_profile_hash": attestation.tool_profile_hash,
        "fk_contract_id": attestation.fk_contract_id,
        "fk_contract_hash": attestation.fk_contract_hash,
        "measurement_clock_domain_id": attestation.measurement_clock_domain_id,
    }
    assert attestation.semantic_content_hash == canonical_content_hash(
        aggregate_payload,
        domain=RM65_POSE_SOURCE_ATTESTATION_HASH_DOMAIN_V1,
    )
    assert (
        len(
            {
                tool.semantic_content_hash,
                fk.semantic_content_hash,
                attestation.source_binding_hash,
                attestation.semantic_content_hash,
            }
        )
        == 4
    )


def test_identical_urdf_bytes_are_path_and_mtime_independent(tmp_path: Path) -> None:
    first_path = _write_urdf(tmp_path / "first" / "arm.urdf")
    second_path = _write_urdf(tmp_path / "relocated" / "renamed.urdf")
    os.utime(second_path, (1_000_000_000, 1_000_000_000))

    first = _build(first_path)
    second = _build(second_path)

    assert first == second
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.forward_kinematics_contract.urdf_raw_bytes_hash == (
        second.forward_kinematics_contract.urdf_raw_bytes_hash
    )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("effector_name", "right_arm_shadow"),
        ("source_instance_id", "rm65-right-runtime-2"),
        ("source_epoch_id", "rm65-right-epoch-5"),
        ("arm_ip", "192.168.1.19"),
        ("tcp_pose_flange_m_rad", (0.02, -0.02, 0.30, 0.1, -0.2, 0.3)),
        ("tcp_evidence", "configured_tcp_pose"),
        ("tool_name", "inspire_right_v2"),
        ("measurement_clock_domain_id", "rm65-controller-clock"),
        ("solver", _solver(version="3.3.0")),
    ],
)
def test_each_runtime_fact_changes_attestation_hash(
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    baseline = _build(urdf_path)
    changed = _build(urdf_path, **{override: value})

    assert changed.semantic_content_hash != baseline.semantic_content_hash


def test_side_changes_reference_and_source_bindings(tmp_path: Path) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    right = _build(urdf_path)
    left = _build(
        urdf_path,
        side="left",
        effector_name="left_arm",
        source_instance_id="rm65-left-runtime-1",
        source_epoch_id="rm65-left-epoch-4",
    )

    assert left.reference_frame_id == "left_arm_base"
    assert left.reference_frame_spec_hash != right.reference_frame_spec_hash
    assert left.source_binding_hash != right.source_binding_hash
    assert left.semantic_content_hash != right.semantic_content_hash


def test_changed_raw_urdf_bytes_change_fk_and_reference_hashes(tmp_path: Path) -> None:
    first_path = _write_urdf(tmp_path / "first.urdf")
    changed_bytes = _URDF.replace(
        b'<robot name="rm65-test">', b'<robot name="rm65-test-v2">'
    )
    second_path = _write_urdf(tmp_path / "second.urdf", changed_bytes)

    first = _build(first_path)
    second = _build(second_path)

    assert first.forward_kinematics_contract.urdf_raw_bytes_hash != (
        second.forward_kinematics_contract.urdf_raw_bytes_hash
    )
    assert first.fk_contract_hash != second.fk_contract_hash
    assert first.reference_frame_spec_hash != second.reference_frame_spec_hash
    assert first.semantic_content_hash != second.semantic_content_hash


def test_solver_provenance_is_construction_time_and_path_change_independent(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    original_bytes = urdf_path.read_bytes()
    solver = _solver()
    _bind_rm65_solver_urdf_provenance(
        solver=solver,
        urdf_path=urdf_path,
        expected_raw_bytes=original_bytes,
    )
    original = _build(urdf_path, solver=solver)

    changed_bytes = _URDF.replace(
        b'<robot name="rm65-test">', b'<robot name="changed-after-build">'
    )
    _write_urdf(urdf_path, changed_bytes)
    after_path_change = _build(urdf_path, solver=solver)

    assert after_path_change == original
    assert after_path_change.forward_kinematics_contract.urdf_raw_bytes_hash == (
        original.forward_kinematics_contract.urdf_raw_bytes_hash
    )


def test_solver_provenance_rejects_missing_or_changed_construction_bytes(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    original_bytes = urdf_path.read_bytes()

    with pytest.raises(TypeError, match="construction-time bytes"):
        _bind_rm65_solver_urdf_provenance(
            solver=_solver(),
            urdf_path=urdf_path,
            expected_raw_bytes=None,  # type: ignore[arg-type]
        )

    _write_urdf(
        urdf_path,
        _URDF.replace(b"rm65-test", b"rm65-mutated-during-build"),
    )
    with pytest.raises(
        ValueError, match="changed while the solver was being constructed"
    ):
        _bind_rm65_solver_urdf_provenance(
            solver=_solver(),
            urdf_path=urdf_path,
            expected_raw_bytes=original_bytes,
        )


def test_backend_solver_construction_binds_the_exact_pre_read_urdf(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    solver = _solver()
    solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.20",
            side="right",
            arm_urdf_path=urdf_path,
            connect_hardware=False,
            auto_load_tool_profile=False,
        )
    )

    with mock.patch.object(
        rm65_module,
        "build_rm65_single_arm_ik_solver",
        return_value=solver,
    ) as build_solver:
        resolved = backend._solver_for_tcp_frame("tool_tcp")

    assert resolved is solver
    build_solver.assert_called_once()
    attestation = _build(urdf_path, solver=resolved)
    assert attestation.forward_kinematics_contract.urdf_raw_bytes_hash == (
        f"sha256:{hashlib.sha256(_URDF).hexdigest()}"
    )


def test_backend_solver_construction_keeps_legacy_fk_when_urdf_drifts(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    solver = _solver()
    solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.20",
            side="right",
            arm_urdf_path=urdf_path,
            connect_hardware=False,
            auto_load_tool_profile=False,
        )
    )

    def mutate_during_build(_config: object) -> object:
        _write_urdf(
            urdf_path,
            _URDF.replace(b"rm65-test", b"changed-during-builder-call"),
        )
        return solver

    with mock.patch.object(
        rm65_module,
        "build_rm65_single_arm_ik_solver",
        side_effect=mutate_during_build,
    ):
        resolved = backend._solver_for_tcp_frame("tool_tcp")

    assert resolved is solver
    assert "changed while the solver was being constructed" in str(
        resolved._rm65_pose_source_provenance_error
    )
    np.testing.assert_allclose(
        backend.compute_tcp_pose(qpos=np.zeros(6, dtype=float)),
        hand_base_to_arm_endpoint_transform(),
    )
    tcp_pose = (0.01, -0.02, 0.30, 0.1, -0.2, 0.3)
    with pytest.raises(
        TypeError,
        match="lacks construction-time RM65 URDF provenance",
    ):
        build_rm65_pose_source_attestation(
            side="right",
            effector_name="right_arm",
            source_instance_id="source",
            source_epoch_id="epoch",
            arm_ip="192.0.2.20",
            tcp_pose_flange_m_rad=tcp_pose,
            tcp_evidence="configured_tcp_pose",
            tool_name=None,
            configured_tcp_enabled=True,
            applied_solver_eef_to_tool_tcp_matrix=_applied_tcp_matrix(tcp_pose),
            solver=resolved,
        )


def test_unbound_or_mutated_solver_cannot_attest(tmp_path: Path) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    tcp_pose = (0.01, -0.02, 0.30, 0.1, -0.2, 0.3)
    with pytest.raises(TypeError, match="lacks construction-time RM65 URDF provenance"):
        build_rm65_pose_source_attestation(
            side="right",
            effector_name="right_arm",
            source_instance_id="source",
            source_epoch_id="epoch",
            arm_ip="192.0.2.1",
            tcp_pose_flange_m_rad=tcp_pose,
            tcp_evidence="configured_tcp_pose",
            tool_name=None,
            configured_tcp_enabled=True,
            applied_solver_eef_to_tool_tcp_matrix=_applied_tcp_matrix(tcp_pose),
            solver=_solver(),
        )

    solver = _solver()
    _bind_rm65_solver_urdf_provenance(
        solver=solver,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    solver.model.jointPlacements[2].homogeneous[0, 3] = 0.125
    with pytest.raises(ValueError, match="model/frame identity changed"):
        _build(urdf_path, solver=solver)

    solver = _solver()
    _bind_rm65_solver_urdf_provenance(
        solver=solver,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    with pytest.raises(ValueError, match="compute_pose changed after URDF binding"):
        _build(urdf_path, solver=solver)


def test_solver_binding_rejects_nonidentity_root_and_nonrevolute_joint(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    shifted_root_solver = _solver()
    shifted_root_solver.model.frames[1].placement.homogeneous[0, 3] = 0.1
    with pytest.raises(ValueError, match="world identity"):
        _bind_rm65_solver_urdf_provenance(
            solver=shifted_root_solver,
            urdf_path=urdf_path,
            expected_raw_bytes=urdf_path.read_bytes(),
        )

    prismatic_urdf = _URDF.replace(
        b'<joint name="joint_1" type="revolute">',
        b'<joint name="joint_1" type="prismatic">',
    )
    prismatic_path = _write_urdf(tmp_path / "prismatic.urdf", prismatic_urdf)
    with pytest.raises(ValueError, match="must all be revolute"):
        _bind_rm65_solver_urdf_provenance(
            solver=_solver(),
            urdf_path=prismatic_path,
            expected_raw_bytes=prismatic_path.read_bytes(),
        )

    nonrevolute_solver = _solver()
    joints = list(nonrevolute_solver.model.joints)
    joints[1] = _FakeJoint("JointModelPX")
    nonrevolute_solver.model.joints = tuple(joints)
    with pytest.raises(ValueError, match="six one-DoF revolute"):
        _bind_rm65_solver_urdf_provenance(
            solver=nonrevolute_solver,
            urdf_path=urdf_path,
            expected_raw_bytes=urdf_path.read_bytes(),
        )

    for index_field in ("idx_qs", "idx_vs"):
        noncontiguous_solver = _solver()
        setattr(
            noncontiguous_solver.model,
            index_field,
            (0, 0, 1, 2, 3, 5, 4),
        )
        with pytest.raises(ValueError, match="contiguous q/v indices"):
            _bind_rm65_solver_urdf_provenance(
                solver=noncontiguous_solver,
                urdf_path=urdf_path,
                expected_raw_bytes=urdf_path.read_bytes(),
            )

    duplicate_eef_solver = _solver()
    duplicate_eef_solver.model.frames[-1].name = "hand_base"
    with pytest.raises(ValueError, match="exactly one solver.eef_name frame"):
        _bind_rm65_solver_urdf_provenance(
            solver=duplicate_eef_solver,
            urdf_path=urdf_path,
            expected_raw_bytes=urdf_path.read_bytes(),
        )


def test_actual_solver_kinematics_are_hashed_into_fk_contract(tmp_path: Path) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    baseline_solver = _solver()
    changed_solver = _solver()
    changed_solver.model.jointPlacements[2].homogeneous[0, 3] = 0.125

    baseline = _build(urdf_path, solver=baseline_solver)
    changed = _build(urdf_path, solver=changed_solver)

    assert RM65_SOLVER_MODEL_KINEMATICS_HASH_DOMAIN_V1.endswith(
        "solver_model_kinematics.v1"
    )
    assert baseline.forward_kinematics_contract.solver_model_kinematics_hash != (
        changed.forward_kinematics_contract.solver_model_kinematics_hash
    )
    assert baseline.fk_contract_hash != changed.fk_contract_hash


def test_builder_requires_hand_base_and_rejects_bound_solver_version_drift(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    with pytest.raises(ValueError, match="requires the actual solver eef_name"):
        _build(urdf_path, solver=_solver(eef_name="link1"))

    solver = _solver(version="3.2.0")
    _bind_rm65_solver_urdf_provenance(
        solver=solver,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    solver.pin.__version__ = "changed-after-binding"

    with pytest.raises(ValueError, match="version changed after URDF binding"):
        _build(urdf_path, solver=solver)


def test_solver_binding_requires_actual_pinocchio_version(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    solver = _solver()
    del solver.pin.__version__

    with pytest.raises(ValueError, match="pinocchio_version is required"):
        _build(urdf_path, solver=solver)


def test_contracts_are_immutable(tmp_path: Path) -> None:
    attestation = _build(_write_urdf(tmp_path / "model.urdf"))

    with pytest.raises(FrozenInstanceError):
        attestation.source_epoch_id = "replacement"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        attestation.tool_tcp_contract.tool_name = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("override", "error", "match"),
    [
        ({"side": "center"}, ValueError, "side"),
        ({"effector_name": "  "}, ValueError, "effector_name"),
        ({"source_instance_id": ""}, ValueError, "source_instance_id"),
        ({"source_epoch_id": None}, TypeError, "source_epoch_id"),
        ({"arm_ip": ""}, ValueError, "arm_ip"),
        ({"tcp_pose_flange_m_rad": (0.0,) * 5}, ValueError, "exactly six"),
        (
            {"tcp_pose_flange_m_rad": (0.0, 0.0, 0.0, 0.0, 0.0, float("nan"))},
            ValueError,
            "finite",
        ),
        ({"tcp_evidence": ""}, ValueError, "tcp_evidence"),
        ({"tool_name": "profiles/tool.json"}, ValueError, "filesystem path"),
        ({"measurement_clock_domain_id": ""}, ValueError, "measurement_clock"),
        (
            {"solver": SimpleNamespace(pin=SimpleNamespace(__version__="3.2.0"))},
            TypeError,
            "solver.model",
        ),
        (
            {"solver": _solver(eef_name="not_in_urdf")},
            ValueError,
            "eef_frame_id",
        ),
        (
            {"applied_solver_eef_to_tool_tcp_matrix": tuple(np.eye(3).reshape(-1))},
            ValueError,
            "sixteen",
        ),
        (
            {"configured_tcp_enabled": False},
            ValueError,
            "configured_tcp_enabled=False",
        ),
    ],
)
def test_builder_rejects_invalid_runtime_facts(
    tmp_path: Path,
    override: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    with pytest.raises(error, match=match):
        _build(urdf_path, **override)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not xml", "well-formed XML"),
        (b"<not_robot><link name='base'/></not_robot>", "root element"),
        (
            b"<robot name='r'><link name='a'/><link name='b'/></robot>",
            "exactly one root link",
        ),
        (
            b"""<robot name='r'>
              <link name='base'/><link name='tip'/>
              <joint name='fixed' type='fixed'>
                <parent link='base'/><child link='tip'/>
              </joint>
            </robot>""",
            "non-fixed joint",
        ),
        (
            b"""<robot name='r'>
              <link name='base'/><link name='tip'/>
              <joint name='j1' type='revolute'>
                <parent link='missing'/><child link='tip'/>
              </joint>
            </robot>""",
            "declared links",
        ),
    ],
)
def test_builder_rejects_invalid_urdf_structure(
    tmp_path: Path,
    payload: bytes,
    match: str,
) -> None:
    urdf_path = _write_urdf(tmp_path / "invalid.urdf", payload)
    with pytest.raises(ValueError, match=match):
        _build(urdf_path, solver=_solver(eef_name="tip"))


def test_builder_rejects_missing_urdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _bind_rm65_solver_urdf_provenance(
            solver=_solver(),
            urdf_path=tmp_path / "missing.urdf",
            expected_raw_bytes=b"",
        )


def test_direct_contract_construction_rejects_malformed_hashes_and_types(
    tmp_path: Path,
) -> None:
    attestation = _build(_write_urdf(tmp_path / "model.urdf"))

    with pytest.raises(ValueError, match="sha256"):
        replace(
            attestation.forward_kinematics_contract,
            urdf_raw_bytes_hash="ABCDEF",
        )
    with pytest.raises(ValueError, match="canonical RM65 tool_tcp FK frame"):
        replace(
            attestation.forward_kinematics_contract,
            solver_eef_name="arm_link6",
        )
    with pytest.raises(TypeError, match="tool_tcp_contract"):
        replace(attestation, tool_tcp_contract=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical fixed-base frame"):
        replace(
            attestation,
            reference_frame_spec=CoordinateFrameSpec(
                frame_id="right_arm_base",
                frame_version="1",
                semantic_parameters={"unbound": True},
            ),
        )


def test_unnamed_tool_is_an_explicit_stable_semantic_value(tmp_path: Path) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    first = _build(urdf_path, tool_name=None, tcp_evidence="configured_tcp_pose")
    second = _build(urdf_path, tool_name=None, tcp_evidence="configured_tcp_pose")

    assert first.tool_tcp_contract.tool_name is None
    assert first == second


class _LifecycleFakeArm:
    def __init__(
        self,
        state_result: tuple[object, ...],
        *,
        tool_read_error: Exception | None = None,
        tool_name: str = "empty",
        state_entered: threading.Event | None = None,
        state_release: threading.Event | None = None,
    ) -> None:
        self.state_result = state_result
        self.tool_read_error = tool_read_error
        self.tool_name = tool_name
        self.state_entered = state_entered
        self.state_release = state_release
        self.state_read_calls = 0
        self.tool_read_calls = 0
        self.uninit_calls = 0
        self.socket_close_calls = 0

    def Get_Current_Arm_State(self) -> tuple[object, ...]:
        self.state_read_calls += 1
        if self.state_entered is not None:
            self.state_entered.set()
        if self.state_release is not None and not self.state_release.wait(timeout=5.0):
            raise RuntimeError("test timed out waiting to release state read")
        return self.state_result

    def Get_Current_Tool_Frame(self) -> tuple[int, dict[str, str]]:
        self.tool_read_calls += 1
        if self.tool_read_error is not None:
            raise self.tool_read_error
        return 0, {"name": self.tool_name}

    def RM_API_UnInit(self) -> int:
        self.uninit_calls += 1
        return 0

    def Arm_Socket_Close(self) -> int:
        self.socket_close_calls += 1
        return 0


def _hardware_backend_for_attestation(
    urdf_path: Path,
    *,
    auto_load_tool_profile: bool = False,
) -> RM65ArmBackend:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.20",
            side="right",
            arm_urdf_path=urdf_path,
            connect_hardware=True,
            observation_only=True,
            auto_load_tool_profile=auto_load_tool_profile,
            hardware_read_retries=0,
            singularity_protection_enabled=False,
        ),
        pose_source_instance_id="rm65-right-test-instance",
    )
    solver = _solver()
    solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    _bind_rm65_solver_urdf_provenance(
        solver=solver,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    backend._solvers["tool_tcp"] = solver
    return backend


def _start_with_arm(backend: RM65ArmBackend, arm: _LifecycleFakeArm) -> None:
    with mock.patch.object(
        rm65_module,
        "_load_rm65_sdk",
        return_value=(lambda _model, _ip: arm, object()),
    ):
        backend.start()


def _successful_state_result() -> tuple[object, ...]:
    return (0, [0.0] * 6, [0.0] * 6, 0, 0)


def test_hardware_start_attests_existing_fk_without_a_second_sdk_read(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    arm = _LifecycleFakeArm(_successful_state_result())
    backend = _hardware_backend_for_attestation(urdf_path)

    _start_with_arm(backend, arm)
    state = backend.get_state(include_pose_source_attestation=True)

    attestation = state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
    assert isinstance(attestation, RM65PoseSourceAttestation)
    assert state["hardware_pose_source_attestation_status"] == "attested"
    assert state["hardware_pose_source_attestation_error"] is None
    assert state["hardware_pose_source_binding_hash"] == (
        attestation.source_binding_hash
    )
    assert np.array_equal(
        np.asarray(
            attestation.tool_tcp_contract.applied_solver_eef_to_tool_tcp_matrix
        ).reshape(4, 4),
        hand_base_to_arm_endpoint_transform(),
    )
    assert arm.state_read_calls == 1
    assert arm.tool_read_calls == 0

    backend.close()


def test_repeated_start_preserves_attestation_identity_and_close_clears_it(
    tmp_path: Path,
) -> None:
    arm = _LifecycleFakeArm(_successful_state_result())
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))

    _start_with_arm(backend, arm)
    before = backend.get_state()
    before_attestation = before[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
    before_epoch = before["pose_source_epoch_id"]
    _start_with_arm(backend, arm)
    after = backend.get_state()

    assert after[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is before_attestation
    assert after["pose_source_epoch_id"] == before_epoch
    assert arm.state_read_calls == 1

    backend.close()
    closed = backend.get_state()
    assert closed[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert closed["pose_source_epoch_id"] is None
    assert closed["hardware_pose_source_attestation_status"] == "unavailable"
    assert closed["hardware_pose_source_attestation_error"] == "closed"


def test_failed_start_retry_uses_a_fresh_epoch_for_one_explicit_instance(
    tmp_path: Path,
) -> None:
    failed_arm = _LifecycleFakeArm((7, None, None, 0, 0))
    retry_arm = _LifecycleFakeArm(_successful_state_result())
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arms = iter((failed_arm, retry_arm))
    uuid_values = iter(
        (
            SimpleNamespace(hex="failed-start-epoch"),
            SimpleNamespace(hex="retry-start-epoch"),
        )
    )

    with (
        mock.patch.object(
            rm65_module,
            "_load_rm65_sdk",
            return_value=(lambda _model, _ip: next(arms), object()),
        ),
        mock.patch.object(
            rm65_module, "uuid4", side_effect=lambda: next(uuid_values)
        ) as uuid4,
    ):
        with pytest.raises(RuntimeError, match="trusted 6D qpos"):
            backend.start()
        failed_state = backend.get_state()
        assert failed_state["pose_source_epoch_id"] is None
        assert failed_state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None

        backend.start()

    retried = backend.get_state(include_pose_source_attestation=True)
    assert retried["pose_source_instance_id"] == "rm65-right-test-instance"
    assert retried["pose_source_epoch_id"] == (
        "rm65-right-test-instance:attach:retry-start-epoch"
    )
    assert isinstance(
        retried[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )
    assert uuid4.call_count == 2
    assert failed_arm.uninit_calls == 1
    assert failed_arm.socket_close_calls == 1

    backend.close()


def test_unresolved_auto_loaded_tool_keeps_legacy_fk_but_withholds_attestation(
    tmp_path: Path,
) -> None:
    arm = _LifecycleFakeArm(
        _successful_state_result(),
        tool_read_error=RuntimeError("test tool read unavailable"),
    )
    backend = _hardware_backend_for_attestation(
        _write_urdf(tmp_path / "model.urdf"),
        auto_load_tool_profile=True,
    )

    _start_with_arm(backend, arm)
    state = backend.get_state(include_pose_source_attestation=True)

    assert arm.state_read_calls == 1
    assert arm.tool_read_calls == 1
    assert state["current_tcp_pose"] is not None
    assert state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert state["hardware_pose_source_attestation_status"] == "unavailable"
    assert state["hardware_pose_source_attestation_error"] == (
        "active_tool_frame_read_failed"
    )

    backend.close()


def _tool_profile(tool_name: str, *, tcp_z_m: float) -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "tcp": {
            "pose_flange_m_rad": (0.0, 0.0, tcp_z_m, 0.0, 0.0, 0.0),
        },
        "_profile_path": f"ignored/{tool_name}.json",
    }


def test_failed_start_retry_rechecks_changed_sdk_tool_binding(
    tmp_path: Path,
) -> None:
    first_arm = _LifecycleFakeArm(
        (7, None, None, 0, 0),
        tool_name="tool_a",
    )
    retry_arm = _LifecycleFakeArm(
        _successful_state_result(),
        tool_name="tool_b",
    )
    backend = _hardware_backend_for_attestation(
        _write_urdf(tmp_path / "model.urdf"),
        auto_load_tool_profile=True,
    )
    arms = iter((first_arm, retry_arm))

    def load_profile(tool_name: str, *, arm: str) -> dict[str, object]:
        assert arm == "right"
        return _tool_profile(
            tool_name,
            tcp_z_m=0.20 if tool_name == "tool_a" else 0.30,
        )

    with (
        mock.patch.object(
            rm65_module,
            "_load_rm65_sdk",
            return_value=(lambda _model, _ip: next(arms), object()),
        ),
        mock.patch(
            "hardware.rm65.calibration_profiles.load_profile",
            side_effect=load_profile,
        ),
    ):
        with pytest.raises(RuntimeError, match="trusted 6D qpos"):
            backend.start()
        assert backend.config.tcp_pose_flange_m_rad is None
        assert backend._active_tcp_pose_flange_m_rad() == pytest.approx(
            (0.0, 0.0, 0.20, 0.0, 0.0, 0.0)
        )
        backend.start()

    state = backend.get_state(include_pose_source_attestation=True)
    attestation = state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
    assert isinstance(attestation, RM65PoseSourceAttestation)
    assert attestation.tool_tcp_contract.tool_name == "tool_b"
    assert attestation.tool_tcp_contract.tcp_pose_flange_m_rad == pytest.approx(
        (0.0, 0.0, 0.30, 0.0, 0.0, 0.0)
    )
    assert first_arm.tool_read_calls == 1
    assert retry_arm.tool_read_calls == 1
    backend.close()


def test_failed_start_retry_unreadable_tool_clears_old_sdk_tcp_proof(
    tmp_path: Path,
) -> None:
    first_arm = _LifecycleFakeArm(
        (7, None, None, 0, 0),
        tool_name="tool_a",
    )
    retry_arm = _LifecycleFakeArm(
        _successful_state_result(),
        tool_read_error=RuntimeError("retry tool unavailable"),
    )
    backend = _hardware_backend_for_attestation(
        _write_urdf(tmp_path / "model.urdf"),
        auto_load_tool_profile=True,
    )
    arms = iter((first_arm, retry_arm))

    with (
        mock.patch.object(
            rm65_module,
            "_load_rm65_sdk",
            return_value=(lambda _model, _ip: next(arms), object()),
        ),
        mock.patch(
            "hardware.rm65.calibration_profiles.load_profile",
            return_value=_tool_profile("tool_a", tcp_z_m=0.20),
        ),
    ):
        with pytest.raises(RuntimeError, match="trusted 6D qpos"):
            backend.start()
        backend.start()

    state = backend.get_state()
    assert first_arm.tool_read_calls == 1
    assert retry_arm.tool_read_calls == 1
    assert backend.config.tcp_pose_flange_m_rad is None
    assert state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert state["hardware_pose_source_attestation_error"] == (
        "active_tool_frame_read_failed"
    )
    backend.close()


@pytest.mark.parametrize("second_tool_readable", [True, False])
def test_reused_config_never_carries_sdk_tool_binding_to_a_new_backend(
    tmp_path: Path,
    second_tool_readable: bool,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    shared_config = RM65ArmBackendConfig(
        arm_ip="192.0.2.20",
        side="right",
        arm_urdf_path=urdf_path,
        connect_hardware=True,
        observation_only=True,
        auto_load_tool_profile=True,
        hardware_read_retries=0,
        singularity_protection_enabled=False,
    )

    def backend_with_solver(instance_id: str) -> RM65ArmBackend:
        backend = RM65ArmBackend(
            shared_config,
            pose_source_instance_id=instance_id,
        )
        solver = _solver()
        solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
        _bind_rm65_solver_urdf_provenance(
            solver=solver,
            urdf_path=urdf_path,
            expected_raw_bytes=urdf_path.read_bytes(),
        )
        backend._solvers["tool_tcp"] = solver
        return backend

    first_backend = backend_with_solver("shared-config-first")
    first_arm = _LifecycleFakeArm(_successful_state_result(), tool_name="tool_a")
    second_backend: RM65ArmBackend | None = None

    def load_profile(tool_name: str, *, arm: str) -> dict[str, object]:
        assert arm == "right"
        return _tool_profile(
            tool_name,
            tcp_z_m=0.20 if tool_name == "tool_a" else 0.30,
        )

    with mock.patch(
        "hardware.rm65.calibration_profiles.load_profile",
        side_effect=load_profile,
    ):
        _start_with_arm(first_backend, first_arm)
        first_state = first_backend.get_state(include_pose_source_attestation=True)
        first_attestation = first_state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
        assert isinstance(first_attestation, RM65PoseSourceAttestation)
        assert first_attestation.tool_tcp_contract.tool_name == "tool_a"
        first_backend.close()

        assert shared_config.tcp_pose_flange_m_rad is None
        second_backend = backend_with_solver("shared-config-second")
        second_arm = _LifecycleFakeArm(
            _successful_state_result(),
            tool_name="tool_b",
            tool_read_error=(
                None
                if second_tool_readable
                else RuntimeError("second attachment tool unavailable")
            ),
        )
        _start_with_arm(second_backend, second_arm)
        second_state = second_backend.get_state(include_pose_source_attestation=True)

    assert first_arm.tool_read_calls == 1
    assert second_arm.tool_read_calls == 1
    assert shared_config.tcp_pose_flange_m_rad is None
    second_attestation = second_state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
    if second_tool_readable:
        assert isinstance(second_attestation, RM65PoseSourceAttestation)
        assert second_attestation.tool_tcp_contract.tool_name == "tool_b"
        assert second_attestation.tool_tcp_contract.tcp_pose_flange_m_rad == (
            pytest.approx((0.0, 0.0, 0.30, 0.0, 0.0, 0.0))
        )
    else:
        assert second_attestation is None
        assert second_state["hardware_pose_source_attestation_error"] == (
            "active_tool_frame_read_failed"
        )
    assert second_backend is not None
    second_backend.close()


def test_published_attestation_fails_closed_after_solver_model_drift(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    initial = backend.get_state(include_pose_source_attestation=True)
    assert isinstance(
        initial[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )

    solver = backend._solvers["tool_tcp"]
    solver.model.jointPlacements[2].homogeneous[0, 3] = 0.125
    drifted = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )

    assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert drifted["hardware_pose_source_attestation_status"] == "unavailable"
    assert str(drifted["hardware_pose_source_attestation_error"]).startswith(
        "attestation_drift_detected:"
    )
    backend.close()


def test_concurrent_solver_cache_miss_builds_one_shared_solver(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    with backend._solver_lock:
        backend._solvers.clear()
    built_solver = _solver()
    built_solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    builder_entered = threading.Event()
    release_builder = threading.Event()
    second_started = threading.Event()
    call_lock = threading.Lock()
    builder_call_count = 0
    results: list[object] = []
    errors: list[BaseException] = []

    def barrier_builder(_config: object) -> object:
        nonlocal builder_call_count
        with call_lock:
            builder_call_count += 1
        builder_entered.set()
        if not release_builder.wait(timeout=5.0):
            raise RuntimeError("test timed out releasing solver builder")
        return built_solver

    def resolve_solver(*, signal: threading.Event | None = None) -> None:
        if signal is not None:
            signal.set()
        try:
            results.append(backend._solver_for_tcp_frame("tool_tcp"))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            errors.append(exc)

    with mock.patch.object(
        rm65_module,
        "build_rm65_single_arm_ik_solver",
        side_effect=barrier_builder,
    ):
        first = threading.Thread(target=resolve_solver)
        second = threading.Thread(
            target=resolve_solver,
            kwargs={"signal": second_started},
        )
        first.start()
        assert builder_entered.wait(timeout=2.0)
        second.start()
        assert second_started.wait(timeout=2.0)
        assert second.is_alive()
        release_builder.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert builder_call_count == 1
    assert len(results) == 2
    assert results[0] is built_solver
    assert results[1] is built_solver
    with backend._solver_lock:
        assert backend._solvers["tool_tcp"] is built_solver


def test_inflight_validator_cannot_publish_for_replaced_solver_cache_owner(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    backend = _hardware_backend_for_attestation(urdf_path)
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    assert isinstance(
        backend.get_state(include_pose_source_attestation=True)[
            HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY
        ],
        RM65PoseSourceAttestation,
    )
    original_solver = backend._solvers["tool_tcp"]
    replacement_solver = _solver()
    replacement_solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    _bind_rm65_solver_urdf_provenance(
        solver=replacement_solver,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    builder_finished = threading.Event()
    release_builder = threading.Event()
    real_builder = rm65_module.build_rm65_pose_source_attestation
    validation_results: list[dict[str, object]] = []
    validation_errors: list[BaseException] = []

    def barrier_builder(**kwargs: object) -> RM65PoseSourceAttestation:
        result = real_builder(**kwargs)  # type: ignore[arg-type]
        builder_finished.set()
        if not release_builder.wait(timeout=5.0):
            raise RuntimeError("test timed out releasing validator")
        return result

    def validate_original_solver() -> None:
        try:
            validation_results.append(
                backend.get_state(
                    include_derived_pose=False,
                    include_pose_source_attestation=True,
                )
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            validation_errors.append(exc)

    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        side_effect=barrier_builder,
    ):
        validator = threading.Thread(target=validate_original_solver)
        validator.start()
        assert builder_finished.wait(timeout=2.0)
        with backend._solver_lock:
            assert backend._solvers["tool_tcp"] is original_solver
            backend._solvers["tool_tcp"] = replacement_solver
        release_builder.set()
        validator.join(timeout=2.0)

    assert not validator.is_alive()
    assert validation_errors == []
    assert len(validation_results) == 1
    invalidated = validation_results[0]
    assert invalidated[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert invalidated["hardware_pose_source_attestation_status"] == "unavailable"
    assert invalidated["hardware_pose_source_attestation_error"] == (
        "attestation_drift_detected:runtime_changed_during_validation"
    )
    with backend._solver_lock:
        assert backend._solvers["tool_tcp"] is replacement_solver
    final = backend.get_state(include_derived_pose=False)
    assert final[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert final["hardware_pose_source_attestation_status"] == "unavailable"
    assert final["hardware_pose_source_attestation_error"] == (
        "attestation_drift_detected:runtime_changed_during_validation"
    )
    backend.close()


def test_published_attestation_fails_closed_after_solver_replacement(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    backend = _hardware_backend_for_attestation(urdf_path)
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    initial = backend.get_state(include_pose_source_attestation=True)
    assert isinstance(
        initial[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )

    replacement = _solver()
    replacement.compute_pose = lambda _qpos: np.array(
        (
            (1.0, 0.0, 0.0, 0.25),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    _bind_rm65_solver_urdf_provenance(
        solver=replacement,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    backend._solvers["tool_tcp"] = replacement
    drifted = backend.get_state(include_pose_source_attestation=True)

    assert float(np.asarray(drifted["current_tcp_pose"])[0, 3]) == pytest.approx(0.25)
    assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert drifted["hardware_pose_source_attestation_error"] == (
        "attestation_drift_detected:solver_identity_or_compute_pose_changed"
    )
    backend.close()


def test_published_attestation_fails_closed_after_effective_tool_matrix_drift(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    initial = backend.get_state(include_pose_source_attestation=True)
    assert isinstance(
        initial[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )

    backend._hand_base_to_configured_tcp = backend._hand_base_to_configured_tcp.copy()
    backend._hand_base_to_configured_tcp[0, 3] += 0.01
    drifted = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )

    assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert str(drifted["hardware_pose_source_attestation_error"]).startswith(
        "attestation_drift_detected:"
    )
    backend.close()


def test_older_successful_validator_cannot_resurrect_concurrently_cleared_proof(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    assert isinstance(
        backend.get_state(include_pose_source_attestation=True)[
            HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY
        ],
        RM65PoseSourceAttestation,
    )
    solver = backend._solvers["tool_tcp"]
    first_builder_finished = threading.Event()
    release_first_builder = threading.Event()
    call_lock = threading.Lock()
    builder_call_count = 0
    real_builder = rm65_module.build_rm65_pose_source_attestation

    def barrier_builder(**kwargs: object) -> RM65PoseSourceAttestation:
        nonlocal builder_call_count
        with call_lock:
            builder_call_count += 1
            call_number = builder_call_count
        result = real_builder(**kwargs)  # type: ignore[arg-type]
        if call_number == 1:
            first_builder_finished.set()
            if not release_first_builder.wait(timeout=5.0):
                raise RuntimeError("test timed out releasing first validator")
        return result

    validation_errors: list[BaseException] = []

    def validate_old_snapshot() -> None:
        try:
            backend.get_state(
                include_derived_pose=False,
                include_pose_source_attestation=True,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            validation_errors.append(exc)

    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        side_effect=barrier_builder,
    ):
        first = threading.Thread(target=validate_old_snapshot)
        first.start()
        assert first_builder_finished.wait(timeout=2.0)
        solver.model.jointPlacements[2].homogeneous[0, 3] = 0.125
        drifted = backend.get_state(
            include_derived_pose=False,
            include_pose_source_attestation=True,
        )
        assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
        release_first_builder.set()
        first.join(timeout=2.0)

    assert not first.is_alive()
    assert validation_errors == []
    final = backend.get_state(include_derived_pose=False)
    assert final[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert final["hardware_pose_source_attestation_status"] == "unavailable"
    assert str(final["hardware_pose_source_attestation_error"]).startswith(
        "attestation_drift_detected:"
    )
    backend.close()


def test_inflight_validator_cannot_overwrite_close_terminal_state(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    assert isinstance(
        backend.get_state(include_pose_source_attestation=True)[
            HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY
        ],
        RM65PoseSourceAttestation,
    )
    builder_finished = threading.Event()
    release_builder = threading.Event()
    real_builder = rm65_module.build_rm65_pose_source_attestation

    def barrier_builder(**kwargs: object) -> RM65PoseSourceAttestation:
        result = real_builder(**kwargs)  # type: ignore[arg-type]
        builder_finished.set()
        if not release_builder.wait(timeout=5.0):
            raise RuntimeError("test timed out releasing validator")
        return result

    validation_errors: list[BaseException] = []

    def validate_old_snapshot() -> None:
        try:
            backend.get_state(
                include_derived_pose=False,
                include_pose_source_attestation=True,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            validation_errors.append(exc)

    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        side_effect=barrier_builder,
    ):
        validator = threading.Thread(target=validate_old_snapshot)
        validator.start()
        assert builder_finished.wait(timeout=2.0)
        backend.close()
        release_builder.set()
        validator.join(timeout=2.0)

    assert not validator.is_alive()
    assert validation_errors == []
    closed = backend.get_state(include_derived_pose=False)
    assert closed[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert closed["pose_source_epoch_id"] is None
    assert closed["hardware_pose_source_attestation_error"] == "closed"


def test_lean_state_omits_proof_without_running_full_revalidation(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    initial = backend.get_state(include_pose_source_attestation=True)
    assert isinstance(
        initial[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )

    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        wraps=rm65_module.build_rm65_pose_source_attestation,
    ) as builder:
        lean = backend.get_state(
            include_derived_pose=False,
            include_timing_percentiles=False,
        )

    assert builder.call_count == 0
    assert lean[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert lean["hardware_pose_source_attestation_status"] == "not_requested"
    backend.close()


def test_default_derived_state_does_not_run_or_publish_full_attestation(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)

    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        wraps=rm65_module.build_rm65_pose_source_attestation,
    ) as builder:
        default_state = backend.get_state()

    assert builder.call_count == 0
    assert default_state["current_tcp_pose"] is not None
    assert default_state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert default_state["hardware_pose_source_attestation_status"] == "unavailable"

    attested = backend.get_state(include_pose_source_attestation=True)
    assert isinstance(
        attested[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )
    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        wraps=rm65_module.build_rm65_pose_source_attestation,
    ) as builder:
        hidden_cached = backend.get_state()
    assert builder.call_count == 0
    assert hidden_cached[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert hidden_cached["hardware_pose_source_attestation_status"] == "not_requested"
    backend.close()


def test_noncallable_solver_after_attestation_does_not_break_state_api(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    assert isinstance(
        backend.get_state(include_pose_source_attestation=True)[
            HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY
        ],
        RM65PoseSourceAttestation,
    )

    backend._solvers["tool_tcp"].compute_pose = None
    state = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )

    assert state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert state["hardware_pose_source_attestation_status"] == "unavailable"
    assert str(state["hardware_pose_source_attestation_error"]).startswith(
        "attestation_drift_detected:TypeError:"
    )
    backend.close()


def test_attachment_endpoint_is_frozen_before_sdk_construction(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    backend = _hardware_backend_for_attestation(urdf_path)
    arm = _LifecycleFakeArm(_successful_state_result())
    actual_endpoint = str(backend.config.arm_ip)

    def sdk_factory(_model: object, endpoint: str) -> _LifecycleFakeArm:
        assert endpoint == actual_endpoint
        backend.config.arm_ip = "198.51.100.99"
        return arm

    with mock.patch.object(
        rm65_module,
        "_load_rm65_sdk",
        return_value=(sdk_factory, object()),
    ):
        backend.start()

    state = backend.get_state(include_pose_source_attestation=True)
    attestation = state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
    assert isinstance(attestation, RM65PoseSourceAttestation)
    assert backend.config.arm_ip == "198.51.100.99"
    assert attestation.configured_hardware_endpoint_hash == (
        "sha256:"
        + canonical_content_hash(
            {
                "transport": "rm65_direct",
                "side": "right",
                "arm_ip": actual_endpoint,
            },
            domain=RM65_CONFIGURED_HARDWARE_ENDPOINT_HASH_DOMAIN_V1,
        )
    )
    backend.close()


def test_runtime_identity_config_drift_withholds_proof_and_keeps_frozen_pose_frame(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    initial = backend.get_state(include_pose_source_attestation=True)
    initial_attestation = initial[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]
    assert isinstance(initial_attestation, RM65PoseSourceAttestation)
    initial_pose = np.asarray(initial["current_tcp_pose"], dtype=float)

    backend.config.side = "left"
    backend.config.effector_name = "renamed_arm"
    backend.config.default_tcp_frame = rm65_module.ARM_ENDPOINT_FRAME
    after_mutation = backend.get_state(include_pose_source_attestation=True)
    after_attestation = after_mutation[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY]

    assert after_attestation is None
    assert backend.name == "renamed_arm"
    assert after_mutation["hardware_pose_source_attestation_error"] == (
        "pose_source_identity_config_drift"
    )
    np.testing.assert_allclose(after_mutation["current_tcp_pose"], initial_pose)
    backend.close()


def test_identity_drift_before_first_proof_latches_epoch_fail_closed(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    backend.config.side = "left"
    _start_with_arm(backend, arm)

    drifted = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )
    assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert drifted["hardware_pose_source_attestation_error"] == (
        "pose_source_identity_config_drift"
    )

    backend.config.side = "right"
    restored = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )
    assert restored[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert restored["hardware_pose_source_attestation_status"] == "unavailable"
    assert restored["hardware_pose_source_attestation_error"] == (
        "pose_source_identity_config_drift"
    )
    backend.close()


def test_tool_resolution_cannot_clear_identity_drift_epoch_latch(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(
        _write_urdf(tmp_path / "model.urdf"),
        auto_load_tool_profile=True,
    )
    arm = _LifecycleFakeArm(_successful_state_result(), tool_name="empty")
    tool_read_entered = threading.Event()
    release_tool_read = threading.Event()
    original_tool_read = arm.Get_Current_Tool_Frame
    start_errors: list[BaseException] = []

    def barrier_tool_read() -> tuple[int, dict[str, str]]:
        tool_read_entered.set()
        if not release_tool_read.wait(timeout=5.0):
            raise RuntimeError("test timed out releasing active tool read")
        return original_tool_read()

    def run_start() -> None:
        try:
            _start_with_arm(backend, arm)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            start_errors.append(exc)

    with mock.patch.object(
        arm,
        "Get_Current_Tool_Frame",
        side_effect=barrier_tool_read,
    ):
        starter = threading.Thread(target=run_start)
        starter.start()
        try:
            assert tool_read_entered.wait(timeout=2.0)
            backend.config.side = "left"
            drifted = backend.get_state(
                include_derived_pose=False,
                include_pose_source_attestation=True,
            )
            assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
            assert drifted["hardware_pose_source_attestation_error"] == (
                "pose_source_identity_config_drift"
            )
            backend.config.side = "right"
        finally:
            release_tool_read.set()
            starter.join(timeout=2.0)

    assert not starter.is_alive()
    assert start_errors == []
    assert arm.tool_read_calls == 1
    final = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )
    assert final[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert final["hardware_pose_source_attestation_status"] == "unavailable"
    assert final["hardware_pose_source_attestation_error"] == (
        "pose_source_identity_config_drift"
    )
    backend.close()


def test_inflight_first_proof_cannot_publish_after_identity_drift_latch(
    tmp_path: Path,
) -> None:
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    arm = _LifecycleFakeArm(_successful_state_result())
    _start_with_arm(backend, arm)
    builder_finished = threading.Event()
    release_builder = threading.Event()
    real_builder = rm65_module.build_rm65_pose_source_attestation
    validation_errors: list[BaseException] = []

    def barrier_builder(**kwargs: object) -> RM65PoseSourceAttestation:
        result = real_builder(**kwargs)  # type: ignore[arg-type]
        builder_finished.set()
        if not release_builder.wait(timeout=5.0):
            raise RuntimeError("test timed out releasing first proof builder")
        return result

    def run_validation() -> None:
        try:
            backend.get_state(
                include_derived_pose=False,
                include_pose_source_attestation=True,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            validation_errors.append(exc)

    with mock.patch.object(
        rm65_module,
        "build_rm65_pose_source_attestation",
        side_effect=barrier_builder,
    ):
        validator = threading.Thread(target=run_validation)
        validator.start()
        try:
            assert builder_finished.wait(timeout=2.0)
            backend.config.side = "left"
            drifted = backend.get_state(
                include_derived_pose=False,
                include_pose_source_attestation=True,
            )
            assert drifted[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
            assert drifted["hardware_pose_source_attestation_error"] == (
                "pose_source_identity_config_drift"
            )
            backend.config.side = "right"
        finally:
            release_builder.set()
            validator.join(timeout=2.0)

    assert not validator.is_alive()
    assert validation_errors == []
    final = backend.get_state(
        include_derived_pose=False,
        include_pose_source_attestation=True,
    )
    assert final[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert final["hardware_pose_source_attestation_status"] == "unavailable"
    assert final["hardware_pose_source_attestation_error"] == (
        "pose_source_identity_config_drift"
    )
    backend.close()


def test_non_tool_default_frame_withholds_tool_tcp_attestation(
    tmp_path: Path,
) -> None:
    urdf_path = _write_urdf(tmp_path / "model.urdf")
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.20",
            side="right",
            arm_urdf_path=urdf_path,
            default_tcp_frame=rm65_module.ARM_ENDPOINT_FRAME,
            connect_hardware=True,
            observation_only=True,
            auto_load_tool_profile=False,
            hardware_read_retries=0,
            singularity_protection_enabled=False,
        ),
        pose_source_instance_id="non-tool-default",
    )
    solver = _solver()
    solver.compute_pose = lambda _qpos: np.eye(4, dtype=float)
    _bind_rm65_solver_urdf_provenance(
        solver=solver,
        urdf_path=urdf_path,
        expected_raw_bytes=urdf_path.read_bytes(),
    )
    backend._solvers[rm65_module.ARM_ENDPOINT_FRAME] = solver
    backend._solvers["tool_tcp"] = solver
    arm = _LifecycleFakeArm(_successful_state_result())

    _start_with_arm(backend, arm)
    state = backend.get_state(include_pose_source_attestation=True)

    np.testing.assert_allclose(state["current_tcp_pose"], np.eye(4, dtype=float))
    assert state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert state["hardware_pose_source_attestation_status"] == "unavailable"
    assert state["hardware_pose_source_attestation_error"] == (
        "default_tcp_frame_is_not_tool_tcp"
    )
    backend.close()


def test_optional_pose_proof_does_not_reject_legacy_unsupported_side() -> None:
    config = RM65ArmBackendConfig(
        arm_ip="192.0.2.20",
        side="center",
        connect_hardware=False,
        auto_load_tool_profile=False,
        singularity_protection_enabled=False,
    )

    backend = RM65ArmBackend(config)
    state = backend.get_state(include_derived_pose=False)

    assert backend.name == "center_arm"
    assert state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert state["hardware_pose_source_attestation_status"] == "unavailable"
    assert state["hardware_pose_source_attestation_error"] == (
        "hardware_connection_disabled"
    )


def test_epoch_initialization_failure_rolls_back_constructed_sdk_handle(
    tmp_path: Path,
) -> None:
    arm = _LifecycleFakeArm(_successful_state_result())
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))

    with (
        mock.patch.object(
            rm65_module,
            "_load_rm65_sdk",
            return_value=(lambda _model, _ip: arm, object()),
        ),
        mock.patch.object(
            rm65_module,
            "uuid4",
            side_effect=RuntimeError("epoch entropy unavailable"),
        ),
        pytest.raises(RuntimeError, match="epoch entropy unavailable"),
    ):
        backend.start()

    state = backend.get_state(include_pose_source_attestation=True)
    assert backend.arm is None
    assert arm.state_read_calls == 0
    assert arm.uninit_calls == 1
    assert arm.socket_close_calls == 1
    assert state["pose_source_epoch_id"] is None
    assert state["hardware_pose_source_attestation_error"] == "startup_failed"


def test_start_and_close_are_serialized_across_one_attachment(
    tmp_path: Path,
) -> None:
    state_entered = threading.Event()
    state_release = threading.Event()
    close_done = threading.Event()
    arm = _LifecycleFakeArm(
        _successful_state_result(),
        state_entered=state_entered,
        state_release=state_release,
    )
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    errors: list[BaseException] = []

    def run_start() -> None:
        try:
            backend.start()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            errors.append(exc)

    def run_close() -> None:
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            errors.append(exc)
        finally:
            close_done.set()

    with mock.patch.object(
        rm65_module,
        "_load_rm65_sdk",
        return_value=(lambda _model, _ip: arm, object()),
    ):
        start_thread = threading.Thread(target=run_start)
        close_thread = threading.Thread(target=run_close)
        start_thread.start()
        assert state_entered.wait(timeout=2.0)
        close_thread.start()
        assert not close_done.wait(timeout=0.05)
        state_release.set()
        start_thread.join(timeout=2.0)
        close_thread.join(timeout=2.0)

    assert not start_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert backend.arm is None
    state = backend.get_state(include_pose_source_attestation=True)
    assert state["pose_source_epoch_id"] is None
    assert state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY] is None
    assert state["hardware_pose_source_attestation_error"] == "closed"
    assert arm.state_read_calls == 1
    assert arm.uninit_calls == 1
    assert arm.socket_close_calls == 1


def test_concurrent_start_calls_share_one_attachment_and_epoch(
    tmp_path: Path,
) -> None:
    state_entered = threading.Event()
    state_release = threading.Event()
    second_started = threading.Event()
    arm = _LifecycleFakeArm(
        _successful_state_result(),
        state_entered=state_entered,
        state_release=state_release,
    )
    backend = _hardware_backend_for_attestation(_write_urdf(tmp_path / "model.urdf"))
    errors: list[BaseException] = []
    sdk_factory = mock.Mock(return_value=arm)

    def run_start(*, signal: threading.Event | None = None) -> None:
        if signal is not None:
            signal.set()
        try:
            backend.start()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below.
            errors.append(exc)

    with mock.patch.object(
        rm65_module,
        "_load_rm65_sdk",
        return_value=(sdk_factory, object()),
    ):
        first = threading.Thread(target=run_start)
        second = threading.Thread(
            target=run_start,
            kwargs={"signal": second_started},
        )
        first.start()
        assert state_entered.wait(timeout=2.0)
        second.start()
        assert second_started.wait(timeout=2.0)
        state_release.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert sdk_factory.call_count == 1
    assert arm.state_read_calls == 1
    state = backend.get_state(include_pose_source_attestation=True)
    assert state["pose_source_epoch_id"] is not None
    assert isinstance(
        state[HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY],
        RM65PoseSourceAttestation,
    )
    backend.close()
