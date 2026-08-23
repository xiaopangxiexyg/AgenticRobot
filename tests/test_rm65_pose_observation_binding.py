from __future__ import annotations

import argparse
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.frames import hand_base_to_arm_endpoint_transform
from robot_motion.pose import pose6_to_matrix
from robot_motion.runtime.identity import ClockDomainDescriptor
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend
from teleop_runtime.backends.arms.rm65_pose_source import (
    HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY,
    RM65ForwardKinematicsContract,
    RM65PoseSourceAttestation,
    RM65ToolTcpContract,
)
from teleop_runtime.deploy.arm_state_reader import (
    HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY,
    ArmStateReader,
)
from teleop_runtime.deploy.rm65_pose_observation import (
    RM65_TCP_POSE_SCHEMA_HASH_V1,
    assemble_rm65_pose_observation,
    build_policy_deploy_receiver_clock,
)
from teleop_runtime.deploy.run_policy import (
    _build_arm_state_reader,
    add_args,
    main,
    validate_args,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_BOOT_ID = "12345678-1234-5678-9234-567812345678"


def _receiver_clock() -> ClockDomainDescriptor:
    return ClockDomainDescriptor(
        clock_domain_id=f"linux-monotonic:{_BOOT_ID}",
        runtime_session_id="policy-deploy:test-session",
        boot_id=_BOOT_ID,
        reset_epoch=0,
        kind="python.monotonic",
        resolution_sec=1e-9,
        monotonic=True,
    )


def _attestation(side: str) -> RM65PoseSourceAttestation:
    digit = "1" if side == "left" else "2"
    urdf_hash = "sha256:" + digit * 64
    tcp_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    applied_tcp = hand_base_to_arm_endpoint_transform() @ pose6_to_matrix(tcp_pose)
    tool = RM65ToolTcpContract(
        side=side,
        tool_name=f"{side}_tool",
        tcp_evidence="test_active_tool_profile",
        tcp_pose_flange_m_rad=tcp_pose,
        configured_tcp_enabled=True,
        applied_solver_eef_to_tool_tcp_matrix=tuple(applied_tcp.reshape(-1)),
    )
    fk = RM65ForwardKinematicsContract(
        urdf_raw_bytes_hash=urdf_hash,
        urdf_root_link="base_link",
        non_fixed_joint_order=tuple(f"joint_{index}" for index in range(1, 7)),
        solver_eef_name="hand_base",
        pinocchio_version="3.2.0",
        solver_model_kinematics_hash="sha256:" + digit * 64,
        tool_profile_hash=tool.tool_profile_hash,
        output_convention=f"T_{side}_arm_base_tool_tcp",
    )
    reference_frame = CoordinateFrameSpec(
        frame_id=f"{side}_arm_base",
        frame_version="1",
        semantic_parameters={
            "owner": "rm65_position_backend",
            "coordinate_source": "fixed_base_pinocchio_model_world",
            "urdf_root_link": "base_link",
            "urdf_raw_bytes_hash": urdf_hash,
            "matrix_convention": "T_reference_frame_from_pinocchio_world",
            "reference_frame_from_pinocchio_world_matrix": tuple(
                float(item) for item in np.eye(4).reshape(-1)
            ),
            "axis_contract": "urdf_root_xyz_right_handed",
            "translation_unit": "m",
            "rotation_unit": "rad",
        },
    )
    return RM65PoseSourceAttestation(
        side=side,
        effector_name=f"{side}_arm",
        source_instance_id=f"rm65-{side}-runtime",
        source_epoch_id=f"rm65-{side}-epoch-1",
        configured_hardware_endpoint_hash="sha256:" + digit * 64,
        reference_frame_spec=reference_frame,
        tool_tcp_contract=tool,
        forward_kinematics_contract=fk,
        measurement_clock_domain_id="local_wall_time",
    )


class _FakeRM65Backend(RM65ArmBackend):
    def __init__(self, side: str) -> None:
        self.side = side
        self.attestation: RM65PoseSourceAttestation | None = _attestation(side)
        self.ready = True
        self.get_state_calls = 0
        self.get_state_kwargs: list[dict[str, object]] = []
        self.refresh_calls = 0
        self.measurement_sequence = 4
        self.measurement_timestamp = 99.0
        self.spoofed_coordinate_binding: object | None = None

    @property
    def name(self) -> str:
        return f"{self.side}_arm"

    def is_ready(self) -> bool:
        return self.ready

    def refresh_state_from_hardware(self) -> None:
        self.refresh_calls += 1

    def get_state(self, **kwargs: object) -> dict[str, object]:
        self.get_state_calls += 1
        self.get_state_kwargs.append(dict(kwargs))
        attestation = self.attestation
        state: dict[str, object] = {
            "hardware_qpos": np.zeros(6, dtype=float),
            "hardware_measurement_timestamp": self.measurement_timestamp,
            "hardware_measurement_sequence": self.measurement_sequence,
            "hardware_measurement_valid": True,
            "hardware_read_success": True,
            "hardware_measurement_clock_domain_id": "local_wall_time",
            "hardware_availability_clock_domain_id": "local_wall_time",
            "pose_source_instance_id": (
                None if attestation is None else attestation.source_instance_id
            ),
            "pose_source_epoch_id": (
                None if attestation is None else attestation.source_epoch_id
            ),
            HARDWARE_POSE_SOURCE_ATTESTATION_STATE_KEY: attestation,
            "hardware_pose_source_binding_hash": (
                None if attestation is None else attestation.source_binding_hash
            ),
            "hardware_pose_source_attestation_status": (
                "unavailable" if attestation is None else "attested"
            ),
        }
        if self.spoofed_coordinate_binding is not None:
            state[HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY] = (
                self.spoofed_coordinate_binding
            )
        return state

    def compute_tcp_pose(self, *, qpos=None) -> np.ndarray:
        return np.eye(4, dtype=float)


def test_receiver_clock_binds_real_boot_id_and_runtime_session(tmp_path: Path) -> None:
    boot_id_path = tmp_path / "boot_id"
    boot_id_path.write_text(_BOOT_ID + "\n", encoding="utf-8")

    descriptor = build_policy_deploy_receiver_clock(
        runtime_session_id="policy-deploy:test-session",
        boot_id_path=boot_id_path,
    )

    assert descriptor.boot_id == _BOOT_ID
    assert descriptor.runtime_session_id == "policy-deploy:test-session"
    assert descriptor.clock_domain_id == f"linux-monotonic:{_BOOT_ID}"
    assert descriptor.kind == "python.monotonic"
    assert descriptor.monotonic is True


def test_app_root_assembles_exact_binding_and_reader_uses_one_state_read() -> None:
    backend = _FakeRM65Backend("right")
    receipt_clock = _receiver_clock()
    receipt_times = iter((5_000_000_000, 5_100_000_000))
    assembly = assemble_rm65_pose_observation(
        {"right": backend},
        receipt_clock=receipt_clock,
        control_domain_id="deploy-control-0",
        bound_at=90.0,
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    projection = assembly.projection_by_side["right"]
    coordinate = projection.coordinate_binding
    attestation = backend.attestation
    assert attestation is not None

    assert coordinate.source_instance_id == attestation.source_instance_id
    assert coordinate.source_epoch_id == attestation.source_epoch_id
    assert coordinate.source_binding_hash == attestation.source_binding_hash
    assert coordinate.reference_frame_spec_hash == (
        attestation.reference_frame_spec_hash
    )
    assert coordinate.tool_profile_hash == attestation.tool_profile_hash
    assert coordinate.fk_contract_hash == attestation.fk_contract_hash
    assert coordinate.pose_schema_hash == RM65_TCP_POSE_SCHEMA_HASH_V1
    assert coordinate.control_domain_id == "deploy-control-0"
    assert coordinate.component_id == "right_arm"
    assert projection.observation_binding.component_ids == ("right_arm",)
    assert backend.get_state_calls == 1
    assert backend.get_state_kwargs == [
        {
            "include_derived_pose": False,
            "include_timing_percentiles": False,
            "include_pose_source_attestation": True,
        }
    ]

    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=30.0,
        clock=lambda: 100.0,
        pose_projection_by_side=assembly.projection_by_side,
        pose_state_read_by_side=assembly.state_read_by_side,
        receipt_clock=assembly.receipt_clock,
        receipt_monotonic_ns_clock=assembly.receipt_monotonic_ns_clock,
    )
    reader.snapshot(100.0)

    assert backend.refresh_calls == 1
    assert backend.get_state_calls == 2
    assert reader.latest_pose_measurement(side="right") is not None


def test_app_root_requires_typed_attestation_before_reader_construction() -> None:
    backend = _FakeRM65Backend("right")
    backend.attestation = None

    with pytest.raises(TypeError, match="typed pose-source attestation"):
        assemble_rm65_pose_observation(
            {"right": backend},
            receipt_clock=_receiver_clock(),
            control_domain_id="deploy-control-0",
            bound_at=90.0,
        )


def test_runtime_attestation_drift_invalidates_only_passive_pose() -> None:
    backend = _FakeRM65Backend("right")
    receipt_times = iter((5_000_000_000, 5_100_000_000, 5_200_000_000, 5_300_000_000))
    assembly = assemble_rm65_pose_observation(
        {"right": backend},
        receipt_clock=_receiver_clock(),
        control_domain_id="deploy-control-0",
        bound_at=90.0,
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader = ArmStateReader(
        {"right": backend},
        refresh_hz=0.0,
        clock=lambda: 100.0,
        pose_projection_by_side=assembly.projection_by_side,
        pose_state_read_by_side=assembly.state_read_by_side,
        receipt_clock=assembly.receipt_clock,
        receipt_monotonic_ns_clock=assembly.receipt_monotonic_ns_clock,
    )
    reader.snapshot(100.0)
    projection = assembly.projection_by_side["right"]
    original = backend.attestation
    assert original is not None
    backend.attestation = replace(original, source_epoch_id="rm65-right-epoch-2")
    backend.spoofed_coordinate_binding = projection.coordinate_binding
    backend.measurement_sequence = 5
    backend.measurement_timestamp = 99.1

    state = reader.snapshot(101.0)["right"]

    assert state["causal_snapshot_missing"] is False
    assert HARDWARE_POSE_COORDINATE_BINDING_STATE_KEY not in state
    assert reader.latest_pose_measurement(side="right") is None
    assert (
        "changed after binding"
        in (reader.pose_observation_status()["error_by_side"]["right"])
    )


def test_dual_arm_assembly_uses_side_local_observation_bindings() -> None:
    left = _FakeRM65Backend("left")
    right = _FakeRM65Backend("right")

    assembly = assemble_rm65_pose_observation(
        {"left": left, "right": right},
        receipt_clock=_receiver_clock(),
        control_domain_id="deploy-control-0",
        bound_at=90.0,
    )

    assert set(assembly.projection_by_side) == {"left", "right"}
    assert assembly.projection_by_side["left"].observation_binding.component_ids == (
        "left_arm",
    )
    assert assembly.projection_by_side["right"].observation_binding.component_ids == (
        "right_arm",
    )
    assert (
        assembly.projection_by_side["left"].observation_binding
        != assembly.projection_by_side["right"].observation_binding
    )


def test_dual_arm_runtime_drift_invalidates_only_the_affected_side() -> None:
    left = _FakeRM65Backend("left")
    right = _FakeRM65Backend("right")
    receipt_times = iter((5_000_000_000, 5_100_000_000, 5_200_000_000, 5_300_000_000))
    assembly = assemble_rm65_pose_observation(
        {"left": left, "right": right},
        receipt_clock=_receiver_clock(),
        control_domain_id="deploy-control-0",
        bound_at=90.0,
        receipt_monotonic_ns_clock=lambda: next(receipt_times),
    )
    reader = ArmStateReader(
        {"left": left, "right": right},
        refresh_hz=30.0,
        clock=lambda: 100.0,
        pose_projection_by_side=assembly.projection_by_side,
        pose_state_read_by_side=assembly.state_read_by_side,
        receipt_clock=assembly.receipt_clock,
        receipt_monotonic_ns_clock=assembly.receipt_monotonic_ns_clock,
    )
    reader.snapshot(100.0)
    left_attestation = left.attestation
    assert left_attestation is not None
    left.attestation = replace(left_attestation, source_epoch_id="rm65-left-epoch-2")
    for backend in (left, right):
        backend.measurement_sequence = 5
        backend.measurement_timestamp = 99.1

    states = reader.snapshot(101.0)

    assert set(states) == {"left", "right"}
    assert reader.latest_pose_measurement(side="left") is None
    right_latest = reader.latest_pose_measurement(side="right")
    assert right_latest is not None
    assert right_latest.source_sequence == 5
    assert left.refresh_calls == right.refresh_calls == 2
    assert left.get_state_calls == right.get_state_calls == 3


def _parse_and_validate(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(argv)
    validate_args(args, raw_argv=argv)
    return args


def test_rm65_pose_observation_cli_is_default_off_and_accepts_bounded_profile() -> None:
    assert _parse_and_validate([]).rm65_pose_observation is False
    args = _parse_and_validate(
        [
            "--rm65-pose-observation",
            "--active-sides",
            "right",
            "--right-arm-ip",
            "192.0.2.18",
        ]
    )
    assert args.rm65_pose_observation is True
    assert args.arm_control_mode == "position"
    assert args.send_actions is False


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    (
        (["--no-arms"], "requires enabled arms"),
        (["--arm-control-mode", "impedance"], "position mode only"),
        (["--send-actions"], "forbidden with --send-actions"),
        (["--allow-arm-connect-failure"], "requires every selected RM65 backend"),
        (["--arm-state-refresh-hz", "0"], "arm-state-refresh-hz > 0"),
        (["--arm-state-refresh-hz", "nan"], "finite --arm-state-refresh-hz > 0"),
        (["--arm-state-refresh-hz", "inf"], "finite --arm-state-refresh-hz > 0"),
        (["--arm-state-refresh-hz=-inf"], "finite --arm-state-refresh-hz > 0"),
    ),
)
def test_rm65_pose_observation_cli_rejects_unsupported_profiles(
    extra_args: list[str],
    expected: str,
) -> None:
    argv = [
        "--rm65-pose-observation",
        "--active-sides",
        "right",
        "--right-arm-ip",
        "192.0.2.18",
        *extra_args,
    ]

    with pytest.raises(ValueError, match=expected):
        _parse_and_validate(argv)


def test_rm65_pose_observation_cli_requires_each_active_side_ip() -> None:
    with pytest.raises(ValueError, match="--left-arm-ip"):
        _parse_and_validate(
            [
                "--rm65-pose-observation",
                "--right-arm-ip",
                "192.0.2.18",
            ]
        )


def test_application_root_orders_binding_after_start_and_before_snapshot() -> None:
    source = inspect.getsource(main)
    assert source.index("resource_lifecycle.start(") < source.index(
        "assemble_rm65_pose_observation("
    )
    assert source.index("assemble_rm65_pose_observation(") < source.index(
        "arm_reader.snapshot("
    )
    assert source.index("invalidate_pose_measurements(") < source.index(
        "resource_lifecycle.close("
    )
    assert inspect.getsource(_build_arm_state_reader).count("ArmStateReader(") == 1


def test_default_reader_builder_does_not_enable_pose_projection() -> None:
    backend = _FakeRM65Backend("right")
    args = _parse_and_validate([])

    reader = _build_arm_state_reader(
        args=args,
        arm_backends={"right": backend},
        pose_observation=None,
    )

    assert reader.pose_measurement_journal_cut() is None
    assert backend.get_state_calls == 0
