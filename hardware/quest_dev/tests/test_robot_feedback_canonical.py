from __future__ import annotations

import unittest

import numpy as np

from hardware.quest_dev.features.visualization.frame_binding import (
    AcceptedFrameBinding,
    QuestFrameIdentity,
    RobotFrameIdentity,
)
from hardware.quest_dev.features.visualization.pose_projection import (
    project_pose_publication,
)
from robot_motion.action.execution_observation import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import ObservationSampleRef
from robot_motion.runtime.pose_observation import (
    POSE_COORDINATE_PARENT_STAGE_V1,
    PoseCoordinateBinding,
    PoseMeasurementPublication,
)


_CLOCK_ID = "host-boot-1.monotonic"
_TOOL_HASH = "sha256:" + "1" * 64


def make_coordinate() -> PoseCoordinateBinding:
    return PoseCoordinateBinding(
        binding_id="right-tool-pose:rm65-right-epoch-7",
        source_instance_id="rm65-right-runtime",
        source_epoch_id="rm65-right-epoch-7",
        source_binding_hash="rm65-source-binding-hash",
        control_domain_id="real-robot-right",
        component_id="right_arm",
        stream_id="robot-state.right-arm.tool-tcp-pose",
        reference_frame_id="right-arm-base",
        reference_frame_spec_hash="right-base-frame-spec",
        moving_frame_id="tool-tcp",
        tool_profile_hash=_TOOL_HASH,
        pose_schema_hash="pose-schema-xyzquat-v1",
        fk_contract_id="rm65-hardware-qpos-fk-v1",
        fk_contract_hash="rm65-hardware-qpos-fk-hash",
        measurement_clock_domain_id="local-wall-time",
        receipt_clock_domain_id=_CLOCK_ID,
        observation_binding_id="observation-binding",
        observation_binding_hash="observation-binding-hash",
        observation_plan_hash="observation-plan-hash",
        source_kind="real_robot_hardware_qpos_fk",
    )


def make_publication(coordinate: PoseCoordinateBinding) -> PoseMeasurementPublication:
    source_sequence = 7
    measured_at = 90.07
    available_at = 100.07
    received_ns = 1_000_000_000
    measurement = ExecutionMeasurementObservation(
        measurement_id="robot-measurement-7",
        source_measurement_hash="robot-source-measurement-7",
        control_domain_id=coordinate.control_domain_id,
        measured_at=measured_at,
        available_at=available_at,
        clock_domain_id=coordinate.measurement_clock_domain_id,
        value_by_component={
            coordinate.component_id: (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
        representation_by_component={coordinate.component_id: "cartesian_pose"},
        schema_hash_by_component={
            coordinate.component_id: coordinate.pose_schema_hash,
        },
        observation_binding_id=coordinate.observation_binding_id,
        observation_plan_hash=coordinate.observation_plan_hash,
        parent_facts=(
            ExecutionFactReference(
                stage=POSE_COORDINATE_PARENT_STAGE_V1,
                fact_id=coordinate.binding_id,
                fact_hash=coordinate.semantic_content_hash,
                observed_at=available_at,
                clock_domain_id=coordinate.measurement_clock_domain_id,
                status="applied",
            ),
        ),
    )
    sample_ref = ObservationSampleRef(
        stream_id=coordinate.stream_id,
        sample_id=measurement.measurement_id,
        sequence=source_sequence,
        measured_at=measured_at,
        measured_clock_domain_id=coordinate.measurement_clock_domain_id,
        available_at=received_ns / 1_000_000_000.0,
        available_clock_domain_id=coordinate.receipt_clock_domain_id,
        payload_object_id=measurement.semantic_content_hash,
        content_metadata={
            "coordinate_binding_hash": coordinate.semantic_content_hash,
            "source_instance_id": coordinate.source_instance_id,
            "source_epoch_id": coordinate.source_epoch_id,
            "source_kind": coordinate.source_kind,
            "source_sequence": source_sequence,
            "reference_frame_id": coordinate.reference_frame_id,
            "reference_frame_spec_hash": coordinate.reference_frame_spec_hash,
            "moving_frame_id": coordinate.moving_frame_id,
            "tool_profile_hash": coordinate.tool_profile_hash,
            "measurement_available_at": measurement.available_at,
            "measurement_available_clock_domain_id": measurement.clock_domain_id,
            "received_monotonic_ns": received_ns,
        },
    )
    receiver_clock = ClockDomainDescriptor(
        clock_domain_id=_CLOCK_ID,
        runtime_session_id="runtime-1",
        boot_id="host-boot-1",
        reset_epoch=0,
        resolution_sec=1e-9,
    )
    return PoseMeasurementPublication(
        coordinate_binding=coordinate,
        source_sequence=source_sequence,
        measurement=measurement,
        sample_ref=sample_ref,
        received_monotonic_ns=received_ns,
        receiver_clock=receiver_clock,
    )


def make_frame_binding(coordinate: PoseCoordinateBinding) -> AcceptedFrameBinding:
    return AcceptedFrameBinding(
        binding_id="binding-1",
        acceptance_receipt_sha256="a" * 64,
        acceptance_profile_id="profile-1",
        capture_evidence_semantic_sha256="b" * 64,
        diagnostic_fit_semantic_sha256="c" * 64,
        dataset_sha256="d" * 64,
        fit_semantic_sha256="e" * 64,
        quest=QuestFrameIdentity(
            session_id="xr-session-1",
            reference_space_type="local-floor",
            reference_space_generation=0,
            reference_epoch_id="xr-session-1:local-floor:0",
            controller_source_session_id="controller-source-1",
            reference_frame_id="quest-local-floor",
        ),
        robot=RobotFrameIdentity(
            robot_id="ar5_1",
            source_instance_id=coordinate.source_instance_id,
            source_epoch_id=coordinate.source_epoch_id,
            source_binding_hash=coordinate.source_binding_hash,
            component_id=coordinate.component_id,
            stream_id=coordinate.stream_id,
            reference_frame_id=coordinate.reference_frame_id,
            reference_frame_spec_hash=coordinate.reference_frame_spec_hash,
            moving_frame_id=coordinate.moving_frame_id,
            tool_profile_hash=coordinate.tool_profile_hash,
            pose_schema_hash=coordinate.pose_schema_hash,
            fk_contract_id=coordinate.fk_contract_id,
            fk_contract_hash=coordinate.fk_contract_hash,
            coordinate_binding_hash=coordinate.semantic_content_hash,
            source_kind=coordinate.source_kind,
        ),
        matrix_quest_from_robot_reference=np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


class CanonicalPoseProjectionTests(unittest.TestCase):
    def test_passive_publication_projects_into_quest_space(self) -> None:
        coordinate = make_coordinate()
        record = project_pose_publication(
            make_publication(coordinate),
            expected_robot="ar5_1",
            key="journal:right",
            max_age_s=0.25,
            now_monotonic_ns=1_100_000_000,
            frame_binding=make_frame_binding(coordinate),
        )
        self.assertTrue(record["fresh"])
        self.assertTrue(record["alignment_ready"])
        self.assertEqual(record["source"], "passive_pose_journal")
        self.assertEqual(record["matrix_quest_from_tcp"][12:15], [1.2, 0.0, 0.0])

    def test_passive_publication_epoch_mismatch_is_not_projected(self) -> None:
        coordinate = make_coordinate()
        binding = make_frame_binding(coordinate)
        robot = binding.robot
        mismatched = RobotFrameIdentity(
            **{
                **robot.as_dict(),
                "source_epoch_id": "different-epoch",
            }
        )
        mismatched_binding = AcceptedFrameBinding(
            binding_id=binding.binding_id,
            acceptance_receipt_sha256=binding.acceptance_receipt_sha256,
            acceptance_profile_id=binding.acceptance_profile_id,
            capture_evidence_semantic_sha256=binding.capture_evidence_semantic_sha256,
            diagnostic_fit_semantic_sha256=binding.diagnostic_fit_semantic_sha256,
            dataset_sha256=binding.dataset_sha256,
            fit_semantic_sha256=binding.fit_semantic_sha256,
            quest=binding.quest,
            robot=mismatched,
            matrix_quest_from_robot_reference=binding.matrix_quest_from_robot_reference,
        )
        record = project_pose_publication(
            make_publication(coordinate),
            expected_robot="ar5_1",
            key="journal:right",
            max_age_s=0.25,
            now_monotonic_ns=1_100_000_000,
            frame_binding=mismatched_binding,
        )
        self.assertFalse(record["fresh"])
        self.assertEqual(record["error"], "source_epoch_id_mismatch")


if __name__ == "__main__":
    unittest.main()
