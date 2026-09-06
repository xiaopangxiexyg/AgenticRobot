from __future__ import annotations

import unittest

import numpy as np

from hardware.quest_dev.features.visualization.runtime import VisualizationRuntime
from hardware.quest_dev.features.visualization.frame_binding import (
    AcceptedFrameBinding,
    QuestFrameIdentity,
    RobotFrameIdentity,
    parse_accepted_frame_binding,
    project_quest_from_tcp,
    project_tcp_from_quest,
)


def make_binding() -> AcceptedFrameBinding:
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
            source_instance_id="rm65-right-runtime",
            source_epoch_id="rm65-right-epoch-7",
            source_binding_hash="rm65-source-binding-hash",
            component_id="right_arm",
            stream_id="robot-state.right-arm.tool-tcp-pose",
            reference_frame_id="right-arm-base",
            reference_frame_spec_hash="right-base-frame-spec",
            moving_frame_id="tool-tcp",
            tool_profile_hash="sha256:" + "1" * 64,
            pose_schema_hash="pose-schema-xyzquat-v1",
            fk_contract_id="rm65-hardware-qpos-fk-v1",
            fk_contract_hash="rm65-hardware-qpos-fk-hash",
            coordinate_binding_hash="f" * 64,
            source_kind="real_robot_hardware_qpos_fk",
        ),
        matrix_quest_from_robot_reference=np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


class FrameBindingTests(unittest.TestCase):
    def test_canonical_round_trip_preserves_reviewed_identity(self) -> None:
        binding = make_binding()
        parsed = parse_accepted_frame_binding(binding.as_dict())
        self.assertEqual(parsed.as_dict(), binding.as_dict())
        self.assertEqual(
            parsed.telemetry_identity()["quest_reference_epoch_id"],
            "xr-session-1:local-floor:0",
        )

    def test_projector_composes_quest_from_robot_with_robot_from_tcp(self) -> None:
        binding = make_binding()
        robot_from_tcp = np.eye(4)
        robot_from_tcp[:3, 3] = [0.1, -0.2, 0.3]
        expected = np.eye(4)
        expected[:3, 3] = [1.1, 1.8, 3.3]
        projected = project_quest_from_tcp(robot_from_tcp, binding)
        np.testing.assert_allclose(projected, expected, atol=1e-9)
        np.testing.assert_allclose(
            project_tcp_from_quest(projected, binding),
            robot_from_tcp,
            atol=1e-9,
        )

    def test_unaccepted_payload_is_rejected(self) -> None:
        payload = make_binding().as_dict()
        payload["accepted"] = False
        with self.assertRaisesRegex(ValueError, "accepted=true"):
            parse_accepted_frame_binding(payload)

    def test_tampered_column_major_matrix_is_rejected(self) -> None:
        payload = make_binding().as_dict()
        payload["matrix_quest_from_robot_reference_column_major"][12] += 0.1
        with self.assertRaisesRegex(ValueError, "column_major"):
            parse_accepted_frame_binding(payload)

    def test_quest_epoch_identity_is_derived_not_free_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_epoch_id"):
            QuestFrameIdentity(
                session_id="xr-session-1",
                reference_space_type="local-floor",
                reference_space_generation=1,
                reference_epoch_id="xr-session-1:local-floor:0",
                controller_source_session_id="controller-source-1",
                reference_frame_id="quest-local-floor",
            )

    def test_runtime_loads_only_binding_for_matching_robot_key(self) -> None:
        binding = make_binding()
        loaded = VisualizationRuntime._load_frame_bindings(
            {
                "frame_bindings": {"ar5_1": binding.as_dict()},
                "frame_binding_paths": {},
            }
        )
        self.assertEqual(loaded["ar5_1"].robot, binding.robot)
        with self.assertRaisesRegex(ValueError, "differs from key"):
            VisualizationRuntime._load_frame_bindings(
                {
                    "frame_bindings": {"ar5_2": binding.as_dict()},
                    "frame_binding_paths": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
