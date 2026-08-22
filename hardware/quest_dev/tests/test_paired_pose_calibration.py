from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from hardware.quest_dev.features.visualization.alignment import fit_quest_from_robot
from hardware.quest_dev.features.visualization.paired_pose_calibration import (
    DATASET_SCHEMA,
    controller_wire_matrix_to_transform,
    fit_paired_pose_calibration,
    parse_paired_pose_dataset,
)


def rotation(axis: list[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1 - c
    return np.asarray(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ]
    )


def transform(axis: list[float], angle: float, translation: list[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation(axis, angle)
    matrix[:3, 3] = translation
    return matrix


def synthetic_payload() -> tuple[dict, np.ndarray, np.ndarray]:
    second_reference_from_first_reference = transform(
        [0.2, -0.3, 1.0], 0.45, [0.55, -0.22, 0.82]
    )
    first_moving_from_second_moving = transform(
        [1.0, 0.1, -0.2], -0.38, [0.075, 0.028, -0.045]
    )
    first_poses = [
        transform([1, 0, 0], 0.20, [0.10, 0.00, 0.20]),
        transform([0, 1, 0], -0.42, [0.21, 0.10, 0.31]),
        transform([0, 0, 1], 0.63, [0.00, 0.22, 0.41]),
        transform([1, 1, 0], 0.51, [-0.12, 0.12, 0.26]),
        transform([0, 1, 1], -0.57, [0.18, -0.08, 0.36]),
        transform([1, 0, 1], 0.72, [-0.04, -0.16, 0.29]),
    ]
    second_poses = [
        second_reference_from_first_reference
        @ first_pose
        @ first_moving_from_second_moving
        for first_pose in first_poses
    ]
    samples = []
    for index, (first_pose, second_pose) in enumerate(
        zip(first_poses, second_poses, strict=True)
    ):
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "first_observation": {
                    "observation_id": f"robot-{index}",
                    "source_instance_id": "robot-readonly-source",
                    "source_epoch_id": "robot-epoch-1",
                    "sequence": index,
                    "observed_at": 10.0 + index,
                    "clock_domain_id": "robot-monotonic",
                    "matrix_reference_from_moving": first_pose.tolist(),
                },
                "second_observation": {
                    "observation_id": f"quest-{index}",
                    "source_instance_id": "input-1",
                    "source_epoch_id": "quest-session-1:local-floor:0",
                    "sequence": index,
                    "observed_at": 1000.0 + index,
                    "clock_domain_id": "webxr-frame-time",
                    "matrix_reference_from_moving": second_pose.tolist(),
                },
            }
        )
    payload = {
        "schema": DATASET_SCHEMA,
        "frames": {
            "first_reference": "robot_base",
            "first_moving": "tool_tcp",
            "second_reference": "quest_reference",
            "second_moving": "controller_grip",
        },
        "pairing_method": "synthetic_replay",
        "source_identity": {
            "quest_session_id": "quest-session-1",
            "reference_space_type": "local-floor",
            "reference_space_generation": 0,
            "controller_source_session_id": "input-1",
            "robot_source_instance_id": "robot-readonly-source",
            "quest_reference_epoch_id": "quest-session-1:local-floor:0",
            "robot_source_epoch_id": "robot-epoch-1",
            "tool_profile_hash": f"sha256:{'1' * 64}",
        },
        "samples": samples,
    }
    return (
        payload,
        second_reference_from_first_reference,
        first_moving_from_second_moving,
    )


class PairedPoseCalibrationTests(unittest.TestCase):
    def test_recovers_reference_and_mount_extrinsics(self) -> None:
        payload, expected_reference, expected_mount = synthetic_payload()
        dataset = parse_paired_pose_dataset(payload)
        result = fit_paired_pose_calibration(dataset)
        np.testing.assert_allclose(
            result.second_reference_from_first_reference, expected_reference, atol=1e-9
        )
        np.testing.assert_allclose(
            result.first_moving_from_second_moving, expected_mount, atol=1e-9
        )
        self.assertLess(result.translation_max_m, 1e-9)
        self.assertLess(result.rotation_max_deg, 1e-6)
        output = result.as_dict()
        self.assertIsNone(output["accepted"])
        self.assertEqual(output["frames"], payload["frames"])
        self.assertEqual(
            len(output["first_moving_from_second_moving_column_major"]), 16
        )
        self.assertEqual(len(output["fit_semantic_sha256"]), 64)

    def test_dataset_round_trip_preserves_semantic_hash(self) -> None:
        payload, _, _ = synthetic_payload()
        first = parse_paired_pose_dataset(payload)
        second = parse_paired_pose_dataset(first.as_dict())
        self.assertEqual(first.semantic_sha256, second.semantic_sha256)

    def test_dataset_is_frozen_and_fit_rechecks_semantic_hash(self) -> None:
        payload, _, _ = synthetic_payload()
        dataset = parse_paired_pose_dataset(payload)
        with self.assertRaises(TypeError):
            dataset.frames["first_reference"] = "mutated"
        object.__setattr__(dataset, "semantic_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "semantic hash"):
            fit_paired_pose_calibration(dataset)

    def test_unknown_dataset_field_is_rejected(self) -> None:
        payload, _, _ = synthetic_payload()
        payload["unhashed_annotation"] = "must not be silently dropped"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            parse_paired_pose_dataset(payload)

    def test_controller_wire_column_major_adapter_has_one_direction(self) -> None:
        expected = transform([0.3, -0.4, 0.8], 0.71, [0.12, -0.34, 0.56])
        wire = expected.reshape(16, order="F").tolist()
        actual = controller_wire_matrix_to_transform(wire)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "last row"):
            controller_wire_matrix_to_transform(expected.reshape(16).tolist())

    def test_position_only_kabsch_fails_with_unknown_mount_offset(self) -> None:
        payload, _, _ = synthetic_payload()
        dataset = parse_paired_pose_dataset(payload)
        robot_points = [
            sample.first.matrix_reference_from_moving[:3, 3]
            for sample in dataset.samples
        ]
        quest_points = [
            sample.second.matrix_reference_from_moving[:3, 3]
            for sample in dataset.samples
        ]
        point_fit = fit_quest_from_robot(robot_points, quest_points)
        self.assertGreater(point_fit.max_error_m, 0.005)

    def test_single_axis_rotation_is_rejected_as_unobservable(self) -> None:
        payload, _, _ = synthetic_payload()
        for index, sample in enumerate(payload["samples"]):
            pose = transform([0, 0, 1], 0.2 * index, [0.1 * index, 0, 0])
            sample["first_observation"]["matrix_reference_from_moving"] = pose.tolist()
            sample["second_observation"]["matrix_reference_from_moving"] = pose.tolist()
        with self.assertRaisesRegex(ValueError, "separated"):
            fit_paired_pose_calibration(parse_paired_pose_dataset(payload))

    def test_tiny_rotation_excitation_is_rejected(self) -> None:
        payload, expected_reference, expected_mount = synthetic_payload()
        axes = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1])
        for index, (sample, axis) in enumerate(
            zip(payload["samples"], axes, strict=True)
        ):
            first_pose = transform(axis, 1e-4 * (index + 1), [0.03 * index, 0, 0])
            second_pose = expected_reference @ first_pose @ expected_mount
            sample["first_observation"]["matrix_reference_from_moving"] = (
                first_pose.tolist()
            )
            sample["second_observation"]["matrix_reference_from_moving"] = (
                second_pose.tolist()
            )
        with self.assertRaisesRegex(ValueError, "effective relative rotations"):
            fit_paired_pose_calibration(parse_paired_pose_dataset(payload))

    def test_near_collinear_axes_with_small_noise_are_rejected(self) -> None:
        payload, expected_reference, expected_mount = synthetic_payload()
        axes = (
            [1, 0, 0],
            [1, 0.005, 0],
            [1, 0, 0.005],
            [1, -0.005, 0],
            [1, 0, -0.005],
            [1, 0.004, 0.003],
        )
        for index, (sample, axis) in enumerate(
            zip(payload["samples"], axes, strict=True)
        ):
            first_pose = transform(
                axis,
                0.15 * (index + 1),
                [0.03 * index, 0.01 * (index % 2), 0],
            )
            second_pose = expected_reference @ first_pose @ expected_mount
            second_pose[:3, 3] += [
                0.0001 * (index % 3),
                -0.0001 * ((index + 1) % 3),
                0.00005 * index,
            ]
            sample["first_observation"]["matrix_reference_from_moving"] = (
                first_pose.tolist()
            )
            sample["second_observation"]["matrix_reference_from_moving"] = (
                second_pose.tolist()
            )
        with self.assertRaisesRegex(ValueError, "axes separated"):
            fit_paired_pose_calibration(parse_paired_pose_dataset(payload))

    def test_reference_epoch_mixing_is_rejected(self) -> None:
        payload, _, _ = synthetic_payload()
        payload["samples"][2]["second_observation"]["source_epoch_id"] = (
            "quest-session-1:local-floor:1"
        )
        with self.assertRaisesRegex(ValueError, "reference epoch"):
            parse_paired_pose_dataset(payload)

    def test_operator_static_window_requires_stationary_evidence(self) -> None:
        payload, _, _ = synthetic_payload()
        payload["pairing_method"] = "operator_static_window"
        with self.assertRaisesRegex(ValueError, "stationary_evidence_id"):
            parse_paired_pose_dataset(payload)

        with_evidence = copy.deepcopy(payload)
        for sample in with_evidence["samples"]:
            evidence_id = f"stable-{sample['sample_id']}"
            sample["first_observation"]["stationary_evidence_id"] = evidence_id
            sample["second_observation"]["stationary_evidence_id"] = evidence_id
        parsed = parse_paired_pose_dataset(with_evidence)
        self.assertEqual(parsed.pairing_method, "operator_static_window")

    def test_operator_static_window_evidence_must_be_shared_and_unique(self) -> None:
        payload, _, _ = synthetic_payload()
        payload["pairing_method"] = "operator_static_window"
        for index, sample in enumerate(payload["samples"]):
            sample["first_observation"]["stationary_evidence_id"] = f"window-{index}"
            sample["second_observation"]["stationary_evidence_id"] = f"window-{index}"

        mismatched = copy.deepcopy(payload)
        mismatched["samples"][0]["second_observation"]["stationary_evidence_id"] = (
            "different-window"
        )
        with self.assertRaisesRegex(ValueError, "share one"):
            parse_paired_pose_dataset(mismatched)

        duplicated = copy.deepcopy(payload)
        for side in ("first_observation", "second_observation"):
            duplicated["samples"][1][side]["stationary_evidence_id"] = "window-0"
        with self.assertRaisesRegex(ValueError, "must be unique"):
            parse_paired_pose_dataset(duplicated)


if __name__ == "__main__":
    unittest.main()
