from __future__ import annotations

import math
import threading
import unittest

from hardware.quest_dev.features.controller_pose.protocol import (
    parse_controller_pose_sample,
)
from hardware.quest_dev.features.controller_pose.runtime import (
    ControllerPoseTelemetryRuntime,
)
from hardware.quest_dev.protocol import PROTOCOL_VERSION


def identity_matrix(*, x: float = 0.0) -> list[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1]


def pose_payload(
    *,
    sequence: int = 1,
    generation: int = 0,
    matrix: list[float] | None = None,
    monotonic_ns: int = 1_000_000_000,
) -> dict:
    return {
        "type": "controller_pose_sample",
        "feature": "controller_pose",
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": 1,
        "client_instance_id": "browser-instance",
        "xr_session_id": "xr-session",
        "reference_space": {"type": "local-floor", "generation": generation},
        "sample_sequence": sequence,
        "xr_frame_time_ms": 1234.5,
        "controllers": [
            {
                "source_session_id": "input-1",
                "handedness": "right",
                "target_ray_mode": "tracked-pointer",
                "profiles": ["oculus-touch-v3"],
                "emulated_position": False,
                "matrix_reference_from_grip": matrix or identity_matrix(x=0.25),
            }
        ],
        "server_timestamp": 100.0,
        "server_monotonic_ns": monotonic_ns,
    }


def session_payload(*, state: str = "started", generation: int = 0) -> dict:
    return {
        "type": "xr_session_state",
        "protocol_version": PROTOCOL_VERSION,
        "state": state,
        "client_instance_id": "browser-instance",
        "xr_session_id": "xr-session",
        "reference_space_type": "local-floor",
        "reference_space_generation": generation,
    }


class ControllerPoseProtocolTests(unittest.TestCase):
    def test_parses_column_major_grip_pose(self) -> None:
        sample = parse_controller_pose_sample(
            pose_payload(),
            received_wall_time_s=100.0,
            received_monotonic_ns=1_000_000_000,
        )
        self.assertEqual(sample.controllers[0].matrix_reference_from_grip[12], 0.25)
        self.assertEqual(sample.reference_space_generation, 0)

    def test_non_finite_matrix_is_rejected(self) -> None:
        matrix = identity_matrix()
        matrix[0] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_controller_pose_sample(
                pose_payload(matrix=matrix),
                received_wall_time_s=100.0,
                received_monotonic_ns=1,
            )

    def test_reflection_is_rejected(self) -> None:
        matrix = identity_matrix()
        matrix[0] = -1
        with self.assertRaisesRegex(ValueError, "determinant"):
            parse_controller_pose_sample(
                pose_payload(matrix=matrix),
                received_wall_time_s=100.0,
                received_monotonic_ns=1,
            )

    def test_control_fields_are_rejected(self) -> None:
        payload = pose_payload()
        payload["command"] = "move"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            parse_controller_pose_sample(
                payload,
                received_wall_time_s=100.0,
                received_monotonic_ns=1,
            )


class FakeServer:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.disconnect_handler = None
        self.status_provider = None

    def register_client_message_handler(self, message_type: str, handler) -> None:
        self.handlers[message_type] = handler

    def register_client_disconnect_handler(self, handler) -> None:
        self.disconnect_handler = handler

    def register_feature_status_provider(self, feature: str, provider) -> None:
        self.feature = feature
        self.status_provider = provider


class ControllerPoseRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = FakeServer()
        self.runtime = ControllerPoseTelemetryRuntime(
            self.server,
            {"admission": {"max_rate_hz": 40, "max_controllers": 2}},
            threading.Event(),
        )
        self.runtime.start()
        self.server.handlers["xr_session_state"](1, session_payload())

    def test_latest_only_sequence_and_rate_admission(self) -> None:
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))
        latest = self.runtime.latest_sample(client_id=1)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.sample_sequence, 1)

        self.server.handlers["controller_pose_sample"](
            1, pose_payload(sequence=2, monotonic_ns=1_010_000_000)
        )
        self.assertEqual(self.runtime.rejected_count, 1)
        self.assertEqual(self.runtime.latest_sample(client_id=1).sample_sequence, 1)

        self.server.handlers["controller_pose_sample"](
            1, pose_payload(sequence=3, monotonic_ns=1_030_000_000)
        )
        self.assertEqual(self.runtime.latest_sample(client_id=1).sample_sequence, 3)

    def test_reset_clears_latest_and_rejects_old_generation(self) -> None:
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))
        self.server.handlers["reference_space_reset"](
            1,
            {
                "protocol_version": PROTOCOL_VERSION,
                "client_instance_id": "browser-instance",
                "xr_session_id": "xr-session",
                "reference_space_type": "local-floor",
                "previous_generation": 0,
                "new_generation": 1,
            },
        )
        self.assertIsNone(self.runtime.latest_sample(client_id=1))
        self.server.handlers["controller_pose_sample"](
            1, pose_payload(sequence=2, generation=0, monotonic_ns=1_030_000_000)
        )
        self.assertIsNone(self.runtime.latest_sample(client_id=1))
        self.server.handlers["controller_pose_sample"](
            1, pose_payload(sequence=3, generation=1, monotonic_ns=1_060_000_000)
        )
        self.assertEqual(
            self.runtime.latest_sample(client_id=1).reference_space_generation, 1
        )

    def test_session_end_and_disconnect_remove_state(self) -> None:
        self.server.handlers["controller_pose_sample"](1, pose_payload())
        self.server.handlers["xr_session_state"](1, session_payload(state="ended"))
        self.assertIsNone(self.runtime.latest_sample(client_id=1))

        self.server.handlers["xr_session_state"](1, session_payload())
        self.server.disconnect_handler(1)
        self.assertIsNone(self.runtime.latest_sample(client_id=1))

    def test_overlapping_session_start_is_rejected(self) -> None:
        self.server.handlers["xr_session_state"](1, session_payload())
        self.assertEqual(self.runtime.rejected_count, 1)

    def test_wrong_lifecycle_protocol_version_is_rejected(self) -> None:
        payload = session_payload()
        payload["protocol_version"] = 1
        self.server.handlers["xr_session_state"](2, payload)
        self.assertIsNone(self.runtime.latest_sample(client_id=2))
        self.assertEqual(self.runtime.rejected_count, 1)

    def test_session_start_requires_generation_zero_and_known_reference_type(
        self,
    ) -> None:
        self.server.handlers["xr_session_state"](2, session_payload(generation=1))
        invalid_reference = session_payload()
        invalid_reference["reference_space_type"] = "unknown-space"
        self.server.handlers["xr_session_state"](3, invalid_reference)
        self.assertNotIn(2, self.runtime.states)
        self.assertNotIn(3, self.runtime.states)
        self.assertEqual(self.runtime.rejected_count, 2)

    def test_status_is_pose_redacted(self) -> None:
        self.server.handlers["controller_pose_sample"](1, pose_payload())
        status = self.runtime.snapshot_status()
        self.assertEqual(status["mode"], "latest_only")
        self.assertEqual(status["clients"]["1"]["controller_count"], 1)
        self.assertNotIn("matrix", str(status))


if __name__ == "__main__":
    unittest.main()
