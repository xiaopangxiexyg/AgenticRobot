from __future__ import annotations

import math
import threading
import unittest

from hardware.quest_dev.features.controller_pose.protocol import (
    parse_controller_pose_sample,
)
from hardware.quest_dev.features.controller_pose.runtime import (
    ControllerPoseAcceptedEvent,
    ControllerPoseInvalidationEvent,
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
    controllers: list[dict] | None = None,
    controller_diagnostics: list[dict] | None = None,
) -> dict:
    raw_controllers = controllers
    if raw_controllers is None:
        raw_controllers = [
            {
                "source_session_id": "input-1",
                "handedness": "right",
                "target_ray_mode": "tracked-pointer",
                "profiles": ["oculus-touch-v3"],
                "emulated_position": False,
                "matrix_reference_from_grip": matrix or identity_matrix(x=0.25),
            }
        ]
    payload = {
        "type": "controller_pose_sample",
        "feature": "controller_pose",
        "protocol_version": PROTOCOL_VERSION,
        "feature_protocol_version": 1,
        "client_instance_id": "browser-instance",
        "xr_session_id": "xr-session",
        "reference_space": {"type": "local-floor", "generation": generation},
        "sample_sequence": sequence,
        "xr_frame_time_ms": 1234.5,
        "controllers": raw_controllers,
        "server_timestamp": 100.0,
        "server_monotonic_ns": monotonic_ns,
    }
    if controller_diagnostics is not None:
        payload["controller_diagnostics"] = controller_diagnostics
    return payload


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

    def test_untracked_reason_is_admitted_without_a_pose(self) -> None:
        sample = parse_controller_pose_sample(
            pose_payload(
                controllers=[],
                controller_diagnostics=[
                    {
                        "source_session_id": "input-1",
                        "handedness": "left",
                        "target_ray_mode": "tracked-pointer",
                        "tracking_state": "untracked",
                        "reason": "pose_unavailable",
                    }
                ],
            ),
            received_wall_time_s=100.0,
            received_monotonic_ns=1,
        )
        self.assertEqual(sample.controllers, ())
        self.assertEqual(
            sample.controller_diagnostics[0].reason,
            "pose_unavailable",
        )

    def test_pose_and_tracking_diagnostic_cannot_contradict(self) -> None:
        with self.assertRaisesRegex(ValueError, "untracked diagnostic"):
            parse_controller_pose_sample(
                pose_payload(
                    controller_diagnostics=[
                        {
                            "source_session_id": "input-1",
                            "handedness": "right",
                            "target_ray_mode": "tracked-pointer",
                            "tracking_state": "untracked",
                            "reason": "pose_unavailable",
                        }
                    ],
                ),
                received_wall_time_s=100.0,
                received_monotonic_ns=1,
            )


class FakeServer:
    def __init__(self) -> None:
        self.server_instance_id = "quest-host-instance"
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
            {
                "receipt_clock_domain_id": "host-boot-1.monotonic",
                "admission": {"max_rate_hz": 40, "max_controllers": 2},
            },
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

    def test_empty_controller_sample_from_get_pose_loss_is_admitted(self) -> None:
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=1, controllers=[]),
        )

        latest = self.runtime.latest_sample(client_id=1)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.controllers, ())
        self.assertEqual(self.runtime.accepted_count, 1)
        self.assertEqual(self.runtime.rejected_count, 0)

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
        self.assertEqual(
            status["clients"]["1"]["controller_tracking"][0]["reason"],
            "tracked",
        )
        self.assertNotIn("matrix", str(status))

    def test_observer_receives_only_admitted_samples_and_ordered_invalidation(
        self,
    ) -> None:
        events: list[ControllerPoseAcceptedEvent | ControllerPoseInvalidationEvent] = []
        self.runtime.subscribe(
            on_sample=events.append,
            on_invalidation=events.append,
        )

        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=2, monotonic_ns=1_010_000_000),
        )
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
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(
                sequence=3,
                generation=1,
                monotonic_ns=1_030_000_000,
            ),
        )

        self.assertEqual([event.event_sequence for event in events], [0, 1, 2])
        self.assertIsInstance(events[0], ControllerPoseAcceptedEvent)
        self.assertIsInstance(events[1], ControllerPoseInvalidationEvent)
        self.assertEqual(events[1].reason, "reference_space_reset")
        self.assertEqual(events[1].previous_reference_space_generation, 0)
        self.assertEqual(events[1].next_reference_space_generation, 1)
        self.assertEqual(events[1].last_accepted_sample_sequence, 1)
        self.assertEqual(
            events[1].receipt_clock_domain_id,
            "host-boot-1.monotonic",
        )
        self.assertEqual(events[2].sample.reference_space_generation, 1)

    def test_subscription_returns_atomic_pose_redacted_session_cut(self) -> None:
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))
        events: list[ControllerPoseAcceptedEvent] = []

        subscription = self.runtime.subscribe_with_cut(on_sample=events.append)

        self.assertEqual(subscription.next_event_sequence, 1)
        self.assertEqual(subscription.server_instance_id, "quest-host-instance")
        self.assertEqual(subscription.receipt_clock_domain_id, "host-boot-1.monotonic")
        self.assertEqual(len(subscription.active_sessions), 1)
        session = subscription.active_sessions[0]
        self.assertEqual(session.client_id, 1)
        self.assertEqual(session.xr_session_id, "xr-session")
        self.assertEqual(session.reference_space_generation, 0)
        self.assertEqual(session.last_accepted_sample_sequence, 1)
        self.assertNotIn("matrix", repr(subscription))

        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=2, monotonic_ns=1_030_000_000),
        )
        self.assertEqual([event.event_sequence for event in events], [1])
        self.assertEqual(events[0].server_instance_id, "quest-host-instance")
        self.assertTrue(self.runtime.unsubscribe(subscription.token))

    def test_atomic_event_cut_is_pose_redacted_and_does_not_subscribe(self) -> None:
        before = self.runtime.atomic_event_cut()
        self.assertEqual(before.next_event_sequence, 0)
        self.assertEqual(before.server_instance_id, "quest-host-instance")
        self.assertEqual(before.receipt_clock_domain_id, "host-boot-1.monotonic")
        self.assertFalse(before.stopped)
        self.assertEqual(len(before.active_sessions), 1)
        self.assertNotIn("matrix", repr(before))
        self.assertEqual(self.runtime.snapshot_status()["observer_count"], 0)

        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))
        after_sample = self.runtime.atomic_event_cut()
        self.assertEqual(after_sample.next_event_sequence, 1)
        self.assertEqual(
            after_sample.active_sessions[0].last_accepted_sample_sequence,
            1,
        )

        self.runtime.stop()
        after_stop = self.runtime.atomic_event_cut()
        self.assertTrue(after_stop.stopped)
        self.assertEqual(after_stop.next_event_sequence, 2)
        self.assertEqual(after_stop.active_sessions, ())

    def test_unsubscribe_with_cut_has_no_event_allocation_gap(self) -> None:
        events: list[ControllerPoseAcceptedEvent] = []
        subscription = self.runtime.subscribe_with_cut(on_sample=events.append)
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=1),
        )

        close_cut = self.runtime.unsubscribe_with_cut(subscription.token)

        self.assertEqual(close_cut.subscription_token, subscription.token)
        self.assertEqual(close_cut.next_event_sequence, 1)
        self.assertEqual(
            close_cut.active_sessions[0].last_accepted_sample_sequence,
            1,
        )
        self.assertEqual(self.runtime.snapshot_status()["observer_count"], 0)
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=2, monotonic_ns=1_030_000_000),
        )
        self.assertEqual([event.event_sequence for event in events], [0])
        with self.assertRaisesRegex(KeyError, "not active"):
            self.runtime.unsubscribe_with_cut(subscription.token)

    def test_observer_failure_does_not_reclassify_an_admitted_sample(self) -> None:
        def fail(_event) -> None:
            raise RuntimeError("observer boom")

        self.runtime.subscribe(on_sample=fail)
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))

        self.assertEqual(self.runtime.accepted_count, 1)
        self.assertEqual(self.runtime.rejected_count, 0)
        self.assertEqual(self.runtime.latest_sample(client_id=1).sample_sequence, 1)
        status = self.runtime.snapshot_status()
        self.assertEqual(status["observer_error_count"], 1)
        self.assertIn("observer boom", status["last_observer_error"])

    def test_observer_callback_can_unsubscribe_without_source_lock_reentry(
        self,
    ) -> None:
        events: list[ControllerPoseAcceptedEvent] = []
        token: list[int] = []

        def observe(event: ControllerPoseAcceptedEvent) -> None:
            events.append(event)
            self.assertIsNotNone(self.runtime.latest_sample(client_id=1))
            self.assertTrue(self.runtime.unsubscribe(token[0]))

        token.append(self.runtime.subscribe(on_sample=observe))
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=2, monotonic_ns=1_030_000_000),
        )

        self.assertEqual(len(events), 1)

    def test_observer_callback_is_invoked_outside_runtime_lock(self) -> None:
        lock_acquired_from_other_thread = threading.Event()

        def observe(_event: ControllerPoseAcceptedEvent) -> None:
            reader = threading.Thread(
                target=lambda: (
                    self.runtime.snapshot_status(),
                    lock_acquired_from_other_thread.set(),
                )
            )
            reader.start()
            self.assertTrue(lock_acquired_from_other_thread.wait(timeout=1.0))
            reader.join(timeout=1.0)
            self.assertFalse(reader.is_alive())

        self.runtime.subscribe(on_sample=observe)
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))

    def test_disconnect_and_stop_emit_terminal_invalidation(self) -> None:
        events: list[ControllerPoseInvalidationEvent] = []
        self.runtime.subscribe(on_invalidation=events.append)
        self.server.disconnect_handler(1)

        self.assertEqual(events[-1].reason, "client_disconnected")
        self.server.handlers["xr_session_state"](1, session_payload())
        self.runtime.stop()

        self.assertEqual(events[-1].reason, "runtime_stopped")
        self.assertIsNone(self.runtime.latest_sample(client_id=1))

    def test_session_end_emits_complete_invalidation_identity(self) -> None:
        events: list[ControllerPoseInvalidationEvent] = []
        self.runtime.subscribe(on_invalidation=events.append)
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))

        self.server.handlers["xr_session_state"](1, session_payload(state="ended"))

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.reason, "session_ended")
        self.assertEqual(event.server_instance_id, "quest-host-instance")
        self.assertEqual(event.client_instance_id, "browser-instance")
        self.assertEqual(event.xr_session_id, "xr-session")
        self.assertEqual(event.reference_space_type, "local-floor")
        self.assertEqual(event.previous_reference_space_generation, 0)
        self.assertIsNone(event.next_reference_space_generation)
        self.assertEqual(event.last_accepted_sample_sequence, 1)

    def test_invalidation_observer_failure_is_isolated_and_lock_free(self) -> None:
        lock_acquired_from_other_thread = threading.Event()

        def fail(_event: ControllerPoseInvalidationEvent) -> None:
            reader = threading.Thread(
                target=lambda: (
                    self.runtime.snapshot_status(),
                    lock_acquired_from_other_thread.set(),
                )
            )
            reader.start()
            self.assertTrue(lock_acquired_from_other_thread.wait(timeout=1.0))
            reader.join(timeout=1.0)
            self.assertFalse(reader.is_alive())
            raise RuntimeError("invalidation observer boom")

        self.runtime.subscribe(on_invalidation=fail)
        self.server.disconnect_handler(1)

        status = self.runtime.snapshot_status()
        self.assertEqual(status["observer_error_count"], 1)
        self.assertIn("invalidation observer boom", status["last_observer_error"])
        self.assertEqual(self.runtime.rejected_count, 0)

    def test_invalid_lifecycle_messages_do_not_emit_events(self) -> None:
        events: list[ControllerPoseInvalidationEvent] = []
        self.runtime.subscribe(on_invalidation=events.append)

        self.server.disconnect_handler(99)
        invalid_end = session_payload(state="ended")
        invalid_end["xr_session_id"] = "other-session"
        self.server.handlers["xr_session_state"](1, invalid_end)

        self.assertEqual(events, [])
        self.assertEqual(self.runtime.rejected_count, 1)

    def test_observer_count_is_bounded(self) -> None:
        runtime = ControllerPoseTelemetryRuntime(
            self.server,
            {
                "admission": {
                    "max_rate_hz": 40,
                    "max_controllers": 2,
                    "max_observers": 1,
                }
            },
            threading.Event(),
        )
        runtime.subscribe(on_sample=lambda _event: None)
        with self.assertRaisesRegex(RuntimeError, "observer limit"):
            runtime.subscribe(on_sample=lambda _event: None)

    def test_stop_is_terminal_and_idempotent(self) -> None:
        events: list[ControllerPoseInvalidationEvent] = []
        self.runtime.subscribe(on_invalidation=events.append)
        self.server.handlers["controller_pose_sample"](1, pose_payload(sequence=1))

        self.runtime.stop()
        self.runtime.stop()

        self.assertEqual([event.reason for event in events], ["runtime_stopped"])
        self.assertEqual(events[0].last_accepted_sample_sequence, 1)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.runtime.subscribe(on_sample=lambda _event: None)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.runtime.start()

        self.server.handlers["xr_session_state"](1, session_payload())
        self.server.handlers["controller_pose_sample"](
            1,
            pose_payload(sequence=2, monotonic_ns=1_030_000_000),
        )
        self.assertEqual(self.runtime.accepted_count, 1)
        self.assertEqual(self.runtime.rejected_count, 2)
        self.assertEqual(self.runtime.snapshot_status()["clients"], {})


if __name__ == "__main__":
    unittest.main()
