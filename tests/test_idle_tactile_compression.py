from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collection_runtime.idle_maintenance import (
    IdleMaintenanceConfig,
    IdleMaintenanceProcessController,
    IdleMaintenanceQuiesceReceipt,
    IdleMaintenanceSnapshot,
)
from teleop_runtime.real_teleop.tactile_idle_compression import (
    RealTeleopIdleTactileCompression,
    assert_idle_tactile_transition_safe,
)
from dataset_interface.tactile_bundle import TactileBundleAttachmentError


def _cooperative_worker(dataset_root: str, stop_event: object) -> dict[str, object]:
    marker = Path(dataset_root) / "worker.started"
    marker.write_text("started", encoding="utf-8")
    while not stop_event.wait(0.01):
        pass
    return {"outcome": "cancelled"}


def _no_work_worker(dataset_root: str, stop_event: object) -> dict[str, object]:
    del dataset_root, stop_event
    return {"outcome": "no_work"}


def _failing_worker(dataset_root: str, stop_event: object) -> dict[str, object]:
    del dataset_root, stop_event
    raise RuntimeError("synthetic maintenance failure")


def _uncooperative_worker(dataset_root: str, stop_event: object) -> dict[str, object]:
    del dataset_root, stop_event
    while True:
        time.sleep(1.0)


def _wait_until(predicate: object, *, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


class IdleMaintenanceProcessControllerTest(unittest.TestCase):
    def test_default_disabled_controller_never_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = IdleMaintenanceProcessController(
                _no_work_worker,
                enabled=False,
            )
            self.assertFalse(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            self.assertEqual(controller.snapshot().phase, "disabled")

    def test_cooperative_quiesce_joins_worker_before_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = IdleMaintenanceProcessController(
                _cooperative_worker,
                config=IdleMaintenanceConfig(
                    stop_timeout_sec=1.0,
                    terminate_timeout_sec=0.5,
                ),
            )
            self.assertTrue(
                controller.start_if_eligible(root, revision="revision-1")
            )
            _wait_until(lambda: (root / "worker.started").is_file())
            with mock.patch(
                "collection_runtime.idle_maintenance._sync_filesystem"
            ) as sync_filesystem:
                receipt = controller.quiesce(reason="runtime_home")

            self.assertTrue(receipt.quiesced)
            self.assertFalse(receipt.forced_termination)
            self.assertTrue(receipt.filesystem_barrier_applied)
            self.assertFalse(controller.running)
            sync_filesystem.assert_called_once_with(root.resolve())
            # A cancelled revision is explicitly retryable during the next
            # eligible idle interval.
            self.assertTrue(
                controller.start_if_eligible(root, revision="revision-1")
            )
            controller.quiesce(reason="test_cleanup")

    def test_completed_revision_is_not_restarted_until_catalog_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = IdleMaintenanceProcessController(_no_work_worker)
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            _wait_until(lambda: controller.poll().phase == "completed")
            self.assertFalse(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-2")
            )
            _wait_until(lambda: controller.poll().phase == "completed")
            controller.close()

    def test_validated_worker_error_is_not_replayed_to_later_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = IdleMaintenanceProcessController(_failing_worker)
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            _wait_until(lambda: controller.poll().phase == "error")

            receipt = controller.quiesce(reason="episode_start")
            self.assertIn("synthetic maintenance failure", receipt.worker_error or "")
            controller.acknowledge_quiesce_receipt(receipt)

            later = controller.quiesce(reason="runtime_home")
            self.assertIsNone(later.worker_error)
            self.assertEqual(later.worker_outcome, "failed")
            self.assertFalse(
                controller.start_if_eligible(temporary, revision="revision-1")
            )

    def test_forced_stop_blocks_same_revision_until_recovery_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = IdleMaintenanceProcessController(
                _uncooperative_worker,
                config=IdleMaintenanceConfig(
                    stop_timeout_sec=0.01,
                    terminate_timeout_sec=0.5,
                ),
            )
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            receipt = controller.quiesce(reason="runtime_home")
            self.assertTrue(receipt.forced_termination)
            retry_receipt = controller.quiesce(reason="runtime_home_retry")
            self.assertTrue(retry_receipt.forced_termination)
            self.assertFalse(
                controller.start_if_eligible(temporary, revision="revision-1")
            )

            controller.acknowledge_forced_quiesce_receipt(retry_receipt)
            self.assertFalse(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-2")
            )
            controller.quiesce(reason="test_cleanup")

    def test_degraded_quiesce_disables_worker_without_restarting_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = IdleMaintenanceProcessController(_no_work_worker)
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            _wait_until(lambda: controller.poll().phase == "completed")
            receipt = controller.quiesce(reason="runtime_home")

            controller.disable_after_degraded_quiesce_receipt(
                receipt,
                reason="historical tactile payload hash mismatch",
            )

            snapshot = controller.snapshot()
            self.assertFalse(snapshot.enabled)
            self.assertEqual(snapshot.phase, "disabled")
            self.assertEqual(snapshot.outcome, "degraded")
            self.assertFalse(
                controller.start_if_eligible(temporary, revision="revision-2")
            )


class RealTeleopIdleTactileCompressionAdmissionTest(unittest.TestCase):
    def test_historical_payload_hash_mismatch_disables_idle_and_admits_transition(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="runtime_home",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=False,
            filesystem_barrier_applied=True,
            worker_outcome="failed",
            worker_error="TactileBundleAttachmentError: tactile artifact hash mismatch: tactile/foo.zst",
        )
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.controller = mock.Mock()
        facade.require_quiesced = mock.Mock(return_value=receipt)
        logs: list[tuple[str, str]] = []

        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression.assert_idle_tactile_transition_safe",
            side_effect=TactileBundleAttachmentError(
                "tactile artifact hash mismatch: tactile/foo.zst"
            ),
        ):
            returned = facade.require_safe_transition(
                reason="runtime_home",
                dataset_root="/tmp/synthetic-dataset",
                integrity_enabled=True,
                log=lambda category, message: logs.append((category, message)),
            )

        self.assertIs(returned, receipt)
        facade.controller.disable_after_degraded_quiesce_receipt.assert_called_once()
        self.assertFalse(facade.enabled)
        self.assertTrue(logs)
        self.assertIn("Home/数采继续", logs[0][1])

    def test_wrapped_forced_payload_hash_mismatch_uses_same_degraded_path(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="runtime_home",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=True,
            filesystem_barrier_applied=True,
            worker_outcome="cancelled",
            worker_error="maintenance worker required forced termination",
        )
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.controller = mock.Mock()
        facade.require_quiesced = mock.Mock(return_value=receipt)
        logs: list[tuple[str, str]] = []

        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression.assert_idle_tactile_transition_safe",
            side_effect=RuntimeError(
                "idle tactile compression required forced termination; "
                "artifact recovery validation failed: TactileBundleAttachmentError: "
                "tactile artifact hash mismatch: tactile/foo.zst"
            ),
        ):
            returned = facade.require_safe_transition(
                reason="runtime_home",
                dataset_root="/tmp/synthetic-dataset",
                integrity_enabled=True,
                log=lambda category, message: logs.append((category, message)),
            )

        self.assertIs(returned, receipt)
        facade.controller.disable_after_degraded_quiesce_receipt.assert_called_once()
        self.assertFalse(facade.enabled)
        self.assertTrue(logs)

    def test_manifest_hash_mismatch_remains_fail_closed(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="runtime_home",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=False,
            filesystem_barrier_applied=True,
            worker_outcome="failed",
            worker_error=(
                "TactileBundleAttachmentError: tactile artifact hash mismatch: "
                "tactile/chunk-000/episode_000018/right/manifest.json"
            ),
        )
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.controller = mock.Mock()
        facade.require_quiesced = mock.Mock(return_value=receipt)

        error = TactileBundleAttachmentError(
            "tactile artifact hash mismatch: "
            "tactile/chunk-000/episode_000018/right/manifest.json"
        )
        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression.assert_idle_tactile_transition_safe",
            side_effect=error,
        ):
            with self.assertRaisesRegex(TactileBundleAttachmentError, "manifest.json"):
                facade.require_safe_transition(
                    reason="runtime_home",
                    dataset_root="/tmp/synthetic-dataset",
                    integrity_enabled=True,
                    log=lambda *_args: None,
                )

        facade.controller.disable_after_degraded_quiesce_receipt.assert_not_called()
        self.assertTrue(facade.enabled)

    def test_acknowledged_error_does_not_rescan_open_episode_on_runtime_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = IdleMaintenanceProcessController(_failing_worker)
            self.assertTrue(
                controller.start_if_eligible(temporary, revision="revision-1")
            )
            _wait_until(lambda: controller.poll().phase == "error")
            episode_start = controller.quiesce(reason="episode_start")
            with mock.patch(
                "teleop_runtime.real_teleop.tactile_idle_compression."
                "assert_dataset_capture_writable"
            ) as writable:
                assert_idle_tactile_transition_safe(
                    episode_start,
                    dataset_root=temporary,
                    integrity_enabled=False,
                )
            writable.assert_called_once_with(temporary)
            controller.acknowledge_quiesce_receipt(episode_start)

            # A live Episode owns files that are intentionally absent from the
            # finalized catalog until seal.  A later Home boundary must not
            # replay the old worker failure and misclassify those files as an
            # unowned cold graph.
            runtime_home = controller.quiesce(reason="runtime_home")
            self.assertIsNone(runtime_home.worker_error)
            with mock.patch(
                "teleop_runtime.real_teleop.tactile_idle_compression."
                "assert_dataset_capture_writable"
            ) as writable:
                assert_idle_tactile_transition_safe(
                    runtime_home,
                    dataset_root=temporary,
                    integrity_enabled=False,
                )
            writable.assert_not_called()

    def test_forced_worker_stop_admits_after_graph_validation(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="episode_start",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=True,
            filesystem_barrier_applied=True,
            worker_outcome="cancelled",
            worker_error="maintenance worker required forced termination",
        )

        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression."
            "assert_dataset_capture_writable"
        ) as writable:
            assert_idle_tactile_transition_safe(
                receipt,
                dataset_root="/tmp/synthetic-dataset",
                integrity_enabled=True,
            )
        writable.assert_called_once_with("/tmp/synthetic-dataset")

    def test_forced_worker_stop_keeps_transition_closed_when_recovery_fails(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="episode_start",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=True,
            filesystem_barrier_applied=True,
            worker_outcome="cancelled",
            worker_error="maintenance worker required forced termination",
        )

        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression."
            "assert_dataset_capture_writable",
            side_effect=RuntimeError("synthetic recovery failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "artifact recovery validation failed"
            ):
                assert_idle_tactile_transition_safe(
                    receipt,
                    dataset_root="/tmp/synthetic-dataset",
                    integrity_enabled=False,
                )

    def test_cooperative_worker_error_revalidates_integrity_graph(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="episode_start",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=False,
            filesystem_barrier_applied=True,
            worker_outcome="failed",
            worker_error="synthetic worker failure",
        )

        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression."
            "assert_dataset_capture_writable"
        ) as writable, mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression."
            "assert_dataset_capture_integrity_compatible"
        ) as integrity:
            assert_idle_tactile_transition_safe(
                receipt,
                dataset_root="/tmp/synthetic-dataset",
                integrity_enabled=True,
            )

        writable.assert_called_once_with("/tmp/synthetic-dataset")
        integrity.assert_called_once_with("/tmp/synthetic-dataset")

    def test_only_idle_recorder_in_prehome_or_hold_can_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "meta/episodes.jsonl").write_text(
                '{"episode_index":0}\n',
                encoding="utf-8",
            )
            recorder = SimpleNamespace(
                dataset_root=root,
                is_recording=False,
                finalizing=False,
            )
            snapshot = IdleMaintenanceSnapshot(
                enabled=True,
                phase="idle",
                dataset_root=None,
                revision=None,
                pid=None,
                outcome=None,
                error=None,
                started_at=None,
                completed_at=None,
            )
            facade = RealTeleopIdleTactileCompression(enabled=True)
            facade.controller = mock.Mock()
            facade.controller.poll.return_value = snapshot
            facade.controller.snapshot.return_value = snapshot
            facade.controller.running = False

            facade.tick(
                dataset_recorder=recorder,
                teleop_started=True,
                hold_active=False,
                episode_countdown_active=False,
                startup_and_devices_ready=True,
            )
            facade.controller.request_quiesce.assert_called_once()
            facade.controller.start_if_eligible.assert_not_called()

            facade.controller.reset_mock()
            recorder.is_recording = True
            facade.tick(
                dataset_recorder=recorder,
                teleop_started=False,
                hold_active=True,
                episode_countdown_active=False,
                startup_and_devices_ready=True,
            )
            facade.controller.request_quiesce.assert_called_once()
            facade.controller.start_if_eligible.assert_not_called()

            facade.controller.reset_mock()
            recorder.is_recording = False
            facade.tick(
                dataset_recorder=recorder,
                teleop_started=True,
                hold_active=True,
                episode_countdown_active=False,
                startup_and_devices_ready=True,
            )
            facade.controller.start_if_eligible.assert_called_once()
            facade.controller.request_quiesce.assert_not_called()

    def test_facade_acknowledges_only_after_application_validation(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="episode_start",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=False,
            filesystem_barrier_applied=True,
            worker_outcome="failed",
            worker_error="synthetic worker failure",
        )
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.controller = mock.Mock()

        facade.acknowledge_safe_transition(receipt)

        facade.controller.acknowledge_quiesce_receipt.assert_called_once_with(
            receipt
        )

    def test_runtime_tick_logs_only_changed_snapshot(self) -> None:
        snapshot = IdleMaintenanceSnapshot(
            enabled=True,
            phase="idle",
            dataset_root=None,
            revision=None,
            pid=None,
            outcome=None,
            error=None,
            started_at=None,
            completed_at=None,
        )
        logs: list[tuple[str, str]] = []
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.tick = mock.Mock(return_value=snapshot)

        for _ in range(2):
            self.assertIs(
                facade.tick_runtime(
                    dataset_recorder=None,
                    teleop_started=False,
                    hold_active=True,
                    episode_countdown_active=False,
                    startup_and_devices_ready=True,
                    log=lambda tag, message: logs.append((tag, message)),
                ),
                snapshot,
            )

        self.assertEqual(len(logs), 1)
        self.assertIn("phase=idle", logs[0][1])

    def test_runtime_tick_isolates_and_deduplicates_optional_errors(self) -> None:
        logs: list[tuple[str, str]] = []
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.tick = mock.Mock(side_effect=OSError("synthetic admission error"))

        for _ in range(2):
            self.assertIsNone(
                facade.tick_runtime(
                    dataset_recorder=None,
                    teleop_started=False,
                    hold_active=True,
                    episode_countdown_active=False,
                    startup_and_devices_ready=True,
                    log=lambda tag, message: logs.append((tag, message)),
                )
            )

        self.assertEqual(len(logs), 1)
        self.assertIn("OSError: synthetic admission error", logs[0][1])

    def test_safe_transition_validates_before_acknowledging_receipt(self) -> None:
        receipt = IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason="episode_start",
            dataset_root="/tmp/synthetic-dataset",
            requested_at=1.0,
            effective_at=2.0,
            forced_termination=False,
            filesystem_barrier_applied=True,
            worker_outcome="cancelled",
            worker_error=None,
        )
        calls: list[object] = []
        facade = RealTeleopIdleTactileCompression(enabled=True)
        facade.require_quiesced = mock.Mock(
            side_effect=lambda **_kwargs: calls.append("quiesce") or receipt
        )
        facade.acknowledge_safe_transition = mock.Mock(
            side_effect=lambda value: calls.append(("ack", value))
        )

        with mock.patch(
            "teleop_runtime.real_teleop.tactile_idle_compression."
            "assert_idle_tactile_transition_safe",
            side_effect=lambda value, **_kwargs: calls.append(("validate", value)),
        ):
            result = facade.require_safe_transition(
                reason="episode_start",
                dataset_root="/tmp/synthetic-dataset",
                integrity_enabled=True,
                timeout_sec=2.0,
                log=lambda tag, message: calls.append((tag, message)),
            )

        self.assertIs(result, receipt)
        self.assertEqual(calls[0], "quiesce")
        self.assertEqual(calls[1], ("validate", receipt))
        self.assertEqual(calls[2], ("ack", receipt))
        self.assertEqual(calls[3][0], "TactileCompression")


if __name__ == "__main__":
    unittest.main()
