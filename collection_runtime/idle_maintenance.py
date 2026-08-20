"""Application-gated, process-isolated maintenance for sealed dataset artifacts.

The controller knows nothing about robots, operator controls, modalities, or GUI
state.  An application supplies an immutable eligibility decision and a
picklable worker.  Revoking eligibility is fail-closed: callers must obtain a
quiesce receipt before entering a latency-sensitive operation.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Mapping


IdleMaintenanceWorker = Callable[[str, Any], Mapping[str, Any]]


@dataclass(frozen=True)
class IdleMaintenanceConfig:
    process_name: str = "dataset_idle_maintenance"
    worker_nice: int = 19
    stop_timeout_sec: float = 2.0
    terminate_timeout_sec: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.process_name).strip():
            raise ValueError("process_name must be non-empty.")
        if int(self.worker_nice) < 0 or int(self.worker_nice) > 19:
            raise ValueError("worker_nice must be in [0, 19].")
        if float(self.stop_timeout_sec) < 0.0:
            raise ValueError("stop_timeout_sec must be non-negative.")
        if float(self.terminate_timeout_sec) <= 0.0:
            raise ValueError("terminate_timeout_sec must be positive.")


@dataclass(frozen=True)
class IdleMaintenanceSnapshot:
    enabled: bool
    phase: str
    dataset_root: str | None
    revision: str | None
    pid: int | None
    outcome: str | None
    error: str | None
    started_at: float | None
    completed_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdleMaintenanceQuiesceReceipt:
    quiesced: bool
    reason: str
    dataset_root: str | None
    requested_at: float
    effective_at: float
    forced_termination: bool
    filesystem_barrier_applied: bool
    worker_outcome: str | None
    worker_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sync_filesystem(path: Path) -> None:
    """Wait for dirty writes on one filesystem before publishing quiescence."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syncfs = getattr(libc, "syncfs", None)
        if syncfs is None:
            os.sync()
            return
        syncfs.argtypes = [ctypes.c_int]
        syncfs.restype = ctypes.c_int
        if int(syncfs(descriptor)) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), str(path))
    finally:
        os.close(descriptor)


def _worker_entry(
    worker: IdleMaintenanceWorker,
    dataset_root: str,
    stop_event: Any,
    result_connection: Any,
    worker_nice: int,
) -> None:
    started_at = time.time()
    try:
        nice_error = None
        if int(worker_nice) > 0:
            try:
                os.nice(int(worker_nice))
            except OSError as exc:
                nice_error = f"{type(exc).__name__}: {exc}"
        result_connection.send(
            {
                "kind": "started",
                "pid": os.getpid(),
                "started_at": started_at,
                "nice_error": nice_error,
            }
        )
        result = dict(worker(dataset_root, stop_event))
        result_connection.send(
            {
                "kind": "finished",
                "outcome": str(result.get("outcome") or "completed"),
                "result": result,
                "completed_at": time.time(),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - child must report every failure.
        try:
            result_connection.send(
                {
                    "kind": "failed",
                    "outcome": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "completed_at": time.time(),
                }
            )
        except BaseException:
            pass
    finally:
        result_connection.close()


class IdleMaintenanceProcessController:
    """Own one restartable spawned maintenance process and its stop barrier."""

    def __init__(
        self,
        worker: IdleMaintenanceWorker,
        *,
        config: IdleMaintenanceConfig | None = None,
        enabled: bool = True,
    ) -> None:
        if not callable(worker):
            raise TypeError("worker must be callable.")
        self.worker = worker
        self.config = config if config is not None else IdleMaintenanceConfig()
        self.enabled = bool(enabled)
        self._ctx = mp.get_context("spawn")
        self._process: Any | None = None
        self._stop_event: Any | None = None
        self._result_connection: Any | None = None
        self._dataset_root: Path | None = None
        self._revision: str | None = None
        self._last_attempt_revision: str | None = None
        # A forcibly stopped worker may have been killed between two durable
        # artifact boundaries. Keep that revision blocked until the
        # application validates and acknowledges recovery; otherwise the next
        # idle tick immediately starts the same worker again.
        self._blocked_revision: str | None = None
        self._forced_termination_pending = False
        self._phase = "idle" if self.enabled else "disabled"
        self._pid: int | None = None
        self._outcome: str | None = None
        self._error: str | None = None
        self._started_at: float | None = None
        self._completed_at: float | None = None
        self._quiesce_requested_at: float | None = None
        self._needs_filesystem_barrier = False

    @property
    def running(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _consume_messages(self) -> None:
        connection = self._result_connection
        if connection is None:
            return
        while connection.poll():
            try:
                message = dict(connection.recv())
            except EOFError:
                break
            kind = str(message.get("kind") or "")
            if kind == "started":
                self._pid = int(message.get("pid") or 0) or None
                self._started_at = float(message.get("started_at") or time.time())
                nice_error = str(message.get("nice_error") or "")
                if nice_error:
                    self._error = f"worker nice adjustment failed: {nice_error}"
            elif kind == "finished":
                self._outcome = str(message.get("outcome") or "completed")
                self._completed_at = float(message.get("completed_at") or time.time())
                self._error = None
            elif kind == "failed":
                self._outcome = "failed"
                self._completed_at = float(message.get("completed_at") or time.time())
                self._error = str(message.get("error") or "maintenance worker failed")

    def _release_process_handles(self) -> None:
        process = self._process
        if process is not None:
            process.join(timeout=0.0)
            process.close()
        if self._result_connection is not None:
            self._result_connection.close()
        self._process = None
        self._stop_event = None
        self._result_connection = None
        self._pid = None

    def poll(self) -> IdleMaintenanceSnapshot:
        self._consume_messages()
        process = self._process
        if process is not None and not process.is_alive():
            process.join(timeout=0.0)
            self._consume_messages()
            exit_code = process.exitcode
            if self._outcome is None:
                self._outcome = "failed"
                self._error = f"maintenance worker exited without a result (code={exit_code})"
                self._completed_at = time.time()
            if self._outcome in {"completed", "no_work"}:
                self._phase = "completed"
            elif self._outcome == "cancelled":
                self._phase = "quiesced"
                self._last_attempt_revision = None
            else:
                self._phase = "error"
            self._release_process_handles()
        return self.snapshot()

    def snapshot(self) -> IdleMaintenanceSnapshot:
        return IdleMaintenanceSnapshot(
            enabled=self.enabled,
            phase=self._phase,
            dataset_root=(None if self._dataset_root is None else str(self._dataset_root)),
            revision=self._revision,
            pid=self._pid,
            outcome=self._outcome,
            error=self._error,
            started_at=self._started_at,
            completed_at=self._completed_at,
        )

    def start_if_eligible(
        self,
        dataset_root: str | Path,
        *,
        revision: str,
    ) -> bool:
        self.poll()
        if not self.enabled or self.running:
            return False
        root = Path(dataset_root).expanduser().resolve()
        if not root.is_dir():
            return False
        revision_text = str(revision).strip()
        if not revision_text:
            raise ValueError("revision must be non-empty.")
        if self._forced_termination_pending:
            return False
        if self._blocked_revision == revision_text:
            return False
        if self._blocked_revision is not None and self._blocked_revision != revision_text:
            # A changed catalog revision is a new maintenance candidate.
            self._blocked_revision = None
        if self._last_attempt_revision == revision_text and self._outcome in {
            "completed",
            "no_work",
            "failed",
            "recovered",
        }:
            return False
        receive_connection, send_connection = self._ctx.Pipe(duplex=False)
        stop_event = self._ctx.Event()
        process = self._ctx.Process(
            target=_worker_entry,
            args=(
                self.worker,
                str(root),
                stop_event,
                send_connection,
                int(self.config.worker_nice),
            ),
            name=self.config.process_name,
            daemon=True,
        )
        self._dataset_root = root
        self._revision = revision_text
        self._last_attempt_revision = revision_text
        self._stop_event = stop_event
        self._result_connection = receive_connection
        self._phase = "starting"
        self._outcome = None
        self._error = None
        self._started_at = time.time()
        self._completed_at = None
        self._quiesce_requested_at = None
        self._needs_filesystem_barrier = False
        try:
            process.start()
        except BaseException:
            receive_connection.close()
            send_connection.close()
            self._stop_event = None
            self._result_connection = None
            self._phase = "error"
            self._last_attempt_revision = None
            self._blocked_revision = None
            self._forced_termination_pending = False
            raise
        send_connection.close()
        self._process = process
        self._pid = process.pid
        self._phase = "running"
        return True

    def request_quiesce(self, *, reason: str) -> None:
        self.poll()
        if not self.running:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        self._phase = "stopping"
        self._quiesce_requested_at = time.time()
        self._needs_filesystem_barrier = True
        if reason:
            self._error = None

    def quiesce(
        self,
        *,
        reason: str,
        timeout_sec: float | None = None,
    ) -> IdleMaintenanceQuiesceReceipt:
        requested_at = time.time()
        self.request_quiesce(reason=reason)
        process = self._process
        forced = bool(self._forced_termination_pending)
        timeout = (
            float(self.config.stop_timeout_sec)
            if timeout_sec is None
            else max(0.0, float(timeout_sec))
        )
        if process is not None:
            process.join(timeout=timeout)
            if process.is_alive():
                forced = True
                process.terminate()
                process.join(timeout=float(self.config.terminate_timeout_sec))
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=float(self.config.terminate_timeout_sec))
            if process.is_alive():
                raise RuntimeError(
                    f"Idle maintenance process {process.pid} did not stop."
                )
            self._consume_messages()
            if forced:
                self._outcome = "cancelled"
                self._error = "maintenance worker required forced termination"
                self._blocked_revision = self._revision
                self._last_attempt_revision = self._revision
                self._forced_termination_pending = True
            elif self._outcome is None:
                self._outcome = "cancelled"
                self._last_attempt_revision = None
            self._completed_at = self._completed_at or time.time()
            self._release_process_handles()
        barrier_applied = False
        if self._needs_filesystem_barrier and self._dataset_root is not None:
            _sync_filesystem(self._dataset_root)
            barrier_applied = True
        self._needs_filesystem_barrier = False
        self._phase = "quiesced" if self.enabled else "disabled"
        effective_at = time.time()
        return IdleMaintenanceQuiesceReceipt(
            quiesced=True,
            reason=str(reason),
            dataset_root=(None if self._dataset_root is None else str(self._dataset_root)),
            requested_at=requested_at,
            effective_at=effective_at,
            forced_termination=forced,
            filesystem_barrier_applied=barrier_applied,
            worker_outcome=self._outcome,
            worker_error=self._error,
        )

    def acknowledge_quiesce_receipt(
        self,
        receipt: IdleMaintenanceQuiesceReceipt,
    ) -> None:
        """Acknowledge an application-validated terminal worker error.

        A failed worker result remains sticky until the application has both
        quiesced the process and independently validated its owned resource.
        Clearing only the acknowledged error prevents an old, already-proven
        failure from being re-applied to later latency-sensitive transitions.
        The terminal outcome and attempted revision remain intact, so the same
        failed revision is not silently restarted.
        """

        if not receipt.quiesced or receipt.forced_termination:
            raise RuntimeError("Only a cooperatively quiesced receipt may be acknowledged.")
        if self._forced_termination_pending:
            raise RuntimeError(
                "Forced termination requires artifact recovery acknowledgement."
            )
        if receipt.worker_error is None:
            return
        if self.running:
            raise RuntimeError("Cannot acknowledge a worker error while maintenance is running.")
        current_root = None if self._dataset_root is None else str(self._dataset_root)
        if receipt.dataset_root != current_root:
            raise RuntimeError("Quiesce receipt dataset root is stale.")
        if receipt.worker_error != self._error:
            raise RuntimeError("Quiesce receipt worker error is stale.")
        self._error = None

    def acknowledge_forced_quiesce_receipt(
        self,
        receipt: IdleMaintenanceQuiesceReceipt,
    ) -> None:
        """Acknowledge a forced stop after application-level recovery.

        The controller does not inspect dataset files itself. The caller must
        validate its owned artifact graph first. The forced revision remains
        blocked after acknowledgement, so it cannot be restarted until a new
        catalog revision is observed.
        """

        if not receipt.quiesced or not receipt.forced_termination:
            raise RuntimeError("Expected a forcibly quiesced receipt.")
        if self.running:
            raise RuntimeError(
                "Cannot acknowledge forced termination while maintenance is running."
            )
        current_root = None if self._dataset_root is None else str(self._dataset_root)
        if receipt.dataset_root != current_root:
            raise RuntimeError("Quiesce receipt dataset root is stale.")
        if receipt.worker_error != self._error:
            raise RuntimeError("Quiesce receipt worker error is stale.")
        if not self._forced_termination_pending:
            raise RuntimeError("No forced termination is pending acknowledgement.")
        self._error = None
        self._outcome = "recovered"
        self._forced_termination_pending = False
        self._phase = "quiesced" if self.enabled else "disabled"

    def close(self) -> None:
        self.quiesce(reason="controller_close")


__all__ = [
    "IdleMaintenanceConfig",
    "IdleMaintenanceProcessController",
    "IdleMaintenanceQuiesceReceipt",
    "IdleMaintenanceSnapshot",
    "IdleMaintenanceWorker",
]
