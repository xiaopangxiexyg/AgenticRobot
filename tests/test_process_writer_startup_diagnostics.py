from __future__ import annotations

import inspect
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest

from teleop_runtime.collection import process_writer
from teleop_runtime.collection.process_writer import ProcessCollectionWorker


class _ExitedProcess:
    exitcode = 23

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False


class _EmptyReplyQueue:
    def get(self, *, timeout: float):
        del timeout
        raise queue.Empty


class _ProgressThenEmptyReplyQueue:
    def __init__(self) -> None:
        self._sent = False

    def get(self, *, timeout: float):
        del timeout
        if not self._sent:
            self._sent = True
            return {
                "kind": "startup_progress",
                "phase": "initialize_legacy_writer",
            }
        raise queue.Empty


class _AliveProcess:
    exitcode = None

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return True


class _WorkerErrorAckQueue:
    def __init__(self) -> None:
        self._messages = [
            {
                "kind": "worker_error",
                "error": "worker.plan_jobs: DatasetLayoutError: bad sample",
                "traceback": "Traceback (most recent call last):\n  worker.plan_jobs",
            }
        ]

    def get_nowait(self):
        if self._messages:
            return self._messages.pop(0)
        raise queue.Empty


class _SampleRejectionAckQueue:
    def __init__(self) -> None:
        self._messages = [
            {"kind": "sample_rejected", "reason": "left stale measured state"},
        ]

    def get_nowait(self):
        if self._messages:
            return self._messages.pop(0)
        raise queue.Empty


class _InputQueue:
    def __init__(self) -> None:
        self.messages = []

    def put(self, message, *, timeout: float) -> None:
        del timeout
        self.messages.append(message)


class _ReplyQueue:
    def __init__(self, reply) -> None:
        self.reply = dict(reply)

    def get(self, *, timeout: float):
        del timeout
        return dict(self.reply)


def test_process_worker_reports_exit_before_ready_without_waiting_for_timeout() -> None:
    worker = object.__new__(ProcessCollectionWorker)
    worker._started = False
    worker._process = _ExitedProcess()
    worker._reply_queue = _EmptyReplyQueue()
    worker.config = SimpleNamespace(startup_timeout_sec=10.0)
    close_calls: list[bool] = []
    worker.close = lambda *, force=False: close_calls.append(bool(force))

    with pytest.raises(
        RuntimeError,
        match=(
            r"collection worker exited before becoming ready "
            r"\(exitcode=23, last_phase=process_spawn\)"
        ),
    ):
        worker.start()

    assert close_calls == [True]


def test_process_worker_timeout_reports_last_startup_phase() -> None:
    worker = object.__new__(ProcessCollectionWorker)
    worker._started = False
    worker._process = _AliveProcess()
    worker._reply_queue = _ProgressThenEmptyReplyQueue()
    worker.config = SimpleNamespace(startup_timeout_sec=0.01)
    close_calls: list[bool] = []
    worker.close = lambda *, force=False: close_calls.append(bool(force))

    with pytest.raises(
        RuntimeError,
        match=(
            r"collection worker did not become ready within 0\.010s "
            r"\(last_phase=initialize_legacy_writer\)"
        ),
    ):
        worker.start()

    assert close_calls == [True]


def test_process_worker_retains_and_prints_sample_traceback(capsys) -> None:
    worker = object.__new__(ProcessCollectionWorker)
    worker._ack_queue = _WorkerErrorAckQueue()
    worker._worker_error = None
    worker._worker_traceback = None

    worker._drain_acks()

    assert worker._worker_error == "worker.plan_jobs: DatasetLayoutError: bad sample"
    assert worker._worker_traceback == (
        "Traceback (most recent call last):\n  worker.plan_jobs"
    )
    assert "[CollectionWorker] sample processing failed" in capsys.readouterr().err


def test_worker_main_mirrors_post_start_fatal_to_health_ack_queue() -> None:
    source = inspect.getsource(process_writer._worker_main)

    assert '"kind": "ready" if writer is None else "fatal"' in source
    assert '"kind": "worker_error"' in source
    assert '"error": f"worker.fatal: {error}"' in source


def test_process_worker_sample_rejection_is_nonfatal() -> None:
    worker = object.__new__(ProcessCollectionWorker)
    worker._ack_queue = _SampleRejectionAckQueue()
    worker._worker_error = None
    worker._worker_traceback = None
    worker._sample_rejection_count = 0
    worker._pending_sample_rejections = deque()
    worker._started = False

    assert worker.consume_sample_rejection() == "left stale measured state"
    assert worker._worker_error is None
    assert worker._sample_rejection_count == 1
    assert worker.consume_sample_rejection() is None


def test_process_worker_request_prints_child_traceback(capsys) -> None:
    worker = object.__new__(ProcessCollectionWorker)
    worker._raise_if_failed = lambda: None
    worker._request_lock = threading.Lock()
    worker._input_queue = _InputQueue()
    worker.config = SimpleNamespace(request_timeout_sec=1.0)
    request_id = "request-id"
    worker._reply_queue = _ReplyQueue(
        {
            "request_id": request_id,
            "ok": False,
            "error": "DatasetLayoutError: bad episode metadata",
            "traceback": "Traceback (most recent call last):\n  writer.start_episode",
        }
    )

    import teleop_runtime.collection.process_writer as process_writer

    original_uuid4 = process_writer.uuid.uuid4
    process_writer.uuid.uuid4 = lambda: SimpleNamespace(hex=request_id)
    try:
        with pytest.raises(
            RuntimeError,
            match="collection worker start_episode failed: DatasetLayoutError",
        ):
            worker._request("start_episode", metadata={"formal_dataset": True})
    finally:
        process_writer.uuid.uuid4 = original_uuid4

    assert "request 'start_episode' failed" in capsys.readouterr().err


def test_process_worker_request_rejects_unpicklable_payload_synchronously() -> None:
    worker = object.__new__(ProcessCollectionWorker)
    worker._raise_if_failed = lambda: None
    worker._request_lock = threading.Lock()
    worker._input_queue = _InputQueue()
    worker._reply_queue = _ReplyQueue({})
    worker.config = SimpleNamespace(request_timeout_sec=1.0)

    with pytest.raises(
        TypeError,
        match=r"collection worker request 'start_episode' is not serializable:.*lock",
    ):
        worker._request("start_episode", metadata={"lock": threading.Lock()})

    assert worker._input_queue.messages == []
