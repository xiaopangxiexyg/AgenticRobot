from __future__ import annotations

import pickle
import sys
import threading
import time
import types
from typing import Any

import pytest

from deployments.openpi.client.transport import (
    OpenPIWebsocketTransport,
    RtcTransportDeadlineExceeded,
    RtcTransportDeadlineSpec,
    packb,
)


class _ScriptedConnection:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = threading.Event()
        self.close_count = 0
        self.recv_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self, *args, **kwargs) -> bytes:
        self.recv_calls.append((args, dict(kwargs)))
        return self.responses.pop(0)

    def close(self) -> None:
        self.close_count += 1
        self.closed.set()


class _BlockingCloseConnection(_ScriptedConnection):
    def __init__(self) -> None:
        super().__init__([])
        self.close_release = threading.Event()

    def close(self) -> None:
        self.close_count += 1
        self.closed.set()
        self.close_release.wait(timeout=2.0)


class _FailingCloseConnection(_ScriptedConnection):
    def __init__(self) -> None:
        super().__init__([])

    def close(self) -> None:
        self.close_count += 1
        self.closed.set()
        raise RuntimeError("synthetic close failure")


class _SlowCloseConnection(_ScriptedConnection):
    def close(self) -> None:
        self.close_count += 1
        self.closed.set()
        time.sleep(0.02)


class _InferenceTimeoutConnection(_ScriptedConnection):
    def recv(self, *args, **kwargs) -> bytes:
        self.recv_calls.append((args, dict(kwargs)))
        if self.responses:
            return self.responses.pop(0)
        raise TimeoutError("synthetic inference timeout")


class _BlockingRecvConnection(_ScriptedConnection):
    def __init__(self, metadata: bytes) -> None:
        super().__init__([metadata])
        self.inference_entered = threading.Event()

    def recv(self, *args, **kwargs) -> bytes:
        self.recv_calls.append((args, dict(kwargs)))
        if self.responses:
            return self.responses.pop(0)
        self.inference_entered.set()
        if not self.closed.wait(timeout=2.0):
            raise RuntimeError("synthetic blocking recv was not released")
        raise RuntimeError("synthetic socket closed")


def _install_fake_websockets(monkeypatch, connections: list[Any]):
    root = types.ModuleType("websockets")
    sync = types.ModuleType("websockets.sync")
    client = types.ModuleType("websockets.sync.client")

    connect_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def connect(*args, **kwargs):
        connect_calls.append((args, dict(kwargs)))
        return connections.pop(0)

    client.connect = connect
    sync.client = client
    root.sync = sync
    monkeypatch.setitem(sys.modules, "websockets", root)
    monkeypatch.setitem(sys.modules, "websockets.sync", sync)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", client)
    return connect_calls


def _install_refusing_websockets(monkeypatch):
    root = types.ModuleType("websockets")
    sync = types.ModuleType("websockets.sync")
    client = types.ModuleType("websockets.sync.client")
    connect_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def connect(*args, **kwargs):
        connect_calls.append((args, dict(kwargs)))
        raise ConnectionRefusedError("synthetic refusal")

    client.connect = connect
    sync.client = client
    root.sync = sync
    monkeypatch.setitem(sys.modules, "websockets", root)
    monkeypatch.setitem(sys.modules, "websockets.sync", sync)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", client)
    return connect_calls


def _install_fake_msgpack(monkeypatch) -> None:
    msgpack = types.ModuleType("msgpack")
    msgpack.packb = lambda value, default=None: pickle.dumps(value)
    msgpack.unpackb = lambda value, object_hook=None: pickle.loads(value)
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)


def _wait_for_close_receipt(transport: OpenPIWebsocketTransport):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        receipt = transport.last_close_receipt
        if receipt is not None:
            return receipt
        time.sleep(0.001)
    raise AssertionError("transport close receipt timed out")


def test_transport_cancel_is_fast_idempotent_and_reconnects_next_session(
    monkeypatch,
) -> None:
    _install_fake_msgpack(monkeypatch)
    metadata = {"model": {"action_horizon": 4}, "runtime_build": "unit"}
    first = _ScriptedConnection(
        [packb(metadata), packb({"robot_actions": [[1.0]]})]
    )
    second = _ScriptedConnection(
        [packb(metadata), packb({"robot_actions": [[2.0]]})]
    )
    _install_fake_websockets(monkeypatch, [first, second])
    transport = OpenPIWebsocketTransport(retry_interval_s=0.001)

    assert transport.infer({"step": 0})["robot_actions"] == [[1.0]]
    assert transport.connection_generation == 1

    started = time.perf_counter()
    cancel = transport.request_cancel_with_receipt()
    assert time.perf_counter() - started < 0.05
    assert cancel.accepted
    assert cancel.status == "cancellation_latched"
    assert cancel.session_generation == 0
    assert cancel.connection_generation == 1
    assert cancel.connection_detached
    assert cancel.close_pending_count == 1
    repeated = transport.request_cancel_with_receipt()
    assert repeated.status == "cancellation_already_latched"
    assert not repeated.connection_detached
    assert repeated.sequence == cancel.sequence + 1
    assert transport.session_cancel_requested
    assert first.closed.wait(timeout=1.0)
    assert first.close_count == 1
    close = _wait_for_close_receipt(transport)
    assert close.completed
    assert close.status == "closed"
    assert close.connection_generation == 1

    assert transport.begin_session()
    begin = transport.last_lifecycle_receipt
    assert begin is not None
    assert begin.operation == "begin_session"
    assert begin.status == "session_armed"
    assert begin.session_generation == 1
    assert begin.sequence == repeated.sequence + 1
    assert not transport.session_cancel_requested
    assert transport.infer({"step": 1})["robot_actions"] == [[2.0]]
    assert transport.connection_generation == 2


def test_transport_close_alias_only_latches_cancellation() -> None:
    connection = _ScriptedConnection([])
    transport = OpenPIWebsocketTransport()
    transport._ws = connection

    transport.close()

    assert transport.session_cancel_requested
    assert connection.closed.wait(timeout=1.0)
    assert connection.close_count == 1


def test_transport_rearm_fails_closed_until_detached_close_completes() -> None:
    connection = _BlockingCloseConnection()
    transport = OpenPIWebsocketTransport()
    transport._ws = connection

    cancel = transport.request_cancel_with_receipt()
    assert cancel.close_pending_count == 1
    assert connection.closed.wait(timeout=1.0)
    blocked = transport.begin_session_with_receipt()
    assert not blocked.accepted
    assert blocked.status == "close_pending"
    assert blocked.close_pending_count == 1

    connection.close_release.set()
    close = _wait_for_close_receipt(transport)
    assert close.completed
    armed = transport.begin_session_with_receipt()
    assert armed.accepted
    assert armed.status == "session_armed"
    assert armed.close_pending_count == 0


def test_transport_close_failure_latches_and_blocks_rearm() -> None:
    connection = _FailingCloseConnection()
    transport = OpenPIWebsocketTransport()
    transport._ws = connection

    transport.request_cancel_with_receipt()
    close = _wait_for_close_receipt(transport)
    assert not close.completed
    assert close.status == "close_failed"
    assert close.failure_type == "RuntimeError"
    blocked = transport.begin_session_with_receipt()
    assert not blocked.accepted
    assert blocked.status == "close_failed"


def test_explicit_close_deadline_becomes_phase_and_lifecycle_failure() -> None:
    connection = _SlowCloseConnection([])
    transport = OpenPIWebsocketTransport(
        deadline_spec=RtcTransportDeadlineSpec(close_timeout_sec=0.005)
    )
    transport._ws = connection

    transport.request_cancel()
    close = _wait_for_close_receipt(transport)

    assert not close.completed
    assert close.status == "close_failed"
    assert close.failure_type == "RtcTransportDeadlineExceeded"
    phase = transport.last_phase_receipt
    assert phase is not None
    assert phase.phase == "close"
    assert phase.status == "timed_out"
    assert not transport.begin_session()


def test_deadline_mapping_is_opt_in_and_legacy_call_shape_is_unchanged(
    monkeypatch,
) -> None:
    _install_fake_msgpack(monkeypatch)
    metadata = {"model": {"action_horizon": 4}}
    legacy = _ScriptedConnection(
        [packb(metadata), packb({"robot_actions": [[1.0]]})]
    )
    legacy_connect_calls = _install_fake_websockets(monkeypatch, [legacy])
    legacy_transport = OpenPIWebsocketTransport()

    assert legacy_transport.infer({"step": 0})["robot_actions"] == [[1.0]]
    assert "open_timeout" not in legacy_connect_calls[0][1]
    assert "close_timeout" not in legacy_connect_calls[0][1]
    assert legacy.recv_calls == [((), {}), ((), {})]

    bounded = _ScriptedConnection(
        [packb(metadata), packb({"robot_actions": [[2.0]]})]
    )
    bounded_connect_calls = _install_fake_websockets(monkeypatch, [bounded])
    deadline_spec = RtcTransportDeadlineSpec(
        connect_timeout_sec=0.5,
        metadata_timeout_sec=0.4,
        inference_timeout_sec=0.3,
        close_timeout_sec=0.2,
    )
    bounded_transport = OpenPIWebsocketTransport(deadline_spec=deadline_spec)

    assert bounded_transport.infer({"step": 1})["robot_actions"] == [[2.0]]
    connect_kwargs = bounded_connect_calls[0][1]
    assert 0.0 < connect_kwargs["open_timeout"] <= 0.5
    assert connect_kwargs["close_timeout"] == 0.2
    assert bounded.recv_calls[0][0] == ()
    assert 0.0 < bounded.recv_calls[0][1]["timeout"] <= 0.4
    assert bounded.recv_calls[1][0] == ()
    assert 0.0 < bounded.recv_calls[1][1]["timeout"] <= 0.3
    receipts = bounded_transport.phase_receipts_snapshot()
    assert [receipt.phase for receipt in receipts] == [
        "connect",
        "metadata",
        "inference",
    ]
    assert [receipt.status for receipt in receipts] == [
        "completed",
        "completed",
        "completed",
    ]
    assert [receipt.sequence for receipt in receipts] == [0, 1, 2]


def test_connect_retry_expires_with_typed_deadline_receipt(monkeypatch) -> None:
    connect_calls = _install_refusing_websockets(monkeypatch)
    transport = OpenPIWebsocketTransport(
        retry_interval_s=0.001,
        deadline_spec=RtcTransportDeadlineSpec(connect_timeout_sec=0.01),
    )

    started = time.monotonic()
    with pytest.raises(RtcTransportDeadlineExceeded) as raised:
        transport.connect()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert len(connect_calls) >= 1
    assert raised.value.receipt.phase == "connect"
    assert raised.value.receipt.status == "timed_out"
    assert transport.last_phase_receipt is raised.value.receipt
    assert transport.connection_generation == 0


def test_inference_timeout_detaches_late_socket_and_fresh_reconnect_succeeds(
    monkeypatch,
) -> None:
    _install_fake_msgpack(monkeypatch)
    metadata = {"model": {"action_horizon": 4}, "runtime_build": "stable"}
    timed_out = _InferenceTimeoutConnection([packb(metadata)])
    recovered = _ScriptedConnection(
        [packb(metadata), packb({"robot_actions": [[3.0]]})]
    )
    _install_fake_websockets(monkeypatch, [timed_out, recovered])
    transport = OpenPIWebsocketTransport(
        deadline_spec=RtcTransportDeadlineSpec(
            connect_timeout_sec=0.2,
            metadata_timeout_sec=0.2,
            inference_timeout_sec=0.01,
        )
    )

    with pytest.raises(RtcTransportDeadlineExceeded) as raised:
        transport.infer({"step": 0})

    assert raised.value.receipt.phase == "inference"
    assert raised.value.receipt.status == "timed_out"
    assert timed_out.closed.wait(timeout=1.0)
    close = _wait_for_close_receipt(transport)
    assert close.completed
    assert close.connection_generation == 1
    assert not transport.session_cancel_requested

    assert transport.infer({"step": 1})["robot_actions"] == [[3.0]]
    assert transport.connection_generation == 2
    receipts = transport.phase_receipts_snapshot()
    assert [receipt.phase for receipt in receipts] == [
        "connect",
        "metadata",
        "inference",
        "close",
        "connect",
        "metadata",
        "inference",
    ]
    assert receipts[2].status == "timed_out"
    assert all(receipt.status == "completed" for receipt in receipts[3:])


def test_metadata_timeout_never_exposes_partial_connection(monkeypatch) -> None:
    _install_fake_msgpack(monkeypatch)
    connection = _InferenceTimeoutConnection([])
    _install_fake_websockets(monkeypatch, [connection])
    transport = OpenPIWebsocketTransport(
        deadline_spec=RtcTransportDeadlineSpec(
            connect_timeout_sec=0.2,
            metadata_timeout_sec=0.01,
        )
    )

    with pytest.raises(RtcTransportDeadlineExceeded) as raised:
        _ = transport.metadata

    assert raised.value.receipt.phase == "metadata"
    assert connection.closed.wait(timeout=1.0)
    assert transport._ws is None
    assert transport._metadata is None


def test_native_timeout_without_opt_in_budget_keeps_legacy_exception(
    monkeypatch,
) -> None:
    _install_fake_msgpack(monkeypatch)
    metadata = {"model": {"action_horizon": 4}}
    connection = _InferenceTimeoutConnection([packb(metadata)])
    _install_fake_websockets(monkeypatch, [connection])
    transport = OpenPIWebsocketTransport()

    with pytest.raises(TimeoutError, match="synthetic inference timeout") as raised:
        transport.infer({"step": 0})

    assert not isinstance(raised.value, RtcTransportDeadlineExceeded)
    assert transport.last_phase_receipt is not None
    assert transport.last_phase_receipt.phase == "inference"
    assert transport.last_phase_receipt.status == "failed"
    assert transport.last_phase_receipt.timeout_sec is None


def test_cancel_wins_over_blocking_inference_and_receipt_is_not_timeout(
    monkeypatch,
) -> None:
    _install_fake_msgpack(monkeypatch)
    metadata = {"model": {"action_horizon": 4}}
    blocked = _BlockingRecvConnection(packb(metadata))
    recovered = _ScriptedConnection(
        [packb(metadata), packb({"robot_actions": [[4.0]]})]
    )
    _install_fake_websockets(monkeypatch, [blocked, recovered])
    transport = OpenPIWebsocketTransport(
        deadline_spec=RtcTransportDeadlineSpec(
            connect_timeout_sec=0.5,
            metadata_timeout_sec=0.5,
            inference_timeout_sec=1.0,
        )
    )
    failures: list[BaseException] = []

    thread = threading.Thread(
        target=lambda: _capture_failure(
            failures,
            lambda: transport.infer({"step": 0}),
        ),
        daemon=True,
    )
    thread.start()
    assert blocked.inference_entered.wait(timeout=1.0)

    cancel = transport.request_cancel_with_receipt()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert cancel.accepted
    assert len(failures) == 1
    assert type(failures[0]).__name__ == "OpenPITransportSessionCancelledError"
    _wait_for_close_receipt(transport)
    inference_receipts = [
        receipt
        for receipt in transport.phase_receipts_snapshot()
        if receipt.phase == "inference"
    ]
    assert inference_receipts[-1].status == "cancelled"

    assert transport.begin_session()
    assert transport.infer({"step": 1})["robot_actions"] == [[4.0]]
    assert transport.connection_generation == 2


def _capture_failure(
    failures: list[BaseException],
    operation,
) -> None:
    try:
        operation()
    except BaseException as error:  # noqa: BLE001 - test captures thread fact.
        failures.append(error)
