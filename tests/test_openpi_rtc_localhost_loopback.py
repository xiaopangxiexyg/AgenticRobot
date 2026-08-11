"""No-model localhost websocket checks for the OpenPI RTC transport.

These tests exchange only metadata, an observation dictionary, and a synthetic
final chunk.  They don't load a policy, initialize hardware, or dispatch an
action.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator

import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.openpi.client.transport import (
    OpenPIWebsocketTransport,
    RtcTransportDeadlineExceeded,
    RtcTransportDeadlineSpec,
    packb,
    unpackb,
)


@contextmanager
def _localhost_server(
    handler: Callable[[Any], Awaitable[None]],
) -> Iterator[int]:
    server_module = pytest.importorskip("websockets.asyncio.server")
    started = threading.Event()
    stop = threading.Event()
    port_box: list[int] = []
    failures: list[BaseException] = []

    async def run_server() -> None:
        try:
            async with server_module.serve(
                handler,
                "127.0.0.1",
                0,
                compression=None,
            ) as server:
                if not server.sockets:
                    started.set()
                    return
                port_box.append(int(server.sockets[0].getsockname()[1]))
                started.set()
                while not stop.is_set():
                    await asyncio.sleep(0.001)
        except BaseException as error:  # noqa: BLE001 - test-thread evidence.
            failures.append(error)
            started.set()

    def thread_main() -> None:
        asyncio.run(run_server())

    thread = threading.Thread(target=thread_main, daemon=True)
    thread.start()
    assert started.wait(timeout=2.0)
    if failures:
        if isinstance(failures[0], (PermissionError, OSError)):
            pytest.skip("localhost socket binding is unavailable in this sandbox")
        raise failures[0]
    if not port_box:
        pytest.skip("localhost socket binding is unavailable in this sandbox")
    try:
        yield port_box[0]
    finally:
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert not failures


def _wait_for_close(transport: OpenPIWebsocketTransport) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        receipt = transport.last_close_receipt
        if receipt is not None:
            assert receipt.completed
            return
        time.sleep(0.001)
    raise AssertionError("localhost transport close did not complete")


@pytest.mark.optional
def test_localhost_final_chunk_loopback_reconnects_with_identical_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    metadata = {
        "protocol_version": "openpi_local_v1",
        "model": {"action_horizon": 2},
        "runtime_build": "synthetic-no-model",
    }
    received: list[dict[str, Any]] = []

    async def handler(connection: Any) -> None:
        await connection.send(packb(metadata))
        request = await asyncio.wait_for(connection.recv(), timeout=1.0)
        assert isinstance(request, bytes)
        decoded = unpackb(request)
        assert isinstance(decoded, dict)
        received.append(decoded)
        await connection.send(
            packb({"robot_actions": [[1.0, 2.0], [3.0, 4.0]]})
        )

    with _localhost_server(handler) as port:
        transport = OpenPIWebsocketTransport(
            port=port,
            deadline_spec=RtcTransportDeadlineSpec(
                connect_timeout_sec=1.0,
                metadata_timeout_sec=1.0,
                inference_timeout_sec=1.0,
                close_timeout_sec=0.5,
            ),
        )
        first_metadata_hash = canonical_content_hash(
            transport.metadata,
            domain="dexteleop.test.openpi_localhost_metadata.v1",
        )
        first = transport.infer({"observation_id": "first", "state": [0.0]})
        assert first["robot_actions"] == [[1.0, 2.0], [3.0, 4.0]]
        assert transport.connection_generation == 1

        transport.request_cancel()
        _wait_for_close(transport)
        assert transport.begin_session()

        second = transport.infer({"observation_id": "second", "state": [1.0]})
        assert second == first
        assert transport.connection_generation == 2
        assert canonical_content_hash(
            transport.metadata,
            domain="dexteleop.test.openpi_localhost_metadata.v1",
        ) == first_metadata_hash

    assert [request["observation_id"] for request in received] == [
        "first",
        "second",
    ]
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
    assert all(receipt.status == "completed" for receipt in receipts)


@pytest.mark.optional
def test_localhost_late_final_chunk_times_out_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    metadata = {
        "model": {"action_horizon": 1},
        "runtime_build": "synthetic-slow-no-model",
    }
    request_received = threading.Event()

    async def handler(connection: Any) -> None:
        await connection.send(packb(metadata))
        await asyncio.wait_for(connection.recv(), timeout=1.0)
        request_received.set()
        await asyncio.sleep(0.1)
        try:
            await connection.send(packb({"robot_actions": [[99.0]]}))
        except Exception:
            # The client intentionally retires this connection on timeout.
            pass

    with _localhost_server(handler) as port:
        transport = OpenPIWebsocketTransport(
            port=port,
            deadline_spec=RtcTransportDeadlineSpec(
                connect_timeout_sec=1.0,
                metadata_timeout_sec=1.0,
                inference_timeout_sec=0.02,
                close_timeout_sec=0.2,
            ),
        )
        with pytest.raises(RtcTransportDeadlineExceeded) as raised:
            transport.infer({"observation_id": "late"})
        assert request_received.wait(timeout=1.0)
        assert raised.value.receipt.phase == "inference"
        assert raised.value.receipt.status == "timed_out"
        assert transport._ws is None
        _wait_for_close(transport)
