from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.client.rtc_transport import (
    RTC_TRANSPORT_PHASE_CANCELLED,
    RTC_TRANSPORT_PHASE_CLOSE,
    RTC_TRANSPORT_PHASE_COMPLETED,
    RTC_TRANSPORT_PHASE_CONNECT,
    RTC_TRANSPORT_PHASE_FAILED,
    RTC_TRANSPORT_PHASE_INFERENCE,
    RTC_TRANSPORT_PHASE_METADATA,
    RTC_TRANSPORT_PHASE_TIMED_OUT,
    RtcTransportDeadlineExceeded,
    RtcTransportDeadlineSpec,
    RtcTransportPhaseReceipt,
)


OPENPI_TRANSPORT_LIFECYCLE_HASH_DOMAIN_V1 = (
    "dexteleop.openpi_transport_lifecycle.v1"
)
OPENPI_TRANSPORT_CLOSE_HASH_DOMAIN_V1 = "dexteleop.openpi_transport_close.v1"
OPENPI_TRANSPORT_PHASE_RECEIPT_HISTORY_LIMIT = 64


_T = TypeVar("_T")


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(value: Any) -> bytes:
    import msgpack

    return msgpack.packb(value, default=_pack_array)


def unpackb(value: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(value, object_hook=_unpack_array)


@dataclass(frozen=True)
class OpenPIWebsocketTransportConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    retry_interval_s: float = 2.0
    deadline_spec: RtcTransportDeadlineSpec = field(
        default_factory=RtcTransportDeadlineSpec
    )


class OpenPITransportSessionCancelledError(RuntimeError):
    """Raised when an RTC worker session retires an OpenPI connection."""


@dataclass(frozen=True)
class OpenPITransportLifecycleReceipt:
    """Session-local begin/cancel fact; never an inference-worker drain proof."""

    sequence: int
    operation: str
    accepted: bool
    status: str
    session_generation: int
    connection_generation: int
    connection_detached: bool
    cancel_latched: bool
    close_pending_count: int

    def __post_init__(self) -> None:
        if self.operation not in ("begin_session", "request_cancel"):
            raise ValueError(f"Unsupported transport operation: {self.operation!r}.")
        if self.status not in (
            "session_armed",
            "cancellation_latched",
            "cancellation_already_latched",
            "close_pending",
            "close_failed",
        ):
            raise ValueError(f"Unsupported transport lifecycle status: {self.status!r}.")
        if self.accepted != (self.status not in ("close_pending", "close_failed")):
            raise ValueError("Transport lifecycle acceptance/status mismatch.")
        for name in (
            "sequence",
            "session_generation",
            "connection_generation",
            "close_pending_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "sequence": self.sequence,
                "operation": self.operation,
                "accepted": self.accepted,
                "status": self.status,
                "session_generation": self.session_generation,
                "connection_generation": self.connection_generation,
                "connection_detached": self.connection_detached,
                "cancel_latched": self.cancel_latched,
                "close_pending_count": self.close_pending_count,
            },
            domain=OPENPI_TRANSPORT_LIFECYCLE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class OpenPITransportCloseReceipt:
    """Completion fact for one detached connection close attempt."""

    sequence: int
    session_generation: int
    connection_generation: int
    completed: bool
    status: str
    failure_type: str | None = None
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        for name in ("sequence", "session_generation", "connection_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.status not in ("closed", "close_failed"):
            raise ValueError(f"Unsupported transport close status: {self.status!r}.")
        if self.completed != (self.status == "closed"):
            raise ValueError("Transport close completion/status mismatch.")
        if self.status == "close_failed":
            if not self.failure_type or not self.failure_detail:
                raise ValueError("Failed transport close requires failure details.")
        elif self.failure_type is not None or self.failure_detail is not None:
            raise ValueError("Successful transport close cannot carry a failure.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "sequence": self.sequence,
                "session_generation": self.session_generation,
                "connection_generation": self.connection_generation,
                "completed": self.completed,
                "status": self.status,
                "failure_type": self.failure_type,
                "failure_detail": self.failure_detail,
            },
            domain=OPENPI_TRANSPORT_CLOSE_HASH_DOMAIN_V1,
        )


class OpenPIWebsocketTransport:
    """OpenPI msgpack-numpy websocket transport."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        api_key: str | None = None,
        retry_interval_s: float = 2.0,
        deadline_spec: RtcTransportDeadlineSpec | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        resolved_deadline_spec = (
            RtcTransportDeadlineSpec()
            if deadline_spec is None
            else deadline_spec
        )
        if not isinstance(resolved_deadline_spec, RtcTransportDeadlineSpec):
            raise TypeError("deadline_spec must be an RtcTransportDeadlineSpec.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.config = OpenPIWebsocketTransportConfig(
            host=str(host),
            port=int(port),
            api_key=api_key,
            retry_interval_s=float(retry_interval_s),
            deadline_spec=resolved_deadline_spec,
        )
        self._clock = clock
        self._lock = threading.RLock()
        self._cancel_requested = threading.Event()
        self._ws: Any = None
        self._metadata: dict[str, Any] | None = None
        self._connection_generation = 0
        self._session_generation = 0
        self._next_lifecycle_sequence = 0
        self._last_lifecycle_receipt: OpenPITransportLifecycleReceipt | None = None
        self._next_close_sequence = 0
        self._close_pending_count = 0
        self._last_close_receipt: OpenPITransportCloseReceipt | None = None
        self._close_failure_latched: OpenPITransportCloseReceipt | None = None
        self._next_phase_sequence = 0
        self._phase_receipts: deque[RtcTransportPhaseReceipt] = deque(
            maxlen=OPENPI_TRANSPORT_PHASE_RECEIPT_HISTORY_LIMIT
        )

    @property
    def uri(self) -> str:
        host = self.config.host
        uri = host if host.startswith("ws") else f"ws://{host}"
        return uri if ":" in uri.rsplit("/", 1)[-1] else f"{uri}:{self.config.port}"

    @property
    def metadata(self) -> dict[str, Any]:
        self.connect()
        with self._lock:
            return dict(self._metadata or {})

    @property
    def connection_generation(self) -> int:
        with self._lock:
            return self._connection_generation

    @property
    def session_cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def last_lifecycle_receipt(self) -> OpenPITransportLifecycleReceipt | None:
        with self._lock:
            return self._last_lifecycle_receipt

    @property
    def last_close_receipt(self) -> OpenPITransportCloseReceipt | None:
        with self._lock:
            return self._last_close_receipt

    @property
    def last_phase_receipt(self) -> RtcTransportPhaseReceipt | None:
        with self._lock:
            return self._phase_receipts[-1] if self._phase_receipts else None

    def phase_receipts_snapshot(self) -> tuple[RtcTransportPhaseReceipt, ...]:
        """Return bounded transport evidence without consuming it."""

        with self._lock:
            return tuple(self._phase_receipts)

    def begin_session(self) -> bool:
        """Re-arm the transport after the previous worker is proven drained.

        The caller owns the drain proof.  This method only clears the local
        cancellation latch; the next ``connect`` performs a fresh handshake
        when the prior connection was detached.
        """

        return self.begin_session_with_receipt().accepted

    def begin_session_with_receipt(self) -> OpenPITransportLifecycleReceipt:
        with self._lock:
            if self._close_pending_count:
                receipt = OpenPITransportLifecycleReceipt(
                    sequence=self._next_lifecycle_sequence,
                    operation="begin_session",
                    accepted=False,
                    status="close_pending",
                    session_generation=self._session_generation,
                    connection_generation=self._connection_generation,
                    connection_detached=False,
                    cancel_latched=self._cancel_requested.is_set(),
                    close_pending_count=self._close_pending_count,
                )
                self._next_lifecycle_sequence += 1
                self._last_lifecycle_receipt = receipt
                return receipt
            if self._close_failure_latched is not None:
                receipt = OpenPITransportLifecycleReceipt(
                    sequence=self._next_lifecycle_sequence,
                    operation="begin_session",
                    accepted=False,
                    status="close_failed",
                    session_generation=self._session_generation,
                    connection_generation=self._connection_generation,
                    connection_detached=False,
                    cancel_latched=True,
                    close_pending_count=0,
                )
                self._next_lifecycle_sequence += 1
                self._last_lifecycle_receipt = receipt
                return receipt
            self._cancel_requested.clear()
            self._session_generation += 1
            receipt = OpenPITransportLifecycleReceipt(
                sequence=self._next_lifecycle_sequence,
                operation="begin_session",
                accepted=True,
                status="session_armed",
                session_generation=self._session_generation,
                connection_generation=self._connection_generation,
                connection_detached=False,
                cancel_latched=False,
                close_pending_count=0,
            )
            self._next_lifecycle_sequence += 1
            self._last_lifecycle_receipt = receipt
            return receipt

    def connect(self) -> None:
        with self._lock:
            if self._ws is not None:
                return
        self._raise_if_cancelled()
        try:
            import websockets.sync.client
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI websocket transport requires the 'websockets' package."
            ) from exc

        headers = (
            {"Authorization": f"Api-Key {self.config.api_key}"}
            if self.config.api_key
            else None
        )

        def open_connection(deadline: float | None) -> Any:
            while True:
                self._raise_if_cancelled()
                connect_kwargs: dict[str, Any] = {
                    "compression": None,
                    "max_size": None,
                    "additional_headers": headers,
                }
                remaining = self._remaining_or_timeout(deadline)
                if remaining is not None:
                    connect_kwargs["open_timeout"] = remaining
                if self.config.deadline_spec.close_timeout_sec is not None:
                    connect_kwargs["close_timeout"] = (
                        self.config.deadline_spec.close_timeout_sec
                    )
                try:
                    connection = websockets.sync.client.connect(
                        self.uri,
                        **connect_kwargs,
                    )
                except ConnectionRefusedError:
                    remaining = self._remaining_or_timeout(deadline)
                    retry_interval = self.config.retry_interval_s
                    if remaining is not None:
                        retry_interval = min(retry_interval, remaining)
                    if self._cancel_requested.wait(retry_interval):
                        self._raise_if_cancelled()
                    continue
                try:
                    self._remaining_or_timeout(deadline)
                except TimeoutError:
                    self._schedule_close(connection)
                    raise
                with self._lock:
                    if self._cancel_requested.is_set():
                        self._schedule_close(connection)
                        self._raise_if_cancelled()
                    self._ws = connection
                    self._metadata = None
                    self._connection_generation += 1
                return connection

        try:
            connection = self._run_phase(
                RTC_TRANSPORT_PHASE_CONNECT,
                self.config.deadline_spec.connect_timeout_sec,
                open_connection,
            )
        except RtcTransportDeadlineExceeded:
            self._detach_current_connection()
            raise

        def receive_metadata(deadline: float | None) -> dict[str, Any]:
            metadata = self._recv(connection, deadline)
            if isinstance(metadata, str):
                raise RuntimeError(
                    f"OpenPI server returned string metadata: {metadata}"
                )
            return dict(unpackb(metadata))

        try:
            decoded_metadata = self._run_phase(
                RTC_TRANSPORT_PHASE_METADATA,
                self.config.deadline_spec.metadata_timeout_sec,
                receive_metadata,
            )
            with self._lock:
                if self._cancel_requested.is_set() or self._ws is not connection:
                    self._raise_if_cancelled()
                    raise RuntimeError(
                        "OpenPI connection changed during metadata handshake."
                    )
                self._metadata = decoded_metadata
        except Exception:
            self._detach_connection(connection)
            raise

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.connect()
        with self._lock:
            self._raise_if_cancelled()
            connection = self._ws
        if connection is None:
            raise RuntimeError("OpenPI websocket connection is unavailable.")

        def exchange(deadline: float | None) -> Any:
            try:
                connection.send(packb(observation))
                response = self._recv(connection, deadline)
            except Exception as error:
                if self._cancel_requested.is_set():
                    raise OpenPITransportSessionCancelledError(
                        "OpenPI transport session was cancelled."
                    ) from error
                raise
            self._raise_if_cancelled()
            if isinstance(response, str):
                raise RuntimeError(f"OpenPI inference server error:\n{response}")
            result = unpackb(response)
            if not isinstance(result, dict):
                raise RuntimeError(
                    "OpenPI inference response must be a dict, got "
                    f"{type(result).__name__}."
                )
            return result

        try:
            response = self._run_phase(
                RTC_TRANSPORT_PHASE_INFERENCE,
                self.config.deadline_spec.inference_timeout_sec,
                exchange,
            )
        except RtcTransportDeadlineExceeded:
            # A response may arrive after recv() times out.  Retaining that
            # socket could pair the late response with the next request.
            self._detach_connection(connection)
            raise
        return response

    def request_cancel(self) -> bool:
        """Latch cancellation and asynchronously close the current socket.

        This method is intentionally quick and idempotent so it can be called
        by ``BoundedPolicyInferenceWorker.stop``.  Its return value acknowledges
        only the cancellation request; worker thread exit remains the drain
        proof.
        """

        return self.request_cancel_with_receipt().accepted

    def request_cancel_with_receipt(self) -> OpenPITransportLifecycleReceipt:
        with self._lock:
            already_latched = self._cancel_requested.is_set()
            self._cancel_requested.set()
            connection = self._ws
            self._ws = None
            self._metadata = None
            if connection is not None:
                self._schedule_close(connection)
            receipt = OpenPITransportLifecycleReceipt(
                sequence=self._next_lifecycle_sequence,
                operation="request_cancel",
                accepted=True,
                status=(
                    "cancellation_already_latched"
                    if already_latched
                    else "cancellation_latched"
                ),
                session_generation=self._session_generation,
                connection_generation=self._connection_generation,
                connection_detached=connection is not None,
                cancel_latched=True,
                close_pending_count=self._close_pending_count,
            )
            self._next_lifecycle_sequence += 1
            self._last_lifecycle_receipt = receipt
        return receipt

    def close(self) -> None:
        self.request_cancel()

    def reset(self) -> None:
        return None

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise OpenPITransportSessionCancelledError(
                "OpenPI transport session was cancelled."
            )

    def _remaining_or_timeout(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = float(deadline) - float(self._clock())
        if remaining <= 0.0:
            raise TimeoutError("RTC transport phase deadline elapsed.")
        return remaining

    def _recv(self, connection: Any, deadline: float | None) -> Any:
        remaining = self._remaining_or_timeout(deadline)
        if remaining is None:
            return connection.recv()
        return connection.recv(timeout=remaining)

    def _run_phase(
        self,
        phase: str,
        timeout_sec: float | None,
        operation: Callable[[float | None], _T],
    ) -> _T:
        started_at = float(self._clock())
        deadline = None if timeout_sec is None else started_at + timeout_sec
        try:
            result = operation(deadline)
            completed_at = float(self._clock())
            if deadline is not None and completed_at > deadline:
                raise TimeoutError("RTC transport phase completed after its deadline.")
        except Exception as error:
            completed_at = float(self._clock())
            if isinstance(error, OpenPITransportSessionCancelledError):
                status = RTC_TRANSPORT_PHASE_CANCELLED
            elif isinstance(error, TimeoutError) and timeout_sec is not None:
                status = RTC_TRANSPORT_PHASE_TIMED_OUT
            else:
                status = RTC_TRANSPORT_PHASE_FAILED
            receipt = self._record_phase_receipt(
                phase=phase,
                status=status,
                started_at=started_at,
                completed_at=max(started_at, completed_at),
                timeout_sec=timeout_sec,
                failure=error,
            )
            if status == RTC_TRANSPORT_PHASE_TIMED_OUT:
                raise RtcTransportDeadlineExceeded(receipt) from error
            raise
        self._record_phase_receipt(
            phase=phase,
            status=RTC_TRANSPORT_PHASE_COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            timeout_sec=timeout_sec,
            failure=None,
        )
        return result

    def _record_phase_receipt(
        self,
        *,
        phase: str,
        status: str,
        started_at: float,
        completed_at: float,
        timeout_sec: float | None,
        failure: Exception | None,
    ) -> RtcTransportPhaseReceipt:
        with self._lock:
            receipt = RtcTransportPhaseReceipt(
                sequence=self._next_phase_sequence,
                phase=phase,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                timeout_sec=timeout_sec,
                session_generation=self._session_generation,
                connection_generation=self._connection_generation,
                failure_type=None if failure is None else type(failure).__name__,
                failure_detail=(
                    None if failure is None else (str(failure) or repr(failure))
                ),
            )
            self._next_phase_sequence += 1
            self._phase_receipts.append(receipt)
            return receipt

    def _detach_current_connection(self) -> None:
        with self._lock:
            connection = self._ws
        if connection is not None:
            self._detach_connection(connection)

    def _detach_connection(self, connection: Any) -> None:
        with self._lock:
            if self._ws is not connection:
                return
            self._ws = None
            self._metadata = None
            session_generation = self._session_generation
            connection_generation = self._connection_generation
        self._schedule_close(
            connection,
            session_generation=session_generation,
            connection_generation=connection_generation,
        )

    def _schedule_close(
        self,
        connection: Any,
        *,
        session_generation: int | None = None,
        connection_generation: int | None = None,
    ) -> None:
        with self._lock:
            close_sequence = self._next_close_sequence
            self._next_close_sequence += 1
            self._close_pending_count += 1
            bound_session_generation = (
                self._session_generation
                if session_generation is None
                else session_generation
            )
            bound_connection_generation = (
                self._connection_generation
                if connection_generation is None
                else connection_generation
            )
        thread = threading.Thread(
            target=self._close_connection,
            args=(
                connection,
                close_sequence,
                bound_session_generation,
                bound_connection_generation,
            ),
            name="dexteleop-openpi-transport-close",
            daemon=True,
        )
        try:
            thread.start()
        except Exception as error:  # noqa: BLE001 - recorded as typed evidence.
            self._record_close_completion(
                sequence=close_sequence,
                session_generation=bound_session_generation,
                connection_generation=bound_connection_generation,
                failure=error,
            )

    def _close_connection(
        self,
        connection: Any,
        sequence: int,
        session_generation: int,
        connection_generation: int,
    ) -> None:
        failure: Exception | None = None
        try:
            self._run_phase(
                RTC_TRANSPORT_PHASE_CLOSE,
                self.config.deadline_spec.close_timeout_sec,
                lambda _deadline: self._close_connection_now(connection),
            )
        except Exception as error:  # noqa: BLE001 - typed close failure.
            failure = error
        self._record_close_completion(
            sequence=sequence,
            session_generation=session_generation,
            connection_generation=connection_generation,
            failure=failure,
        )

    @staticmethod
    def _close_connection_now(connection: Any) -> None:
        connection_socket = getattr(connection, "socket", None)
        if connection_socket is not None:
            try:
                connection_socket.shutdown(socket.SHUT_RDWR)
            except (OSError, RuntimeError):
                pass
            try:
                connection_socket.close()
            except (OSError, RuntimeError):
                pass
        close = getattr(connection, "close", None)
        if callable(close):
            close()

    def _record_close_completion(
        self,
        *,
        sequence: int,
        session_generation: int,
        connection_generation: int,
        failure: Exception | None,
    ) -> None:
        receipt = OpenPITransportCloseReceipt(
            sequence=sequence,
            session_generation=session_generation,
            connection_generation=connection_generation,
            completed=failure is None,
            status="closed" if failure is None else "close_failed",
            failure_type=None if failure is None else type(failure).__name__,
            failure_detail=(
                None if failure is None else (str(failure) or repr(failure))
            ),
        )
        with self._lock:
            self._close_pending_count -= 1
            self._last_close_receipt = receipt
            if not receipt.completed:
                self._close_failure_latched = receipt


__all__ = [
    "OPENPI_TRANSPORT_PHASE_RECEIPT_HISTORY_LIMIT",
    "OPENPI_TRANSPORT_CLOSE_HASH_DOMAIN_V1",
    "OPENPI_TRANSPORT_LIFECYCLE_HASH_DOMAIN_V1",
    "OpenPITransportCloseReceipt",
    "OpenPITransportLifecycleReceipt",
    "OpenPITransportSessionCancelledError",
    "OpenPIWebsocketTransport",
    "OpenPIWebsocketTransportConfig",
    "RtcTransportDeadlineExceeded",
    "RtcTransportDeadlineSpec",
    "RtcTransportPhaseReceipt",
    "packb",
    "unpackb",
]
