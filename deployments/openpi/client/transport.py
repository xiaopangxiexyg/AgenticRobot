from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


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


class OpenPIWebsocketTransport:
    """OpenPI msgpack-numpy websocket transport."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        api_key: str | None = None,
        retry_interval_s: float = 2.0,
    ) -> None:
        self.config = OpenPIWebsocketTransportConfig(
            host=str(host),
            port=int(port),
            api_key=api_key,
            retry_interval_s=float(retry_interval_s),
        )
        self._ws: Any = None
        self._metadata: dict[str, Any] | None = None

    @property
    def uri(self) -> str:
        host = self.config.host
        uri = host if host.startswith("ws") else f"ws://{host}"
        return uri if ":" in uri.rsplit("/", 1)[-1] else f"{uri}:{self.config.port}"

    @property
    def metadata(self) -> dict[str, Any]:
        self.connect()
        return dict(self._metadata or {})

    def connect(self) -> None:
        if self._ws is not None:
            return
        try:
            import websockets.sync.client
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI websocket transport requires the 'websockets' package."
            ) from exc

        headers = {"Authorization": f"Api-Key {self.config.api_key}"} if self.config.api_key else None
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = self._ws.recv()
                if isinstance(metadata, str):
                    raise RuntimeError(f"OpenPI server returned string metadata: {metadata}")
                self._metadata = dict(unpackb(metadata))
                return
            except ConnectionRefusedError:
                time.sleep(self.config.retry_interval_s)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.connect()
        assert self._ws is not None
        self._ws.send(packb(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference server error:\n{response}")
        result = unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"OpenPI inference response must be a dict, got {type(result).__name__}.")
        return result

    def reset(self) -> None:
        return None

