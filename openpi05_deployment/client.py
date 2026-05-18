from __future__ import annotations

from typing import Any

from openpi_client import websocket_client_policy

from openpi05_deployment.metadata import fetch_server_metadata, resolve_endpoint


class OpenPIClient:
    """HTTP-preflighted websocket client for local OpenPI deployment."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str | None = None,
        wait_timeout_sec: float = 600.0,
        metadata_timeout_sec: float = 3.0,
        metadata_poll_interval_sec: float = 1.0,
    ) -> None:
        self.endpoint = resolve_endpoint(host, port)
        self.api_key = api_key
        self.wait_timeout_sec = float(wait_timeout_sec)
        self.metadata_timeout_sec = float(metadata_timeout_sec)
        self.metadata_poll_interval_sec = float(metadata_poll_interval_sec)
        self._policy: websocket_client_policy.WebsocketClientPolicy | None = None
        self._metadata: dict[str, Any] | None = None
        self.connect()

    def connect(self) -> dict[str, Any]:
        self.close()
        fetch_server_metadata(
            host=self.endpoint.host,
            port=self.endpoint.port,
            timeout_sec=self.wait_timeout_sec,
            poll_interval_sec=self.metadata_poll_interval_sec,
            request_timeout_sec=self.metadata_timeout_sec,
        )
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            self.endpoint.host,
            self.endpoint.port,
            api_key=self.api_key,
            wait_timeout_sec=min(self.wait_timeout_sec, 30.0),
        )
        self._metadata = dict(self._policy.get_server_metadata())
        return dict(self._metadata)

    def ensure_connected(self) -> dict[str, Any]:
        if self._policy is None or self._metadata is None:
            return self.connect()
        return dict(self._metadata)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.ensure_connected()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.ensure_connected()
        assert self._policy is not None
        return dict(self._policy.infer(obs))

    def close(self) -> None:
        if self._policy is not None:
            close = getattr(self._policy, "close", None)
            if callable(close):
                close()
        self._policy = None
        self._metadata = None
