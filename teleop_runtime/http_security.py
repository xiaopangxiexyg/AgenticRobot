from __future__ import annotations

import ipaddress
import secrets
from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import urlsplit


CONTROL_TOKEN_HEADER = "X-DexTeleop-Control-Token"
DEFAULT_MAX_CONTROL_BODY_BYTES = 64 * 1024


class HttpRequestSecurityError(ValueError):
    """An HTTP request that must be rejected before application dispatch."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(str(message))
        self.status = int(status)


def generate_session_token() -> str:
    """Return a per-process control token with 256 bits of entropy."""

    return secrets.token_urlsafe(32)


def _normalized_hostname(value: str) -> str:
    return str(value or "").strip().lower().rstrip(".").strip("[]")


def is_loopback_host(host: str) -> bool:
    normalized = _normalized_hostname(host)
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_bind_host(
    host: str,
    *,
    unsafe_allow_remote: bool = False,
    allowed_hosts: Sequence[str] = (),
) -> None:
    """Reject network-visible HTTP binds unless they were explicitly enabled."""

    normalized = _normalized_hostname(host)
    if is_loopback_host(normalized):
        return
    if not unsafe_allow_remote:
        raise ValueError(
            f"Refusing non-loopback web bind {host!r}; pass the explicit unsafe remote-web opt-in."
        )
    if normalized in {"", "0.0.0.0", "::"} and not tuple(allowed_hosts):
        raise ValueError(
            "A wildcard remote-web bind also requires at least one explicit allowed Host value."
        )


def _authority_parts(authority: str, *, default_port: int) -> tuple[str, int]:
    raw = str(authority or "").strip()
    if not raw or any(char in raw for char in ("/", "\\", "@", "#", "?")):
        raise ValueError("invalid HTTP authority")
    try:
        parsed = urlsplit(f"//{raw}")
        hostname = _normalized_hostname(parsed.hostname or "")
        port = int(parsed.port) if parsed.port is not None else int(default_port)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid HTTP authority") from exc
    if not hostname or not 0 < port <= 65535:
        raise ValueError("invalid HTTP authority")
    return hostname, port


def _allowed_hostname(value: str) -> str:
    raw = str(value or "").strip()
    if "://" in raw:
        parsed = urlsplit(raw)
        return _normalized_hostname(parsed.hostname or "")
    try:
        return _authority_parts(raw, default_port=80)[0]
    except ValueError:
        return _normalized_hostname(raw)


@dataclass(frozen=True)
class WebControlSecurity:
    """Host/origin/token and request-size policy for a local control server."""

    token: str
    bind_host: str = "127.0.0.1"
    allowed_hosts: tuple[str, ...] = ()
    max_control_body_bytes: int = DEFAULT_MAX_CONTROL_BODY_BYTES

    def __post_init__(self) -> None:
        if not str(self.token):
            raise ValueError("web control token must not be empty")
        if int(self.max_control_body_bytes) <= 0:
            raise ValueError("max_control_body_bytes must be positive")

    def _allowed_hostnames(self, server_host: str) -> set[str]:
        allowed = {
            _allowed_hostname(value)
            for value in self.allowed_hosts
            if _allowed_hostname(value)
        }
        bind_host = _normalized_hostname(self.bind_host)
        actual_host = _normalized_hostname(server_host)
        if bind_host not in {"", "0.0.0.0", "::"}:
            allowed.add(bind_host)
        if actual_host not in {"", "0.0.0.0", "::"}:
            allowed.add(actual_host)
        if is_loopback_host(bind_host) or is_loopback_host(actual_host):
            allowed.update({"localhost", "127.0.0.1", "::1"})
        return allowed

    def validate_host(
        self,
        headers: Mapping[str, str],
        *,
        server_host: str,
        server_port: int,
    ) -> tuple[str, int]:
        try:
            request_host = _authority_parts(
                headers.get("Host", ""),
                default_port=80,
            )
        except ValueError as exc:
            raise HttpRequestSecurityError(400, "missing or invalid Host header") from exc
        hostname, port = request_host
        if port != int(server_port) or hostname not in self._allowed_hostnames(server_host):
            raise HttpRequestSecurityError(403, "Host is not allowed for this control server")
        return request_host

    def validate_control_request(
        self,
        headers: Mapping[str, str],
        *,
        server_host: str,
        server_port: int,
    ) -> int:
        request_host = self.validate_host(
            headers,
            server_host=server_host,
            server_port=server_port,
        )

        origin = str(headers.get("Origin", "") or "").strip()
        if not origin:
            raise HttpRequestSecurityError(403, "Origin header is required for control requests")
        try:
            parsed_origin = urlsplit(origin)
            if (
                parsed_origin.scheme.lower() != "http"
                or parsed_origin.username is not None
                or parsed_origin.password is not None
                or parsed_origin.path not in {"", "/"}
                or parsed_origin.query
                or parsed_origin.fragment
            ):
                raise ValueError("invalid Origin")
            origin_host = _authority_parts(parsed_origin.netloc, default_port=80)
        except (TypeError, ValueError) as exc:
            raise HttpRequestSecurityError(403, "Origin is not allowed for this control server") from exc
        if origin_host != request_host:
            raise HttpRequestSecurityError(403, "Origin is not allowed for this control server")

        supplied_token = str(headers.get(CONTROL_TOKEN_HEADER, "") or "")
        if not supplied_token or not secrets.compare_digest(supplied_token, self.token):
            raise HttpRequestSecurityError(401, "missing or invalid control token")

        content_type = str(headers.get("Content-Type", "") or "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "application/json":
            raise HttpRequestSecurityError(415, "control requests require Content-Type: application/json")
        if headers.get("Transfer-Encoding"):
            raise HttpRequestSecurityError(400, "chunked control requests are not supported")
        raw_length = headers.get("Content-Length")
        if raw_length is None:
            raise HttpRequestSecurityError(411, "Content-Length is required")
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise HttpRequestSecurityError(400, "invalid Content-Length") from exc
        if length < 0:
            raise HttpRequestSecurityError(400, "invalid Content-Length")
        if length > int(self.max_control_body_bytes):
            raise HttpRequestSecurityError(
                413,
                f"control request exceeds {int(self.max_control_body_bytes)} bytes",
            )
        return length
