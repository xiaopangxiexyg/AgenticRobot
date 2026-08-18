"""Forced-command stdio bridge to one host-local Agent Gateway socket.

SSH owns authentication and launches this module on the hardware host.  The
bridge opens no device and implements no Skill semantics.  It permits only a
small read-only method surface, while the Gateway independently revalidates
that every submitted Route is strictly read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import struct
import sys
from collections.abc import Mapping
from typing import Any, BinaryIO

from .unix_transport import (
    AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
    MAX_AGENT_GATEWAY_UNIX_FRAME_BYTES,
)


AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1 = frozenset(
    {
        "healthz",
        "get_skill_definition_catalog",
        "get_agent_skill_schema_catalog",
        "get_skill_route_manifest",
        "get_skill_io_catalog",
        "get_gateway_session_capabilities",
        "get_hardware_resource_views",
        "get_hardware_status_views",
        "get_observation_snapshot",
        "submit_read_only_skill_request",
        "get_own_operation",
        "cancel_own_read_only_operation",
        "begin_artifact_download",
        "read_artifact_chunk",
        "acknowledge_artifact_download",
    }
)

_FRAME_PREFIX = struct.Struct(">I")


class ReadOnlySshProxyError(RuntimeError):
    """Bounded local proxy rejection safe to report to an SSH client."""


def _read_exact(stream: BinaryIO | socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        if isinstance(stream, socket.socket):
            chunk = stream.recv(size - len(chunks))
        else:
            chunk = stream.read(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise ReadOnlySshProxyError("stream ended within a framed message")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_frame(
    stream: BinaryIO | socket.socket,
    *,
    max_frame_bytes: int,
) -> bytes | None:
    prefix = _read_exact(stream, _FRAME_PREFIX.size)
    if prefix is None:
        return None
    (size,) = _FRAME_PREFIX.unpack(prefix)
    if not 1 <= size <= max_frame_bytes:
        raise ReadOnlySshProxyError("frame size is outside the configured bound")
    body = _read_exact(stream, size)
    if body is None:  # pragma: no cover - prefix already proved non-empty input.
        raise ReadOnlySshProxyError("stream ended before the framed message body")
    return body


def _request_identity(payload: object) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise ReadOnlySshProxyError("request envelope must be a JSON object")
    if set(payload) != {"protocol", "request_id", "method", "payload"}:
        raise ReadOnlySshProxyError("request envelope fields are not exact")
    if payload["protocol"] != AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2:
        raise ReadOnlySshProxyError("request protocol is not supported")
    request_id = payload["request_id"]
    method = payload["method"]
    if (
        not isinstance(request_id, str)
        or request_id != request_id.strip()
        or not 1 <= len(request_id) <= 128
    ):
        raise ReadOnlySshProxyError("request_id is invalid")
    if not isinstance(method, str) or method not in (
        AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1
    ):
        raise PermissionError("SSH Gateway access is read-only")
    return request_id, method


def _required_payload_text(envelope: Mapping[str, Any], field_name: str) -> str:
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise PermissionError("SSH Artifact access requires a JSON payload")
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise PermissionError(f"SSH Artifact access requires {field_name}")
    return value


def _operation_artifact_ids(response: bytes) -> set[str]:
    try:
        envelope = json.loads(response.decode("utf-8"))
        result = envelope["result"]
        operation = result["operation"]
        bindings = operation["output_artifact_bindings"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if not isinstance(bindings, list):
        return set()
    values = set()
    for binding in bindings:
        try:
            artifact_id = binding["artifact"]["artifact_id"]
        except (KeyError, TypeError):
            continue
        if isinstance(artifact_id, str) and artifact_id:
            values.add(artifact_id)
    return values


def _issued_grant_id(response: bytes) -> str | None:
    try:
        envelope = json.loads(response.decode("utf-8"))
        grant_id = envelope["result"]["grant_id"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return grant_id if isinstance(grant_id, str) and grant_id else None


def _encoded_error(request_id: str, *, code: str, message: str) -> bytes:
    payload = {
        "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
        "request_id": request_id,
        "ok": False,
        "error": {"code": code, "message": message[:512]},
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_frame(stream: BinaryIO, body: bytes) -> None:
    stream.write(_FRAME_PREFIX.pack(len(body)))
    stream.write(body)
    stream.flush()


def serve_ssh_readonly_proxy(
    *,
    socket_path: str | os.PathLike[str],
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    max_frame_bytes: int = 2 * 1024 * 1024,
) -> int:
    """Bridge bounded request/response frames until EOF or a policy denial."""

    path = os.fspath(socket_path)
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise ValueError("socket_path must be a non-empty absolute path.")
    if (
        isinstance(max_frame_bytes, bool)
        or not isinstance(max_frame_bytes, int)
        or not 1024 <= max_frame_bytes <= MAX_AGENT_GATEWAY_UNIX_FRAME_BYTES
    ):
        raise ValueError("max_frame_bytes is outside the Gateway bound.")
    endpoint = os.stat(path, follow_symlinks=False)
    if not stat.S_ISSOCK(endpoint.st_mode):
        raise ValueError("socket_path must identify a Unix socket.")

    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    visible_artifact_ids: set[str] = set()
    issued_grant_ids: set[str] = set()
    try:
        upstream.connect(path)
        while True:
            body = _read_frame(input_stream, max_frame_bytes=max_frame_bytes)
            if body is None:
                return 0
            request_id = "ssh-readonly:invalid"
            try:
                envelope: Any = json.loads(body.decode("utf-8"))
                if isinstance(envelope, Mapping):
                    raw_request_id = envelope.get("request_id")
                    if isinstance(raw_request_id, str) and raw_request_id:
                        request_id = raw_request_id[:128]
                _request_id, method = _request_identity(envelope)
                if method == "begin_artifact_download":
                    artifact_id = _required_payload_text(envelope, "artifact_id")
                    mechanism = _required_payload_text(envelope, "mechanism")
                    if mechanism != "chunked_rpc":
                        raise PermissionError(
                            "SSH Artifact access permits only chunked_rpc"
                        )
                    if artifact_id not in visible_artifact_ids:
                        raise PermissionError(
                            "Artifact was not exposed by this connection's "
                            "own operation"
                        )
                elif method in {
                    "read_artifact_chunk",
                    "acknowledge_artifact_download",
                }:
                    grant_id = _required_payload_text(envelope, "grant_id")
                    if grant_id not in issued_grant_ids:
                        raise PermissionError(
                            "Artifact grant was not issued through this SSH connection"
                        )
            except PermissionError as error:
                _write_frame(
                    output_stream,
                    _encoded_error(
                        request_id,
                        code="permission_denied",
                        message=str(error),
                    ),
                )
                return 3
            except (
                ReadOnlySshProxyError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:
                _write_frame(
                    output_stream,
                    _encoded_error(
                        request_id,
                        code="proxy_request_invalid",
                        message=str(error),
                    ),
                )
                return 2

            upstream.sendall(_FRAME_PREFIX.pack(len(body)) + body)
            response = _read_frame(upstream, max_frame_bytes=max_frame_bytes)
            if response is None:
                raise ReadOnlySshProxyError("Gateway closed before responding")
            if method in {
                "submit_read_only_skill_request",
                "get_own_operation",
            }:
                visible_artifact_ids = _operation_artifact_ids(response)
            elif method == "begin_artifact_download":
                grant_id = _issued_grant_id(response)
                if grant_id is not None:
                    if len(issued_grant_ids) >= 64:
                        raise ReadOnlySshProxyError(
                            "SSH Artifact grant tracking capacity is exhausted"
                        )
                    issued_grant_ids.add(grant_id)
            elif method == "acknowledge_artifact_download":
                issued_grant_ids.discard(grant_id)
            _write_frame(output_stream, response)
    finally:
        upstream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway ssh-readonly-proxy",
        description=(
            "Serve a forced-command SSH stdio bridge to one already-running "
            "host-local Agent Gateway. Motion, lifecycle, HOLD and uploads are "
            "always rejected."
        ),
    )
    parser.add_argument("--socket-path", required=True)
    parser.add_argument(
        "--max-frame-bytes",
        type=int,
        default=2 * 1024 * 1024,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return serve_ssh_readonly_proxy(
            socket_path=args.socket_path,
            input_stream=sys.stdin.buffer,
            output_stream=sys.stdout.buffer,
            max_frame_bytes=args.max_frame_bytes,
        )
    except (OSError, ReadOnlySshProxyError, TypeError, ValueError) as error:
        print(
            f"ssh_readonly_proxy_failed:{type(error).__name__}:{error}",
            file=sys.stderr,
        )
        return 1


__all__ = [
    "AGENT_GATEWAY_SSH_READ_ONLY_METHODS_V1",
    "ReadOnlySshProxyError",
    "build_parser",
    "main",
    "serve_ssh_readonly_proxy",
]
