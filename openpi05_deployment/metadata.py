from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from openpi05_deployment.contracts import (
    OPENPI_LOCAL_PROTOCOL_VERSION,
    TACTILE_KIND_NONE,
    TACTILE_KIND_VECTOR,
    EmbodimentContract,
    resolve_embodiment_contract,
)
from openpi05_deployment.errors import ServerMetadataError


@dataclass(frozen=True)
class ServerContract:
    host: str
    port: int
    websocket_uri: str
    metadata_url: str
    protocol_version: str
    embodiment: EmbodimentContract
    deployment_config_name: str
    checkpoint_dir: str
    checkpoint_step: int | None
    default_prompt: str | None
    action_horizon: int
    tactile_kind: str
    tactile_tokens: int
    tactile_dim: int
    raw_metadata: dict[str, Any]

    @property
    def tactile_required(self) -> bool:
        return self.tactile_kind == TACTILE_KIND_VECTOR and self.tactile_tokens > 0


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    websocket_uri: str
    metadata_url: str


def resolve_endpoint(host: str, port: int) -> Endpoint:
    text = str(host or "127.0.0.1").strip()
    if not text:
        text = "127.0.0.1"
    if "://" not in text:
        netloc = text
    else:
        parsed = urllib_parse.urlparse(text)
        netloc = parsed.netloc or parsed.path
    if not netloc:
        raise ServerMetadataError(f"Could not parse host {host!r}.")
    if ":" in netloc and port is None:
        parsed_host, parsed_port = netloc.rsplit(":", 1)
        netloc = parsed_host
        port = int(parsed_port)
    resolved_port = int(port)
    if resolved_port <= 0:
        raise ServerMetadataError(f"Port must be positive, got {port!r}.")
    host_only = netloc.strip("[]")
    return Endpoint(
        host=host_only,
        port=resolved_port,
        websocket_uri=f"ws://{host_only}:{resolved_port}",
        metadata_url=f"http://{host_only}:{resolved_port}/metadata",
    )


def fetch_server_metadata(
    *,
    host: str,
    port: int,
    timeout_sec: float = 600.0,
    poll_interval_sec: float = 1.0,
    request_timeout_sec: float = 3.0,
) -> dict[str, Any]:
    endpoint = resolve_endpoint(host, port)
    deadline = None if timeout_sec is None or float(timeout_sec) <= 0.0 else time.monotonic() + float(timeout_sec)
    while True:
        try:
            with urllib_request.urlopen(
                endpoint.metadata_url,
                timeout=float(request_timeout_sec),
            ) as response:
                payload = response.read().decode("utf-8")
            metadata = json.loads(payload)
            if not isinstance(metadata, dict):
                raise ServerMetadataError(
                    f"Expected dict metadata from {endpoint.metadata_url}, got {type(metadata).__name__}."
                )
            return metadata
        except (urllib_error.URLError, TimeoutError, OSError, json.JSONDecodeError, ServerMetadataError) as exc:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for OpenPI metadata at "
                    f"{endpoint.metadata_url} after {float(timeout_sec):.1f}s."
                ) from exc
            time.sleep(max(0.1, float(poll_interval_sec)))


def parse_server_contract(
    metadata: Mapping[str, Any],
    *,
    host: str,
    port: int,
    expected_embodiment: str | None = None,
) -> ServerContract:
    payload = dict(metadata or {})
    protocol_version = str(payload.get("protocol_version") or "").strip()
    if protocol_version != OPENPI_LOCAL_PROTOCOL_VERSION:
        raise ServerMetadataError(
            "Unsupported OpenPI protocol version "
            f"{protocol_version!r}; expected {OPENPI_LOCAL_PROTOCOL_VERSION!r}."
        )

    endpoint = resolve_endpoint(host, port)
    model = dict(payload.get("model") or {})
    deployment = dict(payload.get("deployment") or {})
    local_robot = dict(payload.get("local_robot") or {})

    embodiment_name = str(local_robot.get("embodiment") or "").strip()
    contract = resolve_embodiment_contract(embodiment_name)
    if expected_embodiment is not None:
        expected = resolve_embodiment_contract(expected_embodiment)
        if expected.name != contract.name:
            raise ServerMetadataError(
                "OpenPI server embodiment mismatch: "
                f"expected {expected.name!r}, got {contract.name!r}."
            )

    active_sides = tuple(str(side).strip().lower() for side in (local_robot.get("active_sides") or ()))
    if active_sides and active_sides != contract.active_sides:
        raise ServerMetadataError(
            "OpenPI server active_sides mismatch: "
            f"expected {contract.active_sides}, got {active_sides}."
        )

    state_dim = int(local_robot.get("state_dim") or 0)
    output_action_dim = int(local_robot.get("output_action_dim") or 0)
    if state_dim != contract.state_dim:
        raise ServerMetadataError(
            f"OpenPI state_dim mismatch: expected {contract.state_dim}, got {state_dim}."
        )
    if output_action_dim != contract.output_action_dim:
        raise ServerMetadataError(
            "OpenPI output_action_dim mismatch: "
            f"expected {contract.output_action_dim}, got {output_action_dim}."
        )

    action_horizon = int(model.get("action_horizon") or 0)
    if action_horizon <= 0:
        raise ServerMetadataError(f"OpenPI action_horizon must be positive, got {action_horizon}.")

    tactile_kind = str(model.get("tactile_kind") or TACTILE_KIND_NONE).strip().lower()
    if tactile_kind in {"", "null"}:
        tactile_kind = TACTILE_KIND_NONE
    if tactile_kind not in {TACTILE_KIND_NONE, TACTILE_KIND_VECTOR}:
        raise ServerMetadataError(
            f"Unsupported tactile_kind {tactile_kind!r}. Only {TACTILE_KIND_NONE!r} and {TACTILE_KIND_VECTOR!r} are supported."
        )
    tactile_tokens = int(model.get("tactile_tokens") or 0)
    tactile_dim = int(model.get("tactile_dim") or 0)

    local_tactile_kind = str(local_robot.get("tactile_kind") or tactile_kind).strip().lower()
    if local_tactile_kind in {"", "null"}:
        local_tactile_kind = TACTILE_KIND_NONE
    if local_tactile_kind != tactile_kind:
        raise ServerMetadataError(
            "OpenPI tactile kind mismatch between model and local_robot metadata: "
            f"{tactile_kind!r} vs {local_tactile_kind!r}."
        )

    expected_tactile_tokens = len(contract.tactile_token_order) if tactile_kind == TACTILE_KIND_VECTOR else 0
    if tactile_tokens != expected_tactile_tokens:
        raise ServerMetadataError(
            "OpenPI tactile token count mismatch: "
            f"expected {expected_tactile_tokens}, got {tactile_tokens}."
        )
    if tactile_kind == TACTILE_KIND_VECTOR and tactile_dim <= 0:
        raise ServerMetadataError("OpenPI tactile_dim must be positive for tactile vector deployments.")

    return ServerContract(
        host=endpoint.host,
        port=endpoint.port,
        websocket_uri=endpoint.websocket_uri,
        metadata_url=endpoint.metadata_url,
        protocol_version=protocol_version,
        embodiment=contract,
        deployment_config_name=str(deployment.get("config_name") or ""),
        checkpoint_dir=str(deployment.get("checkpoint_dir") or ""),
        checkpoint_step=(
            None
            if deployment.get("checkpoint_step") in (None, "")
            else int(deployment.get("checkpoint_step"))
        ),
        default_prompt=(
            None
            if deployment.get("default_prompt") in (None, "")
            else str(deployment.get("default_prompt"))
        ),
        action_horizon=action_horizon,
        tactile_kind=tactile_kind,
        tactile_tokens=tactile_tokens,
        tactile_dim=tactile_dim,
        raw_metadata=payload,
    )
