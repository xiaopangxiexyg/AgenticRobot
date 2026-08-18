"""Operator one-shot client for a pre-authorized real primitive Gateway."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_gateway.skills.motion.definition import (
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1,
    TCP_CARDINAL_STEP_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
)
from agent_gateway.transport.unix_transport import (
    AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
)

from .motion_primitive_profiles import (
    REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1,
)

_PREFIX = struct.Struct(">I")
_CONFIRM_TOKEN = "EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE"
_ACTION_TO_SKILL_ID = {
    **dict(FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1),
    "level_forward": TCP_LEVEL_FORWARD_SKILL_ID_V1,
    "level_forward_left_30": TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    "level_forward_right_30": TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
}
_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "timed_out"}
)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    body = bytearray()
    while len(body) < size:
        chunk = sock.recv(size - len(body))
        if not chunk:
            raise RuntimeError("Gateway socket closed before a complete response.")
        body.extend(chunk)
    return bytes(body)


class _Client:
    def __init__(self, *, socket_path: str, timeout_sec: float) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(float(timeout_sec))
        self._sock.connect(socket_path)

    def close(self) -> None:
        self._sock.close()

    def call(self, method: str, payload: Mapping[str, Any] | None) -> Any:
        request_id = "operator-client:" + uuid.uuid4().hex
        encoded = json.dumps(
            {
                "protocol": AGENT_GATEWAY_UNIX_JSON_PROTOCOL_V2,
                "request_id": request_id,
                "method": method,
                "payload": payload,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._sock.sendall(_PREFIX.pack(len(encoded)) + encoded)
        (size,) = _PREFIX.unpack(_recv_exact(self._sock, _PREFIX.size))
        response = json.loads(_recv_exact(self._sock, size).decode("utf-8"))
        if response.get("request_id") != request_id:
            raise RuntimeError("Gateway response request_id mismatch.")
        if not bool(response.get("ok", False)):
            error = response.get("error", {})
            raise RuntimeError(
                f"Gateway rejected {method}: {error.get('code')}: "
                f"{error.get('message')}"
            )
        return response.get("result")


def _route_entry(manifest: Mapping[str, Any], *, skill_id: str) -> Mapping[str, Any]:
    matches = tuple(
        item
        for item in manifest["skill_routes"]
        if item["route"]["skill_id"] == skill_id
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one advertised Route for {skill_id}.")
    return matches[0]


def _schema_entry_for_route(
    *,
    manifest: Mapping[str, Any],
    schema_catalog: Mapping[str, Any],
    route: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(schema_catalog, Mapping):
        raise TypeError("Gateway returned an invalid Skill schema catalog.")
    if schema_catalog.get("skill_schema_catalog_hash") != manifest.get(
        "skill_schema_catalog_hash"
    ):
        raise RuntimeError("Skill schema catalog does not match the Route manifest.")
    entries = schema_catalog.get("schema_entries")
    if not isinstance(entries, (tuple, list)):
        raise TypeError("Skill schema catalog has no schema_entries list.")
    matches = tuple(
        item
        for item in entries
        if isinstance(item, Mapping)
        and item.get("skill_id") == route.get("skill_id")
        and item.get("skill_version") == route.get("skill_version")
    )
    if len(matches) != 1:
        raise RuntimeError("Gateway schema catalog has no unique matching Skill.")
    entry = matches[0]
    if entry.get("parameter_schema_hash") != route.get("parameter_schema_hash"):
        raise RuntimeError("Skill schema parameter hash does not match the Route.")
    route_profiles = route.get("allowed_parameter_profiles")
    schema_profiles = entry.get("parameter_profiles")
    if not isinstance(route_profiles, (tuple, list)) or not isinstance(
        schema_profiles, (tuple, list)
    ):
        raise TypeError("Route/schema parameter profiles are malformed.")
    schema_profile_keys = {
        (item.get("parameter_profile_id"), item.get("parameter_profile_hash"))
        for item in schema_profiles
        if isinstance(item, Mapping)
    }
    if any(
        not isinstance(item, Mapping)
        or (
            item.get("parameter_profile_id"),
            item.get("parameter_profile_hash"),
        )
        not in schema_profile_keys
        for item in route_profiles
    ):
        raise RuntimeError("Route parameter profile is absent from its schema.")
    return entry


def _schema_properties(schema_entry: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = schema_entry.get("parameter_schema")
    if not isinstance(schema, Mapping):
        raise TypeError("Skill schema entry has no parameter_schema object.")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise TypeError("Skill parameter schema has no properties object.")
    return properties


def _schema_const(
    properties: Mapping[str, Any],
    field_name: str,
) -> Any:
    field = properties.get(field_name)
    if not isinstance(field, Mapping) or "const" not in field:
        raise RuntimeError(
            f"Skill parameter schema has no trusted const for {field_name}."
        )
    return field["const"]


def _schema_enum(
    properties: Mapping[str, Any],
    field_name: str,
) -> tuple[Any, ...]:
    field = properties.get(field_name)
    values = None if not isinstance(field, Mapping) else field.get("enum")
    if not isinstance(values, (tuple, list)) or not values:
        raise RuntimeError(
            f"Skill parameter schema has no trusted enum for {field_name}."
        )
    return tuple(values)


def _parameters(
    *,
    action: str,
    skill_id: str,
    side: str,
    snapshot: Mapping[str, Any],
    schema_entry: Mapping[str, Any],
) -> dict[str, Any]:
    summary = snapshot["measurement_summary"]
    components = tuple(summary["pose_xyz_quat_xyzw_by_component"])
    expected_component = f"{side}_arm"
    if components != (expected_component,):
        raise RuntimeError(
            "Real primitive observation does not exactly match the selected side."
        )
    properties = _schema_properties(schema_entry)
    if expected_component not in _schema_enum(properties, "component_id"):
        raise RuntimeError("Selected side is not allowed by the Skill schema.")
    fixed_cardinal_direction = (
        FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1.get(skill_id)
    )
    if (
        skill_id == TCP_CARDINAL_STEP_SKILL_ID_V1
        or fixed_cardinal_direction is not None
    ) and action not in _schema_enum(properties, "direction"):
        raise RuntimeError("Requested cardinal action is not allowed by the Skill schema.")
    reference_frames = _schema_enum(properties, "reference_frame_id")
    if len(reference_frames) != 1:
        raise RuntimeError("Real primitive schema must bind one reference frame.")
    payload = {
        "component_id": expected_component,
        "reference_frame_id": reference_frames[0],
        "reference_frame_spec_hash": _schema_const(
            properties, "reference_frame_spec_hash"
        ),
        "controlled_frame_id": _schema_const(properties, "controlled_frame_id"),
        "frame_graph_hash": _schema_const(properties, "frame_graph_hash"),
        "tool_config_hash": _schema_const(properties, "tool_config_hash"),
        "calibration_hash": _schema_const(properties, "calibration_hash"),
    }
    for field_name in ("frame_graph_hash", "tool_config_hash", "calibration_hash"):
        if payload[field_name] != snapshot.get(field_name):
            raise RuntimeError(
                f"Skill schema {field_name} disagrees with the observation snapshot."
            )
    if (
        skill_id == TCP_CARDINAL_STEP_SKILL_ID_V1
        or fixed_cardinal_direction is not None
    ):
        payload["direction"] = action
        payload["primitive_catalog_hash"] = _schema_const(
            properties, "primitive_catalog_hash"
        )
    else:
        if skill_id not in REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1:
            raise RuntimeError("Unsupported fixed-orientation real primitive Skill.")
        payload["primitive_hash"] = _schema_const(properties, "primitive_hash")
    return payload


def execute_one_shot(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.confirm != _CONFIRM_TOKEN:
        raise ValueError(f"--confirm must equal {_CONFIRM_TOKEN}")
    if not Path(args.socket_path).is_absolute():
        raise ValueError("--socket-path must be absolute")
    expected_component = f"{args.side}_arm"
    skill_id = _ACTION_TO_SKILL_ID[args.action]
    client = _Client(
        socket_path=args.socket_path,
        timeout_sec=args.timeout_sec,
    )
    try:
        manifest = client.call("get_skill_route_manifest", None)
        entry = _route_entry(manifest, skill_id=skill_id)
        route = entry["route"]
        schema_catalog = client.call("get_agent_skill_schema_catalog", None)
        schema_entry = _schema_entry_for_route(
            manifest=manifest,
            schema_catalog=schema_catalog,
            route=route,
        )
        snapshot = client.call("get_observation_snapshot", None)
        if snapshot["lifecycle_state"] != "running" or (
            snapshot["execution_gate"] != "ready"
        ):
            raise RuntimeError("Lower real Skill owner is not RUNNING/READY.")
        parameters = _parameters(
            action=args.action,
            skill_id=skill_id,
            side=args.side,
            snapshot=snapshot,
            schema_entry=schema_entry,
        )
        if parameters["component_id"] != expected_component:
            raise RuntimeError(
                "Server component does not match the operator-selected side."
            )
        binding = entry["deployment_binding"]
        profiles = tuple(route["allowed_parameter_profiles"])
        if len(profiles) != 1:
            raise RuntimeError("Reviewed real primitive Route requires one profile.")
        envelopes = tuple(binding.get("safety_envelope_refs", ()))
        matching_envelopes = tuple(
            item for item in envelopes if route["route_id"] in item["allowed_route_ids"]
        )
        if len(matching_envelopes) != 1:
            raise RuntimeError("Reviewed Route requires one matching safety envelope.")
        envelope = matching_envelopes[0]
        request_nonce = uuid.uuid4().hex
        request = {
            "client_id": args.client_id,
            "request_id": "real-primitive-request:" + request_nonce,
            "idempotency_key": "real-primitive-idempotency:" + request_nonce,
            "client_sequence": 1,
            "runtime_session_id": manifest["runtime_session_id"],
            "skill_route_manifest_hash": manifest["route_manifest_hash"],
            "expected_station_revision": snapshot["station_revision"],
            "skill_id": route["skill_id"],
            "skill_version": route["skill_version"],
            "skill_definition_hash": route["skill_definition_hash"],
            "route_id": route["route_id"],
            "route_version": route["route_version"],
            "route_hash": route["route_hash"],
            "deployment_binding_hash": binding["deployment_binding_hash"],
            "parameter_profile_id": profiles[0]["parameter_profile_id"],
            "parameter_profile_hash": profiles[0]["parameter_profile_hash"],
            "parameters": parameters,
            "observation_snapshot_id": snapshot["snapshot_id"],
            "requested_mode": "real_active",
            "requested_timeout_ms": int(args.operation_timeout_sec * 1000.0),
            "skill_execution_grant_id": None,
            "input_artifact_bindings": [],
            "safety_profile_id": envelope["object_id"],
            "safety_profile_hash": envelope["content_hash"],
            "deployment_safety_envelope_id": envelope["object_id"],
            "deployment_safety_envelope_hash": envelope["content_hash"],
            "safety_constraint_set_hash": envelope["constraint_set_hash"],
        }
        submission = client.call("submit_skill_request", request)
        if not bool(submission.get("accepted", False)):
            raise RuntimeError(
                "Primitive request was not admitted: "
                + str(submission.get("reason_code"))
            )
        operation = submission.get("operation")
        if not isinstance(operation, Mapping):
            raise TypeError("Gateway returned no admitted operation.")
        operation_id = str(operation["operation_id"])
        deadline = time.monotonic() + float(args.operation_timeout_sec) + 3.0
        while str(operation["status"]) not in _TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for terminal operation/HOLD.")
            time.sleep(0.05)
            outcome = client.call(
                "get_own_operation",
                {
                    "client_id": args.client_id,
                    "query_id": "real-primitive-query:" + uuid.uuid4().hex,
                    "runtime_session_id": manifest["runtime_session_id"],
                    "skill_route_manifest_hash": manifest["route_manifest_hash"],
                    "operation_id": operation_id,
                },
            )
            if not bool(outcome.get("accepted", False)):
                raise RuntimeError(
                    "Gateway could not return the admitted operation: "
                    + str(outcome.get("reason_code"))
                )
            operation = outcome.get("operation")
            if not isinstance(operation, Mapping):
                raise TypeError("Gateway returned no owned operation.")
        return operation
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the exact primitive pre-authorized by RealTeleop.",
    )
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--client-id", default="dexteleop-local-primitive-operator")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--action", choices=tuple(_ACTION_TO_SKILL_ID), required=True)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--operation-timeout-sec", type=float, default=5.0)
    parser.add_argument("--confirm", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operation = execute_one_shot(args)
    print(json.dumps(operation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if operation["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
