from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from control_station import (
    ControlStationRunManifest,
    ControlStationRuntime,
    ExecutionDomainBinding,
    SourceSlotSpec,
)
from control_station.web import (
    CS1B_SAFE_INTENT_KINDS,
    LocalIntentAccessPolicy,
    build_typed_intent_control_station_server,
)
from control_station.web.api import station_intent_draft_from_json


_TOKEN = "cs1b-test-token-0123456789-abcdef"


def _manifest(
    *,
    target_kind: str = "null",
    explicit_exclusions: tuple[str, ...] = (
        "capture",
        "motion",
        "production_entrypoint",
    ),
) -> ControlStationRunManifest:
    source = SourceSlotSpec(
        source_slot_id="policy-primary",
        source_kind="policy",
        producer_id="policy:primary",
        source_action_contract_hash="policy-action-contract:v1",
        producer_epoch_policy_hash="producer-epoch-policy:v1",
        supported_uses=("shadow",),
        expected_emission_seam="policy-action-emission:v1",
        required_for_run=True,
    )
    target = ExecutionDomainBinding(
        binding_id=f"target:{target_kind}",
        target_kind=target_kind,
        control_domain_id="domain:arms",
        embodiment_instance_id=f"embodiment:{target_kind}",
        target_runtime_action_contract_hash="target-action-contract:v1",
        execution_capability_profile_hash=f"execution-capability:{target_kind}-v1",
        component_catalog_hash="component-catalog:arms-v1",
        selection_group_ids=("arms",),
        owner_registry_id="owner-registry:shared",
        authority_epoch_source_id="authority-epoch:shared",
        execution_epoch_source_id="execution-epoch:shared",
        trace_requirement_profile_id=f"trace:{target_kind}-v1",
        backend_inventory_hash=("inventory:real-v1" if target_kind == "real" else None),
    )
    return ControlStationRunManifest(
        manifest_id="control-station:cs1b-null-v1",
        manifest_version="1",
        execution_binding=target,
        source_slots=(source,),
        authority_policy_hash="authority-policy:no-motion-v1",
        decision_schedule_hash="decision-schedule:none-v1",
        trace_requirement_profile_id="trace:station-cs1b-v1",
        capture_policy_id="capture:none-v1",
        allowed_intent_kinds=CS1B_SAFE_INTENT_KINDS,
        operator_control_lease_sec=10.0,
        explicit_exclusions=explicit_exclusions,
    )


def _runtime(manifest: ControlStationRunManifest | None = None) -> ControlStationRuntime:
    return ControlStationRuntime(
        manifest=manifest or _manifest(),
        runtime_session_id="session:cs1b",
    )


def _policy(**kwargs) -> LocalIntentAccessPolicy:
    return LocalIntentAccessPolicy(
        access_policy_id="local-intent-policy:test-v1",
        bearer_token=_TOKEN,
        **kwargs,
    )


@contextmanager
def _running_server(runtime: ControlStationRuntime, policy: LocalIntentAccessPolicy):
    server = build_typed_intent_control_station_server(
        runtime,
        access_policy=policy,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _intent_payload(
    runtime: ControlStationRuntime,
    *,
    sequence: int,
    intent_kind: str,
    subject_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "client_id": "browser:test",
        "client_sequence": sequence,
        "intent_kind": intent_kind,
        "produced_at": 1000.0 + sequence,
        "produced_clock_domain_id": "browser-performance",
        "expected_state_revision": runtime.state.state_revision,
        "manifest_hash": runtime.manifest.semantic_content_hash,
        "reason": "http-fixture",
    }
    if subject_id is not None:
        payload["subject_id"] = subject_id
    return payload


def _request_json(
    url: str,
    payload: dict[str, object] | bytes,
    *,
    token: str | None = _TOKEN,
    origin: str | None = None,
    method: str = "POST",
) -> tuple[int, dict[str, object], object]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if origin is not None:
        headers["Origin"] = origin
    request = Request(url, data=data, headers=headers, method=method)
    try:
        response = urlopen(request, timeout=2.0)  # noqa: S310 - loopback fixture
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8")), error.headers
    with response:
        return (
            response.status,
            json.loads(response.read().decode("utf-8")),
            response.headers,
        )


def test_cs1b_access_policy_is_narrow_and_does_not_expose_token() -> None:
    policy = _policy()
    assert _TOKEN not in repr(policy)
    assert len(policy.token_fingerprint) == 64
    with pytest.raises(ValueError, match="unsafe intents"):
        _policy(allowed_intent_kinds=("request_motion_authority",))
    with pytest.raises(ValueError, match="at least 32"):
        LocalIntentAccessPolicy(
            access_policy_id="too-short",
            bearer_token="short",
        )
    with pytest.raises(ValueError, match="integer"):
        _policy(max_request_bytes=1024.0)
    with pytest.raises(TypeError, match="string"):
        LocalIntentAccessPolicy(
            access_policy_id="wrong-token-type",
            bearer_token=b"x" * 32,  # type: ignore[arg-type]
        )


def test_cs1b_json_parser_rejects_raw_action_and_nested_string_fields() -> None:
    runtime = _runtime()
    policy = _policy()
    payload = _intent_payload(runtime, sequence=0, intent_kind="hold")
    payload["qpos"] = [0.0] * 12
    with pytest.raises(ValueError, match="unknown fields"):
        station_intent_draft_from_json(payload, access_policy=policy)

    payload = _intent_payload(runtime, sequence=0, intent_kind="connect_source")
    payload["subject_id"] = {"unexpected": "object"}
    with pytest.raises(TypeError, match="subject_id"):
        station_intent_draft_from_json(payload, access_policy=policy)


def test_cs1b_requires_bearer_token_and_same_origin_when_origin_is_present() -> None:
    runtime = _runtime()
    state_hash = runtime.state.semantic_content_hash
    with _running_server(runtime, _policy()) as root:
        payload = _intent_payload(runtime, sequence=0, intent_kind="acquire_control_lease")
        status, body, headers = _request_json(
            f"{root}/api/intents", payload, token=None
        )
        assert status == 401
        assert body["error"] == "bearer_token_required"
        assert headers["WWW-Authenticate"] == "Bearer"

        status, body, _ = _request_json(
            f"{root}/api/intents", payload, token="x" * 32
        )
        assert status == 403
        assert body["error"] == "invalid_bearer_token"

        status, body, _ = _request_json(
            f"{root}/api/intents",
            payload,
            origin="http://attacker.invalid",
        )
        assert status == 403
        assert body["error"] == "cross_origin_request_rejected"
    assert runtime.state.semantic_content_hash == state_hash
    assert runtime.next_admission_sequence == 0


def test_cs1b_valid_intent_is_receiver_admitted_but_grants_no_motion() -> None:
    runtime = _runtime()
    with _running_server(runtime, _policy()) as root:
        status, body, _ = _request_json(
            f"{root}/api/intents",
            _intent_payload(runtime, sequence=0, intent_kind="acquire_control_lease"),
            origin=root,
        )
        state_response = urlopen(f"{root}/api/state", timeout=2.0)  # noqa: S310
        with state_response:
            state = json.loads(state_response.read().decode("utf-8"))
    assert status == 200
    assert body["receipt"]["accepted"] is True
    assert body["motion_authorized"] is False
    assert body["adapter_executed_inline"] is False
    assert runtime.state.operator_control_lease is not None
    assert runtime.next_admission_sequence == 1
    assert state["maturity"] == "cs1b_typed_intent_local_gui"
    assert state["capabilities"]["intent_submission"] is True
    assert state["capabilities"]["motion"] is False


def test_cs1b_web_token_does_not_replace_operator_control_lease() -> None:
    runtime = _runtime()
    with _running_server(runtime, _policy()) as root:
        status, body, _ = _request_json(
            f"{root}/api/intents",
            _intent_payload(
                runtime,
                sequence=0,
                intent_kind="connect_target",
                subject_id=runtime.manifest.execution_binding.binding_id,
            ),
        )
    assert status == 200
    assert body["receipt"]["accepted"] is False
    assert body["receipt"]["status"] == "operator_control_lease_required"
    assert body["motion_authorized"] is False
    assert runtime.state.operator_control_lease is None
    assert runtime.state.target_state == "configured"


def test_cs1b_operation_remains_pending_until_separate_adapter_evidence() -> None:
    runtime = _runtime()
    with _running_server(runtime, _policy()) as root:
        status, _, _ = _request_json(
            f"{root}/api/intents",
            _intent_payload(runtime, sequence=0, intent_kind="acquire_control_lease"),
        )
        assert status == 200
        status, body, _ = _request_json(
            f"{root}/api/intents",
            _intent_payload(
                runtime,
                sequence=1,
                intent_kind="connect_target",
                subject_id=runtime.manifest.execution_binding.binding_id,
            ),
        )
    assert status == 200
    assert body["receipt"]["status"] == "operation_requested"
    assert body["adapter_executed_inline"] is False
    assert runtime.state.target_state == "connect_requested"
    assert runtime.state.pending_operation_by_subject


def test_cs1b_invalid_or_disallowed_payload_never_enters_runtime_admission() -> None:
    runtime = _runtime()
    with _running_server(runtime, _policy(max_request_bytes=1024)) as root:
        raw_action = _intent_payload(runtime, sequence=0, intent_kind="hold")
        raw_action["action"] = {"left_arm_qpos": [0.0] * 6}
        status, body, _ = _request_json(f"{root}/api/intents", raw_action)
        assert status == 422
        assert body["error"] == "invalid_typed_intent"

        disallowed = _intent_payload(
            runtime,
            sequence=0,
            intent_kind="request_motion_authority",
        )
        status, body, _ = _request_json(f"{root}/api/intents", disallowed)
        assert status == 403
        assert body["error"] == "intent_not_enabled"

        status, body, _ = _request_json(
            f"{root}/api/intents",
            b'{' + b'"padding":"' + b'x' * 1100 + b'"}',
        )
        assert status == 413
        assert body["error"] == "request_too_large"
    assert runtime.next_admission_sequence == 0
    assert runtime.state.state_revision == 0


def test_cs1b_non_post_mutation_methods_are_rejected() -> None:
    runtime = _runtime()
    with _running_server(runtime, _policy()) as root:
        status, body, headers = _request_json(
            f"{root}/api/intents",
            _intent_payload(runtime, sequence=0, intent_kind="hold"),
            method="PUT",
        )
    assert status == 405
    assert body["error"] == "method_not_allowed"
    assert headers["Allow"] == "GET, POST"
    assert runtime.next_admission_sequence == 0


def test_cs1b_builder_is_null_target_and_explicit_no_motion_only() -> None:
    with pytest.raises(ValueError, match="null target"):
        build_typed_intent_control_station_server(
            _runtime(_manifest(target_kind="real")),
            access_policy=_policy(),
        )
    with pytest.raises(ValueError, match="explicitly exclude motion"):
        build_typed_intent_control_station_server(
            _runtime(_manifest(explicit_exclusions=("capture",))),
            access_policy=_policy(),
        )
