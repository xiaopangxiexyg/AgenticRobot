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
    build_control_station_status_payload,
)
from control_station.web import (
    build_read_only_control_station_server,
)


def _manifest() -> ControlStationRunManifest:
    source = SourceSlotSpec(
        source_slot_id="policy-primary",
        source_kind="policy",
        producer_id="policy:primary",
        source_action_contract_hash="policy-action-contract:v1",
        producer_epoch_policy_hash="producer-epoch-policy:v1",
        supported_uses=("shadow",),
        expected_emission_seam="policy-action-emission:v1",
        required_binding_refs=("binding:policy",),
        trace_requirement_refs=("trace:proposal",),
        required_for_run=True,
    )
    target = ExecutionDomainBinding(
        binding_id="target:null",
        target_kind="null",
        control_domain_id="domain:arms",
        embodiment_instance_id="embodiment:null",
        target_runtime_action_contract_hash="target-action-contract:v1",
        execution_capability_profile_hash="execution-capability:null-v1",
        component_catalog_hash="component-catalog:arms-v1",
        selection_group_ids=("arms",),
        owner_registry_id="owner-registry:shared",
        authority_epoch_source_id="authority-epoch:shared",
        execution_epoch_source_id="execution-epoch:shared",
        trace_requirement_profile_id="trace:null-v1",
    )
    return ControlStationRunManifest(
        manifest_id="control-station:cs1a-read-only-v1",
        manifest_version="1",
        execution_binding=target,
        source_slots=(source,),
        authority_policy_hash="authority-policy:no-motion-v1",
        decision_schedule_hash="decision-schedule:none-v1",
        trace_requirement_profile_id="trace:station-read-only-v1",
        capture_policy_id="capture:none-v1",
        allowed_intent_kinds=("connect_source", "connect_target", "hold"),
        operator_control_lease_sec=10.0,
        explicit_exclusions=("motion", "capture", "production_entrypoint"),
    )


@contextmanager
def _running_server(runtime: ControlStationRuntime):
    server = build_read_only_control_station_server(runtime, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=2.0) as response:  # noqa: S310 - loopback fixture
        return json.loads(response.read().decode("utf-8"))


def test_status_projection_is_read_only_and_not_a_control_fact() -> None:
    manifest = _manifest()
    runtime = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id="session:cs1a",
    )
    payload = build_control_station_status_payload(manifest, runtime.state)
    assert payload["read_only"] is True
    assert payload["target"]["motion_authorized"] is False
    assert payload["capabilities"] == {
        "state_read": True,
        "intent_submission": False,
        "motion": False,
        "capture": False,
        "source_adapters": "null_or_fake_only",
    }
    assert payload["control_facts"]["dispatch_owner"] == "unavailable_in_cs1a"


def test_cs1a_http_state_and_health_are_loopback_read_only() -> None:
    manifest = _manifest()
    runtime = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id="session:cs1a",
    )
    with _running_server(runtime) as root:
        health = _get_json(f"{root}/api/health")
        state = _get_json(f"{root}/api/state")
    assert health == {
        "maturity": "cs1a_read_only_local_gui",
        "motion_authorized": False,
        "read_only": True,
        "status": "ok",
    }
    assert state["runtime_session_id"] == "session:cs1a"
    assert state["manifest"]["semantic_content_hash"] == manifest.semantic_content_hash
    assert state["target"]["motion_authorized"] is False


def test_cs1a_rejects_post_without_mutating_station_state() -> None:
    runtime = ControlStationRuntime(
        manifest=_manifest(),
        runtime_session_id="session:cs1a",
    )
    state_hash_before = runtime.state.semantic_content_hash
    sequence_before = runtime.next_admission_sequence
    with _running_server(runtime) as root:
        request = Request(
            f"{root}/api/intents",
            data=b'{"intent_kind":"connect_target"}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2.0)  # noqa: S310 - loopback fixture
        body = json.loads(error.value.read().decode("utf-8"))
    assert error.value.code == 405
    assert error.value.headers["Allow"] == "GET"
    assert body["error"] == "cs1a_read_only"
    assert body["motion_authorized"] is False
    assert runtime.state.semantic_content_hash == state_hash_before
    assert runtime.next_admission_sequence == sequence_before


def test_cs1a_serves_separate_static_assets_and_rejects_unknown_paths() -> None:
    runtime = ControlStationRuntime(
        manifest=_manifest(),
        runtime_session_id="session:cs1a",
    )
    with _running_server(runtime) as root:
        with urlopen(f"{root}/", timeout=2.0) as response:  # noqa: S310
            html = response.read().decode("utf-8")
            assert response.headers["X-Frame-Options"] == "DENY"
        with urlopen(f"{root}/app.js", timeout=2.0) as response:  # noqa: S310
            javascript = response.read().decode("utf-8")
        with urlopen(f"{root}/styles.css", timeout=2.0) as response:  # noqa: S310
            stylesheet = response.read().decode("utf-8")
        with pytest.raises(HTTPError) as error:
            urlopen(f"{root}/../contracts.py", timeout=2.0)  # noqa: S310
    assert '<script src="/app.js" defer></script>' in html
    assert '<link rel="stylesheet" href="/styles.css">' in html
    assert "fetch(\"/api/state\"" in javascript
    assert ".panel" in stylesheet
    assert error.value.code == 404


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.10", "::1"])
def test_cs1a_rejects_non_supported_bind_hosts(host: str) -> None:
    runtime = ControlStationRuntime(
        manifest=_manifest(),
        runtime_session_id="session:cs1a",
    )
    with pytest.raises(ValueError, match="loopback"):
        build_read_only_control_station_server(runtime, host=host, port=0)


@pytest.mark.parametrize("port", [True, 1.0, "1", -1, 65536])
def test_cs1a_rejects_invalid_ports(port: object) -> None:
    runtime = ControlStationRuntime(
        manifest=_manifest(),
        runtime_session_id="session:cs1a",
    )
    with pytest.raises(ValueError, match="port"):
        build_read_only_control_station_server(runtime, port=port)  # type: ignore[arg-type]
