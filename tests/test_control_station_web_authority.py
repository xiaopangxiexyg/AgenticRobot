from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from control_station import (
    BoundedStationEventLog,
    ControlStationRuntime,
    SideEffectStationOperationCoordinator,
)
from control_station.composition import build_scoped_cs2a2b_mujoco_human_station
from control_station.sources import SyntheticHumanQposSourceSpec
from control_station.web import (
    LocalIntentAccessPolicy,
    ScopedSimAuthorityAccessPolicy,
    build_coordinated_control_station_server,
    build_scoped_sim_authority_control_station_server,
)
from control_station.web.api import station_intent_draft_from_json
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime


_TOKEN = "cs2a2b-scoped-sim-token-0123456789-abcdef"


class ManualClock:
    def __init__(self, value: float = 20.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        self.qpos = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }


def _build_station(clock: ManualClock):
    rig = MujocoPositionExecutionRig(
        FakeScene(),
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="cs2a2b-web-runtime",
            control_domain_id="cs2a2b-web-domain",
            embodiment_instance_id="cs2a2b-web-rm65",
            producer_id="synthetic-human-main",
            decision_hz=30.0,
            authority_policy_id="mujoco-single-human-authority-v1",
            initial_control_mode="human",
        ),
        clock=clock,
    )
    return build_scoped_cs2a2b_mujoco_human_station(
        runtime,
        SyntheticHumanQposSourceSpec(),
        station_clock=clock,
    )


def _policy() -> ScopedSimAuthorityAccessPolicy:
    return ScopedSimAuthorityAccessPolicy(
        access_policy_id="scoped-sim-authority:test-v1",
        bearer_token=_TOKEN,
    )


def _build_coordinator(station, event_log, clock):
    return SideEffectStationOperationCoordinator(
        coordinator_id="cs2a2b-web-side-effect-worker",
        runtime=station.station,
        adapters=(station.target_adapter, station.source_adapter),
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        event_log=event_log,
        queue_capacity=4,
        clock=clock,
    )


@contextmanager
def _running_server(station, coordinator, event_log):
    server = build_scoped_sim_authority_control_station_server(
        station.station,
        access_policy=_policy(),
        event_log=event_log,
        operation_coordinator=coordinator,
        authority_status_provider=station.authority_status_payload,
        port=0,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _payload(
    station,
    *,
    sequence: int,
    intent_kind: str,
    subject_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "client_id": "browser:scoped-sim-test",
        "client_sequence": sequence,
        "intent_kind": intent_kind,
        "produced_at": 1000.0 + sequence,
        "produced_clock_domain_id": "browser-performance",
        "expected_state_revision": station.station.state.state_revision,
        "manifest_hash": station.station.manifest.semantic_content_hash,
        "reason": "scoped-sim-http-test",
    }
    if subject_id is not None:
        payload["subject_id"] = subject_id
    return payload


def _post(root: str, payload: dict[str, object]):
    request = Request(
        f"{root}/api/intents",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        response = urlopen(request, timeout=2.0)  # noqa: S310 - loopback fixture
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _get(root: str, path: str) -> dict[str, object]:
    with urlopen(f"{root}{path}", timeout=2.0) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def test_scoped_policy_enables_authority_intents_but_rejects_action_payloads() -> None:
    station = _build_station(ManualClock())
    try:
        policy = _policy()
        authority = _payload(
            station,
            sequence=0,
            intent_kind="request_motion_authority",
            subject_id=station.source_adapter.subject_id,
        )
        draft = station_intent_draft_from_json(authority, access_policy=policy)
        assert draft.intent_kind == "request_motion_authority"

        authority["qpos"] = [0.0] * 12
        with pytest.raises(ValueError, match="unknown fields"):
            station_intent_draft_from_json(authority, access_policy=policy)
    finally:
        station.close()


def test_scoped_builder_does_not_widen_cs1c_or_accept_real_target() -> None:
    clock = ManualClock()
    station = _build_station(clock)
    event_log = BoundedStationEventLog(clock=clock)
    coordinator = _build_coordinator(station, event_log, clock)
    server = build_scoped_sim_authority_control_station_server(
        station.station,
        access_policy=_policy(),
        event_log=event_log,
        operation_coordinator=coordinator,
        authority_status_provider=station.authority_status_payload,
        port=0,
    )
    server.server_close()
    try:
        with pytest.raises(TypeError, match="NoIoStationOperationCoordinator"):
            build_coordinated_control_station_server(
                station.station,
                access_policy=LocalIntentAccessPolicy(
                    access_policy_id="cs1c-stays-no-io",
                    bearer_token="x" * 32,
                ),
                event_log=event_log,
                operation_coordinator=coordinator,  # type: ignore[arg-type]
            )

        real_binding = replace(
            station.station.manifest.execution_binding,
            target_kind="real",
            backend_inventory_hash="inventory:real:test",
        )
        real_manifest = replace(
            station.station.manifest,
            execution_binding=real_binding,
        )
        real_runtime = ControlStationRuntime(
            manifest=real_manifest,
            runtime_session_id="cs2a2b-real-builder-rejection",
            clock=clock,
        )
        real_coordinator = SideEffectStationOperationCoordinator(
            coordinator_id="cs2a2b-real-builder-rejection",
            runtime=real_runtime,
            adapters=(station.target_adapter, station.source_adapter),
            operation_prepare=lambda _transition: None,
            operation_executor=station.complete_pending_operation,
            event_log=event_log,
            clock=clock,
        )
        with pytest.raises(ValueError, match="exact sim target"):
            build_scoped_sim_authority_control_station_server(
                real_runtime,
                access_policy=_policy(),
                event_log=event_log,
                operation_coordinator=real_coordinator,
                authority_status_provider=station.authority_status_payload,
                port=0,
            )
    finally:
        station.close()


def test_scoped_http_requires_async_completion_before_publication() -> None:
    clock = ManualClock()
    station = _build_station(clock)
    event_log = BoundedStationEventLog(capacity=64, clock=clock)
    coordinator = _build_coordinator(station, event_log, clock)
    coordinator.start()
    target_id = station.station.manifest.execution_binding.binding_id
    source_id = station.source_adapter.subject_id
    operations = (
        ("acquire_control_lease", None),
        ("connect_target", target_id),
        ("connect_source", source_id),
        ("run_preflight", target_id),
        ("run_preflight", source_id),
        ("request_candidate", source_id),
        ("request_motion_authority", source_id),
    )
    try:
        with _running_server(station, coordinator, event_log) as root:
            initial = _get(root, "/api/state")
            assert initial["maturity"] == "cs2a2b_scoped_sim_authority_api"
            assert initial["target"]["motion_authorized"] is False

            for sequence, (intent_kind, subject_id) in enumerate(operations):
                status, body = _post(
                    root,
                    _payload(
                        station,
                        sequence=sequence,
                        intent_kind=intent_kind,
                        subject_id=subject_id,
                    ),
                )
                assert status == 200
                assert body["adapter_executed_inline"] is False
                if body["receipt"]["status"] == "operation_requested":
                    assert body["operation_enqueue"]["status"] == "operation_enqueued"
                assert coordinator.wait_until_idle(timeout_sec=1.0)

            active = _get(root, "/api/state")
            publication = active["control_facts"]["station_publication"]
            assert active["station"]["lifecycle_state"] == "running"
            assert active["target"]["motion_authorized"] is True
            assert active["capabilities"]["motion"] is False
            assert active["capabilities"]["motion_authority_requests"] is True
            assert "request_motion_authority" in (
                active["capabilities"]["allowed_intent_kinds"]
            )
            assert active["capabilities"]["browser_raw_action_submission"] is False
            assert active["control_facts"]["dispatch_owner"] == (
                publication["owner_token_id"]
            )
            assert active["control_facts"]["authority_epoch"] == (
                publication["authority_epoch"]
            )
            assert active["control_facts"]["execution_epoch"] == (
                publication["execution_epoch"]
            )
            assert publication["publication_enabled"] is True
            assert publication["authoritative"] is False

            raw_action = _payload(
                station,
                sequence=7,
                intent_kind="hold",
            )
            raw_action["action"] = {"left_arm_qpos": [0.0] * 6}
            status, body = _post(root, raw_action)
            assert status == 422
            assert body["error"] == "invalid_typed_intent"

            status, body = _post(
                root,
                _payload(
                    station,
                    sequence=7,
                    intent_kind="release_motion_authority",
                    subject_id=source_id,
                ),
            )
            assert status == 200
            assert body["adapter_executed_inline"] is False
            # prepare_pending_operation closes publication before queueing the
            # lower HOLD work; the HTTP response therefore cannot expose a
            # stale true projection even if the worker is still running.
            assert body["motion_authorized"] is False
            assert coordinator.wait_until_idle(timeout_sec=1.0)
            released = _get(root, "/api/state")
            assert released["target"]["motion_authorized"] is False
            assert released["station"]["lifecycle_state"] == "hold"
    finally:
        assert coordinator.stop(timeout_sec=1.0)
        station.close()
