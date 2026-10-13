from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.observation_gateway import (
    assemble_real_tcp_pose_observe_gateway_session,
)
from agent_gateway.profiles.real.readonly_tcp import (
    ReadonlyRM65TcpSource,
    build_readonly_rm65_tcp_contract,
)
from teleop_runtime.skill_deploy import rm65_readonly as lower


@pytest.fixture
def profile():
    station = np.eye(4)
    station[0, 3] = 1.
    return SimpleNamespace(
        arm_ip_by_side={"left": "left.fixture", "right": "right.fixture"},
        scene_world_from_canonical_station=station,
        scene_world_from_arm_base_by_side={"left": np.eye(4), "right": np.eye(4)},
        collision_scene_profile="dual_arm_hand_proxy_dmtac_strict",
        station_world_frame_id="dual_arm_solver_world",
        tool_profile_by_side={"left": {"name": "left-tool"}, "right": {"name": "right-tool"}},
        tool_tcp_pose_flange_m_rad_by_side={"left": np.zeros(6), "right": np.zeros(6)},
    )


class FakeFK:
    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        assert np.allclose(left_qpos, np.deg2rad([1, 2, 3, 4, 5, 6]))
        assert np.allclose(right_qpos, np.deg2rad([1, 2, 3, 4, 5, 6]))
        result = {"left": np.eye(4), "right": np.eye(4)}
        result["left"][0, 3] = 1.4
        result["right"][0, 3] = 1.6
        return result


class FakeArm:
    def __init__(self, *, status=0, degrees=(1, 2, 3, 4, 5, 6)):
        self.status = status
        self.degrees = degrees
        self.reads = 0
        self.closes = 0

    def Get_Current_Arm_State(self, retry=0):
        assert retry == 0
        self.reads += 1
        return self.status, self.degrees, [999] * 6, 0, 0

    def Arm_Socket_Close(self):
        self.closes += 1
        return 0

    def __getattr__(self, name):
        raise AssertionError(f"Forbidden SDK call: {name}")


def test_sampler_publishes_fk_without_command_owner_and_consumers_do_no_io(profile):
    arms = {ip: FakeArm() for ip in profile.arm_ip_by_side.values()}
    sampler = lower.RM65ReadonlyTcpSampler(profile, arm_factory=arms.__getitem__, target_builder=FakeFK())
    bundle = build_readonly_rm65_tcp_contract(profile, runtime_session_id="readonly:test")
    source = ReadonlyRM65TcpSource(sampler, bundle)
    session = assemble_real_tcp_pose_observe_gateway_session(
        contract_bundle=bundle, source=source, allowed_client_ids=("test",)
    )
    try:
        sampler.start()
        snapshot = session.observation_producer.capture_snapshot(active_operation_id=None)
        assert snapshot.lifecycle_state == "readonly_sampling"
        assert snapshot.execution_gate == "unavailable_no_command_owner"
        assert snapshot.authority_epoch == snapshot.execution_epoch == snapshot.dispatch_owner_generation == 0
        assert bundle.route_manifest.current_mode == "real_observe"
        assert len(bundle.route_manifest.skill_routes) == 1
        assert snapshot.measurement_summary["pose_xyz_quat_xyzw_by_component"]["left_arm"] == pytest.approx([.4, 0, 0, 0, 0, 0, 1])
        cut = session.observation_producer.read(active_operation_id="read:1", parameters={
            "component_ids": ["right_arm"], "reference_frame_id": "canonical_station",
            "controlled_frame_id": "tool_tcp", "max_measurement_age_ms": 200,
        })
        assert cut.execution_result["pose_by_component"]["right_arm"] == pytest.approx([.6, 0, 0, 0, 0, 0, 1])
        assert set(cut.execution_result["pose_by_component"]) == {"right_arm"}
        assert source.buffer.latest_publication.owner_kind == "readonly_sdk_sampler"
        assert all(arm.reads >= 1 for arm in arms.values())
    finally:
        session.close()
        sampler.close()
    assert all(arm.closes == 1 for arm in arms.values())
    with pytest.raises(RuntimeError, match="not publishing"):
        _ = sampler.latest


@pytest.mark.parametrize("status,degrees", [(1, [1]*6), (0, [1]*5), (0, [float("nan")]*6)])
def test_failed_sdk_state_never_publishes_or_retries(profile, status, degrees):
    arms = [FakeArm(status=status, degrees=degrees), FakeArm()]
    sampler = lower.RM65ReadonlyTcpSampler(profile, arm_factory=lambda ip: arms[0 if ip.startswith("left") else 1], target_builder=FakeFK())
    try:
        with pytest.raises(RuntimeError, match="sampler failed"):
            sampler.start()
        assert arms[0].reads == 1 and arms[1].reads == 0
        with pytest.raises(RuntimeError, match="sampler failed"):
            _ = sampler.latest
    finally:
        sampler.close()
    assert [arm.closes for arm in arms] == [1, 1]


def test_slow_sdk_time_counts_in_measurement_age(profile):
    ticks = iter([0., .15, .15, .25, .251])
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), clock=lambda: next(ticks))
    with pytest.raises(RuntimeError, match="200ms"):
        sampler._sample({"left": FakeArm(), "right": FakeArm()}, 1)


def test_partial_connect_failure_closes_only_owned_handle(profile):
    arm = FakeArm()
    def connect(ip):
        if ip.startswith("right"):
            raise RuntimeError("connect failed")
        return arm
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), arm_factory=connect)
    try:
        with pytest.raises(RuntimeError, match="sampler failed"):
            sampler.start()
    finally:
        sampler.close()
    assert arm.closes == 1 and arm.reads == 0


def test_stale_cache_is_rejected_without_refreshing_sdk(profile):
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), clock=lambda: 1.)
    sampler._latest = sampler._sample({"left": FakeArm(), "right": FakeArm()}, 1)
    bundle = build_readonly_rm65_tcp_contract(profile, runtime_session_id="readonly:test")
    source = ReadonlyRM65TcpSource(sampler, bundle, clock=lambda: 1.3)
    session = assemble_real_tcp_pose_observe_gateway_session(contract_bundle=bundle, source=source, allowed_client_ids=("test",))
    with pytest.raises(RuntimeError, match="stale"):
        session.observation_producer.capture_snapshot(active_operation_id=None)


def test_close_never_deletes_inflight_sdk_handle(profile, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    class BlockingArm(FakeArm):
        def Get_Current_Arm_State(self, retry=0):
            entered.set()
            assert release.wait(3)
            return super().Get_Current_Arm_State(retry)
    arm = BlockingArm()
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), arm_factory=lambda _: arm)
    sampler._thread = threading.Thread(target=sampler._run)
    sampler._thread.start()
    assert entered.wait(1)
    original_join = sampler._thread.join
    monkeypatch.setattr(sampler._thread, "join", lambda timeout: original_join(timeout=.01))
    try:
        with pytest.raises(RuntimeError, match="in-flight"):
            sampler.close()
        assert arm.closes == 0
    finally:
        release.set()
        original_join(timeout=2)
    sampler.close()


def test_connection_occupancy_fails_before_sdk(profile, monkeypatch):
    monkeypatch.setattr(lower.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="local:123 left.fixture:8080\n"))
    with pytest.raises(RuntimeError, match="existing RM65 connection"):
        lower.assert_readonly_rm65_connections_idle(profile.arm_ip_by_side)


@pytest.mark.parametrize("ticks", [
    [float("nan"), float("nan"), 1., 1., 1.],
    [0., .1, .05, .1, .15],
    [0., .01, .01, .02, float("nan")],
])
def test_invalid_or_regressed_sdk_clock_never_produces_a_cut(profile, ticks):
    clock = iter(ticks)
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), clock=lambda: next(clock))
    with pytest.raises(RuntimeError, match="clock|invalid time"):
        sampler._sample({"left": FakeArm(), "right": FakeArm()}, 1)


def test_sdk_measurement_clock_cannot_regress_across_samples(profile):
    clock = iter([1., 1.01, 1.01, 1.02, 1.03, .99, 1., 1., 1.01, 1.04])
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), clock=lambda: next(clock))
    sampler._latest = sampler._sample({"left": FakeArm(), "right": FakeArm()}, 1)
    with pytest.raises(RuntimeError, match="clock"):
        sampler._sample({"left": FakeArm(), "right": FakeArm()}, 2)


def test_start_timeout_cancels_pending_connection_and_latches_failure(profile, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    arm = FakeArm()

    def connect(ip):
        entered.set()
        assert release.wait(2)
        calls.append(ip)
        return arm

    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), arm_factory=connect)
    monkeypatch.setattr(sampler._ready, "wait", lambda timeout: entered.wait(1) and False)
    try:
        with pytest.raises(TimeoutError):
            sampler.start()
        assert sampler._stop.is_set()
        with pytest.raises(RuntimeError, match="sampler failed"):
            _ = sampler.latest
    finally:
        release.set()
        sampler.close()
    assert calls == [profile.arm_ip_by_side["left"]]
    assert arm.reads == 0 and arm.closes == 1


def test_close_during_first_sdk_read_does_not_begin_second_read(profile, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockingArm(FakeArm):
        def Get_Current_Arm_State(self, retry=0):
            entered.set()
            assert release.wait(2)
            return super().Get_Current_Arm_State(retry)

    left, right = BlockingArm(), FakeArm()
    sampler = lower.RM65ReadonlyTcpSampler(
        profile, target_builder=FakeFK(), arm_factory=lambda ip: left if ip.startswith("left") else right,
    )
    sampler._thread = threading.Thread(target=sampler._run)
    sampler._thread.start()
    assert entered.wait(1)
    join = sampler._thread.join
    monkeypatch.setattr(sampler._thread, "join", lambda timeout: join(timeout=.01))
    try:
        with pytest.raises(RuntimeError, match="in-flight"):
            sampler.close()
        assert left.closes == right.closes == 0
    finally:
        release.set()
        join(1)
    sampler.close()
    assert left.reads == 1 and right.reads == 0
    assert left.closes == right.closes == 1
    assert sampler._latest is None


def test_worker_start_failure_remains_observable_and_can_close(profile, monkeypatch):
    calls = []
    sampler = lower.RM65ReadonlyTcpSampler(profile, target_builder=FakeFK(), arm_factory=lambda ip: calls.append(ip))

    def fail_start(thread):
        raise RuntimeError("worker start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="worker start failed"):
        sampler.start()
    assert sampler._stop.is_set()
    with pytest.raises(RuntimeError, match="sampler failed"):
        _ = sampler.latest
    sampler.close()
    assert calls == []


def test_sampler_start_checks_occupancy_before_creating_a_worker(profile, monkeypatch):
    calls = []

    def occupied(ips):
        assert ips == profile.arm_ip_by_side
        raise RuntimeError("existing RM65 connection")

    monkeypatch.setattr(lower, "assert_readonly_rm65_connections_idle", occupied)
    sampler = lower.RM65ReadonlyTcpSampler(
        profile, target_builder=FakeFK(), arm_factory=lambda ip: calls.append(ip),
    )
    with pytest.raises(RuntimeError, match="existing RM65 connection"):
        sampler.start()
    assert sampler._thread is None and calls == []
    assert sampler._stop.is_set()
    sampler.close()
