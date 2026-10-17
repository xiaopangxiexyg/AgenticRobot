"""Offline partial-acquisition faults: fake backends and localhost UDP only."""
from __future__ import annotations

import socket
import threading
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real.exclusive_application import (
    ExclusiveRealSkillApplication,
    ExclusiveRealSkillApplicationConfig,
    ExclusiveRealSkillApplicationState,
)
from hardware.rm65.realtime_feedback import RM65UDPFeedbackReceiver
from teleop_runtime.position_force_guard import PositionForceGuardConfig, PositionForceGuardMonitor
from teleop_runtime.skill_deploy.control_runtime import RealSkillControlLifecycle
from tests.test_agent_gateway_real_exclusive_factory import (
    _Resources, _Server, _deployment, _fresh_observation_cut,
)
from tests.test_position_force_guard import _payload
from tests.test_real_skill_deploy_rm65_ports import _ports


def _owner(kind):
    if kind == "application":
        resources, server = _Resources(), _Server()
        app = ExclusiveRealSkillApplication(
            deployment=_deployment(observation_provider=_fresh_observation_cut, hold_reasons=[]),
            resources=resources, startup_admission=lambda: {"ready": True},
            server_factory=lambda: server,
            config=ExclusiveRealSkillApplicationConfig(control_join_timeout_sec=.01),
        )
        return SimpleNamespace(value=app, start=app.start, close=app.close,
            name="exclusive-real-skill-control", thread=lambda: app._control_thread,
            closed=lambda: resources.closes == 1,
            retained=lambda: resources.closes == 0)
    if kind == "sampler":
        ports, backends, _ = _ports(background_measurement_sampling=True)
        return SimpleNamespace(value=ports, start=ports.start, close=ports.close,
            name="exclusive-rm65-measurement-sampler", thread=lambda: ports._measurement_thread,
            closed=lambda: ports.resources_closed and all(b.closes == 1 for b in backends.values()),
            retained=lambda: all(b.closes == 0 for b in backends.values()))
    if kind in {"watchdog", "poller"}:
        guard = PositionForceGuardMonitor(
            readers_by_side={"left": _payload, "right": _payload},
            emergency_stop=lambda fault: None,
            config=PositionForceGuardConfig(poll_hz=20., startup_timeout_sec=.5),
        )
        name = "rm65_position_force_guard_" + ("watchdog" if kind == "watchdog" else "left")
        def thread():
            if kind == "watchdog":
                return guard._watchdog_thread
            return next((t for t in guard._threads if t.name == name), None)
        return SimpleNamespace(value=guard, start=guard.start, close=guard.close, name=name,
            thread=thread, closed=lambda: not guard._threads and guard._watchdog_thread is None,
            retained=lambda: thread() is not None)
    assert kind == "udp"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    receiver = RM65UDPFeedbackReceiver(arm_ip_by_side={"left": "127.0.0.1", "right": "127.0.0.2"},
        port=port, bind_ip="127.0.0.1")
    return SimpleNamespace(value=receiver, start=receiver.start, close=receiver.close,
        name="rm65-udp-feedback", thread=lambda: receiver._thread,
        closed=lambda: receiver._closed and receiver._socket is None,
        retained=lambda: receiver._socket is not None and not receiver._closed)


def _emergency_test_cleanup(owner):
    """Also release old-code test leftovers after a deliberate negative proof."""
    value = owner.value
    for name in ("_control_stop", "_measurement_stop", "_stop_event", "_stop"):
        event = getattr(value, name, None)
        if event is not None:
            event.set()
    threads = list(getattr(value, "_threads", ()))
    threads += [getattr(value, name, None) for name in (
        "_control_thread", "_measurement_thread", "_watchdog_thread", "_thread")]
    for thread in threads:
        if thread is not None and thread.ident is not None:
            threading.Thread.join(thread, timeout=2.)
    executor = getattr(value, "_measurement_refresh_executor", None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    receiver = getattr(value, "_socket", None)
    if receiver is not None:
        receiver.close()


@pytest.mark.parametrize("kind", ["application", "sampler", "watchdog", "poller", "udp"])
def test_native_thread_allocation_failure_can_finish_owned_cleanup(kind, monkeypatch):
    owner = _owner(kind)
    allocate = threading._start_new_thread
    def fail_native_allocator(function, args, *rest, **kwargs):
        thread = getattr(function, "__self__", None)
        if thread is not None and thread.name == owner.name:
            raise RuntimeError("can't start new thread")
        return allocate(function, args, *rest, **kwargs)
    monkeypatch.setattr(threading, "_start_new_thread", fail_native_allocator)
    try:
        with pytest.raises(RuntimeError, match="can't start new thread"):
            owner.start()
        owner.close()
        assert owner.closed()
        owner.close()
        assert owner.closed()
    finally:
        _emergency_test_cleanup(owner)


@pytest.mark.parametrize("kind", ["application", "sampler", "watchdog", "poller", "udp"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_unknown_thread_start_retains_dependencies_until_late_bootstrap(kind, failure, monkeypatch):
    owner = _owner(kind)
    original_start = threading.Thread.start
    def unknown_start(thread):
        if thread.name == owner.name:
            raise failure("unconfirmed thread launch")
        return original_start(thread)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", unknown_start)
            with pytest.raises(failure, match="unconfirmed thread launch"):
                owner.start()
        thread = owner.thread()
        assert thread is not None
        with pytest.raises(RuntimeError):
            owner.close()
        assert owner.retained()
        assert owner.thread() is thread
        # Model an already acquired native thread whose bootstrap was delayed.
        original_start(thread)
        threading.Thread.join(thread, timeout=2.)
        owner.close()
        assert owner.closed()
    finally:
        _emergency_test_cleanup(owner)


@pytest.mark.parametrize("kind", ["application", "sampler", "watchdog", "poller", "udp"])
def test_exception_after_thread_launch_keeps_and_drains_that_worker(kind, monkeypatch):
    owner = _owner(kind)
    original_start = threading.Thread.start
    seen = []
    def start_then_raise(thread):
        result = original_start(thread)
        if thread.name == owner.name:
            seen.append(thread)
            raise RuntimeError("failure after native launch")
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", start_then_raise)
            with pytest.raises(RuntimeError, match="failure after native launch"):
                owner.start()
        owner.close()
        assert owner.closed()
        assert len(seen) == 1 and not seen[0].is_alive()
    finally:
        _emergency_test_cleanup(owner)


@pytest.mark.parametrize("stage", ["start", "wait_ready"])
@pytest.mark.parametrize("failed_owner", ["left", "right", "udp"])
@pytest.mark.parametrize("failures", [1, 2])
def test_udp_pre_sdk_failure_cleanup_retries_never_manufacture_hold(stage, failed_owner, failures):
    ports, backends, _ = _ports(backend_close_failures={failed_owner: failures})
    pending = [failures if failed_owner == "udp" else 0]
    def fail():
        raise RuntimeError("feedback startup failed")
    def close_udp():
        if pending[0]:
            pending[0] -= 1
            raise RuntimeError("UDP close failed")
    ports.udp_feedback = SimpleNamespace(start=fail if stage == "start" else lambda: None,
        wait_ready=(lambda **kw: fail()) if stage == "wait_ready" else lambda **kw: None,
        close=close_udp, diagnostics=lambda: {"transport": "udp"})
    with pytest.raises(RuntimeError, match="startup rollback incomplete"):
        ports.start()
    assert all(b.starts == 0 and b.holds == [] for b in backends.values())
    for _ in range(4):
        try:
            ports.close()
        except RuntimeError:
            assert not ports.resources_closed
        assert all(b.starts == 0 and b.holds == [] for b in backends.values())
        assert ports._shutdown_hold_attempted is False
        assert ports._shutdown_hold_confirmed is False
        if ports.resources_closed:
            break
    assert ports.resources_closed
    assert ports.startup_hold_receipt is None


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_never_started_application_fences_gateway_and_retries_owned_store_cleanup(failures):
    class Resources(_Resources):
        def start(self):
            super().start()
            raise RuntimeError("resource acquisition failed")
    resources = Resources()
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=[])
    closes = []
    def close_store():
        closes.append(1)
        if len(closes) <= failures:
            raise RuntimeError("store close failed")
    session = deployment.gateway_session
    session.artifact_store = SimpleNamespace(close=close_store)
    session.owns_artifact_store = True
    app = ExclusiveRealSkillApplication(deployment=deployment, resources=resources,
        startup_admission=lambda: pytest.fail("startup admission must remain unacquired"),
        server_factory=lambda: pytest.fail("transport must remain unacquired"))
    with pytest.raises(RuntimeError, match="resource acquisition failed"):
        app.start()
    assert session.service.is_closed
    assert deployment.control_runtime.lifecycle == RealSkillControlLifecycle.CREATED
    assert deployment.control_runtime.last_hold_receipt is None
    assert session._close_outcome is None
    assert resources.closes == (0 if failures else 1)
    for _ in range(failures + 1):
        try:
            app.close()
        except RuntimeError as error:
            assert "store close failed" in str(error)
            assert resources.closes == 0
        if app.state == ExclusiveRealSkillApplicationState.CLOSED:
            break
    assert app.state == ExclusiveRealSkillApplicationState.CLOSED
    assert resources.closes == 1
    assert len(closes) == failures + 1
    assert session._artifact_store_closed
    assert session._close_outcome is None


@pytest.mark.parametrize("prior_fact", ["admitted", "runtime_started", "server_created"])
def test_unstarted_cleanup_refuses_inconsistent_acquisition_facts(prior_fact):
    resources, holds = _Resources(), []
    deployment = _deployment(observation_provider=_fresh_observation_cut, hold_reasons=holds)
    app = ExclusiveRealSkillApplication(deployment=deployment, resources=resources,
        startup_admission=lambda: {"ready": True}, server_factory=_Server)
    if prior_fact == "admitted":
        # Monotonic total must protect even terminal or evicted work.
        deployment.gateway_session.service.operation_store._total_admitted = 1
    elif prior_fact == "runtime_started":
        deployment.control_runtime.start()
    else:
        app._server = _Server()
    with pytest.raises(RuntimeError):
        app.close()
    assert app.state == ExclusiveRealSkillApplicationState.FAULT
    assert resources.closes == 0
    if prior_fact == "admitted":
        assert deployment.gateway_session.service.is_closed


def test_owned_worker_completion_is_independent_of_poisoned_thread_liveness(monkeypatch):
    from utils.owned_thread import OwnedThread
    entered, release = threading.Event(), threading.Event()
    def run():
        entered.set()
        assert release.wait(3.)
    thread = OwnedThread(target=run)
    thread.start()
    assert entered.wait(1.)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(thread, "join", lambda **kw: None)
            patch.setattr(thread, "is_alive", lambda: False)
            assert thread.join_and_confirm(timeout=.01) is False
    finally:
        release.set()
        thread.join(timeout=2.)
    assert thread.join_and_confirm(timeout=.01) is True


def test_owned_worker_join_interruption_retains_completion_proof(monkeypatch):
    from utils.owned_thread import OwnedThread
    release = threading.Event()
    thread = OwnedThread(target=lambda: release.wait(3.))
    thread.start()
    def interrupted(**kwargs):
        raise KeyboardInterrupt("join interrupted")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(thread, "join", interrupted)
            with pytest.raises(KeyboardInterrupt, match="join interrupted"):
                thread.join_and_confirm(timeout=.01)
        assert thread.join_and_confirm(timeout=.01) is False
    finally:
        release.set()
        thread.join(timeout=2.)
    assert thread.join_and_confirm(timeout=.01) is True
