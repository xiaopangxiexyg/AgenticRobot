"""Offline UDP ownership faults, preserving the fixed source cleanup checks."""
from __future__ import annotations

import socket
import threading
from types import SimpleNamespace

import pytest

from hardware.rm65.realtime_feedback import RM65UDPFeedbackReceiver
from tests.test_real_skill_deploy_rm65_ports import _ports


def _owner(kind):
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



@pytest.mark.parametrize("kind", ["udp"])
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



@pytest.mark.parametrize("kind", ["udp"])
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



@pytest.mark.parametrize("kind", ["udp"])
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
