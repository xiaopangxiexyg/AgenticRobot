"""Cached native fixture ownership, without constructing a simulator."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
from types import SimpleNamespace
import weakref

import pytest

import mujoco_gateway_fixture as fixture
from sim.mujoco.scene import MujocoScene


class _Session:
    def __init__(self, application, failure=None):
        self.application, self.failure = application, failure

    def close(self):
        self.application.closed = True
        if self.failure == "raise":
            raise OSError("injected cleanup failure after application close")
        return SimpleNamespace(accepted=self.failure != "reject")


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_MUJOCO_CACHE_DIR", str(tmp_path / "model-cache"))


def test_cached_scope_collects_only_after_closed_applications_are_released(cache_root, monkeypatch):
    calls = []
    class Application:
        closed = False
    application = Application()
    alive = weakref.ref(application)

    def collect():
        assert alive() is None
        calls.append(True)

    monkeypatch.setattr(fixture.gc, "collect", collect)
    with fixture.gateway_session_test_scope():
        session = fixture.build_gateway_session(lambda: _Session(application))
        assert session.application is application and not calls
        assert session.close().accepted
        assert not calls
        del session, application
        assert alive() is not None  # The scope owns it until teardown.
    assert calls == [True]


def test_cached_scope_rejects_unclosed_application_without_collecting(cache_root, monkeypatch):
    calls = []
    monkeypatch.setattr(fixture.gc, "collect", lambda: calls.append(True))
    application = SimpleNamespace(closed=False)
    with pytest.raises(AssertionError, match="not closed before test teardown"):
        with fixture.gateway_session_test_scope():
            session = fixture.build_gateway_session(lambda: _Session(application))
    assert not application.closed and not calls
    assert session.close().accepted


@pytest.mark.parametrize("failure", ("raise", "reject"))
def test_application_flag_is_insufficient_when_full_session_close_did_not_succeed(cache_root, monkeypatch, failure):
    calls = []
    monkeypatch.setattr(fixture.gc, "collect", lambda: calls.append(True))
    application = SimpleNamespace(closed=False)
    with pytest.raises(AssertionError, match="not closed before test teardown"):
        with fixture.gateway_session_test_scope():
            session = fixture.build_gateway_session(lambda: _Session(application, failure))
            if failure == "raise":
                with pytest.raises(OSError, match="injected cleanup failure"):
                    session.close()
            else:
                assert not session.close().accepted
            assert application.closed
    assert not calls
    session.failure = None
    assert session.close().accepted


def test_empty_cached_scope_does_not_collect(cache_root, monkeypatch):
    calls = []
    monkeypatch.setattr(fixture.gc, "collect", lambda: calls.append(True))
    with fixture.gateway_session_test_scope():
        pass
    assert not calls


def test_unmanaged_thread_uses_original_factory_even_with_cache_root(cache_root):
    original = inspect.getattr_static(MujocoScene, "from_contract")
    marker = object()

    def factory(*, value):
        assert inspect.getattr_static(MujocoScene, "from_contract") is original
        return value

    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(fixture.build_gateway_session, factory, value=marker).result(timeout=2)
    assert result is marker


def test_uncached_construction_waits_until_cached_port_is_restored(cache_root):
    original = inspect.getattr_static(MujocoScene, "from_contract")
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()

    def cached():
        with fixture.gateway_session_test_scope():
            def factory():
                assert inspect.getattr_static(MujocoScene, "from_contract") is not original
                entered.set()
                assert release.wait(timeout=2)
                return _Session(SimpleNamespace(closed=False))
            session = fixture.build_gateway_session(factory)
            assert session.close().accepted

    def uncached():
        attempted.set()
        def factory():
            assert inspect.getattr_static(MujocoScene, "from_contract") is original
            return "original"
        return fixture.build_gateway_session(factory)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(cached)
        try:
            assert entered.wait(timeout=2)
            second = workers.submit(uncached)
            assert attempted.wait(timeout=2)
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=2)
        assert second.result(timeout=2) == "original"
