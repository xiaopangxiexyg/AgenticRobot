"""Reuse compiled inputs, never mutable models or live Gateway resources."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import gc
import os
import threading

import pytest


_construction_lock = threading.RLock()
_sessions = ContextVar("cached_gateway_sessions", default=None)
_INHERITED_CLOSE = object()


@dataclass
class _TrackedSession:
    session: object
    original_close: object
    instance_close: object
    completed: bool = False


@contextmanager
def gateway_session_test_scope():
    """Collect closed cached sessions after the owning test's teardown only."""
    entries = []
    token = _sessions.set(entries)
    try:
        yield
    finally:
        _sessions.reset(token)
        unclosed = sum(not entry.completed or not entry.session.application.closed for entry in entries)
        if unclosed:
            # Keep the references in the failure traceback; never close or
            # collect an application on behalf of a test that left it live.
            raise AssertionError(f"{unclosed} cached native application(s) not closed before test teardown")
        if entries:
            for entry in entries:
                if entry.instance_close is _INHERITED_CLOSE:
                    entry.session.__dict__.pop("close", None)
                else:
                    entry.session.close = entry.instance_close
            del entry
            entries.clear()
            gc.collect()


def build_gateway_session(factory, *, use_model_cache=True, **parameters):
    """Use an explicitly configured Run cache during synchronous construction."""
    if not use_model_cache:
        with _construction_lock:
            return factory(**parameters)

    from sim.mujoco.artifacts.cache import MODEL_CACHE_ENV, ModelArtifactCache
    root = os.environ.get(MODEL_CACHE_ENV, "").strip()
    entries = _sessions.get()
    if not root or entries is None:
        # Plain focused and higher-scope/unmanaged callers keep the original
        # path; every cached application must have a test teardown owner.
        with _construction_lock:
            return factory(**parameters)

    from sim.mujoco.scene import MujocoScene

    cache = ModelArtifactCache(root)

    with _construction_lock, pytest.MonkeyPatch.context() as patch:
        compile_scene = MujocoScene.from_contract

        def fresh_scene(cls, contract, **profiles):
            # Artifact misses compile to an explicit path through the original
            # method. Do not recursively cache or redirect that publication.
            if "mjcf_path" in profiles:
                return compile_scene(contract, **profiles)
            scene, _artifact = cache.load_scene(contract, **profiles)
            return scene

        patch.setattr(MujocoScene, "from_contract", classmethod(fresh_scene))
        session = factory(**parameters)
    entry = _TrackedSession(session, session.close, session.__dict__.get("close", _INHERITED_CLOSE))
    entries.append(entry)

    @wraps(entry.original_close)
    def observed_close(*args, **kwargs):
        # A returned accepted outcome proves the entire close completed. The
        # application flag alone can already be true when later cleanup fails.
        result = entry.original_close(*args, **kwargs)
        entry.completed = entry.completed or bool(result.accepted)
        return result

    session.close = observed_close
    return session
