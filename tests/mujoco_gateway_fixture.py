"""Reuse compiled inputs, never mutable models or live Gateway resources."""

from functools import lru_cache
import os
import tempfile
import threading

import pytest


_construction_lock = threading.RLock()


@lru_cache(maxsize=1)
def _private_cache_directory():
    # Keep direct wrapper invocations isolated too; no persistent repo cache.
    return tempfile.TemporaryDirectory(prefix="ar-gateway-models-", dir="/tmp")


def build_gateway_session(factory, *, use_model_cache=True, **parameters):
    """Bind the existing artifact loader only during synchronous construction."""
    if not use_model_cache:
        return factory(**parameters)

    from sim.mujoco.artifacts.cache import MODEL_CACHE_ENV, ModelArtifactCache
    from sim.mujoco.scene import MujocoScene

    root = os.environ.get(MODEL_CACHE_ENV, "").strip() or _private_cache_directory().name
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
        return factory(**parameters)
