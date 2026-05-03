"""
Lightweight MANO pickle loader for portable visualization.

This keeps the original module path expected by ``manotorch.manolayer`` but only
implements the subset needed to reconstruct meshes for rendering.
"""

from __future__ import annotations

import pickle
import sys
import types
import warnings
from typing import Any, Dict

import numpy as np


class _PickleCh:
    def __init__(self, *args, **kwargs):
        self._state: Any = None

    def __setstate__(self, state: Any) -> None:
        self._state = state


class _PickleSelect(_PickleCh):
    pass


def _load_pickle_py3_compatible(path: str) -> Dict[str, Any]:
    chumpy_mod = types.ModuleType("chumpy")
    chumpy_ch_mod = types.ModuleType("chumpy.ch")
    chumpy_reordering_mod = types.ModuleType("chumpy.reordering")

    chumpy_mod.Ch = _PickleCh
    chumpy_mod.ch = chumpy_ch_mod
    chumpy_mod.reordering = chumpy_reordering_mod
    chumpy_ch_mod.Ch = _PickleCh
    chumpy_reordering_mod.Select = _PickleSelect

    previous = {
        "chumpy": sys.modules.get("chumpy"),
        "chumpy.ch": sys.modules.get("chumpy.ch"),
        "chumpy.reordering": sys.modules.get("chumpy.reordering"),
    }
    sys.modules["chumpy"] = chumpy_mod
    sys.modules["chumpy.ch"] = chumpy_ch_mod
    sys.modules["chumpy.reordering"] = chumpy_reordering_mod
    try:
        with open(path, "rb") as file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                if hasattr(np, "exceptions") and hasattr(np.exceptions, "VisibleDeprecationWarning"):
                    warnings.simplefilter("ignore", category=np.exceptions.VisibleDeprecationWarning)
                return pickle.load(file, encoding="latin1")
    finally:
        for module_name, module_value in previous.items():
            if module_value is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module_value


def _unwrap_chumpy_like(value: Any) -> Any:
    if isinstance(value, _PickleSelect):
        state = value._state if isinstance(value._state, dict) else {}
        base = np.asarray(_unwrap_chumpy_like(state.get("a")))
        idxs = np.asarray(state.get("idxs", []), dtype=np.int64).reshape(-1)
        preferred_shape = tuple(int(dim) for dim in state.get("preferred_shape", (idxs.size,)))
        if idxs.size == 0:
            return base
        return base.reshape(-1)[idxs].reshape(preferred_shape)
    if isinstance(value, _PickleCh):
        state = value._state if isinstance(value._state, dict) else {}
        if "x" in state:
            return np.asarray(state["x"])
        if "a" in state:
            return _unwrap_chumpy_like(state["a"])
        return state
    return value


def ready_arguments(fname_or_dict, posekey4vposed: str = "pose") -> Dict[str, Any]:
    del posekey4vposed
    if isinstance(fname_or_dict, dict):
        dd = dict(fname_or_dict)
    else:
        dd = _load_pickle_py3_compatible(str(fname_or_dict))

    if "shapedirs" in dd:
        dd["shapedirs"] = np.asarray(_unwrap_chumpy_like(dd["shapedirs"]))
    if "betas" not in dd and "shapedirs" in dd:
        dd["betas"] = np.zeros(int(dd["shapedirs"].shape[-1]), dtype=np.float32)
    return dd


def load_model(*args, **kwargs):
    raise NotImplementedError("portable_vis only supports ready_arguments() for MANO visualization.")
