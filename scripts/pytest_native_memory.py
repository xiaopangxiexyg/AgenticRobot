"""Return unused native heap pages between complete-acceptance test modules."""
from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator

import pytest


def _load_malloc_trim():
    if sys.platform != "linux":
        return None
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        # Non-glibc platforms keep the same tests without this optimization.
        return None
    trim.argtypes = (ctypes.c_size_t,)
    trim.restype = ctypes.c_int
    return trim


_MALLOC_TRIM = _load_malloc_trim()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None,
) -> Iterator[None]:
    outcome = yield
    outcome.get_result()
    if _MALLOC_TRIM is None or (nextitem is not None and nextitem.path == item.path):
        return
    # All current-item teardown has finished. This returns already-free pages;
    # it does not collect Python objects, clear caches, or close live resources.
    _MALLOC_TRIM(128 * 1024)
