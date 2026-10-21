"""Thread acquisition and completion receipts for resource-owning workers."""
from __future__ import annotations

import dis
import threading
import time


_STANDARD_START_CODE = threading.Thread.start.__code__


def _native_allocator_lines() -> frozenset[int]:
    line = _STANDARD_START_CODE.co_firstlineno
    result = set()
    for instruction in dis.get_instructions(_STANDARD_START_CODE):
        if instruction.starts_line is not None:
            line = instruction.starts_line
        if instruction.opname == "LOAD_GLOBAL" and instruction.argval in {
            "_start_new_thread", "_start_joinable_thread",
        }:
            result.add(line)
    return frozenset(result)


_ALLOCATOR_LINES = _native_allocator_lines()


def _allocation_definitely_failed(thread: threading.Thread, error: BaseException) -> bool:
    # Only the native allocator's failure at the standard start call is a
    # no-acquisition receipt. An absent ident alone may mean delayed bootstrap.
    # Unknown runtimes, wrapper errors, wait errors and interrupts stay unknown.
    if (type(error) is not RuntimeError or str(error) != "can't start new thread"
            or thread.ident is not None):
        return False
    traceback = error.__traceback__
    while traceback is not None:
        if (traceback.tb_frame.f_code is _STANDARD_START_CODE
                and traceback.tb_lineno in _ALLOCATOR_LINES):
            return True
        traceback = traceback.tb_next
    return False


class OwnedThread(threading.Thread):
    """One launch attempt, with completion independent of Thread.is_alive()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._launch_attempted = False
        self._launch_lock = threading.Lock()
        self._launch_abandoned = False
        self._allocation_failed = False
        self._runner_entered = threading.Event()
        self._runner_completed = threading.Event()

    def start(self) -> None:
        with self._launch_lock:
            if self._launch_abandoned:
                raise RuntimeError("owned thread launch was abandoned during close")
            if self._launch_attempted:
                raise RuntimeError("owned thread starts once")
            self._launch_attempted = True
        try:
            super().start()
        except BaseException as error:
            self._allocation_failed = _allocation_definitely_failed(self, error)
            raise

    def run(self) -> None:
        self._runner_entered.set()
        try:
            super().run()
        finally:
            self._runner_completed.set()

    def join_and_confirm(self, *, timeout: float) -> bool:
        """Confirm no launch or a completed target followed by normal join."""

        with self._launch_lock:
            if not self._launch_attempted:
                # Atomically cancel acquisition. Every existing consumer is
                # safe even if start() races after this no-launch receipt.
                self._launch_abandoned = True
                return True
        if self._allocation_failed:
            return True
        if self is threading.current_thread():
            return False
        deadline = time.monotonic() + max(0.0, timeout)
        if not self._runner_entered.wait(max(0.0, deadline - time.monotonic())):
            return False
        # A Python 3.10 interrupted join can mark a live thread stopped. The
        # independent target-finally receipt must survive every retry as well.
        self.join(timeout=max(0.0, deadline - time.monotonic()))
        return self._runner_completed.is_set() and not self.is_alive()
