"""Runtime receipt for one sample accepted by the authoritative writer.

Acceptance is an admission boundary, not a durability or Episode-seal
receipt.  It means the sample contract was validated, the writer cursor
advanced, and every required write job was admitted by the writer that owns
the Episode.  Applications may use this receipt to align another runtime
state transition to the first formally accepted sample.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading


def _non_negative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class AcceptedSampleReceipt:
    """Exact sample coordinate at the authoritative writer admission boundary."""

    episode_index: int
    sample_index: int
    episode_frame_index: int
    sample_timestamp: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_index",
            _non_negative_int(self.episode_index, name="episode_index"),
        )
        object.__setattr__(
            self,
            "sample_index",
            _non_negative_int(self.sample_index, name="sample_index"),
        )
        object.__setattr__(
            self,
            "episode_frame_index",
            _non_negative_int(
                self.episode_frame_index,
                name="episode_frame_index",
            ),
        )
        object.__setattr__(
            self,
            "sample_timestamp",
            _finite(self.sample_timestamp, name="sample_timestamp"),
        )


class SampleAcceptanceReceiptBuffer:
    """Thread-safe bounded retention for writer admission receipts.

    The writer remains the authority that decides when to call ``append``.
    This helper owns only bounded retention and fail-closed overflow handling;
    it does not infer acceptance from queue submission or counters.
    """

    def __init__(self, *, capacity: int = 0) -> None:
        self._capacity = _non_negative_int(capacity, name="capacity")
        self._receipts: deque[AcceptedSampleReceipt] = deque()
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, receipt: AcceptedSampleReceipt) -> None:
        if not isinstance(receipt, AcceptedSampleReceipt):
            raise TypeError("receipt must be AcceptedSampleReceipt.")
        if self._capacity <= 0:
            return
        with self._lock:
            if len(self._receipts) >= self._capacity:
                self._receipts.popleft()
                self._dropped += 1
            self._receipts.append(receipt)

    def drain(self) -> tuple[AcceptedSampleReceipt, ...]:
        """Drain in acceptance order or fail if the exact boundary was lost."""

        with self._lock:
            if self._dropped:
                dropped = self._dropped
                self._dropped = 0
                self._receipts.clear()
                raise RuntimeError(
                    "collection sample acceptance receipt buffer overflowed; "
                    f"dropped={dropped}"
                )
            receipts = tuple(self._receipts)
            self._receipts.clear()
            return receipts

    def reset(self) -> None:
        """Forget receipts from a lifecycle that has already been closed."""

        with self._lock:
            self._receipts.clear()
            self._dropped = 0


__all__ = [
    "AcceptedSampleReceipt",
    "SampleAcceptanceReceiptBuffer",
]
