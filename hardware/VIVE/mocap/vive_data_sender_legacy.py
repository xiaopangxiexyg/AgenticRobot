"""Stable VIVE v1 sender entry point.

Use this file when the receiver is a released/legacy deployment such as
``DexTeleop_StableCollect_0705``.  It shares the exact pose-sampling and v1
UDP data-plane implementation with :mod:`vive_data_sender`, but it forcibly
disables the separate clock-control plane even when a caller supplies new
clock-sync flags by mistake.
"""

from __future__ import annotations

from typing import Sequence

try:  # Support both ``python -m`` and direct Windows script execution.
    from . import vive_data_sender as _sender
except ImportError:
    import vive_data_sender as _sender  # type: ignore[no-redef]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the exact v1 pose sender with all clock control disabled."""

    _sender.main(
        None if argv is None else list(argv),
        clock_sync_mode_default="off",
        send_hz_default=_sender.DEFAULT_LEGACY_SEND_HZ,
        force_clock_sync_mode="off",
        legacy_wire_format=True,
    )


if __name__ == "__main__":
    main()
