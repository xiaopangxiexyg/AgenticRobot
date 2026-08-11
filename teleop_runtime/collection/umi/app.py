"""Compatibility import for the Station-owned UMI application binding.

The public import remains available while the existing
``python -m teleop_runtime.collection umi`` command is preserved.  New
application-level callers should import :mod:`control_station.umi_application`.
"""

from control_station.umi_application import run_umi_collection


__all__ = ["run_umi_collection"]
