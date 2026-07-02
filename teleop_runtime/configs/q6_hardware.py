from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _assignment_module():
    from hardware.teleop_hand import q6_assignment

    return q6_assignment


def load_q6_hardware_assignment(
    config: str | Path = "latest",
    *,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    return _assignment_module().load_assignment(config, config_dir=config_dir)


def q6_ports_from_assignment(
    assignment: Mapping[str, Any],
    *,
    active_sides: Sequence[str] = ("left", "right"),
) -> dict[str, str]:
    return _assignment_module().q6_ports_for_sides(
        assignment,
        active_sides=active_sides,
    )


def load_latest_q6_ports(
    *,
    active_sides: Sequence[str] = ("left", "right"),
    config: str | Path = "latest",
    config_dir: str | Path | None = None,
    allow_missing: bool = True,
) -> dict[str, str]:
    return _assignment_module().load_latest_q6_ports(
        active_sides=active_sides,
        config=config,
        config_dir=config_dir,
        allow_missing=allow_missing,
    )
