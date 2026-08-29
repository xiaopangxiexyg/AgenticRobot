"""Specialized CLI for the bounded rule-based real writing collection."""

from __future__ import annotations

from control_station.real_teleop_entrypoint import run_real_teleop_entrypoint
from teleop_runtime.real_teleop.rule_based_writing_plan import (
    add_rule_based_writing_arguments,
    build_rule_based_writing_extension,
)


def main() -> None:
    run_real_teleop_entrypoint(
        configure_parser=add_rule_based_writing_arguments,
        runtime_extension_factory=build_rule_based_writing_extension,
    )


if __name__ == "__main__":
    main()
