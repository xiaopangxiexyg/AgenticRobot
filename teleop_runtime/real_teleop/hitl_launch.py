"""Read-only acceptance and explicit handoff for exclusive RealTeleop HITL.

The default path only parses contracts and local artifacts.  It never opens a
device, socket, camera, policy transport, dataset writer, or robot backend.
RealTeleop remains the sole production entrypoint and is reached only after an
explicit ``--execute`` handoff.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from collection_runtime.dataset_layout import (
    FormalDatasetLayout,
    resolve_formal_dataset_layout,
)
from dataset_contracts.hitl import ExclusiveHITLDatasetContract
from dataset_interface.dataset_task_binding import (
    DatasetTaskBindingError,
    inspect_dataset_task_binding,
)
from dataset_interface.hitl import validate_exclusive_hitl_dataset_append
from deployments.openpi.client import preflight_policy_artifact
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
)
from teleop_runtime.configs import ANNOTATED_DUAL_RM65_BASE_DISTANCE_M

from .cli import build_real_teleop_argument_parser
from .hitl_cli import add_exclusive_hitl_arguments
from .hitl_policy_runtime import (
    HITL_POLICY_MODE_NORMAL,
    HITL_POLICY_MODE_RTC,
    build_exclusive_hitl_dataset_contract,
    preflight_normal_hitl_policy_source,
)
from .hitl_terminal import validate_hitl_terminal_keys


HITL_LAUNCH_ACCEPTANCE_SCHEMA_V1 = (
    "dexteleop.real_teleop.hitl_launch_acceptance.v1"
)
CANONICAL_REAL_TELEOP_MODULE = (
    "teleop_runtime.debug.run_vive_rm65_real_teleop"
)


class HITLLaunchAcceptanceError(ValueError):
    """Raised when the requested launch would not be identity-safe."""


@dataclass(frozen=True)
class PreparedHITLLaunch:
    """Immutable offline result and exact canonical process handoff."""

    canonical_argv: tuple[str, ...]
    policy_mode: str
    execution_supported: bool
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_hitl_launch_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one exclusive RealTeleop HITL collection launch without "
            "opening hardware or policy transports. Use --execute explicitly "
            "to hand the validated arguments to the canonical RealTeleop module."
        ),
        epilog=(
            "Separate launcher options from canonical RealTeleop options with --. "
            "Example: python -m teleop_runtime.real_teleop.hitl_launch -- "
            "--gui --hitl-policy-control ..."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "After offline acceptance, replace this process with the canonical "
            "RealTeleop entrypoint. RTC is intentionally not executable."
        ),
    )
    return parser


def _option_present(argv: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(value == option or value.startswith(prefix) for value in argv)


def _required_explicit_collection_identity(
    args: argparse.Namespace,
    real_teleop_argv: Sequence[str],
) -> None:
    required = (
        ("--dataset-root", "dataset_root"),
        ("--collection-task-name", "collection_task_name"),
        ("--collection-task-id", "collection_task_id"),
    )
    missing = [
        option
        for option, attribute in required
        if not _option_present(real_teleop_argv, option)
        or not str(getattr(args, attribute, "") or "").strip()
    ]
    if missing:
        raise HITLLaunchAcceptanceError(
            "HITL launch acceptance requires explicit collection identity: "
            + ", ".join(missing)
            + "."
        )


def _resolve_pose_frame_contract(args: argparse.Namespace) -> str:
    active_sides = tuple(str(side) for side in args.active_sides)
    if args.pose_frame_contract is not None:
        source = "explicit"
    elif bool(args.no_pose_frame_contract):
        source = "disabled"
    else:
        if active_sides == ("left", "right") and not math.isclose(
            float(args.left_right_base_distance),
            float(ANNOTATED_DUAL_RM65_BASE_DISTANCE_M),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise HITLLaunchAcceptanceError(
                "The default RM65 workspace-near-boundary pose contract is "
                "calibrated for --left-right-base-distance "
                f"{ANNOTATED_DUAL_RM65_BASE_DISTANCE_M}. Provide an explicit "
                "--pose-frame-contract for another station."
            )
        args.pose_frame_contract = (
            default_rm65_workspace_near_boundary_collection_pose_frame_contract(
                active_sides
            )
        )
        source = (
            "default_rm65_dual_arm_workspace_near_boundary_v1"
            if args.pose_frame_contract is not None
            else "unresolved"
        )
    if args.pose_frame_contract is not None:
        contract_sides = tuple(args.pose_frame_contract.source.active_sides)
        if contract_sides != active_sides:
            raise HITLLaunchAcceptanceError(
                "Pose-frame contract active_sides disagree with collection "
                f"--active-sides: {contract_sides!r} != {active_sides!r}."
            )
    args.pose_frame_contract_source = source
    return source


def _read_dataset_info(dataset_root: Path) -> Mapping[str, Any] | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HITLLaunchAcceptanceError(
            f"Cannot read target dataset metadata {info_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise HITLLaunchAcceptanceError(
            f"Target dataset metadata must be an object: {info_path}"
        )
    return payload


def _validate_dataset_identity(
    layout: FormalDatasetLayout,
    contract: ExclusiveHITLDatasetContract,
) -> None:
    validate_exclusive_hitl_dataset_append(layout.dataset_root, contract)
    info = _read_dataset_info(layout.dataset_root)
    if info is None:
        return
    try:
        task_binding = inspect_dataset_task_binding(
            layout.dataset_root,
            require_complete=True,
        )
    except DatasetTaskBindingError as exc:
        raise HITLLaunchAcceptanceError(
            f"Existing target dataset has an invalid canonical task binding: {exc}"
        ) from exc
    if task_binding is None:  # ``require_complete`` makes this defensive only.
        raise HITLLaunchAcceptanceError(
            "Existing target dataset has no authoritative task binding."
        )
    if task_binding.task_id != layout.task_id:
        raise HITLLaunchAcceptanceError(
            "Target dataset task binding differs from the requested HITL launch: "
            f"existing={task_binding.task_id!r}, requested={layout.task_id!r}."
        )
    existing_dataset_name = str(info.get("dataset_name", "") or "").strip()
    if not existing_dataset_name:
        raise HITLLaunchAcceptanceError(
            "Existing target dataset has no authoritative dataset_name."
        )
    if existing_dataset_name != layout.dataset_name:
        raise HITLLaunchAcceptanceError(
            "Target dataset name differs from the requested HITL launch: "
            f"existing={existing_dataset_name!r}, "
            f"requested={layout.dataset_name!r}."
        )


def _parse_real_teleop_args(
    real_teleop_argv: Sequence[str],
) -> argparse.Namespace:
    parser = build_real_teleop_argument_parser()
    add_exclusive_hitl_arguments(parser)
    args = parser.parse_args(list(real_teleop_argv))
    args._dataset_name_explicit = _option_present(
        real_teleop_argv,
        "--dataset-name",
    )
    return args


def prepare_hitl_launch(
    real_teleop_argv: Sequence[str],
    *,
    date_prefix: str | None = None,
    execution_requested: bool = False,
) -> PreparedHITLLaunch:
    """Perform local contract acceptance without opening runtime resources."""

    forwarded = tuple(str(value) for value in real_teleop_argv)
    if not forwarded:
        raise HITLLaunchAcceptanceError(
            "Canonical RealTeleop arguments are required after --."
        )
    args = _parse_real_teleop_args(forwarded)
    if not bool(args.hitl_policy_control):
        raise HITLLaunchAcceptanceError(
            "The exclusive HITL launcher requires --hitl-policy-control."
        )
    if bool(args.no_collection):
        raise HITLLaunchAcceptanceError(
            "Exclusive HITL requires formal collection; remove --no-collection."
        )
    _required_explicit_collection_identity(args, forwarded)
    validate_hitl_terminal_keys(
        args.hitl_takeover_key,
        args.hitl_policy_resume_key,
    )
    pose_frame_contract_source = _resolve_pose_frame_contract(args)

    mode = str(args.hitl_policy_mode).strip().lower()
    normal_source_config = None
    policy_preflight: Mapping[str, Any] | None = None
    if mode == HITL_POLICY_MODE_NORMAL:
        normal_source_config = preflight_normal_hitl_policy_source(args)
        if normal_source_config is None:
            raise HITLLaunchAcceptanceError(
                "Offline launch acceptance does not allow an injected policy source."
            )
        policy_preflight = preflight_policy_artifact(
            normal_source_config.policy_artifact,
            active_sides=normal_source_config.model_observation_sides,
            runtime_pose_frame_contract=(
                normal_source_config.runtime_pose_frame_contract
            ),
        ).to_dict()
    elif mode == HITL_POLICY_MODE_RTC:
        if str(args.hitl_policy_artifact_json or "").strip():
            raise HITLLaunchAcceptanceError(
                "RTC no-action acceptance does not accept "
                "--hitl-policy-artifact-json."
            )
        if execution_requested:
            raise HITLLaunchAcceptanceError(
                "RTC remains no-action: no committed-emission producer is bound "
                "to the canonical launcher, so --execute is forbidden."
            )
    else:
        raise HITLLaunchAcceptanceError(
            "HITL policy mode must be normal or rtc."
        )

    contract = build_exclusive_hitl_dataset_contract(
        args,
        normal_source_config=normal_source_config,
    )
    resolved_date = (
        str(date_prefix).strip()
        if date_prefix is not None
        else datetime.now().strftime("%Y-%m-%d")
    )
    if not resolved_date:
        raise HITLLaunchAcceptanceError("date_prefix must be non-empty.")
    layout = resolve_formal_dataset_layout(
        args,
        date_prefix=resolved_date,
    )
    _validate_dataset_identity(layout, contract)

    canonical_argv = (
        sys.executable,
        "-m",
        CANONICAL_REAL_TELEOP_MODULE,
        *forwarded,
    )
    report: dict[str, Any] = {
        "schema": HITL_LAUNCH_ACCEPTANCE_SCHEMA_V1,
        "ok": True,
        "acceptance_kind": "read_only_offline_contract_preflight",
        "execute_requested": bool(execution_requested),
        "execution_supported": mode == HITL_POLICY_MODE_NORMAL,
        "canonical_command": [
            "python",
            "-m",
            CANONICAL_REAL_TELEOP_MODULE,
            *forwarded,
        ],
        "policy_mode": mode,
        "rtc_no_action": mode == HITL_POLICY_MODE_RTC,
        "exclusive_source_semantics": "one_complete_source_no_blending",
        "formal_collection": True,
        "active_sides": list(args.active_sides),
        "pose_frame_contract_source": pose_frame_contract_source,
        "dataset": {
            "date_prefix": resolved_date,
            "base_root": str(layout.dataset_root.parent),
            "root": str(layout.dataset_root),
            "dataset_name": layout.dataset_name,
            "task_name": layout.task_name,
            "task_id": layout.task_id,
            "task_type": layout.task_type,
        },
        "hitl_dataset_contract": contract.to_payload(),
        "normal_policy_preflight": policy_preflight,
        "policy_endpoint": (
            None
            if normal_source_config is None
            else {
                "host": normal_source_config.host,
                "port": normal_source_config.port,
                "reachability": "not_checked",
            }
        ),
        "resource_effects": {
            "device_opened": False,
            "socket_opened": False,
            "policy_service_contacted": False,
            "camera_opened": False,
            "dataset_writer_opened": False,
            "robot_motion_commanded": False,
        },
        "unverified_boundaries": [
            "policy-service reachability and inference latency",
            "teleoperation-device and camera availability",
            "real-robot safety, dispatch, and supervised motion",
            "physical dataset merge and downstream publication readiness",
        ],
    }
    if mode == HITL_POLICY_MODE_RTC:
        report["unverified_boundaries"].append(
            "RTC committed-emission producer and every RTC Active authorization gate"
        )
    return PreparedHITLLaunch(
        canonical_argv=canonical_argv,
        policy_mode=mode,
        execution_supported=mode == HITL_POLICY_MODE_NORMAL,
        report=report,
    )


def _split_cli_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> tuple[argparse.Namespace, tuple[str, ...]]:
    values = list(argv)
    if "--" not in values:
        options = parser.parse_args(values)
        del options
        parser.error(
            "separate launcher options from canonical RealTeleop arguments with --"
        )
    separator = values.index("--")
    options = parser.parse_args(values[:separator])
    forwarded = tuple(values[separator + 1 :])
    if not forwarded:
        parser.error("canonical RealTeleop arguments are required after --")
    return options, forwarded


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_hitl_launch_argument_parser()
    options, forwarded = _split_cli_args(
        parser,
        sys.argv[1:] if argv is None else argv,
    )
    try:
        prepared = prepare_hitl_launch(
            forwarded,
            execution_requested=bool(options.execute),
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            prepared.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if bool(options.execute):
        os.execv(sys.executable, list(prepared.canonical_argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_REAL_TELEOP_MODULE",
    "HITL_LAUNCH_ACCEPTANCE_SCHEMA_V1",
    "HITLLaunchAcceptanceError",
    "PreparedHITLLaunch",
    "build_hitl_launch_argument_parser",
    "main",
    "prepare_hitl_launch",
]
