"""Read basic information and status from one or both RM65 arms without motion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SIDES = ("left", "right")


@dataclass(frozen=True)
class ArmEndpoint:
    side: str
    ip: str


def selected_sides(value: str) -> tuple[str, ...]:
    return SIDES if value == "all" else (value,)


def _default_assignment_loader(value: str) -> tuple[Path, Mapping[str, Any]]:
    from teleop_runtime.hardware_config.rm65_arm_ips import load_assignment

    payload = load_assignment(value)
    return Path(str(payload["_config_path"])), payload


def resolve_endpoints(
    args: argparse.Namespace,
    *,
    assignment_loader: Callable[[str], tuple[Path, Mapping[str, Any]]] = _default_assignment_loader,
) -> tuple[Path | None, list[ArmEndpoint]]:
    config_path: Path | None = None
    configured: dict[str, str] = {}
    if str(args.config).strip().lower() != "none":
        config_path, payload = assignment_loader(args.config)
        assignments = payload.get("assignments")
        if not isinstance(assignments, Mapping):
            raise ValueError(f"{config_path} does not contain an assignments object")
        for side in SIDES:
            side_payload = assignments.get(side)
            if isinstance(side_payload, Mapping):
                configured[side] = str(side_payload.get("arm_ip") or "").strip()

    endpoints: list[ArmEndpoint] = []
    for side in selected_sides(args.side):
        ip = str(getattr(args, f"{side}_ip") or configured.get(side) or "").strip()
        if not ip:
            raise ValueError(f"no {side} RM65 IP; configure it or pass --{side}-ip")
        endpoints.append(ArmEndpoint(side=side, ip=ip))
    return config_path, endpoints


def _open_arm(endpoint: ArmEndpoint) -> Any:
    from hardware.rm65.rm_api2_adapter import RM65, Arm

    return Arm(RM65, endpoint.ip)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_arm_status(
    endpoint: ArmEndpoint,
    *,
    arm_factory: Callable[[ArmEndpoint], Any] = _open_arm,
) -> dict[str, Any]:
    arm = arm_factory(endpoint)
    result: dict[str, Any] = {
        "side": endpoint.side,
        "endpoint": asdict(endpoint),
        "mode": "read_only",
        "queries": {},
        "errors": [],
    }

    def record_tuple(name: str, call: Callable[[], tuple[Any, Any]]) -> None:
        try:
            code, data = call()
            code = int(code)
            result["queries"][name] = {
                "return_code": code,
                "data": _jsonable(data),
            }
            if code != 0:
                result["errors"].append(f"{name} return_code={code}")
        except Exception as exc:  # noqa: BLE001 - report partial vendor SDK status.
            result["queries"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            result["errors"].append(f"{name} raised {type(exc).__name__}: {exc}")

    try:
        try:
            result["api_version"] = _jsonable(arm.API_Version())
        except Exception as exc:  # noqa: BLE001 - report local SDK version failure.
            result["api_version"] = None
            result["errors"].append(f"API_Version raised {type(exc).__name__}: {exc}")

        robot = arm._robot
        record_tuple("robot_info", robot.rm_get_robot_info)
        record_tuple("arm_software_info", robot.rm_get_arm_software_info)
        record_tuple("current_arm_state", robot.rm_get_current_arm_state)
        record_tuple("joint_enable_state", robot.rm_get_joint_en_state)
        record_tuple("joint_temperature_c", robot.rm_get_current_joint_temperature)
        record_tuple("joint_current_ma", robot.rm_get_current_joint_current)
        record_tuple("joint_voltage_v", robot.rm_get_current_joint_voltage)

        try:
            joint_errors = dict(robot.rm_get_joint_err_flag())
            code = int(joint_errors.get("return_code", -3))
            result["queries"]["joint_error_and_brake_state"] = {
                "return_code": code,
                "data": _jsonable(joint_errors),
            }
            if code != 0:
                result["errors"].append(
                    f"joint_error_and_brake_state return_code={code}"
                )
        except Exception as exc:  # noqa: BLE001 - report partial vendor SDK status.
            result["queries"]["joint_error_and_brake_state"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            result["errors"].append(
                f"joint_error_and_brake_state raised {type(exc).__name__}: {exc}"
            )
    finally:
        try:
            close_code = int(arm.close())
            result["connection_close_return_code"] = close_code
            if close_code != 0:
                result["errors"].append(f"connection close return_code={close_code}")
        except Exception as exc:  # noqa: BLE001 - connection cleanup must be reported.
            result["connection_close_error"] = f"{type(exc).__name__}: {exc}"
            result["errors"].append(
                f"connection close raised {type(exc).__name__}: {exc}"
            )

    result["ok"] = not result["errors"]
    return result


def read_all_statuses(
    endpoints: Sequence[ArmEndpoint],
    *,
    arm_factory: Callable[[ArmEndpoint], Any] = _open_arm,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for endpoint in endpoints:
        try:
            results.append(read_arm_status(endpoint, arm_factory=arm_factory))
        except Exception as exc:  # noqa: BLE001 - continue with the other selected arm.
            results.append(
                {
                    "side": endpoint.side,
                    "endpoint": asdict(endpoint),
                    "mode": "read_only",
                    "ok": False,
                    "errors": [f"connection failed: {type(exc).__name__}: {exc}"],
                }
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read RM65 basic information, software versions, measured state, and joint "
            "status. This command never sends motion, configuration, stop, or enable writes."
        )
    )
    parser.add_argument("--side", choices=("all", *SIDES), default="all")
    parser.add_argument("--config", default="latest", help="RM65 IP assignment path or 'latest'.")
    parser.add_argument("--left-ip", default="")
    parser.add_argument("--right-ip", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path, endpoints = resolve_endpoints(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results = read_all_statuses(endpoints)
    payload = {
        "schema": "rm65_status_check.v1",
        "mode": "read_only",
        "config": None if config_path is None else str(config_path),
        "ok": all(bool(result.get("ok")) for result in results),
        "arms": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
