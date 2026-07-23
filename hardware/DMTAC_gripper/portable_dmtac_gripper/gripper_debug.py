from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .gripper_driver import InspireGripper
except ImportError:  # Allows run_gripper_only.py to avoid importing the tactile package.
    from gripper_driver import InspireGripper


SIDES = ("left", "right")
WRITE_CONFIRMATION = "GRIPPER_ONLY"
HIGH_FORCE_CONFIRMATION = "HIGH_FORCE"


@dataclass(frozen=True)
class GripperEndpoint:
    side: str
    port: str
    gripper_id: int = 1
    baudrate: int = 115200
    timeout: float = 0.10
    max_open_mm: float = 70.0


@dataclass(frozen=True)
class GripperCommand:
    endpoint: GripperEndpoint
    opening: float | None = None
    force: float | None = None
    speed: float | None = None
    continuous: bool | None = None

    @property
    def joint_pos(self) -> float | None:
        # Public debug semantics use opening: 0=closed, 1=open.  The driver uses
        # joint_pos: 0=open, 1=closed.
        return None if self.opening is None else 1.0 - self.opening

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.endpoint.side,
            "endpoint": asdict(self.endpoint),
            "opening": self.opening,
            "opening_semantics": "0=fully closed, 1=fully open",
            "driver_joint_pos": self.joint_pos,
            "force": self.force,
            "speed": self.speed,
            "continuous": self.continuous,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _latest_in(root: Path) -> Path | None:
    explicit = root / "latest.json"
    if explicit.is_file():
        return explicit
    candidates = list(root.glob("*_dmtac_usb_assignment.json")) if root.is_dir() else []
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_config_path(value: str | Path) -> Path:
    text = str(value).strip()
    if text.lower() != "latest":
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DMTac assignment config does not exist: {path}")
        return path

    repo = _repo_root()
    roots = [repo / "teleop_runtime" / "configs" / "hardware" / "dmtac"]
    try:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from project_paths import hardware_interface_configs_root, teleop_records_root

        roots = [
            hardware_interface_configs_root() / "dmtac",
            teleop_records_root() / "dmtac_hardware_configs",
            *roots,
        ]
    except Exception:
        pass

    seen: set[Path] = set()
    for root in roots:
        resolved_root = root.expanduser().resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        path = _latest_in(resolved_root)
        if path is not None:
            return path
    raise FileNotFoundError(
        "No DMTac assignment config found in: "
        + ", ".join(str(path) for path in seen)
    )


def load_gripper_endpoints(value: str | Path = "latest") -> tuple[Path, dict[str, GripperEndpoint]]:
    path = resolve_config_path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError(f"{path} does not contain an assignments object")

    result: dict[str, GripperEndpoint] = {}
    for side in SIDES:
        side_config = assignments.get(side)
        gripper = side_config.get("gripper") if isinstance(side_config, Mapping) else None
        if not isinstance(gripper, Mapping) or gripper.get("enabled", True) is False:
            continue
        port = str(gripper.get("port") or "").strip()
        if not port:
            continue
        result[side] = GripperEndpoint(
            side=side,
            port=port,
            gripper_id=int(gripper.get("gripper_id", 1)),
            baudrate=int(gripper.get("baudrate", 115200)),
            timeout=float(gripper.get("timeout", 0.10)),
            max_open_mm=float(gripper.get("max_open_mm", 70.0)),
        )
    return path, result


def _selected_sides(args: argparse.Namespace, *, default_both: bool) -> tuple[str, ...]:
    selected: list[str] = []
    if args.side == "both":
        selected.extend(SIDES)
    elif args.side:
        selected.append(args.side)
    for side in SIDES:
        if any(
            getattr(args, f"{side}_{name}", None) is not None
            for name in ("opening", "force", "speed")
        ) and side not in selected:
            selected.append(side)
    if not selected and default_both:
        return SIDES
    if not selected:
        raise ValueError("select --side left/right/both, or pass a --left-* / --right-* value")
    return tuple(side for side in SIDES if side in selected)


def _endpoint_for_side(
    args: argparse.Namespace,
    side: str,
    configured: Mapping[str, GripperEndpoint],
) -> GripperEndpoint:
    base = configured.get(side)
    port = getattr(args, f"{side}_port") or (base.port if base else "")
    if not port:
        raise ValueError(f"no {side} gripper endpoint; configure it or pass --{side}-port")
    return GripperEndpoint(
        side=side,
        port=str(port),
        gripper_id=getattr(args, f"{side}_id") or (base.gripper_id if base else 1),
        baudrate=getattr(args, f"{side}_baudrate") or (base.baudrate if base else 115200),
        timeout=(
            getattr(args, f"{side}_timeout")
            if getattr(args, f"{side}_timeout") is not None
            else (base.timeout if base else 0.10)
        ),
        max_open_mm=base.max_open_mm if base else 70.0,
    )


def resolve_endpoints(
    args: argparse.Namespace,
    sides: Sequence[str],
) -> tuple[Path | None, dict[str, GripperEndpoint]]:
    configured: dict[str, GripperEndpoint] = {}
    config_path: Path | None = None
    if str(args.config).strip().lower() != "none":
        config_path, configured = load_gripper_endpoints(args.config)
    return config_path, {
        side: _endpoint_for_side(args, side, configured)
        for side in sides
    }


def _side_value(args: argparse.Namespace, side: str, name: str) -> Any:
    specific = getattr(args, f"{side}_{name}", None)
    return specific if specific is not None else getattr(args, name, None)


def build_commands(args: argparse.Namespace) -> tuple[Path | None, list[GripperCommand]]:
    sides = _selected_sides(args, default_both=False)
    config_path, endpoints = resolve_endpoints(args, sides)
    commands: list[GripperCommand] = []
    for side in sides:
        opening = _side_value(args, side, "opening")
        force = _side_value(args, side, "force")
        speed = _side_value(args, side, "speed")
        if opening is None and force is None and speed is None:
            raise ValueError(f"no opening, force, or speed requested for {side}")
        for name, value in (("opening", opening), ("force", force), ("speed", speed)):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{side} {name} must be within [0,1], got {value}")
        commands.append(
            GripperCommand(
                endpoint=endpoints[side],
                opening=None if opening is None else float(opening),
                force=None if force is None else float(force),
                speed=None if speed is None else float(speed),
                continuous=bool(args.continuous) if opening is not None else None,
            )
        )
    return config_path, commands


def validate_write_unlock(args: argparse.Namespace, commands: Sequence[GripperCommand]) -> None:
    if not args.execute:
        return
    if args.confirmation != WRITE_CONFIRMATION:
        raise ValueError(
            f"real gripper writes require --confirmation {WRITE_CONFIRMATION}"
        )
    high_forces = [command.force for command in commands if (command.force or 0.0) > 0.5]
    if high_forces and (
        not args.allow_high_force
        or args.high_force_confirmation != HIGH_FORCE_CONFIRMATION
    ):
        raise ValueError(
            "force above 0.5 requires --allow-high-force "
            f"--high-force-confirmation {HIGH_FORCE_CONFIRMATION}"
        )


def _open(endpoint: GripperEndpoint) -> InspireGripper:
    return InspireGripper(
        port=endpoint.port,
        gripper_id=endpoint.gripper_id,
        baudrate=endpoint.baudrate,
        timeout=endpoint.timeout,
        max_open_mm=endpoint.max_open_mm,
    )


def execute_commands(
    commands: Sequence[GripperCommand],
    *,
    wait: bool,
    wait_timeout: float,
    gripper_factory: Callable[[GripperEndpoint], Any] = _open,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in commands:
        gripper = gripper_factory(command.endpoint)
        try:
            if command.opening is not None:
                gripper.set_joint_pos(
                    command.joint_pos,
                    speed=command.speed,
                    torque=command.force,
                    continuous=command.continuous,
                    force_send=True,
                )
            else:
                if command.speed is not None:
                    gripper.set_speed(command.speed)
                if command.force is not None:
                    gripper.set_torque(command.force)
            state = (
                gripper.wait_until_idle(wait_timeout)
                if wait and command.opening is not None
                else gripper.get_state()
            )
            results.append({"side": command.endpoint.side, "state": state.to_dict()})
        finally:
            gripper.close()
    return results


def read_status(
    endpoints: Mapping[str, GripperEndpoint],
    *,
    gripper_factory: Callable[[GripperEndpoint], Any] = _open,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for side in SIDES:
        endpoint = endpoints.get(side)
        if endpoint is None:
            continue
        gripper = gripper_factory(endpoint)
        try:
            state = gripper.get_state()
            results.append(
                {
                    "side": side,
                    "endpoint": asdict(endpoint),
                    "state": state.to_dict(),
                    "opening": 1.0 - float(state.joint_pos),
                    "force_setting": float(gripper.get_torque()[0]),
                    "speed_setting": float(gripper.get_speed()[0]),
                }
            )
        finally:
            gripper.close()
    return results


def _add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="latest",
        help="USB assignment path or 'latest'; use 'none' with explicit --left/right-port.",
    )
    for side in SIDES:
        group = parser.add_argument_group(f"{side} endpoint override")
        group.add_argument(f"--{side}-port", default="")
        group.add_argument(f"--{side}-id", type=int, default=None)
        group.add_argument(f"--{side}-baudrate", type=int, default=None)
        group.add_argument(f"--{side}-timeout", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone left/right Inspire gripper debugger. It does not import or connect "
            "DM-Tac tactile devices, robot arms, teleop, or collection runtimes."
        )
    )
    _add_endpoint_args(parser)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Read gripper state and current force/speed settings.")
    status.add_argument("--side", choices=(*SIDES, "both"), default="both")

    set_parser = sub.add_parser(
        "set",
        help="Preview or apply independent opening/force/speed values.",
    )
    set_parser.add_argument("--side", choices=(*SIDES, "both"), default=None)
    set_parser.add_argument("--opening", type=float, default=None, help="0=closed, 1=open.")
    set_parser.add_argument("--force", type=float, default=None, help="Normalized setting [0,1], not Newtons.")
    set_parser.add_argument("--speed", type=float, default=None, help="Normalized setting [0,1].")
    for side in SIDES:
        set_parser.add_argument(f"--{side}-opening", type=float, default=None)
        set_parser.add_argument(f"--{side}-force", type=float, default=None)
        set_parser.add_argument(f"--{side}-speed", type=float, default=None)
    set_parser.add_argument("--continuous", action="store_true")
    set_parser.add_argument("--wait", action="store_true")
    set_parser.add_argument("--wait-timeout", type=float, default=3.0)
    set_parser.add_argument("--execute", action="store_true", help="Apply the previewed writes to real grippers.")
    set_parser.add_argument("--confirmation", default="")
    set_parser.add_argument("--allow-high-force", action="store_true")
    set_parser.add_argument("--high-force-confirmation", default="")
    return parser


def _print(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            sides = _selected_sides(args, default_both=True)
            config_path, endpoints = resolve_endpoints(args, sides)
            _print(
                {"config": None if config_path is None else str(config_path), "grippers": read_status(endpoints)},
                as_json=args.json,
            )
            return

        config_path, commands = build_commands(args)
        validate_write_unlock(args, commands)
        payload: dict[str, Any] = {
            "mode": "execute" if args.execute else "preview",
            "config": None if config_path is None else str(config_path),
            "commands": [command.to_dict() for command in commands],
        }
        if args.execute:
            payload["results"] = execute_commands(
                commands,
                wait=bool(args.wait),
                wait_timeout=float(args.wait_timeout),
            )
        else:
            payload["next_step"] = (
                f"review the endpoints and values, then add --execute --confirmation {WRITE_CONFIRMATION}"
            )
        _print(payload, as_json=args.json)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
