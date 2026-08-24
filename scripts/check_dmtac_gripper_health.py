"""Bounded left/right Inspire-gripper health nudge with explicit motion unlock."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GRIPPER_MODULE_DIR = (
    REPO_ROOT / "hardware" / "DMTAC_gripper" / "portable_dmtac_gripper"
)
if str(GRIPPER_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(GRIPPER_MODULE_DIR))

# Import the gripper-only modules directly so this check never imports DM-Tac
# camera/tactile packages.
from gripper_debug import GripperEndpoint, load_gripper_endpoints
from gripper_driver import InspireGripper

SIDES = ("left", "right")
EXECUTE_CONFIRMATION = "GRIPPER_HEALTH_CHECK"
DEFAULT_AMPLITUDE = 0.04
MAX_AMPLITUDE = 0.10
DEFAULT_FORCE = 0.20
DEFAULT_SPEED = 0.20
POSITION_TOLERANCE = 0.015


@dataclass(frozen=True)
class GripperHealthPlan:
    endpoint: GripperEndpoint
    amplitude: float
    force: float
    speed: float
    wait_timeout: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.endpoint.side,
            "endpoint": asdict(self.endpoint),
            "amplitude": self.amplitude,
            "amplitude_semantics": "total driver-joint travel within normalized [0,1]",
            "force": self.force,
            "force_semantics": "normalized device setting, not Newtons",
            "speed": self.speed,
            "wait_timeout": self.wait_timeout,
        }


def selected_sides(value: str) -> tuple[str, ...]:
    return SIDES if value == "all" else (value,)


def motion_window(center: float, amplitude: float) -> tuple[float, float]:
    """Return a bounded full-width motion window around the measured position."""

    center = float(center)
    amplitude = float(amplitude)
    if not 0.0 <= center <= 1.0:
        raise RuntimeError(f"measured joint position is outside [0,1]: {center}")
    if not 0.0 < amplitude <= MAX_AMPLITUDE:
        raise ValueError(
            f"--amplitude must be within (0,{MAX_AMPLITUDE:g}], got {amplitude}"
        )
    half = amplitude * 0.5
    low = center - half
    high = center + half
    if low < 0.0:
        high = min(1.0, high - low)
        low = 0.0
    if high > 1.0:
        low = max(0.0, low - (high - 1.0))
        high = 1.0
    return low, high


def _configured_endpoints(args: argparse.Namespace) -> tuple[Path | None, dict[str, GripperEndpoint]]:
    configured: dict[str, GripperEndpoint] = {}
    config_path: Path | None = None
    if str(args.config).strip().lower() != "none":
        config_path, configured = load_gripper_endpoints(args.config)

    endpoints: dict[str, GripperEndpoint] = {}
    for side in selected_sides(args.side):
        base = configured.get(side)
        port = str(getattr(args, f"{side}_port") or (base.port if base else "")).strip()
        if not port:
            raise ValueError(
                f"no {side} gripper endpoint; configure it or pass --{side}-port"
            )
        endpoints[side] = GripperEndpoint(
            side=side,
            port=port,
            gripper_id=(
                getattr(args, f"{side}_id")
                if getattr(args, f"{side}_id") is not None
                else (base.gripper_id if base else 1)
            ),
            baudrate=(
                getattr(args, f"{side}_baudrate")
                if getattr(args, f"{side}_baudrate") is not None
                else (base.baudrate if base else 115200)
            ),
            timeout=(
                getattr(args, f"{side}_timeout")
                if getattr(args, f"{side}_timeout") is not None
                else (base.timeout if base else 0.10)
            ),
            max_open_mm=base.max_open_mm if base else 70.0,
        )
    return config_path, endpoints


def build_plans(args: argparse.Namespace) -> tuple[Path | None, list[GripperHealthPlan]]:
    if not 0.0 < float(args.amplitude) <= MAX_AMPLITUDE:
        raise ValueError(
            f"--amplitude must be within (0,{MAX_AMPLITUDE:g}], got {args.amplitude}"
        )
    if not 0.0 < float(args.force) <= 0.5:
        raise ValueError("--force must be within (0,0.5] for this health check")
    if not 0.0 < float(args.speed) <= 0.5:
        raise ValueError("--speed must be within (0,0.5] for this health check")
    if float(args.wait_timeout) <= 0.0:
        raise ValueError("--wait-timeout must be positive")
    if args.execute and args.confirmation != EXECUTE_CONFIRMATION:
        raise ValueError(
            f"real gripper motion requires --confirmation {EXECUTE_CONFIRMATION}"
        )

    config_path, endpoints = _configured_endpoints(args)
    plans = [
        GripperHealthPlan(
            endpoint=endpoints[side],
            amplitude=float(args.amplitude),
            force=float(args.force),
            speed=float(args.speed),
            wait_timeout=float(args.wait_timeout),
        )
        for side in selected_sides(args.side)
    ]
    return config_path, plans


def _open_gripper(endpoint: GripperEndpoint) -> InspireGripper:
    return InspireGripper(
        port=endpoint.port,
        gripper_id=endpoint.gripper_id,
        baudrate=endpoint.baudrate,
        timeout=endpoint.timeout,
        max_open_mm=endpoint.max_open_mm,
    )


def _state_dict(state: Any) -> dict[str, Any]:
    payload = dict(state.to_dict())
    payload["opening"] = 1.0 - float(state.joint_pos)
    payload["opening_semantics"] = "0=fully closed, 1=fully open"
    return payload


def _require_healthy_idle(state: Any, *, side: str, stage: str) -> None:
    if bool(state.is_moving):
        raise RuntimeError(f"{side} gripper is still moving at {stage}")
    if int(state.error_code) != 0:
        raise RuntimeError(
            f"{side} gripper reports error_code={state.error_code} at {stage}"
        )
    if int(state.status_code) not in (1, 2, 3):
        raise RuntimeError(
            f"{side} gripper reports non-idle status_code={state.status_code} at {stage}"
        )


def _require_target_reached(state: Any, *, side: str, stage: str, target: float) -> None:
    error = abs(float(state.joint_pos) - float(target))
    if error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"{side} gripper did not reach {stage} target: measured={state.joint_pos:.6f}, "
            f"target={target:.6f}, error={error:.6f}, tolerance={POSITION_TOLERANCE:.6f}"
        )


def execute_plan(
    plan: GripperHealthPlan,
    *,
    gripper_factory: Callable[[GripperEndpoint], Any] = _open_gripper,
) -> dict[str, Any]:
    """Execute one side sequentially and restore position plus device settings."""

    gripper = gripper_factory(plan.endpoint)
    motion_started = False
    try:
        start = gripper.get_state()
        _require_healthy_idle(start, side=plan.endpoint.side, stage="preflight")
        original_speed = float(gripper.get_speed()[0])
        original_force = float(gripper.get_torque()[0])
        original_continuous = bool(gripper.get_continuous_grasp())
        low, high = motion_window(float(start.joint_pos), plan.amplitude)

        states: dict[str, dict[str, Any]] = {"start": _state_dict(start)}
        targets = ("low", low), ("high", high), ("restored", float(start.joint_pos))
        for stage, target in targets:
            motion_started = True
            gripper.set_joint_pos(
                target,
                speed=plan.speed,
                torque=plan.force,
                continuous=False,
                force_send=True,
            )
            state = gripper.wait_until_idle(plan.wait_timeout)
            _require_healthy_idle(state, side=plan.endpoint.side, stage=stage)
            _require_target_reached(
                state,
                side=plan.endpoint.side,
                stage=stage,
                target=target,
            )
            states[stage] = _state_dict(state)

        gripper.set_speed(original_speed)
        gripper.set_torque(original_force)
        gripper.set_continuous_grasp(original_continuous)
        motion_started = False
        return {
            "side": plan.endpoint.side,
            "ok": True,
            "targets": {
                "driver_joint_pos": {"low": low, "high": high, "restore": float(start.joint_pos)},
                "opening": {"low": 1.0 - low, "high": 1.0 - high, "restore": 1.0 - float(start.joint_pos)},
            },
            "states": states,
            "original_settings": {
                "speed": original_speed,
                "force": original_force,
                "continuous": original_continuous,
            },
            "settings_restored": True,
        }
    except BaseException as exc:
        if motion_started:
            try:
                gripper.estop()
            except Exception as stop_exc:  # noqa: BLE001 - preserve vendor SDK failures.
                raise RuntimeError(
                    f"{plan.endpoint.side} health motion failed ({exc}); "
                    f"stop request also failed ({stop_exc})"
                ) from exc
        raise
    finally:
        gripper.close()


def execute_plans(
    plans: Sequence[GripperHealthPlan],
    *,
    gripper_factory: Callable[[GripperEndpoint], Any] = _open_gripper,
) -> list[dict[str, Any]]:
    return [execute_plan(plan, gripper_factory=gripper_factory) for plan in plans]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute a bounded Inspire-gripper health nudge. Default side is all; "
            "real motion is sequential and requires an explicit confirmation."
        )
    )
    parser.add_argument("--side", choices=("all", *SIDES), default="all")
    parser.add_argument("--config", default="latest", help="DMTAC assignment path or 'latest'.")
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--force", type=float, default=DEFAULT_FORCE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--wait-timeout", type=float, default=3.0)
    for side in SIDES:
        group = parser.add_argument_group(f"{side} endpoint override")
        group.add_argument(f"--{side}-port", default="")
        group.add_argument(f"--{side}-id", type=int, default=None)
        group.add_argument(f"--{side}-baudrate", type=int, default=None)
        group.add_argument(f"--{side}-timeout", type=float, default=None)
    parser.add_argument("--execute", action="store_true", help="Allow real gripper motion.")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path, plans = build_plans(args)
        payload: dict[str, Any] = {
            "schema": "dmtac_gripper_health_check.v1",
            "mode": "execute" if args.execute else "preview",
            "config": None if config_path is None else str(config_path),
            "plans": [plan.to_dict() for plan in plans],
        }
        if args.execute:
            payload["results"] = execute_plans(plans)
        else:
            payload["next_step"] = (
                "stop teleop/collection, clear the jaws, verify E-stop access, then add "
                f"--execute --confirmation {EXECUTE_CONFIRMATION}"
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
