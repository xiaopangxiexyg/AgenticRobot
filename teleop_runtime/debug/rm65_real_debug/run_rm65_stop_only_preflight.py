"""Authorized M5.8-b0 stationary-arm one-shot Slow-Stop preflight.

Unlike the state visualizer, this tool performs a real controller-state
change: one ``Arm_Slow_Stop`` call per arm.  It deliberately has no motion or
lifecycle permit provider and therefore cannot send a joint target, home,
enable, rearm, or force-position startup call.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
import uuid

import numpy as np

from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from teleop_runtime.arm_frames import TOOL_TCP_FRAME
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.backends.arms.rm65_stop_only_binding import (
    RM65StopOnlyHardwareAuthorization,
    RM65StopOnlyLegacyBinding,
)
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M5.8-b0: send exactly one permit-gated Arm_Slow_Stop to each "
            "stationary RM65 arm. This is NOT read-only."
        )
    )
    parser.add_argument("--left-arm-ip", required=True)
    parser.add_argument("--right-arm-ip", required=True)
    parser.add_argument("--operator-authorization-id", required=True)
    parser.add_argument("--execute-stop-only", action="store_true")
    parser.add_argument("--confirm-arms-stationary", action="store_true")
    parser.add_argument("--confirm-estop-ready", action="store_true")
    parser.add_argument("--state-interval-sec", type=float, default=0.25)
    parser.add_argument("--max-stationary-drift-rad", type=float, default=0.005)
    parser.add_argument("--authorization-valid-sec", type=float, default=60.0)
    return parser.parse_args()


def _require_explicit_scope(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "execute_stop_only",
            "confirm_arms_stationary",
            "confirm_estop_ready",
        )
        if not bool(getattr(args, name, False))
    ]
    if missing:
        raise RuntimeError(
            "Refusing M5.8-b0 hardware state change; missing confirmations: "
            + ", ".join(missing)
        )
    if not np.isfinite(float(args.state_interval_sec)) or float(args.state_interval_sec) <= 0:
        raise ValueError("state-interval-sec must be finite and positive.")
    if not np.isfinite(float(args.max_stationary_drift_rad)) or float(
        args.max_stationary_drift_rad
    ) <= 0:
        raise ValueError("max-stationary-drift-rad must be finite and positive.")
    if not np.isfinite(float(args.authorization_valid_sec)) or float(
        args.authorization_valid_sec
    ) <= 0:
        raise ValueError("authorization-valid-sec must be finite and positive.")


def _build_stop_only_backend(
    *,
    side: str,
    arm_ip: str,
    binding: RM65StopOnlyLegacyBinding,
) -> RM65ArmBackend:
    return RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip=str(arm_ip),
            side=str(side),
            default_tcp_frame=TOOL_TCP_FRAME,
            connect_hardware=True,
            initial_qpos=None,
            configure_singularity_thresholds=False,
            force_position_control=False,
            background_streaming_enabled=False,
        ),
        # A missing adapter would preserve the old unmediated motion path.
        # The stop-only profile instead installs an explicit deny-all motion
        # adapter and a narrowly authorized safety-control adapter.
        pre_sdk_dispatch_adapter=binding.motion_adapter,
        pre_sdk_control_adapter=binding.control_adapter,
    )


def _read_fresh_state(backend: RM65ArmBackend, *, side: str) -> np.ndarray:
    state = backend.refresh_state_from_hardware()
    if state.get("hardware_result_code") != 0:
        raise RuntimeError(f"{side}: RM65 state read failed: {state.get('hardware_result_text')}")
    if state.get("hardware_arm_err") != 0 or state.get("hardware_sys_err") != 0:
        raise RuntimeError(
            f"{side}: RM65 error state arm_err={state.get('hardware_arm_err')} "
            f"sys_err={state.get('hardware_sys_err')}"
        )
    qpos = state.get("hardware_qpos")
    if qpos is None:
        raise RuntimeError(f"{side}: no fresh hardware qpos was returned.")
    q = np.asarray(qpos, dtype=float).reshape(-1)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise RuntimeError(f"{side}: hardware qpos is not six finite values.")
    q_min = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    q_max = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    if np.any(q < q_min) or np.any(q > q_max):
        raise RuntimeError(f"{side}: fresh qpos violates the active URDF limits.")
    return q


def _close_without_hiding_primary_error(backends: list[RM65ArmBackend]) -> list[str]:
    errors: list[str] = []
    for backend in reversed(backends):
        try:
            backend.close()
        except Exception as exc:  # pragma: no cover - hardware-only evidence
            errors.append(f"{backend.name}:{type(exc).__name__}:{exc}")
    return errors


def main() -> None:
    args = _parse_args()
    _require_explicit_scope(args)
    print("[M5.8-b0] AUTHORIZED STATE CHANGE: one Arm_Slow_Stop per stationary arm")
    print("[M5.8-b0] motion/home/enable/rearm/force-position/background-streaming are denied")

    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    now = time.monotonic()
    authorization = RM65StopOnlyHardwareAuthorization.create(
        operator_authorization_id=args.operator_authorization_id,
        execution_manifest=manifest,
        dispatch_inventory=inventory,
        control_domain_id="rm65-arms",
        embodiment_instance_id="rm65-bimanual-0",
        allowed_sides=("left", "right"),
        issued_at=now,
        expires_at=now + float(args.authorization_valid_sec),
    )
    binding = RM65StopOnlyLegacyBinding(
        runtime_session_id=f"rm65-stop-only-{uuid.uuid4()}",
        owner_runtime_id="legacy-rm65-stop-only-preflight",
        authorization=authorization,
    )

    backends: list[RM65ArmBackend] = []
    evidence: dict[str, object] = {
        "authorization": asdict(authorization),
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest.semantic_content_hash,
        "inventory_hash": inventory.semantic_content_hash,
        "sides": {},
    }
    primary_error: BaseException | None = None
    try:
        by_side = {
            "left": _build_stop_only_backend(
                side="left",
                arm_ip=args.left_arm_ip,
                binding=binding,
            ),
            "right": _build_stop_only_backend(
                side="right",
                arm_ip=args.right_arm_ip,
                binding=binding,
            ),
        }
        for side in ("left", "right"):
            backend = by_side[side]
            backend.start()
            backends.append(backend)
            if not callable(getattr(backend.arm, "Arm_Slow_Stop", None)):
                raise RuntimeError(f"{side}: SDK does not expose Arm_Slow_Stop; refusing fallback.")

        before: dict[str, np.ndarray] = {}
        for side, backend in by_side.items():
            before[side] = _read_fresh_state(backend, side=side)
        time.sleep(float(args.state_interval_sec))
        for side, backend in by_side.items():
            second = _read_fresh_state(backend, side=side)
            drift = float(np.max(np.abs(second - before[side])))
            if drift > float(args.max_stationary_drift_rad):
                raise RuntimeError(
                    f"{side}: arm is not stationary; max drift {drift:.6f} rad exceeds "
                    f"{float(args.max_stationary_drift_rad):.6f} rad."
                )
            evidence["sides"][side] = {
                "qpos_before": second.tolist(),
                "stationary_drift_rad": drift,
            }

        for side in ("left", "right"):
            backend = by_side[side]
            started = time.monotonic()
            backend.hold_motion("m5_8b0_stop_only:operator_confirmed")
            latency = time.monotonic() - started
            after = _read_fresh_state(backend, side=side)
            side_evidence = evidence["sides"][side]
            side_evidence.update(
                {
                    "slow_stop_latency_sec": latency,
                    "qpos_after": after.tolist(),
                    "post_stop_delta_rad": float(
                        np.max(np.abs(after - np.asarray(side_evidence["qpos_before"])))
                    ),
                    "send_result_code": backend.get_state().get("send_result_code"),
                    "send_result_text": backend.get_state().get("send_result_text"),
                }
            )

        evidence["binding"] = asdict(binding.snapshot())
        evidence["receipts"] = [asdict(receipt) for receipt in binding.receipts]
        evidence["status"] = "stop_only_completed"
    except BaseException as exc:
        primary_error = exc
        evidence["status"] = "failed"
        evidence["error"] = f"{type(exc).__name__}:{exc}"
    finally:
        close_errors = _close_without_hiding_primary_error(backends)
        if close_errors:
            evidence["close_errors"] = close_errors

    print("M5_8B0_EVIDENCE=" + json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    if primary_error is not None:
        raise primary_error
    if evidence.get("close_errors"):
        raise RuntimeError("RM65 close reported errors; inspect M5_8B0_EVIDENCE.")


if __name__ == "__main__":
    main()
