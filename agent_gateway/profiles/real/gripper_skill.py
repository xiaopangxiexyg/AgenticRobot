"""Real DMTAC adapter and contract builder for atomic gripper control."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agent_gateway.skills.gripper.deployment import (
    GripperSkillContractBundle,
    GripperSkillDeploymentBinding,
    build_gripper_skill_contract_bundle,
)
from agent_gateway.skills.gripper.runtime import (
    GripperAgentGatewaySession,
    GripperCommandReceipt,
    GripperHoldReceipt,
    GripperReadiness,
    GripperStateSample,
    assemble_gripper_agent_gateway_session,
    hold_gripper_ports,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

_LOG = logging.getLogger(__name__)


class RealDmtacGripperControlPort:
    """Adapt an already-started DMTAC backend without owning its lifecycle."""

    component_id = "right_gripper"
    target_kind = "real"

    def __init__(
        self, *, backend: Any, clock=time.monotonic,
        component_id: str = "right_gripper",
        command_submission_timeout_sec: float = 0.1,
        hold_confirmation_timeout_sec: float = 0.5,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        for method_name in (
            "is_ready",
            "readiness_report",
            "send_fixed_joint_pos",
            "read_gripper_state_now",
            "get_state",
            "hold_gripper_motion",
        ):
            if not callable(getattr(backend, method_name, None)):
                raise TypeError(f"backend must expose {method_name}().")
        self.backend = backend
        self.clock = clock
        self._sequence = 0
        for name, value in (
            ("command_submission_timeout_sec", command_submission_timeout_sec),
            ("hold_confirmation_timeout_sec", hold_confirmation_timeout_sec),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.command_submission_timeout_sec = float(command_submission_timeout_sec)
        self.hold_confirmation_timeout_sec = float(hold_confirmation_timeout_sec)
        self._motion_guard: Callable[[], bool] | None = None
        self._last_command_attempt_ref: str | None = None
        self._last_hold_receipt: GripperHoldReceipt | None = None
        config = getattr(backend, "config", None)
        side = str(getattr(config, "side", "right"))
        if component_id not in {"left_gripper", "right_gripper"}:
            raise ValueError("component_id must identify a left or right gripper.")
        if side + "_gripper" != component_id:
            raise ValueError(
                f"real gripper Skill is bound to {component_id}; "
                f"backend side does not match: {side!r}."
            )
        self.component_id = component_id
        if bool(getattr(config, "observation_only", False)):
            raise ValueError(
                "real gripper Skill cannot use an observation-only backend."
            )
        if getattr(config, "connect_hardware", True) is not True:
            raise ValueError(
                "real gripper Skill requires a hardware-connected backend."
            )
        if getattr(config, "disconnected_state_scalar", None) is not None:
            raise ValueError("real gripper Skill rejects assumed disconnected state.")
        if getattr(config, "send_commands", True) is not True:
            raise ValueError("real gripper Skill requires command-enabled hardware.")
        identity = {
            "adapter": "real_dmtac_gripper_control.v1",
            "component_id": self.component_id,
            "side": side,
            "gripper_port": str(getattr(config, "gripper_port", "bound")),
            "gripper_id": int(getattr(config, "gripper_id", 1)),
            "command_submission_timeout_sec": self.command_submission_timeout_sec,
            "hold_confirmation_timeout_sec": self.hold_confirmation_timeout_sec,
            "speed": float(getattr(config, "speed", 1.0)),
            "torque": float(getattr(config, "torque", 0.2)),
            "max_open_mm": float(getattr(config, "max_open_mm", 70.0)),
            "continuous_grasp": bool(getattr(config, "continuous_grasp", True)),
        }
        self.port_binding_hash = canonical_content_hash(
            identity,
            domain="dexteleop.agent_skill.gripper_port_binding.v1",
        )

    def _ref(self, kind: str, payload: dict[str, object]) -> str:
        self._sequence += 1
        digest = canonical_content_hash(
            {"kind": kind, "sequence": self._sequence,
             "component_id": self.component_id, **payload},
            domain="dexteleop.real.gripper_skill_evidence.v1",
        )
        return f"dmtac-gripper-{kind}:{digest}"

    def readiness(self) -> GripperReadiness:
        report = self.backend.readiness_report()
        ready = bool(self.backend.is_ready()) and bool(report.get("ready"))
        ref = self._ref(
            "readiness",
            {"ready": ready, "reason": str(report.get("reason", ""))},
        )
        return GripperReadiness(
            ready=ready,
            lower_operation_id=f"dmtac-gripper:{self.component_id}:live-session",
            evidence_refs=(ref,),
        )

    def command_close(
        self, target_close: float, *, cancel_requested: Callable[[], bool] | None = None,
    ) -> GripperCommandReceipt:
        self._require_arm_hold()
        target = float(target_close)
        # A hardware write may change the scene even if its ACK is lost.
        self._last_command_attempt_ref = self._ref(
            "command_attempt", {"target_close": target}
        )
        self._last_hold_receipt = None
        started_at = time.monotonic()
        deadline = started_at + self.command_submission_timeout_sec
        attempts = 0
        diagnostic: dict[str, object] = {}
        accepted = False
        while True:
            if cancel_requested is not None and cancel_requested():
                diagnostic = {"gripper_last_send_status": "preempted_before_dispatch"}
                break
            # A busy retry can outlive the arm's original fresh HOLD cut.
            self._require_arm_hold()
            attempts += 1
            accepted = bool(self.backend.send_fixed_joint_pos(target))
            if accepted:
                break
            try:
                state = self.backend.get_state(
                    include_tactile_arrays=False, include_tactile_vector=False,
                )
                diagnostic = {
                    key: state.get(key) for key in (
                        "gripper_last_send_status", "gripper_last_send_skip_reason",
                        "gripper_send_elapsed_s", "gripper_send_lock_wait_s",
                        "gripper_error_code", "gripper_status_code", "gripper_status_text",
                        "gripper_is_moving", "gripper_measured_scalar", "telemetry_error",
                    )
                }
            except Exception as error:  # noqa: BLE001 - preserve failure and still allow HOLD.
                diagnostic = {"diagnostic_error": f"{type(error).__name__}: {error}"}
            remaining = deadline - time.monotonic()
            # io_busy means the nonblocking backend did not write to the device.
            # Never retry hardware faults, SDK exceptions or uncertain ACKs.
            if diagnostic.get("gripper_last_send_status") != "io_busy" or remaining <= 0:
                break
            time.sleep(min(0.005, remaining))
            if time.monotonic() >= deadline:
                break
        command_ref = self._ref("command", {"target_close": target, "accepted": accepted})
        if not accepted:
            _LOG.error(
                "DMTAC Skill command rejected: component=%s target_close=%s "
                "attempts=%s elapsed_sec=%.6f receipt=%s diagnostic=%r",
                self.component_id, target, attempts, time.monotonic() - started_at,
                command_ref, diagnostic,
            )
        return GripperCommandReceipt(
            accepted=accepted,
            lower_operation_id=f"dmtac-gripper-command:{self._sequence}",
            evidence_refs=(command_ref,),
        )

    def observe_state(self) -> GripperStateSample:
        read_started = time.monotonic()
        reading = self.backend.read_gripper_state_now()
        if time.monotonic() - read_started > 0.2:
            raise RuntimeError(
                "DMTAC fresh gripper state read exceeded its 200 ms budget."
            )
        if reading is None:
            raise RuntimeError("DMTAC fresh gripper state read is unavailable.")
        if not isinstance(reading, Mapping) or "timestamp" not in reading:
            raise RuntimeError(
                "DMTAC fresh gripper state lacks its acquisition timestamp."
            )
        measured_at = float(reading["timestamp"])
        state = self.backend.get_state(
            include_tactile_arrays=False,
            include_tactile_vector=False,
        )
        # Include cached-state access in the acquisition window. A slow getter
        # must not make an old read look fresh by assigning a later timestamp.
        if time.monotonic() - read_started > 0.2:
            raise RuntimeError("DMTAC fresh gripper state read exceeded its 200 ms budget.")
        age = time.time() - measured_at
        if not math.isfinite(age) or not 0.0 <= age <= 0.2:
            raise RuntimeError("DMTAC measured gripper state is stale or future-dated.")
        measured = state.get("gripper_measured_scalar")
        if measured is None:
            raise RuntimeError("DMTAC backend has no measured gripper scalar.")
        error_code = state.get("gripper_error_code")
        fault_code = (
            None if error_code in (None, 0) else f"dmtac_error_{int(error_code)}"
        )
        moving_raw = state.get("gripper_is_moving")
        moving = None if moving_raw is None else bool(moving_raw)
        observed_at = float(self.clock()) - age
        return GripperStateSample(
            measured_close=float(measured),
            moving=moving,
            fault_code=fault_code,
            observed_at=observed_at,
            evidence_ref=self._ref(
                "sample",
                {
                    "measured_close": float(measured),
                    "moving": moving,
                    "fault_code": fault_code,
                    "observed_at": observed_at,
                },
            ),
        )

    def poll_state(self) -> GripperStateSample:
        self._require_arm_hold()
        return self.observe_state()

    def bind_motion_guard(self, guard: Callable[[], bool]) -> None:
        if self._motion_guard is not None or not callable(guard):
            raise ValueError("Gripper arm-HOLD owner is bound exactly once.")
        self._motion_guard = guard

    @property
    def last_command_attempt_ref(self) -> str | None:
        return self._last_command_attempt_ref

    def _require_arm_hold(self) -> None:
        if self._motion_guard is not None and not self._motion_guard():
            raise RuntimeError(
                "Gripper motion requires a fresh, healthy arm owner under confirmed HOLD."
            )

    def hold(self, *, reason: str) -> GripperHoldReceipt:
        self._last_hold_receipt = None
        started_at = time.monotonic()
        deadline = started_at + self.hold_confirmation_timeout_sec
        stop_requested_at = float(self.clock())
        acknowledged = bool(self.backend.hold_gripper_motion(reason))
        sample = None
        samples = []
        succeeded = False
        while acknowledged and time.monotonic() < deadline:
            try:
                sample = self.observe_state()
            except Exception:
                _LOG.exception("DMTAC Skill HOLD state read failed: component=%s reason=%s", self.component_id, reason)
                raise
            samples.append(sample.evidence_ref)
            # A pre-stop sample cannot confirm this stop request.
            if sample.observed_at < stop_requested_at or sample.fault_code is not None:
                break
            if sample.moving is False:
                succeeded = time.monotonic() <= deadline
                break
            if sample.moving is None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.01, remaining))
        if not succeeded:
            _LOG.error(
                "DMTAC Skill HOLD not confirmed: component=%s reason=%s "
                "stop_ack=%s samples=%s elapsed_sec=%.6f moving=%s fault=%s measured_close=%s",
                self.component_id, reason, acknowledged, len(samples),
                time.monotonic() - started_at, sample.moving if sample else None,
                sample.fault_code if sample else None, sample.measured_close if sample else None,
            )
        receipt = GripperHoldReceipt(
            succeeded=succeeded,
            evidence_refs=(
                self._ref("hold", {"reason": reason, "succeeded": succeeded}),
                *samples,
            ),
        )
        self._last_hold_receipt = receipt
        return receipt

    @property
    def last_hold_receipt(self) -> GripperHoldReceipt | None:
        return self._last_hold_receipt


def stop_arm_and_grippers(
    *,
    arm_stop: Callable[[], Any],
    ports: Mapping[str, RealDmtacGripperControlPort],
    reason: str,
) -> tuple[Any, GripperHoldReceipt]:
    """Attempt both independent stop boundaries; surface all failed calls."""
    if not ports:
        raise ValueError("Composed gripper stop requires configured hands.")
    errors = []
    interrupted = None
    arm = None
    hand_hold = None
    try:
        arm = arm_stop()
    except (KeyboardInterrupt, SystemExit) as error:
        if interrupted is None:
            interrupted = error
    except Exception as error:  # noqa: BLE001 - one failed stop must not omit the other.
        errors.append(error)
    try:
        hand_hold = hold_gripper_ports(ports=ports, reason=reason)
    except (KeyboardInterrupt, SystemExit) as error:
        if interrupted is None:
            interrupted = error
    except Exception as error:  # noqa: BLE001 - both lower stop attempts must run.
        errors.append(error)
    if interrupted is not None:
        raise interrupted
    if errors:
        raise RuntimeError(
            "Composed stop failed: " + "; ".join(str(error) for error in errors)
        ) from errors[0]
    return arm, hand_hold



def stop_arm_and_gripper(
    *, arm_stop: Callable[[], Any], port: RealDmtacGripperControlPort, reason: str
) -> tuple[Any, GripperHoldReceipt]:
    """Retain the v1 single-hand caller on the shared stop boundary."""
    return stop_arm_and_grippers(
        arm_stop=arm_stop, ports={port.component_id: port}, reason=reason
    )

class ExclusiveRealGripperResources:
    """Own the optional hand beside the existing lower application resources."""

    def __init__(
        self, *, lower: Any, port: RealDmtacGripperControlPort | None = None,
        ports: Mapping[str, RealDmtacGripperControlPort] | None = None
    ) -> None:
        if port is not None:
            if ports is not None:
                raise ValueError("Select single v1 or explicit bilateral v2 resources.")
            ports = {port.component_id: port}
        if not ports or any(
            not isinstance(port, RealDmtacGripperControlPort)
            or key != port.component_id
            for key, port in ports.items()
        ):
            raise TypeError("Gripper resources require correctly keyed real adapters.")
        for resource in (lower, *(port.backend for port in ports.values())):
            if not all(
                callable(getattr(resource, method, None))
                for method in ("start", "close")
            ):
                raise TypeError(
                    "Application resources must expose start() and close()."
                )
        if any(
            bool(getattr(getattr(port.backend, "config", None), name, False))
            for port in ports.values()
            for name in (
                "open_on_start",
                "clear_fault_on_start",
                "reconnect_on_error",
                "auto_recover_internal_communication_fault",
                "tactile_reset_on_start",
            )
        ):
            raise ValueError(
                "Gripper Skill resources forbid startup opening/reset and automatic recovery."
            )
        self.lower = lower
        self.ports = dict(ports)
        self.port = port
        self._lower_start_attempted = False
        self._hand_attempted: set[str] = set()
        self._hand_closed: set[str] = set()
        self._lower_closed = False

    def start(self) -> None:
        if self._lower_start_attempted or self._hand_attempted:
            raise RuntimeError("Gripper application resources start once.")
        self._lower_start_attempted = True
        self.lower.start()
        try:
            for component, port in self.ports.items():
                self._hand_attempted.add(component)
                port.backend.start()
                if not port.readiness().ready:
                    raise RuntimeError(f"{component} startup is not ready.")
                sample = port.observe_state()
                if sample.fault_code is not None or sample.moving is not False:
                    raise RuntimeError(
                        f"{component} must be fault-free and stationary before transport publication."
                    )
        except Exception as start_error:
            try:
                self.close()
            except Exception as close_error:  # noqa: BLE001 - preserve startup and cleanup failures.
                raise RuntimeError(
                    f"Gripper startup failed: {start_error}; cleanup failed: {close_error}"
                ) from start_error
            raise

    def close(self) -> None:
        errors = []
        interrupted = None
        for component in reversed(tuple(self.ports)):
            if component not in self._hand_attempted or component in self._hand_closed:
                continue
            try:
                self.ports[component].backend.close()
                self._hand_closed.add(component)
            except (KeyboardInterrupt, SystemExit) as error:
                if interrupted is None:
                    interrupted = error
            except Exception as error:  # noqa: BLE001 - preserve independent cleanup attempts.
                errors.append(f"{component}:{error}")
        if interrupted is not None:
            raise interrupted
        if errors:
            raise RuntimeError("Gripper resource close failed: " + "; ".join(errors))
        if self._lower_start_attempted and not self._lower_closed:
            self.lower.close()
            self._lower_closed = True


def build_real_dmtac_grippers_skill_contract_bundle(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    ports: Mapping[str, RealDmtacGripperControlPort],
) -> GripperSkillContractBundle:
    if set(ports) != {"left_gripper", "right_gripper"} or any(
        key != port.component_id for key, port in ports.items()
    ):
        raise ValueError(
            "The composed gripper deployment requires both correctly bound hands."
        )
    paths = [
        str(Path(port.backend.config.gripper_port).resolve()) for port in ports.values()
    ]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Left and right grippers cannot independently own the same serial port."
        )
    hashes = {key: port.port_binding_hash for key, port in ports.items()}
    return build_gripper_skill_contract_bundle(
        GripperSkillDeploymentBinding(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            component_id="grippers",
            target_binding_id=target_binding_id,
            target_kind="real",
            current_mode="real_active",
            executor_id="real.dmtac-grippers-control.v2",
            port_binding_hash=canonical_content_hash(
                hashes, domain="dexteleop.gripper_ports.v2"
            ),
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
            skill_version="2",
            port_binding_hash_by_component=hashes,
        )
    )



def build_real_dmtac_gripper_skill_contract_bundle(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    port: RealDmtacGripperControlPort,
) -> GripperSkillContractBundle:
    if not isinstance(port, RealDmtacGripperControlPort):
        raise TypeError("port must be a RealDmtacGripperControlPort.")
    return build_gripper_skill_contract_bundle(
        GripperSkillDeploymentBinding(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            component_id=port.component_id,
            target_binding_id=target_binding_id,
            target_kind="real",
            current_mode="real_active",
            executor_id="control-station.real-dmtac-gripper-control.v1",
            port_binding_hash=port.port_binding_hash,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
    )


def assemble_real_dmtac_gripper_agent_gateway_session(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    port: RealDmtacGripperControlPort,
    allowed_client_ids: tuple[str, ...],
    current_station_revision,
    clock_domain_id: str,
    convergence=None,
) -> GripperAgentGatewaySession:
    bundle = build_real_dmtac_gripper_skill_contract_bundle(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        port=port,
    )
    return assemble_gripper_agent_gateway_session(
        contract_bundle=bundle,
        port=port,
        allowed_client_ids=allowed_client_ids,
        current_station_revision=current_station_revision,
        clock_domain_id=clock_domain_id,
        convergence=convergence,
        clock=port.clock,
    )


__all__ = [
    "ExclusiveRealGripperResources",
    "RealDmtacGripperControlPort",
    "assemble_real_dmtac_gripper_agent_gateway_session",
    "build_real_dmtac_gripper_skill_contract_bundle",
    "build_real_dmtac_grippers_skill_contract_bundle",
]
