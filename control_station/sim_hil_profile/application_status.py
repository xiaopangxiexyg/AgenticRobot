"""Read-only status projection for the Sim-HIL application lifecycle.

The mixin has no resource construction or control authority.  It only projects
an already-owned application state into GUI-safe snapshots while preserving the
facade payload contract.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from .status import (
    SIM_HIL_STATUS_SCHEMA_V1,
    _composition_step_status,
    _emission_status,
    _execution_difference_status,
    _human_channel_status,
    _human_gripper_target_status,
    _input_submission_status,
    _mix_status,
    _mujoco_gripper_execution_status,
    _runtime_contract_status,
    _target_execution_status,
    _vive_receiver_status,
)
from .validation import _finite


class _SimHILApplicationStatusMixin:
    def status(self) -> dict[str, Any]:
        with self._lock:
            now = _finite(self.clock(), name="status.now")
            wall_now = time.time()
            readiness = self.readiness_provider.snapshot(now=now)
            required_resolver = getattr(
                self.composition,
                "required_readiness_checks",
                None,
            )
            failure_resolver = getattr(
                self.composition,
                "readiness_failures",
                None,
            )
            readiness_selection_error = None
            if callable(required_resolver) and callable(failure_resolver):
                try:
                    required_readiness = tuple(
                        required_resolver(target_effective_at=now)
                    )
                    readiness_failures = tuple(
                        failure_resolver(
                            readiness,
                            target_effective_at=now,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - status remains diagnostic.
                    readiness_selection_error = (
                        f"{type(error).__name__}:{error}"
                    )
                    required_readiness = tuple(
                        readiness.detail_by_check
                    )
                    readiness_failures = tuple(
                        readiness.failed_checks
                    )
            else:
                required_readiness = tuple(readiness.detail_by_check)
                readiness_failures = tuple(readiness.failed_checks)
            policy_status = dict(self.policy_source.status(now=now))
            policy_status["refresh_state"] = (
                "suspended_by_application_hold"
                if self._started and self._held
                else "active"
                if self._started
                else "stopped"
            )
            policy_emission = getattr(self.policy_source, "last_emission", None)
            policy_status["last_emission"] = _emission_status(
                policy_emission,
                now=now,
            )

            human_state = getattr(self.human_source, "session_state", None)
            human_session = getattr(self.human_source, "session", None)
            human_binding = getattr(self.human_source, "binding_receipt", None)
            human_emission = getattr(self.human_source, "last_emission", None)
            last_human_step = (
                self._last_human_input_check_step
                if self._last_cycle is None or self._held
                else getattr(self._last_cycle, "human_step", None)
            )
            human_status = {
                "session_state": getattr(human_state, "value", str(human_state)),
                "source_started": bool(getattr(self.human_source, "started", False)),
                "last_observed_emission_sequence": getattr(
                    self.human_source,
                    "last_observed_emission_sequence",
                    None,
                ),
                "last_rejection_reason": getattr(
                    self.human_source,
                    "last_rejection_reason",
                    None,
                ),
                "last_emission": _emission_status(human_emission, now=now),
                "last_emission_age_sec": (
                    None
                    if human_emission is None
                    else now - float(human_emission.emitted_at)
                ),
                "binding": {
                    "source_contract_hash": getattr(
                        human_binding, "source_contract_hash", None
                    ),
                    "observer_binding_id": getattr(
                        human_binding, "observer_binding_id", None
                    ),
                    "semantic_content_hash": getattr(
                        human_binding, "semantic_content_hash", None
                    ),
                    "rig_absent": bool(getattr(human_binding, "rig_absent", False)),
                    "observer_exclusive": bool(
                        getattr(human_binding, "observer_exclusive", False)
                    ),
                    "authoritative": bool(
                        getattr(human_binding, "authoritative", True)
                    ),
                },
                "session_safety": {
                    "home_recalibration_required": bool(
                        getattr(human_session, "home_recalibration_required", False)
                    ),
                    "motion_hold_mode": getattr(
                        human_session, "motion_hold_mode", ""
                    ),
                    "motion_hold_reason": getattr(
                        human_session, "motion_hold_reason", ""
                    ),
                    "tracker_loss_duration_sec": float(
                        getattr(human_session, "tracker_loss_duration_sec", 0.0)
                    ),
                    "tracker_loss_channels": list(
                        getattr(human_session, "tracker_loss_channels", ())
                    ),
                },
                "channels": _human_channel_status(
                    last_human_step,
                    now=wall_now,
                ),
                "vive_receiver": _vive_receiver_status(
                    human_session,
                    wall_now=wall_now,
                ),
                "last_step": None
                if last_human_step is None
                else {
                    "emission_valid": bool(
                        getattr(last_human_step, "emission_valid", False)
                    ),
                    "rejection_reason": getattr(
                        last_human_step, "rejection_reason", None
                    ),
                    "command_dispatched": bool(
                        getattr(last_human_step, "command_dispatched", False)
                    ),
                    "authoritative": bool(
                        getattr(last_human_step, "authoritative", True)
                    ),
                },
                "passive_observation_error": (
                    self._last_passive_observation_error
                ),
            }

            vive_role_mapping: dict[str, str] = {}
            try:
                from teleop_runtime.configs import (
                    vive_source_role_mapping,
                )

                if human_session is not None:
                    vive_role_mapping = vive_source_role_mapping(
                        human_session
                    )
            except (RuntimeError, TypeError, ValueError):
                vive_role_mapping = {}

            result: dict[str, Any] = {
                "schema": SIM_HIL_STATUS_SCHEMA_V1,
                "sampled_at": now,
                "profile_id": "control_station_mujoco_sim_hil_compound_v1",
                "control_profile_id": self.config.control_profile_id,
                "enabled": bool(self.config.enabled),
                "started": self._started,
                "lifecycle": (
                    "closed"
                    if self._closed
                    else "episode_ended"
                    if self._started and self._episode_ended
                    else "held"
                    if self._started and self._held
                    else "running"
                    if self._started
                    else "stopped"
                ),
                "held": self._held,
                "episode_ended": self._episode_ended,
                "startup": {
                    "restart_required": self._restart_required,
                    "first_failure": self._startup_failure,
                    "closed": self._closed,
                    "mujoco_constructed": True,
                },
                "first_control_fault": self._first_control_fault,
                "visualization": (
                    {
                        "enabled": False,
                        "state": "disabled",
                        "running": False,
                        "sync_count": 0,
                        "last_error": None,
                    }
                    if self.viewer is None
                    else dict(self.viewer.status())
                ),
                "control_schedule": {
                    "next_decision_at": self._next_decision_at,
                    "overrun_dropped_cycles": self._overrun_dropped_cycles,
                    "last_overrun": (
                        None
                        if self._last_schedule_overrun is None
                        else dict(self._last_schedule_overrun)
                    ),
                    "target_admission_margin_sec": (
                        self.config.target_admission_margin_sec
                    ),
                    "target_lead_time_sec": (
                        getattr(
                            getattr(
                                self.runtime,
                                "reference_activation_spec",
                                None,
                            ),
                            "target_lead_time_sec",
                            None,
                        )
                    ),
                    "proposal_horizon_sec": (
                        getattr(
                            self.runtime.config,
                            "proposal_horizon_sec",
                            None,
                        )
                    ),
                    "last_cycle_timing": (
                        None
                        if self._last_cycle_timing is None
                        else dict(self._last_cycle_timing)
                    ),
                },
                "blend_failure_fallback": {
                    "selected_source": self._blend_failure_fallback,
                    "grace_sec": self._blend_failure_grace_sec,
                    "failure_window_started_at": (
                        self._blend_failure_window_started_at
                    ),
                    "consecutive_failure_count": (
                        self._blend_failure_consecutive_count
                    ),
                    "latched_source": (
                        self._blend_failure_fallback_latched_source
                    ),
                    "last_activation": (
                        None
                        if self._last_blend_failure_fallback is None
                        else copy.deepcopy(
                            self._last_blend_failure_fallback
                        )
                    ),
                },
                "human_timeout_fallback": {
                    "timeout_sec": self.config.human_emission_timeout_sec,
                    "selected_source": self._human_timeout_fallback,
                    "latched": self._human_timeout_fallback_latched,
                    "last_activation": (
                        None
                        if self._last_human_timeout_fallback is None
                        else copy.deepcopy(self._last_human_timeout_fallback)
                    ),
                },
                "last_error": self._last_error,
                "execution_target": "mujoco_only",
                "real_arm_motion": False,
                "single_execution_owner": "MujocoDeployRuntime/composer",
                "calibration": dict(self.calibration_binding),
                "task_initialization": dict(
                    self.task_initialization_info
                ),
                "source_configuration": {
                    "vive_port": int(self.config.vive_port),
                    "vive_clock_sync_port": self.config.vive_clock_sync_port,
                    "human_input_attached": self.human_input_attached,
                    "active_components": list(
                        getattr(
                            self.runtime.action_schema,
                            "component_ids",
                            (),
                        )
                    ),
                    "vive_source_role_by_output_channel": dict(
                        vive_role_mapping
                    ),
                    "vive_source_swap_count": self._vive_source_swap_count,
                    "input_device_bindings": [
                        {
                            "component_id": "right_gripper",
                            "requested_device_kind": getattr(
                                self.gripper_input_selection,
                                "requested_source",
                                self.config.gripper_input_source,
                            ),
                            "device_kind": getattr(
                                self.gripper_input_selection,
                                "resolved_source",
                                self.config.gripper_input_source,
                            ),
                            "endpoint": getattr(
                                self.gripper_input_selection,
                                "q6_port",
                                self.config.right_q6_port,
                            ),
                            "fallback_reason": getattr(
                                self.gripper_input_selection,
                                "fallback_reason",
                                None,
                            ),
                        }
                    ],
                },
                "contracts": _runtime_contract_status(self.runtime, self.composition),
                "readiness": {
                    "ready": not readiness_failures,
                    "required_checks": list(required_readiness),
                    "failed_checks": list(readiness_failures),
                    "all_evidence_ready": readiness.ready,
                    "all_evidence_failed_checks": list(
                        readiness.failed_checks
                    ),
                    "selection_error": readiness_selection_error,
                    "detail_by_check": dict(readiness.detail_by_check),
                    "semantic_content_hash": readiness.semantic_content_hash,
                },
                "policy": policy_status,
                "policy_execution_feedback": (
                    None
                    if self._last_policy_feedback_status is None
                    else dict(self._last_policy_feedback_status)
                ),
                "human": human_status,
                "gripper": {
                    "human_target": _human_gripper_target_status(
                        human_emission
                    ),
                    "execution": _mujoco_gripper_execution_status(
                        getattr(self.runtime, "rig", None),
                        gripper_name=(
                            getattr(
                                self.task_config,
                                "right_gripper_name",
                                "right_gripper",
                            )
                        ),
                    ),
                },
                "human_input_acceptance": (
                    None
                    if self._human_input_check_report is None
                    else dict(self._human_input_check_report)
                ),
                "runtime_home": (
                    None
                    if self._runtime_home_capture is None
                    else dict(self._runtime_home_capture)
                ),
                "execution_gate": {"status": "not_started"},
                "target_execution": None,
                "execution_difference": _execution_difference_status(
                    self.execution_difference_observer
                ),
                "preflight": None,
                "composition": {
                    "started": bool(getattr(self.composition, "started", False)),
                    "mix": _mix_status(
                        getattr(
                            self.composition,
                            "human_alpha_by_component",
                            {"*": self.config.initial_human_alpha},
                        )
                    ),
                    "activation": None,
                    "last_cycle": None,
                    "allocation_operations": (
                        dict(self.control_cycle_port.operation_status())
                        if callable(
                            getattr(
                                self.control_cycle_port,
                                "operation_status",
                                None,
                            )
                        )
                        else {"supported_signals": [], "active": False}
                    ),
                },
            }
            if self._last_preflight is not None:
                result["preflight"] = {
                    "passed": bool(getattr(self._last_preflight, "passed", False)),
                    "checks": dict(getattr(self._last_preflight, "checks", {})),
                    "reasons": list(getattr(self._last_preflight, "reasons", ())),
                    "semantic_content_hash": getattr(
                        self._last_preflight, "semantic_content_hash", None
                    ),
                }
            scope = getattr(self.composition, "activation_scope", None)
            if scope is not None:
                result["composition"]["activation"] = {
                    "activation_id": getattr(scope, "activation_id", None),
                    "session_epoch": getattr(scope, "session_epoch", None),
                    "input_barrier_at": getattr(scope, "input_barrier_at", None),
                    "clock_domain_id": getattr(scope, "clock_domain_id", None),
                    "semantic_content_hash": getattr(
                        scope, "semantic_content_hash", None
                    ),
                }
            if self._started:
                try:
                    gate = self.runtime.capture_execution_gate_snapshot(now=now)
                    result["execution_gate"] = {
                        "status": gate.status,
                        "reason": getattr(gate, "reason", None),
                        "gate_epoch": getattr(gate, "gate_epoch", None),
                        "source_event_ids": list(
                            getattr(gate, "source_event_ids", ())
                        ),
                        "semantic_content_hash": getattr(
                            gate, "semantic_content_hash", None
                        ),
                    }
                except Exception as error:  # noqa: BLE001 - status remains available.
                    result["execution_gate"] = {
                        "status": "error",
                        "status_error": f"{type(error).__name__}:{error}",
                    }
                capture_target_status = getattr(
                    self.runtime,
                    "capture_target_execution_status_snapshot",
                    None,
                )
                if callable(capture_target_status):
                    try:
                        result["target_execution"] = _target_execution_status(
                            capture_target_status(now=now)
                        )
                    except Exception as error:  # noqa: BLE001 - read-only UI remains available.
                        result["target_execution"] = {
                            "status_error": f"{type(error).__name__}:{error}",
                        }
            if self._last_cycle is not None:
                cycle = self._last_cycle
                result["composition"]["last_cycle"] = {
                    "cycle_index": int(cycle.cycle_index),
                    "semantic_content_hash": getattr(
                        cycle, "semantic_content_hash", None
                    ),
                    "human_submission": _input_submission_status(
                        cycle.human_submission
                    ),
                    "policy_submission": _input_submission_status(
                        cycle.policy_submission
                    ),
                    "composition_step": _composition_step_status(
                        cycle.composition_step
                    ),
                    "execution": {
                        "status": getattr(cycle.execution, "status", None),
                        "command_id": getattr(cycle.execution, "command_id", None),
                        "dispatch_receipt_id": getattr(
                            cycle.execution, "dispatch_receipt_id", None
                        ),
                        "target_execution_request_id": getattr(
                            getattr(cycle.execution, "target_execution_request", None),
                            "request_id",
                            None,
                        ),
                        "target_submission_receipt_id": getattr(
                            getattr(cycle.execution, "target_submission_receipt", None),
                            "receipt_id",
                            None,
                        ),
                        "target_completion_receipt_id": getattr(
                            getattr(cycle.execution, "target_completion_receipt", None),
                            "receipt_id",
                            None,
                        ),
                        "target_completion_outcome_code": getattr(
                            getattr(
                                cycle.execution,
                                "target_completion_receipt",
                                None,
                            ),
                            "outcome_code",
                            None,
                        ),
                        "resolution_failure": (
                            self._resolution_failure_status(
                                getattr(
                                    cycle.execution,
                                    "resolution_failure",
                                    None,
                                )
                            )
                        ),
                        "component_execution_target_id": getattr(
                            getattr(
                                getattr(cycle.execution, "target_dispatch_result", None),
                                "target",
                                None,
                            ),
                            "target_id",
                            None,
                        ),
                        "component_target_admission_receipt_id": getattr(
                            getattr(
                                getattr(cycle.execution, "target_dispatch_result", None),
                                "target_admission",
                                None,
                            ),
                            "receipt_id",
                            None,
                        ),
                        "component_target_activation_id": getattr(
                            getattr(
                                getattr(cycle.execution, "target_dispatch_result", None),
                                "target_activation",
                                None,
                            ),
                            "activation_id",
                            None,
                        ),
                        "target_supported_until": getattr(
                            getattr(
                                getattr(cycle.execution, "target_dispatch_result", None),
                                "target",
                                None,
                            ),
                            "supported_until",
                            None,
                        ),
                    },
                }
            return result

    def vive_status(self) -> dict[str, Any]:
        """Return a low-cost, read-only VIVE status snapshot.

        This path intentionally does not acquire the Station application lock
        and does not inspect policy/composition/execution state.  The source
        owns its own short-lived snapshot lock, so the GUI can poll this at a
        modest rate without competing with the control loop.
        """

        session = getattr(self.human_source, "session", None)
        return {
            "schema": "dexteleop.control_station.sim_hil_vive_status.v1",
            "sampled_at": time.time(),
            "vive_receiver": _vive_receiver_status(session),
        }
