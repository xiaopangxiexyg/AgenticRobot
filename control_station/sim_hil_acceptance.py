"""Typed, single-owner interactive control for the Sim-HIL application.

The browser submits application intents only.  The caller that owns the
control loop must call :meth:`SimHILAcceptanceController.process_next`; HTTP
threads never start devices, change composition parameters, seal capture, or
step the simulator directly.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.control_trace.session_capture import (
    CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1,
    CONTROL_TRACE_SESSION_CAPTURE_RUNTIME_CAPACITY_V1,
    CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1,
    ControlTraceEpisodeCapturePort,
    ControlTraceEpisodeSessionCapture,
    ControlTraceSessionFinalizationReceipt,
)
from control_station.hitl import (
    EXCLUSIVE_HITL_HUMAN_TO_POLICY,
    EXCLUSIVE_HITL_POLICY_TO_HUMAN,
    ExclusiveHITLSwitchEvidence,
    run_exclusive_hitl_switch_via_port,
)
from control_station.control_trace import project_exclusive_hitl_switch
from control_station.sim_hil_profile.exclusive_hitl import (
    SimHILExclusiveTransitionPort,
)
from control_station.sim_hil_profile.policy_human_allocation import (
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
    SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS,
)
from control_station.sim_hil_profile.profile_selection import (
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
    SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
)


SIM_HIL_ACCEPTANCE_STATUS_SCHEMA_V1 = (
    "dexteleop.control_station.sim_hil_acceptance_status.v1"
)
SIM_HIL_ACCEPTANCE_INTENT_SCHEMA_V1 = (
    "dexteleop.control_station.sim_hil_acceptance_intent.v1"
)
SIM_HIL_ACCEPTANCE_INTENT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.sim_hil_acceptance_intent_identity.v1"
)
SIM_HIL_ACCEPTANCE_STAGES_V1 = (
    "human_teleop",
    "ppo_only",
    "proportional_blend",
)
SIM_HIL_CONTROL_MODES_V1 = SIM_HIL_ACCEPTANCE_STAGES_V1
SIM_HIL_CAPTURE_PROFILE_ACCEPTANCE_V1 = "acceptance"
SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1 = "human_demonstration"
SIM_HIL_CAPTURE_PROFILES_V1 = (
    SIM_HIL_CAPTURE_PROFILE_ACCEPTANCE_V1,
    SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1,
)
SIM_HIL_BLEND_FAILURE_FALLBACKS_V1 = (
    "hold",
    "human",
    "policy",
)
SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1 = (
    "hold",
    "policy",
)
SIM_HIL_ALLOCATION_MODES_DEVELOPMENT = (
    "automatic",
    "base_only",
    "relative_overlay",
    "absolute_overlay_owner",
)
SIM_HIL_SELECTABLE_CONTROL_PROFILES = (
    SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
)
SIM_HIL_ACCEPTANCE_TRACE_RUNTIME_CAPACITY_V1 = (
    CONTROL_TRACE_SESSION_CAPTURE_RUNTIME_CAPACITY_V1
)
SIM_HIL_ACCEPTANCE_TRACE_REQUIREMENT_PROFILE_V1 = (
    "deployable_sim_immediate_component_policy_human_composer_v1"
)
SIM_HIL_HUMAN_DEMONSTRATION_TRACE_REQUIREMENT_PROFILE_V1 = (
    "deployable_sim_immediate_component_human_composer_v1"
)
STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1 = (
    CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1
)
STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1 = (
    CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1
)
StationControlTraceFinalizationReceipt = ControlTraceSessionFinalizationReceipt


def sim_hil_trace_requirement_profile_for_capture_profile(
    capture_profile: str,
) -> str:
    """Select the trace contract owned by one application capture mode."""

    normalized = _required(capture_profile, name="capture_profile")
    if normalized == SIM_HIL_CAPTURE_PROFILE_ACCEPTANCE_V1:
        return SIM_HIL_ACCEPTANCE_TRACE_REQUIREMENT_PROFILE_V1
    if normalized == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1:
        return SIM_HIL_HUMAN_DEMONSTRATION_TRACE_REQUIREMENT_PROFILE_V1
    raise ValueError(
        "capture_profile must be acceptance or human_demonstration."
    )


class StationControlTraceEpisodeSessionCapture(ControlTraceEpisodeSessionCapture):
    """Sim-HIL profile defaults over the lower-owned capture lifecycle.

    Persistence, WAL, episode binding, and terminal receipts stay in the base
    class.  This application leaf supplies only the established concrete
    session/domain/profile defaults so existing callers remain unchanged.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        trace_session_id: str | None = None,
        max_spool_bytes: int = 64 << 20,
        runtime_capacity: int = SIM_HIL_ACCEPTANCE_TRACE_RUNTIME_CAPACITY_V1,
        runtime_session_id: str = "control-station-sim-hil-ppo-session",
        control_domain_id: str = "mujoco-sim-hil",
        trace_requirement_profile_id: str = (
            SIM_HIL_ACCEPTANCE_TRACE_REQUIREMENT_PROFILE_V1
        ),
        clock_domain_id: str = "mujoco.sim_hil.monotonic",
        capture_purpose: str = "sim_hil_acceptance",
        legacy_episode_index: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            root,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            trace_requirement_profile_id=trace_requirement_profile_id,
            clock_domain_id=clock_domain_id,
            capture_purpose=capture_purpose,
            trace_session_id=trace_session_id,
            max_spool_bytes=max_spool_bytes,
            runtime_capacity=runtime_capacity,
            legacy_episode_index=legacy_episode_index,
            clock=clock,
        )


SIM_HIL_ACCEPTANCE_INTENT_KINDS_V1 = (
    "prepare_sim_preview",
    "select_control_profile",
    "select_sim_collision_stop",
    "start_runtime",
    "start_allocation_runtime",
    "set_control_mode",
    "set_allocation_mode",
    "request_residual_rebase_and_clear",
    "request_human_takeover",
    "request_policy_resume",
    "set_blend_failure_fallback",
    "set_human_timeout_fallback",
    "start_stage",
    "confirm_stage",
    "begin_vive_calibration",
    "capture_vive_calibration",
    "reset_vive_calibration_stage",
    "cancel_vive_calibration",
    "save_vive_calibration_profile",
    "cancel_operation",
    "swap_vive_sources",
    "hold",
    "rearm",
    "capture_runtime_home",
    "rebuild_runtime",
    "finalize_capture",
    "stop_runtime",
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class SimHILAcceptanceIntent:
    """One immutable operator request without raw action payloads."""

    client_id: str
    client_sequence: int
    expected_revision: int
    intent_kind: str
    stage_id: str | None = None
    control_mode: str | None = None
    allocation_mode: str | None = None
    control_profile_id: str | None = None
    sim_collision_stop_enabled: bool | None = None
    human_alpha: float | None = None
    fallback_source: str | None = None
    blend_failure_grace_sec: float | None = None
    profile_name: str | None = None
    reason: str = "operator_request"

    def __post_init__(self) -> None:
        client_id = _required(self.client_id, name="client_id")
        client_sequence = _nonnegative_integer(
            self.client_sequence,
            name="client_sequence",
        )
        expected_revision = _nonnegative_integer(
            self.expected_revision,
            name="expected_revision",
        )
        intent_kind = _required(self.intent_kind, name="intent_kind")
        if intent_kind not in SIM_HIL_ACCEPTANCE_INTENT_KINDS_V1:
            raise ValueError(f"Unsupported acceptance intent: {intent_kind}.")
        stage_id = _optional(self.stage_id)
        control_mode = _optional(self.control_mode)
        allocation_mode = _optional(self.allocation_mode)
        control_profile_id = _optional(self.control_profile_id)
        if intent_kind in {"start_stage", "confirm_stage"}:
            if stage_id not in SIM_HIL_ACCEPTANCE_STAGES_V1:
                raise ValueError(
                    "start_stage/confirm_stage requires a known stage_id."
                )
        elif stage_id is not None:
            raise ValueError(
                f"{intent_kind} must not carry stage_id."
            )
        if intent_kind in {"start_runtime", "set_control_mode"}:
            if control_mode not in SIM_HIL_CONTROL_MODES_V1:
                raise ValueError(
                    "start_runtime/set_control_mode requires a known "
                    "control_mode."
                )
        elif control_mode is not None:
            raise ValueError(
                f"{intent_kind} must not carry control_mode."
            )
        if intent_kind == "set_allocation_mode":
            if allocation_mode not in SIM_HIL_ALLOCATION_MODES_DEVELOPMENT:
                raise ValueError(
                    "set_allocation_mode requires automatic, base_only, "
                    "relative_overlay, or absolute_overlay_owner."
                )
        elif allocation_mode is not None:
            raise ValueError(
                f"{intent_kind} must not carry allocation_mode."
            )
        if intent_kind == "select_control_profile":
            if control_profile_id is None:
                raise ValueError(
                    "select_control_profile requires control_profile_id."
                )
        elif control_profile_id is not None:
            raise ValueError(
                "control_profile_id is only valid for "
                "select_control_profile."
            )
        collision_stop_enabled = self.sim_collision_stop_enabled
        if intent_kind == "select_sim_collision_stop":
            if not isinstance(collision_stop_enabled, bool):
                raise ValueError(
                    "select_sim_collision_stop requires a boolean "
                    "sim_collision_stop_enabled."
                )
        elif collision_stop_enabled is not None:
            raise ValueError(
                "sim_collision_stop_enabled is only valid for "
                "select_sim_collision_stop."
            )
        alpha = self.human_alpha
        if alpha is not None:
            alpha = float(alpha)
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError("human_alpha must be finite and in [0, 1].")
        blend_requested = bool(
            (
                intent_kind == "start_stage"
                and stage_id == "proportional_blend"
            )
            or (
                intent_kind in {"start_runtime", "set_control_mode"}
                and control_mode == "proportional_blend"
            )
        )
        if blend_requested:
            if alpha is None or alpha <= 0.0 or alpha >= 1.0:
                raise ValueError(
                    "proportional_blend requires human_alpha strictly "
                    "between 0 and 1."
                )
        elif alpha is not None:
            raise ValueError(
                "human_alpha is only valid for proportional_blend."
            )
        fallback_source = _optional(self.fallback_source)
        if intent_kind == "set_blend_failure_fallback":
            if fallback_source not in SIM_HIL_BLEND_FAILURE_FALLBACKS_V1:
                raise ValueError(
                    "set_blend_failure_fallback requires hold, human, or "
                    "policy."
                )
        elif intent_kind == "set_human_timeout_fallback":
            if fallback_source not in SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1:
                raise ValueError(
                    "set_human_timeout_fallback requires hold or policy."
                )
        elif fallback_source is not None:
            raise ValueError(
                "fallback_source is only valid for "
                "a fallback-selection intent."
            )
        blend_failure_grace = self.blend_failure_grace_sec
        if blend_failure_grace is not None:
            blend_failure_grace = float(blend_failure_grace)
            if (
                not math.isfinite(blend_failure_grace)
                or blend_failure_grace < 0.0
            ):
                raise ValueError(
                    "blend_failure_grace_sec must be finite and non-negative."
                )
        if (
            intent_kind != "set_blend_failure_fallback"
            and blend_failure_grace is not None
        ):
            raise ValueError(
                "blend_failure_grace_sec is only valid for "
                "set_blend_failure_fallback."
            )
        profile_name = _optional(self.profile_name)
        if intent_kind == "save_vive_calibration_profile":
            pass
        elif profile_name is not None:
            raise ValueError(
                "profile_name is only valid for save_vive_calibration_profile."
            )
        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "client_sequence", client_sequence)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "intent_kind", intent_kind)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "control_mode", control_mode)
        object.__setattr__(self, "allocation_mode", allocation_mode)
        object.__setattr__(self, "control_profile_id", control_profile_id)
        object.__setattr__(
            self,
            "sim_collision_stop_enabled",
            collision_stop_enabled,
        )
        object.__setattr__(self, "human_alpha", alpha)
        object.__setattr__(self, "fallback_source", fallback_source)
        object.__setattr__(
            self,
            "blend_failure_grace_sec",
            blend_failure_grace,
        )
        object.__setattr__(self, "profile_name", profile_name)
        object.__setattr__(
            self,
            "reason",
            _required(self.reason, name="reason"),
        )

    @property
    def intent_id(self) -> str:
        payload = {
            "schema": SIM_HIL_ACCEPTANCE_INTENT_SCHEMA_V1,
            "client_id": self.client_id,
            "client_sequence": self.client_sequence,
            "expected_revision": self.expected_revision,
            "intent_kind": self.intent_kind,
            "stage_id": self.stage_id,
            "control_mode": self.control_mode,
            "human_alpha": self.human_alpha,
            "fallback_source": self.fallback_source,
            "profile_name": self.profile_name,
            "reason": self.reason,
        }
        # Keep existing fixed-alpha intent identities byte-for-byte stable.
        # The development allocation field enters the identity only when used.
        if self.allocation_mode is not None:
            payload["allocation_mode"] = self.allocation_mode
        if self.control_profile_id is not None:
            payload["control_profile_id"] = self.control_profile_id
        if self.sim_collision_stop_enabled is not None:
            payload["sim_collision_stop_enabled"] = (
                self.sim_collision_stop_enabled
            )
        if self.blend_failure_grace_sec is not None:
            payload["blend_failure_grace_sec"] = (
                self.blend_failure_grace_sec
            )
        return "sim-hil-acceptance-intent:" + canonical_content_hash(
            payload,
            domain=SIM_HIL_ACCEPTANCE_INTENT_HASH_DOMAIN_V1,
        )

    @property
    def semantic_content_hash(self) -> str:
        """Canonical intent identity without parsing the public ID string."""

        return self.intent_id.rsplit(":", 1)[-1]


def sim_hil_acceptance_intent_from_json(
    payload: Mapping[str, Any],
) -> SimHILAcceptanceIntent:
    """Validate an exact browser payload before constructing an intent."""

    if not isinstance(payload, Mapping):
        raise TypeError("Acceptance intent payload must be an object.")
    allowed = {
        "client_id",
        "client_sequence",
        "expected_revision",
        "intent_kind",
        "stage_id",
        "control_mode",
        "allocation_mode",
        "control_profile_id",
        "sim_collision_stop_enabled",
        "human_alpha",
        "fallback_source",
        "blend_failure_grace_sec",
        "profile_name",
        "reason",
    }
    required = {
        "client_id",
        "client_sequence",
        "expected_revision",
        "intent_kind",
    }
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError(f"Unknown acceptance intent fields: {unknown}.")
    if missing:
        raise ValueError(f"Missing acceptance intent fields: {missing}.")
    return SimHILAcceptanceIntent(
        client_id=payload["client_id"],
        client_sequence=payload["client_sequence"],
        expected_revision=payload["expected_revision"],
        intent_kind=payload["intent_kind"],
        stage_id=payload.get("stage_id"),
        control_mode=payload.get("control_mode"),
        allocation_mode=payload.get("allocation_mode"),
        control_profile_id=payload.get("control_profile_id"),
        sim_collision_stop_enabled=payload.get(
            "sim_collision_stop_enabled"
        ),
        human_alpha=payload.get("human_alpha"),
        fallback_source=payload.get("fallback_source"),
        blend_failure_grace_sec=payload.get("blend_failure_grace_sec"),
        profile_name=payload.get("profile_name"),
        reason=payload.get("reason", "operator_request"),
    )


@dataclass(frozen=True)
class SimHILAcceptanceSubmissionReceipt:
    intent_id: str
    accepted: bool
    revision: int
    rejection_code: str | None = None
    message: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "accepted": self.accepted,
            "revision": self.revision,
            "rejection_code": self.rejection_code,
            "message": self.message,
        }


# Compatibility names remain at the Station facade while the concrete WAL,
# writer, episode binding, seal, and commit lifecycle is lower-owned.
SimHILAcceptanceTraceCapture = StationControlTraceEpisodeSessionCapture


class SimHILAcceptanceController:
    """Interactive Sim-HIL state owned by the Station control thread.

    One whole application control profile may be selected while the concrete
    runtime is absent. Construction freezes that lifecycle/control-cycle pair;
    changing it requires the existing explicit whole-runtime rebuild. The
    fixed-alpha profile uses ``start_runtime`` and ``set_control_mode``;
    allocation profiles use the distinct ``start_allocation_runtime`` and
    typed allocation operations. The exclusive profile exposes only the
    HOLD-bounded Policy/Human switch transaction. Older staged intents remain
    temporarily accepted for migration but are not used by the formal GUI.
    """

    def __init__(
        self,
        application: Any | None,
        *,
        trace_capture: ControlTraceEpisodeCapturePort | None = None,
        device_check_timeout_sec: float = 10.0,
        queue_capacity: int = 8,
        application_factory: Callable[..., Any] | None = None,
        application_factory_accepts_control_profile: bool = False,
        application_factory_accepts_sim_collision_stop: bool = False,
        vive_setup_controller: Any | None = None,
        vive_setup_factory: Callable[[str], Any] | None = None,
        coordinate_profile_path: str | Path | None = None,
        allocation_runtime_enabled: bool = False,
        exclusive_hitl_enabled: bool = False,
        selected_control_profile_id: str | None = None,
        available_control_profile_ids: tuple[str, ...] | None = None,
        selected_sim_collision_stop_enabled: bool = True,
        capture_profile: str = SIM_HIL_CAPTURE_PROFILE_ACCEPTANCE_V1,
    ) -> None:
        capacity = int(queue_capacity)
        if capacity <= 0:
            raise ValueError("queue_capacity must be positive.")
        timeout = float(device_check_timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("device_check_timeout_sec must be positive.")
        if application is None and (
            application_factory is None
            or vive_setup_controller is None
            or vive_setup_factory is None
            or coordinate_profile_path is None
        ):
            raise ValueError(
                "Lazy Sim-HIL acceptance requires application_factory, "
                "vive_setup_controller, vive_setup_factory, and "
                "coordinate_profile_path."
            )
        self.application = application
        self.application_factory = application_factory
        self._application_factory_accepts_control_profile = bool(
            application_factory_accepts_control_profile
        )
        self._application_factory_accepts_sim_collision_stop = bool(
            application_factory_accepts_sim_collision_stop
        )
        if not isinstance(selected_sim_collision_stop_enabled, bool):
            raise TypeError(
                "selected_sim_collision_stop_enabled must be a bool."
            )
        self._selected_sim_collision_stop_enabled = (
            selected_sim_collision_stop_enabled
        )
        self._vive_setup_controller = vive_setup_controller
        self._vive_setup_factory = vive_setup_factory
        requested_coordinate_profile_path = (
            None
            if coordinate_profile_path is None
            else Path(coordinate_profile_path).expanduser()
        )
        loaded_coordinate_profile_path = (
            None
            if vive_setup_controller is None
            else getattr(vive_setup_controller, "current_profile_path", None)
        )
        # Freeze a symbolic ``latest`` selection to the exact validated file
        # already loaded by the setup controller. GUI status and later runtime
        # construction must not resolve two different files.
        self._coordinate_profile_path = (
            Path(loaded_coordinate_profile_path).expanduser()
            if loaded_coordinate_profile_path is not None
            else requested_coordinate_profile_path
        )
        self._selected_vive_source_role = "left_wrist"
        self._setup_phase = (
            "runtime_constructed"
            if application is not None
            else "vive_setup"
        )
        self._cancel_requested = threading.Event()
        self.trace_capture = trace_capture
        self.device_check_timeout_sec = timeout
        self.allocation_runtime_enabled = bool(allocation_runtime_enabled)
        self.exclusive_hitl_enabled = bool(exclusive_hitl_enabled)
        normalized_capture_profile = _required(
            capture_profile,
            name="capture_profile",
        )
        if normalized_capture_profile not in SIM_HIL_CAPTURE_PROFILES_V1:
            raise ValueError(
                "capture_profile must be acceptance or human_demonstration."
            )
        if (
            normalized_capture_profile
            == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
            and (self.allocation_runtime_enabled or self.exclusive_hitl_enabled)
        ):
            raise ValueError(
                "human_demonstration requires the fixed-alpha control profile."
            )
        self.capture_profile = normalized_capture_profile
        selected_profile = _optional(selected_control_profile_id)
        if selected_profile is None:
            selected_profile = (
                SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
                if self.exclusive_hitl_enabled
                else SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
                if self.allocation_runtime_enabled
                else SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY
            )
        available_profiles = tuple(
            _required(value, name="available_control_profile_id")
            for value in (
                (selected_profile,)
                if available_control_profile_ids is None
                else available_control_profile_ids
            )
        )
        if len(set(available_profiles)) != len(available_profiles):
            raise ValueError("available_control_profile_ids must be unique.")
        if selected_profile not in available_profiles:
            raise ValueError(
                "selected_control_profile_id must be present in "
                "available_control_profile_ids."
            )
        if any(
            profile_id not in SIM_HIL_SELECTABLE_CONTROL_PROFILES
            for profile_id in available_profiles
        ):
            raise ValueError("Unknown selectable Sim-HIL control profile.")
        if (
            normalized_capture_profile
            == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
            and selected_profile
            != SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY
        ):
            raise ValueError(
                "human_demonstration requires the fixed-alpha control profile."
            )
        self._selected_control_profile_id = selected_profile
        self._available_control_profile_ids = available_profiles
        if self._application_factory_accepts_control_profile:
            self._apply_selected_control_profile_capabilities(selected_profile)
        if self.exclusive_hitl_enabled and not self.allocation_runtime_enabled:
            raise ValueError(
                "Exclusive truncated HITL is a streaming allocation profile."
            )
        self.queue_capacity = capacity
        self._queue: deque[SimHILAcceptanceIntent] = deque()
        self._revision = 0
        self._stage_status_by_id = {
            stage_id: "locked"
            for stage_id in SIM_HIL_ACCEPTANCE_STAGES_V1
        }
        self._stage_status_by_id["human_teleop"] = "available"
        self._completed: list[str] = []
        self._active_stage_id: str | None = None
        self._active_control_mode: str | None = None
        self._active_human_alpha: float | None = None
        self._control_mode_change_count = 0
        self._formal_runtime_started = False
        self._formal_runtime_started_at: float | None = None
        self._formal_runtime_finalized_at: float | None = None
        self._active_intent_id: str | None = None
        self._active_intent_kind: str | None = None
        self._baseline_cycle_by_stage: dict[str, int] = {}
        self._requested_alpha_by_stage: dict[str, float] = {}
        self._stage_evidence_by_id: dict[str, dict[str, object]] = {}
        self._last_result: dict[str, object] | None = None
        self._last_error: str | None = None
        self._last_exclusive_hitl_switch: (
            ExclusiveHITLSwitchEvidence | None
        ) = None
        self._last_exclusive_hitl_trace_error: str | None = None
        self._last_sequence_by_client: dict[str, int] = {}
        self._shutdown_requested = False
        self._lock = threading.RLock()

    def _apply_selected_control_profile_capabilities(
        self,
        profile_id: str,
    ) -> None:
        """Project one pre-construction profile choice into accepted intents."""

        selected = _required(profile_id, name="control_profile_id")
        self.allocation_runtime_enabled = (
            selected in SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS
        )
        self.exclusive_hitl_enabled = (
            selected == SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
        )

    def _select_control_profile(
        self,
        profile_id: str,
    ) -> dict[str, object]:
        """Select one whole application profile before runtime construction."""

        selected = _required(profile_id, name="control_profile_id")
        with self._lock:
            if not self._application_factory_accepts_control_profile:
                raise RuntimeError(
                    "This Sim-HIL binding does not support GUI profile selection."
                )
            if self.capture_profile == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1:
                raise RuntimeError(
                    "Human demonstration has a fixed Human-only control workflow."
                )
            if self.application is not None or self._formal_runtime_started:
                raise RuntimeError(
                    "Control profile is fixed after MuJoCo construction; rebuild "
                    "the runtime before selecting another workflow."
                )
            if selected not in self._available_control_profile_ids:
                raise ValueError(
                    f"Control profile {selected!r} is not available in this GUI."
                )
            self._selected_control_profile_id = selected
            self._apply_selected_control_profile_capabilities(selected)
        return {
            "intent_kind": "select_control_profile",
            "status": "control_profile_selected",
            "control_profile_id": selected,
            "allocation_runtime_enabled": self.allocation_runtime_enabled,
            "exclusive_hitl_enabled": self.exclusive_hitl_enabled,
        }

    def _select_sim_collision_stop(
        self,
        enabled: bool,
    ) -> dict[str, object]:
        """Select terminal Sim collision handling before target construction."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool.")
        with self._lock:
            if not self._application_factory_accepts_sim_collision_stop:
                raise RuntimeError(
                    "This Sim-HIL binding does not expose collision-stop "
                    "selection."
                )
            if self.application is not None or self._formal_runtime_started:
                raise RuntimeError(
                    "Sim collision-stop behavior is fixed after MuJoCo "
                    "construction; reset the runtime before changing it."
                )
            self._selected_sim_collision_stop_enabled = enabled
        return {
            "intent_kind": "select_sim_collision_stop",
            "status": "sim_collision_stop_selected",
            "sim_collision_stop_enabled": enabled,
        }

    @property
    def current_application(self) -> Any | None:
        with self._lock:
            return self.application

    def _require_application(self) -> Any:
        application = self.current_application
        if application is None:
            raise RuntimeError(
                "Sim-HIL runtime has not been constructed; complete VIVE "
                "setup first."
            )
        return application

    def _setup_snapshot(self) -> dict[str, object]:
        with self._lock:
            setup = self._vive_setup_controller
            selected_profile = self._coordinate_profile_path
            setup_phase = self._setup_phase
            selected_role = self._selected_vive_source_role
        if setup is None:
            return {
                "phase": setup_phase,
                "runtime_launch_ready": self.application is not None,
                "policy_runtime_launch_ready": self.application is not None,
                "human_runtime_launch_ready": self.application is not None,
                "selected_profile_path": (
                    None
                    if selected_profile is None
                    else str(selected_profile)
                ),
                "selected_vive_source_role": selected_role,
                "calibration": None,
            }
        calibration = dict(setup.snapshot())
        saved_profile = getattr(setup, "last_saved_profile_path", None)
        if saved_profile is not None:
            selected_profile = Path(saved_profile).expanduser()
        profile_ready = False
        if selected_profile is not None:
            if str(selected_profile) == "latest":
                profile_ready = any(
                    bool(item.get("valid", False))
                    for item in calibration.get("profiles", [])
                    if isinstance(item, Mapping)
                )
            else:
                profile_ready = selected_profile.is_file()
        input_preflight = calibration.get("input_preflight", {})
        input_ready = bool(
            isinstance(input_preflight, Mapping)
            and input_preflight.get("ready", False)
        )
        workflow_active = bool(calibration.get("workflow_active", False))
        stage = str(calibration.get("stage") or "")
        unsaved_completed = stage == "completed" and saved_profile is None
        mapping = dict(
            calibration.get("source_role_by_output_channel") or {}
        )
        selected_role = str(
            mapping.get("right_wrist_pose") or selected_role
        )
        return {
            "phase": setup_phase,
            "runtime_launch_ready": bool(
                input_ready
                and profile_ready
                and not workflow_active
                and not unsaved_completed
            ),
            # Policy-only construction neither consumes this receiver nor
            # loads its coordinate profile. Human setup may remain active in
            # parallel and is checked only at a later human-mode boundary.
            "policy_runtime_launch_ready": True,
            "human_runtime_launch_ready": bool(
                input_ready
                and profile_ready
                and not workflow_active
                and not unsaved_completed
            ),
            "input_ready": input_ready,
            "profile_ready": profile_ready,
            "unsaved_calibration": unsaved_completed,
            "selected_profile_path": (
                None if selected_profile is None else str(selected_profile)
            ),
            "selected_vive_source_role": selected_role,
            "calibration": calibration,
        }

    def setup_status(self) -> dict[str, object]:
        return self._setup_snapshot()

    def vive_status(self) -> dict[str, object]:
        application = self.current_application
        with self._lock:
            setup = self._vive_setup_controller
        if application is not None and (
            setup is None
            or bool(getattr(application, "human_input_attached", True))
        ):
            return dict(application.vive_status())
        if setup is None:
            return {
                "schema": "dexteleop.control_station.sim_hil_vive_status.v1",
                "sampled_at": time.time(),
                "vive_receiver": {
                    "status": "stopped",
                    "message": "VIVE setup receiver is not active.",
                    "trackers": {},
                },
            }
        source = getattr(setup, "source", None)
        from types import SimpleNamespace

        from control_station.sim_hil_profile.status import (
            _vive_receiver_status,
        )

        return {
            "schema": "dexteleop.control_station.sim_hil_vive_status.v1",
            "sampled_at": time.time(),
            "vive_receiver": _vive_receiver_status(
                SimpleNamespace(sources=(source,)),
                required_output_channels=("right_wrist_pose",),
            ),
        }

    def _cancellation_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    def _reject(
        self,
        intent: SimHILAcceptanceIntent,
        *,
        code: str,
        message: str,
    ) -> SimHILAcceptanceSubmissionReceipt:
        return SimHILAcceptanceSubmissionReceipt(
            intent_id=intent.intent_id,
            accepted=False,
            revision=self._revision,
            rejection_code=code,
            message=message,
        )

    def submit(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> SimHILAcceptanceSubmissionReceipt:
        with self._lock:
            previous = self._last_sequence_by_client.get(intent.client_id, -1)
            if intent.client_sequence <= previous:
                return self._reject(
                    intent,
                    code="non_monotonic_client_sequence",
                    message=f"last accepted sequence is {previous}",
                )
            # Cancellation and shutdown are out-of-band lifecycle controls.
            # The status request that supplied their revision may be blocked
            # behind the operation being cancelled, so requiring an exact UI
            # revision here makes the only recovery action unreachable.
            if intent.intent_kind in {"cancel_operation", "stop_runtime"}:
                if self._active_intent_id is None and not self._queue:
                    if intent.intent_kind == "stop_runtime":
                        self._queue.append(intent)
                        self._last_sequence_by_client[intent.client_id] = (
                            intent.client_sequence
                        )
                        self._revision += 1
                        return SimHILAcceptanceSubmissionReceipt(
                            intent_id=intent.intent_id,
                            accepted=True,
                            revision=self._revision,
                            message="runtime stop queued",
                        )
                    return self._reject(
                        intent,
                        code="no_operation_in_progress",
                        message="there is no active or queued operation",
                    )
                self._cancel_requested.set()
                self._queue.clear()
                if intent.intent_kind == "stop_runtime":
                    self._queue.append(intent)
                self._last_sequence_by_client[intent.client_id] = (
                    intent.client_sequence
                )
                self._revision += 1
                return SimHILAcceptanceSubmissionReceipt(
                    intent_id=intent.intent_id,
                    accepted=True,
                    revision=self._revision,
                    message=(
                        "runtime stop queued after active-operation cancellation"
                        if intent.intent_kind == "stop_runtime"
                        else "cancellation requested"
                    ),
                )
            if intent.expected_revision != self._revision:
                return self._reject(
                    intent,
                    code="stale_revision",
                    message=(
                        f"expected {intent.expected_revision}, current "
                        f"{self._revision}"
                    ),
                )
            if len(self._queue) >= self.queue_capacity:
                return self._reject(
                    intent,
                    code="queue_full",
                    message="acceptance intent queue is full",
                )
            if self._active_intent_id is not None or self._queue:
                return self._reject(
                    intent,
                    code="operation_in_progress",
                    message="wait for the current acceptance intent",
                )
            self._queue.append(intent)
            self._last_sequence_by_client[intent.client_id] = (
                intent.client_sequence
            )
            self._revision += 1
            return SimHILAcceptanceSubmissionReceipt(
                intent_id=intent.intent_id,
                accepted=True,
                revision=self._revision,
                message="queued for the Station control thread",
            )

    def submit_json(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, object]:
        intent = sim_hil_acceptance_intent_from_json(payload)
        return self.submit(intent).to_payload()

    def _next_stage(self, stage_id: str) -> str | None:
        index = SIM_HIL_ACCEPTANCE_STAGES_V1.index(stage_id)
        if index + 1 >= len(SIM_HIL_ACCEPTANCE_STAGES_V1):
            return None
        return SIM_HIL_ACCEPTANCE_STAGES_V1[index + 1]

    def _dispatch_vive_setup(
        self,
        event: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            application = self.application
            if application is not None and bool(
                getattr(application, "human_input_attached", True)
            ):
                raise RuntimeError(
                    "VIVE calibration is unavailable after runtime "
                    "human-input attachment."
                )
            setup = self._vive_setup_controller
        if setup is None:
            raise RuntimeError("VIVE setup controller is unavailable.")
        setup.dispatch(event, payload)
        if event == "save_profile":
            saved = setup.last_saved_profile_path
            if saved is None:
                raise RuntimeError(
                    "Calibration reported save success without a profile path."
                )
            with self._lock:
                self._coordinate_profile_path = Path(saved).expanduser()
        return self._setup_snapshot()

    def _construct_application_after_vive_setup(
        self,
        *,
        initial_human_alpha: float = 1.0,
        human_required: bool | None = None,
    ) -> Any:
        setup_status = self._setup_snapshot()
        requires_human = (
            float(initial_human_alpha) > 1e-12
            if human_required is None
            else bool(human_required)
        )
        if requires_human and not bool(
            setup_status.get("runtime_launch_ready", False)
        ):
            raise RuntimeError(
                "MuJoCo construction is blocked until live VIVE input and a "
                "saved coordinate profile are both ready."
            )
        with self._lock:
            setup = self._vive_setup_controller
            factory = self.application_factory
            profile_path = self._coordinate_profile_path
            selected_control_profile_id = self._selected_control_profile_id
            selected_sim_collision_stop_enabled = (
                self._selected_sim_collision_stop_enabled
            )
        if setup is None or factory is None or profile_path is None:
            raise RuntimeError("Lazy Sim-HIL construction is not configured.")
        calibration = setup_status.get("calibration", {})
        mapping = (
            {}
            if not isinstance(calibration, Mapping)
            else dict(
                calibration.get("source_role_by_output_channel") or {}
            )
        )
        source_role = str(
            mapping.get("right_wrist_pose")
            or setup_status.get("selected_vive_source_role")
            or "left_wrist"
        )
        if source_role not in {"left_wrist", "right_wrist"}:
            raise RuntimeError(
                "Selected VIVE source role for right_wrist_pose is invalid: "
                f"{source_role!r}."
            )
        saved = setup.last_saved_profile_path
        if saved is not None:
            profile_path = Path(saved).expanduser()
        if self._cancellation_requested():
            raise InterruptedError(
                "Sim-HIL startup was cancelled before MuJoCo construction."
            )
        with self._lock:
            self._setup_phase = "constructing_runtime"
            self._selected_vive_source_role = source_role
        try:
            # Keep the setup receiver alive while model/scene construction is
            # potentially slow. The new application's source is only
            # constructed here; it is not started until after the handoff.
            if (
                self._application_factory_accepts_control_profile
                and self._application_factory_accepts_sim_collision_stop
            ):
                application = factory(
                    profile_path,
                    source_role,
                    float(initial_human_alpha),
                    selected_control_profile_id,
                    selected_sim_collision_stop_enabled,
                )
            elif self._application_factory_accepts_control_profile:
                application = factory(
                    profile_path,
                    source_role,
                    float(initial_human_alpha),
                    selected_control_profile_id,
                )
            else:
                application = factory(
                    profile_path,
                    source_role,
                    float(initial_human_alpha),
                )
        except BaseException:
            with self._lock:
                self._setup_phase = "vive_setup"
            raise
        if self._cancellation_requested():
            try:
                application.close()
            finally:
                with self._lock:
                    self._setup_phase = "vive_setup"
                raise InterruptedError(
                    "Sim-HIL startup was cancelled after runtime construction "
                    "and before activation."
                )
        with self._lock:
            self._setup_phase = "vive_handoff"
        if requires_human:
            try:
                setup.close()
            except BaseException:
                try:
                    application.close()
                finally:
                    with self._lock:
                        self._setup_phase = "vive_setup_cleanup_failed"
                raise
        with self._lock:
            self.application = application
            self._coordinate_profile_path = profile_path
            if requires_human:
                self._vive_setup_controller = None
                self._setup_phase = "runtime_constructed"
            else:
                self._setup_phase = "policy_runtime_human_setup_available"
        return application

    def _restore_vive_setup_after_failed_launch(self) -> None:
        with self._lock:
            factory = self._vive_setup_factory
            application = self.application
            source_role = self._selected_vive_source_role
            existing_setup = self._vive_setup_controller
        if factory is None:
            raise RuntimeError(
                "Runtime rebuild is unavailable without a VIVE setup factory."
            )
        if application is not None:
            try:
                application.close()
            except Exception as error:
                # Do not start a replacement VIVE receiver while the previous
                # runtime reports an uncertain teardown.  That would turn a
                # cleanup fault into a port/GLFW resource leak.  The explicit
                # rebuild operation can retry the idempotent close and then
                # create a fresh setup owner.
                raise RuntimeError(
                    "Cannot restore VIVE setup after Sim-HIL runtime cleanup "
                    f"failed: {type(error).__name__}:{error}"
                ) from error
        setup = (
            existing_setup
            if existing_setup is not None
            else factory(source_role)
        )
        with self._lock:
            self.application = None
            self._vive_setup_controller = setup
            self._setup_phase = "vive_setup"

    def _abort_trace_capture_for_runtime_reset(
        self,
        application: Any | None,
    ) -> None:
        """Release an uncommitted acceptance episode before rebuilding.

        A failed first control cycle may already have prepared the trace.  A
        rebuild starts a new MuJoCo episode, so retaining that preparation
        would either collide with the next ``prepare`` call or make two
        runtime instances appear to share one capture identity.
        """

        capture = self.trace_capture
        if capture is None:
            return
        abort = getattr(capture, "abort", None)
        if not callable(abort):
            raise RuntimeError(
                "Acceptance capture cannot be reset because its abort hook "
                "is unavailable."
            )
        clock = getattr(application, "clock", None)
        now = (
            float(clock())
            if callable(clock)
            else time.monotonic()
        )
        abort(now=now)

    def _rotate_trace_capture_after_runtime_reset(self) -> None:
        """Give the next runtime a disjoint ControlTrace fact namespace."""

        capture = self.trace_capture
        if capture is None:
            return
        rotate = getattr(capture, "rotate_runtime_session", None)
        if not callable(rotate):
            raise RuntimeError(
                "Acceptance capture cannot rotate after runtime rebuild."
            )
        rotate(now=time.monotonic())

    @staticmethod
    def _control_mode_alpha(
        control_mode: str,
        *,
        requested_alpha: float | None,
    ) -> float:
        mode = str(control_mode)
        if mode == "human_teleop":
            return 1.0
        if mode == "ppo_only":
            return 0.0
        if mode == "proportional_blend":
            if requested_alpha is None:
                raise ValueError(
                    "proportional_blend requires human_alpha."
                )
            return float(requested_alpha)
        raise ValueError(f"Unsupported Sim-HIL control mode {mode!r}.")

    def _select_control_mode(
        self,
        *,
        control_mode: str,
        human_alpha: float | None,
    ) -> dict[str, object]:
        if self.allocation_runtime_enabled:
            raise RuntimeError(
                "Allocation-aware Sim-HIL does not accept fixed-alpha control "
                "modes; use set_allocation_mode."
            )
        application = self._require_application()
        application_status = application.status()
        if not bool(application_status.get("started", False)):
            raise RuntimeError(
                "Start the Sim-HIL runtime before selecting a control mode."
            )
        if bool(application_status.get("episode_ended", False)):
            raise RuntimeError(
                "The Sim-HIL episode has ended; start a new session."
            )
        alpha = self._control_mode_alpha(
            control_mode,
            requested_alpha=human_alpha,
        )
        previous_alpha = self._active_human_alpha
        if previous_alpha is None:
            mix = dict(
                dict(application_status.get("composition") or {}).get("mix")
                or {}
            )
            previous_values = tuple(
                float(value)
                for value in dict(
                    mix.get("human_alpha_by_component") or {}
                ).values()
            )
            if previous_values and all(
                abs(value - previous_values[0]) <= 1e-12
                for value in previous_values[1:]
            ):
                previous_alpha = previous_values[0]
            elif mix.get("mode") == "policy_only":
                previous_alpha = 0.0
            elif mix.get("mode") == "human_only":
                previous_alpha = 1.0
        human_preparation = None
        automatically_held = False
        previous_human_active = bool(
            previous_alpha is not None and previous_alpha > 1e-12
        )
        previous_policy_active = bool(
            previous_alpha is not None
            and previous_alpha < 1.0 - 1e-12
        )
        activates_human = alpha > 1e-12 and not previous_human_active
        activates_policy = (
            alpha < 1.0 - 1e-12 and not previous_policy_active
        )
        if activates_human:
            human_preparation, automatically_held = (
                self._prepare_human_mode(
                    application,
                    recapture_runtime_home=True,
                )
            )
        elif activates_policy:
            current = application.status()
            automatically_held = not bool(current.get("held", False))
            if automatically_held:
                application.hold(reason="policy_mode_preflight")
        try:
            receipt = application.set_human_blend_alpha(alpha)
            if automatically_held:
                application.rearm(
                    cancel_requested=self._cancellation_requested,
                )
        except BaseException:
            if previous_alpha is not None:
                try:
                    application.set_human_blend_alpha(previous_alpha)
                except BaseException:
                    pass
            if automatically_held:
                try:
                    application.rearm(
                        cancel_requested=self._cancellation_requested,
                    )
                except BaseException:
                    pass
            raise
        with self._lock:
            self._active_control_mode = str(control_mode)
            self._active_human_alpha = alpha
            self._control_mode_change_count += 1
        parameter_record = getattr(receipt, "parameter_record", None)
        selection_record_id = getattr(
            parameter_record,
            "parameter_record_id",
            getattr(
                receipt,
                "parameter_snapshot_id",
                getattr(receipt, "semantic_content_hash", None),
            ),
        )
        selection_record_hash = getattr(
            parameter_record,
            "semantic_content_hash",
            getattr(receipt, "semantic_content_hash", None),
        )
        return {
            "control_mode": str(control_mode),
            "human_alpha": alpha,
            # The admitted parameter record is the immutable selection fact.
            # The admission receipt is only a transient success/failure carrier.
            "blend_receipt_id": selection_record_id,
            "blend_receipt_hash": selection_record_hash,
            "human_preparation": human_preparation,
        }

    @staticmethod
    def _hold_evidence_refs(application: Any) -> tuple[str, ...]:
        outcome = getattr(application, "last_hold_outcome", None)
        refs = tuple(getattr(outcome, "lower_event_ids", ()) or ())
        if refs:
            return tuple(str(value) for value in refs)
        event = getattr(outcome, "event", None)
        event_id = getattr(event, "event_id", None)
        return () if event_id is None else (str(event_id),)

    def _record_exclusive_hitl_switch(
        self,
        evidence: ExclusiveHITLSwitchEvidence,
    ) -> None:
        self._last_exclusive_hitl_switch = evidence
        self._last_exclusive_hitl_trace_error = None
        capture = self.trace_capture
        if capture is None:
            return
        admit_trace = getattr(capture, "admit_trace", None)
        if not callable(admit_trace):
            return
        try:
            fact = project_exclusive_hitl_switch(
                evidence,
                runtime_session_id=str(
                    getattr(capture, "runtime_session_id", "")
                    or getattr(capture, "_runtime_session_id", "")
                ),
                control_domain_id=str(
                    getattr(capture, "control_domain_id", "")
                    or getattr(capture, "_control_domain_id", "")
                ),
            )
            admit_trace(fact)
        except Exception as error:
            # Trace health remains visible to the capture owner.  Recording
            # cannot become an implicit source-switch authority gate.
            self._last_exclusive_hitl_trace_error = (
                f"{type(error).__name__}:{error}"
            )

    def _execute_exclusive_hitl_switch(
        self,
        intent: SimHILAcceptanceIntent,
        *,
        target_source: str | None = None,
    ) -> dict[str, object]:
        """Run one HOLD-bounded 0/1 source switch and record its lineage."""

        if not self.exclusive_hitl_enabled:
            raise RuntimeError(
                "Exclusive HITL triggers require the exclusive_truncated_hitl "
                "control profile."
            )
        if target_source is None:
            target_source = (
                "human"
                if intent.intent_kind == "request_human_takeover"
                else "policy"
            )
        if target_source not in {"human", "policy"}:
            raise ValueError("target_source must be human or policy.")
        direction = (
            EXCLUSIVE_HITL_POLICY_TO_HUMAN
            if target_source == "human"
            else EXCLUSIVE_HITL_HUMAN_TO_POLICY
        )
        target_mode = "human_teleop" if target_source == "human" else "ppo_only"
        application = self._require_application()
        current_status = application.status()
        if not bool(current_status.get("started", False)):
            raise RuntimeError("Start the exclusive HITL runtime first.")
        if bool(current_status.get("episode_ended", False)):
            raise RuntimeError(
                "An ended Sim-HIL episode can only be reset/rebuilt."
            )
        if bool(current_status.get("held", False)):
            raise RuntimeError(
                "Exclusive source switching requires a running episode; "
                "resolve a non-terminal HOLD through rearm, or reset an "
                "ended episode."
            )
        current_mode = self._active_control_mode
        current_source = {
            "ppo_only": "policy",
            "human_teleop": "human",
        }.get(str(current_mode))
        if current_source is None:
            raise RuntimeError(
                "Exclusive HITL requires an exact policy-only or human-only "
                f"current mode, got {current_mode or 'unknown'}."
            )

        requested_at = float(application.clock())
        runtime_config = getattr(
            getattr(application, "runtime", None),
            "config",
            None,
        )
        clock_domain_id = str(
            getattr(
                runtime_config,
                "clock_domain_id",
                "mujoco.sim_hil.monotonic",
            )
        )

        transition_port = SimHILExclusiveTransitionPort(
            application=application,
            select_streaming_source=lambda **kwargs: (
                self._select_exclusive_streaming_source(
                    intent=intent,
                    **kwargs,
                )
            ),
            hold_evidence_refs=self._hold_evidence_refs,
            target_source=target_source,
            target_mode=target_mode,
            requested_at=requested_at,
        )

        mode, evidence = run_exclusive_hitl_switch_via_port(
            trigger_intent_id=intent.intent_id,
            trigger_intent_hash=intent.semantic_content_hash,
            direction=direction,
            current_source=current_source,
            requested_at=requested_at,
            clock_domain_id=clock_domain_id,
            port=transition_port,
            evidence_observer=self._record_exclusive_hitl_switch,
            clock=application.clock,
        )
        with self._lock:
            self._active_control_mode = str(target_mode)
            self._active_human_alpha = (
                1.0 if target_source == "human" else 0.0
            )
            self._control_mode_change_count += 1
        return {
            "control_mode": target_mode,
            "human_alpha": 1.0 if target_source == "human" else 0.0,
            **mode,
            "exclusive_hitl_switch": evidence.to_payload(),
        }

    def _select_exclusive_streaming_source(
        self,
        *,
        intent: SimHILAcceptanceIntent,
        target_source: str,
        target_mode: str,
        requested_at: float,
    ) -> dict[str, object]:
        """Commit one 0/1 allocation override inside a HOLD/rearm transaction."""

        from control_station.composition.mujoco_policy_human_allocation_operations import (
            MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER,
            MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
        )

        application = self._require_application()
        if target_source == "human":
            human_preparation, automatically_held = self._prepare_human_mode(
                application,
                recapture_runtime_home=True,
            )
            signal_name = MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER
        elif target_source == "policy":
            human_preparation = None
            current = application.status()
            automatically_held = not bool(current.get("held", False))
            if automatically_held:
                application.hold(reason="exclusive_hitl_policy_resume")
            signal_name = MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY
        else:
            raise ValueError("target_source must be policy or human.")
        if not automatically_held:
            raise RuntimeError(
                "Exclusive source selection requires an established lower "
                "HOLD boundary."
            )
        _report, operation_receipt = (
            application.rearm_with_allocation_operation(
                signal_name=signal_name,
                event_id=intent.intent_id,
                source_fact_id=intent.intent_id,
                source_fact_hash=intent.semantic_content_hash,
                occurred_at=float(requested_at),
                human_emission_required=(target_source == "human"),
                cancel_requested=self._cancellation_requested,
            )
        )
        override_receipt = getattr(operation_receipt, "override_receipt", None)
        if override_receipt is None:
            raise RuntimeError(
                "Exclusive streaming selection produced no allocation receipt."
            )
        return {
            "control_mode": str(target_mode),
            "human_alpha": 1.0 if target_source == "human" else 0.0,
            "allocation_mode": (
                "absolute_overlay_owner"
                if target_source == "human"
                else "base_only"
            ),
            "selection_record_id": override_receipt.receipt_id,
            "selection_record_hash": override_receipt.semantic_content_hash,
            "operation_receipt_id": operation_receipt.receipt_id,
            "operation_receipt_hash": operation_receipt.semantic_content_hash,
            "human_preparation": human_preparation,
        }

    def _prepare_human_mode(
        self,
        application: Any,
        *,
        recapture_runtime_home: bool,
    ) -> tuple[dict[str, object], bool]:
        """Make the passive human contributor ready before mode publication."""

        status = application.status()
        runtime_started = bool(status.get("started", False))
        preview_started = bool(
            getattr(application, "viewer_preview_started", False)
        )
        if not runtime_started and not preview_started:
            raise RuntimeError(
                "Human input preparation requires a pre-start viewer preview "
                "or a started runtime."
            )
        automatically_held = bool(
            runtime_started and not bool(status.get("held", False))
        )
        if not bool(getattr(application, "human_input_attached", True)):
            setup_status = self._setup_snapshot()
            if not bool(setup_status.get("runtime_launch_ready", False)):
                raise RuntimeError(
                    "Human/blend mode requires live human input and a saved "
                    "VIVE coordinate profile."
                )
            with self._lock:
                setup = self._vive_setup_controller
                profile_path = self._coordinate_profile_path
            if setup is None or profile_path is None:
                raise RuntimeError(
                    "Human input setup is unavailable for attachment."
                )
            calibration = setup_status.get("calibration", {})
            mapping = (
                {}
                if not isinstance(calibration, Mapping)
                else dict(
                    calibration.get(
                        "source_role_by_output_channel",
                        {},
                    )
                    or {}
                )
            )
            source_role = str(
                mapping.get("right_wrist_pose")
                or setup_status.get("selected_vive_source_role")
                or "left_wrist"
            )
            saved = getattr(setup, "last_saved_profile_path", None)
            if saved is not None:
                profile_path = Path(saved).expanduser()
            if automatically_held:
                application.hold(reason="human_input_attachment")
            attachment = None
            setup_closed = False
            try:
                attachment = application.prepare_human_attachment(
                    coordinate_profile_path=profile_path,
                    source_role=source_role,
                )
                setup.close()
                setup_closed = True
                activation = application.activate_human_attachment(
                    attachment
                )
            except BaseException:
                close_source = getattr(
                    getattr(attachment, "source", None),
                    "close",
                    None,
                )
                if callable(close_source):
                    try:
                        close_source()
                    except BaseException:
                        pass
                if (
                    setup_closed
                    and not bool(
                        getattr(
                            application,
                            "human_input_attached",
                            True,
                        )
                    )
                ):
                    replacement = (
                        None
                        if self._vive_setup_factory is None
                        else self._vive_setup_factory(source_role)
                    )
                    with self._lock:
                        self._vive_setup_controller = replacement
                        self._setup_phase = (
                            "policy_runtime_human_setup_available"
                            if replacement is not None
                            else "human_setup_unavailable"
                        )
                if automatically_held:
                    try:
                        application.rearm(
                            cancel_requested=self._cancellation_requested,
                        )
                    except BaseException:
                        pass
                raise
            with self._lock:
                self._vive_setup_controller = None
                self._coordinate_profile_path = profile_path
                self._selected_vive_source_role = source_role
                self._setup_phase = "runtime_constructed"
        else:
            activation = {
                "status": "already_attached",
                "command_dispatched": False,
                "authoritative": False,
            }
            if automatically_held:
                application.hold(reason="human_mode_preflight")

        try:
            passive_input = (
                application.run_human_input_acceptance_check(
                    timeout_sec=self.device_check_timeout_sec,
                )
                if self.application_factory is None
                else application.run_human_input_acceptance_check(
                    timeout_sec=self.device_check_timeout_sec,
                    cancel_requested=self._cancellation_requested,
                )
            )
            current = application.status()
            if (
                recapture_runtime_home
                or current.get("runtime_home") is None
            ):
                runtime_home = application.capture_runtime_home()
            else:
                runtime_home = dict(current["runtime_home"])
        except BaseException:
            if automatically_held:
                try:
                    application.rearm(
                        cancel_requested=self._cancellation_requested,
                    )
                except BaseException:
                    pass
            raise
        return (
            {
                "attachment": dict(activation),
                "passive_input": dict(passive_input),
                "runtime_home": dict(runtime_home),
            },
            automatically_held,
        )

    def _prepare_sim_preview(self) -> dict[str, object]:
        """Construct the demo scene and open only its passive viewer."""

        if (
            self.capture_profile
            != SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
        ):
            raise RuntimeError(
                "prepare_sim_preview is available only for "
                "human_demonstration capture."
            )
        application = self.current_application
        if application is None:
            # Build the final Human-only composition contract now while
            # deliberately leaving its Human source unattached.  This avoids
            # ever constructing a policy-authoritative preview that must be
            # switched after startup.
            application = self._construct_application_after_vive_setup(
                initial_human_alpha=1.0,
                human_required=False,
            )
        if bool(getattr(application, "started", False)):
            raise RuntimeError(
                "The Sim-HIL runtime is already started; preview is pre-start only."
            )
        start_preview = getattr(application, "start_viewer_preview", None)
        if not callable(start_preview):
            raise RuntimeError(
                "The selected Sim-HIL application does not support passive "
                "viewer preview."
            )
        report = dict(start_preview())
        with self._lock:
            self._setup_phase = "sim_preview_ready"
        return {
            "intent_kind": "prepare_sim_preview",
            "status": "viewer_preview",
            "trace_recording": False,
            "human_input_attached": bool(
                getattr(application, "human_input_attached", False)
            ),
            "preview": report,
        }

    def _start_runtime(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        if self.exclusive_hitl_enabled:
            return self._start_allocation_runtime(intent)
        if self.allocation_runtime_enabled:
            raise RuntimeError(
                "The selected allocation profile requires "
                "start_allocation_runtime."
            )
        control_mode = str(intent.control_mode)
        if (
            self.capture_profile
            == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
            and not bool(
                getattr(
                    self.current_application,
                    "viewer_preview_started",
                    False,
                )
            )
        ):
            raise RuntimeError(
                "Human demonstration must open the passive Sim preview before "
                "starting teleoperation and recording."
            )
        initial_alpha = self._control_mode_alpha(
            control_mode,
            requested_alpha=intent.human_alpha,
        )
        return self._start_application_runtime(
            intent,
            initial_human_alpha=initial_alpha,
            human_required=initial_alpha > 1e-12,
            control_mode=control_mode,
        )

    def _start_allocation_runtime(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        """Start an allocation profile without publishing fixed-alpha state."""

        if not self.allocation_runtime_enabled:
            raise RuntimeError(
                "start_allocation_runtime requires an allocation-aware "
                "application profile."
            )
        requested_mode = (
            str(intent.control_mode)
            if self.exclusive_hitl_enabled
            else None
        )
        start_human = requested_mode == "human_teleop"
        result = self._start_application_runtime(
            intent,
            # Allocation profiles ignore this fixed-alpha compatibility
            # field for composition. It still selects whether the Human input
            # module must be attached before construction. The allocation
            # session starts from its baseline and, for a Human launch, commits
            # the Human owner through the same HOLD-bounded switch used later.
            initial_human_alpha=(1.0 if start_human else 0.0),
            human_required=start_human,
            control_mode=None,
            prepare_human_before_start=not start_human,
        )
        if not start_human:
            return result
        switch = self._execute_exclusive_hitl_switch(
            intent,
            target_source="human",
        )
        return {
            **result,
            **switch,
            "initial_control_mode_requested": "human_teleop",
        }

    def _start_application_runtime(
        self,
        intent: SimHILAcceptanceIntent,
        *,
        initial_human_alpha: float,
        human_required: bool,
        control_mode: str | None,
        prepare_human_before_start: bool = True,
    ) -> dict[str, object]:
        initial_alpha = float(initial_human_alpha)
        existing = self.current_application
        existing_preview = bool(
            existing is not None
            and getattr(existing, "viewer_preview_started", False)
        )
        if existing is not None and bool(
            getattr(existing, "started", False)
        ):
            raise RuntimeError("The Sim-HIL runtime is already started.")
        application = (
            self._require_application()
            if existing is not None
            else self._construct_application_after_vive_setup(
                initial_human_alpha=initial_alpha,
                human_required=human_required,
            )
        )
        if human_required and prepare_human_before_start:
            if not bool(
                getattr(application, "human_input_attached", True)
            ):
                human_preparation, _automatically_held = (
                    self._prepare_human_mode(
                        application,
                        recapture_runtime_home=True,
                    )
                )
                passive_input = dict(human_preparation["passive_input"])
                runtime_home = dict(human_preparation["runtime_home"])
            else:
                if self.application_factory is None:
                    passive_input = (
                        application.run_human_input_acceptance_check(
                            timeout_sec=self.device_check_timeout_sec,
                        )
                    )
                else:
                    passive_input = (
                        application.run_human_input_acceptance_check(
                            timeout_sec=self.device_check_timeout_sec,
                            cancel_requested=self._cancellation_requested,
                        )
                    )
                if self._cancellation_requested():
                    raise InterruptedError(
                        "Sim-HIL startup was cancelled after VIVE readiness."
                    )
                runtime_home = application.capture_runtime_home()
        else:
            passive_input = {
                "status": "not_required",
                "reason": (
                    "deferred_to_exclusive_source_switch"
                    if human_required
                    else "policy_only_start"
                ),
                "command_dispatched": False,
                "authoritative": False,
            }
            runtime_home = {
                "status": "not_required",
                "reason": (
                    "deferred_to_exclusive_source_switch"
                    if human_required
                    else "policy_only_start"
                ),
                "command_dispatched": False,
                "authoritative": False,
            }
        report = (
            application.start()
            if self.application_factory is None
            else application.start(
                cancel_requested=self._cancellation_requested,
            )
        )
        formal_started_at = float(application.clock())
        if self.trace_capture is not None:
            self.trace_capture.prepare(now=formal_started_at)
        if (existing is None or existing_preview) and control_mode is not None:
            # Production constructs the composition contract with the selected
            # alpha. Startup readiness and first activation therefore use the
            # requested mode without a redundant post-start parameter update.
            mode = {
                "control_mode": control_mode,
                "human_alpha": initial_alpha,
                "blend_receipt_id": None,
                "mode_activation": "initial_profile",
            }
            with self._lock:
                self._active_control_mode = control_mode
                self._active_human_alpha = initial_alpha
                self._control_mode_change_count += 1
        elif existing is not None and control_mode is not None:
            # Direct application injection is retained for lightweight tests
            # and compatibility callers whose contract already exists.
            mode = self._select_control_mode(
                control_mode=control_mode,
                human_alpha=intent.human_alpha,
            )
        else:
            mode = {
                "control_profile_start": "allocation",
                "human_input_required": bool(human_required),
                "human_input_deferred": not bool(human_required),
                "initial_allocation_mode": (
                    "base_only" if self.exclusive_hitl_enabled else "automatic"
                ),
            }
            if self.exclusive_hitl_enabled:
                mode.update(
                    {
                        "control_mode": "ppo_only",
                        "human_alpha": 0.0,
                    }
                )
                with self._lock:
                    self._active_control_mode = "ppo_only"
                    self._active_human_alpha = 0.0
                    self._control_mode_change_count += 1
        with self._lock:
            self._formal_runtime_started = True
            self._formal_runtime_started_at = formal_started_at
            self._formal_runtime_finalized_at = None
        return {
            "intent_kind": intent.intent_kind,
            "status": "running",
            "passive_input": dict(passive_input),
            "runtime_home": dict(runtime_home),
            "preflight_passed": bool(report.passed),
            "mujoco_viewer_required": True,
            **mode,
        }

    def _start_stage(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        stage_id = str(intent.stage_id)
        expected_index = len(self._completed)
        expected_stage = (
            None
            if expected_index >= len(SIM_HIL_ACCEPTANCE_STAGES_V1)
            else SIM_HIL_ACCEPTANCE_STAGES_V1[expected_index]
        )
        if stage_id != expected_stage:
            raise RuntimeError(
                f"Next acceptance stage is {expected_stage}, not {stage_id}."
            )
        if self._stage_status_by_id[stage_id] not in {
            "available",
            "faulted",
        }:
            raise RuntimeError(
                f"Acceptance stage {stage_id} is not startable."
            )
        self._stage_status_by_id[stage_id] = "running"
        self._active_stage_id = stage_id
        if stage_id == "human_teleop":
            application = (
                self._require_application()
                if self.current_application is not None
                else self._construct_application_after_vive_setup()
            )
            if self.application_factory is None:
                passive_input = (
                    application.run_human_input_acceptance_check(
                        timeout_sec=self.device_check_timeout_sec,
                    )
                )
            else:
                passive_input = (
                    application.run_human_input_acceptance_check(
                        timeout_sec=self.device_check_timeout_sec,
                        cancel_requested=self._cancellation_requested,
                    )
                )
            if self._cancellation_requested():
                raise InterruptedError(
                    "Sim-HIL startup was cancelled after VIVE readiness."
                )
            runtime_home = application.capture_runtime_home()
            before = application.status()
            self._baseline_cycle_by_stage[stage_id] = int(
                (
                    before.get("composition", {})
                    .get("last_cycle", {})
                    or {}
                ).get("cycle_index", -1)
            )
            self._requested_alpha_by_stage[stage_id] = 1.0
            # Do not prepare the episode before the native MuJoCo viewer is
            # known to have started.  ``launch_passive`` is a native call and
            # a SIGSEGV cannot run Python ``finally`` blocks; preparing here
            # would leave the recorder lifecycle active in a retryable Python
            # process and the next start would fail with
            # ``episode preparation is already active``.  Runtime startup
            # facts emitted before this point remain session-level facts; the
            # acceptance episode begins only after the complete startup
            # boundary has succeeded.
            report = (
                application.start()
                if self.application_factory is None
                else application.start(
                    cancel_requested=self._cancellation_requested,
                )
            )
            receipt = application.set_human_blend_alpha(1.0)
            if self.trace_capture is not None:
                self.trace_capture.prepare(now=application.clock())
            result = {
                "passive_input": dict(passive_input),
                "runtime_home": dict(runtime_home),
                "preflight_passed": bool(report.passed),
                "mujoco_viewer_required": True,
                "blend_receipt_id": getattr(
                    receipt,
                    "parameter_snapshot_id",
                    getattr(receipt, "semantic_content_hash", None),
                ),
                "human_alpha": 1.0,
            }
        elif stage_id == "ppo_only":
            application = self._require_application()
            before = application.status()
            self._baseline_cycle_by_stage[stage_id] = int(
                (
                    before.get("composition", {})
                    .get("last_cycle", {})
                    or {}
                ).get("cycle_index", -1)
            )
            self._requested_alpha_by_stage[stage_id] = 0.0
            receipt = application.set_human_blend_alpha(0.0)
            result = {
                "blend_receipt_id": getattr(
                    receipt,
                    "parameter_snapshot_id",
                    getattr(receipt, "semantic_content_hash", None),
                ),
                "human_alpha": 0.0,
            }
        else:
            application = self._require_application()
            before = application.status()
            self._baseline_cycle_by_stage[stage_id] = int(
                (
                    before.get("composition", {})
                    .get("last_cycle", {})
                    or {}
                ).get("cycle_index", -1)
            )
            self._requested_alpha_by_stage[stage_id] = float(
                intent.human_alpha
            )
            receipt = application.set_human_blend_alpha(
                float(intent.human_alpha)
            )
            result = {
                "blend_receipt_id": getattr(
                    receipt,
                    "parameter_snapshot_id",
                    getattr(receipt, "semantic_content_hash", None),
                ),
                "human_alpha": float(intent.human_alpha),
            }
        self._stage_status_by_id[stage_id] = "awaiting_confirmation"
        self._stage_evidence_by_id[stage_id] = dict(result)
        return {
            "intent_kind": intent.intent_kind,
            "stage_id": stage_id,
            "status": "awaiting_confirmation",
            "evidence": result,
        }

    def _stage_confirmation_failures(
        self,
        stage_id: str,
    ) -> tuple[str, ...]:
        status = self._require_application().status()
        failures: list[str] = []
        if stage_id == "human_teleop":
            evidence = self._stage_evidence_by_id.get(stage_id, {})
            passive = (
                evidence.get("passive_input", {})
                if isinstance(evidence, Mapping)
                else {}
            )
            if not isinstance(passive, Mapping) or passive.get("status") != "ready":
                failures.append("passive_input_not_ready")
            visualization = status.get("visualization", {})
            if (
                not isinstance(visualization, Mapping)
                or not bool(visualization.get("enabled", False))
                or not bool(visualization.get("running", False))
            ):
                failures.append("mujoco_viewer_not_running")
        if not bool(status.get("started", False)):
            failures.append("application_not_started")
        if bool(status.get("held", False)):
            failures.append("application_held")
        gate = status.get("execution_gate", {})
        if not isinstance(gate, Mapping) or gate.get("status") != "ready":
            failures.append("execution_gate_not_ready")
        composition = status.get("composition", {})
        last_cycle = (
            {}
            if not isinstance(composition, Mapping)
            else composition.get("last_cycle", {}) or {}
        )
        cycle_index = (
            -1
            if not isinstance(last_cycle, Mapping)
            else int(last_cycle.get("cycle_index", -1))
        )
        baseline = int(self._baseline_cycle_by_stage.get(stage_id, -1))
        if cycle_index <= baseline:
            failures.append("no_post_stage_control_cycle")
        execution = (
            {}
            if not isinstance(last_cycle, Mapping)
            else last_cycle.get("execution", {}) or {}
        )
        if (
            not isinstance(execution, Mapping)
            or execution.get("status") != "dispatched"
        ):
            failures.append("latest_cycle_not_dispatched")
        mix = (
            {}
            if not isinstance(composition, Mapping)
            else composition.get("mix", {}) or {}
        )
        expected_mode = {
            "human_teleop": "human_only",
            "ppo_only": "policy_only",
            "proportional_blend": "proportional_blend",
        }[stage_id]
        if not isinstance(mix, Mapping) or mix.get("mode") != expected_mode:
            failures.append("composition_mix_mode_mismatch")
        expected_alpha = self._requested_alpha_by_stage.get(stage_id)
        alpha_by_component = (
            {}
            if not isinstance(mix, Mapping)
            else mix.get("human_alpha_by_component", {}) or {}
        )
        if expected_alpha is None or not isinstance(
            alpha_by_component,
            Mapping,
        ) or not alpha_by_component or any(
            abs(float(value) - expected_alpha) > 1e-9
            for value in alpha_by_component.values()
        ):
            failures.append("composition_alpha_mismatch")
        if stage_id == "proportional_blend":
            composition_step = (
                {}
                if not isinstance(last_cycle, Mapping)
                else last_cycle.get("composition_step", {}) or {}
            )
            if (
                not isinstance(composition_step, Mapping)
                or composition_step.get("immediate_difference") is None
            ):
                failures.append("immediate_difference_missing")
            execution_difference = status.get(
                "execution_difference",
                {},
            )
            latest_difference = (
                {}
                if not isinstance(execution_difference, Mapping)
                else execution_difference.get("latest", {}) or {}
            )
            if (
                isinstance(execution_difference, Mapping)
                and bool(execution_difference.get("enabled", False))
                and (
                    not isinstance(latest_difference, Mapping)
                    or latest_difference.get("status") != "available"
                )
            ):
                failures.append("execution_difference_missing")
        return tuple(failures)

    def _confirm_stage(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        stage_id = str(intent.stage_id)
        if stage_id != self._active_stage_id:
            raise RuntimeError(
                f"Active acceptance stage is {self._active_stage_id}."
            )
        if self._stage_status_by_id[stage_id] != "awaiting_confirmation":
            raise RuntimeError(
                f"Acceptance stage {stage_id} is not awaiting confirmation."
            )
        failures = self._stage_confirmation_failures(stage_id)
        if failures:
            raise RuntimeError(
                "Acceptance evidence is incomplete: "
                + ",".join(failures)
            )
        self._stage_status_by_id[stage_id] = "passed"
        self._completed.append(stage_id)
        self._active_stage_id = None
        next_stage = self._next_stage(stage_id)
        if next_stage is not None:
            self._stage_status_by_id[next_stage] = "available"
        return {
            "intent_kind": intent.intent_kind,
            "stage_id": stage_id,
            "status": "passed",
            "next_stage_id": next_stage,
        }

    def _execute(
        self,
        intent: SimHILAcceptanceIntent,
    ) -> dict[str, object]:
        human_demonstration = (
            self.capture_profile
            == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
        )
        if human_demonstration:
            if intent.intent_kind in {"start_runtime", "set_control_mode"}:
                if intent.control_mode != "human_teleop":
                    raise RuntimeError(
                        "human_demonstration permits only human_teleop control."
                    )
            if intent.intent_kind in {
                "start_allocation_runtime",
                "request_human_takeover",
                "request_policy_resume",
                "set_allocation_mode",
                "request_residual_rebase_and_clear",
                "start_stage",
                "confirm_stage",
            }:
                raise RuntimeError(
                    "human_demonstration rejects policy, allocation, and staged "
                    "acceptance operations."
                )
            if (
                intent.intent_kind
                in {
                    "set_blend_failure_fallback",
                    "set_human_timeout_fallback",
                }
                and intent.fallback_source != "hold"
            ):
                raise RuntimeError(
                    "human_demonstration permits only HOLD fallback."
                )
        if self.exclusive_hitl_enabled and intent.intent_kind in {
            "start_stage",
            "confirm_stage",
            "set_blend_failure_fallback",
            "set_human_timeout_fallback",
            "set_allocation_mode",
            "request_residual_rebase_and_clear",
        }:
            raise RuntimeError(
                "exclusive_truncated_hitl does not expose staged or blended control."
            )
        if intent.intent_kind in {
            "prepare_sim_preview",
            "start_runtime",
            "start_allocation_runtime",
            "set_control_mode",
            "request_human_takeover",
            "request_policy_resume",
            "set_allocation_mode",
            "request_residual_rebase_and_clear",
            "start_stage",
            "confirm_stage",
            "rearm",
        } and self.trace_capture is not None:
            capture_status = self.trace_capture.status()
            if capture_status.get("status") == "faulted":
                raise RuntimeError(
                    str(
                        capture_status.get("message")
                        or "ControlTrace writer fault"
                    )
                )
        if intent.intent_kind == "prepare_sim_preview":
            return self._prepare_sim_preview()
        if intent.intent_kind == "begin_vive_calibration":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup(
                    "begin_calibration"
                ),
            }
        if intent.intent_kind == "capture_vive_calibration":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup("capture"),
            }
        if intent.intent_kind == "reset_vive_calibration_stage":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup("reset_stage"),
            }
        if intent.intent_kind == "cancel_vive_calibration":
            return {
                "intent_kind": intent.intent_kind,
                "status": "accepted",
                "setup": self._dispatch_vive_setup(
                    "cancel_calibration"
                ),
            }
        if intent.intent_kind == "save_vive_calibration_profile":
            return {
                "intent_kind": intent.intent_kind,
                "status": "saved",
                "setup": self._dispatch_vive_setup(
                    "save_profile",
                    {"profile_name": intent.profile_name or ""},
                ),
            }
        if intent.intent_kind == "select_control_profile":
            assert intent.control_profile_id is not None
            return self._select_control_profile(intent.control_profile_id)
        if intent.intent_kind == "select_sim_collision_stop":
            assert intent.sim_collision_stop_enabled is not None
            return self._select_sim_collision_stop(
                intent.sim_collision_stop_enabled
            )
        if intent.intent_kind == "start_runtime":
            if self.exclusive_hitl_enabled and intent.control_mode not in {
                "ppo_only",
                "human_teleop",
            }:
                raise RuntimeError(
                    "exclusive_truncated_hitl must start from the Policy or "
                    "Human input module."
                )
            return self._start_runtime(intent)
        if intent.intent_kind == "start_allocation_runtime":
            return self._start_allocation_runtime(intent)
        if intent.intent_kind == "set_control_mode":
            if self.exclusive_hitl_enabled:
                raise RuntimeError(
                    "exclusive_truncated_hitl accepts only the explicit "
                    "takeover/resume triggers."
                )
            mode = self._select_control_mode(
                control_mode=str(intent.control_mode),
                human_alpha=intent.human_alpha,
            )
            return {
                "intent_kind": intent.intent_kind,
                "status": "mode_selected",
                **mode,
            }
        if intent.intent_kind in {
            "request_human_takeover",
            "request_policy_resume",
        }:
            switch = self._execute_exclusive_hitl_switch(intent)
            return {
                "intent_kind": intent.intent_kind,
                "status": "switch_ready",
                **switch,
            }
        if intent.intent_kind in {
            "set_allocation_mode",
            "request_residual_rebase_and_clear",
        }:
            from control_station.composition.mujoco_policy_human_allocation_operations import (
                MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER,
                MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
                MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
                MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR,
                MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY,
            )

            application = self._require_application()
            requested_mode = (
                None
                if intent.intent_kind == "request_residual_rebase_and_clear"
                else str(intent.allocation_mode)
            )
            needs_human = (
                intent.intent_kind == "request_residual_rebase_and_clear"
                or requested_mode != "base_only"
            )
            human_preparation = None
            automatically_held = False
            if needs_human:
                human_preparation, automatically_held = (
                    self._prepare_human_mode(
                        application,
                        # Runtime Home belongs to the device/target binding,
                        # not to every allocation-mode click.  First attach
                        # captures it because none exists; later mode changes
                        # retain the established Home.
                        recapture_runtime_home=False,
                    )
                )
            signal_name = (
                MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR
                if intent.intent_kind == "request_residual_rebase_and_clear"
                else {
                    "automatic": MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
                    "base_only": MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
                    "relative_overlay": (
                        MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY
                    ),
                    "absolute_overlay_owner": (
                        MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER
                    ),
                }[str(intent.allocation_mode)]
            )
            try:
                receipt = application.submit_allocation_button_operation(
                    signal_name=signal_name,
                    event_id=intent.intent_id,
                    source_fact_id=intent.intent_id,
                    source_fact_hash=intent.intent_id.rsplit(":", 1)[-1],
                    occurred_at=time.monotonic(),
                )
                if automatically_held:
                    application.rearm(
                        cancel_requested=self._cancellation_requested,
                    )
            except BaseException:
                if automatically_held:
                    try:
                        application.rearm(
                            cancel_requested=self._cancellation_requested,
                        )
                    except BaseException:
                        pass
                raise
            override_receipt = getattr(receipt, "override_receipt", None)
            anchor_request = getattr(receipt, "anchor_request", None)
            return {
                "intent_kind": intent.intent_kind,
                "status": (
                    "residual_rebase_requested"
                    if anchor_request is not None
                    else "allocation_mode_selected"
                ),
                "signal_name": signal_name,
                "allocation_mode": intent.allocation_mode,
                "operation_receipt_id": getattr(receipt, "receipt_id", None),
                "operation_receipt_hash": getattr(
                    receipt,
                    "semantic_content_hash",
                    None,
                ),
                "allocation_override_receipt_id": getattr(
                    override_receipt,
                    "receipt_id",
                    None,
                ),
                "anchor_request_id": getattr(
                    anchor_request,
                    "request_id",
                    None,
                ),
                "human_preparation": human_preparation,
            }
        if intent.intent_kind == "set_blend_failure_fallback":
            application = self._require_application()
            if not bool(application.status().get("started", False)):
                raise RuntimeError(
                    "Start the Sim-HIL runtime before selecting a blend "
                    "failure fallback."
                )
            receipt = application.set_blend_failure_fallback(
                str(intent.fallback_source),
                grace_sec=intent.blend_failure_grace_sec,
            )
            return {
                "intent_kind": intent.intent_kind,
                "status": "fallback_selected",
                **dict(receipt),
            }
        if intent.intent_kind == "set_human_timeout_fallback":
            application = self._require_application()
            if not bool(application.status().get("started", False)):
                raise RuntimeError(
                    "Start the Sim-HIL runtime before selecting a human "
                    "input timeout fallback."
                )
            receipt = application.set_human_timeout_fallback(
                str(intent.fallback_source)
            )
            return {
                "intent_kind": intent.intent_kind,
                "status": "human_timeout_fallback_selected",
                **dict(receipt),
            }
        if intent.intent_kind == "start_stage":
            return self._start_stage(intent)
        if intent.intent_kind == "confirm_stage":
            return self._confirm_stage(intent)
        if intent.intent_kind == "swap_vive_sources":
            # Swapping is a pure pre-runtime source-role mapping operation.
            # It must remain available even when the currently selected
            # sender is missing/invalid; readiness is evaluated only when the
            # resulting mapping is used to start the stage.
            application = self.current_application
            human_attached = bool(
                application is not None
                and getattr(application, "started", False)
                and getattr(application, "human_input_attached", True)
            )
            if human_attached or "human_teleop" in self._completed:
                raise RuntimeError(
                    "VIVE sources may be swapped only before human input is "
                    "attached to the Sim-HIL runtime."
                )
            with self._lock:
                setup_available = self._vive_setup_controller is not None
            if setup_available:
                setup = self._dispatch_vive_setup("swap_vive_sources")
                receipt = {
                    "status": "swapped",
                    "source_role_by_output_channel": dict(
                        (
                            setup.get("calibration", {})
                            if isinstance(
                                setup.get("calibration", {}),
                                Mapping,
                            )
                            else {}
                        ).get(
                            "source_role_by_output_channel",
                            {},
                        )
                    ),
                }
            elif application is not None:
                receipt = (
                    self._require_application()
                    .swap_vive_left_right_sources()
                )
            else:
                raise RuntimeError("VIVE setup controller is unavailable.")
            self._stage_status_by_id["human_teleop"] = "available"
            self._active_stage_id = None
            self._stage_evidence_by_id.pop("human_teleop", None)
            return {
                "intent_kind": "swap_vive_sources",
                "status": "swapped",
                "receipt": receipt,
            }
        if intent.intent_kind == "hold":
            self._require_application().hold(reason=intent.reason)
            return {"intent_kind": "hold", "status": "held"}
        if intent.intent_kind == "rearm":
            report = self._require_application().rearm(
                cancel_requested=self._cancellation_requested,
            )
            return {
                "intent_kind": "rearm",
                "status": "rearmed",
                "preflight_passed": bool(report.passed),
            }
        if intent.intent_kind == "capture_runtime_home":
            receipt = self._require_application().capture_runtime_home()
            return {
                "intent_kind": intent.intent_kind,
                "status": "runtime_home_captured",
                "receipt": dict(receipt),
            }
        if intent.intent_kind == "rebuild_runtime":
            # This is a destructive whole-runtime reset, not Home capture.
            # The formal runtime may rebuild from HOLD at any time; its
            # unfinalized trace is explicitly aborted.  Preserve the older
            # staged restriction only for migration callers.
            if self._completed and not self._formal_runtime_started:
                raise RuntimeError(
                    "Runtime rebuild is only available before an acceptance "
                    "stage has been committed."
                )
            application = self.current_application
            if application is None:
                raise RuntimeError("No constructed Sim-HIL runtime to rebuild.")
            if bool(getattr(application, "started", False)) and not bool(
                getattr(application, "_held", False)
            ):
                application.hold(reason=intent.reason)
            self._abort_trace_capture_for_runtime_reset(application)
            self._restore_vive_setup_after_failed_launch()
            self._rotate_trace_capture_after_runtime_reset()
            with self._lock:
                self._stage_status_by_id["human_teleop"] = "available"
                self._active_stage_id = None
                self._stage_evidence_by_id.pop("human_teleop", None)
                self._baseline_cycle_by_stage.pop("human_teleop", None)
                self._requested_alpha_by_stage.pop("human_teleop", None)
                self._active_control_mode = None
                self._active_human_alpha = None
                self._formal_runtime_started = False
                self._formal_runtime_started_at = None
                self._formal_runtime_finalized_at = None
                self._last_exclusive_hitl_switch = None
                self._last_exclusive_hitl_trace_error = None
                self._last_error = None
            return {
                "intent_kind": intent.intent_kind,
                "status": "runtime_rebuilt",
                "setup": self._setup_snapshot(),
            }
        if intent.intent_kind == "finalize_capture":
            if human_demonstration and (
                not self._formal_runtime_started
                or self._active_control_mode != "human_teleop"
            ):
                raise RuntimeError(
                    "human_demonstration finalization requires an active "
                    "human_teleop session."
                )
            if (
                not self._formal_runtime_started
                and tuple(self._completed) != SIM_HIL_ACCEPTANCE_STAGES_V1
            ):
                raise RuntimeError(
                    "All acceptance stages must pass before capture finalize."
                )
            application = self._require_application()
            application.end_episode(
                reason="operator_sim_hil_session_complete"
            )
            if self.trace_capture is None:
                raise RuntimeError(
                    "Acceptance capture is not bound; refusing to report "
                    "a completed data-retention stage."
                )
            finalized_at = float(application.clock())
            capture_status = self.trace_capture.finalize(now=finalized_at)
            with self._lock:
                self._formal_runtime_finalized_at = finalized_at
            return {
                "intent_kind": "finalize_capture",
                "status": "complete",
                "capture": capture_status,
            }
        if intent.intent_kind == "stop_runtime":
            application = self.current_application
            if application is not None and bool(
                getattr(application, "started", False)
            ):
                application.hold(reason=intent.reason)
            self._shutdown_requested = True
            return {
                "intent_kind": "stop_runtime",
                "status": "shutdown_requested",
            }
        raise RuntimeError(
            f"Unsupported acceptance intent {intent.intent_kind}."
        )

    def process_next(self) -> dict[str, object] | None:
        """Run at most one queued operation on the Station control thread."""

        with self._lock:
            if not self._queue:
                return None
            self._cancel_requested.clear()
            intent = self._queue.popleft()
            self._active_intent_id = intent.intent_id
            self._active_intent_kind = intent.intent_kind
            self._revision += 1
        try:
            result = self._execute(intent)
        except Exception as error:
            capture_reset_error: BaseException | None = None
            startup_intent = bool(
                intent.intent_kind in {
                    "prepare_sim_preview",
                    "start_runtime",
                    "start_allocation_runtime",
                }
                or (
                    intent.intent_kind == "start_stage"
                    and intent.stage_id == "human_teleop"
                )
            )
            application = self.current_application
            retryable_demo_preview = bool(
                self.capture_profile
                == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
                and intent.intent_kind == "start_runtime"
                and application is not None
                and getattr(application, "viewer_preview_started", False)
                and not bool(getattr(application, "started", False))
                and not bool(getattr(application, "restart_required", False))
                and not bool(getattr(application, "closed", False))
            )
            if startup_intent and not retryable_demo_preview:
                try:
                    self._abort_trace_capture_for_runtime_reset(
                        self.current_application
                    )
                except BaseException as reset_error:  # noqa: BLE001 - preserve startup cause.
                    capture_reset_error = reset_error
                application = self.current_application
                if application is not None and bool(
                    getattr(application, "started", False)
                ) and not bool(getattr(application, "_held", False)):
                    try:
                        application.hold(
                            reason="sim_hil_startup_failed"
                        )
                    except BaseException as hold_error:  # noqa: BLE001
                        if capture_reset_error is None:
                            capture_reset_error = hold_error
            should_restore_setup = (
                startup_intent
                and not retryable_demo_preview
                and self.application_factory is not None
                and (
                    self.current_application is not None
                    or self._vive_setup_controller is None
                )
                and not bool(
                    getattr(self.current_application, "started", False)
                )
            )
            if should_restore_setup:
                try:
                    self._restore_vive_setup_after_failed_launch()
                    self._rotate_trace_capture_after_runtime_reset()
                except Exception as restore_error:
                    error = RuntimeError(
                        f"{type(error).__name__}:{error}; "
                        "vive_setup_restore_failed="
                        f"{type(restore_error).__name__}:{restore_error}"
                    )
            with self._lock:
                if (
                    self._active_stage_id is not None
                    and self._stage_status_by_id[
                        self._active_stage_id
                    ]
                    == "running"
                ):
                    failed_status = (
                        "restart_required"
                        if bool(
                            getattr(
                                self.application,
                                "restart_required",
                                False,
                            )
                        )
                        else "faulted"
                    )
                    self._stage_status_by_id[
                        self._active_stage_id
                    ] = failed_status
                self._last_error = f"{type(error).__name__}:{error}"
                if capture_reset_error is not None:
                    self._last_error += (
                        "; capture_reset_failed="
                        f"{type(capture_reset_error).__name__}:"
                        f"{capture_reset_error}"
                    )
                self._last_result = {
                    "intent_id": intent.intent_id,
                    "intent_kind": intent.intent_kind,
                    "status": "failed",
                    "error": self._last_error,
                }
                self._active_intent_id = None
                self._active_intent_kind = None
                self._revision += 1
                return dict(self._last_result)
        with self._lock:
            self._last_error = None
            self._last_result = {
                "intent_id": intent.intent_id,
                **result,
            }
            self._active_intent_id = None
            self._active_intent_kind = None
            self._revision += 1
            return dict(self._last_result)

    def status(self) -> dict[str, object]:
        setup = self._setup_snapshot()
        with self._lock:
            if (
                self.capture_profile
                == SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
            ):
                available_control_modes = ["human_teleop"]
            elif self.exclusive_hitl_enabled:
                available_control_modes = ["ppo_only", "human_teleop"]
            elif self.allocation_runtime_enabled:
                available_control_modes = []
            else:
                available_control_modes = list(SIM_HIL_CONTROL_MODES_V1)
            stage_status = MappingProxyType(
                dict(self._stage_status_by_id)
            )
            capture = (
                {"enabled": False, "status": "not_bound"}
                if self.trace_capture is None
                else self.trace_capture.status()
            )
            return {
                "schema": SIM_HIL_ACCEPTANCE_STATUS_SCHEMA_V1,
                "revision": self._revision,
                "active_stage_id": self._active_stage_id,
                "capture_profile": self.capture_profile,
                "selected_control_profile_id": (
                    self._selected_control_profile_id
                ),
                "available_control_profile_ids": list(
                    self._available_control_profile_ids
                ),
                "control_profile_selection_available": bool(
                    self._application_factory_accepts_control_profile
                    and self.capture_profile
                    != SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1
                ),
                "control_profile_selection_locked": bool(
                    self.application is not None
                    or self._formal_runtime_started
                ),
                "selected_sim_collision_stop_enabled": (
                    self._selected_sim_collision_stop_enabled
                ),
                "sim_collision_stop_selection_available": bool(
                    self._application_factory_accepts_sim_collision_stop
                ),
                "sim_collision_stop_selection_locked": bool(
                    self.application is not None
                    or self._formal_runtime_started
                ),
                "available_control_modes": available_control_modes,
                "available_blend_failure_fallbacks": list(
                    SIM_HIL_BLEND_FAILURE_FALLBACKS_V1
                ),
                "available_human_timeout_fallbacks": list(
                    SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1
                ),
                "active_control_mode": self._active_control_mode,
                "active_human_alpha": self._active_human_alpha,
                "control_mode_change_count": (
                    self._control_mode_change_count
                ),
                "formal_runtime_started": self._formal_runtime_started,
                "formal_runtime_started_at": (
                    self._formal_runtime_started_at
                ),
                "formal_runtime_finalized_at": (
                    self._formal_runtime_finalized_at
                ),
                "sim_preview_ready": bool(
                    self.application is not None
                    and getattr(
                        self.application,
                        "viewer_preview_started",
                        False,
                    )
                ),
                "allocation_runtime_enabled": (
                    self.allocation_runtime_enabled
                ),
                "exclusive_hitl_enabled": self.exclusive_hitl_enabled,
                "last_exclusive_hitl_switch": (
                    None
                    if self._last_exclusive_hitl_switch is None
                    else self._last_exclusive_hitl_switch.to_payload()
                ),
                "last_exclusive_hitl_trace_error": (
                    self._last_exclusive_hitl_trace_error
                ),
                "active_intent_id": self._active_intent_id,
                "active_intent_kind": self._active_intent_kind,
                "pending_intent_count": len(self._queue),
                "completed_stage_ids": list(self._completed),
                "stage_status_by_id": dict(stage_status),
                "last_result": (
                    None
                    if self._last_result is None
                    else dict(self._last_result)
                ),
                "last_error": self._last_error,
                "capture": capture,
                "shutdown_requested": self._shutdown_requested,
                "operation_cancel_available": bool(
                    self._active_intent_id is not None
                ),
                "setup": setup,
            }

    def enforce_capture_health(self) -> str | None:
        """Latch formal execution in HOLD after a capture-path fault."""

        capture = self.trace_capture
        if capture is None:
            return None
        capture_status = capture.status()
        if capture_status.get("status") != "faulted":
            return None
        message = str(
            capture_status.get("message")
            or "ControlTrace writer fault"
        )
        application = self.current_application
        hold_error = None
        if (
            application is not None
            and bool(getattr(application, "started", False))
            and not bool(getattr(application, "_held", False))
            and not bool(getattr(application, "_episode_ended", False))
        ):
            try:
                application.hold(
                    reason="formal_control_trace_capture_fault"
                )
            except BaseException as error:  # noqa: BLE001 - retain both facts.
                hold_error = f"{type(error).__name__}:{error}"
        with self._lock:
            failure = (
                message
                if hold_error is None
                else f"{message}; hold_failed={hold_error}"
            )
            if self._last_error != failure:
                self._last_error = failure
                self._revision += 1
            return self._last_error

    def close(self) -> None:
        with self._lock:
            setup = self._vive_setup_controller
            application = self.application
        cleanup_errors: list[BaseException] = []
        # Close the constructed runtime first so its viewer, composition
        # worker, and passive VIVE source are drained before the setup owner is
        # touched.  The outer Station loop may already have closed it; the
        # application lifecycle is intentionally idempotent.
        if application is not None:
            close_application = getattr(application, "close", None)
            if callable(close_application):
                try:
                    close_application()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    cleanup_errors.append(error)
        if setup is not None:
            try:
                setup.close()
            except BaseException as error:  # noqa: BLE001 - cleanup continues.
                cleanup_errors.append(error)
        capture = self.trace_capture
        if capture is not None:
            application = self.current_application
            try:
                capture.close(
                    now=(
                        time.monotonic()
                        if application is None
                        else application.clock()
                    )
                )
            except BaseException as error:  # noqa: BLE001 - report after all cleanup.
                cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError(
                "Sim-HIL acceptance cleanup encountered errors: "
                + "; ".join(
                    f"{type(error).__name__}:{error}"
                    for error in cleanup_errors
                )
            ) from cleanup_errors[0]


__all__ = [
    "SIM_HIL_CAPTURE_PROFILE_ACCEPTANCE_V1",
    "SIM_HIL_CAPTURE_PROFILE_HUMAN_DEMONSTRATION_V1",
    "SIM_HIL_CAPTURE_PROFILES_V1",
    "SIM_HIL_ALLOCATION_MODES_DEVELOPMENT",
    "SIM_HIL_ACCEPTANCE_INTENT_KINDS_V1",
    "SIM_HIL_ACCEPTANCE_INTENT_SCHEMA_V1",
    "SIM_HIL_ACCEPTANCE_STAGES_V1",
    "SIM_HIL_ACCEPTANCE_STATUS_SCHEMA_V1",
    "SIM_HIL_ACCEPTANCE_TRACE_REQUIREMENT_PROFILE_V1",
    "SIM_HIL_CONTROL_MODES_V1",
    "SIM_HIL_HUMAN_DEMONSTRATION_TRACE_REQUIREMENT_PROFILE_V1",
    "SIM_HIL_HUMAN_TIMEOUT_FALLBACKS_V1",
    "STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1",
    "STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1",
    "SimHILAcceptanceController",
    "SimHILAcceptanceIntent",
    "SimHILAcceptanceSubmissionReceipt",
    "SimHILAcceptanceTraceCapture",
    "StationControlTraceEpisodeSessionCapture",
    "StationControlTraceFinalizationReceipt",
    "sim_hil_trace_requirement_profile_for_capture_profile",
    "sim_hil_acceptance_intent_from_json",
]
