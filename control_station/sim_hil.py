"""Control Station application root for the early Sim-HIL profile.

The profile is deliberately narrow and default-off:

* the temporary MuJoCo PPO checkpoint contributes ``right_arm`` pose plus
  ``right_hand`` qpos from its task-local gripper delta;
* real VIVE and Q6 are constructed through the passive teleop builder and can
  only produce a human emission;
* the existing formal composition and one ``MujocoDeployRuntime`` own
  authority, safety, coupled IK, reference scheduling, and MuJoCo dispatch;
* no real RM65 arm backend is constructed or enabled.

This module is an application leaf.  Model inference, device lifecycle,
composition algebra, safety, IK, and execution remain in their lower-owned
packages.  Concrete imports are lazy so invoking the default-off CLI does not
open a socket, load a device, or initialize MuJoCo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE_NAME,
    POSITION_60_120_PROFILE_NAME,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "sim"
    / "mujoco_temp_pipeline"
    / "records"
    / "20260728_215045_cube_lift_pregrasp_bridge_v5_center_guard"
    / "checkpoints"
    / "step_000050176_final.pt"
)
DEFAULT_TASK_CONFIG = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
)
DEFAULT_CALIBRATION_PROFILE = (
    REPO_ROOT
    / "teleop_records"
    / "calibration_profiles"
    / "20260627_112114_vive_rm65_virtual_desk.json"
)
DEFAULT_VIVE_COORDINATE_PROFILE = Path("latest")
DEFAULT_VIVE_PORT = 9999
DEFAULT_STATUS_HOST = "127.0.0.1"
DEFAULT_STATUS_PORT = 8787
DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ = 30.0
SIM_HIL_STATUS_SCHEMA_V1 = "dexteleop.control_station.sim_hil_status.v1"
SIM_HIL_BLEND_FAILURE_FALLBACKS = frozenset({"hold", "human", "policy"})
SIM_HIL_HUMAN_TIMEOUT_FALLBACKS = frozenset({"hold", "policy"})
_LOOPBACK_STATUS_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _advance_control_deadline(
    *,
    previous_deadline: float,
    completed_at: float,
    period: float,
) -> tuple[float, int]:
    """Advance one periodic Station deadline without burst catch-up."""

    previous = _finite(previous_deadline, name="previous_deadline")
    completed = _finite(completed_at, name="completed_at")
    cadence = _finite(period, name="period")
    if cadence <= 0.0:
        raise ValueError("period must be positive.")
    candidate = previous + cadence
    skipped = 0
    if candidate <= completed:
        skipped = int(math.floor((completed - candidate) / cadence)) + 1
        candidate += skipped * cadence
    return candidate, skipped


def _acceptance_profile_config(
    config: "MujocoSimHILApplicationConfig",
) -> "MujocoSimHILApplicationConfig":
    """Return the base configuration for interactive control.

    The lazy application factory replaces ``initial_human_alpha`` with the
    operator-selected mode before constructing composition contracts.
    """

    return replace(config, initial_human_alpha=1.0)


@dataclass(frozen=True)
class MujocoSimHILApplicationConfig:
    """Station-only profile selection and lifecycle settings."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    task_config_path: Path = DEFAULT_TASK_CONFIG
    calibration_profile_path: Path = DEFAULT_CALIBRATION_PROFILE
    vive_coordinate_profile_path: Path | None = DEFAULT_VIVE_COORDINATE_PROFILE
    vive_port: int = DEFAULT_VIVE_PORT
    vive_clock_sync_port: int | None = None
    vive_source_tracker_role: str = "left_wrist"
    left_q6_port: str | None = None
    right_q6_port: str | None = None
    gripper_input_source: str = "auto"
    q6_config: str = "latest"
    angle_trigger_config_root: Path | None = None
    angle_trigger_rate_hz: float = 30.0
    device: str = "auto"
    enabled: bool = False
    realtime_worker: bool = True
    # Explicit final compose/authority target cadence.  ``control_rate_hz``
    # remains a compatibility alias for the historical --rate option.
    position_control_profile_name: str = POSITION_15_60_120_PROFILE_NAME
    # 30 Hz is the measured interactive compromise for the current
    # synchronous composition path.  Policy inference retains its own lower
    # cadence; the selected 15->60->120 position profile permits composed
    # targets anywhere from 15 through 60 Hz.
    compose_target_hz: float | None = None
    control_rate_hz: float | None = None
    warmup_timeout_sec: float = 10.0
    # Sim-HIL-only human input freshness window.  This is deliberately
    # separate from teleop_runtime's tracker-loss HOLD thresholds: it allows
    # a short network gap to be observed without changing real-robot safety
    # semantics or retaining an old pose as a fresh command.
    human_emission_timeout_sec: float = 5.0
    target_admission_margin_sec: float = 0.05
    initial_human_alpha: float = 0.5
    blend_failure_fallback: str = "hold"
    human_timeout_fallback: str = "hold"
    defer_human_input: bool = False
    execution_difference_enabled: bool = True
    mujoco_viewer_enabled: bool = False
    status_host: str = DEFAULT_STATUS_HOST
    status_port: int = DEFAULT_STATUS_PORT

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_path",
            "task_config_path",
            "calibration_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if self.vive_coordinate_profile_path is not None:
            object.__setattr__(
                self,
                "vive_coordinate_profile_path",
                Path(self.vive_coordinate_profile_path).expanduser(),
            )
        if self.angle_trigger_config_root is not None:
            object.__setattr__(
                self,
                "angle_trigger_config_root",
                Path(self.angle_trigger_config_root).expanduser(),
            )
        for name in ("left_q6_port", "right_q6_port"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when supplied.")
        vive_role = str(self.vive_source_tracker_role).strip()
        if vive_role not in {"left_wrist", "right_wrist"}:
            raise ValueError(
                "vive_source_tracker_role must be left_wrist or right_wrist."
            )
        object.__setattr__(
            self,
            "vive_source_tracker_role",
            vive_role,
        )
        source = str(self.gripper_input_source).strip().lower()
        if source not in {"auto", "q6", "angle_trigger"}:
            raise ValueError(
                "gripper_input_source must be auto, q6, or angle_trigger."
            )
        object.__setattr__(self, "gripper_input_source", source)
        q6_config = str(self.q6_config).strip()
        if not q6_config:
            raise ValueError("q6_config must be non-empty.")
        object.__setattr__(self, "q6_config", q6_config)
        angle_rate = _finite(
            self.angle_trigger_rate_hz,
            name="angle_trigger_rate_hz",
        )
        if angle_rate <= 0.0:
            raise ValueError("angle_trigger_rate_hz must be positive.")
        object.__setattr__(self, "angle_trigger_rate_hz", angle_rate)
        vive_port = int(self.vive_port)
        if not 0 <= vive_port <= 65535:
            raise ValueError("vive_port must be in [0, 65535].")
        object.__setattr__(self, "vive_port", vive_port)
        if self.vive_clock_sync_port is not None:
            sync_port = int(self.vive_clock_sync_port)
            if not 0 <= sync_port <= 65535:
                raise ValueError("vive_clock_sync_port must be in [0, 65535].")
            object.__setattr__(self, "vive_clock_sync_port", sync_port)
        timeout = _finite(self.warmup_timeout_sec, name="warmup_timeout_sec")
        if timeout <= 0.0:
            raise ValueError("warmup_timeout_sec must be positive.")
        object.__setattr__(self, "warmup_timeout_sec", timeout)
        human_timeout = _finite(
            self.human_emission_timeout_sec,
            name="human_emission_timeout_sec",
        )
        if human_timeout <= 0.0:
            raise ValueError("human_emission_timeout_sec must be positive.")
        object.__setattr__(self, "human_emission_timeout_sec", human_timeout)
        target_margin = _finite(
            self.target_admission_margin_sec,
            name="target_admission_margin_sec",
        )
        if target_margin <= 0.0:
            raise ValueError("target_admission_margin_sec must be positive.")
        object.__setattr__(
            self,
            "target_admission_margin_sec",
            target_margin,
        )
        human_alpha = _finite(
            self.initial_human_alpha,
            name="initial_human_alpha",
        )
        if human_alpha < 0.0 or human_alpha > 1.0:
            raise ValueError("initial_human_alpha must be in [0, 1].")
        object.__setattr__(self, "initial_human_alpha", human_alpha)
        blend_failure_fallback = str(
            self.blend_failure_fallback
        ).strip().lower()
        if blend_failure_fallback not in SIM_HIL_BLEND_FAILURE_FALLBACKS:
            raise ValueError(
                "blend_failure_fallback must be hold, human, or policy."
            )
        object.__setattr__(
            self,
            "blend_failure_fallback",
            blend_failure_fallback,
        )
        human_timeout_fallback = str(
            self.human_timeout_fallback
        ).strip().lower()
        if human_timeout_fallback not in SIM_HIL_HUMAN_TIMEOUT_FALLBACKS:
            raise ValueError(
                "human_timeout_fallback must be hold or policy."
            )
        object.__setattr__(
            self,
            "human_timeout_fallback",
            human_timeout_fallback,
        )
        object.__setattr__(
            self,
            "defer_human_input",
            bool(self.defer_human_input),
        )
        profile_name = str(self.position_control_profile_name).strip()
        if not profile_name:
            raise ValueError("position_control_profile_name must be non-empty.")
        object.__setattr__(self, "position_control_profile_name", profile_name)
        compose_rate = None
        if self.compose_target_hz is not None:
            compose_rate = _finite(
                self.compose_target_hz,
                name="compose_target_hz",
            )
            if compose_rate <= 0.0:
                raise ValueError("compose_target_hz must be positive.")
            object.__setattr__(self, "compose_target_hz", compose_rate)
        if self.control_rate_hz is not None:
            legacy_rate = _finite(self.control_rate_hz, name="control_rate_hz")
            if legacy_rate <= 0.0:
                raise ValueError("control_rate_hz must be positive.")
            if compose_rate is not None and not math.isclose(
                legacy_rate,
                compose_rate,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "control_rate_hz is a compatibility alias and must match "
                    "compose_target_hz when both are supplied."
                )
            object.__setattr__(self, "control_rate_hz", legacy_rate)
            if compose_rate is None:
                compose_rate = legacy_rate
                object.__setattr__(self, "compose_target_hz", legacy_rate)
        elif compose_rate is None:
            compose_rate = DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ
            object.__setattr__(self, "compose_target_hz", compose_rate)
        status_port = int(self.status_port)
        if not 0 <= status_port <= 65535:
            raise ValueError("status_port must be in [0, 65535].")
        object.__setattr__(self, "status_port", status_port)
        if not isinstance(self.execution_difference_enabled, bool):
            raise TypeError("execution_difference_enabled must be a bool.")
        if not isinstance(self.mujoco_viewer_enabled, bool):
            raise TypeError("mujoco_viewer_enabled must be a bool.")
        status_host = str(self.status_host).strip().lower()
        if status_host not in _LOOPBACK_STATUS_HOSTS:
            raise ValueError(
                "status_host must be loopback-only (127.0.0.1 or localhost)."
            )
        object.__setattr__(self, "status_host", status_host)


@dataclass(frozen=True)
class _SimHILHumanAttachment:
    source: Any
    gripper_input_selection: Any
    calibration_binding: Mapping[str, Any]
    runtime_home_pose_desk_by_side_capture: Callable[
        [],
        Mapping[str, Any],
    ]


def _file_sha256(path: Path) -> str | None:
    """Return a stable file identity for Station status, if the file exists."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_vive_coordinate_profile(
    calibration: Any,
    *,
    coordinate_profile_path: Path | None,
    home_profile_path: Path | None = None,
    required_sides: tuple[str, ...] = ("left", "right"),
) -> tuple[Any, dict[str, Any]]:
    """Bind Station VIVE coordinates to a stored human-Home profile.

    ``TeleopCalibration`` remains owned by ``teleop_runtime``.  The Station
    profile only selects two immutable inputs: the compatibility profile
    supplies stored human Home, while the independent Control Station profile
    supplies the VIVE world-to-retarget transform and tracker offsets. Target
    geometry and robot Home are bound later from the concrete target. No data
    is written back to either source profile.
    """

    home_profile_value = getattr(calibration, "metadata", {}).get(
        "source_profile_path"
    )
    binding: dict[str, Any] = {
        "mode": "home_profile_only_compatibility",
        "home_profile_path": (
            None if not str(home_profile_value or "").strip() else str(home_profile_value)
        ),
        "home_profile_sha256": None,
        "vive_coordinate_profile_path": None,
        "vive_coordinate_profile_sha256": None,
        "coordinate_profile_kind": None,
        "coordinate_profile_schema": None,
        "active_sides": None,
    }
    if coordinate_profile_path is None:
        return calibration, binding

    from .vive_calibration_profiles import (
        load_coordinate_profile,
        validate_coordinate_profile,
    )

    coordinate_source = Path(coordinate_profile_path).expanduser()
    if (
        str(coordinate_source).strip().lower() != "latest"
        and not coordinate_source.is_absolute()
        and coordinate_source.exists()
    ):
        coordinate_source = coordinate_source.resolve()
    coordinate, resolved_path = load_coordinate_profile(coordinate_source)
    active_sides = validate_coordinate_profile(coordinate, path=resolved_path)
    required = tuple(str(side).strip().lower() for side in required_sides)
    if (
        not required
        or any(side not in {"left", "right"} for side in required)
        or len(required) != len(set(required))
    ):
        raise ValueError("required_sides must contain unique known sides.")
    missing = tuple(side for side in required if side not in active_sides)
    if missing:
        raise ValueError(
            "The selected Sim-HIL profile requires VIVE coordinate sides "
            f"{required}, but the profile provides {active_sides}."
        )
    world_to_desk = coordinate.world_to_desk_pose()
    if world_to_desk is None:
        raise ValueError("VIVE coordinate profile is missing world_to_vive_retarget.")
    coordinate_source_hash = _file_sha256(resolved_path)
    home_profile_hash = (
        None
        if home_profile_path is None
        else _file_sha256(Path(home_profile_path).expanduser())
    )
    derived_home_hash = str(
        coordinate.metadata.get("source_profile_sha256", "")
    ).strip()
    if (
        derived_home_hash
        and home_profile_hash is not None
        and derived_home_hash != home_profile_hash
    ):
        raise ValueError(
            "Stored human Home belongs to another VIVE coordinate profile; "
            "capture a new session Home instead of reinterpreting it."
        )

    merged = copy.deepcopy(calibration)
    merged.set_world_to_desk(world_to_desk)
    for side in active_sides:
        role = f"{side}_wrist"
        merged.set_tracker_offset(role, coordinate.tracker_offsets[role])
    metadata = dict(getattr(merged, "metadata", {}))
    metadata.update(
        {
            "sim_hil_coordinate_profile_path": str(resolved_path),
            "sim_hil_coordinate_profile_kind": str(
                coordinate.metadata.get("profile_kind", "")
            ),
            "sim_hil_coordinate_profile_schema": str(
                coordinate.metadata.get("profile_schema", "")
            ),
            "sim_hil_coordinate_profile_name": str(
                coordinate.metadata.get("profile_name", resolved_path.stem)
            ),
            "canonical_target_frame_contract_id": coordinate.metadata.get(
                "canonical_target_frame_contract_id"
            ),
            "sim_hil_calibration_binding": (
                "independent_vive_coordinate_plus_stored_human_home_v1"
            ),
        }
    )
    merged.metadata = metadata
    binding.update(
        {
            "mode": (
                "independent_vive_coordinate_plus_stored_human_home_v1"
            ),
            "vive_coordinate_profile_path": str(resolved_path),
            "vive_coordinate_profile_sha256": coordinate_source_hash,
            "vive_coordinate_profile_id": str(
                coordinate.metadata.get("profile_name", resolved_path.stem)
            ),
            "coordinate_profile_kind": coordinate.metadata.get("profile_kind"),
            "coordinate_profile_schema": coordinate.metadata.get("profile_schema"),
            "active_sides": list(active_sides),
        }
    )
    return merged, binding


def _bind_sim_target_session_anchors(
    calibration: Any,
    *,
    target: Any,
    calibration_binding: dict[str, Any],
    captured_at: float,
) -> tuple[Any, dict[str, Any]]:
    """Replace stored robot Home with the measured reset-state MuJoCo TCP."""

    from sim.mujoco.deploy import (
        capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    )
    from teleop_runtime.configs import (
        bind_passive_vive_target_session_anchors,
    )

    measured_by_side = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
        target.scene,
        target.task_config,
    )
    exposed_components = set(
        target.target_frame_profile.reference_frame_by_component
    )
    measured_by_side = {
        side: pose
        for side, pose in measured_by_side.items()
        if f"{side}_arm" in exposed_components
    }
    coordinate_profile_id = str(
        calibration_binding.get("vive_coordinate_profile_id")
        or calibration_binding.get("coordinate_profile_kind")
        or "legacy-combined-home-profile"
    )
    coordinate_profile_hash = str(
        calibration_binding.get("vive_coordinate_profile_sha256")
        or calibration_binding.get("home_profile_sha256")
        or ""
    ).strip()
    if not coordinate_profile_hash:
        raise ValueError("Session anchor requires a coordinate profile hash.")
    merged, receipt = bind_passive_vive_target_session_anchors(
        calibration,
        session_id=target.runtime.config.runtime_session_id,
        coordinate_profile_id=coordinate_profile_id,
        coordinate_profile_hash=coordinate_profile_hash,
        target_frame_profile=target.target_frame_profile,
        tool_frame_profile=target.tool_frame_profile,
        side_by_component={
            f"{side}_arm": side for side in measured_by_side
        },
        target_home_pose_reference_by_component={
            f"{side}_arm": pose for side, pose in measured_by_side.items()
        },
        source_anchor_origin="stored_human_home_from_selected_profile",
        target_anchor_origin="measured_mujoco_reset_state_tool_tcp",
        captured_at=float(captured_at),
        clock_domain_id=target.runtime.config.clock_domain_id,
    )
    binding = dict(calibration_binding)
    binding.update(
        {
            "mode": "independent_vive_plus_scene_target_session_anchor_v1",
            "target_frame_profile": target.target_frame_profile.to_payload(),
            "tool_frame_profile": target.tool_frame_profile.to_payload(),
            "session_anchor_receipt": receipt.to_payload(),
        }
    )
    return merged, binding


class _SimHILReadinessProvider:
    """Station binding of lower-owned source and MuJoCo readiness evidence."""

    def __init__(
        self,
        *,
        policy_source: Any,
        human_source: Any,
        scene: Any,
        calibration_ready: bool,
        home_ready: bool,
        max_human_age_sec: float,
        max_policy_age_sec: float,
        clock,
        scene_capture: Callable[[Callable[[Any], object]], object] | None = None,
    ) -> None:
        self.policy_source = policy_source
        self.human_source = human_source
        self.scene = scene
        self.calibration_ready = bool(calibration_ready)
        self.home_ready = bool(home_ready)
        self.max_human_age_sec = float(max_human_age_sec)
        self.max_policy_age_sec = float(max_policy_age_sec)
        self.clock = clock
        if scene_capture is not None and not callable(scene_capture):
            raise TypeError("scene_capture must be callable or None.")
        self._scene_capture = (
            (lambda capture: capture(self.scene))
            if scene_capture is None
            else scene_capture
        )

    @staticmethod
    def _age_detail(
        emission: Any,
        *,
        now: float,
        max_age_sec: float,
    ) -> tuple[bool, str]:
        if emission is None:
            return False, "no_emission"
        emitted_at = float(getattr(emission, "emitted_at", float("nan")))
        if not math.isfinite(emitted_at):
            return False, "emission_timestamp_invalid"
        age = float(now) - emitted_at
        if age < -1e-6:
            return False, f"emission_in_future:{age:.6f}s"
        if age > float(max_age_sec):
            return False, f"emission_stale:{age:.6f}s>{max_age_sec:.6f}s"
        return True, f"fresh:{age:.6f}s"

    def snapshot(self, *, now: float) -> Any:
        from control_station.composition.mujoco_sim_hil import (
            MujocoSimHILReadinessEvidence,
        )

        timestamp = _finite(now, name="readiness.now")
        human_session = self.human_source.session
        raw_human_state = human_session.state_machine.state
        human_state = str(
            getattr(raw_human_state, "value", raw_human_state)
        ).strip()
        human_state_key = human_state.upper()
        human_emission = self.human_source.last_emission
        human_fresh, human_detail = self._age_detail(
            human_emission,
            now=timestamp,
            max_age_sec=self.max_human_age_sec,
        )
        human_rejection = getattr(self.human_source, "last_rejection_reason", None)
        if not human_fresh:
            diagnostic_parts = []
            if human_rejection:
                diagnostic_parts.append(f"rejection={human_rejection}")
            if human_state_key in {"FAULT", "ESTOP", "FROZEN"}:
                diagnostic_parts.append(f"session_state={human_state}")
            hold_reason = str(
                getattr(human_session, "motion_hold_reason", "") or ""
            ).strip()
            hold_mode = str(
                getattr(human_session, "motion_hold_mode", "") or ""
            ).strip()
            if hold_mode and hold_mode not in {"none", ""}:
                diagnostic_parts.append(f"hold_mode={hold_mode}")
            if hold_reason:
                diagnostic_parts.append(f"hold_reason={hold_reason}")
            if bool(getattr(human_session, "home_recalibration_required", False)):
                diagnostic_parts.append("runtime_home_recalibration_required=true")
            if diagnostic_parts:
                human_detail = f"{human_detail};" + ";".join(diagnostic_parts)
        policy_emission = self.policy_source.last_emission
        policy_fresh, policy_detail = self._age_detail(
            policy_emission,
            now=timestamp,
            max_age_sec=self.max_policy_age_sec,
        )
        policy_ttl = False
        policy_ttl_detail = "no_emission"
        if policy_emission is not None:
            effective_until = float(getattr(policy_emission, "effective_until", float("nan")))
            policy_ttl = math.isfinite(effective_until) and effective_until > timestamp
            policy_ttl_detail = (
                f"valid_until:{effective_until:.6f}"
                if policy_ttl
                else "policy_chunk_ttl_expired"
            )

        faulted = human_state_key in {"FAULT", "ESTOP", "FROZEN"}
        home_recalibration_required = bool(
            getattr(human_session, "home_recalibration_required", False)
        )
        hold_mode = str(getattr(human_session, "motion_hold_mode", ""))
        human_ready = human_fresh and not faulted and not home_recalibration_required
        rebase_ready = self.home_ready and not home_recalibration_required
        # The passive source never arms a real target.  DEBUG_ONLY is the
        # explicit non-motion deadman/paused gate for this sim-only profile.
        deadman_ready = (
            human_state_key == "DEBUG_ONLY"
            and not faulted
            and not home_recalibration_required
            # ``grace`` means the passive session noticed a transient tracker
            # gap but has not crossed its profile-local automatic HOLD
            # threshold.  Sim-HIL keeps the composition alive during that
            # network window; proposal TTL still prevents an old pose from
            # being executed.  Real teleop uses its own readiness gate.
            and hold_mode in {"", "none", "grace"}
        )
        policy_ready = bool(
            getattr(self.policy_source, "ready", False)
            and policy_fresh
        )
        measured_state_ready = bool(
            self._scene_capture(lambda scene: scene.state_is_finite())
        )
        details = {
            "policy_source_ready": (
                "ready" if policy_ready else str(getattr(self.policy_source, "last_error", None) or policy_detail)
            ),
            "human_device_ready": human_detail,
            "calibration_ready": "loaded_and_acknowledged" if self.calibration_ready else "calibration_not_ready",
            "home_ready": "anchors_seeded" if self.home_ready else "home_anchors_missing",
            "rebase_ready": "not_required" if rebase_ready else "rebase_required",
            "deadman_ready": (
                "passive_debug_only"
                if deadman_ready and hold_mode in {"", "none"}
                else "passive_debug_only_grace"
                if deadman_ready
                else f"teleop_state:{human_state}"
            ),
            "policy_chunk_ttl_ready": policy_ttl_detail,
            "measured_state_ready": "finite" if measured_state_ready else "mujoco_state_nonfinite",
        }
        return MujocoSimHILReadinessEvidence(
            policy_source_ready=policy_ready,
            human_device_ready=human_ready,
            calibration_ready=self.calibration_ready,
            home_ready=self.home_ready,
            rebase_ready=rebase_ready,
            deadman_ready=deadman_ready,
            policy_chunk_ttl_ready=bool(policy_ttl and policy_fresh),
            measured_state_ready=measured_state_ready,
            detail_by_check=details,
        )


def _build_application(
    config: MujocoSimHILApplicationConfig,
    *,
    trace_sink: Any | None = None,
) -> "MujocoSimHILApplication":
    if not config.enabled:
        raise RuntimeError("MuJoCo Sim-HIL application is default-off; pass --enable.")
    if config.left_q6_port is not None:
        raise ValueError(
            "The selected right-arm profile does not consume --left-q6-port."
        )

    from teleop_runtime.configs import resolve_single_gripper_input

    from control_station.composition.mujoco_sim_hil import (
        MujocoSimHILProfileConfig,
        build_component_keyed_mujoco_sim_hil_contracts,
        build_mujoco_sim_hil_policy_binding,
        build_scoped_mujoco_sim_hil_composition,
    )
    from control_composition.observability import SourceDifferenceSpec
    from robot_motion.action import (
        BIMANUAL_EE_POSE_V1,
        ExecutionObservationPlan,
        HandActionSpec,
    )
    from sim.mujoco.deploy import (
        MujocoComponentActionRobotActionV1Materializer,
        MujocoExecutionDifferenceObserver,
        MujocoRobotActionV1ComponentIngressAdapter,
        MujocoSimHILTargetConfig,
        build_mujoco_component_execution_binding,
        build_rm65_right_cube_lift_component_target_profile,
        build_mujoco_sim_hil_target,
        capture_mujoco_sim_hil_tool_tcp_pose_by_side,
        prepare_mujoco_sim_hil_target,
    )
    from sim.mujoco_temp_pipeline.hil_policy import TempPPOHILPolicySource
    from teleop_runtime.calibration import TeleopCalibration
    from teleop_runtime.configs import PassiveSingleViveGripperSessionConfig

    hand_specs = {"right": HandActionSpec.dmtac_gripper("right")}
    policy_effectors = ("right_arm", "right_hand")
    execution_observation_plan = (
        ExecutionObservationPlan(
            plan_id="control-station-sim-hil-right-arm-execution-difference-v1",
            reference_mode="full_frames",
            measurement_mode="full_frames",
            component_ids=("right_arm",),
        )
        if config.execution_difference_enabled
        else None
    )
    execution_difference_observer = (
        MujocoExecutionDifferenceObserver(
            selection_group_id="arms",
            component_ids=("right_arm",),
        )
        if execution_observation_plan is not None
        else None
    )
    target_config = MujocoSimHILTargetConfig(
        task_config_path=config.task_config_path,
        realtime_worker=config.realtime_worker,
        position_control_profile_name=config.position_control_profile_name,
        compose_target_hz=config.compose_target_hz,
        target_admission_margin_sec=config.target_admission_margin_sec,
        execution_observation_plan=execution_observation_plan,
        exposed_pose_sides=("right",),
    )
    prepared_target = prepare_mujoco_sim_hil_target(target_config)
    component_profile = build_rm65_right_cube_lift_component_target_profile(
        prepared_target.scene,
        prepared_target.task_config,
    )
    action_schema = component_profile.action_schema
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        action_schema,
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.25,
        policy_active_components=action_schema.component_ids,
        human_weight=config.initial_human_alpha,
        policy_weight=1.0 - config.initial_human_alpha,
        max_pose_translation_disagreement_m=None,
        max_pose_rotation_disagreement_rad=None,
    )
    runtime_rig_holder: dict[str, Any] = {}

    def capture_support_arm_pose() -> Mapping[str, Any]:
        rig = runtime_rig_holder.get("rig")
        if rig is None:
            raise RuntimeError(
                "MuJoCo support-pose capture requested before target binding."
            )
        return rig.capture_scene(
            lambda scene: {
                "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    prepared_target.task_config,
                )["left"]
            }
        )

    execution_materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared_target.scene,
        component_profile,
        hand_action_spec_by_component={
            "right_gripper": hand_specs["right"],
        },
        support_arm_pose_provider=capture_support_arm_pose,
        support_arm_sides=("left",),
        support_pose_contract_id=(
            "mujoco.measured_left_tool_tcp_support_for_coupled_ik.v1"
        ),
    )
    component_execution_binding = build_mujoco_component_execution_binding(
        component_profile,
        execution_materializer,
    )
    target = build_mujoco_sim_hil_target(
        config=target_config,
        contracts=contracts,
        hand_specs_by_side=hand_specs,
        prepared_target=prepared_target,
        component_execution_binding=component_execution_binding,
        execution_observation_sink=execution_difference_observer,
        trace_sink=trace_sink,
        composition_input_action_schema_by_role={
            "human": action_schema,
            "policy": action_schema,
        },
    )
    runtime_rig_holder["rig"] = target.runtime.rig
    if target.runtime.action_schema.semantic_content_hash != (
        action_schema.semantic_content_hash
    ):
        raise RuntimeError(
            "MuJoCo Sim-HIL target built a different ComponentActionSchema."
        )
    from control_station.sources.teleop_pose import (
        AttachablePassiveTeleopComponentSource,
    )

    target_action_hashes = dict(
        target.runtime.target_runtime_action_contract
        .action_spec_hash_by_component
    )
    arm_payload_schema = action_schema.payload_schema_by_component[
        "right_arm"
    ]
    human_source_spec = (
        _build_single_vive_gripper_component_source_spec(
            arm_reference_frame=arm_payload_schema.reference_frame,
            target_action_spec_hash_by_component=target_action_hashes,
        )
    )
    human_source = AttachablePassiveTeleopComponentSource(
        human_source_spec
    )

    def build_human_attachment(
        coordinate_profile_path: Path,
        source_role: str,
    ) -> _SimHILHumanAttachment:
        gripper_selection = resolve_single_gripper_input(
            side="right",
            requested_source=config.gripper_input_source,
            q6_port=config.right_q6_port,
            q6_config=config.q6_config,
        )
        calibration = TeleopCalibration.load_json(
            config.calibration_profile_path
        )
        calibration, calibration_binding = _bind_vive_coordinate_profile(
            calibration,
            coordinate_profile_path=coordinate_profile_path,
            home_profile_path=config.calibration_profile_path,
            required_sides=("right",),
        )
        calibration_binding["home_profile_path"] = str(
            config.calibration_profile_path
        )
        calibration_binding["home_profile_sha256"] = _file_sha256(
            config.calibration_profile_path
        )
        calibration, calibration_binding = _bind_sim_target_session_anchors(
            calibration,
            target=target,
            calibration_binding=calibration_binding,
            captured_at=target.runtime.clock(),
        )
        if calibration.metadata.get("home_ready") is not True:
            raise ValueError(
                "Sim-HIL requires a Home-capable calibration profile with "
                "metadata.home_ready=true."
            )
        source, actual_spec = (
            _build_passive_single_vive_gripper_component_source(
                calibration=calibration,
                source_config=PassiveSingleViveGripperSessionConfig(
                    side="right",
                    gripper_input_source=(
                        gripper_selection.resolved_source
                    ),
                    vive_port=config.vive_port,
                    vive_clock_sync_port=config.vive_clock_sync_port,
                    vive_source_tracker_role=source_role,
                    q6_port=gripper_selection.q6_port,
                    angle_trigger_config_root=(
                        config.angle_trigger_config_root
                    ),
                    angle_trigger_rate_hz=config.angle_trigger_rate_hz,
                    tracker_auto_hold_sec=config.human_emission_timeout_sec,
                    tracker_manual_hold_sec=config.human_emission_timeout_sec,
                ),
                target_action_spec_hash_by_component=(
                    target_action_hashes
                ),
                target_units_convention_id=(
                    target.runtime.target_runtime_action_contract
                    .units_convention_id
                ),
                source_spec=human_source_spec,
            )
        )
        if actual_spec.semantic_content_hash != (
            human_source_spec.semantic_content_hash
        ):
            source.close()
            raise RuntimeError(
                "Human attachment changed the reserved source contract."
            )

        def capture_runtime_home_pose() -> Mapping[str, Any]:
            return {
                side: calibration.base_to_desk(pose, side)
                for side, pose in dict(
                    target.runtime.rig.capture_scene(
                        lambda scene: (
                            capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                                scene,
                                target.task_config,
                            )
                        )
                    )
                ).items()
                if side == "right"
            }

        return _SimHILHumanAttachment(
            source=source,
            gripper_input_selection=gripper_selection,
            calibration_binding=calibration_binding,
            runtime_home_pose_desk_by_side_capture=(
                capture_runtime_home_pose
            ),
        )

    if config.defer_human_input:
        gripper_input_selection = SimpleNamespace(
            requested_source=config.gripper_input_source,
            resolved_source=None,
            q6_port=None,
            fallback_reason="deferred_until_human_mode",
        )
        calibration_binding = {
            "mode": "human_input_deferred",
            "vive_coordinate_profile_path": None,
            "home_profile_path": None,
        }
        calibration_ready = False
        home_ready = False
        runtime_home_capture = None
    else:
        if config.vive_coordinate_profile_path is None:
            raise ValueError(
                "An attached human source requires a VIVE coordinate profile."
            )
        initial_attachment = build_human_attachment(
            config.vive_coordinate_profile_path,
            config.vive_source_tracker_role,
        )
        human_source.attach(initial_attachment.source)
        gripper_input_selection = (
            initial_attachment.gripper_input_selection
        )
        calibration_binding = dict(
            initial_attachment.calibration_binding
        )
        calibration_ready = True
        home_ready = True
        runtime_home_capture = (
            initial_attachment.runtime_home_pose_desk_by_side_capture
        )
    policy_source = TempPPOHILPolicySource(
        target.scene,
        checkpoint_path=config.checkpoint_path,
        task_config_path=config.task_config_path,
        policy_node_id=target.runtime.config.policy_node_id,
        policy_clock_domain_id=target.runtime.config.clock_domain_id,
        device=config.device,
        task_initialization_info=target.task_initialization_info,
        downstream_target_lead_sec=(
            target.runtime.reference_activation_spec.target_lead_time_sec
        ),
        scene_capture=target.runtime.rig.capture_scene,
    )
    ingress_projector = MujocoRobotActionV1ComponentIngressAdapter(
        target.scene,
        component_profile,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )
    policy_binding = build_mujoco_sim_hil_policy_binding(
        target.runtime,
        policy_active_effectors=policy_effectors,
        source_action_space=BIMANUAL_EE_POSE_V1,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
        component_projection_contract_hash=(
            ingress_projector.semantic_content_hash
        ),
    )
    profile_config = MujocoSimHILProfileConfig(
        enabled=True,
        initial_human_alpha=config.initial_human_alpha,
        max_human_emission_age_sec=config.human_emission_timeout_sec,
        max_policy_emission_age_sec=policy_source.maximum_hold_sec,
        physics_ticks_after_dispatch=(
            0
            if config.realtime_worker
            else max(
                1,
                int(
                    round(
                        target.task_config.physics_hz
                        / target.runtime.config.decision_hz
                    )
                ),
            )
        ),
    )
    readiness_provider = _SimHILReadinessProvider(
        policy_source=policy_source,
        human_source=human_source,
        scene=target.scene,
        calibration_ready=calibration_ready,
        home_ready=home_ready,
        max_human_age_sec=profile_config.max_human_emission_age_sec,
        max_policy_age_sec=profile_config.max_policy_emission_age_sec,
        clock=target.runtime.clock,
        scene_capture=target.runtime.rig.capture_scene,
    )
    composition = build_scoped_mujoco_sim_hil_composition(
        runtime=target.runtime,
        human_source=human_source,
        human_source_spec=human_source_spec,
        readiness_provider=readiness_provider,
        contracts=contracts,
        config=profile_config,
        policy_active_effectors=policy_effectors,
        policy_binding=policy_binding,
        policy_component_action_projector=ingress_projector,
        human_component_action_projector=ingress_projector,
        human_component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
        source_difference_spec=SourceDifferenceSpec(
            spec_id="control-station-sim-hil-immediate-difference-v1",
            composition_spec_hash=(
                contracts.composition_spec.semantic_content_hash
            ),
            compared_reference_role_pairs=(("human", "policy"),),
            output_residual_roles=("human", "policy"),
        ),
    )
    viewer = None
    if config.mujoco_viewer_enabled:
        from sim.mujoco.viewer import MujocoPassiveViewerSession

        viewer = MujocoPassiveViewerSession(
            target.scene,
            sync_callback=target.runtime.rig.sync_viewer,
        )
    return MujocoSimHILApplication(
        config=config,
        task_config=target.task_config,
        scene=target.scene,
        runtime=target.runtime,
        composition=composition,
        policy_source=policy_source,
        human_source=human_source,
        readiness_provider=readiness_provider,
        execution_difference_observer=execution_difference_observer,
        gripper_input_selection=gripper_input_selection,
        calibration_binding=calibration_binding,
        task_initialization_info=dict(target.task_initialization_info),
        viewer=viewer,
        runtime_home_pose_desk_by_side_capture=runtime_home_capture,
        human_attachment_factory=build_human_attachment,
    )


def _build_passive_vive_q6_component_source(
    *,
    calibration: Any,
    source_config: Any,
    target_action_spec_hash_by_component: dict[str, str],
    target_units_convention_id: str,
) -> tuple[Any, Any]:
    """Bind the concrete VIVE/Q6 session at the Sim-HIL application leaf.

    ``control_station.sources.teleop_pose`` remains a device-neutral adapter.
    The lower ``teleop_runtime`` builder still owns VIVE/Q6 construction,
    calibration, and device lifecycle; this helper only selects that source
    for this concrete MuJoCo profile and binds it to target component hashes.
    """

    from control_station.sources.teleop_pose import (
        PassiveTeleopComponentClockBridge,
        PassiveTeleopComponentSessionSource,
        PassiveTeleopComponentSourceSpec,
        PassiveTeleopComponentSpec,
    )
    from robot_motion.action import ArmActionSpec, HandActionSpec
    from teleop_runtime.configs.passive_vive_q6 import (
        build_passive_vive_q6_session,
    )

    required_components = {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    hashes = {
        str(component): str(value).strip()
        for component, value in dict(target_action_spec_hash_by_component).items()
    }
    if set(hashes) != required_components or any(not value for value in hashes.values()):
        raise ValueError(
            "Passive VIVE/Q6 source requires exact target hashes for the four "
            "bimanual arm/hand components."
        )
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    session = build_passive_vive_q6_session(
        calibration=calibration,
        source_config=source_config,
    )

    components: list[Any] = []
    for side in ("left", "right"):
        arm_spec = ArmActionSpec.tool_tcp(side, reference_frame=f"{side}_arm_base")
        components.append(
            PassiveTeleopComponentSpec.arm_pose(
                f"{side}_arm",
                f"{side}_arm",
                action_spec_hash=hashes[f"{side}_arm"],
                arm_action_spec=arm_spec,
            )
        )
    for side in ("left", "right"):
        components.append(
            PassiveTeleopComponentSpec.hand_qpos(
                f"{side}_hand",
                f"{side}_hand",
                action_spec_hash=hashes[f"{side}_hand"],
                hand_action_spec=hand_specs[side],
            )
        )
    source_spec = PassiveTeleopComponentSourceSpec(
        components=tuple(components),
        source_id="vive-q6-human-passive",
        observer_binding_id="vive-q6-human-passive-sim-hil-v1",
        source_profile_id="real-vive-q6-passive-mujoco-target-v1",
        units_convention_id=str(target_units_convention_id),
    )
    source = PassiveTeleopComponentSessionSource(session, source_spec)
    return PassiveTeleopComponentClockBridge(source), source_spec


def _build_single_vive_gripper_component_source_spec(
    *,
    arm_reference_frame: str,
    target_action_spec_hash_by_component: Mapping[str, str],
) -> Any:
    """Declare the stable human component contract before devices exist."""

    from control_composition.composition import (
        hand_action_spec_semantic_hash,
    )
    from control_station.sources.teleop_pose import (
        PassiveTeleopComponentSourceSpec,
        PassiveTeleopComponentSpec,
    )
    from robot_motion.action import ArmActionSpec, HandActionSpec

    hashes = {
        str(component): str(value).strip()
        for component, value in dict(
            target_action_spec_hash_by_component
        ).items()
    }
    if set(hashes) != {"right_arm", "right_gripper"} or any(
        not value for value in hashes.values()
    ):
        raise ValueError(
            "Single-side human source requires right_arm/right_gripper "
            "target payload identities."
        )
    reference_frame = str(arm_reference_frame).strip()
    if not reference_frame:
        raise ValueError("arm_reference_frame must be non-empty.")
    hand_spec = HandActionSpec.dmtac_gripper("right")
    return PassiveTeleopComponentSourceSpec(
        components=(
            PassiveTeleopComponentSpec.arm_pose(
                "right_arm",
                "right_arm",
                action_spec_hash=hashes["right_arm"],
                arm_action_spec=ArmActionSpec.tool_tcp(
                    "right",
                    reference_frame=reference_frame,
                ),
            ),
            PassiveTeleopComponentSpec.hand_qpos(
                "right_gripper",
                "right_hand",
                action_spec_hash=hand_action_spec_semantic_hash(hand_spec),
                hand_action_spec=hand_spec,
            ),
        ),
        source_id="sim-hil-human-passive-right",
        observer_binding_id="sim-hil-human-passive-right-v1",
        source_profile_id="single-human-right-mujoco-target-v1",
        units_convention_id=(
            "si_m_rad_xyzw_plus_normalized_hand_v1"
        ),
    )


def _build_passive_single_vive_gripper_component_source(
    *,
    calibration: Any,
    source_config: Any,
    target_action_spec_hash_by_component: dict[str, str],
    target_units_convention_id: str,
    source_spec: Any | None = None,
) -> tuple[Any, Any]:
    """Bind one right-side VIVE/gripper producer without motion ownership."""

    if not str(target_units_convention_id).strip():
        raise ValueError("target_units_convention_id must be non-empty.")

    from control_station.sources.teleop_pose import (
        PassiveTeleopComponentClockBridge,
        PassiveTeleopComponentSessionSource,
        PassiveTeleopComponentSourceSpec,
    )
    from teleop_runtime.configs import (
        build_passive_single_vive_gripper_session,
    )
    from teleop_runtime.real_teleop.defaults import (
        DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
        DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
    )
    from teleop_runtime.safety import TeleopSafetyConfig

    hashes = {
        str(component): str(value).strip()
        for component, value in dict(
            target_action_spec_hash_by_component
        ).items()
    }
    if set(hashes) != {"right_arm", "right_gripper"} or any(
        not value for value in hashes.values()
    ):
        raise ValueError(
            "Single-side VIVE/gripper source requires right_arm/right_gripper "
            "target payload identities."
        )
    session = build_passive_single_vive_gripper_session(
        calibration=calibration,
        source_config=source_config,
        safety_config=TeleopSafetyConfig(
            # Sim-HIL consumes the same physical VIVE stream as formal
            # teleoperation.  Do not silently apply the much tighter generic
            # 10 cm / 45 degree fixture defaults at this application binding.
            max_translation_jump_m=(
                DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M
            ),
            max_rotation_jump_deg=(
                DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG
            ),
            # The receiver is latest-value while composition is lower-rate.
            # Never compare poses across a consumer stall as though they were
            # adjacent tracker samples.
            pose_jump_baseline_max_gap_sec=0.25,
        ),
    )
    # The passive source contract must be derived from the retargeter that
    # will actually emit commands.  The single-side profile currently binds
    # ``right_arm_base`` through ``seed_vive_home_anchors``; deriving it here
    # keeps the producer contract correct if a future target profile changes
    # the output frame, and avoids a second hard-coded frame identity.
    arm_retargeters = tuple(
        retargeter
        for retargeter in tuple(getattr(session, "retargeters", ()) or ())
        if str(
            getattr(getattr(retargeter, "config", None), "effector_name", "")
        ).strip()
        == "right_arm"
    )
    if len(arm_retargeters) != 1:
        raise ValueError(
            "Single-side VIVE/gripper session must expose exactly one "
            f"right_arm pose retargeter; found {len(arm_retargeters)}."
        )
    arm_reference_frame = str(
        getattr(
            getattr(arm_retargeters[0], "config", None),
            "output_frame_name",
            "",
        )
    ).strip()
    if not arm_reference_frame:
        raise ValueError(
            "The right_arm pose retargeter must declare output_frame_name."
        )
    expected_spec = (
        _build_single_vive_gripper_component_source_spec(
            arm_reference_frame=arm_reference_frame,
            target_action_spec_hash_by_component=hashes,
        )
        if source_spec is None
        else source_spec
    )
    if not isinstance(
        expected_spec,
        PassiveTeleopComponentSourceSpec,
    ):
        raise TypeError(
            "source_spec must be a PassiveTeleopComponentSourceSpec."
        )
    component_by_id = {
        item.component_id: item for item in expected_spec.components
    }
    if set(component_by_id) != {"right_arm", "right_gripper"}:
        raise ValueError(
            "The single-side source contract must expose the target component "
            "IDs right_arm/right_gripper."
        )
    arm_components = tuple(
        item
        for item in expected_spec.components
        if item.effector_name == "right_arm"
    )
    if (
        len(arm_components) != 1
        or arm_components[0].arm_action_spec.reference_frame
        != arm_reference_frame
    ):
        raise ValueError(
            "The attached VIVE retargeter frame does not match the reserved "
            "human source contract."
        )
    source = PassiveTeleopComponentSessionSource(session, expected_spec)
    return PassiveTeleopComponentClockBridge(source), expected_spec


def _emission_status(emission: Any, *, now: float) -> dict[str, Any] | None:
    if emission is None:
        return None
    emitted_at = getattr(emission, "emitted_at", None)
    result: dict[str, Any] = {
        "emission_id": getattr(emission, "emission_id", None),
        "semantic_content_hash": getattr(emission, "semantic_content_hash", None),
        "action_payload_kind": getattr(emission, "action_payload_kind", None),
        "representation_trace_id": getattr(
            emission, "representation_trace_id", None
        ),
        "source_binding_hash": getattr(emission, "source_binding_hash", None),
        "canonical_policy_action_hash": getattr(
            getattr(emission, "canonical_action", None),
            "semantic_content_hash",
            None,
        ),
        "canonical_action_hash": getattr(
            getattr(
                getattr(emission, "canonical_action", None),
                "canonicalization_receipt",
                None,
            ),
            "canonical_action_hash",
            None,
        ),
        "canonicalization_receipt_hash": getattr(
            getattr(
                getattr(emission, "canonical_action", None),
                "canonicalization_receipt",
                None,
            ),
            "semantic_content_hash",
            None,
        ),
        "emitted_at": emitted_at,
        "age_sec": None if emitted_at is None else float(now) - float(emitted_at),
        "target_effective_at": getattr(emission, "target_effective_at", None),
        "effective_from": getattr(emission, "effective_from", None),
        "effective_until": getattr(emission, "effective_until", None),
        "chunk_id": getattr(emission, "chunk_id", None),
        "chunk_keyframe_index": getattr(emission, "chunk_keyframe_index", None),
        "inference_id": getattr(emission, "inference_id", None),
    }
    return result


def _input_submission_status(submission: Any) -> dict[str, Any] | None:
    if submission is None:
        return None
    receipt = getattr(submission, "input_receipt", None)
    return {
        "role": getattr(submission, "role", None),
        "source_emission_id": getattr(submission, "source_emission_id", None),
        "accepted": bool(getattr(submission, "accepted", False)),
        "error": getattr(submission, "error", None),
        "semantic_content_hash": getattr(submission, "semantic_content_hash", None),
        "receipt": None
        if receipt is None
        else {
            "accepted": bool(getattr(receipt, "accepted", False)),
            "rejection_code": getattr(receipt, "rejection_code", None),
            "message": getattr(receipt, "message", ""),
            "composition_input_id": getattr(receipt, "composition_input_id", None),
            "available_at": getattr(receipt, "available_at", None),
            "available_clock_domain_id": getattr(
                receipt, "available_clock_domain_id", None
            ),
            "control_admission_sequence": getattr(
                receipt, "control_admission_sequence", None
            ),
            "semantic_content_hash": getattr(
                receipt, "semantic_content_hash", None
            ),
        },
    }


def _composition_step_status(step: Any) -> dict[str, Any] | None:
    if step is None:
        return None
    evaluation = getattr(step, "evaluation", None)
    composition_evaluation = (
        None if evaluation is None else getattr(evaluation, "composition_evaluation", None)
    )
    disposition = getattr(step, "disposition", None)
    proposal = getattr(step, "proposal_admission", None)
    hold = getattr(step, "hold_transition", None)
    hold_event = None if hold is None else getattr(hold, "event", None)
    difference = (
        None
        if evaluation is None
        else getattr(evaluation, "source_difference_observation", None)
    )

    def difference_payload(item: Any) -> dict[str, Any]:
        return {
            "compared_role": item.compared_role,
            "reference_role": item.reference_role,
            "available_at_delta_sec": item.available_at_delta_sec,
            "effective_from_delta_sec": item.effective_from_delta_sec,
            "valid_for_sec": item.valid_for_sec,
            "components": [
                {
                    "component_id": metric.component_id,
                    "representation": metric.representation,
                    "translation_delta_xyz_m": list(
                        metric.translation_delta_xyz_m
                    ),
                    "translation_norm_m": metric.translation_norm_m,
                    "rotation_delta_rotvec_rad": list(
                        metric.rotation_delta_rotvec_rad
                    ),
                    "rotation_angle_rad": metric.rotation_angle_rad,
                    "qpos_active_indices": list(metric.qpos_active_indices),
                    "qpos_delta": list(metric.qpos_delta),
                    "qpos_l2_norm": metric.qpos_l2_norm,
                    "qpos_max_abs": metric.qpos_max_abs,
                }
                for metric in item.component_metrics
            ],
        }

    return {
        "decision_key": getattr(step, "decision_key", None),
        "target_effective_at": getattr(step, "target_effective_at", None),
        "semantic_content_hash": getattr(step, "semantic_content_hash", None),
        "readiness_hash": (
            None
            if getattr(step, "readiness", None) is None
            else getattr(step.readiness, "semantic_content_hash", None)
        ),
        "composition_evaluation": None
        if composition_evaluation is None
        else {
            "evaluation_id": getattr(
                composition_evaluation, "composition_evaluation_id", None
            ),
            "semantic_content_hash": getattr(
                composition_evaluation, "semantic_content_hash", None
            ),
            "composition_supported_until": getattr(
                composition_evaluation, "composition_supported_until", None
            ),
        },
        "immediate_difference": None
        if difference is None
        else {
            "observation_id": difference.observation_id,
            "semantic_content_hash": difference.semantic_content_hash,
            "valid_until": difference.valid_until,
            "pairwise": [
                difference_payload(item)
                for item in difference.pairwise_differences
            ],
            "output_residuals": [
                difference_payload(item)
                for item in difference.output_residuals
            ],
        },
        "disposition": None
        if disposition is None
        else {
            "disposition_id": getattr(disposition, "disposition_id", None),
            "downstream_request": getattr(disposition, "downstream_request", None),
            "reason": getattr(disposition, "reason", None),
            "semantic_content_hash": getattr(
                disposition, "semantic_content_hash", None
            ),
        },
        "proposal": None
        if proposal is None
        else {
            "accepted": bool(getattr(proposal, "accepted", False)),
            "proposal_id": getattr(proposal, "proposal_id", None),
            "rejection_code": getattr(proposal, "rejection_code", None),
            "message": getattr(proposal, "message", ""),
            "semantic_content_hash": getattr(
                proposal, "semantic_content_hash", None
            ),
            "effective_from": getattr(
                getattr(proposal, "proposal", None), "effective_from", None
            ),
            "effective_until": getattr(
                getattr(proposal, "proposal", None), "effective_until", None
            ),
            "target_effective_at": getattr(
                getattr(proposal, "proposal", None), "target_effective_at", None
            ),
        },
        "hold": None
        if hold is None
        else {
            "event_id": None if hold_event is None else getattr(hold_event, "event_id", None),
            "status": getattr(getattr(hold, "snapshot", None), "status", None),
            "semantic_content_hash": getattr(
                getattr(hold, "snapshot", None), "semantic_content_hash", None
            ),
        },
        "blocked_reason": getattr(step, "blocked_reason", None),
        "error": getattr(step, "error", None),
    }


def _human_channel_status(step: Any, *, now: float) -> dict[str, Any]:
    session_step = None if step is None else getattr(step, "session_step", None)
    channels = {} if session_step is None else getattr(session_step, "channels", {})
    result: dict[str, Any] = {}
    for name, channel in dict(channels or {}).items():
        timestamp = getattr(channel, "timestamp", None)
        status = {
            "valid": bool(getattr(channel, "valid", False)),
            "capability": getattr(channel, "capability", None),
            "timestamp": timestamp,
            "age_sec": None
            if timestamp is None
            else float(now) - float(timestamp),
        }
        value = getattr(channel, "value", None)
        try:
            serialized = (
                value.tolist()
                if callable(getattr(value, "tolist", None))
                else value
            )
            numeric = (
                [float(serialized)]
                if isinstance(serialized, (int, float))
                else [float(item) for item in serialized]
            )
            if len(numeric) <= 16 and all(math.isfinite(item) for item in numeric):
                status["value"] = numeric[0] if len(numeric) == 1 else numeric
        except (TypeError, ValueError):
            pass
        result[str(name)] = status
    return result


def _human_gripper_target_status(emission: Any) -> dict[str, Any] | None:
    if emission is None:
        return None
    for command in tuple(getattr(emission, "merged_commands", ()) or ()):
        if (
            getattr(command, "effector_name", None) != "right_hand"
            or getattr(command, "command_type", None) != "joint_vector"
        ):
            continue
        metadata = dict(getattr(command, "metadata", {}) or {})
        target = metadata.get("gripper_joint_target")
        if target is None:
            values = getattr(command, "value", ())
            try:
                target = float(values[0])
            except (IndexError, TypeError, ValueError):
                return None
        return {
            "effector_name": "right_hand",
            "component_id": "right_gripper",
            "normalized_close": float(target),
            "input_channel": metadata.get("input_channel"),
            "source_value": metadata.get("source_value"),
        }
    return None


def _mujoco_gripper_execution_status(
    rig: Any,
    *,
    gripper_name: str,
) -> dict[str, Any] | None:
    snapshot = getattr(rig, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        execution = dict(snapshot().get("gripper_execution") or {})
    except Exception:  # noqa: BLE001 - status must not affect control.
        return None
    if not execution:
        return None

    def selected(field: str) -> Any:
        return dict(execution.get(field) or {}).get(gripper_name)

    return {
        "gripper_name": gripper_name,
        "accepted_close": selected("accepted_close_by_gripper"),
        "servo_goal_close": selected("servo_goal_close_by_gripper"),
        "actuator_target_close": selected(
            "actuator_target_close_by_gripper"
        ),
        "measured_close": selected("measured_close_by_gripper"),
        "status": selected("last_status_by_gripper"),
        "pending_delayed_targets": execution.get(
            "pending_delayed_targets"
        ),
    }


def _vive_receiver_status(
    session: Any,
    *,
    wall_now: float | None = None,
) -> dict[str, Any]:
    """Project the passive VIVE receiver into GUI-safe diagnostic facts.

    Upstream role channels are intentionally read before the active
    source-role mapping.  They are display-only and cannot enter the teleop or
    composition paths.
    """

    from teleop_runtime.real_teleop.gui_tracker_status import (
        TRACKER_GUI_STALE_AGE_SEC,
        raw_tracker_pose_state_from_channels,
    )
    from teleop_runtime.sources.pose.vive import VivePoseSource

    sources = [
        source
        for source in tuple(getattr(session, "sources", ()) or ())
        if isinstance(source, VivePoseSource)
    ]
    if len(sources) != 1:
        return {
            "status": "unavailable",
            "message": (
                "Expected exactly one VivePoseSource; "
                f"found {len(sources)}."
            ),
            "trackers": raw_tracker_pose_state_from_channels(
                {},
                active_sides=("left", "right"),
                now=time.time() if wall_now is None else float(wall_now),
            ),
        }

    source = sources[0]
    now = time.time() if wall_now is None else float(wall_now)
    stats = dict(source.debug_stats())
    upstream = source.get_latest_upstream_role_channels()
    tracker_status = raw_tracker_pose_state_from_channels(
        {
            "left_wrist_pose": upstream.get("left_wrist"),
            "right_wrist_pose": upstream.get("right_wrist"),
        },
        active_sides=("left", "right"),
        now=now,
    )
    role_by_output = source.source_role_by_output_channel
    for side, tracker_role in (
        ("left", "left_wrist"),
        ("right", "right_wrist"),
    ):
        tracker_status[side]["selected_for_output_channels"] = sorted(
            channel_name
            for channel_name, selected_role in role_by_output.items()
            if selected_role == tracker_role
        )

    # Transport freshness must be measured from socket arrival, not from the
    # later parse/publication completion.  The latter can be delayed by a
    # malformed or unusually large batch and is kept as a separate diagnostic.
    last_packet_at = stats.get("last_datagram_received_at")
    if last_packet_at is None:
        last_packet_at = stats.get("last_packet_received_at")
    packet_age_wall = (
        None
        if last_packet_at is None
        else max(0.0, now - float(last_packet_at))
    )
    last_packet_monotonic = stats.get(
        "last_datagram_received_monotonic_time"
    )
    if last_packet_monotonic is None:
        last_packet_monotonic = stats.get("last_packet_received_monotonic_time")
    packet_age_monotonic = (
        None
        if last_packet_monotonic is None
        else max(0.0, time.perf_counter() - float(last_packet_monotonic))
    )
    # Tests/diagnostic callers may pass an explicit wall-time cut.  The live
    # GUI path does not, so it uses the monotonic age and cannot be confused by
    # NTP/manual wall-clock changes.
    packet_age = (
        packet_age_wall
        if wall_now is not None or packet_age_monotonic is None
        else packet_age_monotonic
    )
    running = bool(stats.get("running", False))
    worker_alive = bool(stats.get("worker_alive", running))
    worker_started_once = bool(stats.get("worker_started_once", False))
    if not running:
        receiver_status = "stopped"
    elif worker_started_once and not worker_alive:
        receiver_status = "worker_dead"
    elif packet_age is None:
        receiver_status = "waiting_for_packet"
    elif packet_age > TRACKER_GUI_STALE_AGE_SEC:
        receiver_status = "packet_stale"
    else:
        receiver_status = "receiving"

    return {
        "status": receiver_status,
        "bind_host": stats.get("bind_host"),
        "port": stats.get("port"),
        "packet_count": int(
            stats.get("datagram_count")
            if int(stats.get("datagram_count") or 0) > 0
            else (stats.get("packet_count") or 0)
        ),
        "processed_packet_count": int(stats.get("packet_count") or 0),
        "worker_alive": worker_alive,
        "worker_started_once": worker_started_once,
        "last_packet_received_at": last_packet_at,
        "packet_age_sec": (
            None if packet_age is None else round(packet_age, 3)
        ),
        "packet_age_wall_sec": (
            None if packet_age_wall is None else round(packet_age_wall, 3)
        ),
        "packet_age_monotonic_sec": (
            None
            if packet_age_monotonic is None
            else round(packet_age_monotonic, 3)
        ),
        "recv_error_count": int(stats.get("recv_error_count") or 0),
        "last_recv_error": stats.get("last_recv_error"),
        "process_error_count": int(stats.get("process_error_count") or 0),
        "last_process_error": stats.get("last_process_error"),
        "last_process_duration_sec": stats.get("last_process_duration_sec"),
        "last_process_finished_monotonic_time": stats.get(
            "last_process_finished_monotonic_time"
        ),
        "last_process_batch_size": int(
            stats.get("last_process_batch_size") or 0
        ),
        "last_process_parse_error_count": int(
            stats.get("last_process_parse_error_count") or 0
        ),
        "dropped_stale_packet_count": int(
            stats.get("dropped_stale_packet_count") or 0
        ),
        "source_role_by_output_channel": dict(role_by_output),
        "source_mapping_generation": int(
            stats.get("source_mapping_generation") or 0
        ),
        "channel_update_counts": dict(
            stats.get("channel_update_counts") or {}
        ),
        "last_channel_update_at": dict(
            stats.get("last_channel_update_at") or {}
        ),
        "upstream_role_update_counts": dict(
            stats.get("upstream_role_update_counts") or {}
        ),
        "last_upstream_role_update_at": dict(
            stats.get("last_upstream_role_update_at") or {}
        ),
        "trackers": tracker_status,
    }


def _runtime_contract_status(runtime: Any, composition: Any) -> dict[str, Any]:
    contracts = getattr(composition, "contracts", None)
    composition_spec = getattr(contracts, "composition_spec", None)
    strategy_spec = getattr(contracts, "strategy_spec", None)
    authority_profile = getattr(contracts, "authority_profile", None)
    runtime_config = getattr(runtime, "config", None)
    target_contract = getattr(runtime, "target_runtime_action_contract", None)
    target_capability = getattr(runtime, "component_target_capability", None)
    target_port = getattr(runtime, "target_execution_port", None)
    return {
        "runtime_session_id": getattr(runtime_config, "runtime_session_id", None),
        "control_domain_id": getattr(runtime_config, "control_domain_id", None),
        "clock_domain_id": getattr(runtime_config, "clock_domain_id", None),
        "action_schema_hash": getattr(
            getattr(runtime, "action_schema", None), "semantic_content_hash", None
        ),
        "target": {
            "handoff_enabled": bool(
                getattr(runtime_config, "enable_component_execution_handoff", False)
            ),
            "target_handler_id": getattr(target_port, "target_handler_id", None),
            "activation_mode": getattr(target_port, "activation_mode", None),
            "target_runtime_action_contract_hash": getattr(
                target_contract, "semantic_content_hash", None
            ),
            "component_target_capability_id": getattr(
                target_capability, "capability_id", None
            ),
            "component_target_capability_hash": getattr(
                target_capability, "semantic_content_hash", None
            ),
            "component_target_action_schema_hash": getattr(
                target_capability, "action_schema_hash", None
            ),
            "execution_capability_profile_hash": getattr(
                runtime, "execution_capability_profile_hash", None
            ),
        },
        "composition": {
            "spec_id": getattr(composition_spec, "spec_id", None),
            "spec_hash": getattr(composition_spec, "semantic_content_hash", None),
            "required_roles": list(
                getattr(composition_spec, "required_roles", ())
            ),
        },
        "strategy": {
            "spec_id": getattr(strategy_spec, "strategy_id", None),
            "spec_hash": getattr(strategy_spec, "semantic_content_hash", None),
        },
        "authority": {
            "profile_id": getattr(authority_profile, "profile_id", None),
            "profile_hash": getattr(authority_profile, "semantic_content_hash", None),
            "single_execution_owner": (
                "MujocoDeployRuntime/composer"
                if getattr(runtime_config, "enable_composer_authority", False)
                and not getattr(runtime_config, "enable_primary_policy_authority", True)
                and not getattr(runtime_config, "enable_replay_authority", True)
                else "invalid"
            ),
        },
    }


def _target_execution_status(snapshot: Any) -> dict[str, Any] | None:
    """Render a lower-owned immutable snapshot without deriving control state."""

    if snapshot is None:
        return None
    stages: dict[str, dict[str, Any]] = {}
    for name in ("request", "submission", "completion", "admission", "activation"):
        stage = getattr(snapshot, name)
        stages[name] = {
            "status": stage.status,
            "record_id": stage.record_id,
            "record_hash": stage.record_hash,
            "available_at": stage.available_at,
        }
    return {
        "snapshot_id": snapshot.snapshot_id,
        "semantic_content_hash": snapshot.semantic_content_hash,
        "captured_at": snapshot.captured_at,
        "control_domain_id": snapshot.control_domain_id,
        "target_handler_id": snapshot.target_handler_id,
        "clock_domain_id": snapshot.clock_domain_id,
        "activation_mode": snapshot.activation_mode,
        "request_id": snapshot.request_id,
        "target_id": snapshot.target_id,
        "target_hash": snapshot.target_hash,
        "stages": stages,
    }


def _execution_difference_status(observer: Any | None) -> dict[str, Any]:
    if observer is None:
        return {"enabled": False}
    snapshot = observer.snapshot()
    association = snapshot.latest_association
    return {
        "enabled": True,
        "reference_count": snapshot.reference_count,
        "measurement_count": snapshot.measurement_count,
        "unmatched_measurement_count": snapshot.unmatched_measurement_count,
        "pending_reference_count": snapshot.pending_reference_count,
        "watermarks": {
            stream: {
                "status": watermark.status,
                "reason": watermark.reason,
                "processed_through": watermark.processed_through,
                "emitted_count": watermark.emitted_count,
                "dropped_count": watermark.dropped_count,
                "semantic_content_hash": watermark.semantic_content_hash,
            }
            for stream, watermark in snapshot.watermark_by_stream.items()
        },
        "latest": None
        if association is None
        else {
            "association_id": association.association_id,
            "semantic_content_hash": association.semantic_content_hash,
            "status": association.status,
            "reason": association.reason,
            "measurement_id": association.measurement_id,
            "reference_frame_id": association.reference_frame_id,
            "measured_at": association.measured_at,
            "execution_fence_sequence": (
                association.execution_fence_sequence
            ),
            "reference_revision": association.reference_revision,
            "component_difference": {
                component_id: dict(value)
                for component_id, value in (
                    association.component_difference.items()
                )
            },
        },
    }


def _mix_status(
    alpha_by_component: Mapping[str, float],
) -> dict[str, Any]:
    values = {
        str(component_id): float(value)
        for component_id, value in dict(alpha_by_component).items()
    }
    if values and all(value <= 1e-12 for value in values.values()):
        mode = "policy_only"
    elif values and all(
        value >= 1.0 - 1e-12 for value in values.values()
    ):
        mode = "human_only"
    else:
        mode = "proportional_blend"
    return {
        "mode": mode,
        "human_alpha_by_component": values,
    }


class MujocoSimHILApplication:
    """Own only the lifecycle of one Station-selected Sim-HIL composition."""

    def __init__(
        self,
        *,
        config: MujocoSimHILApplicationConfig,
        task_config: Any,
        scene: Any,
        runtime: Any,
        composition: Any,
        policy_source: Any,
        human_source: Any,
        readiness_provider: Any,
        execution_difference_observer: Any | None = None,
        gripper_input_selection: Any | None = None,
        calibration_binding: dict[str, Any] | None = None,
        task_initialization_info: dict[str, object] | None = None,
        viewer: Any | None = None,
        runtime_home_pose_desk_by_side_capture: (
            Callable[[], Mapping[str, Any]] | None
        ) = None,
        human_attachment_factory: (
            Callable[[Path, str], _SimHILHumanAttachment] | None
        ) = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.config = config
        self.task_config = task_config
        self.scene = scene
        self.runtime = runtime
        self.composition = composition
        self.policy_source = policy_source
        self.human_source = human_source
        self.readiness_provider = readiness_provider
        self.execution_difference_observer = execution_difference_observer
        self.gripper_input_selection = gripper_input_selection
        self.calibration_binding = dict(calibration_binding or {})
        self.task_initialization_info = dict(task_initialization_info or {})
        self.viewer = viewer
        self._runtime_home_pose_desk_by_side_capture = (
            runtime_home_pose_desk_by_side_capture
        )
        self._human_attachment_factory = human_attachment_factory
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.RLock()
        self._started = False
        self._sources_started = False
        self._last_preflight: Any | None = None
        self._last_cycle: Any | None = None
        self._last_human_input_check_step: Any | None = None
        self._human_input_check_report: dict[str, object] | None = None
        self._human_input_check_active = False
        self._runtime_home_capture: dict[str, object] | None = None
        self._last_passive_observation_error: str | None = None
        self._last_error: str | None = None
        self._first_control_fault: str | None = None
        self._held = False
        self._episode_ended = False
        self._last_hold_outcome: Any | None = None
        self._next_decision_at: float | None = None
        self._overrun_dropped_cycles = 0
        self._last_schedule_overrun: dict[str, object] | None = None
        self._last_cycle_timing: dict[str, float] | None = None
        self._blend_failure_fallback = config.blend_failure_fallback
        self._blend_failure_fallback_latched_source: str | None = None
        self._last_blend_failure_fallback: dict[str, object] | None = None
        self._human_timeout_fallback = config.human_timeout_fallback
        self._human_timeout_fallback_latched = False
        self._last_human_timeout_fallback: dict[str, object] | None = None
        self._vive_source_swap_count = 0
        self._startup_failure: str | None = None
        self._restart_required = False
        self._closed = False
        self._viewer_exit_requested = False

    @property
    def human_input_attached(self) -> bool:
        return bool(getattr(self.human_source, "attached", True))

    @property
    def viewer_exit_requested(self) -> bool:
        with self._lock:
            return bool(self._viewer_exit_requested)

    def prepare_human_attachment(
        self,
        *,
        coordinate_profile_path: Path,
        source_role: str,
    ) -> _SimHILHumanAttachment:
        """Build a passive human source without mutating the live runtime."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot attach human input to a closed runtime.")
            if self.human_input_attached:
                raise RuntimeError("Human input is already attached.")
            factory = self._human_attachment_factory
            if factory is None:
                raise RuntimeError("Human input attachment is not configured.")
        return factory(
            Path(coordinate_profile_path).expanduser(),
            str(source_role),
        )

    def activate_human_attachment(
        self,
        attachment: _SimHILHumanAttachment,
    ) -> dict[str, object]:
        """Commit a prepared passive source while the execution gate is held."""

        if not isinstance(attachment, _SimHILHumanAttachment):
            raise TypeError("attachment must be a _SimHILHumanAttachment.")
        with self._lock:
            if not self._started or not self._held:
                raise RuntimeError(
                    "Human input attachment requires a started held runtime."
                )
            receipt = self.human_source.attach(attachment.source)
            self.gripper_input_selection = (
                attachment.gripper_input_selection
            )
            self.calibration_binding = dict(
                attachment.calibration_binding
            )
            self._runtime_home_pose_desk_by_side_capture = (
                attachment.runtime_home_pose_desk_by_side_capture
            )
            self.readiness_provider.calibration_ready = True
            self.readiness_provider.home_ready = False
            return {
                **dict(receipt),
                "calibration_binding": dict(self.calibration_binding),
            }

    @property
    def started(self) -> bool:
        return self._started

    @property
    def restart_required(self) -> bool:
        return self._restart_required

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _raise_if_cancelled(
        cancel_requested: Callable[[], bool] | None,
        *,
        phase: str,
    ) -> None:
        if cancel_requested is not None and bool(cancel_requested()):
            raise InterruptedError(
                f"Sim-HIL startup cancelled during {phase}."
            )

    def _policy_contributor_active(self) -> bool:
        """Return whether the current composition needs policy values.

        A human-only acceptance stage must not synchronously run a potentially
        cold PPO inference just to discover that its result is unused.  The
        source is still started during application startup so a later stage
        can enable it without rebuilding the application.
        """

        values = getattr(self.composition, "human_alpha_by_component", None)
        if values is None:
            # Keep lightweight/fake application bindings conservative: older
            # callers without the typed alpha view retain the old behavior.
            return True
        return any(
            float(value) < 1.0 - 1e-12
            for value in dict(values).values()
        )

    def _warmup_sources(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._raise_if_cancelled(
            cancel_requested,
            phase="source_start",
        )
        self.policy_source.start()
        self.human_source.start()
        # The application now owns a live lower source even if the following
        # calibration transition raises.
        self._sources_started = True
        self.human_source.set_calibration_ready()
        self._wait_for_source_readiness(
            cancel_requested=cancel_requested,
            phase="source_warmup",
        )

    def _wait_for_source_readiness(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        phase: str,
    ) -> None:
        """Acquire genuinely current contributor emissions and verify them."""

        deadline = self.clock() + self.config.warmup_timeout_sec
        while True:
            self._raise_if_cancelled(
                cancel_requested,
                phase=phase,
            )
            now = _finite(self.clock(), name="warmup.now")
            if self._policy_contributor_active():
                self.policy_source.step(now=now)
            self.human_source.step(now=now)
            # Source emissions timestamp themselves with the local monotonic
            # clock during ``step``.  Observe after both calls so a freshly
            # produced emission is not compared against the pre-step cut and
            # reported as ``emission_in_future``.
            observed_at = max(
                now,
                _finite(self.clock(), name="warmup.observed_at"),
            )
            evidence = self.readiness_provider.snapshot(now=observed_at)
            readiness_failures = self.composition.readiness_failures(
                evidence,
                target_effective_at=observed_at,
            )
            if not readiness_failures:
                return
            if observed_at >= deadline:
                raise TimeoutError(
                    f"Sim-HIL source readiness timed out during {phase}: "
                    + ";".join(
                        f"{name}={evidence.detail_by_check.get(name, '')}"
                        for name in readiness_failures
                    )
                )
            self.sleep(0.005)

    def _prime_activation_inputs(self) -> None:
        """Admit one fresh contributor pair after an activation barrier."""

        policy_active = self._policy_contributor_active()
        requested_at = _finite(self.clock(), name="activation.requested_at")
        policy_emission = None
        if policy_active:
            self.policy_source.reset_for_activation()
            policy_emission = self.policy_source.step(now=requested_at)
        human_step = self.human_source.step(now=requested_at)
        observed_at = max(
            requested_at,
            _finite(self.runtime.clock(), name="activation.observed_at"),
            _finite(self.clock(), name="activation.station_observed_at"),
        )
        alpha = dict(self.composition.human_alpha_by_component)
        human_active = any(value > 1e-12 for value in alpha.values())
        policy_active = any(value < 1.0 - 1e-12 for value in alpha.values())
        if policy_emission is None and policy_active:
            raise RuntimeError(
                "Policy source produced no fresh activation emission."
            )
        if policy_emission is not None:
            policy_submission = self.composition.submit_policy_emission(
                policy_emission,
                observed_at=observed_at,
            )
            if not policy_submission.accepted and policy_active:
                raise RuntimeError(
                    "Fresh policy activation emission was rejected: "
                    f"{policy_submission.error}"
                )
        human_emission = getattr(human_step, "emission", None)
        if human_emission is None and human_active:
            # A network packet may arrive between the viewer-start boundary
            # and this activation cut.  Wait for a genuinely new emission up
            # to the configured Sim-HIL window; never promote last_emission
            # merely because it is within the age budget.
            deadline = observed_at + self.config.human_emission_timeout_sec
            while human_emission is None:
                if _finite(self.clock(), name="activation.wait_now") >= deadline:
                    break
                self.sleep(0.005)
                poll_at = _finite(self.clock(), name="activation.poll_at")
                human_step = self.human_source.step(now=poll_at)
                human_emission = getattr(human_step, "emission", None)
        if human_emission is None and human_active:
            raise RuntimeError(
                "Human source produced no fresh activation emission within "
                f"{self.config.human_emission_timeout_sec:.3f}s."
            )
        if human_emission is not None:
            human_submission = self.composition.submit_human_emission(
                human_emission,
                observed_at=observed_at,
            )
            if (
                human_active
                and (
                    human_submission is None
                    or not human_submission.accepted
                )
            ):
                raise RuntimeError(
                    "Fresh human activation emission was rejected: "
                    f"{None if human_submission is None else human_submission.error}"
                )

    def set_human_blend_alpha(
        self,
        alpha: float | Mapping[str, float],
        *,
        effective_from: float | None = None,
    ) -> Any:
        """Switch or proportionally blend human input without changing owners."""

        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoSimHILApplication.start() is required.")
            receipt = self.composition.set_human_blend_alpha(
                alpha,
                effective_from=effective_from,
            )
            if not receipt.accepted:
                raise RuntimeError(
                    "Sim-HIL blend update was rejected: "
                    f"{receipt.rejection_code}:{receipt.message}"
                )
            self._blend_failure_fallback_latched_source = None
            if isinstance(alpha, Mapping):
                human_enabled = any(
                    float(value) > 1e-12 for value in alpha.values()
                )
            else:
                human_enabled = float(alpha) > 1e-12
            if human_enabled:
                self._human_timeout_fallback_latched = False
            return receipt

    def set_blend_failure_fallback(self, source: str) -> dict[str, object]:
        """Select the Station response to an infeasible blended target.

        This changes no target, safety, IK, or dispatch state.  A human/policy
        choice is applied only after a typed resolution failure and then uses
        the existing formal composition-parameter admission path.
        """

        normalized = str(source).strip().lower()
        if normalized not in SIM_HIL_BLEND_FAILURE_FALLBACKS:
            raise ValueError(
                "blend failure fallback must be hold, human, or policy."
            )
        with self._lock:
            self._blend_failure_fallback = normalized
            return {
                "status": "selected",
                "source": normalized,
                "latched_source": self._blend_failure_fallback_latched_source,
            }

    def set_human_timeout_fallback(self, source: str) -> dict[str, object]:
        """Select what to do after the Sim-HIL human input window expires.

        ``hold`` leaves the lower composition gate responsible for stopping
        publication.  ``policy`` is an explicit, latched source handoff; it
        never republishes the last human pose and never switches back
        automatically when VIVE packets resume.
        """

        normalized = str(source).strip().lower()
        if normalized not in SIM_HIL_HUMAN_TIMEOUT_FALLBACKS:
            raise ValueError(
                "human timeout fallback must be hold or policy."
            )
        with self._lock:
            self._human_timeout_fallback = normalized
            return {
                "status": "selected",
                "source": normalized,
                "latched": self._human_timeout_fallback_latched,
            }

    @staticmethod
    def _resolution_failure_status(failure: Any | None) -> dict[str, object] | None:
        if failure is None:
            return None
        return {
            "reason_code": getattr(failure, "reason_code", None),
            "resolver_status": getattr(failure, "resolver_status", None),
            "message": getattr(failure, "message", None),
            "diagnostics": dict(getattr(failure, "diagnostics", {}) or {}),
        }

    def _apply_blend_failure_fallback(
        self,
        cycle: Any,
        *,
        observed_at: float,
    ) -> None:
        execution = getattr(cycle, "execution", None)
        if getattr(execution, "status", None) != (
            "compound_resolution_failure_hold"
        ):
            return
        failure = getattr(execution, "resolution_failure", None)
        if failure is None:
            return
        alpha_by_component = dict(
            getattr(self.composition, "human_alpha_by_component", {}) or {}
        )
        if not alpha_by_component or not all(
            1e-12 < float(value) < 1.0 - 1e-12
            for value in alpha_by_component.values()
        ):
            return
        selected = self._blend_failure_fallback
        event: dict[str, object] = {
            "status": "hold_selected" if selected == "hold" else "latched",
            "selected_source": selected,
            "triggered_at": observed_at,
            "cycle_index": getattr(cycle, "cycle_index", None),
            "alpha_before": alpha_by_component,
            "failure": self._resolution_failure_status(failure),
            "parameter_receipt_id": None,
        }
        if selected != "hold":
            target_alpha = 1.0 if selected == "human" else 0.0
            receipt = self.composition.set_human_blend_alpha(
                target_alpha,
                effective_from=max(
                    observed_at,
                    _finite(
                        self.runtime.clock(),
                        name="blend_fallback.effective_from",
                    ),
                ),
            )
            if not receipt.accepted:
                raise RuntimeError(
                    "Sim-HIL blend failure fallback was rejected: "
                    f"{receipt.rejection_code}:{receipt.message}"
                )
            self._blend_failure_fallback_latched_source = selected
            event["alpha_after"] = {
                component_id: target_alpha
                for component_id in alpha_by_component
            }
            event["parameter_receipt_id"] = getattr(
                receipt,
                "parameter_snapshot_id",
                getattr(receipt, "semantic_content_hash", None),
            )
        self._last_blend_failure_fallback = event

    def _apply_human_timeout_fallback(
        self,
        *,
        observed_at: float,
        policy_emission: Any | None,
    ) -> Any | None:
        """Apply the configured network-gap fallback without stale-pose reuse."""

        if self._human_timeout_fallback_latched:
            return policy_emission
        alpha_by_component = dict(
            getattr(self.composition, "human_alpha_by_component", {}) or {}
        )
        if not alpha_by_component or not any(
            float(value) > 1e-12 for value in alpha_by_component.values()
        ):
            return policy_emission
        snapshot = getattr(self.readiness_provider, "snapshot", None)
        if not callable(snapshot):
            # Lightweight test/compatibility bindings may not expose the
            # optional readiness observer; retain their previous behavior.
            return policy_emission
        readiness = snapshot(now=observed_at)
        if bool(getattr(readiness, "human_device_ready", True)):
            return policy_emission
        event: dict[str, object] = {
            "status": "hold_selected",
            "selected_source": self._human_timeout_fallback,
            "triggered_at": observed_at,
            "human_detail": getattr(
                readiness,
                "detail_by_check",
                {},
            ).get("human_device_ready"),
            "parameter_receipt_id": None,
        }
        if self._human_timeout_fallback == "policy":
            if policy_emission is None:
                try:
                    policy_emission = self.policy_source.step(now=observed_at)
                except Exception as error:  # noqa: BLE001 - lower gate still holds.
                    event["status"] = "policy_fallback_failed"
                    event["error"] = f"{type(error).__name__}:{error}"
                    self._last_human_timeout_fallback = event
                    return None
            if policy_emission is None:
                event["status"] = "policy_fallback_unavailable"
                self._last_human_timeout_fallback = event
                return None
            receipt = self.composition.set_human_blend_alpha(
                0.0,
                effective_from=observed_at,
            )
            if not receipt.accepted:
                event["status"] = "policy_fallback_rejected"
                event["error"] = (
                    f"{receipt.rejection_code}:{receipt.message}"
                )
                self._last_human_timeout_fallback = event
                return policy_emission
            event["status"] = "policy_latched"
            event["parameter_receipt_id"] = getattr(
                receipt,
                "parameter_snapshot_id",
                getattr(receipt, "semantic_content_hash", None),
            )
            self._human_timeout_fallback_latched = True
        self._last_human_timeout_fallback = event
        return policy_emission

    def swap_vive_left_right_sources(self) -> dict[str, object]:
        """Swap upstream VIVE wrist roles before composition starts."""

        with self._lock:
            if self._started:
                raise RuntimeError(
                    "VIVE source roles cannot be swapped after Sim-HIL "
                    "composition has started."
                )
            if self._human_input_check_active:
                raise RuntimeError(
                    "VIVE source roles cannot be swapped while passive input "
                    "acceptance is running."
                )
            session = getattr(self.human_source, "session", None)
            if session is None:
                raise RuntimeError(
                    "Passive human source does not expose its teleop session."
                )
            from teleop_runtime.sources.pose.vive import VivePoseSource

            sources = [
                source
                for source in tuple(
                    getattr(session, "sources", ()) or ()
                )
                if isinstance(source, VivePoseSource)
            ]
            if len(sources) != 1:
                raise RuntimeError(
                    "Expected exactly one VivePoseSource in the passive "
                    f"human session; found {len(sources)}."
                )
            source = sources[0]
            receipt = dict(source.swap_left_right_source_roles())
            self._vive_source_swap_count += 1
            self._last_human_input_check_step = None
            self._human_input_check_report = None
            self._last_error = None
            receipt.update(
                {
                    "swapped_at": _finite(
                        self.clock(),
                        name="vive_source_swap.swapped_at",
                    ),
                    "swap_count": self._vive_source_swap_count,
                }
            )
            return receipt

    def run_human_input_acceptance_check(
        self,
        *,
        timeout_sec: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        """Start only the passive human source and wait for one valid emission.

        This is an input-device acceptance step, not target authorization or
        simulator dispatch. It may run before startup or while a started
        runtime is explicitly held for deferred human attachment.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Human input acceptance requires a newly built Sim-HIL "
                    "application; this instance is closed."
                )
            if self._started and not self._held:
                raise RuntimeError(
                    "Hold a started Sim-HIL runtime before checking human input."
                )
            if not self.human_input_attached:
                raise RuntimeError(
                    "Human input acceptance requires an attached human source."
                )
            if self._human_input_check_active:
                raise RuntimeError(
                    "Human input acceptance is already running."
                )
            timeout = (
                self.config.warmup_timeout_sec
                if timeout_sec is None
                else _finite(timeout_sec, name="input_check.timeout_sec")
            )
            if timeout <= 0.0:
                raise ValueError("input acceptance timeout must be positive.")
            self._human_input_check_active = True

        try:
            self.human_source.start()
            # Preserve cleanup ownership once lower resources are live.
            with self._lock:
                self._sources_started = True
            self.human_source.set_calibration_ready()
            deadline = self.clock() + timeout
            while True:
                self._raise_if_cancelled(
                    cancel_requested,
                    phase="human_input_readiness",
                )
                now = _finite(self.clock(), name="input_check.now")
                step = self.human_source.step(now=now)
                with self._lock:
                    self._last_human_input_check_step = step
                emission = getattr(step, "emission", None)
                emission_valid = bool(
                    getattr(step, "emission_valid", False)
                )
                if bool(getattr(step, "command_dispatched", False)):
                    raise RuntimeError(
                        "Passive human input check attempted dispatch."
                    )
                if bool(getattr(step, "authoritative", False)):
                    raise RuntimeError(
                        "Passive human input check became authoritative."
                    )
                if emission_valid and emission is not None:
                    session_step = getattr(step, "session_step", None)
                    channels = dict(
                        getattr(session_step, "channels", {}) or {}
                    )
                    report = {
                        "status": "ready",
                        "checked_at": now,
                        "emission_id": getattr(
                            emission,
                            "emission_id",
                            None,
                        ),
                        "emission_hash": getattr(
                            emission,
                            "semantic_content_hash",
                            None,
                        ),
                        "channel_names": sorted(channels),
                        "resolved_gripper_input": getattr(
                            self.gripper_input_selection,
                            "resolved_source",
                            None,
                        ),
                        "q6_port": getattr(
                            self.gripper_input_selection,
                            "q6_port",
                            None,
                        ),
                        "fallback_reason": getattr(
                            self.gripper_input_selection,
                            "fallback_reason",
                            None,
                        ),
                        "rig_absent": bool(
                            getattr(
                                getattr(
                                    self.human_source,
                                    "binding_receipt",
                                    None,
                                ),
                                "rig_absent",
                                False,
                            )
                        ),
                        "command_dispatched": False,
                        "authoritative": False,
                    }
                    with self._lock:
                        self._human_input_check_report = report
                        self._last_error = None
                    return dict(report)
                if now >= deadline:
                    rejection_reason = getattr(
                        step,
                        "rejection_reason",
                        None,
                    )
                    session_step = getattr(step, "session_step", None)
                    channels = dict(
                        getattr(session_step, "channels", {}) or {}
                    )
                    expected_channels: set[str] = set()
                    session = getattr(self.human_source, "session", None)
                    for source in tuple(
                        getattr(session, "sources", ()) or ()
                    ):
                        expected = getattr(
                            source,
                            "expected_channel_names",
                            None,
                        )
                        if callable(expected):
                            expected_channels.update(expected())
                    missing_channels = sorted(
                        expected_channels - set(channels)
                    )
                    invalid_channels = sorted(
                        name
                        for name, channel in channels.items()
                        if not bool(getattr(channel, "valid", False))
                    )
                    fault_codes = sorted(
                        {
                            str(getattr(fault, "code", "unknown"))
                            for fault in tuple(
                                getattr(session_step, "faults", ()) or ()
                            )
                        }
                    )
                    vive_status = _vive_receiver_status(session)
                    vive_trackers = dict(
                        vive_status.get("trackers") or {}
                    )
                    vive_role_facts = {
                        side: {
                            "status": tracker.get("status_level"),
                            "valid": tracker.get("valid"),
                            "age_sec": tracker.get("age_sec"),
                            "seq": tracker.get("seq"),
                            "error": tracker.get("sender_error"),
                            "selected_outputs": tracker.get(
                                "selected_for_output_channels"
                            ),
                        }
                        for side, tracker in vive_trackers.items()
                    }
                    raise TimeoutError(
                        "Passive human input acceptance timed out: "
                        f"rejection={rejection_reason or 'none'}; "
                        f"missing_channels={missing_channels}; "
                        f"invalid_channels={invalid_channels}; "
                        f"available_channels={sorted(channels)}; "
                        f"fault_codes={fault_codes}; "
                        f"vive_receiver={vive_status.get('status')}; "
                        f"vive_packet_count={vive_status.get('packet_count')}; "
                        f"vive_packet_age_sec={vive_status.get('packet_age_sec')}; "
                        f"vive_roles={vive_role_facts}"
                    )
                self.sleep(0.005)
        except Exception as error:
            with self._lock:
                self._last_error = f"{type(error).__name__}:{error}"
            raise
        finally:
            with self._lock:
                self._human_input_check_active = False

    @staticmethod
    def _channel_pose_in_desk(
        channel: Any,
        calibration: Any,
    ) -> Any:
        from teleop_runtime.transforms import as_pose_matrix

        pose = as_pose_matrix(channel.value)
        frame_name = str(getattr(channel, "frame_name", "") or "")
        metadata = dict(getattr(channel, "metadata", {}) or {})
        tracker_role = metadata.get("tracker_role")
        if frame_name in {"vive_world", "world"}:
            return calibration.world_to_desk(
                pose,
                tracker_role=(
                    None if tracker_role is None else str(tracker_role)
                ),
            )
        if frame_name.endswith("_operator") or frame_name == "operator":
            return calibration.operator_to_desk(pose)
        if frame_name in {calibration.desk_frame.name, "desk"}:
            return pose
        raise ValueError(
            f"Unsupported runtime-Home tracker frame {frame_name or '<none>'!r}."
        )

    def capture_runtime_home(
        self,
        *,
        samples_required: int = 5,
        timeout_sec: float = 1.0,
    ) -> dict[str, object]:
        """Re-anchor live human pose to current MuJoCo pose without dispatch."""

        sample_count = int(samples_required)
        timeout = _finite(timeout_sec, name="runtime_home.timeout_sec")
        if sample_count <= 0:
            raise ValueError("samples_required must be positive.")
        if timeout <= 0.0:
            raise ValueError("runtime Home timeout must be positive.")
        with self._lock:
            if self._closed:
                raise RuntimeError("Runtime Home requires an open application.")
            if not self._sources_started:
                raise RuntimeError(
                    "Runtime Home requires passive human input acceptance first."
                )
            if self._episode_ended:
                raise RuntimeError(
                    "Runtime Home cannot mutate an ended Sim-HIL episode."
                )
            if self._started and not self._held:
                raise RuntimeError(
                    "Hold Sim-HIL before recapturing runtime Home."
                )
            capture_robot_home = (
                self._runtime_home_pose_desk_by_side_capture
            )
            if capture_robot_home is None:
                raise RuntimeError(
                    "Runtime Home robot-pose capture is not bound."
                )
            session = getattr(self.human_source, "session", None)
            calibration = getattr(session, "calibration", None)
            retargeters = tuple(
                getattr(session, "retargeters", ()) or ()
            )
            if session is None or calibration is None:
                raise RuntimeError(
                    "Passive human source does not expose its calibration session."
                )

        pose_retargeters: dict[str, Any] = {}
        for retargeter in retargeters:
            config = getattr(retargeter, "config", None)
            side = str(getattr(config, "side", "") or "").strip()
            if (
                side
                and callable(getattr(retargeter, "set_home_anchors", None))
                and str(
                    getattr(config, "input_channel_name", "") or ""
                ).strip()
            ):
                if side in pose_retargeters:
                    raise RuntimeError(
                        f"Multiple pose retargeters expose side {side!r}."
                    )
                pose_retargeters[side] = retargeter
        active_sides = tuple(sorted(pose_retargeters))
        if not active_sides:
            raise RuntimeError(
                "Runtime Home has no pose retargeter."
            )

        samples: dict[str, list[Any]] = {
            side: [] for side in active_sides
        }
        deadline = self.clock() + timeout
        last_step = None
        while any(len(values) < sample_count for values in samples.values()):
            now = _finite(self.clock(), name="runtime_home.now")
            if now >= deadline:
                counts = {
                    side: len(values) for side, values in samples.items()
                }
                raise TimeoutError(
                    "Timed out waiting for runtime-Home VIVE samples: "
                    f"{counts}"
                )
            step = self.human_source.step(now=now)
            if bool(getattr(step, "command_dispatched", False)) or bool(
                getattr(step, "authoritative", False)
            ):
                raise RuntimeError(
                    "Runtime Home passive input unexpectedly gained dispatch "
                    "or authority."
                )
            last_step = step
            channels = dict(
                getattr(getattr(step, "session_step", None), "channels", {})
                or {}
            )
            for side in active_sides:
                if len(samples[side]) >= sample_count:
                    continue
                retargeter = pose_retargeters[side]
                channel_name = str(
                    retargeter.config.input_channel_name
                )
                channel = channels.get(channel_name)
                if channel is None or not bool(
                    getattr(channel, "valid", False)
                ):
                    continue
                pose_desk = self._channel_pose_in_desk(
                    channel,
                    calibration,
                )
                # ``set_home_anchors`` owns the tracker-local translation and
                # rotation adjustment.  Keep capture samples in the raw desk
                # frame so that adjustment is applied exactly once.
                samples[side].append(pose_desk)
            self.sleep(0.005)

        from teleop_runtime.calibration_flow import average_pose

        human_home_by_side = {
            side: average_pose(samples[side]) for side in active_sides
        }
        # Capture the robot pose after the sample interval so a preceding HOLD
        # has time to reach the physics owner. The lower rig serializes this
        # read against mj_step.
        robot_home_by_side = {
            str(side): pose
            for side, pose in dict(capture_robot_home()).items()
        }
        missing_robot_sides = set(active_sides) - set(robot_home_by_side)
        if missing_robot_sides:
            raise RuntimeError(
                "Runtime Home robot-pose capture is missing sides: "
                f"{sorted(missing_robot_sides)}"
            )
        for side in active_sides:
            pose_retargeters[side].set_home_anchors(
                human_pose_desk=human_home_by_side[side],
                robot_pose_desk=robot_home_by_side[side],
            )
        session.mark_home_recalibrated(
            reason="sim_hil_runtime_home_captured"
        )
        captured_at = _finite(
            self.clock(),
            name="runtime_home.captured_at",
        )
        receipt: dict[str, object] = {
            "capture_id": f"sim-hil-runtime-home:{captured_at:.9f}",
            "captured_at": captured_at,
            "clock_domain_id": getattr(
                getattr(self.runtime, "config", None),
                "clock_domain_id",
                None,
            ),
            "sample_count_by_side": {
                side: len(samples[side]) for side in active_sides
            },
            "human_home_pose_desk_by_side": {
                side: (
                    human_home_by_side[side].tolist()
                    if callable(
                        getattr(human_home_by_side[side], "tolist", None)
                    )
                    else human_home_by_side[side]
                )
                for side in active_sides
            },
            "robot_home_pose_desk_by_side": {
                side: (
                    robot_home_by_side[side].tolist()
                    if callable(
                        getattr(robot_home_by_side[side], "tolist", None)
                    )
                    else robot_home_by_side[side]
                )
                for side in active_sides
            },
            "command_dispatched": False,
            "authoritative": False,
        }
        with self._lock:
            self._last_human_input_check_step = last_step
            self._runtime_home_capture = receipt
            self.readiness_provider.home_ready = True
            self._last_passive_observation_error = None
        return dict(receipt)

    def poll_passive_human_input(
        self,
        *,
        now: float | None = None,
    ) -> Any:
        """Refresh human observation while HOLD without composing or dispatching."""

        with self._lock:
            if (
                self._closed
                or not self._sources_started
                or not self._started
                or not self._held
                or self._episode_ended
            ):
                return None
            timestamp = _finite(
                self.clock() if now is None else now,
                name="passive_observation.now",
            )
            try:
                step = self.human_source.step(now=timestamp)
                if bool(getattr(step, "command_dispatched", False)) or bool(
                    getattr(step, "authoritative", False)
                ):
                    raise RuntimeError(
                        "Held passive observation unexpectedly gained dispatch "
                        "or authority."
                    )
                self._last_human_input_check_step = step
                self._last_passive_observation_error = None
                return step
            except Exception as error:
                self._last_passive_observation_error = (
                    f"{type(error).__name__}:{error}"
                )
                raise

    def start(
        self,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Any:
        with self._lock:
            if self._started:
                return self._last_preflight
            if self._restart_required:
                raise RuntimeError(
                    "The Sim-HIL application was closed after a failed startup "
                    "and must be rebuilt. First startup failure: "
                    f"{self._startup_failure or 'unknown'}"
                )
            if self._closed:
                raise RuntimeError(
                    "The Sim-HIL application is closed and must be rebuilt."
                )
            if self._human_input_check_active:
                raise RuntimeError(
                    "Sim-HIL cannot start while passive human input "
                    "acceptance is running."
                )
            try:
                self._raise_if_cancelled(
                    cancel_requested,
                    phase="runtime_start",
                )
                self._warmup_sources(
                    cancel_requested=cancel_requested,
                )
                self._raise_if_cancelled(
                    cancel_requested,
                    phase="viewer_start",
                )
                # Open the native viewer before starting the realtime physics
                # worker.  ``launch_passive`` and the worker both touch the
                # same MjModel/MjData; starting them concurrently has caused
                # native crashes on some GLFW/GLX builds.  Input readiness was
                # already checked by the passive human acceptance step, so
                # this still cannot open a MuJoCo window for stale VIVE input.
                if self.viewer is not None:
                    self.viewer.start()
                    # Native window construction may exceed the contributor
                    # TTL even though the receiver remains healthy.  Re-poll
                    # the real sources after that boundary instead of passing
                    # the pre-viewer emission to composition preflight.
                    self._wait_for_source_readiness(
                        cancel_requested=cancel_requested,
                        phase="post_viewer_source_refresh",
                    )
                report = self.composition.start()
                # The warm-up emissions predate the composition activation barrier.
                # Acquire and admit one fresh pair after that barrier; no
                # second queue or authority is introduced.
                self._prime_activation_inputs()
                self._last_preflight = report
                self._started = True
                self._held = False
                self._episode_ended = False
                self._next_decision_at = _finite(
                    self.runtime.clock(),
                    name="start.next_decision_at",
                )
                self._overrun_dropped_cycles = 0
                self._last_schedule_overrun = None
                self._last_error = None
                return report
            except Exception as error:
                primary_error = f"{type(error).__name__}:{error}"
                self._startup_failure = primary_error
                self._restart_required = True
                cleanup_error: str | None = None
                try:
                    self.close()
                except Exception as close_error:  # noqa: BLE001 - preserve primary cause.
                    cleanup_error = (
                        f"{type(close_error).__name__}:{close_error}"
                    )
                self._last_error = (
                    primary_error
                    if cleanup_error is None
                    else f"{primary_error}; cleanup_error={cleanup_error}"
                )
                raise

    def _skip_missed_compose_cuts(
        self,
        *,
        decision_at: float,
        observed_at: float,
        period: float,
        phase: str,
    ) -> float:
        aligned_decision_at, skipped = _advance_control_deadline(
            previous_deadline=decision_at - period,
            completed_at=observed_at,
            period=period,
        )
        self._overrun_dropped_cycles += skipped
        self._last_schedule_overrun = {
            "observed_at": observed_at,
            "late_by_sec": observed_at - decision_at,
            "skipped_cycle_count": skipped,
            "previous_decision_at": decision_at,
            "resumed_decision_at": aligned_decision_at,
            "resolution": "skip_to_next_compose_grid",
            "phase": _required(phase, name="overrun.phase"),
        }
        return aligned_decision_at

    def step(self, *, now: float | None = None) -> Any:
        with self._lock:
            cycle_started_at = time.perf_counter()
            if not self._started:
                raise RuntimeError("MujocoSimHILApplication.start() is required.")
            if self._held:
                raise RuntimeError(
                    "MuJoCo Sim-HIL is held; explicit rearm is required."
                )
            timestamp = _finite(
                self.clock() if now is None else now,
                name="step.now",
            )
            decision_at = (
                timestamp
                if self._next_decision_at is None
                else self._next_decision_at
            )
            period = 1.0 / float(self.runtime.config.decision_hz)
            lateness = timestamp - decision_at
            maximum_lateness = max(0.02, period)
            if lateness > maximum_lateness:
                # Station composition is not the canonical execution watchdog.
                # Skip missed upstream compose cuts without burst catch-up;
                # the lower target gate remains authoritative for whether the
                # newly scheduled target can still be admitted and executed.
                decision_at = self._skip_missed_compose_cuts(
                    decision_at=decision_at,
                    observed_at=timestamp,
                    period=period,
                    phase="step_entry",
                )
            try:
                policy_started_at = time.perf_counter()
                policy_emission = None
                if self._policy_contributor_active():
                    policy_emission = self.policy_source.step(now=timestamp)
                policy_elapsed_sec = time.perf_counter() - policy_started_at
                observed_at = max(
                    timestamp,
                    _finite(self.runtime.clock(), name="step.observed_at"),
                )
                policy_emission = self._apply_human_timeout_fallback(
                    observed_at=observed_at,
                    policy_emission=policy_emission,
                )
                # A policy fallback may have produced a synchronous emission
                # after the first observation cut; close the cut again before
                # admitting it so clock ordering cannot reject a valid fresh
                # fallback as ``emission_in_future``.
                observed_at = max(
                    observed_at,
                    _finite(
                        self.runtime.clock(),
                        name="step.fallback_observed_at",
                    ),
                    _finite(
                        self.clock(),
                        name="step.fallback_station_observed_at",
                    ),
                )
                if observed_at - decision_at > maximum_lateness:
                    # Synchronous policy work can consume the original target
                    # lead after entry-time alignment. Move the upstream cut
                    # forward before publication; the target still owns final
                    # late admission and HOLD.
                    decision_at = self._skip_missed_compose_cuts(
                        decision_at=decision_at,
                        observed_at=observed_at,
                        period=period,
                        phase="post_policy_source",
                    )
                composition_started_at = time.perf_counter()
                self._last_cycle = self.composition.step(
                    now=observed_at,
                    policy_emission=policy_emission,
                    poll_human=True,
                    target_effective_at=(
                        observed_at
                        if getattr(
                            self.runtime.reference_activation_spec,
                            "activation_mode",
                            None,
                        )
                        == "immediate_target"
                        else (
                            decision_at
                            + self.runtime.reference_activation_spec
                            .target_lead_time_sec
                        )
                    ),
                )
                self._apply_blend_failure_fallback(
                    self._last_cycle,
                    observed_at=observed_at,
                )
                composition_elapsed_sec = (
                    time.perf_counter() - composition_started_at
                )
                nominal_next = decision_at + period
                lower_due = getattr(
                    self.runtime,
                    "next_decision_due_at",
                    None,
                )
                if (
                    getattr(
                        getattr(self._last_cycle, "execution", None),
                        "status",
                        None,
                    )
                    == "not_due"
                    and lower_due is not None
                    and lower_due > observed_at
                ):
                    # A small phase offset is not a missed compose cycle. Retry
                    # at the lower scheduler's exact cut instead of discarding
                    # a complete input period.
                    self._next_decision_at = lower_due
                else:
                    # Never let a stale lower deadline move Station backwards.
                    self._next_decision_at = (
                        nominal_next
                        if lower_due is None
                        else max(nominal_next, lower_due)
                    )
                gate = self.runtime.capture_execution_gate_snapshot(
                    now=max(
                        observed_at,
                        _finite(
                            self.runtime.clock(),
                            name="step.gate_observed_at",
                        ),
                    )
                )
                if gate.status != "ready":
                    self._held = True
                    gate_reason = str(
                        getattr(gate, "reason", "") or "unspecified"
                    )
                    self._last_error = (
                        "ExecutionGateNotReady:"
                        f"{gate.status}:{gate_reason}"
                    )
                    if self._first_control_fault is None:
                        self._first_control_fault = self._last_error
                if self.viewer is not None and not self.viewer.sync():
                    viewer_state = str(
                        dict(self.viewer.status()).get("state") or ""
                    )
                    if viewer_state in {"closed_by_user", "faulted"}:
                        # Closing the native window is an operator exit
                        # request, not merely a missing visualization frame.
                        # Establish lower HOLD now; the Station loop performs
                        # the one authoritative application teardown.
                        self._viewer_exit_requested = True
                        self.hold(reason=f"mujoco_viewer_{viewer_state}")
                self._last_cycle_timing = {
                    "total_sec": time.perf_counter() - cycle_started_at,
                    "policy_source_sec": policy_elapsed_sec,
                    "composition_and_execution_sec": composition_elapsed_sec,
                    "configured_period_sec": period,
                }
                if gate.status == "ready":
                    self._last_error = None
                return self._last_cycle
            except Exception as error:
                self._last_cycle_timing = {
                    "total_sec": time.perf_counter() - cycle_started_at,
                    "configured_period_sec": period,
                }
                self._last_error = f"{type(error).__name__}:{error}"
                if self._first_control_fault is None:
                    self._first_control_fault = self._last_error
                raise

    def hold(self, *, reason: str) -> Any | None:
        """Pause publication through the lower execution gate."""

        with self._lock:
            # A startup failure can leave the lower composition active before
            # this application publishes ``started=True``.  That partial state
            # still requires a lower HOLD before cleanup.
            if not self._started and not bool(
                getattr(self.composition, "started", False)
            ):
                return None
            try:
                outcome = self.composition.request_hold(
                    reason=(
                        "sim_hil_application:"
                        f"{_required(reason, name='reason')}"
                    ),
                )
            except Exception as error:
                # The lower transition failed or is uncertain.  Keep the
                # application locally latched so no later step can infer READY.
                self._held = True
                self._last_error = f"{type(error).__name__}:{error}"
                raise
            self._last_hold_outcome = outcome
            self._held = True
            return outcome

    def end_episode(self, *, reason: str = "episode_end") -> Any | None:
        """Establish terminal HOLD while passive inputs remain connected.

        Reset remains owned by the simulator task.  A subsequent episode must
        close and rebuild this target instance instead of mutating the scene
        behind a live execution rig.
        """

        with self._lock:
            if not self._started:
                return None
            self._episode_ended = True
            return self.hold(reason=f"episode_end:{reason}")

    def rearm(
        self,
        *,
        now: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Any:
        """Refresh both contributors around a new composition activation."""

        with self._lock:
            if self._episode_ended:
                raise RuntimeError(
                    "An ended Sim-HIL episode requires target rebuild, not rearm."
                )
            if not self._started or not self._held:
                raise RuntimeError(
                    "MuJoCo Sim-HIL rearm requires a started held application."
                )
            timestamp = _finite(
                self.clock() if now is None else now,
                name="rearm.now",
            )
            # Establish the authoritative lower HOLD before resetting or
            # waiting on contributors. Application-local ``_held`` is only a
            # latch and cannot protect the target while a stale lower READY
            # record is still current.
            self.composition.request_hold(
                reason="sim_hil_application:rearm_boundary"
            )
            hold_observed_at = _finite(
                self.runtime.clock(),
                name="rearm.hold_observed_at",
            )
            if self._policy_contributor_active():
                self.policy_source.reset_for_activation()
            # Human/device sources are asynchronous.  One poll may legally
            # return no new emission even while the receiver is healthy, so
            # reuse the bounded startup freshness loop before rearm preflight.
            self._wait_for_source_readiness(
                cancel_requested=cancel_requested,
                phase="rearm_source_refresh",
            )
            # Source production may finish after the requested rearm cut.
            # Validate readiness against an observation cut that can causally
            # include those fresh emissions.
            rearm_at = max(
                timestamp,
                hold_observed_at,
                _finite(self.runtime.clock(), name="rearm.observed_at"),
            )
            report = self.composition.rearm(now=rearm_at)
            self._prime_activation_inputs()
            self._next_decision_at = _finite(
                self.runtime.clock(),
                name="rearm.next_decision_at",
            )
            self._last_schedule_overrun = None
            self._held = False
            self._last_error = None
            return report

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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors: list[BaseException] = []
            if self.composition.started:
                try:
                    self.hold(reason="shutdown")
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
                try:
                    self.composition.close()
                except BaseException as error:  # noqa: BLE001 - continue lower cleanup.
                    errors.append(error)
            elif self._sources_started:
                try:
                    self.human_source.close()
                except BaseException as error:  # noqa: BLE001 - continue lower cleanup.
                    errors.append(error)
            try:
                self.policy_source.close()
            except BaseException as error:  # noqa: BLE001 - continue lower cleanup.
                errors.append(error)
            # The composition/runtime close above joins the physics worker.
            # Only then may the native viewer tear down the shared MjData.
            if self.viewer is not None:
                try:
                    self.viewer.close()
                except BaseException as error:  # noqa: BLE001 - cleanup continues.
                    errors.append(error)
            self._started = False
            self._sources_started = False
            self._held = False
            if errors:
                self._last_error = ";".join(
                    f"{type(error).__name__}:{error}" for error in errors
                )
                raise RuntimeError(
                    "Sim-HIL close encountered lower-runtime errors: "
                    + self._last_error
                ) from errors[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m control_station sim-hil",
        description=(
            "Default-off MuJoCo-only Sim-HIL using temporary PPO + passive "
            "VIVE with Q6-first gripper input."
        ),
    )
    parser.add_argument("--enable", action="store_true", help="Explicitly enable source startup and MuJoCo execution.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate PPO/task/calibration bindings and simulator task "
            "initialization without starting devices, authority, or workers."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        default=DEFAULT_CALIBRATION_PROFILE,
        help="Compatibility profile supplying the stored human Home only.",
    )
    parser.add_argument(
        "--vive-coordinate-profile",
        type=Path,
        default=DEFAULT_VIVE_COORDINATE_PROFILE,
        help=(
            "Independent Control Station VIVE coordinate profile. "
            "The default 'latest' selects the newest valid record from the "
            "independent Station profile directory. Target geometry and robot "
            "Home come from the loaded MuJoCo target."
        ),
    )
    parser.add_argument(
        "--vive-calibration-gui",
        action="store_true",
        help=(
            "Compatibility flag. VIVE readiness and on-demand calibration "
            "are now always integrated into --control-gui; no separate "
            "startup calibration page is opened."
        ),
    )
    parser.add_argument(
        "--vive-calibration-ui-port",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option; integrated calibration uses "
            "the --control-gui port."
        ),
    )
    parser.add_argument(
        "--vive-calibration-open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated compatibility option; integrated calibration does "
            "not open a second browser page."
        ),
    )
    parser.add_argument("--vive-port", type=int, default=DEFAULT_VIVE_PORT)
    parser.add_argument("--vive-clock-sync-port", type=int, default=None)
    parser.add_argument(
        "--left-q6-port",
        default=None,
        help="Reserved for another topology profile; rejected by this right-only profile.",
    )
    parser.add_argument(
        "--right-q6-port",
        default=None,
        help=(
            "Optional preferred right Q6 endpoint. In the default auto mode, "
            "the latest assignment/discovery is used when omitted."
        ),
    )
    parser.add_argument(
        "--gripper-input-source",
        choices=("auto", "q6", "angle_trigger"),
        default="auto",
        help=(
            "Gripper input selection. auto prefers a reachable Q6 and falls "
            "back to the calibrated angle trigger."
        ),
    )
    parser.add_argument(
        "--q6-config",
        default="latest",
        help="Saved Q6 side assignment used before unassigned discovery.",
    )
    parser.add_argument(
        "--angle-trigger-config-root",
        type=Path,
        default=None,
        help="Optional calibrated angle-trigger configuration directory.",
    )
    parser.add_argument(
        "--angle-trigger-rate-hz",
        type=float,
        default=30.0,
        help="Angle-trigger polling rate used after explicit selection or fallback.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--duration", type=float, default=0.0, help="Run seconds; 0 means until interrupted.")
    parser.add_argument(
        "--position-control-profile",
        type=str,
        default=POSITION_15_60_120_PROFILE_NAME,
        help=(
            "Lower position-controller profile. The default keeps the policy "
            "source at 15 Hz, reference at 60 Hz, and commands at 120 Hz; "
            "the independent composition rate defaults to 30 Hz."
        ),
    )
    parser.add_argument(
        "--compose-hz",
        type=float,
        default=None,
        help=(
            "Final HIL compose/authority target rate (default: 30 Hz). "
            "Policy inference keeps its own cadence."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=(
            "Compatibility alias for --compose-hz and the Station loop rate; "
            "when omitted, composition uses its 30 Hz default."
        ),
    )
    parser.add_argument("--warmup-timeout", type=float, default=10.0)
    parser.add_argument(
        "--human-emission-timeout",
        type=float,
        default=5.0,
        help=(
            "Maximum Sim-HIL human emission age tolerated across a VIVE "
            "network gap (default: 5 seconds). This does not change real "
            "teleop tracker-loss HOLD thresholds."
        ),
    )
    parser.add_argument(
        "--human-timeout-fallback",
        choices=("hold", "policy"),
        default="hold",
        help=(
            "Sim-HIL fallback after the human emission window expires: "
            "hold (default) or policy. Policy fallback is latched until "
            "the operator selects a human mode again."
        ),
    )
    parser.add_argument(
        "--human-alpha",
        type=float,
        default=0.5,
        help=(
            "Initial human contribution in [0,1]: 0 is PPO-only, 1 is "
            "human-only, and intermediate values are proportional mixing. "
            "The interactive control GUI publishes its selected mode before "
            "the first control cycle."
        ),
    )
    parser.add_argument(
        "--execution-difference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request passive right-arm MuJoCo reference/measurement "
            "difference observations (default: enabled)."
        ),
    )
    parser.add_argument(
        "--mujoco-viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Open the lower-owned passive MuJoCo viewer. Defaults to enabled "
            "and is required for --control-gui; disabled by default for "
            "non-acceptance runs."
        ),
    )
    parser.add_argument(
        "--target-admission-margin-sec",
        type=float,
        default=0.05,
        help=(
            "Compatibility field reserved for scheduled-target profiles; "
            "the current immediate-qref Sim-HIL profile does not use it "
            "(default: 0.05 seconds)."
        ),
    )
    worker = parser.add_mutually_exclusive_group()
    worker.add_argument("--realtime-worker", dest="realtime_worker", action="store_true", default=True)
    worker.add_argument("--no-realtime-worker", dest="realtime_worker", action="store_false")
    parser.add_argument("--gui", action="store_true", help="Start the read-only status page.")
    parser.add_argument(
        "--control-gui",
        "--acceptance-gui",
        dest="acceptance_gui",
        action="store_true",
        help=(
            "Enable the loopback typed-intent GUI for formal Sim-HIL use. "
            "Human-only, PPO-only, and proportional blend may be selected in "
            "any order. Requires --enable and writes ControlTrace. "
            "--acceptance-gui is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        default=REPO_ROOT / "teleop_records" / "sim_hil_acceptance",
        help=(
            "Root for acceptance ControlTrace WAL, seal, and commit evidence."
        ),
    )
    parser.add_argument(
        "--acceptance-device-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for a valid passive VIVE/gripper emission.",
    )
    parser.add_argument(
        "--gui-host",
        default=DEFAULT_STATUS_HOST,
        help="Local-only status GUI host (127.0.0.1 or localhost).",
    )
    parser.add_argument("--gui-port", type=int, default=DEFAULT_STATUS_PORT)
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open the local Sim-HIL webpage automatically after the server "
            "starts (default: enabled; use --no-open-browser for headless use)."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> MujocoSimHILApplicationConfig:
    compose_target_hz = (
        args.compose_hz
        if args.compose_hz is not None
        else (
            None
            if args.rate is not None
            else DEFAULT_SIM_HIL_COMPOSE_TARGET_HZ
        )
    )
    return MujocoSimHILApplicationConfig(
        checkpoint_path=args.checkpoint,
        task_config_path=args.task_config,
        calibration_profile_path=args.calibration_profile,
        vive_coordinate_profile_path=args.vive_coordinate_profile,
        vive_port=args.vive_port,
        vive_clock_sync_port=args.vive_clock_sync_port,
        left_q6_port=args.left_q6_port,
        right_q6_port=args.right_q6_port,
        gripper_input_source=args.gripper_input_source,
        q6_config=args.q6_config,
        angle_trigger_config_root=args.angle_trigger_config_root,
        angle_trigger_rate_hz=args.angle_trigger_rate_hz,
        device=args.device,
        enabled=bool(args.enable),
        realtime_worker=bool(args.realtime_worker),
        position_control_profile_name=args.position_control_profile,
        compose_target_hz=compose_target_hz,
        control_rate_hz=args.rate,
        warmup_timeout_sec=args.warmup_timeout,
        human_emission_timeout_sec=args.human_emission_timeout,
        target_admission_margin_sec=args.target_admission_margin_sec,
        initial_human_alpha=args.human_alpha,
        human_timeout_fallback=args.human_timeout_fallback,
        execution_difference_enabled=bool(args.execution_difference),
        mujoco_viewer_enabled=(
            bool(args.acceptance_gui)
            if args.mujoco_viewer is None
            else bool(args.mujoco_viewer)
        ),
        status_host=args.gui_host,
        status_port=args.gui_port,
    )


def _run_dry_run(config: MujocoSimHILApplicationConfig) -> int:
    from control_station.composition.mujoco_sim_hil import (
        build_component_keyed_mujoco_sim_hil_contracts,
    )
    from sim.mujoco.deploy import (
        MujocoSimHILTargetConfig,
        build_rm65_right_cube_lift_component_target_profile,
        prepare_mujoco_sim_hil_target,
    )
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_policy_from_checkpoint,
        check_task_config_compatibility,
    )
    from sim.mujoco_temp_pipeline.policy import resolve_device
    from sim.mujoco_temp_pipeline.hil_policy import (
        validate_temp_ppo_hil_profile,
    )
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from teleop_runtime.calibration import TeleopCalibration

    task = CubeLiftTaskConfig.load(config.task_config_path)
    policy, _normalizer, payload = build_policy_from_checkpoint(
        config.checkpoint_path,
        device=resolve_device(config.device),
    )
    compatibility = check_task_config_compatibility(
        payload,
        config.task_config_path,
    )
    validate_temp_ppo_hil_profile(task, policy)
    calibration = TeleopCalibration.load_json(config.calibration_profile_path)
    calibration, calibration_binding = _bind_vive_coordinate_profile(
        calibration,
        coordinate_profile_path=config.vive_coordinate_profile_path,
        home_profile_path=config.calibration_profile_path,
        required_sides=("right",),
    )
    calibration_binding["home_profile_path"] = str(config.calibration_profile_path)
    calibration_binding["home_profile_sha256"] = _file_sha256(
        config.calibration_profile_path
    )
    home_ready = bool(calibration.metadata.get("home_ready") is True)
    prepared_target = prepare_mujoco_sim_hil_target(
        MujocoSimHILTargetConfig(
            task_config_path=config.task_config_path,
            realtime_worker=config.realtime_worker,
            position_control_profile_name=config.position_control_profile_name,
            compose_target_hz=config.compose_target_hz,
            target_admission_margin_sec=config.target_admission_margin_sec,
            exposed_pose_sides=("right",),
        )
    )
    component_profile = build_rm65_right_cube_lift_component_target_profile(
        prepared_target.scene,
        prepared_target.task_config,
    )
    action_schema = component_profile.action_schema
    policy_components = action_schema.component_ids
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        action_schema,
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.25,
        policy_active_components=policy_components,
        human_weight=config.initial_human_alpha,
        policy_weight=1.0 - config.initial_human_alpha,
    )
    contract_preflight = {
        "passed": True,
        "action_schema_hash": action_schema.semantic_content_hash,
        "composition_spec_hash": contracts.composition_spec.semantic_content_hash,
        "strategy_spec_hash": contracts.strategy_spec.semantic_content_hash,
        "authority_profile_hash": contracts.authority_profile.semantic_content_hash,
        "required_roles": list(contracts.composition_spec.required_roles),
        "component_ids": list(action_schema.component_ids),
        "policy_active_components": list(policy_components),
        "arm_representation": "cartesian_pose",
        "end_effector_representation": "physical_joint_qpos",
        "target_handoff_required": True,
        "target_binding_validation": "startup_target_construction",
        "initial_human_alpha": config.initial_human_alpha,
        "target_admission_margin_sec": (
            config.target_admission_margin_sec
        ),
    }
    print(
        json.dumps(
            {
                "dry_run": True,
                "enabled": bool(config.enabled),
                "execution_target": "mujoco_only",
                "real_arm_motion": False,
                "checkpoint": str(config.checkpoint_path),
                "task_config": str(config.task_config_path),
                "task_name": task.name,
                "position_control_profile": (
                    prepared_target.position_profile.name
                ),
                "compose_target_hz": prepared_target.compose_target_hz,
                "reference_hz": prepared_target.timing.reference_hz,
                "command_hz": prepared_target.timing.command_hz,
                "policy_dimensions": [policy.observation_dim, policy.action_dim],
                "policy_profile_validated": True,
                "compatibility": compatibility,
                "calibration_profile": str(config.calibration_profile_path),
                "vive_coordinate_profile": calibration_binding[
                    "vive_coordinate_profile_path"
                ],
                "calibration_binding": calibration_binding,
                "home_ready": home_ready,
                "contract_preflight": contract_preflight,
                "task_initialization_info": dict(
                    prepared_target.task_initialization_info
                ),
                "device_start_skipped": True,
                "authority_and_runtime_start_skipped": True,
                "mujoco_task_initialization_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if not home_ready:
        return 2
    return 0


def run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.vive_calibration_gui and not args.acceptance_gui:
        raise ValueError(
            "--vive-calibration-gui is a compatibility flag for the "
            "integrated --control-gui calibration controls."
        )
    if args.dry_run:
        return _run_dry_run(config)
    if args.acceptance_gui and not args.enable:
        raise ValueError("--control-gui requires explicit --enable.")
    if args.acceptance_gui and not config.mujoco_viewer_enabled:
        raise ValueError(
            "--control-gui requires the MuJoCo viewer for interactive "
            "teleoperation and policy visualization."
        )
    if not args.enable:
        print(
            json.dumps(
                {
                    "enabled": False,
                    "default_off": True,
                    "profile": "control_station_mujoco_sim_hil_compound_v1",
                    "message": (
                        "Pass --enable to start MuJoCo and real passive "
                        "VIVE/gripper inputs."
                    ),
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.duration < 0.0:
        raise ValueError("duration must be non-negative.")
    if args.acceptance_gui:
        return _run_acceptance_gui(args, config)
    app = _build_application(config)
    gui_server = None
    if args.gui:
        from .sim_hil_gui import SimHILStatusServer

        gui_server = SimHILStatusServer(
            app.status,
            host=config.status_host,
            port=config.status_port,
            live_status_provider=app.vive_status,
            open_browser=bool(args.open_browser),
        )
        host, port = gui_server.start()
        print(f"Sim-HIL read-only GUI: http://{host}:{port}/", flush=True)
    try:
        report = app.start()
        print(
            json.dumps(
                {
                    "started": True,
                    "preflight_passed": report.passed,
                    "readiness_hash": report.readiness.semantic_content_hash,
                    "execution_target": "mujoco_only",
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        rate = config.control_rate_hz
        if rate is None:
            # The policy source keeps its own inference cadence.  The Station
            # loop must drive the final compose/authority schedule selected by
            # the target controller profile (60 Hz for the default Sim-HIL
            # position_60_120hz profile), rather than falling back to policy_hz.
            rate = float(app.runtime.config.decision_hz)
        period = 1.0 / rate
        started_at = time.monotonic()
        next_tick = started_at
        overrun_dropped_cycles = 0
        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - started_at >= args.duration:
                break
            app.step(now=now)
            if app.viewer_exit_requested:
                break
            next_tick, skipped = _advance_control_deadline(
                previous_deadline=next_tick,
                completed_at=time.monotonic(),
                period=period,
            )
            overrun_dropped_cycles += skipped
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(min(remaining, period))
    finally:
        try:
            app.close()
        finally:
            if gui_server is not None:
                gui_server.close()
    return 0


def _build_sim_hil_vive_setup_controller(
    config: MujocoSimHILApplicationConfig,
    *,
    source_role: str,
) -> Any:
    """Start the pre-target VIVE owner used by the unified Station GUI."""

    from .vive_calibration import (
        ViveCoordinateCalibrationConfig,
        ViveCoordinateCalibrationController,
    )

    role = str(source_role).strip()
    if role not in {"left_wrist", "right_wrist"}:
        raise ValueError(
            "Sim-HIL VIVE setup source_role must be left_wrist or "
            "right_wrist."
        )
    controller = ViveCoordinateCalibrationController(
        ViveCoordinateCalibrationConfig(
            active_sides=("right",),
            desk_tracker="auto",
            vive_port=config.vive_port,
            vive_clock_sync_port=config.vive_clock_sync_port,
        )
    )
    if role == "left_wrist":
        # The right-only Sim-HIL compatibility profile historically selects
        # the sender's left_wrist role for logical right_wrist_pose. Preserve
        # that mapping visibly, then let the unified GUI swap it if needed.
        swap = getattr(
            controller.source,
            "swap_left_right_source_roles",
            None,
        )
        if not callable(swap):
            raise RuntimeError(
                "Sim-HIL VIVE setup source does not support the declared "
                "single-tracker role mapping."
            )
        swap()
    try:
        controller.start(start_source=True)
    except Exception:
        # The setup receiver is optional for a policy-only runtime. Keep its
        # typed stopped/error snapshot in the GUI so PPO construction remains
        # independent while Human modes stay unavailable.
        pass
    return controller


def _sim_hil_bootstrap_status(
    config: MujocoSimHILApplicationConfig,
    *,
    setup: Mapping[str, object],
    vive_status: Mapping[str, object],
) -> dict[str, Any]:
    """Project VIVE/profile setup without constructing a MuJoCo target."""

    calibration = (
        setup.get("calibration", {})
        if isinstance(setup.get("calibration", {}), Mapping)
        else {}
    )
    input_ready = bool(setup.get("input_ready", False))
    profile_ready = bool(setup.get("profile_ready", False))
    failed_checks = [
        name
        for name, ready in (
            ("vive_input_ready", input_ready),
            ("vive_coordinate_profile_ready", profile_ready),
        )
        if not ready
    ]
    receiver = (
        vive_status.get("vive_receiver", {})
        if isinstance(vive_status.get("vive_receiver", {}), Mapping)
        else {}
    )
    return {
        "schema": SIM_HIL_STATUS_SCHEMA_V1,
        "sampled_at": time.monotonic(),
        "profile_id": "control_station_mujoco_sim_hil_compound_v1",
        "enabled": bool(config.enabled),
        "started": False,
        "lifecycle": str(setup.get("phase") or "vive_setup"),
        "held": False,
        "episode_ended": False,
        "startup": {
            "restart_required": False,
            "first_failure": None,
            "closed": False,
            "phase": setup.get("phase"),
            "mujoco_constructed": False,
        },
        "first_control_fault": None,
        "visualization": {
            "enabled": bool(config.mujoco_viewer_enabled),
            "state": "not_constructed",
            "running": False,
            "sync_count": 0,
            "last_error": None,
        },
        "control_schedule": {
            "next_decision_at": None,
            "overrun_dropped_cycles": 0,
            "last_overrun": None,
            "target_admission_margin_sec": (
                config.target_admission_margin_sec
            ),
            "target_lead_time_sec": None,
            "proposal_horizon_sec": None,
        },
        "readiness": {
            "ready": False,
            "required_checks": [
                "vive_input_ready",
                "vive_coordinate_profile_ready",
            ],
            "failed_checks": failed_checks,
            "detail_by_check": {
                "vive_input_ready": (
                    "live finite fresh right_wrist_pose"
                    if input_ready
                    else (
                        calibration.get("input_preflight", {})
                        if isinstance(
                            calibration.get("input_preflight", {}),
                            Mapping,
                        )
                        else {}
                    ).get("message", "VIVE input is not ready")
                ),
                "vive_coordinate_profile_ready": (
                    str(setup.get("selected_profile_path"))
                    if profile_ready
                    else "No valid saved coordinate profile is selected."
                ),
            },
        },
        "execution_gate": {
            "status": "not_constructed",
            "reason": "waiting_for_vive_setup",
        },
        "policy": {
            "started": False,
            "ready": False,
            "refresh_state": "not_constructed",
            "checkpoint": str(config.checkpoint_path),
            "last_emission": {},
        },
        "human": {
            "session_state": "VIVE_SETUP",
            "source_started": bool(
                calibration.get("source_started", False)
            ),
            "last_emission": {},
            "last_emission_age_sec": None,
            "binding": {
                "rig_absent": True,
                "authoritative": False,
            },
            "session_safety": {},
            "channels": dict(calibration.get("live_channels") or {}),
            "vive_receiver": dict(receiver),
        },
        "composition": {
            "started": False,
            "activation": {},
            "last_cycle": {},
            "mix": {},
        },
        "execution_difference": {
            "enabled": bool(config.execution_difference_enabled),
            "latest": {},
        },
        "last_error": None,
        "execution_target": "mujoco_not_constructed",
        "real_arm_motion": False,
        "single_execution_owner": "not_created",
        "calibration": {
            "vive_coordinate_profile_path": setup.get(
                "selected_profile_path"
            ),
            "mode": "station_vive_setup_before_target",
        },
        "task_initialization": {},
        "source_configuration": {
            "vive_port": int(config.vive_port),
            "vive_clock_sync_port": config.vive_clock_sync_port,
            "active_components": [],
            "vive_source_role_by_output_channel": dict(
                calibration.get("source_role_by_output_channel") or {}
            ),
            "vive_source_swap_count": calibration.get(
                "source_swap_count",
                0,
            ),
            "input_device_bindings": [
                {
                    "component_id": "right_gripper",
                    "requested_device_kind": config.gripper_input_source,
                    "device_kind": None,
                    "endpoint": None,
                    "fallback_reason": "resolved_after_vive_setup",
                }
            ],
        },
        "contracts": {},
    }


def _run_acceptance_gui(
    args: argparse.Namespace,
    config: MujocoSimHILApplicationConfig,
) -> int:
    """Run formal interactive control with one Station-owned control loop."""

    import secrets

    from .sim_hil_acceptance import (
        SimHILAcceptanceController,
        SimHILAcceptanceTraceCapture,
    )
    from .sim_hil_gui import SimHILStatusServer

    capture = None
    acceptance_config = _acceptance_profile_config(config)
    controller = None
    gui_server = None
    try:
        capture = SimHILAcceptanceTraceCapture(args.acceptance_root)

        def setup_factory(source_role: str) -> Any:
            return _build_sim_hil_vive_setup_controller(
                acceptance_config,
                source_role=source_role,
            )

        setup_controller = setup_factory(
            acceptance_config.vive_source_tracker_role
        )

        def application_factory(
            coordinate_profile_path: Path,
            source_role: str,
            initial_human_alpha: float,
        ) -> MujocoSimHILApplication:
            defer_human_input = initial_human_alpha <= 1e-12
            runtime_config = replace(
                acceptance_config,
                vive_coordinate_profile_path=(
                    None
                    if defer_human_input
                    else coordinate_profile_path
                ),
                vive_source_tracker_role=source_role,
                initial_human_alpha=float(initial_human_alpha),
                defer_human_input=defer_human_input,
            )
            return _build_application(
                runtime_config,
                trace_sink=capture.admit_trace,
            )

        controller = SimHILAcceptanceController(
            None,
            trace_capture=capture,
            device_check_timeout_sec=float(
                args.acceptance_device_timeout
            ),
            application_factory=application_factory,
            vive_setup_controller=setup_controller,
            vive_setup_factory=setup_factory,
            coordinate_profile_path=(
                acceptance_config.vive_coordinate_profile_path
                or DEFAULT_VIVE_COORDINATE_PROFILE
            ),
        )

        cached_runtime_status: dict[str, Any] | None = None

        def status_provider() -> dict[str, Any]:
            nonlocal cached_runtime_status
            acceptance_status = controller.status()
            app = controller.current_application
            if app is None:
                setup = controller.setup_status()
                value = _sim_hil_bootstrap_status(
                    acceptance_config,
                    setup=setup,
                    vive_status=controller.vive_status(),
                )
            elif acceptance_status.get("active_intent_id") is not None:
                # start/rearm/Home may hold the application lock while waiting
                # on lower resources.  Serving the last immutable projection
                # keeps cancel/exit reachable instead of blocking the HTTP
                # status request behind the operation it must interrupt.
                if cached_runtime_status is None:
                    value = _sim_hil_bootstrap_status(
                        acceptance_config,
                        setup=controller.setup_status(),
                        vive_status=controller.vive_status(),
                    )
                    value["startup"]["mujoco_constructed"] = True
                else:
                    value = dict(cached_runtime_status)
            else:
                value = app.status()
                cached_runtime_status = dict(value)
            value["acceptance"] = acceptance_status
            return value

        gui_server = SimHILStatusServer(
            status_provider,
            host=acceptance_config.status_host,
            port=acceptance_config.status_port,
            live_status_provider=controller.vive_status,
            intent_submitter=controller.submit_json,
            control_token=secrets.token_urlsafe(32),
            open_browser=bool(args.open_browser),
        )
        host, port = gui_server.start()
        print(
            f"Sim-HIL control GUI: http://{host}:{port}/",
            flush=True,
        )
        started_at = time.monotonic()
        passive_observation_due_at = started_at
        while not controller.shutdown_requested:
            now = time.monotonic()
            if (
                args.duration > 0.0
                and now - started_at >= args.duration
            ):
                break
            controller.process_next()
            controller.enforce_capture_health()
            app = controller.current_application
            if (
                app is not None
                and app.started
                and not app._held
                and not app._episode_ended
            ):
                due_at = app._next_decision_at
                if due_at is None or now >= due_at:
                    try:
                        app.step(now=now)
                    except Exception:
                        # The application/lower gate retains the fault and
                        # HOLD evidence. Keep the GUI alive for inspection.
                        if not app._held:
                            try:
                                app.hold(
                                    reason="acceptance_control_loop_fault"
                                )
                            except Exception:
                                pass
            elif (
                app is not None
                and app.started
                and app._held
                and not app._episode_ended
                and now >= passive_observation_due_at
            ):
                # Keep the passive device observation fresh for diagnosis and
                # Home/rearm while HOLD. This never steps policy, composition,
                # the target, or MuJoCo dispatch.
                try:
                    app.poll_passive_human_input(now=now)
                except Exception:
                    pass
                passive_observation_due_at = now + 0.05
            if app is not None and app.viewer_exit_requested:
                break
            next_due = None if app is None else app._next_decision_at
            delay = 0.01
            if (
                app is not None
                and app.started
                and not app._held
                and next_due is not None
            ):
                rate = (
                    float(acceptance_config.control_rate_hz)
                    if acceptance_config.control_rate_hz is not None
                    else float(app.runtime.config.decision_hz)
                )
                period = 1.0 / rate
                delay = max(
                    0.0005,
                    min(period, float(next_due) - time.monotonic()),
                )
            time.sleep(delay)
        return 0
    finally:
        if gui_server is not None:
            gui_server.close()
        app = (
            None
            if controller is None
            else controller.current_application
        )
        if app is not None:
            try:
                app.close()
            finally:
                if controller is not None:
                    controller.close()
                elif capture is not None:
                    capture.close()
        elif controller is not None:
            controller.close()
        elif capture is not None:
            capture.close()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return int(run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - CLI reports a typed failure.
        print(f"control_station sim-hil failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CALIBRATION_PROFILE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_TASK_CONFIG",
    "DEFAULT_VIVE_COORDINATE_PROFILE",
    "MujocoSimHILApplication",
    "MujocoSimHILApplicationConfig",
    "SIM_HIL_HUMAN_TIMEOUT_FALLBACKS",
    "SIM_HIL_STATUS_SCHEMA_V1",
    "build_arg_parser",
    "main",
    "run",
]
