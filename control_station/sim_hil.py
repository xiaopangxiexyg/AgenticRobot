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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_motion.position_control import POSITION_60_120_PROFILE_NAME


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "sim"
    / "mujoco_temp_pipeline"
    / "checkpoints"
    / "cube_lift_contact_curriculum_v3.pt"
)
DEFAULT_TASK_CONFIG = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
)
DEFAULT_CALIBRATION_PROFILE = (
    REPO_ROOT
    / "teleop_records"
    / "calibration_profiles"
    / "20260627_112114_vive_rm65_virtual_desk.json"
)
DEFAULT_VIVE_PORT = 9999
DEFAULT_STATUS_HOST = "127.0.0.1"
DEFAULT_STATUS_PORT = 8787
SIM_HIL_STATUS_SCHEMA_V1 = "dexteleop.control_station.sim_hil_status.v1"
_LOOPBACK_STATUS_HOSTS = frozenset({"127.0.0.1", "localhost"})


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class MujocoSimHILApplicationConfig:
    """Station-only profile selection and lifecycle settings."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    task_config_path: Path = DEFAULT_TASK_CONFIG
    calibration_profile_path: Path = DEFAULT_CALIBRATION_PROFILE
    vive_coordinate_profile_path: Path | None = None
    vive_port: int = DEFAULT_VIVE_PORT
    vive_clock_sync_port: int | None = None
    left_q6_port: str | None = None
    right_q6_port: str | None = None
    device: str = "auto"
    enabled: bool = False
    realtime_worker: bool = True
    # Explicit final compose/authority target cadence.  ``control_rate_hz``
    # remains a compatibility alias for the historical --rate option.
    position_control_profile_name: str = POSITION_60_120_PROFILE_NAME
    compose_target_hz: float | None = None
    control_rate_hz: float | None = None
    warmup_timeout_sec: float = 10.0
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
        for name in ("left_q6_port", "right_q6_port"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when supplied.")
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
        status_port = int(self.status_port)
        if not 0 <= status_port <= 65535:
            raise ValueError("status_port must be in [0, 65535].")
        object.__setattr__(self, "status_port", status_port)
        status_host = str(self.status_host).strip().lower()
        if status_host not in _LOOPBACK_STATUS_HOSTS:
            raise ValueError(
                "status_host must be loopback-only (127.0.0.1 or localhost)."
            )
        object.__setattr__(self, "status_host", status_host)


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
) -> tuple[Any, dict[str, Any]]:
    """Bind Station VIVE coordinates onto a Home-capable target profile.

    ``TeleopCalibration`` remains owned by ``teleop_runtime``.  The Station
    profile only selects and combines two immutable inputs: the target/session
    profile supplies robot transforms and Home anchors; the independent
    Control Station profile supplies the VIVE world-to-desk transform and
    tracker offsets.  No Home data is written back to either source profile.
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
    if not coordinate_source.is_absolute() and coordinate_source.exists():
        coordinate_source = coordinate_source.resolve()
    coordinate, resolved_path = load_coordinate_profile(coordinate_source)
    active_sides = validate_coordinate_profile(coordinate, path=resolved_path)
    if active_sides != ("left", "right"):
        raise ValueError(
            "The bimanual Sim-HIL profile requires a VIVE coordinate profile "
            "with active_sides=left,right."
        )
    world_to_desk = coordinate.world_to_desk_pose()
    if world_to_desk is None:
        raise ValueError("VIVE coordinate profile is missing world_to_vive_retarget.")

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
            "sim_hil_calibration_binding": (
                "independent_vive_coordinate_plus_target_home_v1"
            ),
        }
    )
    merged.metadata = metadata
    binding.update(
        {
            "mode": "independent_vive_coordinate_plus_target_home_v1",
            "vive_coordinate_profile_path": str(resolved_path),
            "vive_coordinate_profile_sha256": _file_sha256(resolved_path),
            "coordinate_profile_kind": coordinate.metadata.get("profile_kind"),
            "coordinate_profile_schema": coordinate.metadata.get("profile_schema"),
            "active_sides": list(active_sides),
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
    ) -> None:
        self.policy_source = policy_source
        self.human_source = human_source
        self.scene = scene
        self.calibration_ready = bool(calibration_ready)
        self.home_ready = bool(home_ready)
        self.max_human_age_sec = float(max_human_age_sec)
        self.max_policy_age_sec = float(max_policy_age_sec)
        self.clock = clock

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
        human_state = getattr(human_session.state_machine.state, "value", str(human_session.state_machine.state))
        human_emission = self.human_source.last_emission
        human_fresh, human_detail = self._age_detail(
            human_emission,
            now=timestamp,
            max_age_sec=self.max_human_age_sec,
        )
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

        faulted = human_state in {"fault", "estop", "frozen"}
        home_recalibration_required = bool(
            getattr(human_session, "home_recalibration_required", False)
        )
        hold_mode = str(getattr(human_session, "motion_hold_mode", ""))
        human_ready = human_fresh and not faulted and not home_recalibration_required
        rebase_ready = self.home_ready and not home_recalibration_required
        # The passive source never arms a real target.  DEBUG_ONLY is the
        # explicit non-motion deadman/paused gate for this sim-only profile.
        deadman_ready = (
            human_state == "debug_only"
            and not faulted
            and not home_recalibration_required
            and hold_mode in {"", "none"}
        )
        policy_ready = bool(
            getattr(self.policy_source, "ready", False)
            and policy_fresh
        )
        measured_state_ready = bool(self.scene.state_is_finite())
        details = {
            "policy_source_ready": (
                "ready" if policy_ready else str(getattr(self.policy_source, "last_error", None) or policy_detail)
            ),
            "human_device_ready": human_detail,
            "calibration_ready": "loaded_and_acknowledged" if self.calibration_ready else "calibration_not_ready",
            "home_ready": "anchors_seeded" if self.home_ready else "home_anchors_missing",
            "rebase_ready": "not_required" if rebase_ready else "rebase_required",
            "deadman_ready": (
                "passive_debug_only" if deadman_ready else f"teleop_state:{human_state}"
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


def _build_application(config: MujocoSimHILApplicationConfig) -> "MujocoSimHILApplication":
    if not config.enabled:
        raise RuntimeError("MuJoCo Sim-HIL application is default-off; pass --enable.")
    if config.left_q6_port is None or config.right_q6_port is None:
        raise ValueError("Enabled Sim-HIL requires --left-q6-port and --right-q6-port.")

    from control_station.composition.mujoco_sim_hil import (
        MujocoSimHILProfileConfig,
        build_default_mujoco_sim_hil_contracts,
        build_scoped_mujoco_sim_hil_composition,
    )
    from robot_motion.action import HandActionSpec
    from sim.mujoco.deploy import (
        MujocoSimHILTargetConfig,
        build_mujoco_pose_action_schema,
        build_mujoco_sim_hil_target,
    )
    from sim.mujoco_temp_pipeline.hil_policy import TempPPOHILPolicySource
    from teleop_runtime.calibration import TeleopCalibration
    from teleop_runtime.configs import PassiveViveQ6SessionConfig

    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    policy_effectors = ("right_arm", "right_hand")
    action_schema = build_mujoco_pose_action_schema(hand_specs)
    contracts = build_default_mujoco_sim_hil_contracts(
        hand_specs,
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.1,
        policy_active_effectors=policy_effectors,
        action_schema_hash=action_schema.semantic_content_hash,
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=config.task_config_path,
            realtime_worker=config.realtime_worker,
            position_control_profile_name=config.position_control_profile_name,
            compose_target_hz=config.compose_target_hz,
        ),
        contracts=contracts,
        hand_specs_by_side=hand_specs,
    )
    if target.runtime.action_schema.semantic_content_hash != (
        action_schema.semantic_content_hash
    ):
        raise RuntimeError(
            "MuJoCo Sim-HIL target built a different ComponentActionSchema."
        )
    calibration = TeleopCalibration.load_json(config.calibration_profile_path)
    calibration, calibration_binding = _bind_vive_coordinate_profile(
        calibration,
        coordinate_profile_path=config.vive_coordinate_profile_path,
    )
    calibration_binding["home_profile_path"] = str(config.calibration_profile_path)
    calibration_binding["home_profile_sha256"] = _file_sha256(
        config.calibration_profile_path
    )
    if calibration.metadata.get("home_ready") is not True:
        raise ValueError(
            "Sim-HIL requires a Home-capable calibration profile with "
            "metadata.home_ready=true."
        )
    human_source, human_source_spec = _build_passive_vive_q6_component_source(
        calibration=calibration,
        source_config=PassiveViveQ6SessionConfig(
            vive_port=config.vive_port,
            vive_clock_sync_port=config.vive_clock_sync_port,
            left_q6_port=config.left_q6_port,
            right_q6_port=config.right_q6_port,
        ),
        target_action_spec_hash_by_component=(
            target.runtime.target_runtime_action_contract.action_spec_hash_by_component
        ),
        target_units_convention_id=(
            target.runtime.target_runtime_action_contract.units_convention_id
        ),
    )
    policy_source = TempPPOHILPolicySource(
        target.scene,
        checkpoint_path=config.checkpoint_path,
        task_config_path=config.task_config_path,
        policy_node_id=target.runtime.config.policy_node_id,
        policy_clock_domain_id=target.runtime.config.clock_domain_id,
        device=config.device,
    )
    profile_config = MujocoSimHILProfileConfig(
        enabled=True,
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
        calibration_ready=True,
        home_ready=True,
        max_human_age_sec=profile_config.max_human_emission_age_sec,
        max_policy_age_sec=profile_config.max_policy_emission_age_sec,
        clock=target.runtime.clock,
    )
    composition = build_scoped_mujoco_sim_hil_composition(
        runtime=target.runtime,
        human_source=human_source,
        human_source_spec=human_source_spec,
        readiness_provider=readiness_provider,
        contracts=contracts,
        config=profile_config,
        policy_active_effectors=policy_effectors,
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
        calibration_binding=calibration_binding,
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
        result[str(name)] = {
            "valid": bool(getattr(channel, "valid", False)),
            "capability": getattr(channel, "capability", None),
            "timestamp": timestamp,
            "age_sec": None
            if timestamp is None
            else float(now) - float(timestamp),
        }
    return result


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
        calibration_binding: dict[str, Any] | None = None,
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
        self.calibration_binding = dict(calibration_binding or {})
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.RLock()
        self._started = False
        self._sources_started = False
        self._last_preflight: Any | None = None
        self._last_cycle: Any | None = None
        self._last_error: str | None = None

    @property
    def started(self) -> bool:
        return self._started

    def _warmup_sources(self) -> None:
        self.policy_source.start()
        self.human_source.start()
        self.human_source.set_calibration_ready()
        deadline = self.clock() + self.config.warmup_timeout_sec
        self._sources_started = True
        while True:
            now = _finite(self.clock(), name="warmup.now")
            self.policy_source.step(now=now)
            self.human_source.step(now=now)
            evidence = self.readiness_provider.snapshot(now=now)
            if evidence.ready:
                return
            if now >= deadline:
                raise TimeoutError(
                    "Sim-HIL source readiness timed out: "
                    + ";".join(
                        f"{name}={evidence.detail_by_check.get(name, '')}"
                        for name in evidence.failed_checks
                    )
                )
            self.sleep(0.005)

    def start(self) -> Any:
        with self._lock:
            if self._started:
                return self._last_preflight
            try:
                self._warmup_sources()
                report = self.composition.start()
                # The warm-up emissions predate the composition activation barrier.
                # Reset the lower-owned policy schedule and acquire one fresh pair
                # after that barrier; no second queue or authority is introduced.
                self.policy_source.reset_for_activation()
                now = _finite(self.clock(), name="activation.now")
                self.policy_source.step(now=now)
                self.human_source.step(now=now)
                self._last_preflight = report
                self._started = True
                self._last_error = None
                return report
            except Exception as error:
                self._last_error = f"{type(error).__name__}:{error}"
                self.close()
                raise

    def step(self, *, now: float | None = None) -> Any:
        with self._lock:
            if not self._started:
                raise RuntimeError("MujocoSimHILApplication.start() is required.")
            timestamp = _finite(
                self.clock() if now is None else now,
                name="step.now",
            )
            try:
                policy_emission = self.policy_source.step(now=timestamp)
                self._last_cycle = self.composition.step(
                    now=timestamp,
                    policy_emission=policy_emission,
                    poll_human=True,
                )
                self._last_error = None
                return self._last_cycle
            except Exception as error:
                self._last_error = f"{type(error).__name__}:{error}"
                raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = _finite(self.clock(), name="status.now")
            readiness = self.readiness_provider.snapshot(now=now)
            policy_status = dict(self.policy_source.status(now=now))
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
                None
                if self._last_cycle is None
                else getattr(self._last_cycle, "human_step", None)
            )
            human_status = {
                "session_state": getattr(human_state, "value", str(human_state)),
                "source_started": bool(getattr(self.human_source, "started", False)),
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
                "channels": _human_channel_status(last_human_step, now=now),
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
            }

            result: dict[str, Any] = {
                "schema": SIM_HIL_STATUS_SCHEMA_V1,
                "sampled_at": now,
                "profile_id": "control_station_mujoco_sim_hil_compound_v1",
                "enabled": bool(self.config.enabled),
                "started": self._started,
                "lifecycle": "running" if self._started else "stopped",
                "last_error": self._last_error,
                "execution_target": "mujoco_only",
                "real_arm_motion": False,
                "single_execution_owner": "MujocoDeployRuntime/composer",
                "calibration": dict(self.calibration_binding),
                "source_configuration": {
                    "vive_port": int(self.config.vive_port),
                    "vive_clock_sync_port": self.config.vive_clock_sync_port,
                    "left_q6_port": self.config.left_q6_port,
                    "right_q6_port": self.config.right_q6_port,
                },
                "contracts": _runtime_contract_status(self.runtime, self.composition),
                "readiness": {
                    "ready": readiness.ready,
                    "failed_checks": list(readiness.failed_checks),
                    "detail_by_check": dict(readiness.detail_by_check),
                    "semantic_content_hash": readiness.semantic_content_hash,
                },
                "policy": policy_status,
                "human": human_status,
                "execution_gate": {"status": "not_started"},
                "target_execution": None,
                "preflight": None,
                "composition": {
                    "started": bool(getattr(self.composition, "started", False)),
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

    def close(self) -> None:
        with self._lock:
            errors: list[BaseException] = []
            if self.composition.started:
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
            self._started = False
            self._sources_started = False
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
        description="Default-off MuJoCo-only Sim-HIL using temporary PPO + VIVE/Q6 passive input.",
    )
    parser.add_argument("--enable", action="store_true", help="Explicitly enable source startup and MuJoCo execution.")
    parser.add_argument("--dry-run", action="store_true", help="Validate PPO/task/calibration bindings without starting MuJoCo or devices.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        default=DEFAULT_CALIBRATION_PROFILE,
        help="Home-capable target/session calibration profile.",
    )
    parser.add_argument(
        "--vive-coordinate-profile",
        type=Path,
        default=None,
        help=(
            "Optional independent Control Station VIVE coordinate profile. "
            "The calibration profile still supplies Home/target anchors."
        ),
    )
    parser.add_argument("--vive-port", type=int, default=DEFAULT_VIVE_PORT)
    parser.add_argument("--vive-clock-sync-port", type=int, default=None)
    parser.add_argument("--left-q6-port", default=None)
    parser.add_argument("--right-q6-port", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--duration", type=float, default=0.0, help="Run seconds; 0 means until interrupted.")
    parser.add_argument(
        "--position-control-profile",
        type=str,
        default=POSITION_60_120_PROFILE_NAME,
        help=(
            "Position controller rate profile. The default accepts composed "
            "targets at 60 Hz; select position_15_60_120hz for a slower input "
            "cadence with the same 60/120 downstream clocks."
        ),
    )
    parser.add_argument(
        "--compose-hz",
        type=float,
        default=None,
        help=(
            "Final HIL compose/authority target rate; defaults to the selected "
            "position profile input rate."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=(
            "Compatibility alias for --compose-hz and the Station loop rate; "
            "defaults to the selected target profile rate."
        ),
    )
    parser.add_argument("--warmup-timeout", type=float, default=10.0)
    worker = parser.add_mutually_exclusive_group()
    worker.add_argument("--realtime-worker", dest="realtime_worker", action="store_true", default=True)
    worker.add_argument("--no-realtime-worker", dest="realtime_worker", action="store_false")
    parser.add_argument("--gui", action="store_true", help="Start the read-only status page.")
    parser.add_argument(
        "--gui-host",
        default=DEFAULT_STATUS_HOST,
        help="Local-only status GUI host (127.0.0.1 or localhost).",
    )
    parser.add_argument("--gui-port", type=int, default=DEFAULT_STATUS_PORT)
    return parser


def _config_from_args(args: argparse.Namespace) -> MujocoSimHILApplicationConfig:
    return MujocoSimHILApplicationConfig(
        checkpoint_path=args.checkpoint,
        task_config_path=args.task_config,
        calibration_profile_path=args.calibration_profile,
        vive_coordinate_profile_path=args.vive_coordinate_profile,
        vive_port=args.vive_port,
        vive_clock_sync_port=args.vive_clock_sync_port,
        left_q6_port=args.left_q6_port,
        right_q6_port=args.right_q6_port,
        device=args.device,
        enabled=bool(args.enable),
        realtime_worker=bool(args.realtime_worker),
        position_control_profile_name=args.position_control_profile,
        compose_target_hz=args.compose_hz,
        control_rate_hz=args.rate,
        warmup_timeout_sec=args.warmup_timeout,
        status_host=args.gui_host,
        status_port=args.gui_port,
    )


def _run_dry_run(config: MujocoSimHILApplicationConfig) -> int:
    from control_station.composition.mujoco_sim_hil import (
        build_default_mujoco_sim_hil_contracts,
    )
    from robot_motion.action import HandActionSpec
    from sim.mujoco.deploy import build_mujoco_pose_action_schema
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_policy_from_checkpoint,
        check_task_config_compatibility,
    )
    from sim.mujoco_temp_pipeline.policy import resolve_device
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
    calibration = TeleopCalibration.load_json(config.calibration_profile_path)
    calibration, calibration_binding = _bind_vive_coordinate_profile(
        calibration,
        coordinate_profile_path=config.vive_coordinate_profile_path,
    )
    calibration_binding["home_profile_path"] = str(config.calibration_profile_path)
    calibration_binding["home_profile_sha256"] = _file_sha256(
        config.calibration_profile_path
    )
    home_ready = bool(calibration.metadata.get("home_ready") is True)
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    policy_effectors = ("right_arm", "right_hand")
    action_schema = build_mujoco_pose_action_schema(hand_specs)
    contracts = build_default_mujoco_sim_hil_contracts(
        hand_specs,
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.1,
        policy_active_effectors=policy_effectors,
        action_schema_hash=action_schema.semantic_content_hash,
    )
    contract_preflight = {
        "passed": True,
        "action_schema_hash": action_schema.semantic_content_hash,
        "composition_spec_hash": contracts.composition_spec.semantic_content_hash,
        "strategy_spec_hash": contracts.strategy_spec.semantic_content_hash,
        "authority_profile_hash": contracts.authority_profile.semantic_content_hash,
        "required_roles": list(contracts.composition_spec.required_roles),
        "policy_active_effectors": list(policy_effectors),
        "target_handoff_required": True,
        "target_binding_validation": "startup_target_construction",
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
                "policy_dimensions": [policy.observation_dim, policy.action_dim],
                "compatibility": compatibility,
                "calibration_profile": str(config.calibration_profile_path),
                "vive_coordinate_profile": calibration_binding[
                    "vive_coordinate_profile_path"
                ],
                "calibration_binding": calibration_binding,
                "home_ready": home_ready,
                "contract_preflight": contract_preflight,
                "device_start_skipped": True,
                "mujoco_start_skipped": True,
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
    if args.dry_run:
        return _run_dry_run(config)
    if not args.enable:
        print(
            json.dumps(
                {
                    "enabled": False,
                    "default_off": True,
                    "profile": "control_station_mujoco_sim_hil_compound_v1",
                    "message": "Pass --enable to start MuJoCo and real VIVE/Q6 passive inputs.",
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.duration < 0.0:
        raise ValueError("duration must be non-negative.")
    app = _build_application(config)
    gui_server = None
    if args.gui:
        from .sim_hil_gui import SimHILStatusServer

        gui_server = SimHILStatusServer(
            app.status,
            host=config.status_host,
            port=config.status_port,
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
        while True:
            now = time.monotonic()
            if args.duration > 0.0 and now - started_at >= args.duration:
                break
            app.step(now=now)
            next_tick += period
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
    "MujocoSimHILApplication",
    "MujocoSimHILApplicationConfig",
    "SIM_HIL_STATUS_SCHEMA_V1",
    "build_arg_parser",
    "main",
    "run",
]
