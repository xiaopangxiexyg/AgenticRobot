"""Concrete MuJoCo target/source/composition graph assembly for Sim-HIL.

The factory constructs one graph and returns its application lifecycle owner.
It does not run the control loop or own lower execution semantics.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from .application import MujocoSimHILApplication
from .config import MujocoSimHILApplicationConfig
from .human import (
    _SimHILHumanAttachment,
    _bind_vive_coordinate_profile,
    _build_passive_single_vive_gripper_component_source,
    _build_single_vive_gripper_component_source_spec,
    _file_sha256,
)
from .profile_selection import (
    SimHILControlProfileAssembly,
    SimHILControlProfileBinding,
    require_sim_hil_control_profile_available,
    select_sim_hil_application_control_profile,
)
from .policy_human_allocation import (
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
    SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS,
    build_policy_human_allocation_scheme_a_control_profile,
    build_policy_human_allocation_scheme_b_control_profile,
    prepare_policy_human_allocation_scheme_a,
    prepare_policy_human_allocation_scheme_b,
)
from .policy_source import (
    SimHILPolicySourceBuildContext,
    SimHILPolicySourcePort,
    require_sim_hil_policy_source,
)
from .validation import _finite


def _allocation_profile_builders(profile_id: str) -> tuple[Any, Any]:
    """Return the paired contract/runtime builders for one built-in profile."""

    if profile_id == SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A:
        return (
            prepare_policy_human_allocation_scheme_a,
            build_policy_human_allocation_scheme_a_control_profile,
        )
    if profile_id == SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B:
        return (
            prepare_policy_human_allocation_scheme_b,
            build_policy_human_allocation_scheme_b_control_profile,
        )
    raise ValueError(f"Not a built-in allocation profile: {profile_id!r}.")


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
                "ready"
                if policy_ready
                else str(
                    getattr(self.policy_source, "last_error", None)
                    or policy_detail
                )
            ),
            "human_device_ready": human_detail,
            "calibration_ready": (
                "loaded_and_acknowledged"
                if self.calibration_ready
                else "calibration_not_ready"
            ),
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


def build_application_graph(
    config: MujocoSimHILApplicationConfig,
    *,
    trace_sink: Any | None = None,
    human_source_builder: Callable[..., tuple[Any, Any]] | None = None,
    policy_source_builder: (
        Callable[[SimHILPolicySourceBuildContext], SimHILPolicySourcePort]
        | None
    ) = None,
    control_profile_bindings: Sequence[
        SimHILControlProfileBinding
    ] = (),
) -> MujocoSimHILApplication:
    profile_bindings = tuple(control_profile_bindings)
    allocation_profile_selected = (
        config.control_profile_id in SIM_HIL_POLICY_HUMAN_ALLOCATION_PROFILE_IDS
    )
    allocation_prepare = None
    allocation_control_profile_builder = None
    if allocation_profile_selected:
        (
            allocation_prepare,
            allocation_control_profile_builder,
        ) = _allocation_profile_builders(config.control_profile_id)
    if not allocation_profile_selected:
        require_sim_hil_control_profile_available(
            config.control_profile_id,
            additional_bindings=profile_bindings,
        )
    if human_source_builder is None:
        human_source_builder = (
            _build_passive_single_vive_gripper_component_source
        )
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
            required_representation_by_component={
                "right_arm": "cartesian_pose",
            },
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
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
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
    allocation_preparation = (
        allocation_prepare(
            action_schema=action_schema,
            target_frame_graph=prepared_target.target_frame_profile.frame_graph,
            target_frame_profile_hash=(
                prepared_target.target_frame_profile.semantic_content_hash
            ),
            tool_frame_profile_hash=(
                prepared_target.tool_frame_profile.semantic_content_hash
            ),
            composer_producer_id="control-station-sim-hil-composer",
        )
        if allocation_profile_selected
        else None
    )
    contracts = (
        allocation_preparation.bundle
        if allocation_preparation is not None
        else build_component_keyed_mujoco_sim_hil_contracts(
            action_schema,
            composer_producer_id="control-station-sim-hil-composer",
            proposal_ttl_sec=0.25,
            policy_active_components=action_schema.component_ids,
            human_weight=config.initial_human_alpha,
            policy_weight=1.0 - config.initial_human_alpha,
            max_pose_translation_disagreement_m=None,
            max_pose_rotation_disagreement_rad=None,
        )
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
        composition_input_action_schema_by_role=(
            allocation_preparation.bundle.input_action_schema_by_role
            if allocation_preparation is not None
            else {
                "human": action_schema,
                "policy": action_schema,
            }
        ),
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
            human_source_builder(
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
    policy_build_context = SimHILPolicySourceBuildContext(
        scene=target.scene,
        task_config=target.task_config,
        task_config_path=Path(config.task_config_path),
        target_runtime=target.runtime,
        task_initialization_info=dict(target.task_initialization_info),
        downstream_target_lead_sec=(
            target.runtime.reference_activation_spec.target_lead_time_sec
        ),
        scene_capture=target.runtime.rig.capture_scene,
    )
    policy_source = require_sim_hil_policy_source(
        TempPPOHILPolicySource(
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
        if policy_source_builder is None
        else policy_source_builder(policy_build_context)
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
    composition = None
    if allocation_preparation is None:
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
    control_profile = (
        allocation_control_profile_builder(
            preparation=allocation_preparation,
            runtime=target.runtime,
            human_source=human_source,
            human_source_spec=human_source_spec,
            readiness_provider=readiness_provider,
            policy_binding=policy_binding,
            component_action_projector=ingress_projector,
            physics_ticks_after_dispatch=(
                profile_config.physics_ticks_after_dispatch
            ),
        )
        if allocation_preparation is not None
        else select_sim_hil_application_control_profile(
            config.control_profile_id,
            assembly=SimHILControlProfileAssembly(
                compatibility_session=composition,
                runtime=target.runtime,
                policy_source=policy_source,
                human_source=human_source,
                readiness_provider=readiness_provider,
            ),
            additional_bindings=profile_bindings,
        )
    )
    return MujocoSimHILApplication(
        config=config,
        task_config=target.task_config,
        scene=target.scene,
        runtime=target.runtime,
        composition=control_profile.session,
        policy_source=policy_source,
        human_source=human_source,
        readiness_provider=readiness_provider,
        control_cycle_port=control_profile.control_cycle_port,
        execution_difference_observer=execution_difference_observer,
        target_frame_profile=target.target_frame_profile,
        tool_frame_profile=target.tool_frame_profile,
        gripper_input_selection=gripper_input_selection,
        calibration_binding=calibration_binding,
        task_initialization_info=dict(target.task_initialization_info),
        viewer=viewer,
        runtime_home_pose_desk_by_side_capture=runtime_home_capture,
        human_attachment_factory=build_human_attachment,
    )


__all__ = ["build_application_graph"]
