from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import control_station.sim_hil as sim_hil_module
from control_station.sim_hil import (
    MujocoSimHILApplicationConfig,
    _build_dry_run_composition_contracts,
    _config_from_args,
    build_arg_parser,
)
from control_station.composition.mujoco_policy_human_allocation_operations import (
    MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER,
    MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
    MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
    MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR,
    MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY,
)
from control_station.sim_hil_profile.policy_human_allocation import (
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
    SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
    prepare_exclusive_truncated_hitl_allocation_profile,
    prepare_policy_human_allocation_scheme_a,
    prepare_policy_human_allocation_scheme_b,
)
from control_station.sim_hil_profile.profile_selection import (
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
    SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    HandActionSpec,
    RobotAction,
    SourceValidationResult,
)
from robot_motion.coordinate_frames import CoordinateFrameGraph, CoordinateFrameSpec
from robot_motion.deploy import NormalizedAction, PolicyActionEmission
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from sim.mujoco.deploy import capture_mujoco_sim_hil_tool_tcp_pose_by_side
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from tests.test_control_station_policy_human_residual_profile import _schema
from tests.test_control_station_sim_hil_device_adapters import (
    _install_local_virtual_device_io,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_scheme_a_preparation_is_resource_free_and_component_keyed() -> None:
    schema = _schema()
    preparation = prepare_policy_human_allocation_scheme_a(
        action_schema=schema,
        target_frame_graph=CoordinateFrameGraph(
            graph_id="virtual-sim-hil-target-frames",
            frames=(CoordinateFrameSpec("arm/base"),),
        ),
        target_frame_profile_hash="target-frame-profile-hash",
        tool_frame_profile_hash="tool-frame-profile-hash",
        composer_producer_id="virtual-composer",
    )

    bundle = preparation.bundle
    assert bundle.config.enabled
    assert bundle.config.profile_id == (
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
    )
    assert bundle.composition_spec is bundle.composition_contracts.composition_spec
    assert set(bundle.input_action_schema_by_role) == {
        "policy",
        "human_absolute",
        "human_residual",
    }
    assert bundle.config.require_human_roles_at_start is False
    assert bundle.composition_spec.required_roles == ("policy",)
    assert bundle.information_plan.required_context_channels == ()
    assert bundle.information_plan.optional_context_channels == (
        bundle.config.allocation_context_channel,
    )
    assert (
        bundle.information_plan.context_requirements[0].maximum_age_sec
        == pytest.approx(0.5)
    )
    assert {"arm_base", "tool_tcp"}.issubset(
        bundle.input_contracts.residual_adapter.frame_graph.frame_by_id
    )
    assert preparation.station_profiles.proposal.send_actions
    assert preparation.station_profiles.proposal.promotable_allocation_modes == (
        "base_only",
        "relative_overlay",
    )
    assert all(
        policy.automatic_mode_ceiling == "relative_overlay"
        for policy in bundle.allocation_spec.component_policies
    )
    pose_component = next(
        component_id
        for component_id in schema.component_ids
        if schema.payload_schema_by_component[
            component_id
        ].representation_family
        == "cartesian_pose"
    )
    assert bundle.evidence_window.activity_normalization_scale_by_component[
        pose_component
    ]["rotation_displacement_rad"] == pytest.approx(np.deg2rad(10.0))
    assert bundle.evidence_window.residual_normalization_scale_by_component[
        pose_component
    ]["rotation_from_anchor_rad"] == pytest.approx(np.deg2rad(10.0))
    assert bundle.allocation_spec.transition_continuity_policy_hash is None


def test_scheme_b_preparation_enables_three_state_target_continuity() -> None:
    schema = _schema()
    preparation = prepare_policy_human_allocation_scheme_b(
        action_schema=schema,
        target_frame_graph=CoordinateFrameGraph(
            graph_id="virtual-sim-hil-target-frames",
            frames=(CoordinateFrameSpec("arm/base"),),
        ),
        target_frame_profile_hash="target-frame-profile-hash",
        tool_frame_profile_hash="tool-frame-profile-hash",
        composer_producer_id="virtual-composer",
    )

    bundle = preparation.bundle
    assert bundle.config.profile_id == (
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B
    )
    assert preparation.station_profiles.proposal.promotable_allocation_modes == (
        "base_only",
        "relative_overlay",
        "absolute_overlay_owner",
    )
    assert all(
        policy.automatic_mode_ceiling == "absolute_overlay_owner"
        for policy in bundle.allocation_spec.component_policies
    )
    assert bundle.composition_spec.required_roles == ("policy",)
    assert bundle.allocation_spec.transition_continuity_policy_hash is not None


def test_exclusive_takeover_is_a_discrete_streaming_allocation_profile() -> None:
    schema = _schema()
    preparation = prepare_exclusive_truncated_hitl_allocation_profile(
        action_schema=schema,
        target_frame_graph=CoordinateFrameGraph(
            graph_id="virtual-exclusive-hitl-target-frames",
            frames=(CoordinateFrameSpec("arm/base"),),
        ),
        target_frame_profile_hash="target-frame-profile-hash",
        tool_frame_profile_hash="tool-frame-profile-hash",
        composer_producer_id="virtual-composer",
    )

    bundle = preparation.bundle
    assert bundle.config.profile_id == (
        SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
    )
    assert bundle.config.automatic_override_enabled is False
    assert bundle.config.rebase_operation_enabled is False
    assert bundle.config.initial_override_mode == "base_only"
    assert bundle.config.persistent_override_selection_lease_sec == pytest.approx(
        0.5
    )
    assert bundle.config.require_human_roles_at_start is False
    assert preparation.station_profiles.proposal.promotable_allocation_modes == (
        "base_only",
        "absolute_overlay_owner",
    )
    assert bundle.composition_spec.required_roles == ()


def test_cli_keeps_fixed_alpha_default_and_allocation_is_explicit() -> None:
    default_config = _config_from_args(build_arg_parser().parse_args([]))
    allocation_config = _config_from_args(
        build_arg_parser().parse_args(
            [
                "--control-profile",
                SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A,
            ]
        )
    )
    three_state_config = _config_from_args(
        build_arg_parser().parse_args(
            [
                "--control-profile",
                SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B,
            ]
        )
    )

    assert default_config.control_profile_id == (
        SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY
    )
    assert allocation_config.control_profile_id == (
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
    )
    assert three_state_config.control_profile_id == (
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B
    )


def test_scheme_a_dry_run_uses_selected_allocation_contracts() -> None:
    schema = _schema()
    contracts, preparation = _build_dry_run_composition_contracts(
        MujocoSimHILApplicationConfig(
            control_profile_id=(
                SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
            )
        ),
        action_schema=schema,
        prepared_target=SimpleNamespace(
            target_frame_profile=SimpleNamespace(
                frame_graph=CoordinateFrameGraph(
                    graph_id="virtual-dry-run-target-frames",
                    frames=(
                        CoordinateFrameSpec("arm_base"),
                        CoordinateFrameSpec("tool_tcp"),
                    ),
                ),
                semantic_content_hash="virtual-dry-run-target-profile",
            ),
            tool_frame_profile=SimpleNamespace(
                semantic_content_hash="virtual-dry-run-tool-profile"
            ),
        ),
    )

    assert preparation is not None
    assert contracts is preparation.bundle
    assert contracts.config.profile_id == (
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
    )
    assert contracts.config.require_human_roles_at_start is False
    assert contracts.composition_spec.required_roles == ("policy",)
    assert preparation.station_profiles.proposal.promotable_allocation_modes == (
        "base_only",
        "relative_overlay",
    )


def test_scheme_b_dry_run_uses_three_state_contracts() -> None:
    schema = _schema()
    contracts, preparation = _build_dry_run_composition_contracts(
        MujocoSimHILApplicationConfig(
            control_profile_id=(
                SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B
            )
        ),
        action_schema=schema,
        prepared_target=SimpleNamespace(
            target_frame_profile=SimpleNamespace(
                frame_graph=CoordinateFrameGraph(
                    graph_id="virtual-dry-run-target-frames",
                    frames=(
                        CoordinateFrameSpec("arm_base"),
                        CoordinateFrameSpec("tool_tcp"),
                    ),
                ),
                semantic_content_hash="virtual-dry-run-target-profile",
            ),
            tool_frame_profile=SimpleNamespace(
                semantic_content_hash="virtual-dry-run-tool-profile"
            ),
        ),
    )

    assert preparation is not None
    assert contracts is preparation.bundle
    assert contracts.config.profile_id == (
        SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B
    )
    assert contracts.allocation_spec.transition_continuity_policy_hash is not None
    assert preparation.station_profiles.proposal.promotable_allocation_modes == (
        "base_only",
        "relative_overlay",
        "absolute_overlay_owner",
    )


class _VirtualAllocationPolicySource:
    """Deterministic one-step policy using the production emission seam."""

    maximum_hold_sec = 0.25
    policy_period_sec = 1.0 / 15.0

    def __init__(
        self,
        scene,
        *,
        task_config_path,
        policy_node_id,
        policy_clock_domain_id,
        **_kwargs,
    ) -> None:
        self.scene = scene
        self.task_config = CubeLiftTaskConfig.load(task_config_path)
        self.policy_node_id = str(policy_node_id)
        self.clock_domain_id = str(policy_clock_domain_id)
        self.started = False
        self.last_emission = None
        self.last_error = None
        self.sequence = 0

    @property
    def ready(self) -> bool:
        return self.started and self.last_error is None

    def start(self) -> None:
        self.started = True
        self.reset_for_activation()

    def reset_for_activation(self) -> None:
        self.last_emission = None

    def close(self) -> None:
        self.started = False
        self.last_emission = None

    def step(self, *, now: float, step_index: int | None = None):
        del step_index
        if not self.started:
            raise RuntimeError("Virtual policy source is not started.")
        pose = pose_matrix_to_xyzquat(
            capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                self.scene,
                self.task_config,
            )["right"]
        )
        assert pose is not None
        hand_spec = HandActionSpec.dmtac_gripper("right")
        hand_qpos = np.zeros(hand_spec.slot_count, dtype=np.float32)
        hand_qpos[hand_spec.slot_map["gripper_close"]] = float(
            self.scene.get_gripper_close(self.task_config.right_gripper_name)
        )
        action = RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            arm_pose_by_side={"right": pose},
            hand_qpos_by_side={"right": hand_qpos},
            active_effectors=("right_arm", "right_hand"),
            frame_by_effector={"right_arm": "right_arm_base"},
            controlled_frame_by_effector={"right_arm": "tool_tcp"},
            source="virtual-allocation-policy",
            metadata={
                "hand_action_specs_by_side": {
                    "right": hand_spec.to_metadata()
                }
            },
        )
        sequence = self.sequence
        self.sequence += 1
        emission = PolicyActionEmission(
            emission_id=f"virtual-allocation-policy:{sequence}",
            admission_mode="on_emission_pop_v1",
            policy_node_id=self.policy_node_id,
            inference_id=f"virtual-inference:{sequence}",
            policy_output_id=f"virtual-output:{sequence}",
            chunk_id=sequence,
            chunk_keyframe_index=0,
            chunk_size=1,
            decoded_absolute_action=NormalizedAction(
                vector=np.zeros(26, dtype=np.float32)
            ),
            pre_safety_robot_action=action,
            source_validation=SourceValidationResult(accepted=True),
            representation_receipt=None,
            pre_safety_faults=(),
            produced_at=now,
            produced_clock_domain_id=self.clock_domain_id,
            emitted_at=now,
            target_effective_at=now,
            effective_from=now,
            effective_until=now + self.maximum_hold_sec,
            effective_clock_domain_id=self.clock_domain_id,
            schedule_spec_id="virtual-allocation-schedule",
            schedule_spec_hash="virtual-allocation-schedule-hash",
            queue_len_before=1,
            queue_len_after=0,
            queue_pop_count=1,
            semantic_content_hash=f"virtual-allocation-emission-hash:{sequence}",
        )
        self.last_emission = emission
        return emission

    def status(self, *, now: float | None = None):
        return {
            "source_id": "virtual-allocation-policy",
            "started": self.started,
            "ready": self.ready,
            "last_emission_age_sec": (
                None
                if now is None or self.last_emission is None
                else float(now) - self.last_emission.emitted_at
            ),
            "last_error": self.last_error,
        }


@pytest.mark.slow_sim
def test_allocation_policy_only_start_does_not_open_human_devices(
    monkeypatch,
) -> None:
    def reject_human_source(**_kwargs):
        raise AssertionError("policy-only allocation touched a human device")

    monkeypatch.setattr(
        sim_hil_module,
        "_build_passive_single_vive_gripper_component_source",
        reject_human_source,
    )
    monkeypatch.setattr(
        "sim.mujoco_temp_pipeline.hil_policy.TempPPOHILPolicySource",
        _VirtualAllocationPolicySource,
    )
    app = sim_hil_module._build_application(
        MujocoSimHILApplicationConfig(
            enabled=True,
            control_profile_id=(
                SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A
            ),
            defer_human_input=True,
            vive_coordinate_profile_path=None,
            device="cpu",
            realtime_worker=False,
            mujoco_viewer_enabled=False,
            compose_target_hz=15.0,
        )
    )
    try:
        assert app.human_input_attached is False
        assert app.composition.required_readiness_checks() == (
            "policy_source_ready",
            "policy_chunk_ttl_ready",
            "measured_state_ready",
        )
        report = app.start()
        assert report.passed
        assert app.status()["source_configuration"][
            "human_input_attached"
        ] is False

        admitted = None
        publication_diagnostics = []
        for _ in range(10):
            due = app._next_decision_at
            if due is not None:
                time.sleep(max(0.0, due - app.clock()))
            candidate = app.step()
            publication_diagnostics.append(
                (
                    candidate.publication.proposal_publication.status,
                    candidate.publication.proposal_publication.submission
                    .composition_evaluation.output_status,
                    candidate.publication.proposal_publication.submission
                    .composition_evaluation.reason,
                    candidate.publication.proposal_publication
                    .selected_mode_by_component,
                )
            )
            if candidate.publication.proposal_publication.status == (
                "proposal_admitted"
            ):
                admitted = candidate
                break
        assert admitted is not None, publication_diagnostics
        assert set(
            admitted.publication.proposal_publication
            .selected_mode_by_component.values()
        ) == {"base_only"}
    finally:
        app.close()


@pytest.mark.parametrize(
    ("profile_id", "supports_absolute"),
    (
        (SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_A, False),
        (SIM_HIL_CONTROL_PROFILE_POLICY_HUMAN_ALLOCATION_SCHEME_B, True),
    ),
)
@pytest.mark.slow_sim
def test_allocation_profiles_virtual_policy_and_human_run_button_operations(
    monkeypatch,
    profile_id: str,
    supports_absolute: bool,
) -> None:
    """Exercise MuJoCo plus concrete human adapters without device or model I/O."""

    concrete_human_builder = (
        sim_hil_module._build_passive_single_vive_gripper_component_source
    )
    local_human_sources: list[Any] = []
    local_human_lifecycles: list[dict[str, int]] = []

    def local_human_builder(**kwargs):
        source, source_spec = concrete_human_builder(**kwargs)
        local_human_lifecycles.append(
            _install_local_virtual_device_io(
                monkeypatch,
                source,
                input_kind=kwargs["source_config"].gripper_input_source,
            )
        )
        local_human_sources.append(source)
        return source, source_spec

    monkeypatch.setattr(
        sim_hil_module,
        "_build_passive_single_vive_gripper_component_source",
        local_human_builder,
    )
    monkeypatch.setattr(
        "sim.mujoco_temp_pipeline.hil_policy.TempPPOHILPolicySource",
        _VirtualAllocationPolicySource,
    )
    app = sim_hil_module._build_application(
        MujocoSimHILApplicationConfig(
            enabled=True,
            control_profile_id=profile_id,
            gripper_input_source="angle_trigger",
            device="cpu",
            realtime_worker=False,
            mujoco_viewer_enabled=False,
            warmup_timeout_sec=1.0,
            human_emission_timeout_sec=1.0,
            compose_target_hz=15.0,
        )
    )
    try:
        def step_when_due():
            due = app._next_decision_at
            if due is not None:
                time.sleep(max(0.0, due - app.clock()))
            return app.step()

        report = app.start()
        assert report.passed
        assert len(local_human_sources) == 1
        assert local_human_sources[0].session.rig is None
        assert [item.name for item in local_human_sources[0].session.sources] == [
            "vive_pose",
            "angle_trigger",
        ]
        assert local_human_lifecycles == [
            {
                "vive_start": 1,
                "hand_start": 1,
                "vive_close": 0,
                "hand_close": 0,
            }
        ]
        assert app.status()["control_profile_id"] == profile_id

        first_cycle = step_when_due()
        assert first_cycle.publication.proposal_publication.status in {
            "no_action",
            "proposal_admitted",
        }
        if first_cycle.publication.proposal_publication.status == "no_action":
            # The allocation producer is intentionally asynchronous. Until
            # its required context receipt is visible, the unchanged lower
            # missing-action behavior cancels/holds instead of inventing a
            # source choice.
            assert first_cycle.execution.status == "execution_cancelled"

        admitted_cycle = (
            first_cycle
            if first_cycle.publication.proposal_publication.status
            == "proposal_admitted"
            else None
        )
        for _ in range(10):
            if admitted_cycle is not None:
                break
            candidate = step_when_due()
            if candidate.publication.proposal_publication.status == (
                "proposal_admitted"
            ):
                admitted_cycle = candidate
        assert admitted_cycle is not None
        assert admitted_cycle.execution.status in {
            "dispatched",
            "execution_kept_existing",
            "not_due",
        }
        if not supports_absolute:
            with pytest.raises(ValueError, match="Unsupported allocation button"):
                app.submit_allocation_button_operation(
                    signal_name=(
                        MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER
                    ),
                    event_id="virtual-button:absolute-not-supported",
                    source_fact_id="virtual-button-fact:absolute-not-supported",
                    source_fact_hash=(
                        "virtual-button-fact-hash:absolute-not-supported"
                    ),
                    occurred_at=app.clock(),
                )

        signals = [
            MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
            MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY,
        ]
        if supports_absolute:
            signals.append(MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER)
        signals.extend(
            (
                MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
                MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR,
            )
        )
        receipts = []
        for index, signal in enumerate(signals):
            receipt = app.submit_allocation_button_operation(
                signal_name=signal,
                event_id=f"virtual-button:{index}",
                source_fact_id=f"virtual-button-fact:{index}",
                source_fact_hash=f"virtual-button-fact-hash:{index}",
                occurred_at=app.clock(),
            )
            receipts.append(receipt)
            if signal == MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER:
                absolute_cycle = None
                for _ in range(10):
                    candidate = step_when_due()
                    publication = candidate.publication.proposal_publication
                    if publication.status == "proposal_admitted" and set(
                        publication.selected_mode_by_component.values()
                    ) == {"absolute_overlay_owner"}:
                        absolute_cycle = candidate
                        break
                assert absolute_cycle is not None
                evaluation = (
                    absolute_cycle.publication.proposal_publication.submission
                    .composition_evaluation
                )
                assert "allocation_transition_continuity" in (
                    evaluation.strategy_result_data
                )
        assert all(receipt.receipt_id for receipt in receipts)
        assert receipts[-1].anchor_request is not None
        operation_status = app.status()["composition"]["allocation_operations"]
        assert (
            MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER
            in operation_status["supported_signals"]
        ) is supports_absolute
        assert operation_status["allocation_override_mode"] == "automatic"
        assert operation_status["last_operation_signal"] == (
            MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR
        )

        post_rebase_cycle = step_when_due()
        assert post_rebase_cycle.human_submission.accepted
        assert post_rebase_cycle.human_submission.input_cycle.derivation.captured_anchor
        if post_rebase_cycle.execution.status == "execution_cancelled":
            # Rebase publishes a new anchor generation before the asynchronous
            # allocation result for that generation is available.  The exact
            # rebase cycle therefore remains a valid no-action/HOLD cut.
            assert post_rebase_cycle.publication.proposal_publication.status == (
                "no_action"
            )
            post_rebase_cycle = None
            for _ in range(10):
                candidate = step_when_due()
                if candidate.publication.proposal_publication.status == (
                    "proposal_admitted"
                ):
                    post_rebase_cycle = candidate
                    break
            assert post_rebase_cycle is not None
        assert post_rebase_cycle.execution.status in {
            "dispatched",
            "execution_kept_existing",
            "not_due",
        }
    finally:
        app.close()
    assert local_human_lifecycles == [
        {
            "vive_start": 1,
            "hand_start": 1,
            "vive_close": 1,
            "hand_close": 1,
        }
    ]
