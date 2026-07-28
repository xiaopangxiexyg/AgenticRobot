from __future__ import annotations

import numpy as np

from control_composition.composition import (
    CompositionContextDraft,
    CompositionInputDraft,
    CompositionSpec,
)
from robot_motion.action import (
    RobotAction,
    project_robot_action_v1_to_component_action,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import MujocoPositionExecutionConfig, MujocoPositionExecutionRig
from sim.mujoco.deploy import (
    DeterministicAsyncTopologyHarness,
    MujocoDeployConfig,
    MujocoDeployRuntime,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[dict[str, np.ndarray]] = []

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        command = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


def build_runtime() -> tuple[MujocoDeployRuntime, FakeScene, ManualClock]:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="async-topology-session",
            control_domain_id="async-topology-domain",
            embodiment_instance_id="async-topology-embodiment",
            enable_composer_authority=True,
            decision_hz=30.0,
        ),
        clock=clock,
    )
    runtime.start()
    for producer_id in ("slow-context", "slow-base", "fast-residual"):
        runtime.activate_policy_node_producer(producer_id)
    return runtime, scene, clock


def composition_spec() -> CompositionSpec:
    return CompositionSpec(
        spec_id="async-base-residual-v1",
        algebra="base_plus_bounded_residual",
        input_roles=("base", "residual"),
        required_roles=("base", "residual"),
        residual_limit_by_component={"left_arm": 0.05, "right_arm": 0.05},
    )


def qpos_action(value: float, *, source: str, timestamp: float) -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, value, dtype=np.float32),
        right_arm=np.full(6, -value, dtype=np.float32),
        timestamp=timestamp,
        source=source,
    )


def context_draft(
    runtime: MujocoDeployRuntime,
    *,
    sequence: int,
    epoch: int,
    produced_at: float,
    effective_until: float,
) -> CompositionContextDraft:
    observation = runtime.capture_policy_observation(
        policy_node_id="slow-vision",
        trigger_reason="async_fixture",
    )
    return CompositionContextDraft(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        producer_id="slow-context",
        producer_epoch=epoch,
        sequence=sequence,
        producer_node_id="slow-vision",
        context_channel="motion-plan",
        context_kind="trajectory_plan",
        semantic_payload={"sequence": sequence},
        produced_at=produced_at,
        produced_clock_domain_id=runtime.config.clock_domain_id,
        effective_from=produced_at,
        effective_until=effective_until,
        effective_clock_domain_id=runtime.config.clock_domain_id,
        payload_contract_hash="async-motion-plan-v1",
        observation_cut_id=observation.record.observation_cut_id,
        policy_output_id=f"slow-context-output-{epoch}-{sequence}",
    )


def input_draft(
    runtime: MujocoDeployRuntime,
    *,
    producer_id: str,
    epoch: int,
    sequence: int,
    role: str,
    value: float,
    context_id: str,
    produced_at: float,
    effective_until: float,
) -> CompositionInputDraft:
    observation = runtime.capture_policy_observation(
        policy_node_id="slow-vision" if role == "base" else "fast-tactile",
        trigger_reason="async_fixture",
    )
    return CompositionInputDraft(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        producer_id=producer_id,
        producer_epoch=epoch,
        sequence=sequence,
        input_role=role,
        action=project_robot_action_v1_to_component_action(
            qpos_action(
                value,
                source=f"async_{role}",
                timestamp=produced_at,
            ),
            action_schema=runtime.action_schema,
        ),
        produced_at=produced_at,
        produced_clock_domain_id=runtime.config.clock_domain_id,
        effective_from=produced_at,
        effective_until=effective_until,
        effective_clock_domain_id=runtime.config.clock_domain_id,
        context_ids=(context_id,),
        observation_cut_id=observation.record.observation_cut_id,
        policy_output_id=f"{producer_id}-output-{epoch}-{sequence}",
    )


def harness(
    runtime: MujocoDeployRuntime,
    clock: ManualClock,
    *,
    capacity: int = 4096,
) -> DeterministicAsyncTopologyHarness:
    return DeterministicAsyncTopologyHarness(
        runtime,
        set_control_time=lambda value: setattr(clock, "value", value),
        start_at=clock.value,
        capacity=capacity,
    )


def test_async_completion_cutoff_restart_and_burst_remain_receiver_admitted() -> None:
    runtime, scene, clock = build_runtime()
    async_runtime = harness(runtime, clock)

    context_job = context_draft(
        runtime,
        sequence=0,
        epoch=1,
        produced_at=10.0,
        effective_until=10.2,
    )
    async_runtime.schedule_composition_context(
        job_id="context-0",
        draft=context_job,
        submitted_at=10.0,
        completed_at=10.005,
    )
    context_release = async_runtime.advance_to(10.005)[0]
    assert context_release.status == "admitted"
    context_id = context_release.result.context_id

    base = input_draft(
        runtime,
        producer_id="slow-base",
        epoch=1,
        sequence=0,
        role="base",
        value=0.1,
        context_id=context_id,
        produced_at=10.006,
        effective_until=10.2,
    )
    residual_0 = input_draft(
        runtime,
        producer_id="fast-residual",
        epoch=1,
        sequence=0,
        role="residual",
        value=0.01,
        context_id=context_id,
        produced_at=10.007,
        effective_until=10.2,
    )
    async_runtime.schedule_composition_input(
        job_id="base-0",
        draft=base,
        submitted_at=10.006,
        completed_at=10.010,
    )
    async_runtime.schedule_composition_input(
        job_id="residual-0",
        draft=residual_0,
        submitted_at=10.007,
        completed_at=10.020,
        deadline_at=10.015,
    )
    async_runtime.schedule_composition(
        job_id="compose-0",
        spec=composition_spec(),
        target_effective_at=10.025,
        submitted_at=10.0,
        completed_at=10.025,
        required_context_channels=("motion-plan",),
    )
    async_runtime.schedule_decision(
        job_id="decision-0",
        decision_at=10.025,
        physics_ticks_after_dispatch=1,
    )
    first_releases = async_runtime.advance_to(10.025)
    first_step = next(item.result for item in first_releases if item.job_id == "decision-0")
    assert first_step.status == "dispatched"
    np.testing.assert_allclose(
        first_step.decision.selected_action.value("left_arm").values,
        np.full(6, 0.11),
    )
    missed = next(item for item in first_releases if item.job_id == "residual-0")
    assert missed.deadline_missed
    assert missed.status == "admitted"

    async_runtime.schedule_producer_restart(
        job_id="restart-fast",
        producer_id="fast-residual",
        producer_epoch=2,
        submitted_at=10.026,
        completed_at=10.030,
        reason="synthetic_worker_restart",
    )
    stale_old_epoch = input_draft(
        runtime,
        producer_id="fast-residual",
        epoch=1,
        sequence=1,
        role="residual",
        value=0.02,
        context_id=context_id,
        produced_at=10.026,
        effective_until=10.2,
    )
    async_runtime.schedule_composition_input(
        job_id="old-epoch-late",
        draft=stale_old_epoch,
        submitted_at=10.026,
        completed_at=10.031,
    )
    for sequence, (completed_at, value) in enumerate(
        ((10.040, 0.021), (10.045, 0.022), (10.050, 0.023)),
    ):
        draft = input_draft(
            runtime,
            producer_id="fast-residual",
            epoch=2,
            sequence=sequence,
            role="residual",
            value=value,
            context_id=context_id,
            produced_at=completed_at - 0.002,
            effective_until=10.2,
        )
        async_runtime.schedule_composition_input(
            job_id=f"new-residual-{sequence}",
            draft=draft,
            submitted_at=completed_at - 0.002,
            completed_at=completed_at,
        )
    async_runtime.schedule_composition(
        job_id="compose-1",
        spec=composition_spec(),
        target_effective_at=10.060,
        submitted_at=10.026,
        completed_at=10.060,
        required_context_channels=("motion-plan",),
    )
    async_runtime.schedule_decision(
        job_id="decision-1",
        decision_at=10.060,
        physics_ticks_after_dispatch=1,
    )
    second_releases = async_runtime.advance_to(10.060)
    old_release = next(item for item in second_releases if item.job_id == "old-epoch-late")
    assert old_release.status == "rejected"
    assert old_release.result.rejection_code == "inactive_producer_epoch"
    second_step = next(item.result for item in second_releases if item.job_id == "decision-1")
    assert second_step.status == "dispatched"
    np.testing.assert_allclose(
        second_step.decision.selected_action.value("left_arm").values,
        np.full(6, 0.123),
    )
    assert len(scene.physics_commands) == 2
    decisions = [
        item.result.decision
        for item in async_runtime.releases
        if item.job_kind == "decision" and item.result.decision is not None
    ]
    assert len(decisions) == 2
    assert decisions[1].published_at - decisions[0].published_at >= 1.0 / 30.0


def test_stale_context_blocks_composite_proposal_at_one_causal_cut() -> None:
    runtime, scene, clock = build_runtime()
    async_runtime = harness(runtime, clock)
    context_job = context_draft(
        runtime,
        sequence=0,
        epoch=1,
        produced_at=10.0,
        effective_until=10.030,
    )
    async_runtime.schedule_composition_context(
        job_id="context-short",
        draft=context_job,
        submitted_at=10.0,
        completed_at=10.001,
    )
    context_release = async_runtime.advance_to(10.001)[0]
    context_id = context_release.result.context_id
    for role, producer_id, value in (
        ("base", "slow-base", 0.1),
        ("residual", "fast-residual", 0.01),
    ):
        draft = input_draft(
            runtime,
            producer_id=producer_id,
            epoch=1,
            sequence=0,
            role=role,
            value=value,
            context_id=context_id,
            produced_at=10.002,
            effective_until=10.2,
        )
        async_runtime.schedule_composition_input(
            job_id=f"{role}-short-context",
            draft=draft,
            submitted_at=10.002,
            completed_at=10.003,
        )
    async_runtime.schedule_composition(
        job_id="compose-after-context-expiry",
        spec=composition_spec(),
        target_effective_at=10.060,
        submitted_at=10.002,
        completed_at=10.060,
        required_context_channels=("motion-plan",),
    )
    async_runtime.schedule_decision(job_id="decision-no-context", decision_at=10.060)

    releases = async_runtime.advance_to(10.060)
    composition = next(
        item.result for item in releases if item.job_id == "compose-after-context-expiry"
    )
    assert composition.composition_evaluation.output_status == "no_action"
    assert composition.composition_evaluation.reason == "missing_context:motion-plan"
    assert composition.proposal_admission is None
    assert len(runtime.core.proposals) == 0
    decision = next(item.result for item in releases if item.job_id == "decision-no-context")
    assert decision.status == "execution_cancelled"
    assert decision.cancellation is not None
    assert scene.physics_commands == []


def test_hold_between_composition_and_decision_prevents_dispatch() -> None:
    runtime, scene, clock = build_runtime()
    async_runtime = harness(runtime, clock)
    context_job = context_draft(
        runtime,
        sequence=0,
        epoch=1,
        produced_at=10.0,
        effective_until=10.2,
    )
    async_runtime.schedule_composition_context(
        job_id="context",
        draft=context_job,
        submitted_at=10.0,
        completed_at=10.001,
    )
    context_id = async_runtime.advance_to(10.001)[0].result.context_id
    for role, producer_id, value in (
        ("base", "slow-base", 0.1),
        ("residual", "fast-residual", 0.01),
    ):
        async_runtime.schedule_composition_input(
            job_id=role,
            draft=input_draft(
                runtime,
                producer_id=producer_id,
                epoch=1,
                sequence=0,
                role=role,
                value=value,
                context_id=context_id,
                produced_at=10.002,
                effective_until=10.2,
            ),
            submitted_at=10.002,
            completed_at=10.010,
        )
    async_runtime.schedule_composition(
        job_id="compose",
        spec=composition_spec(),
        target_effective_at=10.030,
        submitted_at=10.002,
        completed_at=10.020,
        required_context_channels=("motion-plan",),
    )
    async_runtime.schedule_execution_gate(
        job_id="hold",
        status="hold",
        submitted_at=10.021,
        completed_at=10.021,
        reason="synthetic_cancel_during_composition",
        source_event_ids=("synthetic-hold",),
    )
    async_runtime.schedule_decision(job_id="decision", decision_at=10.030)

    releases = async_runtime.advance_to(10.030)
    assert next(item for item in releases if item.job_id == "compose").status == "ready"
    decision = next(item.result for item in releases if item.job_id == "decision")
    assert decision.status == "execution_hold_active"
    assert scene.physics_commands == []


def test_async_queue_capacity_rejects_without_dropping_existing_job() -> None:
    runtime, _scene, clock = build_runtime()
    async_runtime = harness(runtime, clock, capacity=1)
    first = async_runtime.schedule_decision(job_id="first", decision_at=10.01)
    second = async_runtime.schedule_decision(job_id="second", decision_at=10.02)
    assert first.accepted
    assert not second.accepted
    assert second.rejection_code == "capacity_exhausted"
    releases = async_runtime.run_all()
    assert [item.job_id for item in releases] == ["first"]
