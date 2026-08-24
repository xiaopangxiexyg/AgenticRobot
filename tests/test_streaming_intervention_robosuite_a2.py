from __future__ import annotations

import pytest

from experiments.sim_learning.streaming_intervention import (
    A2_COMPONENT_IDS,
    A2_POLICY_SOURCE_KIND_RTC,
    A2_POLICY_SOURCE_KIND_STANDARD,
    ComponentOwnershipA2,
    DeterministicRobosuiteMockBackend,
    PolicyInferenceRequestA2,
    PolicyProposalA2,
    RobosuiteComponentComposerA2,
    SequencePolicySourceA2,
    StreamingInterventionControlFrameA2,
    StreamingInterventionPolicyRunnerA2,
)
from sim.robosuite import RobosuiteResetManifest


def _manifest(*, max_episode_steps: int = 8) -> RobosuiteResetManifest:
    return RobosuiteResetManifest(
        environment_id="ReplayFake",
        environment_contract_hash="replay-fake-contract-a2",
        task_id="streaming-a2",
        episode_id="episode-a2",
        reset_sequence=0,
        seed=17,
        action_dimension=7,
        observation_keys=("state", "step_index", "episode_seed"),
        max_episode_steps=max_episode_steps,
    )


def _composer() -> RobosuiteComponentComposerA2:
    return RobosuiteComponentComposerA2(
        action_low=(-1.0,) * 7,
        action_high=(1.0,) * 7,
    )


def test_component_composer_makes_gripper_intervention_non_additive() -> None:
    selection = _composer().compose(
        mode="relative_overlay",
        base_action=(0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8),
        intervention_action=(0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7),
        ownership=ComponentOwnershipA2.for_components(A2_COMPONENT_IDS),
    )

    assert selection.preclip_action[:6] == pytest.approx(
        (1.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert selection.preclip_action[6] == 0.7
    assert selection.applied_action[:6] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert selection.applied_action[6] == 0.7
    assert selection.clipped_dimensions == (0,)


def test_standard_and_rtc_sources_share_a_revisioned_contract() -> None:
    request = PolicyInferenceRequestA2(
        episode_id="episode-a2",
        frame_id="frame-0",
        step_index=0,
        observation={"state": (0.0,), "step_index": 0},
    )
    standard = SequencePolicySourceA2(
        source_id="policy-standard",
        source_kind=A2_POLICY_SOURCE_KIND_STANDARD,
        actions=((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2),),
        revision=4,
    )
    rtc = SequencePolicySourceA2(
        source_id="policy-rtc",
        source_kind=A2_POLICY_SOURCE_KIND_RTC,
        actions=((0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9),),
        revision=9,
    )

    standard_proposal = standard.infer(
        PolicyInferenceRequestA2(
            episode_id=request.episode_id,
            frame_id=request.frame_id,
            step_index=request.step_index,
            observation=request.observation,
            expected_source_revision=standard.revision,
        )
    )
    rtc_proposal = rtc.infer(
        PolicyInferenceRequestA2(
            episode_id=request.episode_id,
            frame_id=request.frame_id,
            step_index=request.step_index,
            observation=request.observation,
            nominal_action=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4),
            expected_source_revision=rtc.revision,
        )
    )

    assert standard_proposal.action_role == "nominal"
    assert standard_proposal.revision == 4
    assert rtc_proposal.action_role == "intervention"
    assert rtc_proposal.revision == 9
    with pytest.raises(ValueError, match="nominal_action"):
        rtc.infer(request)


def test_standard_policy_run_covers_base_takeover_streaming_and_lineage() -> None:
    policy = SequencePolicySourceA2(
        source_id="policy-standard",
        source_kind=A2_POLICY_SOURCE_KIND_STANDARD,
        actions=(
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2),
            (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3),
            (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4),
        ),
        revision=2,
    )
    frames = (
        StreamingInterventionControlFrameA2(
            frame_id="base-only",
            mode="base_only",
        ),
        StreamingInterventionControlFrameA2(
            frame_id="takeover",
            mode="absolute_overlay_owner",
            human_raw_action=(0.0,) * 7,
            human_mapped_action=(
                0.6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
            ),
            human_event_id="takeover-1",
        ),
        StreamingInterventionControlFrameA2(
            frame_id="streaming-arm-only",
            mode="relative_overlay",
            human_raw_action=(0.0,) * 7,
            human_mapped_action=(
                0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.9,
            ),
            human_component_ids=("arm",),
            human_event_id="stream-1",
        ),
    )

    result = StreamingInterventionPolicyRunnerA2(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
        policy_source=policy,
    ).run(manifest=_manifest(), frames=frames)

    assert result.stop_reason == "input_exhausted"
    assert result.steps[0].selection.applied_action == (
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.2,
    )
    assert result.steps[1].selection.applied_action == (
        0.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.8,
    )
    assert result.steps[2].selection.preclip_action == (
        0.7,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
    )
    evidence = result.steps[2].evidence
    assert evidence.a_policy == policy.actions[2]
    assert evidence.a_rtc is None
    assert evidence.human_raw_action == (0.0,) * 7
    assert evidence.human_mapped_action[-1] == 0.9
    assert evidence.active_component_ids == ("arm",)
    assert evidence.ownership_hash == result.steps[2].selection.ownership.semantic_content_hash
    assert evidence.observation_hash
    assert evidence.policy_source_id == "policy-standard"
    assert evidence.policy_source_revision == 2
    assert evidence.policy_inference_id is not None
    assert evidence.a_safe == evidence.a_exec


def test_ordinary_teleop_base_mode_runs_without_a_policy_source() -> None:
    frame = StreamingInterventionControlFrameA2(
        frame_id="teleop-only",
        mode="base_only",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=(0.2, -0.1, 0.0, 0.0, 0.0, 0.0, -0.5),
        human_event_id="teleop-1",
    )

    result = StreamingInterventionPolicyRunnerA2(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
    ).run(manifest=_manifest(max_episode_steps=2), frames=(frame,))

    step = result.steps[0]
    assert step.selection.applied_action == frame.human_mapped_action
    assert step.evidence.a_policy is None
    assert step.evidence.a_rtc is None
    assert step.evidence.active_component_ids == ()
    assert step.evidence.human_mapped_action == frame.human_mapped_action


def test_rtc_policy_is_the_only_intervention_source_and_gripper_stays_base() -> None:
    rtc = SequencePolicySourceA2(
        source_id="policy-rtc",
        source_kind=A2_POLICY_SOURCE_KIND_RTC,
        actions=((0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9),),
    )
    frame = StreamingInterventionControlFrameA2(
        frame_id="rtc",
        mode="relative_overlay",
        base_action=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6),
    )

    result = StreamingInterventionPolicyRunnerA2(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
        policy_source=rtc,
    ).run(manifest=_manifest(max_episode_steps=2), frames=(frame,))

    assert result.steps[0].selection.preclip_action == (
        0.9,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.6,
    )
    assert result.steps[0].evidence.a_policy is None
    assert result.steps[0].evidence.a_rtc == rtc.actions[0]

    conflicting = StreamingInterventionControlFrameA2(
        frame_id="rtc-human-conflict",
        mode="relative_overlay",
        base_action=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6),
        human_raw_action=(0.0,) * 7,
        human_mapped_action=(0.1,) * 7,
        human_event_id="human-1",
    )
    with pytest.raises(ValueError, match="RTC and human"):
        StreamingInterventionPolicyRunnerA2(
            composer=_composer(),
            backend_factory=DeterministicRobosuiteMockBackend,
            policy_source=rtc,
        ).run(manifest=_manifest(), frames=(conflicting,))


class _StaleProposalSource:
    source_id = "stale-source"
    source_kind = A2_POLICY_SOURCE_KIND_STANDARD
    action_role = "nominal"
    action_dimension = 7
    component_ids = A2_COMPONENT_IDS
    revision = 3

    def infer(self, request: PolicyInferenceRequestA2) -> PolicyProposalA2:
        return PolicyProposalA2(
            source_id=self.source_id,
            source_kind=self.source_kind,
            action_role=self.action_role,
            revision=2,
            inference_id="stale",
            frame_id=request.frame_id,
            step_index=request.step_index,
            observation_hash=request.observation_hash,
            action=(0.0,) * 7,
            component_ids=self.component_ids,
        )


def test_stale_policy_revision_fails_closed_before_backend_step() -> None:
    frame = StreamingInterventionControlFrameA2(
        frame_id="stale",
        mode="base_only",
    )
    with pytest.raises(RuntimeError, match="does not bind the installed source"):
        StreamingInterventionPolicyRunnerA2(
            composer=_composer(),
            backend_factory=DeterministicRobosuiteMockBackend,
            policy_source=_StaleProposalSource(),
        ).run(manifest=_manifest(), frames=(frame,))


def test_human_raw_mapped_and_event_lineage_are_required_together() -> None:
    with pytest.raises(ValueError, match="paired"):
        StreamingInterventionControlFrameA2(
            frame_id="unpaired",
            mode="absolute_overlay_owner",
            human_raw_action=(0.0,) * 7,
            human_event_id="event",
        )
