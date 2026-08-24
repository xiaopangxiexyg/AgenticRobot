from __future__ import annotations

from collections.abc import Mapping

import pytest

from experiments.sim_learning.robosuite_mixed_control_a3 import (
    A3_COLLECTION_MODE_ORDINARY_TELEOP,
    A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
    A3_COLLECTION_MODE_TAKEOVER_TELEOP,
    A3_POLICY_VARIANT_RTC,
    A3_POLICY_VARIANT_STANDARD,
    RobosuiteMixedControlCaptureRunnerA3,
    make_rtc_policy_source_a3,
    make_standard_policy_source_a3,
)
from experiments.sim_learning.streaming_intervention import (
    A2_POLICY_SOURCE_KIND_RTC,
    DeterministicRobosuiteMockBackend,
    PolicyInferenceRequestA2,
    RobosuiteComponentComposerA2,
    SequencePolicySourceA2,
    StreamingInterventionControlFrameA2,
)
from sim.robosuite import RobosuiteResetManifest


def _manifest(*, max_episode_steps: int = 4) -> RobosuiteResetManifest:
    return RobosuiteResetManifest(
        environment_id="ReplayFake",
        environment_contract_hash="replay-fake-contract-a3",
        task_id="streaming-a3",
        episode_id="episode-a3",
        reset_sequence=0,
        seed=23,
        action_dimension=7,
        observation_keys=("state", "step_index", "episode_seed"),
        max_episode_steps=max_episode_steps,
    )


def _composer() -> RobosuiteComponentComposerA2:
    return RobosuiteComponentComposerA2(
        action_low=(-1.0,) * 7,
        action_high=(1.0,) * 7,
    )


def _action(value: float, gripper: float = -0.5) -> tuple[float, ...]:
    return (value, 0.0, 0.0, 0.0, 0.0, 0.0, gripper)


class _StandardProvider:
    def __init__(self, actions: tuple[tuple[float, ...], ...]) -> None:
        self.actions = actions
        self.observations: list[Mapping[str, object]] = []

    def predict(self, observation: Mapping[str, object]) -> Mapping[str, object]:
        self.observations.append(observation)
        return {"action": self.actions[len(self.observations) - 1]}


class _RtcProvider:
    def __init__(self, action: tuple[float, ...]) -> None:
        self.action = action
        self.requests: list[PolicyInferenceRequestA2] = []

    def infer(self, request: PolicyInferenceRequestA2) -> Mapping[str, object]:
        self.requests.append(request)
        return {"action_chunk": (self.action,)}


def test_standard_callable_source_invokes_predict_and_binds_revision() -> None:
    provider = _StandardProvider((_action(0.2),))
    source = make_standard_policy_source_a3(
        source_id="standard-a3",
        provider=provider,
        revision=4,
    )
    request = PolicyInferenceRequestA2(
        episode_id="episode-a3",
        frame_id="frame-0",
        step_index=0,
        observation={"state": (0.0,), "step_index": 0},
        expected_source_revision=4,
    )

    proposal = source.infer(request)

    assert proposal.action == _action(0.2)
    assert proposal.source_kind == "standard"
    assert proposal.action_role == "nominal"
    assert proposal.revision == 4
    assert provider.observations == [request.observation]
    assert source.semantic_content_hash


def test_rtc_callable_source_receives_nominal_request_and_emits_intervention() -> None:
    provider = _RtcProvider(_action(0.3, gripper=0.8))
    source = make_rtc_policy_source_a3(
        source_id="rtc-a3",
        provider=provider,
        revision=7,
    )
    request = PolicyInferenceRequestA2(
        episode_id="episode-a3",
        frame_id="rtc-frame",
        step_index=0,
        observation={"state": (0.0,), "step_index": 0},
        nominal_action=_action(0.1),
        expected_source_revision=7,
    )

    proposal = source.infer(request)

    assert proposal.action == _action(0.3, gripper=0.8)
    assert proposal.source_kind == A2_POLICY_SOURCE_KIND_RTC
    assert proposal.action_role == "intervention"
    assert provider.requests == [request]


def test_callable_source_rejects_stale_revision_and_malformed_action() -> None:
    source = make_standard_policy_source_a3(
        source_id="standard-a3",
        provider=lambda _observation: {"action": (0.0,) * 6},
        revision=3,
    )
    request = PolicyInferenceRequestA2(
        episode_id="episode-a3",
        frame_id="stale",
        step_index=0,
        observation={"state": (0.0,), "step_index": 0},
        expected_source_revision=2,
    )

    with pytest.raises(RuntimeError, match="stale input"):
        source.infer(request)

    valid_request = PolicyInferenceRequestA2(
        episode_id="episode-a3",
        frame_id="bad-action",
        step_index=0,
        observation={"state": (0.0,), "step_index": 0},
        expected_source_revision=3,
    )
    with pytest.raises(ValueError, match="dimension 7"):
        source.infer(valid_request)


def test_ordinary_teleop_capture_records_reset_and_next_observations() -> None:
    frames = (
        StreamingInterventionControlFrameA2(
            frame_id="teleop-0",
            mode="base_only",
            human_raw_action=(0.0,) * 7,
            human_mapped_action=_action(0.2),
            human_event_id="human-0",
        ),
        StreamingInterventionControlFrameA2(
            frame_id="teleop-1",
            mode="base_only",
            human_raw_action=(0.1,) * 7,
            human_mapped_action=_action(0.3),
            human_event_id="human-1",
        ),
    )

    dataset = RobosuiteMixedControlCaptureRunnerA3(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
    ).run(
        manifest=_manifest(),
        frames=frames,
        collection_mode=A3_COLLECTION_MODE_ORDINARY_TELEOP,
    )

    assert dataset.policy_variant == "none"
    assert len(dataset.rows) == 2
    assert dataset.rows[0].observation != dataset.rows[0].next_observation
    assert dataset.rows[1].observation == dataset.rows[0].next_observation
    assert dataset.rows[0].a_policy is None
    assert dataset.rows[0].a_rtc is None
    assert dataset.rows[0].a_exec == dataset.rows[0].a_safe
    assert dataset.dataset_id.startswith("robosuite-mixed-control-a3:")
    assert dataset.to_payload()["rows"][0]["collection_mode"] == (
        A3_COLLECTION_MODE_ORDINARY_TELEOP
    )


def test_takeover_capture_binds_standard_policy_and_full_human_components() -> None:
    policy = make_standard_policy_source_a3(
        source_id="standard-takeover",
        provider=_StandardProvider((_action(0.1),)),
    )
    frame = StreamingInterventionControlFrameA2(
        frame_id="takeover-0",
        mode="absolute_overlay_owner",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.7, gripper=0.9),
        human_event_id="takeover-0",
    )

    dataset = RobosuiteMixedControlCaptureRunnerA3(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
        policy_source=policy,
    ).run(
        manifest=_manifest(max_episode_steps=2),
        frames=(frame,),
        collection_mode=A3_COLLECTION_MODE_TAKEOVER_TELEOP,
    )

    row = dataset.rows[0]
    assert dataset.policy_variant == A3_POLICY_VARIANT_STANDARD
    assert row.a_policy == _action(0.1)
    assert row.a_rtc is None
    assert row.a_exec == _action(0.7, gripper=0.9)
    assert row.active_component_ids == ("arm", "gripper")
    assert row.policy_source_id == "standard-takeover"


def test_streaming_capture_supports_standard_human_residual() -> None:
    policy = make_standard_policy_source_a3(
        source_id="standard-stream",
        provider=_StandardProvider((_action(0.2, gripper=-0.4),)),
    )
    frame = StreamingInterventionControlFrameA2(
        frame_id="stream-0",
        mode="relative_overlay",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.1, gripper=0.9),
        human_component_ids=("arm",),
        human_event_id="stream-0",
    )

    dataset = RobosuiteMixedControlCaptureRunnerA3(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
        policy_source=policy,
    ).run(
        manifest=_manifest(max_episode_steps=2),
        frames=(frame,),
        collection_mode=A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
    )

    row = dataset.rows[0]
    assert dataset.policy_variant == A3_POLICY_VARIANT_STANDARD
    assert row.a_policy == _action(0.2, gripper=-0.4)
    assert row.a_rtc is None
    assert row.a_mix[0] == pytest.approx(0.3)
    assert row.a_mix[6] == pytest.approx(-0.4)
    assert row.active_component_ids == ("arm",)


def test_streaming_capture_supports_rtc_intervention_without_human_source() -> None:
    rtc = make_rtc_policy_source_a3(
        source_id="rtc-stream",
        provider=_RtcProvider(_action(0.2, gripper=0.9)),
        revision=2,
    )
    frame = StreamingInterventionControlFrameA2(
        frame_id="rtc-stream-0",
        mode="relative_overlay",
        base_action=_action(0.4, gripper=-0.6),
    )

    dataset = RobosuiteMixedControlCaptureRunnerA3(
        composer=_composer(),
        backend_factory=DeterministicRobosuiteMockBackend,
        policy_source=rtc,
    ).run(
        manifest=_manifest(max_episode_steps=2),
        frames=(frame,),
        collection_mode=A3_COLLECTION_MODE_STREAMING_MIXED_CONTROL,
    )

    row = dataset.rows[0]
    assert dataset.policy_variant == A3_POLICY_VARIANT_RTC
    assert row.a_policy is None
    assert row.a_rtc == _action(0.2, gripper=0.9)
    assert row.human_mapped_action is None
    assert row.active_component_ids == ("arm",)
    assert row.a_mix[6] == pytest.approx(-0.6)


def test_capture_rejects_mode_source_mismatches_before_backend_creation() -> None:
    policy = SequencePolicySourceA2(
        source_id="standard-sequence",
        source_kind="standard",
        actions=(_action(0.1),),
    )
    frame = StreamingInterventionControlFrameA2(
        frame_id="ordinary-with-policy",
        mode="base_only",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.1),
        human_event_id="human",
    )

    with pytest.raises(ValueError, match="ordinary teleop"):
        RobosuiteMixedControlCaptureRunnerA3(
            composer=_composer(),
            backend_factory=DeterministicRobosuiteMockBackend,
            policy_source=policy,
        ).run(
            manifest=_manifest(max_episode_steps=2),
            frames=(frame,),
            collection_mode=A3_COLLECTION_MODE_ORDINARY_TELEOP,
        )


def test_repeated_capture_with_same_inputs_is_content_addressed_identically() -> None:
    frame = StreamingInterventionControlFrameA2(
        frame_id="teleop-deterministic",
        mode="base_only",
        human_raw_action=(0.0,) * 7,
        human_mapped_action=_action(0.2),
        human_event_id="human-deterministic",
    )

    def run_once():
        return RobosuiteMixedControlCaptureRunnerA3(
            composer=_composer(),
            backend_factory=DeterministicRobosuiteMockBackend,
        ).run(
            manifest=_manifest(max_episode_steps=2),
            frames=(frame,),
            collection_mode=A3_COLLECTION_MODE_ORDINARY_TELEOP,
        )

    first = run_once()
    second = run_once()
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.rows[0].semantic_content_hash == second.rows[0].semantic_content_hash
