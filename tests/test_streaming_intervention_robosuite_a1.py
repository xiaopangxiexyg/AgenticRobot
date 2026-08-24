from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from experiments.sim_learning.streaming_intervention import (
    StreamingInterventionFrame,
    StreamingInterventionMockRunner,
    StreamingInterventionRunner,
)
from experiments.sim_learning.streaming_intervention.mock_runner import (
    MOCK_OBSERVATION_KEYS_A0,
)
from sim.robosuite import (
    PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1,
    ROBOSUITE_A1_PACKAGE_VERSIONS,
    PandaLiftOscPoseBackend,
    RobosuiteBackendStep,
    RobosuiteEnvironmentContractA1,
    RobosuiteEpisodeStartArtifactA1,
    RobosuiteMixedControlAdapter,
    RobosuiteResetManifest,
)
from sim.robosuite.adapter import (
    ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
    ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
)


def _manifest(*, max_episode_steps: int = 4) -> RobosuiteResetManifest:
    return RobosuiteResetManifest(
        environment_id="ReplayFake",
        environment_contract_hash="replay-fake-contract",
        task_id="streaming-a1",
        episode_id="episode-1",
        reset_sequence=0,
        seed=31,
        action_dimension=2,
        observation_keys=MOCK_OBSERVATION_KEYS_A0,
        max_episode_steps=max_episode_steps,
    )


def _episode_start_artifact(
    manifest: RobosuiteResetManifest,
    state: tuple[float, float],
    effective_xml_hash: str,
) -> RobosuiteEpisodeStartArtifactA1:
    return RobosuiteEpisodeStartArtifactA1(
        manifest_hash=manifest.semantic_content_hash,
        environment_contract_hash=manifest.environment_contract_hash,
        runtime_fingerprint={"runtime": "replay-fake-a1"},
        environment_kwargs={"seed": manifest.seed},
        controller_config={"type": "fake"},
        active_controller_profile=PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1,
        compiled_model_xml="<mujoco model='replay-fake-a1'/>",
        effective_model_xml_sha256=effective_xml_hash,
        initial_state=state,
        episode_metadata={},
        rng_state={"bit_generator": "fake", "state": manifest.seed},
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
        model_dimensions={"nq": 1, "nv": 1, "na": 0, "nu": 2},
        model_timestep=0.002,
        control_timestep=0.05,
    )


class _ReplayBackend:
    backend_id = "streaming-a1-replay-fake"

    def __init__(
        self,
        *,
        mutate_at_step: int | None = None,
        reset_observation_offset: float = 0.0,
        effective_xml_hash: str | None = None,
    ) -> None:
        self.mutate_at_step = mutate_at_step
        self.reset_observation_offset = reset_observation_offset
        self.manifest: RobosuiteResetManifest | None = None
        self.state = (0.0, 0.0)
        self.step_index = 0
        self.closed = False
        self._artifact: RobosuiteEpisodeStartArtifactA1 | None = None
        self._effective_xml_hash = (
            effective_xml_hash or hashlib.sha256(b"replay-fake").hexdigest()
        )

    @property
    def reset_artifact(self) -> RobosuiteEpisodeStartArtifactA1:
        if self._artifact is None:
            raise RuntimeError("no reset artifact")
        return self._artifact

    @property
    def last_effective_model_xml_sha256(self) -> str:
        return self._effective_xml_hash

    def reset(self, manifest: RobosuiteResetManifest) -> Mapping[str, object]:
        self.manifest = manifest
        self.state = (manifest.seed / 100.0, -(manifest.seed / 100.0))
        self.step_index = 0
        self.closed = False
        self._artifact = _episode_start_artifact(
            manifest,
            self.state,
            self._effective_xml_hash,
        )
        return self._observation()

    def restore_episode_start(
        self,
        manifest: RobosuiteResetManifest,
        artifact: RobosuiteEpisodeStartArtifactA1,
    ) -> Mapping[str, object]:
        if artifact.manifest_hash != manifest.semantic_content_hash:
            raise ValueError("artifact mismatch")
        self.manifest = manifest
        self._artifact = artifact
        self.state = tuple(artifact.initial_state)  # type: ignore[assignment]
        self.step_index = 0
        self.closed = False
        observation = dict(self._observation())
        if self.reset_observation_offset:
            observation["state"] = (
                observation["state"][0] + self.reset_observation_offset,
                observation["state"][1],
            )
        return MappingProxyType(observation)

    def step(self, action: tuple[float, ...]) -> RobosuiteBackendStep:
        if self.manifest is None:
            raise RuntimeError("reset required")
        self.step_index += 1
        self.state = (
            self.state[0] + (0.1 * action[0]),
            self.state[1] + (0.1 * action[1]),
        )
        if self.mutate_at_step == self.step_index:
            self.state = (self.state[0] + 0.001, self.state[1])
        state_hash = canonical_content_hash(
            self.state,
            domain="tests.streaming_intervention.replay_fake_state.a1",
        )
        return RobosuiteBackendStep(
            observation=self._observation(),
            reward=-sum(abs(value) for value in self.state),
            terminated=False,
            truncated=self.step_index >= self.manifest.max_episode_steps,
            info={
                "state_hash": state_hash,
                "step_index": self.step_index,
            },
        )

    def close(self) -> None:
        self.closed = True
        self.manifest = None

    def _observation(self) -> Mapping[str, object]:
        if self.manifest is None:
            raise RuntimeError("reset required")
        return MappingProxyType(
            {
                "state": self.state,
                "step_index": self.step_index,
                "episode_seed": self.manifest.seed,
            }
        )


class _BackendFactory:
    def __init__(
        self,
        *,
        replay_mutate_at_step: int | None = None,
        replay_reset_offset: float = 0.0,
        replay_effective_xml_hash: str | None = None,
    ) -> None:
        self.replay_mutate_at_step = replay_mutate_at_step
        self.replay_reset_offset = replay_reset_offset
        self.replay_effective_xml_hash = replay_effective_xml_hash
        self.instances: list[_ReplayBackend] = []

    def __call__(self) -> _ReplayBackend:
        is_replay = bool(self.instances)
        backend = _ReplayBackend(
            mutate_at_step=(self.replay_mutate_at_step if is_replay else None),
            reset_observation_offset=(self.replay_reset_offset if is_replay else 0.0),
            effective_xml_hash=(self.replay_effective_xml_hash if is_replay else None),
        )
        self.instances.append(backend)
        return backend


def _adapter() -> RobosuiteMixedControlAdapter:
    return RobosuiteMixedControlAdapter(
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )


def _frames() -> tuple[StreamingInterventionFrame, ...]:
    return (
        StreamingInterventionFrame(
            frame_id="base-none",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
            base_action=(0.2, -0.2),
        ),
        StreamingInterventionFrame(
            frame_id="base-explicit-zero",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
            base_action=(0.2, -0.2),
            intervention_action=(0.0, 0.0),
        ),
        StreamingInterventionFrame(
            frame_id="relative",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
            base_action=(0.8, -0.8),
            intervention_action=(0.4, -0.4),
        ),
        StreamingInterventionFrame(
            frame_id="absolute",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
            base_action=(-0.9, -0.9),
            intervention_action=(0.3, 0.7),
        ),
    )


def test_injected_runner_preserves_frames_runs_three_modes_and_closes_backend() -> None:
    factory = _BackendFactory()
    runner = StreamingInterventionRunner(
        adapter=_adapter(),
        backend_factory=factory,
    )

    result = runner.run(manifest=_manifest(), frames=_frames())

    assert [step.selection.applied_action for step in result.steps] == [
        (0.2, -0.2),
        (0.2, -0.2),
        (1.0, -1.0),
        (0.3, 0.7),
    ]
    assert result.steps[0].input_frame is not None
    assert result.steps[0].input_frame.intervention_action is None
    assert result.steps[1].input_frame is not None
    assert result.steps[1].input_frame.intervention_action == (0.0, 0.0)
    assert result.steps[0].selection.semantic_content_hash == (
        result.steps[1].selection.semantic_content_hash
    )
    assert result.steps[0].frame_hash != result.steps[1].frame_hash
    assert result.reset_artifact is not None
    assert result.stop_reason == "manifest_step_limit"
    assert factory.instances[0].closed


def test_exact_prefix_replay_restarts_at_episode_start_and_matches_all_steps() -> None:
    factory = _BackendFactory()
    runner = StreamingInterventionRunner(
        adapter=_adapter(),
        backend_factory=factory,
    )
    source = runner.run(manifest=_manifest(), frames=_frames())

    receipt = runner.replay_prefix(
        source_result=source,
        prefix_length=len(source.steps),
    )

    assert receipt.matched
    assert receipt.matched_steps == len(source.steps)
    assert receipt.reset_observation_match
    assert receipt.first_mismatch_index is None
    assert len(factory.instances) == 2
    assert all(instance.closed for instance in factory.instances)


def test_prefix_replay_stops_at_first_backend_step_mismatch() -> None:
    factory = _BackendFactory(replay_mutate_at_step=2)
    runner = StreamingInterventionRunner(
        adapter=_adapter(),
        backend_factory=factory,
    )
    source = runner.run(manifest=_manifest(), frames=_frames())

    receipt = runner.replay_prefix(source_result=source, prefix_length=4)

    assert not receipt.matched
    assert receipt.reset_observation_match
    assert receipt.first_mismatch_index == 1
    assert receipt.matched_steps == 1
    assert receipt.mismatch_kind == "backend_step_hash"


def test_prefix_replay_fails_closed_on_reset_observation_mismatch() -> None:
    factory = _BackendFactory(replay_reset_offset=0.01)
    runner = StreamingInterventionRunner(
        adapter=_adapter(),
        backend_factory=factory,
    )
    source = runner.run(manifest=_manifest(), frames=_frames())

    receipt = runner.replay_prefix(source_result=source, prefix_length=3)

    assert not receipt.matched
    assert not receipt.reset_observation_match
    assert receipt.first_mismatch_index is None
    assert receipt.matched_steps == 0
    assert receipt.mismatch_kind == "reset_observation_hash"


def test_prefix_replay_fails_closed_on_effective_model_xml_drift() -> None:
    factory = _BackendFactory(replay_effective_xml_hash="b" * 64)
    runner = StreamingInterventionRunner(
        adapter=_adapter(),
        backend_factory=factory,
    )
    source = runner.run(manifest=_manifest(), frames=_frames())

    receipt = runner.replay_prefix(source_result=source, prefix_length=4)

    assert not receipt.matched
    assert receipt.reset_observation_match
    assert receipt.first_mismatch_index is None
    assert receipt.matched_steps == 0
    assert receipt.mismatch_kind == "effective_model_xml_sha256"
    assert receipt.expected_hash == source.source_effective_model_xml_sha256
    assert receipt.actual_hash == "b" * 64


def test_mock_evidence_cannot_be_promoted_to_exact_episode_start_replay() -> None:
    mock_runner = StreamingInterventionMockRunner(_adapter())
    source = mock_runner.run(manifest=_manifest(), frames=_frames())
    generic = StreamingInterventionRunner(
        adapter=_adapter(),
        backend_factory=_ReplayBackend,
    )

    with pytest.raises(ValueError, match="episode-start artifact"):
        generic.replay_prefix(source_result=source, prefix_length=1)


def _real_runtime_available() -> bool:
    try:
        return all(
            importlib.metadata.version(name) == version
            for name, version in ROBOSUITE_A1_PACKAGE_VERSIONS.items()
        )
    except importlib.metadata.PackageNotFoundError:
        return False


@pytest.mark.optional
@pytest.mark.integration
def test_real_runner_executes_three_modes_and_replays_full_prefix_exactly() -> None:
    if not _real_runtime_available():
        pytest.skip("isolated pinned robosuite A1 runtime is not active")
    contract = RobosuiteEnvironmentContractA1()
    manifest = contract.make_reset_manifest(
        task_id="streaming-a1-real",
        episode_id="episode-real",
        reset_sequence=0,
        seed=20260824,
        max_episode_steps=8,
    )
    adapter = RobosuiteMixedControlAdapter(
        action_low=contract.action_low,
        action_high=contract.action_high,
    )
    runner = StreamingInterventionRunner(
        adapter=adapter,
        backend_factory=lambda: PandaLiftOscPoseBackend(contract),
    )
    frames = (
        StreamingInterventionFrame(
            frame_id="real-base",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_BASE_ONLY_A0,
            base_action=(0.05, -0.02, 0.01, 0.0, 0.01, -0.01, -0.5),
        ),
        StreamingInterventionFrame(
            frame_id="real-relative",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_RELATIVE_OVERLAY_A0,
            base_action=(0.02, 0.01, 0.0, 0.01, 0.0, 0.0, -0.2),
            intervention_action=(0.01, -0.01, 0.02, 0.0, 0.01, 0.0, 0.1),
        ),
        StreamingInterventionFrame(
            frame_id="real-absolute",
            mode=ROBOSUITE_MIXED_CONTROL_MODE_ABSOLUTE_OVERLAY_OWNER_A0,
            base_action=(0.0,) * 7,
            intervention_action=(-0.03, 0.02, 0.0, 0.0, -0.01, 0.01, 0.5),
        ),
    )

    source = runner.run(manifest=manifest, frames=frames)
    receipt = runner.replay_prefix(
        source_result=source,
        prefix_length=len(source.steps),
    )

    assert len(source.steps) == 3
    assert source.reset_artifact is not None
    assert receipt.matched
    assert receipt.matched_steps == 3
