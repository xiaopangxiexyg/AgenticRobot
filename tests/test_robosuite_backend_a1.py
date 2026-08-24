from __future__ import annotations

import copy
import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sim.robosuite import (
    PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1,
    ROBOSUITE_A1_PACKAGE_VERSIONS,
    PandaLiftOscPoseBackend,
    RobosuiteEnvironmentContractA1,
    RobosuiteMixedControlAdapter,
    RobosuiteRuntimeBindingA1,
)
from sim.robosuite import runtime as runtime_module


def _controller_config() -> dict[str, Any]:
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_type": "delta",
                "input_ref_frame": "base",
                "gripper": {"type": "GRIP"},
            }
        },
    }


def _active_arm_controller() -> SimpleNamespace:
    return SimpleNamespace(
        name="OSC_POSE",
        control_dim=6,
        use_ori=True,
        input_type="delta",
        input_ref_frame="base",
        _goal_update_mode="achieved",
        impedance_mode="fixed",
        input_min=np.asarray([-1.0] * 6),
        input_max=np.asarray([1.0] * 6),
        output_min=np.asarray([-0.05] * 3 + [-0.5] * 3),
        output_max=np.asarray([0.05] * 3 + [0.5] * 3),
        control_limits=(np.asarray([-1.0] * 6), np.asarray([1.0] * 6)),
        interpolator_pos=None,
        interpolator_ori=None,
        kp=np.asarray([150.0] * 6),
        kd=np.asarray([2.0 * np.sqrt(150.0)] * 6),
        uncoupling=True,
        position_limits=None,
        orientation_limits=None,
        control_freq=20,
        lite_physics=True,
    )


class _FakeModel:
    nq = 2
    nv = 1
    na = 0
    nu = 1
    nbody = 4
    ngeom = 5
    nsensor = 2

    def __init__(self, seed: int) -> None:
        self._xml = f"<mujoco><asset file='/original/{seed}.stl'/></mujoco>"

    def get_xml(self) -> str:
        return self._xml


class _FakeSim:
    def __init__(self, seed: int) -> None:
        self.model = _FakeModel(seed)
        self.state = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_state(self) -> Any:
        return SimpleNamespace(flatten=lambda: self.state.copy())

    def reset(self) -> None:
        self.state[:] = 0.0

    def set_state_from_flattened(self, state: np.ndarray[Any, Any]) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()

    def forward(self) -> None:
        return None


class _FakeRobosuiteEnv:
    action_dim = 7
    action_spec = (
        np.asarray([-1.0] * 7, dtype=np.float64),
        np.asarray([1.0] * 7, dtype=np.float64),
    )
    control_freq = 20
    model_timestep = 0.002
    control_timestep = 0.05

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = copy.deepcopy(kwargs)
        self.seed = int(kwargs["seed"])
        self.horizon = int(kwargs["horizon"])
        self.sim = _FakeSim(self.seed)
        self.rng = np.random.default_rng(self.seed)
        arm_controller = _active_arm_controller()
        self.robots = [
            SimpleNamespace(
                composite_controller=SimpleNamespace(
                    name="BASIC",
                    part_controllers={
                        "right": arm_controller,
                        "right_gripper": SimpleNamespace(name="GRIP"),
                    },
                    _action_split_indexes={
                        "right": (0, 6),
                        "right_gripper": (6, 7),
                    },
                )
            )
        ]
        self.step_count = 0
        self.closed = False
        self.applied_xml: str | None = None
        self.ep_meta: dict[str, Any] = {}
        self.effective_asset_root = "/effective/"

    def reset(self) -> dict[str, np.ndarray[Any, Any]]:
        self.step_count = 0
        self.sim.state = np.asarray(
            [self.seed / 1000.0, 0.25, -0.5, 0.0],
            dtype=np.float64,
        )
        return self._get_observations(force_update=True)

    def get_ep_meta(self) -> dict[str, int]:
        return {"seed": self.seed}

    def set_ep_meta(self, metadata: dict[str, Any]) -> None:
        self.ep_meta = copy.deepcopy(metadata)

    def edit_model_xml(self, xml: str) -> str:
        return xml.replace("/original/", self.effective_asset_root)

    def reset_from_xml_string(self, xml: str) -> None:
        self.applied_xml = xml
        self.step_count = 0

    def _get_observations(
        self,
        *,
        force_update: bool,
    ) -> dict[str, np.ndarray[Any, Any]]:
        assert force_update
        offset = float(self.sim.state[0])
        return {
            "robot0_joint_pos": np.arange(7, dtype=np.float64) + offset,
            "robot0_joint_vel": np.full(7, self.sim.state[1]),
            "robot0_eef_pos": np.asarray([offset, 0.1, 0.2]),
            "robot0_eef_quat_site": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.01, -0.01]),
            "robot0_gripper_qvel": np.asarray([0.0, 0.0]),
            "cube_pos": np.asarray([0.0, offset, 0.8]),
            "cube_quat": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "gripper_to_cube_pos": np.asarray([0.0, -offset, 0.6]),
        }

    def step(
        self,
        action: np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, dict[str, Any]]:
        self.step_count += 1
        self.sim.state[0] += float(np.sum(action)) * 0.01
        self.sim.state[-1] = self.step_count * self.control_timestep
        return (
            self._get_observations(force_update=True),
            -abs(float(self.sim.state[0])),
            self.step_count >= self.horizon,
            {"fake": True},
        )

    def _check_success(self) -> bool:
        return self.step_count >= 2

    def close(self) -> None:
        self.closed = True


class _FakeSuite:
    def __init__(self) -> None:
        self.environments: list[_FakeRobosuiteEnv] = []
        self.effective_asset_root = "/effective/"

    def make(self, **kwargs: Any) -> _FakeRobosuiteEnv:
        env = _FakeRobosuiteEnv(kwargs)
        env.effective_asset_root = self.effective_asset_root
        self.environments.append(env)
        return env


def _fake_runtime(suite: _FakeSuite) -> RobosuiteRuntimeBindingA1:
    return RobosuiteRuntimeBindingA1(
        suite_module=suite,
        load_part_controller_config=lambda **_kwargs: {"type": "OSC_POSE"},
        refactor_composite_controller_config=(
            lambda _part, _robot, _arms: _controller_config()
        ),
        fingerprint={
            "python": "3.10.20",
            "python_implementation": "CPython",
            "sys_platform": "linux",
            "machine": "x86_64",
            "robosuite": "1.5.2",
            "mujoco": "3.9.0",
            "mujoco_runtime": "3.9.0",
            "numpy": "1.26.4",
            "scipy": "1.13.1",
        },
    )


def test_a1_environment_contract_is_closed_and_builds_matching_manifest() -> None:
    contract = RobosuiteEnvironmentContractA1()
    manifest = contract.make_reset_manifest(
        task_id="a1",
        episode_id="episode-1",
        reset_sequence=0,
        seed=19,
        max_episode_steps=8,
    )

    assert manifest.environment_contract_hash == contract.semantic_content_hash
    assert manifest.observation_keys == contract.observation_keys
    assert manifest.action_dimension == 7
    with pytest.raises(ValueError, match="A1 requires robot"):
        RobosuiteEnvironmentContractA1(robot="GR1")


def test_pinned_runtime_rejects_drift_before_importing_robosuite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def version(distribution: str) -> str:
        if distribution == "robosuite":
            return "1.5.1"
        return ROBOSUITE_A1_PACKAGE_VERSIONS[distribution]

    monkeypatch.setattr(runtime_module.importlib.metadata, "version", version)
    monkeypatch.setattr(
        runtime_module.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(RuntimeError, match="robosuite==1.5.2"):
        runtime_module.load_pinned_runtime()
    assert imports == []


def test_backend_normalizes_reset_restores_start_and_maps_done_semantics() -> None:
    suite = _FakeSuite()
    contract = RobosuiteEnvironmentContractA1()
    manifest = contract.make_reset_manifest(
        task_id="a1",
        episode_id="episode-2",
        reset_sequence=0,
        seed=23,
        max_episode_steps=3,
    )
    backend = PandaLiftOscPoseBackend(
        contract,
        runtime_loader=lambda: _fake_runtime(suite),
    )
    adapter = RobosuiteMixedControlAdapter(
        action_low=contract.action_low,
        action_high=contract.action_high,
    )

    observation, receipt = adapter.reset_backend(backend, manifest)
    artifact = backend.reset_artifact
    first_env = suite.environments[0]

    assert tuple(observation) == contract.observation_keys
    assert artifact.manifest_hash == manifest.semantic_content_hash
    assert artifact.compiled_model_xml_sha256 != (
        backend.last_effective_model_xml_sha256
    )
    assert artifact.effective_model_xml_sha256 == (
        backend.last_effective_model_xml_sha256
    )
    assert dict(artifact.active_controller_profile) == dict(
        PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1
    )
    assert receipt.initial_observation_hash
    assert first_env.kwargs["has_renderer"] is False
    assert first_env.kwargs["has_offscreen_renderer"] is False
    assert first_env.kwargs["use_camera_obs"] is False
    assert first_env.kwargs["initialization_noise"] is None

    first_step = backend.step((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    assert not first_step.terminated
    assert not first_step.truncated
    assert first_step.info["task_success"] is False
    assert first_step.info["sim_state_f64_sha256"]

    restored = backend.restore_episode_start(manifest, artifact)
    assert first_env.closed
    assert restored == observation
    replay_step = backend.step((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    assert replay_step.semantic_content_hash == first_step.semantic_content_hash
    second_step = backend.step((0.0,) * 7)
    assert second_step.terminated
    assert not second_step.truncated

    with pytest.raises(ValueError, match="envelope"):
        backend.step((1.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    backend.close()
    backend.close()
    assert suite.environments[-1].closed


def test_backend_rejects_effective_model_xml_drift_before_restore() -> None:
    suite = _FakeSuite()
    contract = RobosuiteEnvironmentContractA1()
    manifest = contract.make_reset_manifest(
        task_id="a1",
        episode_id="episode-xml-drift",
        reset_sequence=0,
        seed=29,
        max_episode_steps=2,
    )
    backend = PandaLiftOscPoseBackend(
        contract,
        runtime_loader=lambda: _fake_runtime(suite),
    )

    backend.reset(manifest)
    artifact = backend.reset_artifact
    suite.effective_asset_root = "/drifted/"

    with pytest.raises(RuntimeError, match="model XML drift"):
        backend.restore_episode_start(manifest, artifact)
    assert suite.environments[-1].closed


def test_backend_close_keeps_failed_resource_retriable() -> None:
    class _CloseOnce:
        def __init__(self) -> None:
            self.attempts = 0

        def close(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("synthetic close failure")

    backend = PandaLiftOscPoseBackend(
        RobosuiteEnvironmentContractA1(),
        runtime_loader=lambda: _fake_runtime(_FakeSuite()),
    )
    resource = _CloseOnce()
    backend._env = resource

    with pytest.raises(RuntimeError, match="synthetic close failure"):
        backend.close()
    assert backend._env is resource

    backend.close()
    assert backend._env is None
    assert resource.attempts == 2


def test_backend_rejects_manifest_or_runtime_drift_before_environment_creation() -> (
    None
):
    suite = _FakeSuite()
    contract = RobosuiteEnvironmentContractA1()
    backend = PandaLiftOscPoseBackend(
        contract,
        runtime_loader=lambda: _fake_runtime(suite),
    )
    manifest = contract.make_reset_manifest(
        task_id="a1",
        episode_id="episode-3",
        reset_sequence=0,
        seed=1,
        max_episode_steps=2,
    )
    bad_manifest = type(manifest)(
        **{
            **dict(manifest.to_payload()),
            "environment_contract_hash": "wrong-contract",
        }
    )

    with pytest.raises(ValueError, match="contract hash"):
        backend.reset(bad_manifest)
    assert suite.environments == []

    drifted_runtime = _fake_runtime(suite)
    object.__setattr__(
        drifted_runtime,
        "fingerprint",
        {**dict(drifted_runtime.fingerprint), "mujoco": "3.10.0"},
    )
    drifted_backend = PandaLiftOscPoseBackend(
        contract,
        runtime_loader=lambda: drifted_runtime,
    )
    with pytest.raises(RuntimeError, match="version drift"):
        drifted_backend.reset(manifest)
    assert suite.environments == []


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
def test_real_headless_backend_replays_one_nonzero_step_exactly() -> None:
    if not _real_runtime_available():
        pytest.skip("isolated pinned robosuite A1 runtime is not active")
    contract = RobosuiteEnvironmentContractA1()
    manifest = contract.make_reset_manifest(
        task_id="a1-real",
        episode_id="episode-real",
        reset_sequence=0,
        seed=20260824,
        max_episode_steps=5,
    )
    backend = PandaLiftOscPoseBackend(contract)
    action = (0.08, -0.04, 0.03, 0.02, 0.0, -0.01, -1.0)
    try:
        initial = backend.reset(manifest)
        artifact = backend.reset_artifact
        source_step = backend.step(action)
        restored = backend.restore_episode_start(manifest, artifact)
        replay_step = backend.step(action)
    finally:
        backend.close()

    assert restored == initial
    assert replay_step.semantic_content_hash == source_step.semantic_content_hash
    fingerprint = artifact.runtime_fingerprint
    lock_path = Path(__file__).parents[1] / "environments/robosuite/uv.lock"
    assert (
        fingerprint["uv_lock_sha256"]
        == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    )
    assert fingerprint["python_soabi"]
    assert set(fingerprint["direct_distributions"]) == {
        "mujoco",
        "numpy",
        "robosuite",
        "scipy",
    }
    assert len(fingerprint["mujoco_native"]) == 2
    assert "/tmp/" not in repr(fingerprint)
