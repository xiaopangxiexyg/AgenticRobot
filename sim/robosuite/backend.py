"""Headless Panda / Lift / OSC_POSE backend for the pinned robosuite A1 profile."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from .adapter import RobosuiteA0Backend, RobosuiteBackendStep
from .contracts import (
    PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1,
    RobosuiteEnvironmentContractA1,
    RobosuiteEpisodeStartArtifactA1,
    RobosuiteResetManifest,
)
from .runtime import RobosuiteRuntimeBindingA1, load_pinned_runtime


def _normalize_primitive(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float.")
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} contains a non-finite float.")
        return numeric
    if isinstance(value, np.ndarray):
        return tuple(
            _normalize_primitive(item, name=f"{name}.item") for item in value.tolist()
        )
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_primitive(item, name=f"{name}.item") for item in value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError(f"{name} requires non-empty string keys.")
            normalized[key] = _normalize_primitive(item, name=f"{name}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}.")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_state(state: Any) -> str:
    array = np.asarray(state, dtype=np.dtype("<f8"))
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("simulator state must be a finite flat vector.")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _controller_vector(value: Any, *, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError(f"active controller {name} must be a finite vector.")
    return tuple(float(item) for item in array)


class RobosuiteReplayableA1Backend(RobosuiteA0Backend, Protocol):
    """Backend that can restore only a captured episode start."""

    @property
    def reset_artifact(self) -> RobosuiteEpisodeStartArtifactA1: ...

    @property
    def last_effective_model_xml_sha256(self) -> str: ...

    def restore_episode_start(
        self,
        manifest: RobosuiteResetManifest,
        artifact: RobosuiteEpisodeStartArtifactA1,
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class PandaLiftOscPoseBackend:
    """Concrete headless backend for one reviewed A1 environment contract.

    The backend never owns policy, intervention, safety, collection, or GUI
    semantics. It only binds one simulation profile and exposes deterministic
    episode-start restore plus normalized steps.
    """

    def __init__(
        self,
        contract: RobosuiteEnvironmentContractA1,
        *,
        runtime_loader: Callable[[], RobosuiteRuntimeBindingA1] = load_pinned_runtime,
    ) -> None:
        if not isinstance(contract, RobosuiteEnvironmentContractA1):
            raise TypeError("contract must be a RobosuiteEnvironmentContractA1.")
        if not callable(runtime_loader):
            raise TypeError("runtime_loader must be callable.")
        self._contract = contract
        self._runtime_loader = runtime_loader
        self._runtime: RobosuiteRuntimeBindingA1 | None = None
        self._env: Any | None = None
        self._manifest: RobosuiteResetManifest | None = None
        self._reset_artifact: RobosuiteEpisodeStartArtifactA1 | None = None
        self._last_effective_model_xml_sha256: str | None = None
        self._step_index = 0

    @property
    def backend_id(self) -> str:
        return (
            "robosuite-1.5.2:panda-lift-osc-pose-a1:"
            f"{self._contract.semantic_content_hash}"
        )

    @property
    def reset_artifact(self) -> RobosuiteEpisodeStartArtifactA1:
        if self._reset_artifact is None:
            raise RuntimeError("backend has no captured episode start.")
        return self._reset_artifact

    @property
    def last_effective_model_xml_sha256(self) -> str:
        if self._last_effective_model_xml_sha256 is None:
            raise RuntimeError("backend has not applied an episode-start XML.")
        return self._last_effective_model_xml_sha256

    def reset(self, manifest: RobosuiteResetManifest) -> Mapping[str, Any]:
        self._validate_manifest(manifest)
        self.close()
        self._reset_artifact = None
        self._last_effective_model_xml_sha256 = None
        runtime = self._runtime_binding()
        controller_config = self._controller_config(runtime)
        environment_kwargs = self._environment_kwargs(manifest)
        try:
            env = runtime.suite_module.make(
                controller_configs=copy.deepcopy(controller_config),
                **environment_kwargs,
            )
            self._env = env
            env.reset()
            self._validate_environment(env)
            artifact = self._capture_episode_start(
                manifest=manifest,
                runtime=runtime,
                environment_kwargs=environment_kwargs,
                controller_config=controller_config,
            )
            observation = self._apply_episode_start(env, artifact)
            self._validate_environment(env)
        except Exception:
            self.close()
            raise
        self._manifest = manifest
        self._reset_artifact = artifact
        self._step_index = 0
        return observation

    def restore_episode_start(
        self,
        manifest: RobosuiteResetManifest,
        artifact: RobosuiteEpisodeStartArtifactA1,
    ) -> Mapping[str, Any]:
        self._validate_manifest(manifest)
        if not isinstance(artifact, RobosuiteEpisodeStartArtifactA1):
            raise TypeError("artifact must be a RobosuiteEpisodeStartArtifactA1.")
        if artifact.manifest_hash != manifest.semantic_content_hash:
            raise ValueError("episode-start artifact does not bind this manifest.")
        if artifact.environment_contract_hash != self._contract.semantic_content_hash:
            raise ValueError("episode-start artifact contract hash mismatch.")
        self.close()
        self._reset_artifact = None
        self._last_effective_model_xml_sha256 = None
        runtime = self._runtime_binding()
        if dict(artifact.runtime_fingerprint) != dict(runtime.fingerprint):
            raise RuntimeError(
                "exact prefix replay requires the captured runtime fingerprint."
            )
        controller_config = self._controller_config(runtime)
        environment_kwargs = self._environment_kwargs(manifest)
        controller_hash = canonical_content_hash(
            _normalize_primitive(controller_config, name="controller_config"),
            domain="dexteleop.robosuite.controller_config.a1",
        )
        kwargs_hash = canonical_content_hash(
            _normalize_primitive(environment_kwargs, name="environment_kwargs"),
            domain="dexteleop.robosuite.environment_kwargs.a1",
        )
        if controller_hash != artifact.controller_config_hash:
            raise RuntimeError("controller configuration drift blocks exact replay.")
        if kwargs_hash != artifact.environment_kwargs_hash:
            raise RuntimeError("environment construction drift blocks exact replay.")
        try:
            env = runtime.suite_module.make(
                controller_configs=copy.deepcopy(controller_config),
                **environment_kwargs,
            )
            self._env = env
            observation = self._apply_episode_start(env, artifact)
            self._validate_environment(env)
            self._validate_artifact_runtime_shape(env, artifact)
        except Exception:
            self.close()
            raise
        self._manifest = manifest
        self._reset_artifact = artifact
        self._step_index = 0
        return observation

    def step(self, action: tuple[float, ...]) -> RobosuiteBackendStep:
        env = self._env
        manifest = self._manifest
        if env is None or manifest is None:
            raise RuntimeError("backend must be reset or restored before stepping.")
        normalized = np.asarray(
            tuple(float(value) for value in action), dtype=np.float64
        )
        if normalized.shape != (len(self._contract.action_layout),):
            raise ValueError(
                f"action must have dimension {len(self._contract.action_layout)}."
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("action must contain only finite values.")
        low, high = env.action_spec
        if np.any(normalized < np.asarray(low)) or np.any(
            normalized > np.asarray(high)
        ):
            raise ValueError("action exceeds the validated robosuite action envelope.")
        raw_observation, reward, raw_done, raw_info = env.step(normalized)
        self._step_index += 1
        task_success = bool(env._check_success())
        truncated = bool(raw_done) or self._step_index >= manifest.max_episode_steps
        state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        return RobosuiteBackendStep(
            observation=self._project_observation(raw_observation),
            reward=float(reward),
            terminated=task_success,
            truncated=truncated,
            info={
                "backend_kind": "robosuite_headless_a1",
                "manifest_hash": manifest.semantic_content_hash,
                "step_index": self._step_index,
                "task_success": task_success,
                "raw_done": bool(raw_done),
                "raw_info": _normalize_primitive(raw_info, name="raw_info"),
                "sim_state_f64_sha256": _sha256_state(state),
                "sim_state_size": int(state.size),
            },
        )

    def close(self) -> None:
        env = self._env
        if env is not None:
            env.close()
        self._env = None
        self._manifest = None
        self._step_index = 0

    def _runtime_binding(self) -> RobosuiteRuntimeBindingA1:
        if self._runtime is None:
            runtime = self._runtime_loader()
            if not isinstance(runtime, RobosuiteRuntimeBindingA1):
                raise TypeError("runtime_loader must return RobosuiteRuntimeBindingA1.")
            expected_versions = {
                "robosuite": self._contract.robosuite_version,
                "mujoco": self._contract.mujoco_version,
                "mujoco_runtime": self._contract.mujoco_version,
                "numpy": self._contract.numpy_version,
                "scipy": self._contract.scipy_version,
            }
            actual_versions = {
                name: str(runtime.fingerprint.get(name, ""))
                for name in expected_versions
            }
            if actual_versions != expected_versions:
                raise RuntimeError(
                    "runtime fingerprint version drift: "
                    f"expected {expected_versions}, got {actual_versions}."
                )
            python_version = str(runtime.fingerprint.get("python", ""))
            if not python_version.startswith(f"{self._contract.python_version}."):
                raise RuntimeError(
                    "runtime fingerprint Python version does not match A1."
                )
            self._runtime = runtime
        return self._runtime

    def _validate_manifest(self, manifest: RobosuiteResetManifest) -> None:
        if not isinstance(manifest, RobosuiteResetManifest):
            raise TypeError("manifest must be a RobosuiteResetManifest.")
        if manifest.environment_id != self._contract.environment_id:
            raise ValueError("manifest environment does not match the A1 contract.")
        if manifest.environment_contract_hash != self._contract.semantic_content_hash:
            raise ValueError("manifest environment contract hash mismatch.")
        if manifest.action_dimension != len(self._contract.action_layout):
            raise ValueError(
                "manifest action dimension does not match the A1 contract."
            )
        if manifest.observation_keys != self._contract.observation_keys:
            raise ValueError("manifest observation keys do not match the A1 contract.")
        if manifest.options:
            raise ValueError("A1 rejects reset options outside the fixed contract.")

    def _controller_config(
        self,
        runtime: RobosuiteRuntimeBindingA1,
    ) -> dict[str, Any]:
        part = runtime.load_part_controller_config(
            default_controller=self._contract.arm_controller
        )
        config = runtime.refactor_composite_controller_config(
            part,
            self._contract.robot,
            ["right"],
        )
        if not isinstance(config, Mapping):
            raise TypeError("robosuite controller factory returned a non-mapping.")
        normalized = copy.deepcopy(dict(config))
        body_parts = normalized.get("body_parts")
        right = body_parts.get("right") if isinstance(body_parts, Mapping) else None
        gripper = right.get("gripper") if isinstance(right, Mapping) else None
        expected = {
            "composite": normalized.get("type"),
            "arm": right.get("type") if isinstance(right, Mapping) else None,
            "input_type": (
                right.get("input_type") if isinstance(right, Mapping) else None
            ),
            "input_ref_frame": (
                right.get("input_ref_frame") if isinstance(right, Mapping) else None
            ),
            "gripper": gripper.get("type") if isinstance(gripper, Mapping) else None,
        }
        required = {
            "composite": self._contract.composite_controller,
            "arm": self._contract.arm_controller,
            "input_type": self._contract.controller_input_type,
            "input_ref_frame": self._contract.controller_input_reference_frame,
            "gripper": "GRIP",
        }
        if expected != required:
            raise RuntimeError(
                f"robosuite controller profile drift: expected {required}, got {expected}."
            )
        return normalized

    def _environment_kwargs(
        self,
        manifest: RobosuiteResetManifest,
    ) -> dict[str, Any]:
        return {
            "env_name": self._contract.environment_id,
            "robots": self._contract.robot,
            "has_renderer": self._contract.has_renderer,
            "has_offscreen_renderer": self._contract.has_offscreen_renderer,
            "use_camera_obs": self._contract.use_camera_observations,
            "control_freq": self._contract.control_frequency_hz,
            "horizon": manifest.max_episode_steps,
            "ignore_done": self._contract.ignore_done,
            "reward_shaping": self._contract.reward_shaping,
            "lite_physics": self._contract.lite_physics,
            "hard_reset": self._contract.hard_reset,
            "seed": manifest.seed,
            "initialization_noise": self._contract.initialization_noise,
        }

    def _validate_environment(self, env: Any) -> None:
        if int(env.action_dim) != len(self._contract.action_layout):
            raise RuntimeError("robosuite action dimension drifted from A1.")
        low, high = env.action_spec
        if not np.array_equal(np.asarray(low), np.asarray(self._contract.action_low)):
            raise RuntimeError("robosuite action lower bounds drifted from A1.")
        if not np.array_equal(np.asarray(high), np.asarray(self._contract.action_high)):
            raise RuntimeError("robosuite action upper bounds drifted from A1.")
        if int(env.control_freq) != self._contract.control_frequency_hz:
            raise RuntimeError("robosuite control frequency drifted from A1.")
        self._active_controller_profile(env)

    def _active_controller_profile(self, env: Any) -> Mapping[str, Any]:
        try:
            composite = env.robots[0].composite_controller
            parts = composite.part_controllers
            arm_controller = parts["right"]
            action_splits = composite._action_split_indexes
            control_low, control_high = arm_controller.control_limits
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "robosuite did not expose the pinned A1 controller structure."
            ) from exc
        input_low = np.asarray(arm_controller.input_min, dtype=np.float64)
        input_high = np.asarray(arm_controller.input_max, dtype=np.float64)
        output_low = np.asarray(arm_controller.output_min, dtype=np.float64)
        output_high = np.asarray(arm_controller.output_max, dtype=np.float64)
        if any(
            array.shape != (6,)
            for array in (input_low, input_high, output_low, output_high)
        ) or any(
            not np.all(np.isfinite(array))
            for array in (input_low, input_high, output_low, output_high)
        ):
            raise RuntimeError("active OSC_POSE scaling bounds drifted from A1.")
        action_scale = (output_high - output_low) / (input_high - input_low)
        input_transform = (input_high + input_low) / 2.0
        output_transform = (output_high + output_low) / 2.0
        profile = _normalize_primitive(
            {
                "composite_name": composite.name,
                "part_names": tuple(parts),
                "action_splits": tuple(
                    (name, tuple(int(index) for index in bounds))
                    for name, bounds in action_splits.items()
                ),
                "arm_name": arm_controller.name,
                "control_dimension": int(arm_controller.control_dim),
                "use_orientation": arm_controller.use_ori,
                "input_type": arm_controller.input_type,
                "input_reference_frame": arm_controller.input_ref_frame,
                "goal_update_mode": arm_controller._goal_update_mode,
                "impedance_mode": arm_controller.impedance_mode,
                "input_low": input_low,
                "input_high": input_high,
                "output_low": output_low,
                "output_high": output_high,
                "control_low": _controller_vector(control_low, name="control_low"),
                "control_high": _controller_vector(
                    control_high,
                    name="control_high",
                ),
                "action_scale": action_scale,
                "action_input_transform": input_transform,
                "action_output_transform": output_transform,
                "interpolator_position": arm_controller.interpolator_pos,
                "interpolator_orientation": arm_controller.interpolator_ori,
                "kp": _controller_vector(arm_controller.kp, name="kp"),
                "kd": _controller_vector(arm_controller.kd, name="kd"),
                "uncoupling": arm_controller.uncoupling,
                "position_limits": arm_controller.position_limits,
                "orientation_limits": arm_controller.orientation_limits,
                "control_frequency_hz": int(arm_controller.control_freq),
                "lite_physics": arm_controller.lite_physics,
            },
            name="active_controller_profile",
        )
        if dict(profile) != dict(PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1):
            raise RuntimeError(
                "active robosuite controller semantics drifted from the A1 profile."
            )
        return profile

    def _capture_episode_start(
        self,
        *,
        manifest: RobosuiteResetManifest,
        runtime: RobosuiteRuntimeBindingA1,
        environment_kwargs: Mapping[str, Any],
        controller_config: Mapping[str, Any],
    ) -> RobosuiteEpisodeStartArtifactA1:
        env = self._env
        if env is None:
            raise RuntimeError("cannot capture without an environment.")
        model = env.sim.model
        state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        compiled_model_xml = str(model.get_xml())
        effective_model_xml = env.edit_model_xml(compiled_model_xml)
        if not isinstance(effective_model_xml, str) or not effective_model_xml.strip():
            raise RuntimeError("robosuite returned an invalid effective model XML.")
        rng_state = _normalize_primitive(
            copy.deepcopy(env.rng.bit_generator.state),
            name="rng_state",
        )
        return RobosuiteEpisodeStartArtifactA1(
            manifest_hash=manifest.semantic_content_hash,
            environment_contract_hash=self._contract.semantic_content_hash,
            runtime_fingerprint=_normalize_primitive(
                runtime.fingerprint,
                name="runtime_fingerprint",
            ),
            environment_kwargs=_normalize_primitive(
                environment_kwargs,
                name="environment_kwargs",
            ),
            controller_config=_normalize_primitive(
                controller_config,
                name="controller_config",
            ),
            active_controller_profile=self._active_controller_profile(env),
            compiled_model_xml=compiled_model_xml,
            effective_model_xml_sha256=_sha256_text(effective_model_xml),
            initial_state=tuple(float(value) for value in state),
            episode_metadata=_normalize_primitive(
                copy.deepcopy(env.get_ep_meta()),
                name="episode_metadata",
            ),
            rng_state=rng_state,
            action_low=tuple(float(value) for value in env.action_spec[0]),
            action_high=tuple(float(value) for value in env.action_spec[1]),
            model_dimensions={
                name: int(getattr(model, name))
                for name in ("nq", "nv", "na", "nu", "nbody", "ngeom", "nsensor")
            },
            model_timestep=float(env.model_timestep),
            control_timestep=float(env.control_timestep),
        )

    def _apply_episode_start(
        self,
        env: Any,
        artifact: RobosuiteEpisodeStartArtifactA1,
    ) -> Mapping[str, Any]:
        env.set_ep_meta(_thaw(artifact.episode_metadata))
        effective_xml = env.edit_model_xml(artifact.compiled_model_xml)
        if not isinstance(effective_xml, str) or not effective_xml.strip():
            raise RuntimeError("robosuite returned an invalid effective model XML.")
        effective_xml_hash = _sha256_text(effective_xml)
        if effective_xml_hash != artifact.effective_model_xml_sha256:
            raise RuntimeError(
                "effective robosuite model XML drift blocks exact prefix replay."
            )
        env.reset_from_xml_string(effective_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(
            np.asarray(artifact.initial_state, dtype=np.float64)
        )
        env.sim.forward()
        env.rng.bit_generator.state = _thaw(artifact.rng_state)
        raw_observation = env._get_observations(force_update=True)
        self._last_effective_model_xml_sha256 = effective_xml_hash
        return self._project_observation(raw_observation)

    def _project_observation(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("robosuite observation must be a mapping.")
        projected: dict[str, tuple[float, ...]] = {}
        for key in self._contract.observation_keys:
            if key not in raw:
                raise RuntimeError(f"robosuite observation is missing {key!r}.")
            value = np.asarray(raw[key], dtype=np.float64)
            if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"robosuite observation {key!r} must be a finite vector."
                )
            projected[key] = tuple(float(item) for item in value)
        return MappingProxyType(projected)

    def _validate_artifact_runtime_shape(
        self,
        env: Any,
        artifact: RobosuiteEpisodeStartArtifactA1,
    ) -> None:
        model = env.sim.model
        actual_dimensions = {
            name: int(getattr(model, name)) for name in artifact.model_dimensions
        }
        if actual_dimensions != dict(artifact.model_dimensions):
            raise RuntimeError("MuJoCo model dimensions drifted from the artifact.")
        active_controller_profile = self._active_controller_profile(env)
        if dict(active_controller_profile) != dict(artifact.active_controller_profile):
            raise RuntimeError("active controller profile drifted from the artifact.")
        if float(env.model_timestep) != artifact.model_timestep or (
            float(env.control_timestep) != artifact.control_timestep
        ):
            raise RuntimeError("simulation timesteps drifted from the artifact.")
        low, high = env.action_spec
        if (
            tuple(float(value) for value in low) != artifact.action_low
            or tuple(float(value) for value in high) != artifact.action_high
        ):
            raise RuntimeError("action bounds drifted from the artifact.")


__all__ = [
    "PandaLiftOscPoseBackend",
    "RobosuiteReplayableA1Backend",
]
