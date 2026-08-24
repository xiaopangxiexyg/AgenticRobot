"""Content-addressed reset/replay contracts for the robosuite seam."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

ROBOSUITE_RESET_MANIFEST_SCHEMA_A0 = "dexteleop.robosuite.reset_manifest.a0"
ROBOSUITE_RESET_MANIFEST_HASH_DOMAIN_A0 = "dexteleop.robosuite.reset_manifest.a0"
ROBOSUITE_RESET_RECEIPT_HASH_DOMAIN_A0 = "dexteleop.robosuite.reset_receipt.a0"
ROBOSUITE_ENVIRONMENT_CONTRACT_SCHEMA_A1 = "dexteleop.robosuite.environment_contract.a1"
ROBOSUITE_ENVIRONMENT_CONTRACT_HASH_DOMAIN_A1 = (
    "dexteleop.robosuite.environment_contract.a1"
)
ROBOSUITE_EPISODE_START_ARTIFACT_SCHEMA_A1 = (
    "dexteleop.robosuite.episode_start_artifact.a1"
)
ROBOSUITE_EPISODE_START_ARTIFACT_HASH_DOMAIN_A1 = (
    "dexteleop.robosuite.episode_start_artifact.a1"
)

PANDA_LIFT_OSC_POSE_OBSERVATION_KEYS_A1 = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat_site",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "cube_pos",
    "cube_quat",
    "gripper_to_cube_pos",
)
PANDA_LIFT_OSC_POSE_ACTION_LAYOUT_A1 = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
    "gripper",
)
PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1 = MappingProxyType(
    {
        "composite_name": "BASIC",
        "part_names": ("right", "right_gripper"),
        "action_splits": (("right", (0, 6)), ("right_gripper", (6, 7))),
        "arm_name": "OSC_POSE",
        "control_dimension": 6,
        "use_orientation": True,
        "input_type": "delta",
        "input_reference_frame": "base",
        "goal_update_mode": "achieved",
        "impedance_mode": "fixed",
        "input_low": (-1.0,) * 6,
        "input_high": (1.0,) * 6,
        "output_low": (-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        "output_high": (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        "control_low": (-1.0,) * 6,
        "control_high": (1.0,) * 6,
        "action_scale": (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        "action_input_transform": (0.0,) * 6,
        "action_output_transform": (0.0,) * 6,
        "interpolator_position": None,
        "interpolator_orientation": None,
        "kp": (150.0,) * 6,
        "kd": (2.0 * math.sqrt(150.0),) * 6,
        "uncoupling": True,
        "position_limits": None,
        "orientation_limits": None,
        "control_frequency_hz": 20,
        "lite_physics": True,
    }
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _freeze_primitive(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float.")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_primitive(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _required(key, name=f"{name}.key")
            if normalized_key in normalized:
                raise ValueError(f"{name} contains duplicate normalized keys.")
            normalized[normalized_key] = _freeze_primitive(
                item,
                name=f"{name}[{normalized_key!r}]",
            )
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(
        f"{name} accepts only explicit primitive reset options; "
        f"got {type(value).__name__}."
    )


def _finite_vector(
    values: tuple[float, ...] | list[float],
    *,
    name: str,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values.")
    return normalized


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_digest(value: object, *, name: str) -> str:
    normalized = _required(value, name=name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _sha256_f64(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


@dataclass(frozen=True)
class RobosuiteEnvironmentContractA1:
    """Exact sim-only Panda / Lift / OSC_POSE runtime profile.

    A1 is intentionally one closed profile. Supporting another robot, task,
    controller, renderer, or dependency set requires another reviewed contract
    instead of a permissive option being silently ignored.
    """

    environment_id: str = "Lift"
    robot: str = "Panda"
    composite_controller: str = "BASIC"
    arm_controller: str = "OSC_POSE"
    controller_input_type: str = "delta"
    controller_input_reference_frame: str = "base"
    action_layout: tuple[str, ...] = PANDA_LIFT_OSC_POSE_ACTION_LAYOUT_A1
    action_low: tuple[float, ...] = (-1.0,) * 7
    action_high: tuple[float, ...] = (1.0,) * 7
    observation_keys: tuple[str, ...] = PANDA_LIFT_OSC_POSE_OBSERVATION_KEYS_A1
    control_frequency_hz: int = 20
    has_renderer: bool = False
    has_offscreen_renderer: bool = False
    use_camera_observations: bool = False
    initialization_noise: None = None
    reward_shaping: bool = True
    lite_physics: bool = True
    hard_reset: bool = True
    ignore_done: bool = False
    python_version: str = "3.10"
    robosuite_version: str = "1.5.2"
    mujoco_version: str = "3.9.0"
    numpy_version: str = "1.26.4"
    scipy_version: str = "1.13.1"
    schema_id: str = ROBOSUITE_ENVIRONMENT_CONTRACT_SCHEMA_A1
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        expected = {
            "environment_id": "Lift",
            "robot": "Panda",
            "composite_controller": "BASIC",
            "arm_controller": "OSC_POSE",
            "controller_input_type": "delta",
            "controller_input_reference_frame": "base",
            "action_layout": PANDA_LIFT_OSC_POSE_ACTION_LAYOUT_A1,
            "action_low": (-1.0,) * 7,
            "action_high": (1.0,) * 7,
            "observation_keys": PANDA_LIFT_OSC_POSE_OBSERVATION_KEYS_A1,
            "control_frequency_hz": 20,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_observations": False,
            "initialization_noise": None,
            "reward_shaping": True,
            "lite_physics": True,
            "hard_reset": True,
            "ignore_done": False,
            "python_version": "3.10",
            "robosuite_version": "1.5.2",
            "mujoco_version": "3.9.0",
            "numpy_version": "1.26.4",
            "scipy_version": "1.13.1",
            "schema_id": ROBOSUITE_ENVIRONMENT_CONTRACT_SCHEMA_A1,
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    f"A1 requires {name}={expected_value!r}; "
                    f"got {getattr(self, name)!r}."
                )
        payload = dict(expected)
        payload.update(
            {
                "active_controller_profile": (
                    PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1
                ),
                "reset_semantics": "compiled_xml_plus_flattened_initial_state",
                "replay_semantics": "restore_initial_state_then_replay_full_prefix",
                "mid_episode_state_resume": "forbidden_unproven",
                "rendering": "forbidden",
            }
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_ENVIRONMENT_CONTRACT_HASH_DOMAIN_A1,
            ),
        )

    @property
    def contract_id(self) -> str:
        return f"robosuite-environment-contract:{self.semantic_content_hash}"

    def make_reset_manifest(
        self,
        *,
        task_id: str,
        episode_id: str,
        reset_sequence: int,
        seed: int,
        max_episode_steps: int,
    ) -> RobosuiteResetManifest:
        return RobosuiteResetManifest(
            environment_id=self.environment_id,
            environment_contract_hash=self.semantic_content_hash,
            task_id=task_id,
            episode_id=episode_id,
            reset_sequence=reset_sequence,
            seed=seed,
            action_dimension=len(self.action_layout),
            observation_keys=self.observation_keys,
            max_episode_steps=max_episode_steps,
        )


@dataclass(frozen=True)
class RobosuiteEpisodeStartArtifactA1:
    """Replayable episode start captured after a real robosuite reset.

    It deliberately represents only the episode start. A flattened MuJoCo
    state captured in the middle of an episode is not sufficient to restore a
    stateful OSC controller and is outside this contract.
    """

    manifest_hash: str
    environment_contract_hash: str
    runtime_fingerprint: Mapping[str, Any]
    environment_kwargs: Mapping[str, Any]
    controller_config: Mapping[str, Any]
    active_controller_profile: Mapping[str, Any]
    compiled_model_xml: str
    effective_model_xml_sha256: str
    initial_state: tuple[float, ...]
    episode_metadata: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    model_dimensions: Mapping[str, int]
    model_timestep: float
    control_timestep: float
    schema_id: str = ROBOSUITE_EPISODE_START_ARTIFACT_SCHEMA_A1
    compiled_model_xml_sha256: str = field(init=False)
    initial_state_f64_sha256: str = field(init=False)
    controller_config_hash: str = field(init=False)
    active_controller_profile_hash: str = field(init=False)
    environment_kwargs_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_id != ROBOSUITE_EPISODE_START_ARTIFACT_SCHEMA_A1:
            raise ValueError(f"Unsupported episode-start schema {self.schema_id!r}.")
        manifest_hash = _required(self.manifest_hash, name="manifest_hash")
        contract_hash = _required(
            self.environment_contract_hash,
            name="environment_contract_hash",
        )
        runtime_fingerprint = _freeze_primitive(
            dict(self.runtime_fingerprint),
            name="runtime_fingerprint",
        )
        if not runtime_fingerprint:
            raise ValueError("runtime_fingerprint must not be empty.")
        environment_kwargs = _freeze_primitive(
            dict(self.environment_kwargs),
            name="environment_kwargs",
        )
        controller_config = _freeze_primitive(
            dict(self.controller_config),
            name="controller_config",
        )
        active_controller_profile = _freeze_primitive(
            dict(self.active_controller_profile),
            name="active_controller_profile",
        )
        if not active_controller_profile:
            raise ValueError("active_controller_profile must not be empty.")
        compiled_model_xml = _required(
            self.compiled_model_xml,
            name="compiled_model_xml",
        )
        effective_xml_hash = _sha256_digest(
            self.effective_model_xml_sha256,
            name="effective_model_xml_sha256",
        )
        initial_state = _finite_vector(self.initial_state, name="initial_state")
        episode_metadata = _freeze_primitive(
            dict(self.episode_metadata),
            name="episode_metadata",
        )
        rng_state = _freeze_primitive(dict(self.rng_state), name="rng_state")
        if not rng_state:
            raise ValueError("rng_state must not be empty.")
        action_low = _finite_vector(self.action_low, name="action_low")
        action_high = _finite_vector(self.action_high, name="action_high")
        if len(action_low) != len(action_high) or any(
            lower >= upper for lower, upper in zip(action_low, action_high, strict=True)
        ):
            raise ValueError("action bounds must have equal size and low < high.")
        model_dimensions_raw = dict(self.model_dimensions)
        if not model_dimensions_raw:
            raise ValueError("model_dimensions must not be empty.")
        model_dimensions: dict[str, int] = {}
        for name, value in model_dimensions_raw.items():
            key = _required(name, name="model_dimensions.key")
            model_dimensions[key] = _nonnegative_int(
                value,
                name=f"model_dimensions[{key!r}]",
            )
        model_dimensions_frozen = MappingProxyType(
            dict(sorted(model_dimensions.items()))
        )
        model_timestep = float(self.model_timestep)
        control_timestep = float(self.control_timestep)
        if (
            not math.isfinite(model_timestep)
            or not math.isfinite(control_timestep)
            or model_timestep <= 0.0
            or control_timestep <= 0.0
        ):
            raise ValueError("model and control timesteps must be finite and positive.")
        xml_hash = _sha256_text(compiled_model_xml)
        state_hash = _sha256_f64(initial_state)
        controller_hash = canonical_content_hash(
            controller_config,
            domain="dexteleop.robosuite.controller_config.a1",
        )
        active_controller_hash = canonical_content_hash(
            active_controller_profile,
            domain="dexteleop.robosuite.active_controller_profile.a1",
        )
        environment_kwargs_hash = canonical_content_hash(
            environment_kwargs,
            domain="dexteleop.robosuite.environment_kwargs.a1",
        )
        payload = {
            "schema_id": self.schema_id,
            "manifest_hash": manifest_hash,
            "environment_contract_hash": contract_hash,
            "runtime_fingerprint": runtime_fingerprint,
            "environment_kwargs_hash": environment_kwargs_hash,
            "controller_config_hash": controller_hash,
            "active_controller_profile_hash": active_controller_hash,
            "compiled_model_xml_sha256": xml_hash,
            "effective_model_xml_sha256": effective_xml_hash,
            "initial_state_shape": (len(initial_state),),
            "initial_state_dtype": "<f8",
            "initial_state_f64_sha256": state_hash,
            "episode_metadata": episode_metadata,
            "rng_state": rng_state,
            "action_low": action_low,
            "action_high": action_high,
            "model_dimensions": model_dimensions_frozen,
            "model_timestep": model_timestep,
            "control_timestep": control_timestep,
            "restore_semantics": "episode_start_only_full_prefix_required",
        }
        normalized = {
            "manifest_hash": manifest_hash,
            "environment_contract_hash": contract_hash,
            "runtime_fingerprint": runtime_fingerprint,
            "environment_kwargs": environment_kwargs,
            "controller_config": controller_config,
            "active_controller_profile": active_controller_profile,
            "compiled_model_xml": compiled_model_xml,
            "effective_model_xml_sha256": effective_xml_hash,
            "initial_state": initial_state,
            "episode_metadata": episode_metadata,
            "rng_state": rng_state,
            "action_low": action_low,
            "action_high": action_high,
            "model_dimensions": model_dimensions_frozen,
            "model_timestep": model_timestep,
            "control_timestep": control_timestep,
            "compiled_model_xml_sha256": xml_hash,
            "initial_state_f64_sha256": state_hash,
            "controller_config_hash": controller_hash,
            "active_controller_profile_hash": active_controller_hash,
            "environment_kwargs_hash": environment_kwargs_hash,
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_EPISODE_START_ARTIFACT_HASH_DOMAIN_A1,
            ),
        )

    @property
    def artifact_id(self) -> str:
        return f"robosuite-episode-start:{self.semantic_content_hash}"


@dataclass(frozen=True)
class RobosuiteResetManifest:
    """One deterministic reset request that can be replayed by exact hash.

    The manifest does not claim that a real robosuite environment was started.
    A backend receipt is required to prove that a reset was actually applied.
    """

    environment_id: str
    environment_contract_hash: str
    task_id: str
    episode_id: str
    reset_sequence: int
    seed: int
    action_dimension: int
    observation_keys: tuple[str, ...]
    max_episode_steps: int
    options: Mapping[str, Any] = field(default_factory=dict)
    schema_id: str = ROBOSUITE_RESET_MANIFEST_SCHEMA_A0
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_id = _required(self.schema_id, name="schema_id")
        if schema_id != ROBOSUITE_RESET_MANIFEST_SCHEMA_A0:
            raise ValueError(f"Unsupported robosuite reset schema {schema_id!r}.")
        environment_id = _required(self.environment_id, name="environment_id")
        environment_hash = _required(
            self.environment_contract_hash,
            name="environment_contract_hash",
        )
        task_id = _required(self.task_id, name="task_id")
        episode_id = _required(self.episode_id, name="episode_id")
        reset_sequence = _nonnegative_int(
            self.reset_sequence,
            name="reset_sequence",
        )
        seed = _nonnegative_int(self.seed, name="seed")
        action_dimension = _positive_int(
            self.action_dimension,
            name="action_dimension",
        )
        max_episode_steps = _positive_int(
            self.max_episode_steps,
            name="max_episode_steps",
        )
        observation_keys = tuple(
            _required(key, name="observation_keys.item")
            for key in self.observation_keys
        )
        if not observation_keys:
            raise ValueError("observation_keys must not be empty.")
        if len(observation_keys) != len(set(observation_keys)):
            raise ValueError("observation_keys must be unique.")
        options = _freeze_primitive(dict(self.options), name="options")
        payload = {
            "schema_id": schema_id,
            "environment_id": environment_id,
            "environment_contract_hash": environment_hash,
            "task_id": task_id,
            "episode_id": episode_id,
            "reset_sequence": reset_sequence,
            "seed": seed,
            "action_dimension": action_dimension,
            "observation_keys": observation_keys,
            "max_episode_steps": max_episode_steps,
            "options": options,
            "reset_semantics": "seeded_full_environment_reset",
            "replay_binding": "exact_manifest_hash",
        }
        for name, value in payload.items():
            if name not in {"reset_semantics", "replay_binding"}:
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_RESET_MANIFEST_HASH_DOMAIN_A0,
            ),
        )

    @property
    def manifest_id(self) -> str:
        return f"robosuite-reset-manifest:{self.semantic_content_hash}"

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_id": self.schema_id,
                "environment_id": self.environment_id,
                "environment_contract_hash": self.environment_contract_hash,
                "task_id": self.task_id,
                "episode_id": self.episode_id,
                "reset_sequence": self.reset_sequence,
                "seed": self.seed,
                "action_dimension": self.action_dimension,
                "observation_keys": self.observation_keys,
                "max_episode_steps": self.max_episode_steps,
                "options": self.options,
            }
        )


@dataclass(frozen=True)
class RobosuiteResetReceipt:
    """Deterministic proof that one backend reset used one exact manifest."""

    backend_id: str
    manifest_id: str
    manifest_hash: str
    initial_observation_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "backend_id": _required(self.backend_id, name="backend_id"),
            "manifest_id": _required(self.manifest_id, name="manifest_id"),
            "manifest_hash": _required(self.manifest_hash, name="manifest_hash"),
            "initial_observation_hash": _required(
                self.initial_observation_hash,
                name="initial_observation_hash",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_RESET_RECEIPT_HASH_DOMAIN_A0,
            ),
        )

    @property
    def receipt_id(self) -> str:
        return f"robosuite-reset-receipt:{self.semantic_content_hash}"


__all__ = [
    "PANDA_LIFT_OSC_POSE_ACTION_LAYOUT_A1",
    "PANDA_LIFT_OSC_POSE_ACTIVE_CONTROLLER_PROFILE_A1",
    "PANDA_LIFT_OSC_POSE_OBSERVATION_KEYS_A1",
    "ROBOSUITE_ENVIRONMENT_CONTRACT_HASH_DOMAIN_A1",
    "ROBOSUITE_ENVIRONMENT_CONTRACT_SCHEMA_A1",
    "ROBOSUITE_EPISODE_START_ARTIFACT_HASH_DOMAIN_A1",
    "ROBOSUITE_EPISODE_START_ARTIFACT_SCHEMA_A1",
    "ROBOSUITE_RESET_MANIFEST_HASH_DOMAIN_A0",
    "ROBOSUITE_RESET_MANIFEST_SCHEMA_A0",
    "ROBOSUITE_RESET_RECEIPT_HASH_DOMAIN_A0",
    "RobosuiteEnvironmentContractA1",
    "RobosuiteEpisodeStartArtifactA1",
    "RobosuiteResetManifest",
    "RobosuiteResetReceipt",
]
