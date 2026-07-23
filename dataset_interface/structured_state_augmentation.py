"""Physical-unit-aware augmentation for absolute v2.2 observation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.pose_frames import PoseFrameContract
from dataset_contracts.state_input import (
    STATE_INPUT_MODE_RAW,
    StateInputMode,
    normalize_state_input_mode,
    uses_state_mask,
)
from dataset_contracts.v22 import PolicyIOLayout, SIDES


StateSampleScope = Literal["frame", "clip"]


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1].")


@dataclass(frozen=True)
class StructuredStateAugmentationConfig:
    """SE(3) and end-effector state augmentation before model normalization.

    This configuration is enabled by default because it preserves the v2.2
    model dimension and simulates small physical sensor/control noise. Dropout
    remains disabled by default; enabling any dropout requires a masked state
    artifact so zero cannot be confused with a valid pose/scalar.
    """

    enabled: bool = True
    global_seed: int = 20260715
    sample_scope: StateSampleScope = "frame"
    arm_translation_std_m: float = 0.001
    arm_rotation_std_deg: float = 0.75
    gripper_noise_std: float = 0.01
    gripper_clip_range: tuple[float, float] | None = None
    side_dropout_probability: float = 0.0
    arm_dropout_probability: float = 0.0
    end_effector_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if str(self.sample_scope).strip().lower() not in {"frame", "clip"}:
            raise ValueError("sample_scope must be 'frame' or 'clip'.")
        for name in (
            "arm_translation_std_m",
            "arm_rotation_std_deg",
            "gripper_noise_std",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name in (
            "side_dropout_probability",
            "arm_dropout_probability",
            "end_effector_dropout_probability",
        ):
            _check_probability(name, float(getattr(self, name)))
        clip_range = None
        if self.gripper_clip_range is not None:
            low, high = (float(value) for value in self.gripper_clip_range)
            if low > high:
                raise ValueError("gripper_clip_range must be ordered.")
            clip_range = (low, high)
        object.__setattr__(self, "sample_scope", str(self.sample_scope).strip().lower())
        object.__setattr__(self, "gripper_clip_range", clip_range)

    @property
    def requires_masked_state_input(self) -> bool:
        return bool(
            self.enabled
            and (
                self.side_dropout_probability > 0.0
                or self.arm_dropout_probability > 0.0
                or self.end_effector_dropout_probability > 0.0
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop_structured_state_augmentation_v1",
            "enabled": bool(self.enabled),
            "global_seed": int(self.global_seed),
            "sample_scope": self.sample_scope,
            "pose": {
                "space": "canonical_desk_se3",
                "translation_std_m": float(self.arm_translation_std_m),
                "rotation_std_deg": float(self.arm_rotation_std_deg),
                "composition": "delta_left_multiply",
            },
            "end_effector": {
                "noise_std": float(self.gripper_noise_std),
                "clip_range": (
                    None if self.gripper_clip_range is None else list(self.gripper_clip_range)
                ),
            },
            "dropout": {
                "side_probability": float(self.side_dropout_probability),
                "arm_probability": float(self.arm_dropout_probability),
                "end_effector_probability": float(
                    self.end_effector_dropout_probability
                ),
            },
            "excluded": [
                "action",
                "observation.last_action",
                "observation.state.discrete",
                "observation.arm.qpos",
            ],
        }


def _as_batch(state: Any, *, state_dim: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(state, dtype=np.float32)
    squeeze = values.ndim == 1
    if squeeze:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != int(state_dim):
        raise ValueError(
            f"raw state must have shape ({int(state_dim)},) or [batch, {int(state_dim)}], "
            f"got {np.asarray(state).shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("raw state must be finite before structured augmentation.")
    return values, squeeze


def _pose_component_slices(layout: PolicyIOLayout) -> dict[str, slice]:
    result: dict[str, slice] = {}
    for component in layout.components("state", "continuous"):
        if component.name.endswith(".arm.pose"):
            if component.side not in SIDES or component.dim != 7:
                raise ValueError(
                    "Each arm pose component must be a 7D left/right component."
                )
            result[str(component.side)] = slice(component.offset, component.stop)
    if set(result) != set(SIDES):
        raise ValueError(
            "Structured state augmentation requires both left and right arm poses."
        )
    return result


def _end_effector_slices(layout: PolicyIOLayout) -> dict[str, tuple[slice, ...]]:
    result: dict[str, list[slice]] = {side: [] for side in SIDES}
    for component in layout.components("state", "continuous"):
        if component.name.endswith(".arm.pose"):
            continue
        if component.side not in result:
            raise ValueError(f"Unexpected state component side {component.side!r}.")
        result[str(component.side)].append(slice(component.offset, component.stop))
    return {side: tuple(values) for side, values in result.items()}


def _gripper_clip_range(
    layout: PolicyIOLayout,
    side: str,
    *,
    config: StructuredStateAugmentationConfig,
) -> tuple[float, float] | None:
    if config.gripper_clip_range is not None:
        return config.gripper_clip_range
    channel = layout.channel("state", side)
    if str(channel.units or "").strip().lower() == "normalized_0_1":
        return (0.0, 1.0)
    return None


def _set_mask(
    mask: np.ndarray,
    rows: np.ndarray,
    slices: slice | tuple[slice, ...],
) -> None:
    if isinstance(slices, slice):
        mask[rows, slices] = 0.0
        return
    for value in slices:
        mask[rows, value] = 0.0


@dataclass(frozen=True)
class StructuredStateAugmentationResult:
    state_augmented: np.ndarray
    mask: np.ndarray
    translation_noise_m: np.ndarray
    rotation_noise_rad: np.ndarray
    end_effector_noise: np.ndarray

    def __post_init__(self) -> None:
        state = np.asarray(self.state_augmented, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=np.float32)
        translation = np.asarray(self.translation_noise_m, dtype=np.float32)
        rotation = np.asarray(self.rotation_noise_rad, dtype=np.float32)
        end_effector = np.asarray(self.end_effector_noise, dtype=np.float32)
        if state.ndim not in (1, 2) or mask.shape != state.shape:
            raise ValueError(
                "state_augmented and mask must share [dim] or [batch, dim] shape."
            )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(mask)):
            raise ValueError("structured state result must be finite.")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("structured state mask must contain only 0 and 1.")
        if (
            translation.shape[-2:] != (len(SIDES), 3)
            or rotation.shape[-2:] != (len(SIDES), 3)
        ):
            raise ValueError("pose noise must have [..., 2, 3] shape.")
        if end_effector.ndim < 2 or end_effector.shape[-2] != len(SIDES):
            raise ValueError("end_effector_noise must have [..., 2, dim] shape.")
        object.__setattr__(self, "state_augmented", state.copy())
        object.__setattr__(self, "mask", mask.copy())
        object.__setattr__(self, "translation_noise_m", translation.copy())
        object.__setattr__(self, "rotation_noise_rad", rotation.copy())
        object.__setattr__(self, "end_effector_noise", end_effector.copy())


class StructuredStateInputAdapter:
    """Compose raw structured state noise with the existing normalizer stage.

    The normalizer remains owned by the model/trainer. This adapter only
    defines the ordering and carries the structured availability mask into the
    existing normalized-state augmenter.
    """

    def __init__(
        self,
        *,
        layout: PolicyIOLayout,
        pose_frame_contract: PoseFrameContract | None = None,
        structured_config: StructuredStateAugmentationConfig | None = None,
        normalized_config: Any | None = None,
        state_input_mode: StateInputMode | str = STATE_INPUT_MODE_RAW,
    ) -> None:
        from .state_augmentation import (
            NormalizedStateInputAdapter,
            StateAugmentationConfig,
        )

        self.layout = layout
        self.pose_frame_contract = pose_frame_contract
        self.structured_config = (
            StructuredStateAugmentationConfig()
            if structured_config is None
            else structured_config
        )
        self.normalized_config = (
            StateAugmentationConfig() if normalized_config is None else normalized_config
        )
        self.state_input_mode = normalize_state_input_mode(state_input_mode)
        self._normalized_adapter = NormalizedStateInputAdapter(
            layout=layout,
            state_input_mode=self.state_input_mode,
            config=self.normalized_config,
        )
        if self.structured_config.requires_masked_state_input and not uses_state_mask(
            self.state_input_mode
        ):
            raise ValueError(
                "Structured state dropout requires "
                "state_input_mode='masked_normalized_concat_v1'."
            )

    def apply_raw(
        self,
        state: Any,
        *,
        rng: np.random.Generator | None = None,
    ) -> StructuredStateAugmentationResult:
        return augment_structured_state(
            state,
            layout=self.layout,
            pose_frame_contract=self.pose_frame_contract,
            config=self.structured_config,
            rng=rng,
        )

    def model_input(
        self,
        state: Any,
        normalize: Any,
        *,
        rng: np.random.Generator | None = None,
        normalized_rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        structured = self.apply_raw(state, rng=rng)
        normalized = normalize(structured.state_augmented)
        return self._normalized_adapter.model_input(
            normalized,
            rng=normalized_rng,
            initial_mask=structured.mask,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop_state_augmentation_pipeline_v1",
            "structured": self.structured_config.to_dict(),
            "normalized": self.normalized_config.to_dict(),
            "state_input_mode": self.state_input_mode,
            "order": [
                "raw_structured_state",
                "model_normalizer",
                "normalized_state_augmentation",
            ],
        }


def augment_structured_state(
    state: Any,
    *,
    layout: PolicyIOLayout,
    pose_frame_contract: PoseFrameContract | None = None,
    config: StructuredStateAugmentationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> StructuredStateAugmentationResult:
    """Apply physical-unit-aware noise to raw absolute observation state.

    Pose noise is sampled and composed in canonical desk coordinates, then
    converted back to the dataset storage frame using the declared contract.
    The action and all audit/discrete channels are intentionally outside this
    function.
    """

    resolved = StructuredStateAugmentationConfig() if config is None else config
    values, squeeze = _as_batch(state, state_dim=layout.state_dim)
    mask = np.ones_like(values, dtype=np.float32)
    pose_slices = _pose_component_slices(layout)
    effector_slices = _end_effector_slices(layout)
    translation_noise = np.zeros(
        (values.shape[0], len(SIDES), 3), dtype=np.float32
    )
    rotation_noise = np.zeros(
        (values.shape[0], len(SIDES), 3), dtype=np.float32
    )
    effector_dim = max(
        (
            component.dim
            for component in layout.components("state", "continuous")
            if not component.name.endswith(".arm.pose")
        ),
        default=0,
    )
    effector_noise = np.zeros(
        (values.shape[0], len(SIDES), effector_dim), dtype=np.float32
    )
    if not resolved.enabled:
        if squeeze:
            return StructuredStateAugmentationResult(
                state_augmented=values[0],
                mask=mask[0],
                translation_noise_m=translation_noise[0],
                rotation_noise_rad=rotation_noise[0],
                end_effector_noise=effector_noise[0],
            )
        return StructuredStateAugmentationResult(
            state_augmented=values,
            mask=mask,
            translation_noise_m=translation_noise,
            rotation_noise_rad=rotation_noise,
            end_effector_noise=effector_noise,
        )

    random = np.random.default_rng() if rng is None else rng
    contract = pose_frame_contract or PoseFrameContract.legacy_v22_raw(
        active_sides=SIDES
    )
    output = values.copy()
    sample_count = 1 if resolved.sample_scope == "clip" else values.shape[0]
    for side_index, side in enumerate(SIDES):
        translations = random.normal(
            0.0,
            float(resolved.arm_translation_std_m),
            size=(sample_count, 3),
        ).astype(np.float32)
        rotations = random.normal(
            0.0,
            float(np.deg2rad(resolved.arm_rotation_std_deg)),
            size=(sample_count, 3),
        ).astype(np.float32)
        if resolved.sample_scope == "clip":
            translations = np.repeat(translations, values.shape[0], axis=0)
            rotations = np.repeat(rotations, values.shape[0], axis=0)
        translation_noise[:, side_index, :] = translations
        rotation_noise[:, side_index, :] = rotations
        pose_slice = pose_slices[side]
        for row_index in range(values.shape[0]):
            source_pose = values[row_index, pose_slice]
            canonical = contract.storage_pose_vector_to_canonical_desk(
                source_pose, side=side
            )
            source_rotation = Rotation.from_quat(canonical[3:7])
            delta_rotation = Rotation.from_rotvec(rotations[row_index])
            augmented_rotation = delta_rotation * source_rotation
            augmented_pose = np.asarray(canonical, dtype=np.float32).copy()
            augmented_pose[:3] += translations[row_index]
            augmented_pose[3:7] = augmented_rotation.as_quat().astype(np.float32)
            restored = contract.canonical_desk_pose_vector_to_storage(
                augmented_pose, side=side
            )
            if float(np.dot(restored[3:7], source_pose[3:7])) < 0.0:
                restored[3:7] *= -1.0
            output[row_index, pose_slice] = restored

        side_dropped = (
            random.random(values.shape[0])
            < float(resolved.side_dropout_probability)
        )
        arm_dropped = (
            random.random(values.shape[0])
            < float(resolved.arm_dropout_probability)
        )
        _set_mask(mask, side_dropped | arm_dropped, pose_slices[side])

        for component_index, component_slice in enumerate(effector_slices[side]):
            component_dim = component_slice.stop - component_slice.start
            noise = random.normal(
                0.0,
                float(resolved.gripper_noise_std),
                size=(values.shape[0], component_dim),
            ).astype(np.float32)
            output[:, component_slice] += noise
            effector_noise[:, side_index, :component_dim] = noise
            if component_dim > 0:
                clip_range = _gripper_clip_range(
                    layout, side, config=resolved
                )
                if clip_range is not None:
                    low, high = clip_range
                    output[:, component_slice] = np.clip(
                        output[:, component_slice], low, high
                    )
            dropped = (
                random.random(values.shape[0])
                < float(resolved.end_effector_dropout_probability)
            )
            _set_mask(mask, side_dropped | dropped, component_slice)

    output *= mask
    if squeeze:
        return StructuredStateAugmentationResult(
            state_augmented=output[0],
            mask=mask[0],
            translation_noise_m=translation_noise[0],
            rotation_noise_rad=rotation_noise[0],
            end_effector_noise=effector_noise[0],
        )
    return StructuredStateAugmentationResult(
        state_augmented=output,
        mask=mask,
        translation_noise_m=translation_noise,
        rotation_noise_rad=rotation_noise,
        end_effector_noise=effector_noise,
    )


__all__ = [
    "StateSampleScope",
    "StructuredStateAugmentationConfig",
    "StructuredStateInputAdapter",
    "StructuredStateAugmentationResult",
    "augment_structured_state",
]
