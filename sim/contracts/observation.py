from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from robot_motion.deploy.observation import DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
from robot_motion.deploy.schema import OBSERVATION_STATE_DIM

from .versions import OBSERVATION_SPEC_VERSION, require_version


OBSERVATION_STATE_LAYOUT_VERSION = "deploy_observation_state.qpos6_pose7_hand6.v1"


@dataclass(frozen=True)
class ObservationSpec:
    """Versioned policy-observation layout shared by simulated and real sources."""

    schema_version: str = OBSERVATION_SPEC_VERSION
    state_layout_version: str = OBSERVATION_STATE_LAYOUT_VERSION
    state_dim: int = OBSERVATION_STATE_DIM
    state_dtype: str = "float32"
    state_semantics: str = "measured_qpos_tcp_pose_hand_state"
    image_keys: tuple[str, ...] = ("front", "left_wrist", "right_wrist")
    image_layout: str = "HWC"
    image_dtype: str = "uint8"
    image_color_space: str = "RGB"
    image_timestamp_policy: str = "latest_not_after_observation"
    max_image_age_sec: float = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC

    def __post_init__(self) -> None:
        require_version(
            self.schema_version,
            OBSERVATION_SPEC_VERSION,
            field="observation spec version",
        )
        if self.state_layout_version != OBSERVATION_STATE_LAYOUT_VERSION:
            raise ValueError("Unsupported observation state layout.")
        if int(self.state_dim) != OBSERVATION_STATE_DIM or self.state_dtype != "float32":
            raise ValueError("ObservationSpec state disagrees with deploy schema.")
        if self.state_semantics != "measured_qpos_tcp_pose_hand_state":
            raise ValueError("ObservationSpec v1 requires measured state semantics.")
        if (self.image_layout, self.image_dtype, self.image_color_space) != (
            "HWC",
            "uint8",
            "RGB",
        ):
            raise ValueError("ObservationSpec v1 requires HWC uint8 RGB images.")
        if self.image_timestamp_policy != "latest_not_after_observation":
            raise ValueError("ObservationSpec v1 requires causal image selection.")
        keys = tuple(dict.fromkeys(str(key) for key in self.image_keys))
        if not keys:
            raise ValueError("ObservationSpec.image_keys must not be empty.")
        if float(self.max_image_age_sec) <= 0.0:
            raise ValueError("ObservationSpec.max_image_age_sec must be positive.")
        object.__setattr__(self, "image_keys", keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_layout_version": self.state_layout_version,
            "state_dim": int(self.state_dim),
            "state_dtype": self.state_dtype,
            "state_semantics": self.state_semantics,
            "image_keys": list(self.image_keys),
            "image_layout": self.image_layout,
            "image_dtype": self.image_dtype,
            "image_color_space": self.image_color_space,
            "image_timestamp_policy": self.image_timestamp_policy,
            "max_image_age_sec": float(self.max_image_age_sec),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationSpec":
        return cls(
            schema_version=str(payload.get("schema_version", OBSERVATION_SPEC_VERSION)),
            state_layout_version=str(
                payload.get("state_layout_version", OBSERVATION_STATE_LAYOUT_VERSION)
            ),
            state_dim=int(payload.get("state_dim", OBSERVATION_STATE_DIM)),
            state_dtype=str(payload.get("state_dtype", "float32")),
            state_semantics=str(
                payload.get("state_semantics", "measured_qpos_tcp_pose_hand_state")
            ),
            image_keys=tuple(payload.get("image_keys", ("front", "left_wrist", "right_wrist"))),
            image_layout=str(payload.get("image_layout", "HWC")),
            image_dtype=str(payload.get("image_dtype", "uint8")),
            image_color_space=str(payload.get("image_color_space", "RGB")),
            image_timestamp_policy=str(
                payload.get("image_timestamp_policy", "latest_not_after_observation")
            ),
            max_image_age_sec=float(
                payload.get("max_image_age_sec", DEFAULT_CAMERA_FRAME_MAX_AGE_SEC)
            ),
        )


__all__ = ["OBSERVATION_STATE_LAYOUT_VERSION", "ObservationSpec"]
