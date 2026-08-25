"""State-only policy observation adapter for the MuJoCo deployment seam.

The FormalCollection learner consumes the 16-dimensional v2.2 state (TCP
poses plus measured gripper scalars).  MuJoCo's neutral observation source
still exposes the canonical 38-dimensional deployment observation, so this
module performs the one explicit projection at the model boundary.  It never
steps physics or dispatches an action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.deploy import (
    DeploymentObservation,
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    V22DeploymentAdapter,
    prepare_observation_for_policy_pose_mode,
)
from sim.mujoco.dataset_v22.layout import load_rm65_bimanual_dmtac_layout


# Kept local to the simulator adapter so the production ``sim`` package does
# not import the experiment package.  The value is the exact layout identity
# shared by FormalCollection v2.2 and ``load_rm65_bimanual_dmtac_layout``.
STATE_BC_LAYOUT_HASH_V1 = (
    "sha256:52e4bc42ad0a0e562c357e41da1ee2cbd827b4dc98643ffb73fe7acfeafc66ab"
)
STATE_BC_SOURCE_LAYOUT_HASH_V1 = (
    "sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16"
)
STATE_BC_ACCEPTED_LAYOUT_HASHES = frozenset(
    {STATE_BC_LAYOUT_HASH_V1, STATE_BC_SOURCE_LAYOUT_HASH_V1}
)
STATE_BC_STATE_DIM = 16
STATE_BC_ACTION_DIM = 16
STATE_BC_SCHEMA_V1 = "dexteleop.state_bc_experiment.v1"


def _normalize_sides(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(value).strip().lower() for value in values)
    if set(requested) != {"left", "right"} or len(requested) != 2:
        raise ValueError("state BC observation requires active_sides=('left', 'right').")
    return ("left", "right")


@dataclass(frozen=True)
class StatePolicyObservationConfig:
    """Explicit model-frame and layout binding for one observation adapter."""

    active_sides: tuple[str, ...] = ("left", "right")
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None
    expected_layout_hash: str = STATE_BC_LAYOUT_HASH_V1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        frame = str(self.policy_absolute_frame).strip().lower()
        if frame not in {
            POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
            POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
        }:
            raise ValueError(f"unsupported policy_absolute_frame {self.policy_absolute_frame!r}.")
        expected = str(self.expected_layout_hash).strip()
        if not expected:
            raise ValueError("expected_layout_hash must be non-empty.")
        object.__setattr__(self, "policy_absolute_frame", frame)
        object.__setattr__(self, "expected_layout_hash", expected)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StatePolicyObservation:
    """Validated 16D model input with provenance for one simulator step."""

    state: np.ndarray
    timestamp: float
    step_index: int
    active_sides: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        if state.shape != (STATE_BC_STATE_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError(f"state BC observation must be finite with shape ({STATE_BC_STATE_DIM},).")
        timestamp = float(self.timestamp)
        step_index = int(self.step_index)
        if not np.isfinite(timestamp) or step_index < 0:
            raise ValueError("state BC observation timestamp/step_index is invalid.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        object.__setattr__(self, "metadata", dict(self.metadata))


class StatePolicyObservationAdapter:
    """Project a deployment observation into the FormalCollection state."""

    def __init__(self, config: StatePolicyObservationConfig | None = None) -> None:
        self.config = StatePolicyObservationConfig() if config is None else config
        self.layout = load_rm65_bimanual_dmtac_layout()
        if self.layout.layout_hash != STATE_BC_LAYOUT_HASH_V1:
            raise ValueError(
                "state BC layout binding mismatch: "
                f"expected_runtime={STATE_BC_LAYOUT_HASH_V1!r}, "
                f"actual={self.layout.layout_hash!r}."
            )
        if self.config.expected_layout_hash not in STATE_BC_ACCEPTED_LAYOUT_HASHES:
            raise ValueError(
                "state BC expected layout hash is not an accepted source/runtime alias: "
                f"{self.config.expected_layout_hash!r}."
            )
        self._v22 = V22DeploymentAdapter(self.layout)

    def capture(self, observation: DeploymentObservation) -> StatePolicyObservation:
        """Build one immutable model view; no physics or runtime state changes."""

        if not isinstance(observation, DeploymentObservation):
            raise TypeError("observation must be a DeploymentObservation.")
        if not set(self.config.active_sides).issubset(set(observation.active_sides)):
            raise ValueError(
                "deployment observation does not contain both state BC active sides: "
                f"{observation.active_sides!r}."
            )
        policy_view = prepare_observation_for_policy_pose_mode(
            observation,
            target_mode="absolute",
            active_sides=self.config.active_sides,
            policy_absolute_frame=self.config.policy_absolute_frame,
            runtime_pose_frame_contract=self.config.runtime_pose_frame_contract,
        )
        typed = self._v22.observation_from_deployment(policy_view)
        state = np.asarray(typed.state, dtype=np.float32).reshape(-1)
        if state.shape != (STATE_BC_STATE_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError("v2.2 deployment adapter returned an invalid state BC vector.")
        metadata = {
            **dict(self.config.metadata),
            "state_bc_schema": STATE_BC_SCHEMA_V1,
            "policy_io_layout_hash": self.layout.layout_hash,
            "policy_absolute_frame": self.config.policy_absolute_frame,
            "state_dim": STATE_BC_STATE_DIM,
            "state_source": "deployment_observation_v22",
        }
        if self.config.runtime_pose_frame_contract is not None:
            contract = self.config.runtime_pose_frame_contract
            metadata["runtime_pose_frame_contract"] = (
                contract.to_dict() if isinstance(contract, PoseFrameContract) else dict(contract)
            )
        return StatePolicyObservation(
            state=state,
            timestamp=typed.timestamp,
            step_index=typed.step_index,
            active_sides=self.config.active_sides,
            metadata=metadata,
        )


__all__ = [
    "STATE_BC_LAYOUT_HASH_V1",
    "STATE_BC_SOURCE_LAYOUT_HASH_V1",
    "STATE_BC_ACCEPTED_LAYOUT_HASHES",
    "STATE_BC_ACTION_DIM",
    "STATE_BC_SCHEMA_V1",
    "STATE_BC_STATE_DIM",
    "StatePolicyObservation",
    "StatePolicyObservationAdapter",
    "StatePolicyObservationConfig",
]
