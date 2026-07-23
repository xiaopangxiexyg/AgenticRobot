"""Typed observation binding for a lower-owned replay emission stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.replay import ReplaySourceFrameEmission


REPLAY_SOURCE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2b_replay_source_binding.v1"
)


@dataclass(frozen=True)
class ReplaySourceObserverBinding:
    """Bind one replay stream without opening its dataset or owning its cursor."""

    source_slot_id: str
    source_locator_id: str
    expected_source_path: str
    source_episode_index: int
    group_key: str
    source_start_index: int
    clock_domain_id: str
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        source_slot_id: str,
        source_locator_id: str,
        expected_source_path: str,
        source_episode_index: int,
        group_key: str,
        source_start_index: int,
        clock_domain_id: str,
    ) -> "ReplaySourceObserverBinding":
        stable = {
            "source_slot_id": str(source_slot_id).strip(),
            "source_locator_id": str(source_locator_id).strip(),
            "source_episode_index": int(source_episode_index),
            "group_key": str(group_key).strip(),
            "source_start_index": int(source_start_index),
            "clock_domain_id": str(clock_domain_id).strip(),
        }
        if any(
            not stable[name]
            for name in (
                "source_slot_id",
                "source_locator_id",
                "group_key",
                "clock_domain_id",
            )
        ):
            raise ValueError("Replay observer binding identity must be non-empty.")
        if stable["source_episode_index"] < 0 or stable["source_start_index"] < 0:
            raise ValueError("Replay observer indices must be non-negative.")
        source_path = str(Path(expected_source_path).expanduser().resolve())
        if not source_path:
            raise ValueError("expected_source_path must be non-empty.")
        return cls(
            **stable,
            expected_source_path=source_path,
            semantic_content_hash=canonical_content_hash(
                {
                    **stable,
                    "local_path_semantics": "deployment_context_not_semantic_identity",
                },
                domain=REPLAY_SOURCE_BINDING_HASH_DOMAIN_V1,
            ),
        )

    def validate_emission(self, emission: ReplaySourceFrameEmission) -> None:
        if not isinstance(emission, ReplaySourceFrameEmission):
            raise TypeError("emission must be a ReplaySourceFrameEmission.")
        if str(Path(emission.source_path).expanduser().resolve()) != (
            self.expected_source_path
        ):
            raise ValueError("Replay emission belongs to another source path.")
        if emission.source_episode_index != self.source_episode_index:
            raise ValueError("Replay emission belongs to another source episode.")
        if emission.group_key != self.group_key:
            raise ValueError("Replay emission belongs to another replay group.")
        if emission.clock_domain_id != self.clock_domain_id:
            raise ValueError("Replay emission clock domain does not match binding.")
        expected_local = emission.source_action_index - self.source_start_index
        if expected_local < 0 or emission.local_frame_index != expected_local:
            raise ValueError("Replay emission source/local indices do not match binding.")


__all__ = [
    "REPLAY_SOURCE_BINDING_HASH_DOMAIN_V1",
    "ReplaySourceObserverBinding",
]
