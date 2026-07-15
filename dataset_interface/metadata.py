from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.v22 import (
    PolicyIOLayout,
    policy_io_layout_from_metadata,
    validate_policy_io_layout_binding,
)
from dataset_contracts.v23 import V23DatasetContract, resolve_v23_dataset_contract
from dataset_contracts.pose_frames import PoseFrameContract


MODALITY_VALIDITY_SCHEMA_VERSION = "dexteleop.modality_validity.v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


@dataclass(frozen=True)
class TrainingEligibility:
    """Whether an episode can supply the requested video views to training."""

    episode_index: int
    camera_keys: tuple[str, ...]
    eligible: bool
    reasons: tuple[str, ...] = ()

    def message(self) -> str:
        requested = ", ".join(self.camera_keys) if self.camera_keys else "episode data"
        detail = "; ".join(self.reasons) if self.reasons else "training policy rejected the episode"
        return f"episode={self.episode_index} views=[{requested}] {detail}"


@dataclass(frozen=True)
class InvalidTrainingDataError(RuntimeError):
    """Raised before an invalid episode/view is read by a training pipeline."""

    eligibility: TrainingEligibility

    def __str__(self) -> str:
        return f"invalid training data: {self.eligibility.message()}"


@dataclass(frozen=True)
class _ValidityStatus:
    valid: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class _EpisodeTrainingValidity:
    status: _ValidityStatus = field(default_factory=_ValidityStatus)
    video_by_camera: dict[str, _ValidityStatus] = field(default_factory=dict)


@dataclass(frozen=True)
class _TrainingValidityPolicy:
    status: _ValidityStatus = field(default_factory=_ValidityStatus)
    allowed_video_camera_keys: tuple[str, ...] | None = None
    episodes: dict[int, _EpisodeTrainingValidity] = field(default_factory=dict)


def _camera_keys_from_records(records: list["EpisodeRecord"]) -> tuple[str, ...]:
    """Return the stable union of cameras declared by episode metadata."""

    keys: set[str] = set()
    for record in records:
        paths = record.payload.get("video_paths")
        if isinstance(paths, Mapping):
            keys.update(str(key) for key, value in paths.items() if isinstance(value, str) and value)
    return tuple(sorted(keys))


def _status_from_mapping(value: Any, *, label: str) -> _ValidityStatus:
    if isinstance(value, bool):
        return _ValidityStatus(valid=value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a boolean or mapping with a boolean valid field.")
    valid = value.get("valid", True)
    if not isinstance(valid, bool):
        raise ValueError(f"{label}.valid must be a boolean when present.")
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"{label}.reason must be a string when present.")
    return _ValidityStatus(valid=valid, reason=None if reason is None else reason.strip() or None)


def _unique_camera_keys(values: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list of camera keys.")
    out: list[str] = []
    for value in values:
        key = str(value).strip()
        if not key:
            raise ValueError(f"{label} must not contain an empty camera key.")
        if key not in out:
            out.append(key)
    return tuple(out)


@dataclass(frozen=True)
class EpisodeRecord:
    dataset_root: Path
    payload: Mapping[str, Any]

    @property
    def episode_index(self) -> int:
        return int(self.payload["episode_index"])

    @property
    def frame_count(self) -> int:
        return int(self.payload.get("frame_count", 0) or 0)

    def video_path(self, camera_key: str) -> Path:
        video_paths = self.payload.get("video_paths")
        if not isinstance(video_paths, Mapping):
            raise KeyError(f"episode {self.episode_index} has no video_paths")
        value = video_paths.get(camera_key)
        if not isinstance(value, str) or not value:
            raise KeyError(f"episode {self.episode_index} has no video path for {camera_key!r}")
        path = Path(value)
        return path if path.is_absolute() else self.dataset_root / path

    def has_video(self, camera_key: str) -> bool:
        video_paths = self.payload.get("video_paths")
        if not isinstance(video_paths, Mapping):
            return False
        value = video_paths.get(str(camera_key))
        return isinstance(value, str) and bool(value)

    def explicit_mask_path(self, camera_key: str) -> Path | None:
        for key in ("final_mask_paths", "wrist_final_mask_paths", "wrist_keep_mask_paths"):
            paths = self.payload.get(key)
            if isinstance(paths, Mapping):
                value = paths.get(camera_key)
                if isinstance(value, str) and value:
                    path = Path(value)
                    return path if path.is_absolute() else self.dataset_root / path
        return None


class DexTeleopDatasetIndex:
    """Fast metadata resolver for DexTeleop/LeRobot-style datasets."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        self.info_path = self.dataset_root / "meta" / "info.json"
        self.modality_validity_path = self.dataset_root / "meta" / "modality_validity.json"
        self.info = _read_json(self.info_path)
        self.modality_validity = _read_json(self.modality_validity_path)
        records = [EpisodeRecord(self.dataset_root, row) for row in _read_jsonl(self.episodes_path)]
        self._episodes = {record.episode_index: record for record in records}
        if len(self._episodes) != len(records):
            raise ValueError(f"duplicate episode_index in {self.episodes_path}")
        self._declared_camera_keys = _camera_keys_from_records(records)
        self._training_validity = self._parse_training_validity()

    @property
    def dataset_name(self) -> str:
        return str(self.info.get("dataset_name") or self.dataset_root.name)

    @property
    def policy_io_layout(self) -> PolicyIOLayout:
        """Resolve the dataset's v2.2 layout or explicit legacy q6 adapter."""

        return policy_io_layout_from_metadata(self.info, allow_legacy_v21=True)

    @property
    def policy_io_layout_hash(self) -> str:
        return self.policy_io_layout.layout_hash

    @property
    def v23_contract(self) -> V23DatasetContract:
        """Resolve the canonical pose/source contract for this dataset.

        v2.2 and older datasets are exposed through the marked legacy RM65 raw
        compatibility view; v2.3 requires its explicit pose-frame metadata.
        """

        return resolve_v23_dataset_contract(self.info)

    @property
    def pose_frame_contract(self) -> PoseFrameContract:
        """Return the dataset's raw/desk storage-frame contract."""

        return self.v23_contract.pose_frames

    def validate_policy_io_layout(
        self,
        expected: PolicyIOLayout | str,
        *,
        allow_legacy_v21: bool = False,
    ) -> PolicyIOLayout:
        """Fail before a training reader mixes an incompatible model layout."""

        return validate_policy_io_layout_binding(
            expected,
            self.info,
            context=f"dataset:{self.dataset_root}",
            allow_legacy_v21=allow_legacy_v21,
        )

    @property
    def episode_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._episodes))

    @property
    def declared_camera_keys(self) -> tuple[str, ...]:
        return self._declared_camera_keys

    def episode(self, episode_index: int) -> EpisodeRecord:
        try:
            return self._episodes[int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"episode_index {episode_index} not found in {self.episodes_path}") from exc

    @property
    def training_camera_keys(self) -> tuple[str, ...]:
        """Globally allowed video views for training.

        This preserves the old remote allow-list proposal.  If no policy exists,
        every camera declared by ``episodes.jsonl`` remains trainable.
        """

        allowed = self._training_validity.allowed_video_camera_keys
        return self._declared_camera_keys if allowed is None else allowed

    def training_eligibility(
        self,
        episode_index: int,
        *,
        camera_keys: tuple[str, ...] | list[str] = (),
    ) -> TrainingEligibility:
        """Evaluate dataset, episode, and requested video-view validity.

        The optional policy is intentionally fail-closed once it exists and
        declares a relevant invalid value.  A missing policy is legacy-safe:
        all declared data remains eligible for training.
        """

        episode = self.episode(episode_index)
        requested = _unique_camera_keys(camera_keys, label="camera_keys")
        reasons: list[str] = []
        policy = self._training_validity
        if not policy.status.valid:
            reasons.append(_invalid_reason("dataset training data", policy.status))

        episode_policy = policy.episodes.get(episode.episode_index)
        if episode_policy is not None and not episode_policy.status.valid:
            reasons.append(_invalid_reason(f"episode {episode.episode_index} data", episode_policy.status))

        allowed = policy.allowed_video_camera_keys
        for camera_key in requested:
            if allowed is not None and camera_key not in allowed:
                reasons.append(
                    f"video view {camera_key!r} is not in training.allowed_video_camera_keys"
                )
            if episode_policy is not None:
                status = episode_policy.video_by_camera.get(camera_key)
                if status is not None and not status.valid:
                    reasons.append(_invalid_reason(f"episode {episode.episode_index} video {camera_key!r}", status))
        return TrainingEligibility(
            episode_index=episode.episode_index,
            camera_keys=requested,
            eligible=not reasons,
            reasons=tuple(reasons),
        )

    def require_training_eligibility(
        self,
        episode_index: int,
        *,
        camera_keys: tuple[str, ...] | list[str] = (),
    ) -> TrainingEligibility:
        eligibility = self.training_eligibility(episode_index, camera_keys=camera_keys)
        if not eligibility.eligible:
            raise InvalidTrainingDataError(eligibility)
        return eligibility

    def require_training_camera(self, camera_key: str) -> None:
        """Reject a video view excluded by the dataset-level training policy."""

        key = str(camera_key).strip()
        if key not in self.training_camera_keys:
            raise InvalidTrainingDataError(
                TrainingEligibility(
                    episode_index=-1,
                    camera_keys=(key,),
                    eligible=False,
                    reasons=(f"video view {key!r} is not allowed for training",),
                )
            )

    def video_path(self, episode_index: int, camera_key: str) -> Path:
        return self.episode(episode_index).video_path(camera_key)

    def has_video(self, episode_index: int, camera_key: str) -> bool:
        return self.episode(episode_index).has_video(camera_key)

    def representative_video_path(self, camera_key: str) -> Path | None:
        key = str(camera_key)
        for episode_index in self.episode_indices:
            episode = self.episode(episode_index)
            if episode.has_video(key):
                return episode.video_path(key)
        return None

    def training_video_path(self, episode_index: int, camera_key: str) -> Path:
        """Resolve a training video only after validity checks, without reading it."""

        self.require_training_eligibility(episode_index, camera_keys=(str(camera_key),))
        return self.video_path(episode_index, camera_key)

    def explicit_mask_path(self, episode_index: int, camera_key: str) -> Path | None:
        return self.episode(episode_index).explicit_mask_path(camera_key)

    def _parse_training_validity(self) -> _TrainingValidityPolicy:
        payload = self.modality_validity
        if not payload:
            return _TrainingValidityPolicy()
        schema_version = payload.get("schema_version")
        if schema_version not in (None, MODALITY_VALIDITY_SCHEMA_VERSION):
            raise ValueError(
                f"unsupported modality validity schema_version={schema_version!r} in {self.modality_validity_path}"
            )
        training = payload.get("training")
        if training is None:
            return _TrainingValidityPolicy()
        if not isinstance(training, Mapping):
            raise ValueError(f"{self.modality_validity_path}.training must be a JSON object.")

        status = _status_from_mapping(training, label=f"{self.modality_validity_path}.training")
        allowed_raw = training.get("allowed_video_camera_keys")
        allowed = (
            None
            if allowed_raw is None
            else _unique_camera_keys(
                allowed_raw,
                label=f"{self.modality_validity_path}.training.allowed_video_camera_keys",
            )
        )
        if allowed is not None:
            unknown = sorted(set(allowed) - set(self._declared_camera_keys))
            if unknown:
                raise ValueError(
                    f"{self.modality_validity_path} permits undeclared video camera(s): {unknown}"
                )

        raw_episodes = training.get("episodes", {})
        if not isinstance(raw_episodes, Mapping):
            raise ValueError(f"{self.modality_validity_path}.training.episodes must be a JSON object.")
        episodes: dict[int, _EpisodeTrainingValidity] = {}
        for raw_episode_index, raw_episode in raw_episodes.items():
            if isinstance(raw_episode_index, bool):
                raise ValueError(f"{self.modality_validity_path}.training.episodes has an invalid episode index.")
            try:
                episode_index = int(raw_episode_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.modality_validity_path}.training.episodes has an invalid episode index "
                    f"{raw_episode_index!r}."
                ) from exc
            if episode_index not in self._episodes:
                raise ValueError(
                    f"{self.modality_validity_path} declares unknown episode_index {episode_index}."
                )
            if not isinstance(raw_episode, Mapping):
                raise ValueError(
                    f"{self.modality_validity_path}.training.episodes[{episode_index}] must be a JSON object."
                )
            episode_status = _status_from_mapping(
                raw_episode,
                label=f"{self.modality_validity_path}.training.episodes[{episode_index}]",
            )
            modalities = raw_episode.get("modalities", {})
            if not isinstance(modalities, Mapping):
                raise ValueError(
                    f"{self.modality_validity_path}.training.episodes[{episode_index}].modalities must be a JSON object."
                )
            raw_video = modalities.get("video", {})
            if not isinstance(raw_video, Mapping):
                raise ValueError(
                    f"{self.modality_validity_path}.training.episodes[{episode_index}].modalities.video "
                    "must be a JSON object."
                )
            video_by_camera: dict[str, _ValidityStatus] = {}
            for raw_camera_key, raw_status in raw_video.items():
                camera_key = str(raw_camera_key).strip()
                if not camera_key:
                    raise ValueError(
                        f"{self.modality_validity_path}.training.episodes[{episode_index}].modalities.video "
                        "contains an empty camera key."
                    )
                if camera_key not in self._declared_camera_keys:
                    raise ValueError(
                        f"{self.modality_validity_path} declares undeclared video camera {camera_key!r}."
                    )
                video_by_camera[camera_key] = _status_from_mapping(
                    raw_status,
                    label=(
                        f"{self.modality_validity_path}.training.episodes[{episode_index}]"
                        f".modalities.video[{camera_key!r}]"
                    ),
                )
            episodes[episode_index] = _EpisodeTrainingValidity(
                status=episode_status,
                video_by_camera=video_by_camera,
            )
        return _TrainingValidityPolicy(
            status=status,
            allowed_video_camera_keys=allowed,
            episodes=episodes,
        )


def _invalid_reason(label: str, status: _ValidityStatus) -> str:
    suffix = "" if not status.reason else f": {status.reason}"
    return f"{label} is marked invalid{suffix}"
