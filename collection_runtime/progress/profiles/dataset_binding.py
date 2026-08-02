"""Dataset-wide immutable binding for collection progress profiles.

The bound artifact is the existing :class:`TaskProgressProfileV1` itself.
There is deliberately no second binding schema or copied identity vocabulary:
``meta/progress/profile.cjson`` is the one canonical dataset-level profile.
Episode-local copies remain self-contained and must match it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_facts import PROGRESS_PROFILE_FILE_DOMAIN_V1
from dataset_contracts.progress_profile import TaskProgressProfileV1
from dataset_contracts.progress_stages import (
    TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
    TaskStageProfileV1,
    validate_task_stage_profile,
)
from dataset_interface.capture_modalities import dataset_is_blank

from ...immutable_canonical_store import publish_immutable_canonical


DATASET_PROGRESS_PROFILE_RELATIVE_PATH = Path("meta/progress/profile.cjson")
DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH = Path(
    "meta/progress/stages.cjson"
)
DATASET_PROGRESS_PROFILE_LOCK_RELATIVE_PATH = Path(
    "meta/progress/.profile.lock"
)
PROGRESS_EPISODES_RELATIVE_ROOT = Path("meta/progress/episodes")


class DatasetProgressProfileBindingError(ValueError):
    """Raised when an append could mix progress profiles in one dataset."""


@dataclass(frozen=True)
class DatasetProgressProfileAdmission:
    """Read-only result used by collection applications before an Episode."""

    profile: TaskProgressProfileV1 | None
    locked: bool
    source: str

    @property
    def profile_hash(self) -> str | None:
        return (
            None
            if self.profile is None
            else self.profile.semantic_content_hash
        )


@dataclass(frozen=True)
class DatasetTaskStageProfileAdmission:
    """Read-only dataset admission for optional fine-stage semantics."""

    profile: TaskStageProfileV1 | None
    locked: bool
    source: str

    @property
    def profile_hash(self) -> str | None:
        return (
            None
            if self.profile is None
            else self.profile.semantic_content_hash
        )


def _load_profile(path: Path) -> TaskProgressProfileV1:
    try:
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("canonical payload is not an object")
        return TaskProgressProfileV1.from_payload(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise DatasetProgressProfileBindingError(
            f"Invalid dataset progress profile artifact {path}: {exc}"
        ) from exc


def read_dataset_progress_profile(
    dataset_root: str | Path,
) -> TaskProgressProfileV1 | None:
    """Read the explicit dataset-level profile, if one has been bound."""

    root = Path(dataset_root).expanduser()
    path = root / DATASET_PROGRESS_PROFILE_RELATIVE_PATH
    return None if not path.exists() else _load_profile(path)


def _load_stage_profile(path: Path) -> TaskStageProfileV1:
    try:
        payload = canonical_loads(
            path.read_bytes(),
            expected_domain=TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("canonical payload is not an object")
        return TaskStageProfileV1.from_payload(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise DatasetProgressProfileBindingError(
            f"Invalid dataset task stage profile artifact {path}: {exc}"
        ) from exc


def read_dataset_task_stage_profile(
    dataset_root: str | Path,
) -> TaskStageProfileV1 | None:
    """Read the dataset-wide fine-stage profile, if explicitly bound."""

    root = Path(dataset_root).expanduser()
    path = root / DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH
    return None if not path.exists() else _load_stage_profile(path)


def _legacy_episode_stage_profile(
    dataset_root: Path,
) -> tuple[TaskStageProfileV1 | None, bool]:
    """Return one exact legacy Episode stage profile and whether Episodes exist."""

    episode_root = dataset_root / PROGRESS_EPISODES_RELATIVE_ROOT
    if not episode_root.exists():
        return None, False
    episode_dirs = tuple(
        sorted(
            (path for path in episode_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    )
    if not episode_dirs:
        return None, False
    profiles: list[TaskStageProfileV1] = []
    missing = 0
    for episode_dir in episode_dirs:
        stage_path = episode_dir / "stages.cjson"
        if stage_path.is_file():
            profiles.append(_load_stage_profile(stage_path))
        else:
            missing += 1
    if profiles and missing:
        raise DatasetProgressProfileBindingError(
            "Existing progress Episodes mix stage-profiled and unprofiled "
            "semantics."
        )
    if not profiles:
        return None, True
    expected = profiles[0]
    if any(
        item.semantic_content_hash != expected.semantic_content_hash
        for item in profiles[1:]
    ):
        raise DatasetProgressProfileBindingError(
            "Existing progress Episodes contain multiple task stage profiles."
        )
    return expected, True


def _legacy_episode_profile(
    dataset_root: Path,
) -> TaskProgressProfileV1 | None:
    """Return one exact profile shared by every legacy progress Episode."""

    episode_root = dataset_root / PROGRESS_EPISODES_RELATIVE_ROOT
    if not episode_root.exists():
        return None
    profiles: list[TaskProgressProfileV1] = []
    for episode_dir in sorted(
        (path for path in episode_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        profile_path = episode_dir / "profile.cjson"
        if not profile_path.is_file():
            raise DatasetProgressProfileBindingError(
                "Existing progress Episode is missing profile.cjson: "
                f"{episode_dir}."
            )
        profiles.append(_load_profile(profile_path))
    if not profiles:
        return None
    expected = profiles[0]
    mismatched = tuple(
        profile.semantic_content_hash
        for profile in profiles[1:]
        if profile.semantic_content_hash != expected.semantic_content_hash
    )
    if mismatched:
        raise DatasetProgressProfileBindingError(
            "Existing dataset already contains multiple collection progress "
            "profiles; refuse to infer or append another profile."
        )
    return expected


def inspect_dataset_progress_profile(
    dataset_root: str | Path,
    *,
    requested_profile: TaskProgressProfileV1 | None = None,
) -> DatasetProgressProfileAdmission:
    """Validate one selection without mutating the dataset.

    A populated profile-less dataset is treated as permanently profile-less.
    This prevents enabling progress halfway through a dataset and producing a
    mixture of profiled and unprofiled Episodes.
    """

    if requested_profile is not None and not isinstance(
        requested_profile,
        TaskProgressProfileV1,
    ):
        raise TypeError("requested_profile must be TaskProgressProfileV1 or None.")
    root = Path(dataset_root).expanduser()
    explicit = read_dataset_progress_profile(root)
    legacy = _legacy_episode_profile(root)
    if (
        explicit is not None
        and legacy is not None
        and explicit.semantic_content_hash != legacy.semantic_content_hash
    ):
        raise DatasetProgressProfileBindingError(
            "Dataset-level profile differs from an existing progress Episode."
        )
    existing = explicit or legacy
    if existing is not None:
        if (
            requested_profile is not None
            and requested_profile.semantic_content_hash
            != existing.semantic_content_hash
        ):
            raise DatasetProgressProfileBindingError(
                "Requested collection progress profile differs from the "
                "profile already used by this dataset: "
                f"existing={existing.profile_id}@{existing.profile_version}"
                f"/{existing.semantic_content_hash}, "
                f"requested={requested_profile.profile_id}@"
                f"{requested_profile.profile_version}/"
                f"{requested_profile.semantic_content_hash}."
            )
        return DatasetProgressProfileAdmission(
            profile=existing,
            locked=True,
            source=(
                "dataset_binding"
                if explicit is not None
                else "legacy_episode_profiles"
            ),
        )
    if not dataset_is_blank(root):
        if requested_profile is not None:
            raise DatasetProgressProfileBindingError(
                "Cannot add a collection progress profile to a populated "
                "profile-less dataset; create/select a new dataset root."
            )
        return DatasetProgressProfileAdmission(
            profile=None,
            locked=True,
            source="populated_profileless_dataset",
        )
    return DatasetProgressProfileAdmission(
        profile=requested_profile,
        locked=False,
        source="blank_dataset",
    )


def inspect_dataset_task_stage_profile(
    dataset_root: str | Path,
    *,
    progress_profile: TaskProgressProfileV1,
    requested_profile: TaskStageProfileV1 | None = None,
) -> DatasetTaskStageProfileAdmission:
    """Prevent fine-stage profile mixing within one progress dataset."""

    if not isinstance(progress_profile, TaskProgressProfileV1):
        raise TypeError("progress_profile must be TaskProgressProfileV1.")
    if requested_profile is not None:
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=requested_profile,
        )
    root = Path(dataset_root).expanduser()
    dataset_progress = read_dataset_progress_profile(root)
    if (
        dataset_progress is not None
        and dataset_progress.semantic_content_hash
        != progress_profile.semantic_content_hash
    ):
        raise DatasetProgressProfileBindingError(
            "Task stage admission used a different dataset progress profile."
        )
    explicit = read_dataset_task_stage_profile(root)
    legacy, has_progress_episodes = _legacy_episode_stage_profile(root)
    if explicit is not None:
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=explicit,
        )
    if legacy is not None:
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=legacy,
        )
    if (
        explicit is not None
        and legacy is not None
        and explicit.semantic_content_hash != legacy.semantic_content_hash
    ):
        raise DatasetProgressProfileBindingError(
            "Dataset-level task stage profile differs from an existing "
            "progress Episode."
        )
    existing = explicit or legacy
    if existing is not None:
        if (
            requested_profile is not None
            and requested_profile.semantic_content_hash
            != existing.semantic_content_hash
        ):
            raise DatasetProgressProfileBindingError(
                "Requested task stage profile differs from the profile "
                "already used by this dataset."
            )
        return DatasetTaskStageProfileAdmission(
            profile=existing,
            locked=True,
            source=(
                "dataset_binding"
                if explicit is not None
                else "legacy_episode_profiles"
            ),
        )
    if has_progress_episodes:
        if requested_profile is not None:
            raise DatasetProgressProfileBindingError(
                "Cannot enable fine-stage semantics after unprofiled progress "
                "Episodes have already been sealed in this dataset."
            )
        return DatasetTaskStageProfileAdmission(
            profile=None,
            locked=True,
            source="existing_stage_profileless_progress_episodes",
        )
    return DatasetTaskStageProfileAdmission(
        profile=requested_profile,
        locked=False,
        source="no_progress_episodes",
    )


def bind_dataset_progress_profile(
    dataset_root: str | Path,
    profile: TaskProgressProfileV1,
) -> Path:
    """Atomically bind one exact profile before the first profiled Episode."""

    if not isinstance(profile, TaskProgressProfileV1):
        raise TypeError("profile must be TaskProgressProfileV1.")
    root = Path(dataset_root).expanduser()
    lock_path = root / DATASET_PROGRESS_PROFILE_LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            admission = inspect_dataset_progress_profile(
                root,
                requested_profile=profile,
            )
            effective = admission.profile
            if (
                effective is None
                or effective.semantic_content_hash
                != profile.semantic_content_hash
            ):
                raise DatasetProgressProfileBindingError(
                    "Dataset progress profile admission did not preserve the "
                    "requested profile."
                )
            path = root / DATASET_PROGRESS_PROFILE_RELATIVE_PATH
            publish_immutable_canonical(
                path,
                profile.to_payload(),
                domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
            )
            return path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def bind_dataset_task_stage_profile(
    dataset_root: str | Path,
    *,
    progress_profile: TaskProgressProfileV1,
    stage_profile: TaskStageProfileV1,
) -> Path:
    """Atomically bind one exact fine-stage profile before its first Episode."""

    validate_task_stage_profile(
        progress_profile=progress_profile,
        stage_profile=stage_profile,
    )
    root = Path(dataset_root).expanduser()
    lock_path = root / DATASET_PROGRESS_PROFILE_LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            progress_admission = inspect_dataset_progress_profile(
                root,
                requested_profile=progress_profile,
            )
            if progress_admission.profile_hash != (
                progress_profile.semantic_content_hash
            ):
                raise DatasetProgressProfileBindingError(
                    "Dataset progress binding was not established before its "
                    "task stage profile."
                )
            admission = inspect_dataset_task_stage_profile(
                root,
                progress_profile=progress_profile,
                requested_profile=stage_profile,
            )
            if admission.profile_hash != stage_profile.semantic_content_hash:
                raise DatasetProgressProfileBindingError(
                    "Dataset task stage admission did not preserve the "
                    "requested profile."
                )
            path = root / DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH
            publish_immutable_canonical(
                path,
                stage_profile.to_payload(),
                domain=TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
            )
            return path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = [
    "DATASET_PROGRESS_PROFILE_RELATIVE_PATH",
    "DATASET_TASK_STAGE_PROFILE_RELATIVE_PATH",
    "DatasetProgressProfileAdmission",
    "DatasetProgressProfileBindingError",
    "DatasetTaskStageProfileAdmission",
    "bind_dataset_progress_profile",
    "bind_dataset_task_stage_profile",
    "inspect_dataset_progress_profile",
    "inspect_dataset_task_stage_profile",
    "read_dataset_progress_profile",
    "read_dataset_task_stage_profile",
]
