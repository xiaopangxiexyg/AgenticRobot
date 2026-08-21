"""Versioned site profiles for the real OpenPI deployment application.

The profile owns reviewed lab facts only.  It cannot grant motion authority or
inject arbitrary runtime flags; callers still opt in to ``--send-actions`` at
the canonical real deployment entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


OPENPI_REAL_DEPLOYMENT_PROFILE_SCHEMA_V1 = (
    "dexteleop_openpi_real_deployment_profile_v1"
)
RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID = "rm65_bimanual_dmtac"
_HASH_DOMAIN = "dexteleop.openpi.real_deployment_profile.v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_PATH_BY_ID = {
    RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID: (
        _PROJECT_ROOT
        / "teleop_runtime/configs/deployments/openpi/"
        "rm65_bimanual_dmtac_v1.json"
    ),
}


class OpenPIRealDeploymentProfileError(ValueError):
    """Raised when a reviewed real OpenPI profile is malformed."""


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OpenPIRealDeploymentProfileError(
                f"OpenPI deployment profile contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def _exact_object(
    value: Any,
    *,
    field_name: str,
    expected_fields: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a JSON object."
        )
    actual_fields = set(value)
    unknown = sorted(actual_fields - expected_fields)
    missing = sorted(expected_fields - actual_fields)
    if unknown:
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} contains unknown fields: {unknown}."
        )
    if missing:
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} is missing required fields: {missing}."
        )
    return dict(value)


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a non-empty string."
        )
    return value.strip()


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a boolean."
        )
    return value


def _nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a non-negative integer."
        )
    return value


def _nonnegative_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a non-negative number."
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a non-negative number."
        )
    return numeric


def _repository_relative_path(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise OpenPIRealDeploymentProfileError(
            f"{field_name} must be a repository-relative path without '..'."
        )
    return path.as_posix()


@dataclass(frozen=True)
class OpenPIRealDeploymentProfile:
    """Validated immutable facts for one real OpenPI application profile."""

    profile_id: str
    runtime: Mapping[str, object]
    paths: Mapping[str, str]
    interactive_startup: Mapping[str, object]
    rtc_real_shadow_startup: Mapping[str, object]
    source_path: Path
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))
        object.__setattr__(self, "paths", MappingProxyType(dict(self.paths)))
        object.__setattr__(
            self,
            "interactive_startup",
            MappingProxyType(dict(self.interactive_startup)),
        )
        object.__setattr__(
            self,
            "rtc_real_shadow_startup",
            MappingProxyType(dict(self.rtc_real_shadow_startup)),
        )
        object.__setattr__(self, "source_path", Path(self.source_path))

    def runtime_args(
        self,
        *,
        project_root: Path = _PROJECT_ROOT,
        rtc_real_shadow: bool = False,
    ) -> list[str]:
        """Project the reviewed fields to the legacy CLI in the same order."""

        root = Path(project_root).resolve()
        runtime = self.runtime
        paths = self.paths
        args = [
            "--active-sides",
            str(runtime["active_sides"]),
            "--end-effector-sides",
            str(runtime["end_effector_sides"]),
            "--arm-execution-profile",
            str(runtime["arm_execution_profile"]),
            "--left-arm-ip",
            str(runtime["left_arm_ip"]),
            "--right-arm-ip",
            str(runtime["right_arm_ip"]),
            "--left-tool-profile",
            str(root / paths["left_tool_profile"]),
            "--right-tool-profile",
            str(root / paths["right_tool_profile"]),
        ]
        if bool(runtime["enable_hands"]):
            args.append("--enable-hands")
        args.extend(
            [
                "--hand-backend",
                str(runtime["hand_backend"]),
                "--dmtac-config",
                str(runtime["dmtac_config"]),
            ]
        )
        if bool(runtime["require_dmtac_config"]):
            args.append("--require-dmtac-config")
        # The prepared policy consumes only each gripper's scalar state.  The
        # reviewed profile keeps large tactile payloads out of the parent so
        # they cannot starve the 60/120 Hz arm workers; a later explicit CLI
        # --dmtac-with-tactile still preserves the established rollback.
        args.append(
            "--dmtac-with-tactile"
            if bool(runtime["dmtac_tactile"])
            else "--dmtac-no-tactile"
        )
        args.extend(
            [
                "--camera-mode",
                str(runtime["camera_mode"]),
                "--camera-config",
                str(root / paths["camera_config"]),
            ]
        )
        if bool(runtime["require_camera_frame"]):
            args.append("--require-camera-frame")
        if bool(runtime["require_ready_observation"]):
            args.append("--require-ready-observation")
        args.extend(
            [
                "--action-queue-policy",
                str(runtime["action_queue_policy"]),
                "--min-action-queue",
                str(runtime["min_action_queue"]),
                "--init-source",
                str(runtime["init_source"]),
            ]
        )

        if rtc_real_shadow:
            startup = self.rtc_real_shadow_startup
            if bool(startup["auto_start"]):
                args.append("--auto-start")
            args.append(
                "--wait-for-ready-confirm"
                if bool(startup["wait_for_ready_confirm"])
                else "--no-wait-for-ready-confirm"
            )
            args.extend(
                [
                    "--camera-start-stagger-sec",
                    str(startup["camera_start_stagger_sec"]),
                ]
            )
            return args

        startup = self.interactive_startup
        args.append(
            "--wait-for-ready-confirm"
            if bool(startup["wait_for_ready_confirm"])
            else "--no-wait-for-ready-confirm"
        )
        args.extend(
            [
                "--policy-qpos-init-dataset",
                str(root / str(startup["policy_qpos_init_dataset"])),
                "--policy-qpos-init-episode",
                str(startup["policy_qpos_init_episode"]),
                "--policy-qpos-init-frame",
                str(startup["policy_qpos_init_frame"]),
                "--policy-qpos-init-source",
                str(startup["policy_qpos_init_source"]),
            ]
        )
        if bool(startup["policy_qpos_init_collision_check"]):
            args.append("--policy-qpos-init-collision-check")
        return args


def load_openpi_real_deployment_profile_file(
    path: str | Path,
) -> OpenPIRealDeploymentProfile:
    """Load one exact profile without opening devices or validating assets."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"OpenPI deployment profile is not a file: {source_path}")
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise OpenPIRealDeploymentProfileError(
            f"OpenPI deployment profile is invalid JSON: {source_path}."
        ) from exc

    top = _exact_object(
        payload,
        field_name="profile",
        expected_fields={
            "schema",
            "profile_id",
            "runtime",
            "paths",
            "interactive_startup",
            "rtc_real_shadow_startup",
        },
    )
    schema = _text(top["schema"], field_name="schema")
    if schema != OPENPI_REAL_DEPLOYMENT_PROFILE_SCHEMA_V1:
        raise OpenPIRealDeploymentProfileError(
            f"Unsupported OpenPI deployment profile schema: {schema!r}."
        )
    profile_id = _text(top["profile_id"], field_name="profile_id")
    runtime = _exact_object(
        top["runtime"],
        field_name="runtime",
        expected_fields={
            "active_sides",
            "end_effector_sides",
            "arm_execution_profile",
            "left_arm_ip",
            "right_arm_ip",
            "enable_hands",
            "hand_backend",
            "dmtac_config",
            "require_dmtac_config",
            "dmtac_tactile",
            "camera_mode",
            "require_camera_frame",
            "require_ready_observation",
            "action_queue_policy",
            "min_action_queue",
            "init_source",
        },
    )
    for field_name in (
        "active_sides",
        "end_effector_sides",
        "arm_execution_profile",
        "left_arm_ip",
        "right_arm_ip",
        "hand_backend",
        "dmtac_config",
        "camera_mode",
        "action_queue_policy",
        "init_source",
    ):
        runtime[field_name] = _text(
            runtime[field_name], field_name=f"runtime.{field_name}"
        )
    for field_name in (
        "enable_hands",
        "require_dmtac_config",
        "dmtac_tactile",
        "require_camera_frame",
        "require_ready_observation",
    ):
        runtime[field_name] = _boolean(
            runtime[field_name], field_name=f"runtime.{field_name}"
        )
    runtime["min_action_queue"] = _nonnegative_integer(
        runtime["min_action_queue"], field_name="runtime.min_action_queue"
    )

    paths = _exact_object(
        top["paths"],
        field_name="paths",
        expected_fields={"left_tool_profile", "right_tool_profile", "camera_config"},
    )
    for field_name in tuple(paths):
        paths[field_name] = _repository_relative_path(
            paths[field_name], field_name=f"paths.{field_name}"
        )

    interactive = _exact_object(
        top["interactive_startup"],
        field_name="interactive_startup",
        expected_fields={
            "wait_for_ready_confirm",
            "policy_qpos_init_dataset",
            "policy_qpos_init_episode",
            "policy_qpos_init_frame",
            "policy_qpos_init_source",
            "policy_qpos_init_collision_check",
        },
    )
    interactive["wait_for_ready_confirm"] = _boolean(
        interactive["wait_for_ready_confirm"],
        field_name="interactive_startup.wait_for_ready_confirm",
    )
    interactive["policy_qpos_init_dataset"] = _repository_relative_path(
        interactive["policy_qpos_init_dataset"],
        field_name="interactive_startup.policy_qpos_init_dataset",
    )
    for field_name in ("policy_qpos_init_episode", "policy_qpos_init_frame"):
        interactive[field_name] = _nonnegative_integer(
            interactive[field_name],
            field_name=f"interactive_startup.{field_name}",
        )
    interactive["policy_qpos_init_source"] = _text(
        interactive["policy_qpos_init_source"],
        field_name="interactive_startup.policy_qpos_init_source",
    )
    interactive["policy_qpos_init_collision_check"] = _boolean(
        interactive["policy_qpos_init_collision_check"],
        field_name="interactive_startup.policy_qpos_init_collision_check",
    )

    shadow = _exact_object(
        top["rtc_real_shadow_startup"],
        field_name="rtc_real_shadow_startup",
        expected_fields={
            "auto_start",
            "wait_for_ready_confirm",
            "camera_start_stagger_sec",
        },
    )
    for field_name in ("auto_start", "wait_for_ready_confirm"):
        shadow[field_name] = _boolean(
            shadow[field_name],
            field_name=f"rtc_real_shadow_startup.{field_name}",
        )
    shadow["camera_start_stagger_sec"] = _nonnegative_number(
        shadow["camera_start_stagger_sec"],
        field_name="rtc_real_shadow_startup.camera_start_stagger_sec",
    )

    semantic_payload = {
        "schema": schema,
        "profile_id": profile_id,
        "runtime": runtime,
        "paths": paths,
        "interactive_startup": interactive,
        "rtc_real_shadow_startup": shadow,
    }
    return OpenPIRealDeploymentProfile(
        profile_id=profile_id,
        runtime=runtime,
        paths=paths,
        interactive_startup=interactive,
        rtc_real_shadow_startup=shadow,
        source_path=source_path,
        semantic_content_hash=canonical_content_hash(
            semantic_payload,
            domain=_HASH_DOMAIN,
        ),
    )


def available_openpi_real_deployment_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILE_PATH_BY_ID))


def load_openpi_real_deployment_profile(
    profile_id: str,
) -> OpenPIRealDeploymentProfile:
    normalized = str(profile_id).strip()
    try:
        path = _PROFILE_PATH_BY_ID[normalized]
    except KeyError as exc:
        raise OpenPIRealDeploymentProfileError(
            f"Unknown real OpenPI deployment profile: {normalized!r}."
        ) from exc
    profile = load_openpi_real_deployment_profile_file(path)
    if profile.profile_id != normalized:
        raise OpenPIRealDeploymentProfileError(
            "OpenPI deployment profile id does not match its registered name: "
            f"expected {normalized!r}, got {profile.profile_id!r}."
        )
    return profile


def openpi_real_deployment_profile_args(
    profile_id: str,
    *,
    rtc_real_shadow: bool = False,
    project_root: Path = _PROJECT_ROOT,
) -> list[str]:
    return load_openpi_real_deployment_profile(profile_id).runtime_args(
        project_root=project_root,
        rtc_real_shadow=rtc_real_shadow,
    )


__all__ = [
    "OPENPI_REAL_DEPLOYMENT_PROFILE_SCHEMA_V1",
    "OpenPIRealDeploymentProfile",
    "OpenPIRealDeploymentProfileError",
    "RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID",
    "available_openpi_real_deployment_profiles",
    "load_openpi_real_deployment_profile",
    "load_openpi_real_deployment_profile_file",
    "openpi_real_deployment_profile_args",
]
