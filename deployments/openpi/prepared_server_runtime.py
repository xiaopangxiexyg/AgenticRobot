"""Serve one prepared OpenPI checkpoint with its exact derived config.

OpenPI keeps training configs in an in-process registry.  This runtime rebuilds
the immutable inference-relevant config from the checked-in training profile,
validates it against the checkpoint-bound deployment artifact, registers it
only for this process, and then delegates to OpenPI's existing server script.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import dataclasses
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from dataset_contracts.state_input import (
    STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    deployment_state_input,
)

from deployments.openpi.image_wire import (
    OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1,
    OpenPIImageWireServerConfig,
    build_openpi_image_wire_capability,
    install_openpi_image_wire_server_adapter,
)
from deployments.openpi.prepared_deployment import (
    PreparedDeploymentResolution,
    validate_prepared_deployment_artifact,
)
from deployments.openpi.state_only import (
    OpenPIStateOnlyInputBridge,
    install_openpi_state_only_server_adapter,
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OpenPI server script: {path}")
    module = importlib.util.module_from_spec(spec)
    had_previous = name in sys.modules
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if had_previous:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
        raise
    return module


def _module_origin(module: Any, module_name: str) -> Path:
    raw_origin = getattr(module, "__file__", None)
    if not raw_origin:
        raise ImportError(
            f"Prepared OpenPI runtime cannot verify {module_name!r}: "
            "the imported module has no filesystem origin."
        )
    return Path(str(raw_origin)).expanduser().resolve()


def _require_module_under(
    module: Any,
    *,
    module_name: str,
    source_root: Path,
) -> Path:
    origin = _module_origin(module, module_name)
    if not origin.is_relative_to(source_root):
        raise ImportError(
            "Prepared OpenPI runtime imported a module from the wrong "
            f"checkout: module={module_name!r} origin={origin} "
            f"expected_source_root={source_root}."
        )
    return origin


def _activate_runtime_openpi_sources(runtime_root: Path) -> dict[str, str]:
    """Bind OpenPI imports to ``runtime_root`` before importing its packages.

    ``PYTHONPATH`` is still set by the parent launcher, but an editable install
    may also install an import hook in the selected Python environment.  Make
    the intended source roots authoritative in this process and verify the
    modules that Python actually selected.  A mismatched package that was
    imported before this boundary is unsafe to unload piecemeal, so fail
    clearly instead of attempting a silent module replacement.
    """

    resolved_root = runtime_root.expanduser().resolve()
    source_roots = {
        "openpi": (resolved_root / "src").resolve(),
        "openpi_client": (
            resolved_root / "packages" / "openpi-client" / "src"
        ).resolve(),
    }
    missing = [
        str(source_root / package_name)
        for package_name, source_root in source_roots.items()
        if not (source_root / package_name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Prepared OpenPI runtime source packages are missing: "
            + ", ".join(missing)
        )

    for package_name, source_root in source_roots.items():
        loaded = tuple(
            (name, module)
            for name, module in sys.modules.items()
            if module is not None
            and (name == package_name or name.startswith(package_name + "."))
        )
        for name, module in loaded:
            _require_module_under(
                module,
                module_name=name,
                source_root=source_root,
            )

    selected_roots = [str(source_root) for source_root in source_roots.values()]
    sys.path[:] = selected_roots + [
        entry for entry in sys.path if entry not in selected_roots
    ]
    importlib.invalidate_caches()

    origins: dict[str, str] = {}
    for package_name, source_root in source_roots.items():
        module = importlib.import_module(package_name)
        origins[package_name] = str(
            _require_module_under(
                module,
                module_name=package_name,
                source_root=source_root,
            )
        )
    return origins


def _import_runtime_openpi_modules(
    runtime_root: Path,
) -> tuple[Any, Any, dict[str, str]]:
    origins = _activate_runtime_openpi_sources(runtime_root)
    training_config = importlib.import_module("openpi.training.config")
    dataset_policy = importlib.import_module("openpi.policies.dexteleop_v23_policy")
    openpi_source_root = (runtime_root.expanduser().resolve() / "src").resolve()
    for module_name, module in (
        ("openpi.training.config", training_config),
        ("openpi.policies.dexteleop_v23_policy", dataset_policy),
    ):
        origins[module_name] = str(
            _require_module_under(
                module,
                module_name=module_name,
                source_root=openpi_source_root,
            )
        )
    return training_config, dataset_policy, origins


def _validate_runtime_openpi_binding(
    *,
    runtime_root: Path,
    training_root: Path,
    model_rtc_profile: Path | None,
    rtc_inference_steps: int,
    bundled_root: Path | None = None,
) -> None:
    """Allow source separation only for the pinned 8-step RTC runtime."""

    if runtime_root == training_root:
        return
    expected_runtime_root = (
        Path(__file__).resolve().parents[2] / "model" / "openpi"
        if bundled_root is None
        else bundled_root
    ).resolve()
    if (
        model_rtc_profile is None
        or rtc_inference_steps != 8
        or runtime_root != expected_runtime_root
    ):
        raise ValueError(
            "a distinct --openpi-root is only allowed for the 8-step "
            "trained model-RTC deployment and must be this AgenticRobot "
            "snapshot's pinned model/openpi submodule."
        )


def _mixing_sources(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    mixing = payload.get("dataset_mixing")
    if not isinstance(mixing, Mapping):
        raise ValueError("Prepared deployment artifact lacks dataset_mixing.")
    values = mixing.get("sources")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Prepared deployment dataset_mixing.sources is invalid.")
    if not values or any(not isinstance(value, Mapping) for value in values):
        raise ValueError("Prepared deployment requires dataset source mappings.")
    return tuple(values)


def _occlusion_schedule(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("front_occlusion_schedule must be a sequence.")
    schedule: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("front_occlusion_schedule entries must be mappings.")
        schedule.append(
            (float(item["ratio"]), float(item["probability"]))
        )
    if not schedule:
        raise ValueError("front_occlusion_schedule must not be empty.")
    return tuple(schedule)


def _prepared_weight_loader(
    profile: Mapping[str, Any],
    weight_loader: Any,
    checkout: Path,
) -> Any:
    """Restore the profile-pinned base loader used to train the checkpoint."""

    configured = str(
        profile.get("openpi_weight_loader_params_path") or ""
    ).strip()
    if not configured:
        return weight_loader
    params_path = Path(configured).expanduser()
    if not params_path.is_absolute():
        params_path = checkout / params_path
    params_path = params_path.resolve()
    if not params_path.is_dir():
        raise FileNotFoundError(
            f"Pinned OpenPI base parameter directory is missing: {params_path}."
        )
    required = (
        "_CHECKPOINT_METADATA",
        "_METADATA",
        "_sharding",
        "commit_success.txt",
        "manifest.ocdbt",
    )
    missing = [name for name in required if not (params_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Pinned OpenPI base parameter directory is incomplete: "
            f"path={params_path} missing={missing}."
        )
    if not dataclasses.is_dataclass(weight_loader) or not hasattr(
        weight_loader, "params_path"
    ):
        raise TypeError(
            "openpi_weight_loader_params_path requires a dataclass weight "
            "loader with params_path."
        )
    return dataclasses.replace(weight_loader, params_path=str(params_path))


@contextmanager
def _use_checkpoint_policy_metadata(
    training_config: Any,
    config: Any,
    resolution: PreparedDeploymentResolution,
) -> Iterator[None]:
    """Keep prepared inference on the already validated checkpoint artifact.

    OpenPI normally derives policy metadata again when it creates a policy.
    That is appropriate for training configs, but it makes inference reopen
    every historical dataset source even though the prepared deployment
    artifact has already been recomputed and validated against the checkpoint.
    Intercept only the exact process-local config registered below; every other
    config continues through OpenPI's original resolver.
    """

    expected_artifact = resolution.deployment_artifact.to_dict()
    metadata = getattr(config, "policy_metadata", None)
    if not isinstance(metadata, Mapping) or (
        metadata.get("policy_artifact") != expected_artifact
    ):
        raise ValueError(
            "Prepared OpenPI config lacks its validated checkpoint policy artifact."
        )
    original = training_config.resolve_policy_metadata

    def _resolve(candidate: Any) -> Any:
        if candidate is not config:
            return original(candidate)
        candidate_metadata = getattr(candidate, "policy_metadata", None)
        if not isinstance(candidate_metadata, Mapping) or (
            candidate_metadata.get("policy_artifact") != expected_artifact
        ):
            raise ValueError(
                "Prepared OpenPI config policy artifact changed after registration."
            )
        return candidate

    training_config.resolve_policy_metadata = _resolve
    try:
        yield
    finally:
        current = training_config.resolve_policy_metadata
        training_config.resolve_policy_metadata = original
        if current is not _resolve:
            raise RuntimeError(
                "OpenPI policy metadata resolver changed during prepared serving."
            )


def build_prepared_openpi_config(
    resolution: PreparedDeploymentResolution,
    *,
    training_config: Any,
    dataset_policy: Any,
) -> Any:
    """Rebuild the exact training-derived config needed for policy creation."""

    profile = resolution.profile
    artifact = resolution.deployment_artifact
    base = training_config.get_config(str(profile["openpi_base_config"]))
    augmentation_source_name = str(
        profile.get("openpi_state_augmentation_from_config") or ""
    )
    state_augmentation = base.data.state_augmentation
    if augmentation_source_name:
        augmentation_source = training_config.get_config(
            augmentation_source_name
        )
        state_augmentation = getattr(
            augmentation_source.data,
            "state_augmentation",
            None,
        )
        if state_augmentation is None:
            raise ValueError(
                "Prepared augmentation source has no state_augmentation."
            )
    state_augmentation_mode = str(
        profile.get("openpi_state_augmentation_mode", "configured_v1")
    ).strip()
    if state_augmentation_mode == "disabled_v1":
        state_augmentation = dataclasses.replace(
            state_augmentation,
            enabled=False,
            noise_std=0.0,
            dropout_probability=0.0,
            structured_enabled=False,
            arm_translation_std_m=0.0,
            arm_rotation_std_deg=0.0,
            gripper_noise_std=0.0,
            side_dropout_probability=0.0,
            arm_dropout_probability=0.0,
            end_effector_dropout_probability=0.0,
        )
    elif state_augmentation_mode != "configured_v1":
        raise ValueError(
            "openpi_state_augmentation_mode must be configured_v1 or disabled_v1."
        )
    raw_state_augmentation = resolution.deployment_payload.get(
        "state_augmentation"
    )
    if not isinstance(raw_state_augmentation, Mapping):
        raise ValueError(
            "Prepared deployment artifact lacks state_augmentation metadata."
        )
    to_dict = getattr(state_augmentation, "to_dict", None)
    if not callable(to_dict) or to_dict() != dict(raw_state_augmentation):
        raise ValueError(
            "OpenPI state-augmentation config differs from the checkpoint "
            "artifact."
        )
    if str(getattr(state_augmentation, "state_input_mode", "")) != str(
        artifact.state_input_mode
    ):
        raise ValueError(
            "OpenPI state input mode differs from the checkpoint artifact."
        )

    expected_slots = tuple(
        str(value) for value in profile["camera_slot_to_dataset_key"].keys()
    )
    if (
        int(base.model.action_horizon) != int(profile["action_horizon"])
        or tuple(base.model.image_keys) != expected_slots
        or str(base.data.action_pose_mode) != str(profile["action_pose_mode"])
        or str(base.data.last_action_pose_mode)
        != str(profile["action_pose_mode"])
        or str(base.data.tail_policy) != str(profile["tail_policy"])
    ):
        raise ValueError(
            "OpenPI base config differs from the prepared model contract."
        )

    if resolution.model_rtc_profile is None:
        source_configs = tuple(
            dataset_policy.DexTeleopV23DatasetSourceConfig(
                name=str(source["name"]),
                dataset_root=str(source["dataset_root"]),
                sample_weight=int(source.get("sample_weight", 1)),
                front_occlusion_schedule=_occlusion_schedule(
                    source["front_occlusion_schedule"]
                ),
                explicit_mask_keys=tuple(
                    str(value)
                    for value in source.get("explicit_mask_keys", ())
                ),
                front_force_black=bool(
                    source.get("front_force_black", False)
                ),
                episode_manifest=str(source["episode_manifest"]),
            )
            for source in _mixing_sources(resolution.deployment_payload)
        )
    else:
        explicit_mask_keys = tuple(
            str(value)
            for value in getattr(base.data, "explicit_mask_keys", ())
        )
        source_configs = tuple(
            dataset_policy.DexTeleopV23DatasetSourceConfig(
                name=str(source["source_dataset_id"]),
                dataset_root=str(source["source_dataset_root"]),
                sample_weight=1,
                front_occlusion_schedule=((0.0, 1.0),),
                explicit_mask_keys=explicit_mask_keys,
                front_force_black=False,
                episode_manifest=source.get("episode_manifest"),
            )
            for source in resolution.prepared_source_manifests
        )
    if not source_configs:
        raise ValueError("Prepared deployment has no source configs.")
    first_root = Path(str(source_configs[0].dataset_root)).expanduser().resolve()
    layout_path = resolution.deployment_artifact_path.expanduser().resolve()
    if not layout_path.is_file():
        raise FileNotFoundError(
            f"Prepared deployment policy artifact is missing: {layout_path}"
        )
    layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(layout_payload, Mapping):
        raise ValueError(
            f"Prepared deployment policy artifact must be an object: {layout_path}"
        )
    actual_layout_hash = str(
        layout_payload.get("policy_io_layout_hash") or ""
    )
    if actual_layout_hash != artifact.layout_hash:
        raise ValueError(
            "Prepared deployment policy-I/O layout differs from the "
            "checkpoint artifact."
        )
    assets = dataclasses.replace(
        base.data.assets,
        assets_dir=str(resolution.checkpoint_dir / "assets"),
        asset_id=str(profile["openpi_asset_id"]),
    )
    data = dataclasses.replace(
        base.data,
        assets=assets,
        dataset_root=str(first_root),
        layout_path=str(layout_path),
        layout_hash=artifact.layout_hash,
        camera_slots=expected_slots,
        action_pose_mode=str(profile["action_pose_mode"]),
        last_action_pose_mode=str(profile["action_pose_mode"]),
        tail_policy=str(profile["tail_policy"]),
        strict_media_integrity=True,
        state_augmentation=state_augmentation,
        dataset_sources=source_configs,
        default_prompt=str(
            profile.get("default_prompt", base.data.default_prompt)
        ),
    )
    config_overrides: dict[str, Any] = {
        "name": resolution.openpi_config_name,
        "data": data,
        "exp_name": str(profile.get("default_exp_name") or base.exp_name),
        "resume": False,
        "overwrite": False,
        "num_workers": int(
            profile.get("openpi_num_workers", base.num_workers)
        ),
        "weight_loader": _prepared_weight_loader(
            profile,
            base.weight_loader,
            Path(str(profile["openpi_checkout"])).expanduser().resolve(),
        ),
    }
    if not hasattr(base, "policy_metadata"):
        raise ValueError(
            "Prepared OpenPI base config has no policy_metadata field."
        )
    policy_metadata = dict(getattr(base, "policy_metadata", None) or {})
    policy_metadata["policy_artifact"] = artifact.to_dict()
    config_overrides["policy_metadata"] = policy_metadata
    if resolution.model_rtc_profile is not None:
        mask_aware = bool(
            resolution.model_rtc_profile.get(
                "rtc_prefix_mask_conditioning",
                False,
            )
        )
        config_overrides["model"] = dataclasses.replace(
            base.model,
            rtc_prefix_mask_conditioning=mask_aware,
        )
    config = dataclasses.replace(base, **config_overrides)
    if config.name != artifact.model_id:
        raise ValueError(
            "Rebuilt OpenPI config name differs from the deployment artifact."
        )
    if str(config.data.assets.asset_id) != str(profile["openpi_asset_id"]):
        raise ValueError("Rebuilt OpenPI config asset id is incorrect.")
    if Path(str(config.data.assets.assets_dir)).resolve() != (
        resolution.checkpoint_dir / "assets"
    ).resolve():
        raise ValueError(
            "Rebuilt OpenPI config must load normalization assets from the "
            "validated checkpoint."
        )
    return config


def _prepared_request_image_keys(
    resolution: PreparedDeploymentResolution,
    config: Any,
) -> tuple[str, ...]:
    """Resolve the keys present in the actual OpenPI inference request."""

    camera_input = resolution.deployment_artifact.camera_input
    if camera_input is None:
        raise ValueError(
            "Prepared image-wire deployment requires a camera input contract."
        )
    expected_slots = tuple(str(value) for value in config.model.image_keys)
    if camera_input.server_camera_slots != expected_slots:
        raise ValueError(
            "Prepared image-wire camera slots differ from the loaded model."
        )
    request_keys = tuple(item.model_input_key for item in camera_input.inputs)
    if (
        not request_keys
        or any(not key for key in request_keys)
        or len(set(request_keys)) != len(request_keys)
    ):
        raise ValueError(
            "Prepared image-wire request keys must be unique and non-empty."
        )
    return request_keys


def prepared_config_summary(
    resolution: PreparedDeploymentResolution,
    config: Any,
    *,
    image_wire_config: OpenPIImageWireServerConfig | None = None,
    runtime_module_origins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return an auditable, JSON-safe summary of the restored config."""

    summary = {
        "ok": True,
        "config_name": str(config.name),
        "base_config_name": str(
            resolution.profile["openpi_base_config"]
        ),
        "asset_id": str(config.data.assets.asset_id),
        "action_horizon": int(config.model.action_horizon),
        "image_keys": [str(value) for value in config.model.image_keys],
        "camera_slots": [str(value) for value in config.data.camera_slots],
        "action_pose_mode": str(config.data.action_pose_mode),
        "last_action_pose_mode": str(config.data.last_action_pose_mode),
        "tail_policy": str(config.data.tail_policy),
        "default_prompt": str(config.data.default_prompt),
        "layout_hash": str(config.data.layout_hash),
        "layout_path": str(config.data.layout_path),
        "dataset_sources": [
            {
                "name": str(source.name),
                "dataset_root": str(source.dataset_root),
                "sample_weight": int(source.sample_weight),
                "episode_manifest": (
                    None
                    if source.episode_manifest is None
                    else str(source.episode_manifest)
                ),
            }
            for source in config.data.dataset_sources
        ],
        "profile_hash": str(resolution.binding["profile_hash"]),
        "norm_stats_hash": str(resolution.binding["norm_stats_hash"]),
        "binding_hash": str(resolution.binding["binding_hash"]),
        "model_rtc_profile_id": (
            None
            if resolution.model_rtc_profile is None
            else str(resolution.model_rtc_profile["profile_id"])
        ),
        "rtc_prefix_mask_conditioning": bool(
            getattr(config.model, "rtc_prefix_mask_conditioning", False)
        ),
    }
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        resolution.profile.get("openpi_observation_mode"),
        image_keys=tuple(str(value) for value in config.model.image_keys),
    )
    summary["observation_mode"] = (
        "state_only" if state_only_bridge is not None else "configured_cameras"
    )
    summary["external_camera_slots"] = (
        []
        if state_only_bridge is not None
        else [str(value) for value in config.data.camera_slots]
    )
    if state_only_bridge is not None:
        summary["state_only_input_contract"] = state_only_bridge.contract
    if image_wire_config is not None and image_wire_config.enabled:
        summary["image_wire_capability"] = build_openpi_image_wire_capability(
            image_wire_config,
            image_keys=_prepared_request_image_keys(resolution, config),
        )
    if runtime_module_origins is not None:
        summary["runtime_module_origins"] = dict(runtime_module_origins)
    return summary


def _serve_until_stopped(server_module: Any, server_args: Any) -> int:
    """Run the delegated server and keep an operator stop traceback-free."""

    try:
        server_module.main(server_args)
    except KeyboardInterrupt:
        print("OpenPI policy server stopped by operator.", file=sys.stderr)
        return 130
    return 0


@contextmanager
def install_openpi_rtc_inference_steps_server_adapter(
    server_module: Any,
    *,
    rtc_inference_steps: int,
) -> Iterator[None]:
    """Bind the prepared RTC sampler budget across OpenPI server revisions.

    The OpenPI ``serve_dexteleop_policy.Args`` dataclass is not the model
    contract: older checkouts do not expose an ``rtc_inference_steps`` CLI
    field, while the policy factory already accepts the canonical
    ``sample_kwargs={"num_steps": ...}`` binding.  Passing the deployment
    option through ``Args`` therefore crashes before the server can listen.
    Inject the validated budget at the policy-construction seam instead and
    restore the imported module after the foreground server exits.
    """

    steps = int(rtc_inference_steps)
    policy_config = getattr(server_module, "_policy_config", None)
    create_trained_policy = getattr(policy_config, "create_trained_policy", None)
    if not callable(create_trained_policy):
        raise RuntimeError(
            "OpenPI server module does not expose policy_config.create_trained_policy"
        )

    def create_prepared_policy(*args: Any, **kwargs: Any) -> Any:
        sample_kwargs_value = kwargs.get("sample_kwargs")
        sample_kwargs = (
            {}
            if sample_kwargs_value is None
            else dict(sample_kwargs_value)
        )
        existing_steps = sample_kwargs.get("num_steps")
        if existing_steps is not None and int(existing_steps) != steps:
            raise ValueError(
                "prepared RTC inference steps differ from the OpenPI policy "
                f"factory request: requested={steps} existing={existing_steps}"
            )
        sample_kwargs["num_steps"] = steps
        kwargs["sample_kwargs"] = sample_kwargs
        return create_trained_policy(*args, **kwargs)

    policy_config.create_trained_policy = create_prepared_policy
    try:
        yield
    finally:
        policy_config.create_trained_policy = create_trained_policy


@dataclasses.dataclass(frozen=True)
class _RtcStartupWarmupSpec:
    """The already-validated tensor contract needed for one RTC warmup."""

    image_keys: tuple[str, ...]
    image_shape: tuple[int, int, int]
    base_state_dim: int
    state_input_mode: str
    action_dim: int
    image_mask_value: bool = True


def _prepared_rtc_startup_warmup_spec(
    resolution: PreparedDeploymentResolution,
    config: Any,
    *,
    state_only_bridge: OpenPIStateOnlyInputBridge | None,
) -> _RtcStartupWarmupSpec:
    """Derive the warmup shape from the validated artifact, not from guesses."""

    artifact = resolution.deployment_artifact
    layout = artifact.policy_io_layout
    base_state_dim = int(layout.state_dim)
    action_dim = int(layout.action_dim)
    state_input_mode = str(artifact.state_input_mode)
    state_input_dim = int(artifact.state_input_dim)
    if base_state_dim <= 0 or action_dim <= 0:
        raise ValueError(
            "RTC startup warmup requires positive state and action dimensions."
        )
    expected_state_input_dim = (
        base_state_dim * 2
        if state_input_mode == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT
        else base_state_dim
    )
    if state_input_dim != expected_state_input_dim:
        raise ValueError(
            "RTC startup warmup state dimension disagrees with the prepared "
            f"artifact: mode={state_input_mode!r} expected="
            f"{expected_state_input_dim} actual={state_input_dim}."
        )
    image_keys = tuple(str(value) for value in config.model.image_keys)
    if not image_keys or len(set(image_keys)) != len(image_keys):
        raise ValueError(
            "RTC startup warmup requires unique configured OpenPI image keys."
        )
    if state_only_bridge is not None:
        image_shape = tuple(int(value) for value in state_only_bridge.image_shape)
        image_mask_value = False
    else:
        # OpenPI's visual model contract is fixed at this resolution; the
        # prepared image artifact has already validated the camera slots.
        image_shape = (224, 224, 3)
        image_mask_value = True
    return _RtcStartupWarmupSpec(
        image_keys=image_keys,
        image_shape=image_shape,
        base_state_dim=base_state_dim,
        state_input_mode=state_input_mode,
        action_dim=action_dim,
        image_mask_value=image_mask_value,
    )


def _run_rtc_startup_warmup(
    policy: Any,
    spec: _RtcStartupWarmupSpec,
) -> float:
    """Compile the conditioned RTC path before the server accepts clients.

    The all-zero prefix is an explicit provider bootstrap operation.  Its
    result is deliberately never published or returned to a caller; this is a
    server-local JAX/model warmup only.  Any failure is propagated so the
    server does not expose a cold or partially initialized endpoint.
    """

    infer_rtc = getattr(policy, "infer_rtc", None)
    if not callable(infer_rtc):
        raise RuntimeError(
            "RTC startup warmup requires a policy with infer_rtc()."
        )
    observation = {
        "state": deployment_state_input(
            np.zeros((spec.base_state_dim,), dtype=np.float32),
            spec.state_input_mode,
        ),
        "image": {
            key: np.zeros(spec.image_shape, dtype=np.uint8)
            for key in spec.image_keys
        },
        "image_mask": {
            key: np.bool_(spec.image_mask_value) for key in spec.image_keys
        },
    }
    print(
        "[OpenPIRuntime] RTC startup warmup starting; "
        "server_listen=blocked; action_publication=none",
        flush=True,
    )
    started = time.monotonic()
    result = infer_rtc(
        observation,
        conditioning_actions=np.zeros(
            (1, spec.action_dim), dtype=np.float32
        ),
        prefix_mask=np.zeros((1,), dtype=np.bool_),
        conditioning_mode="unconditioned_bootstrap_v1",
        receipt_context={"warmup": "prepared_server_startup_v1"},
    )
    elapsed = time.monotonic() - started
    if not isinstance(result, Mapping):
        raise RuntimeError("RTC startup warmup returned a non-mapping result.")
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != spec.action_dim:
        raise RuntimeError(
            "RTC startup warmup returned an unexpected action shape: "
            f"expected second dimension {spec.action_dim}, got {actions.shape}."
        )
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("RTC startup warmup returned non-finite actions.")
    print(
        "[OpenPIRuntime] RTC startup warmup complete "
        f"elapsed={elapsed:.3f}s; result_discarded=true; "
        "action_publication=none; server_listen=starting",
        flush=True,
    )
    return elapsed


@contextmanager
def install_openpi_rtc_startup_warmup_server_adapter(
    websocket_policy_server_module: Any,
    spec: _RtcStartupWarmupSpec | None,
) -> Iterator[None]:
    """Run one local RTC bootstrap before the delegated server listens."""

    if spec is None:
        yield
        return
    original = websocket_policy_server_module.WebsocketPolicyServer

    class RtcStartupWarmupWebsocketPolicyServer(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            policy = kwargs.get("policy")
            if policy is None and args:
                policy = args[0]
            if policy is None:
                raise TypeError(
                    "OpenPI WebsocketPolicyServer warmup wrapper requires policy."
                )
            super().__init__(*args, **kwargs)
            self._prepared_rtc_warmup_policy = policy
            self._prepared_rtc_warmup_done = False

        def serve_forever(self) -> Any:
            if not self._prepared_rtc_warmup_done:
                _run_rtc_startup_warmup(
                    self._prepared_rtc_warmup_policy,
                    spec,
                )
                self._prepared_rtc_warmup_done = True
            return super().serve_forever()

    RtcStartupWarmupWebsocketPolicyServer.__name__ = original.__name__
    RtcStartupWarmupWebsocketPolicyServer.__qualname__ = original.__qualname__
    websocket_policy_server_module.WebsocketPolicyServer = (
        RtcStartupWarmupWebsocketPolicyServer
    )
    try:
        yield
    finally:
        if (
            websocket_policy_server_module.WebsocketPolicyServer
            is RtcStartupWarmupWebsocketPolicyServer
        ):
            websocket_policy_server_module.WebsocketPolicyServer = original


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a checkpoint-bound prepared OpenPI config."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy-artifact-json", type=Path, required=True)
    parser.add_argument("--prepared-training-profile", type=Path, required=True)
    parser.add_argument("--model-rtc-profile", type=Path, default=None)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument(
        "--training-openpi-root",
        type=Path,
        default=None,
        help=(
            "Profile-pinned training checkout when --openpi-root selects the "
            "8-step deployment runtime. Defaults to --openpi-root."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--enable-rtc-continuation", action="store_true")
    parser.add_argument(
        "--rtc-inference-steps",
        type=int,
        choices=(8, 10),
        default=10,
        help=(
            "Versioned conditioned-RTC sampler budget; 10 preserves the "
            "trained default and 8 selects the latency deployment binding."
        ),
    )
    parser.add_argument(
        "--image-wire-profile",
        choices=("off", "resize-raw", "jpeg"),
        default="off",
        help=(
            "Additive prepared-server image capability. off keeps the exact "
            "historical server class and metadata."
        ),
    )
    parser.add_argument("--image-wire-height", type=int, default=224)
    parser.add_argument("--image-wire-width", type=int, default=224)
    parser.add_argument("--image-wire-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Rebuild and temporarily register the prepared OpenPI config, "
            "print its bound summary, and exit without loading a model or "
            "starting a server."
        ),
    )
    args = parser.parse_args(argv)
    if not args.enable_rtc_continuation and args.rtc_inference_steps != 10:
        parser.error(
            "--rtc-inference-steps 8 requires --enable-rtc-continuation."
        )
    image_wire_config = OpenPIImageWireServerConfig(
        profile=args.image_wire_profile,
        target_height=args.image_wire_height,
        target_width=args.image_wire_width,
        jpeg_quality=args.image_wire_jpeg_quality,
    )

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    openpi_root = args.openpi_root.expanduser().resolve()
    training_openpi_root = (
        openpi_root
        if args.training_openpi_root is None
        else args.training_openpi_root.expanduser().resolve()
    )
    resolution = validate_prepared_deployment_artifact(
        checkpoint_dir,
        args.prepared_training_profile,
        args.policy_artifact_json,
        model_rtc_profile_path=args.model_rtc_profile,
    )
    profile_default_prompt = str(
        resolution.profile.get("default_prompt") or ""
    ).strip()
    if (
        args.default_prompt is not None
        and profile_default_prompt
        and args.default_prompt != profile_default_prompt
    ):
        parser.error(
            "--default-prompt differs from the prepared training profile."
        )
    resolved_default_prompt = (
        args.default_prompt
        if args.default_prompt is not None
        else (profile_default_prompt or None)
    )
    if resolution.deployment_artifact.model_id != str(args.model):
        parser.error("--model differs from the prepared deployment artifact.")
    profile_openpi_root = Path(
        str(resolution.profile["openpi_checkout"])
    ).expanduser().resolve()
    if profile_openpi_root != training_openpi_root:
        parser.error(
            "--training-openpi-root differs from the prepared training profile."
        )
    try:
        _validate_runtime_openpi_binding(
            runtime_root=openpi_root,
            training_root=training_openpi_root,
            model_rtc_profile=args.model_rtc_profile,
            rtc_inference_steps=args.rtc_inference_steps,
        )
    except ValueError as exc:
        parser.error(str(exc))

    training_config, dexteleop_v23_policy, runtime_module_origins = (
        _import_runtime_openpi_modules(openpi_root)
    )

    config = build_prepared_openpi_config(
        resolution,
        training_config=training_config,
        dataset_policy=dexteleop_v23_policy,
    )
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        resolution.profile.get("openpi_observation_mode"),
        image_keys=tuple(str(value) for value in config.model.image_keys),
    )
    request_image_keys: tuple[str, ...] | None = None
    if state_only_bridge is not None and image_wire_config.enabled:
        raise ValueError(
            "State-only prepared deployment cannot enable image-wire transport."
        )
    if image_wire_config.enabled:
        from openpi.models import model as openpi_model

        model_resolution = tuple(int(value) for value in openpi_model.IMAGE_RESOLUTION)
        requested_resolution = (
            image_wire_config.target_height,
            image_wire_config.target_width,
        )
        if requested_resolution != model_resolution:
            raise ValueError(
                "Prepared image-wire target differs from the loaded OpenPI "
                f"model resolution: {requested_resolution!r} != "
                f"{model_resolution!r}."
            )
        request_image_keys = _prepared_request_image_keys(resolution, config)
        training_alignment = resolution.checkpoint_payload.get(
            OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1
        )
        if training_alignment is not None:
            expected_training_capability = build_openpi_image_wire_capability(
                image_wire_config,
                image_keys=tuple(str(value) for value in config.model.image_keys),
            )
            if not isinstance(training_alignment, Mapping) or (
                training_alignment.get("server_capability")
                != expected_training_capability
            ):
                raise ValueError(
                    "Prepared image-wire training capability differs from the "
                    "checkpoint training alignment contract."
                )
    registry = training_config._CONFIGS_DICT
    previous = registry.get(config.name)
    if previous is not None and previous != config:
        raise ValueError(
            "OpenPI config registry already contains a different config with "
            f"name {config.name!r}."
        )
    rtc_warmup_spec = (
        _prepared_rtc_startup_warmup_spec(
            resolution,
            config,
            state_only_bridge=state_only_bridge,
        )
        if args.enable_rtc_continuation and not args.validate_only
        else None
    )
    registry[config.name] = config
    try:
        if args.validate_only:
            print(
                json.dumps(
                    prepared_config_summary(
                        resolution,
                        config,
                        image_wire_config=image_wire_config,
                        runtime_module_origins=runtime_module_origins,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(
            "[OpenPIRuntime] READY: selected module origins "
            + " ".join(
                f"{name}={origin}"
                for name, origin in sorted(runtime_module_origins.items())
            ),
            flush=True,
        )
        server_module = _load_module(
            openpi_root / "scripts" / "serve_dexteleop_policy.py",
            "_dexteleop_prepared_openpi_server",
        )
        server_kwargs = {
            "model": str(args.model),
            "registry": args.registry.expanduser().resolve(),
            "host": str(args.host),
            "port": int(args.port),
            "checkpoint_dir": None,
            "default_prompt": resolved_default_prompt,
            "record": bool(args.record),
            "enable_rtc_continuation": bool(
                args.enable_rtc_continuation
            ),
        }
        # Newer OpenPI server revisions expose the sampler budget on their
        # Args dataclass.  Bind the prepared runtime's explicit CLI value
        # when that field exists; older revisions are handled by the policy
        # factory adapter below and must not receive an unknown constructor
        # keyword.
        server_args_fields = getattr(server_module.Args, "__dataclass_fields__", {})
        if "rtc_inference_steps" in server_args_fields:
            server_kwargs["rtc_inference_steps"] = int(args.rtc_inference_steps)
        server_args = server_module.Args(**server_kwargs)
        rtc_steps_adapter = (
            install_openpi_rtc_inference_steps_server_adapter(
                server_module,
                rtc_inference_steps=int(args.rtc_inference_steps),
            )
            if args.enable_rtc_continuation
            else nullcontext()
        )
        with install_openpi_image_wire_server_adapter(
            server_module.websocket_policy_server,
            image_wire_config,
            image_keys=request_image_keys,
        ), install_openpi_state_only_server_adapter(
            server_module.websocket_policy_server,
            state_only_bridge,
        ), _use_checkpoint_policy_metadata(
            training_config,
            config,
            resolution,
        ), rtc_steps_adapter, install_openpi_rtc_startup_warmup_server_adapter(
            server_module.websocket_policy_server,
            rtc_warmup_spec,
        ):
            return _serve_until_stopped(server_module, server_args)
    finally:
        if previous is None:
            registry.pop(config.name, None)
        else:
            registry[config.name] = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_prepared_openpi_config",
    "install_openpi_rtc_startup_warmup_server_adapter",
    "main",
    "prepared_config_summary",
]
