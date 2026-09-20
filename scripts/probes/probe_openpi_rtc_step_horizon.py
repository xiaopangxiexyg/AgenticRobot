#!/usr/bin/env python3
"""Audit OpenPI model-RTC step/horizon pairs without starting a runtime.

This probe is deliberately diagnostic-only.  It reads checked-in training and
model-RTC profiles plus optional policy/binding sidecars, then reports which
``effective_num_steps``/prediction-horizon pairs are compatible with the
existing contract.  It never imports OpenPI, loads weights, starts a server,
uses a GPU, connects to a device, or edits any input file.

The probe does not manufacture a new artifact or relax a contract.  A pair is
``compatible`` only when the selected sampling step is in the existing
deployment allow-list and the requested horizon agrees with the prepared
training profile, artifact continuation runtime, and optional provider
capability sidecar.  Unsupported pairs remain visible as diagnostic results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from deployments.openpi.model_rtc_profile import load_prepared_model_rtc_profile
from deployments.policy_artifact import load_policy_artifact


DEFAULT_PAIRS = ((8, 32), (6, 32), (4, 32), (8, 24), (6, 24), (4, 24))
# Keep this probe lightweight: importing the runtime binding also imports the
# robot-motion stack and optional solver dependencies.  This is the current
# public trained-artifact deployment allow-list declared by that binding.  The
# report names it explicitly so a future contract change is visible in review.
CURRENT_DEPLOYMENT_STEP_ALLOWLIST = (8, 10)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _optional_nonempty_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_pair(value: str) -> tuple[int, int]:
    raw = str(value).strip().lower()
    for separator in ("x", ":", "/"):
        if separator in raw:
            left, right = raw.split(separator, 1)
            try:
                steps, horizon = int(left), int(right)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid pair {value!r}; use STEPSxHORIZON"
                ) from exc
            if steps <= 0 or horizon <= 0:
                raise argparse.ArgumentTypeError(
                    f"invalid pair {value!r}; values must be positive"
                )
            return steps, horizon
    raise argparse.ArgumentTypeError(
        f"invalid pair {value!r}; use STEPSxHORIZON, e.g. 8x32"
    )


def _profile_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    cameras = profile.get("camera_slot_to_dataset_key")
    if not isinstance(cameras, Mapping):
        raise ValueError("prepared training profile camera mapping is invalid")
    return {
        "profile_id": str(profile["profile_id"]),
        "action_horizon": int(profile["action_horizon"]),
        "action_pose_mode": str(profile["action_pose_mode"]),
        "camera_slot_to_dataset_key": {
            str(key): str(value) for key, value in cameras.items()
        },
        "expected_policy_io_layout_hash": _optional_nonempty_string(
            profile.get("expected_policy_io_layout_hash")
        ),
        "openpi_config_name": _optional_nonempty_string(
            profile.get("openpi_config_name")
        ),
    }


def _model_rtc_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    sampling = profile.get("sampling_spec")
    if not isinstance(sampling, Mapping):
        raise ValueError("model-RTC profile sampling_spec is invalid")
    pairs = sampling.get("delay_prefix_pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ValueError("model-RTC profile delay_prefix_pairs is invalid")
    return {
        "schema": str(profile["schema"]),
        "profile_id": str(profile["profile_id"]),
        "prepared_skill_profile_id": str(profile["prepared_skill_profile_id"]),
        "output_config_name": str(profile["output_config_name"]),
        "training_implementation": _optional_nonempty_string(
            profile.get("training_implementation")
        ),
        "rtc_prefix_mask_conditioning": profile.get(
            "rtc_prefix_mask_conditioning"
        ),
        "minimum_forecast_delay_steps": int(
            profile["minimum_forecast_delay_steps"]
        ),
        "maximum_forecast_delay_steps": int(
            profile["maximum_forecast_delay_steps"]
        ),
        "minimum_prefix_steps": int(profile["minimum_prefix_steps"]),
        "maximum_prefix_steps": int(profile["maximum_prefix_steps"]),
        "sampling_delay_prefix_pairs": [
            [int(item[0]), int(item[1])] for item in pairs
        ],
    }


def _artifact_summary(path: Path) -> dict[str, Any]:
    artifact = load_policy_artifact(path)
    continuation = artifact.model_continuation
    runtime: Mapping[str, Any] | None = None
    if continuation is not None:
        runtime = continuation.runtime.to_dict()
    camera_input = artifact.camera_input
    return {
        "source": str(path.expanduser().resolve()),
        "model_id": artifact.model_id,
        "checkpoint_step": artifact.checkpoint_step,
        "action_dim": artifact.policy_io_layout.action_dim,
        "action_pose_mode": (
            None
            if artifact.pose_action_chunk is None
            else artifact.pose_action_chunk.action_pose_mode
        ),
        "policy_io_layout_hash": artifact.layout_hash,
        "camera_slots": []
        if camera_input is None
        else list(camera_input.server_camera_slots),
        "model_continuation_runtime": runtime,
        "continuation_implementation_id": (
            None
            if runtime is None
            else runtime.get("continuation_implementation_id")
        ),
        "prediction_horizon_steps": (
            None if runtime is None else runtime.get("prediction_horizon_steps")
        ),
        "effective_num_steps": None,
        "effective_num_steps_source": "not published in policy artifact",
    }


def _binding_summary(path: Path) -> dict[str, Any]:
    payload = _json_object(path, label="binding receipt")
    provider = payload.get("provider_capability")
    if not isinstance(provider, Mapping):
        raise ValueError("binding receipt provider_capability is missing")
    parameters = provider.get("continuation_implementation_parameters")
    if not isinstance(parameters, Mapping):
        parameters = payload.get("implementation_parameters")
    if not isinstance(parameters, Mapping):
        parameters = None
    return {
        "source": str(path.expanduser().resolve()),
        "provider_capability_schema": _optional_nonempty_string(
            provider.get("provider_capability_schema")
        ),
        "wire_protocol": _optional_nonempty_string(provider.get("wire_protocol")),
        "provider_prediction_horizon_steps": provider.get(
            "prediction_horizon_steps"
        ),
        "provider_external_action_dim": provider.get("external_action_dim"),
        "continuation_implementation_id": _optional_nonempty_string(
            provider.get("continuation_implementation_id")
        ),
        "effective_num_steps": (
            None if parameters is None else parameters.get("effective_num_steps")
        ),
        "effective_num_steps_source": (
            "provider_capability.continuation_implementation_parameters"
            if isinstance(provider.get("continuation_implementation_parameters"), Mapping)
            else "binding_receipt.implementation_parameters"
            if isinstance(payload.get("implementation_parameters"), Mapping)
            else "not published in binding receipt"
        ),
    }


def _candidate_result(
    *,
    steps: int,
    horizon: int,
    training: Mapping[str, Any],
    model_rtc: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
    binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    training_horizon = int(training["action_horizon"])
    if horizon != training_horizon:
        reasons.append(
            f"horizon_mismatch_training_profile:{training_horizon}"
        )

    expected_cameras = tuple(training["camera_slot_to_dataset_key"])
    if artifact is None:
        reasons.append("artifact_sidecar_required_for_runtime_check")
    else:
        if int(artifact["action_dim"]) <= 0:
            reasons.append("artifact_action_dim_invalid")
        if artifact["policy_io_layout_hash"] != training.get(
            "expected_policy_io_layout_hash"
        ):
            reasons.append("artifact_policy_io_layout_mismatch")
        if tuple(artifact["camera_slots"]) != expected_cameras:
            reasons.append("artifact_camera_slots_mismatch")
        artifact_horizon = artifact.get("prediction_horizon_steps")
        if artifact_horizon is None:
            reasons.append("artifact_prediction_horizon_missing")
        elif int(artifact_horizon) != horizon:
            reasons.append(
                f"horizon_mismatch_artifact_runtime:{int(artifact_horizon)}"
            )

    if steps not in CURRENT_DEPLOYMENT_STEP_ALLOWLIST:
        reasons.append(
            "effective_num_steps_not_in_current_deployment_allowlist:"
            + repr(list(CURRENT_DEPLOYMENT_STEP_ALLOWLIST))
        )

    if binding is None:
        warnings.append("provider_capability_sidecar_not_supplied")
    else:
        binding_steps = binding.get("effective_num_steps")
        if binding_steps is None:
            warnings.append("provider_effective_num_steps_not_published")
        elif int(binding_steps) != steps:
            reasons.append(f"effective_num_steps_mismatch_provider:{binding_steps}")
        provider_horizon = binding.get("provider_prediction_horizon_steps")
        if provider_horizon is None:
            warnings.append("provider_prediction_horizon_not_published")
        elif int(provider_horizon) != horizon:
            reasons.append(
                f"horizon_mismatch_provider:{int(provider_horizon)}"
            )

    return {
        "effective_num_steps": steps,
        "prediction_horizon_steps": horizon,
        "status": "compatible" if not reasons else "rejected",
        "reasons": reasons,
        "warnings": warnings,
        "runtime_action": "none; diagnostic-only",
    }


def audit(
    *,
    prepared_training_profile: Path,
    model_rtc_profile: Path,
    artifact_json: Path | None,
    binding_receipt_json: Path | None,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    training_payload = _json_object(
        prepared_training_profile.expanduser().resolve(),
        label="prepared training profile",
    )
    loaded_model_rtc = load_prepared_model_rtc_profile(model_rtc_profile)
    training = _profile_summary(training_payload)
    model_rtc = _model_rtc_summary(loaded_model_rtc)
    if model_rtc["prepared_skill_profile_id"] != training["profile_id"]:
        raise ValueError(
            "model-RTC profile is bound to a different prepared training profile"
        )
    artifact = None if artifact_json is None else _artifact_summary(artifact_json)
    binding = (
        None
        if binding_receipt_json is None
        else _binding_summary(binding_receipt_json)
    )
    results = [
        _candidate_result(
            steps=steps,
            horizon=horizon,
            training=training,
            model_rtc=model_rtc,
            artifact=artifact,
            binding=binding,
        )
        for steps, horizon in pairs
    ]
    return {
        "schema": "dexteleop.openpi_rtc_step_horizon_diagnostic.v1",
        "runtime_started": False,
        "gpu_used": False,
        "hardware_contacted": False,
        "prepared_training_profile": training,
        "model_rtc_profile": model_rtc,
        "artifact": artifact,
        "binding_receipt": binding,
        "deployment_step_allowlist": list(CURRENT_DEPLOYMENT_STEP_ALLOWLIST),
        "requested_pairs": [
            {"effective_num_steps": steps, "prediction_horizon_steps": horizon}
            for steps, horizon in pairs
        ],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-training-profile", required=True, type=Path)
    parser.add_argument("--model-rtc-profile", required=True, type=Path)
    parser.add_argument("--artifact-json", type=Path)
    parser.add_argument("--binding-receipt-json", type=Path)
    parser.add_argument(
        "--pair",
        dest="pairs",
        action="append",
        type=_parse_pair,
        metavar="STEPSxHORIZON",
        help="repeatable candidate pair; defaults to 8x32, 6x32, 4x32, 8x24, 6x24, 4x24",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pairs = tuple(args.pairs or DEFAULT_PAIRS)
    report = audit(
        prepared_training_profile=args.prepared_training_profile,
        model_rtc_profile=args.model_rtc_profile,
        artifact_json=args.artifact_json,
        binding_receipt_json=args.binding_receipt_json,
        pairs=pairs,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        print(str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
