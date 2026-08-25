from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from dataset_contracts import (
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)
from dataset_contracts.v22 import PolicyIOLayout
from teleop_runtime.deploy import camera_input_contract, camera_runtime, run_policy


def _artifact(*keys: str, required: bool = True) -> PolicyArtifactContract:
    return PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        camera_input=PolicyCameraInputContract(
            inputs=tuple(
                PolicyCameraInput(
                    source_key=key,
                    model_input_key=f"observation.images.{key}",
                    server_camera_slot=f"{key}_0_rgb",
                )
                for key in keys
            ),
            required=required,
        ),
    )


def _args(
    *,
    camera_mode: str,
    camera_serial: str = "",
    camera_key: str = "front",
) -> argparse.Namespace:
    return SimpleNamespace(
        camera_mode=camera_mode,
        camera_serial=camera_serial,
        camera_key=camera_key,
    )


def test_facade_reexports_the_extracted_contract_function() -> None:
    assert (
        camera_runtime.resolve_policy_camera_input_keys
        is camera_input_contract.resolve_policy_camera_input_keys
    )
    assert (
        run_policy.resolve_policy_camera_input_keys
        is camera_input_contract.resolve_policy_camera_input_keys
    )


def test_legacy_artifact_keeps_unconstrained_camera_behavior() -> None:
    args = _args(camera_mode="multi")

    assert (
        camera_input_contract.resolve_policy_camera_input_keys(
            args,
            None,
            raw_argv=[],
        )
        is None
    )


def test_required_artifact_rejects_camera_mode_none() -> None:
    with pytest.raises(ValueError, match="requires camera inputs"):
        camera_input_contract.resolve_policy_camera_input_keys(
            _args(camera_mode="none"),
            _artifact("left_wrist"),
            raw_argv=[],
        )


def test_optional_artifact_allows_camera_mode_none() -> None:
    args = _args(camera_mode="none")

    assert camera_input_contract.resolve_policy_camera_input_keys(
        args,
        _artifact("left_wrist", required=False),
        raw_argv=[],
    ) == ("left_wrist",)


def test_serial_camera_rekeys_to_the_artifact_input() -> None:
    args = _args(camera_mode="single", camera_serial="123")

    selected = camera_input_contract.resolve_policy_camera_input_keys(
        args,
        _artifact("right_wrist"),
        raw_argv=[],
    )

    assert selected == ("right_wrist",)
    assert args.camera_key == "right_wrist"


def test_explicit_serial_camera_key_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagrees with policy artifact"):
        camera_input_contract.resolve_policy_camera_input_keys(
            _args(camera_mode="single", camera_serial="123", camera_key="front"),
            _artifact("right_wrist"),
            raw_argv=["--camera-key", "front"],
        )


def test_multi_input_artifact_rejects_serial_camera() -> None:
    with pytest.raises(ValueError, match="only one artifact camera input"):
        camera_input_contract.resolve_policy_camera_input_keys(
            _args(camera_mode="multi", camera_serial="123"),
            _artifact("left_wrist", "right_wrist"),
            raw_argv=[],
        )


def test_single_mode_requires_exactly_one_artifact_input() -> None:
    with pytest.raises(ValueError, match="exactly one artifact camera input"):
        camera_input_contract.resolve_policy_camera_input_keys(
            _args(camera_mode="single"),
            _artifact("left_wrist", "right_wrist"),
            raw_argv=[],
        )
