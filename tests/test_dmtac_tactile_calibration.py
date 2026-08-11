from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hardware.DMTAC_gripper.portable_dmtac_gripper.tactile_calibration import (
    BASE_FRAME_SHAPE,
    build_stable_base_frame,
    load_tactile_calibration,
    save_tactile_calibration,
)
from hardware.DMTAC_gripper.portable_dmtac_gripper.device import (
    DMTacGripperDevice,
    TactileConfig,
)
from utils.dataset_postprocess.DM_force_calculate.processor import (
    FORCE_SHAPE,
    RAW_SHAPE,
    ForceProcessConfig,
    RawTactileStreamTask,
    process_raw_tactile_stream,
)


class _FakeCalculator:
    def __init__(self, **_: object) -> None:
        self.base: np.ndarray | None = None

    def set_base_frame(self, raw: np.ndarray) -> None:
        self.base = np.asarray(raw).copy()

    def distributed_force(self, raw: np.ndarray) -> np.ndarray:
        del raw
        value = 1.0 if self.base is not None else 2.0
        return np.full(FORCE_SHAPE, value, dtype=np.float32)

    def close(self) -> None:
        return None


def _task(tmp_path: Path, sensor: str = "M_TEST") -> RawTactileStreamTask:
    source = tmp_path / "raw.bin"
    np.zeros((2, *RAW_SHAPE), dtype=np.uint8).tofile(source)
    return RawTactileStreamTask(
        dataset_root=tmp_path,
        output_root=tmp_path,
        source_manifest_path=tmp_path / "raw_manifest.json",
        output_manifest_path=tmp_path / "force_manifest.json",
        side="left",
        episode_index=4,
        sensor_key=sensor,
        source_stream_path=source,
        output_stream_path=tmp_path / "force.bin",
        frame_count=2,
        source_shape=RAW_SHAPE,
        source_dtype="uint8",
    )


@pytest.mark.integration
def test_fixed_calibration_round_trip_and_checksum(tmp_path: Path) -> None:
    frame = np.arange(np.prod(BASE_FRAME_SHAPE), dtype=np.uint8).reshape(BASE_FRAME_SHAPE)
    saved = save_tactile_calibration(
        tmp_path,
        "M_TEST",
        frame,
        source={"kind": "unit_test"},
    )

    loaded = load_tactile_calibration(tmp_path, "M_TEST", required=True)

    assert loaded is not None
    assert loaded.sha256 == saved.sha256
    assert loaded.metadata["source"] == {"kind": "unit_test"}
    np.testing.assert_array_equal(loaded.frame, frame)


@pytest.mark.integration
def test_fixed_calibration_rejects_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        save_tactile_calibration(tmp_path, "M_TEST", np.zeros((4, 5), dtype=np.uint8))


def test_stable_base_frame_uses_pixelwise_median_and_reports_noise() -> None:
    frames = [
        np.full(BASE_FRAME_SHAPE, value, dtype=np.uint8)
        for value in (10, 12, 200)
    ]

    base, metrics = build_stable_base_frame(frames)

    assert base.dtype == np.uint8
    assert base.shape == BASE_FRAME_SHAPE
    assert np.all(base == 12)
    assert metrics["mean_abs_deviation"] > 0
    assert metrics["p99_abs_deviation"] >= metrics["mean_abs_deviation"]


@pytest.mark.integration
def test_postprocess_auto_loads_fixed_sensor_calibration(tmp_path: Path) -> None:
    save_tactile_calibration(
        tmp_path / "calibration",
        "M_TEST",
        np.full(RAW_SHAPE, 7, dtype=np.uint8),
    )
    task = _task(tmp_path)

    result = process_raw_tactile_stream(
        task,
        {
            "calibration_policy": "auto",
            "calibration_dir": str(tmp_path / "calibration"),
            "overwrite": True,
            "backend": "cpu",
            "mode": "standard",
            "compression": "none",
        },
        calculator_factory=_FakeCalculator,
    )

    assert result.status == "processed"
    assert result.calibration["status"] == "loaded_fixed"
    output = np.memmap(task.output_stream_path, dtype=np.float32, mode="r", shape=(2, *FORCE_SHAPE))
    assert float(output[0, 0, 0, 0]) == 1.0


@pytest.mark.integration
def test_postprocess_required_calibration_fails_when_missing(tmp_path: Path) -> None:
    result = process_raw_tactile_stream(
        _task(tmp_path),
        {
            "calibration_policy": "required",
            "calibration_dir": str(tmp_path / "missing"),
            "overwrite": True,
        },
        calculator_factory=_FakeCalculator,
    )

    assert result.status == "failed"
    assert "No fixed DM-Tac calibration" in (result.error or "")


@pytest.mark.integration
def test_save_calibration_requires_one_episode_and_explicit_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="base_frame_index"):
        ForceProcessConfig(dataset_root=tmp_path, save_calibration=True).normalized()
    with pytest.raises(ValueError, match="exactly one episode"):
        ForceProcessConfig(
            dataset_root=tmp_path,
            base_frame_index=0,
            save_calibration=True,
        ).normalized()


@pytest.mark.integration
def test_live_device_auto_applies_and_explicit_reset_persists(tmp_path: Path) -> None:
    fixed = np.full(RAW_SHAPE, 11, dtype=np.uint8)
    reset_frame = np.full(RAW_SHAPE, 23, dtype=np.uint8)
    save_tactile_calibration(tmp_path, "M_TEST", fixed)

    class _Sensor:
        dev_id = "M_TEST"

        def __init__(self) -> None:
            self.applied: np.ndarray | None = None

        def set_base_frame(self, frame: np.ndarray) -> None:
            self.applied = np.asarray(frame).copy()

    class _Multi:
        def __init__(self) -> None:
            self.devices = [_Sensor()]
            self.reset_calls = 0

        def reset_all(self) -> None:
            self.reset_calls += 1

        def get_base_frames(self) -> dict[str, np.ndarray]:
            return {"M_TEST": reset_frame}

    device = DMTacGripperDevice(
        tactile=TactileConfig(calibration_dir=tmp_path),
        connect_gripper=False,
        connect_tactile=False,
    )
    device.tactile = _Multi()  # type: ignore[assignment]

    applied = device.apply_fixed_tactile_calibration()
    np.testing.assert_array_equal(device.tactile.devices[0].applied, fixed)
    assert applied["M_TEST"]["sha256"]

    device.reset_tactile()
    loaded = load_tactile_calibration(tmp_path, "M_TEST", required=True)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.frame, reset_frame)
