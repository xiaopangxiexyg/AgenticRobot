from __future__ import annotations

import importlib

import pytest


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

rgb_distortion = importlib.import_module("sim.mujoco.replay.rgb_distortion")


def _gradient_image(width: int = 64, height: int = 48):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy, np.full_like(xx, 127)], axis=-1)


def test_real_raw_rgb_distorter_noops_zero_distortion() -> None:
    image = _gradient_image()
    distorter = rgb_distortion.RealRawRgbDistorter(
        camera_intrinsics_by_name={
            "left_wrist": {
                "width": 64,
                "height": 48,
                "fx": 40.0,
                "fy": 40.0,
                "ppx": 32.0,
                "ppy": 24.0,
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        },
        camera_metadata_by_name={"left_wrist": {"fovy_deg": 61.9275}},
    )

    output = distorter.distort("left_wrist", image)

    assert output is image
    metadata = distorter.metadata()
    assert metadata["cameras"]["left_wrist"]["enabled"] is False


def test_real_raw_rgb_distorter_remaps_nonzero_distortion() -> None:
    image = _gradient_image()
    distorter = rgb_distortion.RealRawRgbDistorter(
        camera_intrinsics_by_name={
            "left_wrist": {
                "width": 64,
                "height": 48,
                "fx": 40.0,
                "fy": 40.0,
                "ppx": 32.0,
                "ppy": 24.0,
                "coeffs": [-0.08, 0.03, 0.001, 0.0, 0.0],
            }
        },
        camera_metadata_by_name={"left_wrist": {"fovy_deg": 61.9275}},
        source_camera_model="mujoco",
    )

    output = distorter.distort("left_wrist", image)
    metadata = distorter.metadata()["cameras"]["left_wrist"]

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert not np.array_equal(output, image)
    assert metadata["enabled"] is True
    assert metadata["source_camera_matrix"][0][0] == pytest.approx(40.0, abs=1e-2)
    assert metadata["outside_fraction"] >= 0.0


def test_real_raw_rgb_distorter_supports_optimal_source_k() -> None:
    image = _gradient_image()
    distorter = rgb_distortion.RealRawRgbDistorter(
        camera_intrinsics_by_name={
            "right_wrist": {
                "width": 64,
                "height": 48,
                "fx": 39.0,
                "fy": 39.5,
                "ppx": 31.0,
                "ppy": 23.5,
                "coeffs": [-0.05, 0.04, 0.001, 0.001, 0.0],
            }
        },
        camera_metadata_by_name={"right_wrist": {"fovy_deg": 62.0}},
        source_camera_model="undistort-alpha0",
    )

    output = distorter.distort("right_wrist", image)
    metadata = distorter.metadata()

    assert output.shape == image.shape
    assert metadata["source_camera_model"] == "undistort-alpha0"
    assert metadata["cameras"]["right_wrist"]["enabled"] is True


def test_render_intrinsics_for_rgb_distortion_widens_distorted_camera_focal() -> None:
    raw = {
        "left_wrist": {
            "width": 640,
            "height": 480,
            "fx": 393.8441162109375,
            "fy": 393.5601501464844,
            "ppx": 321.1545104980469,
            "ppy": 233.7112579345703,
            "coeffs": [-0.05128170922398567, 0.05344846844673157, 0.001898177433758974, 9.337193478131667e-05, -0.018449459224939346],
        },
        "front": {
            "width": 640,
            "height": 480,
            "fx": 606.5564575195312,
            "fy": 605.2216796875,
            "ppx": 313.12677001953125,
            "ppy": 241.48489379882812,
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }

    render = rgb_distortion.render_intrinsics_for_rgb_distortion(
        raw,
        source_camera_model="undistort-alpha0",
    )

    assert render["left_wrist"]["fy"] < raw["left_wrist"]["fy"]
    assert render["left_wrist"]["coeffs"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert render["left_wrist"]["raw_real_intrinsics"]["fy"] == raw["left_wrist"]["fy"]
    assert render["front"]["fy"] == raw["front"]["fy"]
