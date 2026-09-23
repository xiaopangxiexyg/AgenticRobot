from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/probes/openpi_synthetic_inference_load.py"
)
_SPEC = importlib.util.spec_from_file_location("openpi_synthetic_inference_load", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _artifact() -> dict:
    return {
        "inputs": {"state": {"dtype": "float32", "dim": 32}},
        "camera_input": {
            "dtype": "uint8",
            "layout": "HWC",
            "inputs": [
                {"model_input_key": "observation.images.front"},
                {"model_input_key": "observation.images.right_wrist"},
            ],
        },
    }


def test_synthetic_observation_matches_checkpoint_inputs() -> None:
    observation = _MODULE.build_observation(_artifact(), prompt="stage one")
    assert observation["prompt"] == "stage one"
    assert observation["observation.state"].shape == (32,)
    assert observation["observation.state"].dtype == np.float32
    for key in ("observation.images.front", "observation.images.right_wrist"):
        assert observation[key].shape == (224, 224, 3)
        assert observation[key].dtype == np.uint8


def test_masked_state_declares_all_components_available() -> None:
    artifact = _artifact()
    artifact["state_input_mode"] = "masked_normalized_concat_v1"
    state = _MODULE.build_observation(artifact, prompt="stage one")["observation.state"]
    np.testing.assert_array_equal(state[:16], np.zeros((16,), dtype=np.float32))
    np.testing.assert_array_equal(state[16:], np.ones((16,), dtype=np.float32))


@pytest.mark.parametrize(
    "key,value",
    [
        ("inputs", {"state": {"dtype": "float32", "dim": 0}}),
        ("camera_input", {"dtype": "float32", "layout": "HWC", "inputs": [{"model_input_key": "observation.images.front"}]}),
        ("camera_input", {"dtype": "uint8", "layout": "HWC", "inputs": [{"model_input_key": "bad"}]}),
    ],
)
def test_synthetic_observation_rejects_bad_contract(key: str, value: dict) -> None:
    artifact = _artifact()
    artifact[key] = value
    with pytest.raises(ValueError):
        _MODULE.build_observation(artifact, prompt="stage one")
