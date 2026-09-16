from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deployments.openpi.client.rtc_conditioned import (
    OPENPI_CONDITIONED_RTC_METADATA_KEY_V1,
    OPENPI_CONDITIONED_RTC_OPERATION_V1,
    OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
)
from deployments.openpi.client.transport import packb
from scripts.probes.probe_openpi_rtc_latency import (
    load_conditioned_request,
    run_probe,
    validate_conditioned_response,
)


def _request() -> dict[str, Any]:
    actions = np.arange(24, dtype=np.float32).reshape(4, 6)
    return {
        "__openpi_request__": {
            "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
            "operation": OPENPI_CONDITIONED_RTC_OPERATION_V1,
        },
        "observation": {
            "state": np.zeros((8,), dtype=np.float32),
            "image": np.zeros((4, 5, 3), dtype=np.uint8),
            "prompt": "write",
        },
        "conditioning_actions": actions,
        "prefix_mask": np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "conditioning_mode": "conditioned_prefix_v1",
        "receipt_context": {"request_id": "stable-input"},
    }


def _response(request: dict[str, Any], *, policy_ms: float) -> dict[str, Any]:
    actions = np.zeros_like(request["conditioning_actions"])
    actions[:2] = request["conditioning_actions"][:2]
    return {
        "actions": actions,
        OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1: {
            "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
            "conditioning_mode": request["conditioning_mode"],
            "receipt_context": dict(request["receipt_context"]),
            "consumed_external_shape": list(request["conditioning_actions"].shape),
            "consumed_generation_indices": [0, 1],
        },
        "policy_timing": {"infer_ms": policy_ms},
        "server_timing": {"infer_ms": policy_ms + 1.0},
    }


@dataclass
class _Phase:
    duration_sec: float
    phase: str = "inference"


class _FakeTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.last_phase_receipt = None
        self._policy_ms = iter((90.0, 20.0, 30.0))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": "openpi_local_v1",
            "model_id": "informational-model",
            "deployment": {"checkpoint_step": 100000},
            "model": {"action_horizon": 4},
            OPENPI_CONDITIONED_RTC_METADATA_KEY_V1: {
                "enabled": True,
                "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
                "prediction_horizon_steps": 4,
                "external_action_dim": 6,
                "internal_sampler_action_dim": 32,
                "sampler_constraint": "clamped_known_prefix_v1",
            },
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        policy_ms = next(self._policy_ms)
        self.last_phase_receipt = _Phase(duration_sec=(policy_ms + 2.0) / 1000.0)
        return _response(request, policy_ms=policy_ms)


class _Clock:
    def __init__(self) -> None:
        self._values = iter((1.0, 1.025, 2.0, 2.035))

    def __call__(self) -> float:
        return next(self._values)


def test_load_conditioned_request_preserves_exact_arrays(tmp_path: Path) -> None:
    source = tmp_path / "request.msgpack"
    source.write_bytes(packb(_request()))

    loaded = load_conditioned_request(source)

    np.testing.assert_array_equal(
        loaded["conditioning_actions"], _request()["conditioning_actions"]
    )
    np.testing.assert_array_equal(loaded["prefix_mask"], [1.0, 1.0, 0.0, 0.0])
    assert loaded["observation"]["prompt"] == "write"


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(unexpected=True), "fields differ"),
        (
            lambda value: value.update(prefix_mask=np.asarray([0.5, 1.0, 0.0, 0.0])),
            "exactly 0 or 1",
        ),
        (
            lambda value: value.update(conditioning_mode="unconditioned_bootstrap_v1"),
            "disagrees with prefix_mask",
        ),
    ),
)
def test_load_conditioned_request_rejects_ambiguous_wire_input(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    request = _request()
    mutate(request)
    source = tmp_path / "request.msgpack"
    source.write_bytes(packb(request))

    with pytest.raises(ValueError, match=message):
        load_conditioned_request(source)


def test_run_probe_excludes_warmup_and_reports_observed_budget() -> None:
    request = _request()
    transport = _FakeTransport()

    result = run_probe(
        transport=transport,
        request=request,
        warmup_requests=1,
        measured_requests=2,
        latency_budget_ms=30.0,
        label="same-input-8-step",
        clock=_Clock(),
    )

    assert len(transport.requests) == 3
    assert all(value is request for value in transport.requests)
    assert result["latency_ms"]["wall_samples"] == pytest.approx([25.0, 35.0])
    assert result["latency_ms"]["wall"]["mean"] == pytest.approx(30.0)
    assert result["latency_ms"]["policy_infer"]["mean"] == pytest.approx(25.0)
    assert result["latency_ms"]["non_policy_wall"]["mean"] == pytest.approx(5.0)
    assert result["budget_observation"] == {
        "budget_ms": 30.0,
        "within_budget_count": 1,
        "measured_count": 2,
        "within_budget_ratio": 0.5,
        "note": "informational only; not a deployment admission gate",
    }
    assert result["request"]["selected_generation_indices"] == [0, 1]
    assert result["request"]["action_key"] == "actions"


def test_response_rejects_prefix_rows_that_provider_did_not_preserve() -> None:
    request = _request()
    response = _response(request, policy_ms=10.0)
    response["actions"][0, 0] += 1.0

    with pytest.raises(ValueError, match="preserve the selected prefix"):
        validate_conditioned_response(response, request)


def test_run_probe_rejects_request_shape_incompatible_with_server() -> None:
    request = _request()
    request["conditioning_actions"] = np.zeros((5, 6), dtype=np.float32)
    request["prefix_mask"] = np.zeros((5,), dtype=np.float32)
    request["conditioning_mode"] = "unconditioned_bootstrap_v1"

    with pytest.raises(ValueError, match="request horizon differs"):
        run_probe(
            transport=_FakeTransport(),
            request=request,
            warmup_requests=0,
            measured_requests=1,
            latency_budget_ms=400.0,
        )
