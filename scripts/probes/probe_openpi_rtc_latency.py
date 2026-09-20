#!/usr/bin/env python3
"""Measure conditioned OpenPI RTC inference without owning server or hardware.

The probe deliberately replays one exact msgpack request.  It does not load a
checkpoint, start an OpenPI server, open cameras, initialize robot backends, or
dispatch actions.  Reusing the same request file makes separate 8-step and
10-step server runs comparable without binding the client to either model or
checkpoint identity.

For a diagnostic provider that exposes them, ``policy_timing`` and
``server_timing`` may additionally contain ``python_dispatch_ms``,
``jax_device_ready_ms`` and ``jax_device_ready_wait_ms``.  The latter two are
trusted as device timing only when the provider measured them after an
explicit ``block_until_ready`` and marks
``device_ready_semantics='explicit_block_until_ready'``; the legacy
``infer_ms`` field is reported for compatibility but is never treated as proof
of device readiness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from deployments.client.rtc_transport import RtcTransportDeadlineSpec
from deployments.openpi.client.rtc_conditioned import (
    OPENPI_CONDITIONED_RTC_METADATA_KEY_V1,
    OPENPI_CONDITIONED_RTC_OPERATION_V1,
    OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
)
from deployments.openpi.client.transport import (
    OpenPIWebsocketTransport,
    packb,
    unpackb,
)


_REQUEST_FIELDS = frozenset(
    {
        "__openpi_request__",
        "observation",
        "conditioning_actions",
        "prefix_mask",
        "conditioning_mode",
        "receipt_context",
    }
)
_CONDITIONED_PREFIX_MODE = "conditioned_prefix_v1"
_UNCONDITIONED_BOOTSTRAP_MODE = "unconditioned_bootstrap_v1"
_RAW_STATE_MODE = "raw_state_v1"
_MASKED_STATE_MODE = "masked_normalized_concat_v1"


class _ProbeTransport(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def last_phase_receipt(self) -> Any: ...

    def infer(self, request: dict[str, Any]) -> Mapping[str, Any]: ...


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def load_conditioned_request(path: Path) -> dict[str, Any]:
    """Load and structurally validate one exact conditioned-RTC wire request."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"RTC request snapshot does not exist: {source}")
    decoded = unpackb(source.read_bytes())
    if not isinstance(decoded, dict):
        raise ValueError("RTC request snapshot must decode to a mapping.")
    request = dict(decoded)
    missing = sorted(_REQUEST_FIELDS.difference(request))
    extra = sorted(set(request).difference(_REQUEST_FIELDS))
    if missing or extra:
        raise ValueError(
            "RTC request fields differ from the conditioned wire contract: "
            f"missing={missing}, extra={extra}."
        )

    marker = request["__openpi_request__"]
    if not isinstance(marker, Mapping) or dict(marker) != {
        "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
        "operation": OPENPI_CONDITIONED_RTC_OPERATION_V1,
    }:
        raise ValueError("RTC request marker does not select conditioned inference.")
    if not isinstance(request["observation"], Mapping) or not request["observation"]:
        raise ValueError("RTC request observation must be a non-empty mapping.")
    if not isinstance(request["receipt_context"], Mapping):
        raise ValueError("RTC request receipt_context must be a mapping.")

    actions = np.asarray(request["conditioning_actions"], dtype=np.float32)
    mask = np.asarray(request["prefix_mask"], dtype=np.float32)
    if actions.ndim != 2 or not actions.shape[0] or not actions.shape[1]:
        raise ValueError(
            "conditioning_actions must have non-empty shape [horizon, action_dim]."
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("conditioning_actions must contain only finite values.")
    if mask.shape != (actions.shape[0],):
        raise ValueError(
            "prefix_mask must have shape [horizon] matching conditioning_actions."
        )
    if not np.all(np.logical_or(mask == 0.0, mask == 1.0)):
        raise ValueError("prefix_mask values must be exactly 0 or 1.")
    expected_mode = (
        _CONDITIONED_PREFIX_MODE
        if bool(np.any(mask))
        else _UNCONDITIONED_BOOTSTRAP_MODE
    )
    if request["conditioning_mode"] != expected_mode:
        raise ValueError(
            "conditioning_mode disagrees with prefix_mask: "
            f"expected {expected_mode!r}."
        )

    # Normalize only the two numeric wire arrays.  No inferred state, camera,
    # prompt, model identity, or checkpoint binding is introduced here.
    request["conditioning_actions"] = actions
    request["prefix_mask"] = mask
    request["observation"] = dict(request["observation"])
    request["receipt_context"] = dict(request["receipt_context"])
    return request


def validate_deployment_state_contract(
    request: Mapping[str, Any],
    *,
    state_dim: int,
    state_input_mode: str,
) -> None:
    """Validate the state shape required by a concrete deployment artifact.

    The latency probe remains model-agnostic by default. When the operator
    supplies the loaded artifact's state contract, this check catches a
    malformed synthetic request before it reaches the server. In particular,
    masked deployment state is ``[state, all_available_mask]``; training-time
    dropout masks must never be reused for deployment.
    """

    if state_dim <= 0:
        raise ValueError("state_dim must be positive.")
    if state_input_mode not in {_RAW_STATE_MODE, _MASKED_STATE_MODE}:
        raise ValueError(
            "state_input_mode must be 'raw_state_v1' or "
            "'masked_normalized_concat_v1'."
        )
    observation = request.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("RTC request observation must be a mapping.")
    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("RTC deployment state must be a finite rank-1 array.")
    expected_dim = state_dim * (2 if state_input_mode == _MASKED_STATE_MODE else 1)
    if state.shape != (expected_dim,):
        raise ValueError(
            "RTC deployment state shape disagrees with the supplied artifact "
            f"contract: expected {(expected_dim,)}, got {state.shape}."
        )
    if state_input_mode == _MASKED_STATE_MODE:
        availability = state[state_dim:]
        if not np.all(availability == 1.0):
            raise ValueError(
                "RTC deployment state availability mask must be all ones; "
                "training/dropout masks cannot be used for deployment."
            )


def _provider_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    provider = metadata.get(OPENPI_CONDITIONED_RTC_METADATA_KEY_V1)
    if not isinstance(provider, Mapping):
        raise ValueError(
            "OpenPI server metadata does not advertise conditioned RTC support."
        )
    if provider.get("enabled") is not True:
        raise ValueError("OpenPI conditioned RTC provider is not enabled.")
    if provider.get("wire_protocol") != OPENPI_CONDITIONED_RTC_WIRE_V1:
        raise ValueError("OpenPI conditioned RTC wire protocol is incompatible.")
    fields = (
        "wire_protocol",
        "provider_capability_schema",
        "conditioning_transform_owner",
        "external_action_dim",
        "internal_sampler_action_dim",
        "prediction_horizon_steps",
        "sampler_constraint",
        "supports_unconditioned_bootstrap",
        "continuation_implementation_id",
    )
    return {name: provider[name] for name in fields if name in provider}


def _action_array(
    response: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int],
) -> tuple[str, np.ndarray]:
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in response.items():
        if key in (
            OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1,
            "policy_timing",
            "server_timing",
        ):
            continue
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if array.shape == expected_shape and np.all(np.isfinite(array)):
            candidates.append((str(key), array))
    if len(candidates) != 1:
        raise ValueError(
            "Conditioned response must contain exactly one finite action array "
            f"with shape {expected_shape}; candidates={[key for key, _ in candidates]}."
        )
    return candidates[0]


def validate_conditioned_response(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
) -> str:
    """Validate observable prefix consumption without adding identity gates."""

    if not isinstance(response, Mapping):
        raise ValueError("OpenPI conditioned response must be a mapping.")
    actions = np.asarray(request["conditioning_actions"], dtype=np.float32)
    mask = np.asarray(request["prefix_mask"], dtype=np.float32)
    selected = np.flatnonzero(mask).astype(int).tolist()
    receipt = response.get(OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1)
    if not isinstance(receipt, Mapping):
        raise ValueError("OpenPI conditioned response has no consumption receipt.")
    if receipt.get("wire_protocol") != OPENPI_CONDITIONED_RTC_WIRE_V1:
        raise ValueError("OpenPI conditioned response wire protocol mismatch.")
    if receipt.get("conditioning_mode") != request["conditioning_mode"]:
        raise ValueError("OpenPI conditioned response mode mismatch.")
    if receipt.get("receipt_context") != request["receipt_context"]:
        raise ValueError("OpenPI conditioned response receipt context mismatch.")
    if list(receipt.get("consumed_external_shape", ())) != list(actions.shape):
        raise ValueError("OpenPI consumed conditioning shape mismatch.")
    if list(receipt.get("consumed_generation_indices", ())) != selected:
        raise ValueError("OpenPI consumed different conditioning indices.")

    action_key, generated = _action_array(response, expected_shape=actions.shape)
    if selected and not np.array_equal(generated[selected], actions[selected]):
        raise ValueError(
            "OpenPI conditioned output did not preserve the selected prefix rows."
        )
    return action_key


def _optional_timing_ms(response: Mapping[str, Any], key: str) -> float | None:
    timing = response.get(key)
    if not isinstance(timing, Mapping):
        return None
    value = timing.get("infer_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _optional_explicit_timing_ms(
    response: Mapping[str, Any],
    container_key: str,
    field: str,
) -> float | None:
    """Read one explicitly named provider timing field.

    ``infer_ms`` predates this probe's timing truth work and has no stable
    promise about whether the provider waited for a JAX device result.  The
    fields read here deliberately use explicit names so a diagnostic server
    can distinguish Python dispatch from a measurement taken after
    ``block_until_ready``.  Missing fields stay missing; the probe never
    infers device timing from a wall clock or from ``infer_ms``.
    """

    timing = response.get(container_key)
    if not isinstance(timing, Mapping):
        return None
    value = timing.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _optional_device_ready_timing_ms(
    response: Mapping[str, Any],
    container_key: str,
    field: str,
) -> float | None:
    """Read device timing only when the provider declares synchronization."""

    timing = response.get(container_key)
    if not isinstance(timing, Mapping):
        return None
    if timing.get("device_ready_semantics") != "explicit_block_until_ready":
        return None
    return _optional_explicit_timing_ms(response, container_key, field)


def measure_dispatch_and_device_ready(
    operation: Callable[[], Any],
    *,
    block_until_ready: Callable[[Any], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure Python dispatch separately from optional device readiness.

    This helper is intentionally only for benchmark/diagnostic code.  A
    JAX/Flax operation may return before device execution completes, so the
    interval ending at ``operation()`` is *not* a device-compute measurement.
    When ``block_until_ready`` is supplied (for example ``jax.block_until_ready``)
    it is called on the returned value and the second interval is reported.
    If no blocker is supplied, an object's bound ``block_until_ready`` method
    is used when available.  Otherwise device readiness is explicitly marked
    unavailable rather than guessed.
    """

    started = float(clock())
    result = operation()
    dispatched = float(clock())
    if (
        not math.isfinite(started)
        or not math.isfinite(dispatched)
        or dispatched < started
    ):
        raise ValueError("Benchmark clock regressed or returned a non-finite value.")

    blocker = block_until_ready
    if blocker is None:
        bound_blocker = getattr(result, "block_until_ready", None)
        if callable(bound_blocker):
            blocker = lambda value: value.block_until_ready()

    dispatch_ms = (dispatched - started) * 1000.0
    if blocker is None:
        return {
            "python_dispatch_ms": dispatch_ms,
            "jax_device_ready_wait_ms": None,
            "jax_device_ready_ms": None,
            "device_ready_observed": False,
            "device_ready_semantics": "not_observed",
        }

    blocker(result)
    ready = float(clock())
    if not math.isfinite(ready) or ready < dispatched:
        raise ValueError("Benchmark clock regressed after device-ready wait.")
    return {
        "python_dispatch_ms": dispatch_ms,
        "jax_device_ready_wait_ms": (ready - dispatched) * 1000.0,
        "jax_device_ready_ms": (ready - started) * 1000.0,
        "device_ready_observed": True,
        "device_ready_semantics": "explicit_block_until_ready",
    }


def _statistics(values: Sequence[float]) -> dict[str, float]:
    samples = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError("Latency samples must be a non-empty finite sequence.")
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "p95": float(np.percentile(samples, 95, method="linear")),
        "p99": float(np.percentile(samples, 99, method="linear")),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def run_probe(
    *,
    transport: _ProbeTransport,
    request: dict[str, Any],
    warmup_requests: int,
    measured_requests: int,
    latency_budget_ms: float,
    label: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one model-only benchmark against an already-started server."""

    if warmup_requests < 0 or measured_requests <= 0:
        raise ValueError("warmup_requests must be non-negative and measured_requests positive.")
    if not math.isfinite(latency_budget_ms) or latency_budget_ms <= 0.0:
        raise ValueError("latency_budget_ms must be finite and positive.")

    metadata = dict(transport.metadata)
    provider = _provider_summary(metadata)
    baseline_wire = packb(request)
    request_actions = np.asarray(request["conditioning_actions"], dtype=np.float32)
    provider_horizon = provider.get("prediction_horizon_steps")
    provider_action_dim = provider.get("external_action_dim")
    if provider_horizon != request_actions.shape[0]:
        raise ValueError(
            "Conditioned RTC request horizon differs from the server capability: "
            f"{request_actions.shape[0]} != {provider_horizon}."
        )
    if provider_action_dim != request_actions.shape[1]:
        raise ValueError(
            "Conditioned RTC request action dimension differs from the server "
            f"capability: {request_actions.shape[1]} != {provider_action_dim}."
        )
    action_key: str | None = None
    for _ in range(warmup_requests):
        response = transport.infer(request)
        action_key = validate_conditioned_response(response, request)

    wall_ms: list[float] = []
    transport_ms: list[float] = []
    policy_ms: list[float] = []
    server_ms: list[float] = []
    non_policy_wall_ms: list[float] = []
    policy_python_dispatch_ms: list[float] = []
    policy_jax_device_ready_ms: list[float] = []
    policy_jax_device_ready_wait_ms: list[float] = []
    server_python_dispatch_ms: list[float] = []
    server_jax_device_ready_ms: list[float] = []
    server_jax_device_ready_wait_ms: list[float] = []
    for _ in range(measured_requests):
        started = float(clock())
        response = transport.infer(request)
        completed = float(clock())
        if not math.isfinite(started) or not math.isfinite(completed) or completed < started:
            raise ValueError("Benchmark clock regressed or returned a non-finite value.")
        elapsed_ms = (completed - started) * 1000.0
        wall_ms.append(elapsed_ms)
        action_key = validate_conditioned_response(response, request)

        phase = transport.last_phase_receipt
        if phase is not None and getattr(phase, "phase", None) == "inference":
            duration = float(getattr(phase, "duration_sec")) * 1000.0
            if math.isfinite(duration) and duration >= 0.0:
                transport_ms.append(duration)
        policy_value = _optional_timing_ms(response, "policy_timing")
        if policy_value is not None:
            policy_ms.append(policy_value)
            non_policy_wall_ms.append(elapsed_ms - policy_value)
        server_value = _optional_timing_ms(response, "server_timing")
        if server_value is not None:
            server_ms.append(server_value)
        for values, container_key, field, reader in (
            (
                policy_python_dispatch_ms,
                "policy_timing",
                "python_dispatch_ms",
                _optional_explicit_timing_ms,
            ),
            (
                policy_jax_device_ready_ms,
                "policy_timing",
                "jax_device_ready_ms",
                _optional_device_ready_timing_ms,
            ),
            (
                policy_jax_device_ready_wait_ms,
                "policy_timing",
                "jax_device_ready_wait_ms",
                _optional_device_ready_timing_ms,
            ),
            (
                server_python_dispatch_ms,
                "server_timing",
                "python_dispatch_ms",
                _optional_explicit_timing_ms,
            ),
            (
                server_jax_device_ready_ms,
                "server_timing",
                "jax_device_ready_ms",
                _optional_device_ready_timing_ms,
            ),
            (
                server_jax_device_ready_wait_ms,
                "server_timing",
                "jax_device_ready_wait_ms",
                _optional_device_ready_timing_ms,
            ),
        ):
            timing_value = reader(
                response,
                container_key,
                field,
            )
            if timing_value is not None:
                values.append(timing_value)

    if packb(request) != baseline_wire:
        raise RuntimeError("Benchmark request was mutated during replay.")

    actions = request_actions
    mask = np.asarray(request["prefix_mask"], dtype=np.float32)
    within_budget = sum(value <= latency_budget_ms for value in wall_ms)
    result: dict[str, Any] = {
        "schema": "openpi.conditioned_rtc_latency_probe.v1",
        "label": label,
        "server": {
            "protocol_version": metadata.get("protocol_version"),
            "model_id": metadata.get("model_id"),
            "deployment": metadata.get("deployment"),
            "model": metadata.get("model"),
            "conditioned_rtc": provider,
        },
        "request": {
            "packed_bytes": len(baseline_wire),
            "prediction_horizon_steps": int(actions.shape[0]),
            "external_action_dim": int(actions.shape[1]),
            "conditioning_mode": request["conditioning_mode"],
            "selected_generation_indices": np.flatnonzero(mask).astype(int).tolist(),
            "action_key": action_key,
        },
        "sampling": {
            "warmup_requests": int(warmup_requests),
            "measured_requests": int(measured_requests),
        },
        "latency_ms": {
            "wall": _statistics(wall_ms),
            "wall_samples": wall_ms,
            "transport_exchange": (
                None if len(transport_ms) != measured_requests else _statistics(transport_ms)
            ),
            "policy_infer": (
                None if len(policy_ms) != measured_requests else _statistics(policy_ms)
            ),
            "server_infer": (
                None if len(server_ms) != measured_requests else _statistics(server_ms)
            ),
            "non_policy_wall": (
                None
                if len(non_policy_wall_ms) != measured_requests
                else _statistics(non_policy_wall_ms)
            ),
            "policy_python_dispatch": (
                None
                if len(policy_python_dispatch_ms) != measured_requests
                else _statistics(policy_python_dispatch_ms)
            ),
            "policy_jax_device_ready": (
                None
                if len(policy_jax_device_ready_ms) != measured_requests
                else _statistics(policy_jax_device_ready_ms)
            ),
            "policy_jax_device_ready_wait": (
                None
                if len(policy_jax_device_ready_wait_ms) != measured_requests
                else _statistics(policy_jax_device_ready_wait_ms)
            ),
            "server_python_dispatch": (
                None
                if len(server_python_dispatch_ms) != measured_requests
                else _statistics(server_python_dispatch_ms)
            ),
            "server_jax_device_ready": (
                None
                if len(server_jax_device_ready_ms) != measured_requests
                else _statistics(server_jax_device_ready_ms)
            ),
            "server_jax_device_ready_wait": (
                None
                if len(server_jax_device_ready_wait_ms) != measured_requests
                else _statistics(server_jax_device_ready_wait_ms)
            ),
        },
        "timing_truth": {
            "policy_infer_ms": (
                "legacy provider infer_ms; it does not establish JAX device "
                "readiness unless the provider documents that semantic."
            ),
            "python_dispatch_ms": (
                "explicit provider field: Python/model call returned; this is "
                "not a synchronized device-compute measurement."
            ),
            "jax_device_ready_ms": (
                "explicit provider field: provider must have waited for "
                "block_until_ready before reporting it."
            ),
            "missing_fields": {
                "policy_python_dispatch": len(policy_python_dispatch_ms)
                != measured_requests,
                "policy_jax_device_ready": len(policy_jax_device_ready_ms)
                != measured_requests,
                "server_python_dispatch": len(server_python_dispatch_ms)
                != measured_requests,
                "server_jax_device_ready": len(server_jax_device_ready_ms)
                != measured_requests,
            },
        },
        "budget_observation": {
            "budget_ms": float(latency_budget_ms),
            "within_budget_count": int(within_budget),
            "measured_count": int(measured_requests),
            "within_budget_ratio": float(within_budget / measured_requests),
            "note": "informational only; not a deployment admission gate",
        },
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one exact conditioned-RTC request against an already-running "
            "OpenPI server and report model-only latency."
        )
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--warmup-requests", type=_non_negative_int, default=2)
    parser.add_argument("--measured-requests", type=_positive_int, default=20)
    parser.add_argument("--latency-budget-ms", type=_positive_float, default=400.0)
    parser.add_argument("--connect-timeout-sec", type=_positive_float, default=10.0)
    parser.add_argument("--metadata-timeout-sec", type=_positive_float, default=10.0)
    parser.add_argument("--inference-timeout-sec", type=_positive_float, default=30.0)
    parser.add_argument("--close-timeout-sec", type=_positive_float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--state-dim",
        type=_positive_int,
        default=None,
        help="Base deployment state dimension for optional artifact-contract validation.",
    )
    parser.add_argument(
        "--state-input-mode",
        choices=(_RAW_STATE_MODE, _MASKED_STATE_MODE),
        default=None,
        help="State input mode for optional artifact-contract validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request_path = args.request.expanduser().resolve()
    request = load_conditioned_request(request_path)
    if (args.state_dim is None) != (args.state_input_mode is None):
        raise ValueError("--state-dim and --state-input-mode must be provided together.")
    if args.state_dim is not None and args.state_input_mode is not None:
        validate_deployment_state_contract(
            request,
            state_dim=args.state_dim,
            state_input_mode=args.state_input_mode,
        )
    transport = OpenPIWebsocketTransport(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        deadline_spec=RtcTransportDeadlineSpec(
            connect_timeout_sec=args.connect_timeout_sec,
            metadata_timeout_sec=args.metadata_timeout_sec,
            inference_timeout_sec=args.inference_timeout_sec,
            close_timeout_sec=args.close_timeout_sec,
        ),
    )
    try:
        result = run_probe(
            transport=transport,
            request=request,
            warmup_requests=args.warmup_requests,
            measured_requests=args.measured_requests,
            latency_budget_ms=args.latency_budget_ms,
            label=args.label,
        )
    finally:
        transport.close()
    result["endpoint"] = {"host": str(args.host), "port": int(args.port)}
    result["request"]["source"] = str(request_path)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
