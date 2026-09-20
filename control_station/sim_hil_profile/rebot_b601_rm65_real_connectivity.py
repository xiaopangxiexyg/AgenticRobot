"""Read-only B601/RealMan connectivity and frame-audit command.

This module is the first real-hardware slice for the B601 -> RealMan target.
It deliberately stops at observation: it opens the existing B601 CAN backend
and the existing observation-only RM65 backend, reads both state owners, and
reports the configured frame conversion and timing.  It does not enable,
disable, stop, or send a motion command to either device.

The later motion command must reuse the existing B601 residual/session and
``RM65ArmBackend`` dispatch path.  This probe exists so that connection,
freshness, and coordinate conventions can be checked before that path is
allowed to write a target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

READ_ONLY_CONFIRMATION = "B601_RM65_REAL_READ_ONLY_CONNECTIVITY"
_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and not bool")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _matrix4(value: object, *, name: str) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(_finite(item, name=name) for item in row) for row in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 matrix") from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"{name} must be a numeric 4x4 matrix")
    return rows


def _matmul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            sum(float(left[row][inner]) * float(right[inner][column]) for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )


def _invert_rigid(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Invert a validated homogeneous rigid transform without SciPy."""

    transform = _matrix4(matrix, name="rigid transform")
    rotation = tuple(tuple(transform[row][column] for column in range(3)) for row in range(3))
    translation = tuple(transform[row][3] for row in range(3))
    inverse_rotation = tuple(
        tuple(rotation[column][row] for column in range(3)) for row in range(3)
    )
    inverse_translation = tuple(
        -sum(inverse_rotation[row][column] * translation[column] for column in range(3))
        for row in range(3)
    )
    return tuple(
        tuple(
            inverse_rotation[row][column] if column < 3 else inverse_translation[row]
            for column in range(4)
        )
        for row in range(3)
    ) + (_IDENTITY[3],)


def map_source_pose_to_target(
    source_pose: Sequence[Sequence[float]],
    *,
    matrix_target_reference_source_reference: Sequence[Sequence[float]],
    matrix_source_controlled_target_controlled: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Map ``source_reference -> source_controlled`` into target frames.

    The profile stores ``T_target_reference_source_reference`` and
    ``T_source_controlled_target_controlled``.  Therefore the target pose is

    ``T_target_reference_target_controlled =
    T_target_reference_source_reference @ T_source_reference_source_controlled
    @ inv(T_source_controlled_target_controlled)``.
    """

    return _matmul(
        _matmul(
            _matrix4(matrix_target_reference_source_reference, name="reference mapping"),
            _matrix4(source_pose, name="source pose"),
        ),
        _invert_rigid(
            _matrix4(matrix_source_controlled_target_controlled, name="controlled mapping")
        ),
    )


def _matrix_to_json(value: object) -> list[list[float]]:
    return [list(row) for row in _matrix4(value, name="matrix")]


def _vector_to_json(value: object, *, length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [_finite(item, name="vector item") for item in value]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return values if len(values) == length else None


def _json_safe(value: object) -> object:
    """Convert numpy/scalar-like values in backend state to JSON values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return str(value)


def _pose_error(
    expected: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
) -> dict[str, float | None]:
    expected_matrix = _matrix4(expected, name="expected target pose")
    observed_matrix = _matrix4(observed, name="observed target pose")
    position_error = math.sqrt(
        sum((expected_matrix[row][3] - observed_matrix[row][3]) ** 2 for row in range(3))
    )
    relative = _matmul(_invert_rigid(observed_matrix), expected_matrix)
    trace = max(-1.0, min(3.0, sum(relative[index][index] for index in range(3))))
    rotation_angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    return {
        "translation_error_m": position_error,
        "rotation_error_rad": rotation_angle,
    }


def _b601_state_json(state: object) -> dict[str, object] | None:
    """Expose the safety-relevant B601 freshness fields without vendor objects."""

    if state is None:
        return None
    joints = []
    for joint in getattr(state, "joints", ()):
        joints.append(
            {
                "name": getattr(joint, "name", None),
                "motor_id": getattr(joint, "motor_id", None),
                "feedback_id": getattr(joint, "feedback_id", None),
                "position_rad": getattr(joint, "position_rad", None),
                "observed_at_sec": getattr(joint, "observed_at_sec", None),
                "status_sequence": getattr(joint, "status_sequence", None),
                "receive_age_sec": getattr(joint, "receive_age_sec", None),
                "valid": getattr(joint, "valid", None),
                "stale": getattr(joint, "stale", None),
                "host_receive_freshness_verified": getattr(
                    joint, "host_receive_freshness_verified", None
                ),
                "error": getattr(joint, "error", None),
            }
        )
    return {
        "sequence": getattr(state, "sequence", None),
        "receive_timestamp_sec": getattr(state, "receive_timestamp_sec", None),
        "valid": getattr(state, "valid", None),
        "stale": getattr(state, "stale", None),
        "freshness_verified": getattr(state, "freshness_verified", None),
        "freshness_semantics": getattr(state, "freshness_semantics", None),
        "fault_reason": getattr(state, "fault_reason", None),
        "host_receive_freshness_verified": getattr(
            state, "host_receive_freshness_verified", None
        ),
        "joints": joints,
    }


def _sample_acceptance_counts(samples: Sequence[object]) -> dict[str, int]:
    """Count the independent and paired observations used by the PASS gate."""

    b601_accept_count = 0
    rm65_success_count = 0
    valid_pair_count = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        b601 = sample.get("b601")
        rm65 = sample.get("rm65")
        b601_accepted = isinstance(b601, Mapping) and b601.get("accepted") is True
        rm65_successful = (
            isinstance(rm65, Mapping) and rm65.get("hardware_read_success") is True
        )
        b601_accept_count += int(b601_accepted)
        rm65_success_count += int(rm65_successful)
        valid_pair_count += int(
            b601_accepted
            and rm65_successful
            and "mapped_target_pose" in sample
            and "mapped_vs_observed_target" in sample
        )
    return {
        "b601_accepted_sample_count": b601_accept_count,
        "rm65_successful_sample_count": rm65_success_count,
        "valid_pose_pair_count": valid_pair_count,
    }


def _build_b601_source(config_path: Path, source_root: Path):
    from control_station.sources.rebot_b601_pose import (
        B601ReadOnlyPoseSource,
        B601ReadOnlyPoseSourceSpec,
    )
    from hardware.rebot_b601_rs import B601RSBackend, B601RSBackendConfig
    from robot_motion.rebot_b601_rs import B601PinocchioModel

    config = B601RSBackendConfig.from_rs_yaml(config_path)
    backend = B601RSBackend(config)
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    source = B601ReadOnlyPoseSource(
        backend,
        model,
        spec=B601ReadOnlyPoseSourceSpec(clock_domain_id="local_wall_time"),
    )
    return backend, source


def _build_rm65_backend(arm_ip: str, arm_urdf_path: Path):
    from control_station.sim_hil_profile.rm65_b601_free_drive_application import (
        build_observation_only_rm65_backend,
    )

    return build_observation_only_rm65_backend(
        arm_ip=arm_ip,
        arm_urdf_path=arm_urdf_path,
        connect_hardware=True,
    )


def _read_sample(
    *,
    b601_source,
    rm65_backend,
    mapping,
) -> dict[str, object]:
    sample_started = time.perf_counter()
    b601_started = time.perf_counter()
    b601_step = b601_source.step()
    b601_elapsed = time.perf_counter() - b601_started
    rm65_started = time.perf_counter()
    rm65_state = rm65_backend.refresh_state_from_hardware(
        include_derived_pose=True,
        include_timing_percentiles=False,
    )
    rm65_elapsed = time.perf_counter() - rm65_started
    result: dict[str, object] = {
        "sample_elapsed_sec": max(0.0, time.perf_counter() - sample_started),
        "b601_poll_elapsed_sec": b601_elapsed,
        "rm65_refresh_elapsed_sec": rm65_elapsed,
        "b601": {
            "state": _b601_state_json(getattr(b601_step, "state", None)),
            "accepted": bool(getattr(b601_step, "emission", None) is not None),
            "rejection_reason": getattr(b601_step, "rejection_reason", None),
            "feedback_poll_elapsed_sec": getattr(b601_step, "feedback_poll_elapsed_sec", None),
            "fk_elapsed_sec": getattr(b601_step, "fk_elapsed_sec", None),
        },
        "rm65": {
            "hardware_qpos": _vector_to_json(rm65_state.get("hardware_qpos"), length=6),
            "current_tcp_pose": _json_safe(rm65_state.get("current_tcp_pose")),
            "hardware_measurement_timestamp": _json_safe(
                rm65_state.get("hardware_measurement_timestamp")
            ),
            "hardware_read_timestamp_monotonic": _json_safe(
                rm65_state.get("hardware_read_timestamp_monotonic")
            ),
            "hardware_read_sequence": _json_safe(rm65_state.get("hardware_read_sequence")),
            "hardware_read_success": bool(rm65_state.get("hardware_read_success")),
            "hardware_read_phase_timing": _json_safe(
                rm65_state.get("hardware_read_phase_timing")
            ),
            "hardware_result_code": _json_safe(rm65_state.get("hardware_result_code")),
            "hardware_result_text": _json_safe(rm65_state.get("hardware_result_text")),
        },
    }
    emission = getattr(b601_step, "emission", None)
    if emission is not None and len(emission.merged_commands) == 1:
        source_pose = _matrix4(emission.merged_commands[0].value, name="B601 FK pose")
        result["b601"]["pose_source_reference_to_controlled"] = _matrix_to_json(source_pose)  # type: ignore[index]
        mapped_pose = map_source_pose_to_target(
            source_pose,
            matrix_target_reference_source_reference=(
                mapping.matrix_target_reference_source_reference
            ),
            matrix_source_controlled_target_controlled=(
                mapping.matrix_source_controlled_target_controlled
            ),
        )
        result["mapped_target_pose"] = _matrix_to_json(mapped_pose)
        observed_pose = rm65_state.get("current_tcp_pose")
        if observed_pose is not None:
            try:
                result["mapped_vs_observed_target"] = _pose_error(mapped_pose, observed_pose)
            except (TypeError, ValueError):
                result["mapped_vs_observed_target"] = None
    return result


def _positive(value: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect-read-only", action="store_true", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--arm-ip", required=True)
    parser.add_argument("--arm-urdf-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebot-source-root", type=Path, required=True)
    parser.add_argument("--mapping-profile", type=Path, required=True)
    parser.add_argument("--duration-sec", type=_positive, default=2.0)
    parser.add_argument("--sample-hz", type=_positive, default=5.0)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.confirmation != READ_ONLY_CONFIRMATION:
        parser.error(f"read-only connectivity requires --confirmation {READ_ONLY_CONFIRMATION}")
    if args.output.exists():
        parser.error(f"output already exists; use a new run directory: {args.output}")
    for path, label in (
        (args.arm_urdf_path, "RM65 URDF"),
        (args.config, "B601 config"),
        (args.rebot_source_root, "reBotArm source root"),
        (args.mapping_profile, "mapping profile"),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "status": "ERROR",
        "execution_status": "NOT_STARTED",
        "read_only": True,
        "motion_commands_sent": False,
        "b601_commands_sent": False,
        "rm65_commands_sent": False,
        "confirmation": args.confirmation,
        "argv": [
            sys.executable,
            "-m",
            "control_station.sim_hil_profile.rebot_b601_rm65_real_connectivity",
            *raw_argv,
        ],
        "samples": [],
    }
    with args.log.open("a", encoding="utf-8") as log:
        def write_result() -> None:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n")

        def record(event: Mapping[str, object]) -> None:
            log.write(json.dumps(_json_safe(dict(event)), ensure_ascii=False) + "\n")
            log.flush()

        record({"event": "command", "argv": result["argv"]})
        b601_backend = None
        rm65_backend = None
        try:
            from control_station.composition.rebot_b601_rm65_profile import (
                load_b601_rm65_frame_mapping_profile,
            )
            from control_station.sim_hil_profile.rebot_b601_rm65_application import (
                _prepare_bundled_motorbridge_runtime,
            )

            result["motorbridge_runtime_selection"] = _prepare_bundled_motorbridge_runtime()
            mapping = load_b601_rm65_frame_mapping_profile(args.mapping_profile)
            result["mapping_profile"] = mapping.to_payload()
            b601_backend, b601_source = _build_b601_source(args.config, args.rebot_source_root)
            rm65_backend = _build_rm65_backend(args.arm_ip, args.arm_urdf_path)
            with ExitStack() as resources:
                resources.callback(lambda: b601_backend.close() if b601_backend else None)
                resources.callback(lambda: rm65_backend.close() if rm65_backend else None)
                b601_backend.connect()
                rm65_backend.start()
                result["b601_connected"] = bool(b601_backend.connected)
                result["rm65_started"] = bool(rm65_backend.is_ready())
                result["b601_runtime_evidence"] = _json_safe(
                    b601_backend.motorbridge_runtime_evidence
                )
                deadline = time.monotonic() + float(args.duration_sec)
                period = 1.0 / float(args.sample_hz)
                next_sample = time.monotonic()
                while True:
                    sample = _read_sample(
                        b601_source=b601_source,
                        rm65_backend=rm65_backend,
                        mapping=mapping,
                    )
                    sample["captured_wall_time_sec"] = time.time()
                    sample["captured_monotonic_sec"] = time.monotonic()
                    result["samples"].append(sample)  # type: ignore[union-attr]
                    record({"event": "sample", **sample})
                    if time.monotonic() >= deadline:
                        break
                    next_sample += period
                    time.sleep(max(0.0, next_sample - time.monotonic()))
                samples = result["samples"]
                assert isinstance(samples, list)
                acceptance_counts = _sample_acceptance_counts(samples)
                result.update(acceptance_counts)
                b601_accept_count = acceptance_counts["b601_accepted_sample_count"]
                rm65_success_count = acceptance_counts["rm65_successful_sample_count"]
                valid_pair_count = acceptance_counts["valid_pose_pair_count"]
                if b601_accept_count == 0:
                    raise RuntimeError(
                        "no complete host-fresh B601 FK sample arrived during the probe"
                    )
                if rm65_success_count == 0:
                    raise RuntimeError(
                        "no successful RealMan observation arrived during the probe"
                    )
                if valid_pair_count == 0:
                    raise RuntimeError(
                        "no sample contained both valid poses for frame audit"
                    )
                result["execution_status"] = "COMPLETED"
                result["status"] = "PASS"
            record({"event": "result", **result, "exit_code": 0})
            write_result()
            return 0
        except KeyboardInterrupt:
            result["status"] = "INTERRUPTED"
            result["execution_status"] = "INTERRUPTED"
            record({"event": "result", **result, "exit_code": 130})
            write_result()
            return 130
        except Exception as exc:  # noqa: BLE001 - CLI evidence boundary
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["execution_status"] = "ERROR"
            record({"event": "error", "error": result["error"], "traceback": traceback.format_exc()})
            record({"event": "result", **result, "exit_code": 1})
            write_result()
            return 1

__all__ = [
    "READ_ONLY_CONFIRMATION",
    "main",
    "map_source_pose_to_target",
]


if __name__ == "__main__":
    raise SystemExit(main())
