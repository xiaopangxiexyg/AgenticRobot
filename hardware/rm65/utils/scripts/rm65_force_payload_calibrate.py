from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent
DATA_DIR = UTILS_DIR / "data"
MEASUREMENTS_DIR = DATA_DIR / "measurements"
RESULTS_DIR = UTILS_DIR / "results"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.standard_paths import force_sensor_calib4_measurements_path  # noqa: E402


FORCE_SOURCES = ("raw", "zero", "work", "tool")
EULER_ORDERS = ("zyx", "xyz", "xzy", "yxz", "yzx", "zxy")
GRAVITY = 9.80665
GRAVITY_BASE = [0.0, 0.0, -1.0]


def _default_input_path() -> Path:
    return MEASUREMENTS_DIR / "rm65_right_legacy_force_measurements.json"


def _format(values: Iterable[float], *, precision: int = 5) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    b_t = list(zip(*b))
    return [[_dot(row, list(col)) for col in b_t] for row in a]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _rot_x(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _rotation_from_pose(pose: list[float], order: str) -> list[list[float]]:
    angles = {"x": pose[3], "y": pose[4], "z": pose[5]}
    rotations = {"x": _rot_x, "y": _rot_y, "z": _rot_z}
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for axis in order:
        matrix = _mat_mul(matrix, rotations[axis](angles[axis]))
    return matrix


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    aug = [list(row) + [float(value)] for row, value in zip(matrix, vector)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise RuntimeError("least-squares system is singular; collect more varied poses")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-18:
                continue
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[-1] for row in aug]


def _least_squares(
    rows: list[list[float]],
    values: list[float],
    *,
    ridge: float = 1e-9,
) -> list[float]:
    if not rows:
        raise RuntimeError("no least-squares rows")
    cols = len(rows[0])
    normal = [[0.0 for _ in range(cols)] for _ in range(cols)]
    rhs = [0.0 for _ in range(cols)]
    for row, value in zip(rows, values):
        for i in range(cols):
            rhs[i] += row[i] * value
            for j in range(cols):
                normal[i][j] += row[i] * row[j]
    for i in range(cols):
        normal[i][i] += ridge
    return _solve_linear_system(normal, rhs)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(len(vectors[0]))]


def _load_records(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("data_type") == "force_measurement_dataset" or "measurements" in payload:
        measurements = payload.get("measurements", [])
        if not isinstance(measurements, list) or not measurements:
            raise RuntimeError(f"{path} has no measurements")
        records = []
        for index, measurement in enumerate(measurements, start=1):
            summary = measurement.get("sample_summary", {})
            if "pose_mean" not in summary or "force" not in summary:
                raise RuntimeError("measurements must contain sample_summary.pose_mean and force")
            records.append(
                {
                    "index": int(measurement.get("pose_index", index)),
                    "pose_mean": summary["pose_mean"],
                    "force": summary["force"],
                }
            )
        return payload, records

    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{path} has no records")
    normalized = []
    for record in records:
        if "pose_mean" not in record or "force" not in record:
            raise RuntimeError("records must contain pose_mean and force statistics")
        normalized.append(
            {
                "index": int(record["index"]),
                "pose_mean": record["pose_mean"],
                "force": record["force"],
            }
        )
    return payload, normalized


def _record_wrench(record: dict[str, object], source: str) -> list[float]:
    return [float(value) for value in record["force"][source]["mean"]]


def _fit_tool_model(records: list[dict[str, object]], source: str, order: str) -> dict[str, object]:
    force_rows = []
    force_values = []
    torque_rows = []
    torque_values = []
    samples = []
    for record in records:
        pose = [float(value) for value in record["pose_mean"]]
        rotation_base_tool = _rotation_from_pose(pose, order)
        rotation_tool_base = _transpose(rotation_base_tool)
        gravity_tool = _mat_vec(rotation_tool_base, GRAVITY_BASE)
        wrench = _record_wrench(record, source)
        force = wrench[:3]
        torque = wrench[3:]

        force_rows.extend(
            [
                [1.0, 0.0, 0.0, gravity_tool[0]],
                [0.0, 1.0, 0.0, gravity_tool[1]],
                [0.0, 0.0, 1.0, gravity_tool[2]],
            ]
        )
        force_values.extend(force)

        vx, vy, vz = gravity_tool
        torque_rows.extend(
            [
                [1.0, 0.0, 0.0, 0.0, vz, -vy],
                [0.0, 1.0, 0.0, -vz, 0.0, vx],
                [0.0, 0.0, 1.0, vy, -vx, 0.0],
            ]
        )
        torque_values.extend(torque)
        samples.append(
            {
                "index": int(record["index"]),
                "pose": pose,
                "position_base": pose[:3],
                "rotation_base_tool": rotation_base_tool,
                "gravity_tool": gravity_tool,
                "force_tool": force,
                "torque_tool": torque,
            }
        )

    force_params = _least_squares(force_rows, force_values)
    torque_params = _least_squares(torque_rows, torque_values)
    force_bias = force_params[:3]
    signed_weight_n = force_params[3]
    torque_bias = torque_params[:3]
    weighted_com = torque_params[3:]
    if abs(signed_weight_n) > 1e-9:
        com_m = [value / signed_weight_n for value in weighted_com]
    else:
        com_m = [math.nan, math.nan, math.nan]

    before_force_base = []
    corrected_force_base = []
    before_torque_tool = []
    corrected_torque_tool = []
    per_record = []
    for sample in samples:
        rotation = sample["rotation_base_tool"]
        gravity_tool = sample["gravity_tool"]
        predicted_force_tool = [
            force_bias[i] + signed_weight_n * gravity_tool[i] for i in range(3)
        ]
        predicted_torque_tool = [
            torque_bias[i] + _cross(weighted_com, gravity_tool)[i] for i in range(3)
        ]
        corrected_force_tool = [
            sample["force_tool"][i] - predicted_force_tool[i] for i in range(3)
        ]
        corrected_torque = [
            sample["torque_tool"][i] - predicted_torque_tool[i] for i in range(3)
        ]
        measured_force_base = _mat_vec(rotation, sample["force_tool"])
        corrected_base = _mat_vec(rotation, corrected_force_tool)
        before_force_base.append(measured_force_base)
        corrected_force_base.append(corrected_base)
        before_torque_tool.append(sample["torque_tool"])
        corrected_torque_tool.append(corrected_torque)
        per_record.append(
            {
                "index": sample["index"],
                "position_base": sample["position_base"],
                "measured_force_base": measured_force_base,
                "corrected_force_base": corrected_base,
                "measured_force_tool": sample["force_tool"],
                "predicted_force_tool": predicted_force_tool,
                "corrected_force_tool": corrected_force_tool,
                "measured_torque_tool": sample["torque_tool"],
                "predicted_torque_tool": predicted_torque_tool,
                "corrected_torque_tool": corrected_torque,
                "gravity_tool": gravity_tool,
            }
        )

    before_force_norms = [_norm(vector) for vector in before_force_base]
    corrected_force_norms = [_norm(vector) for vector in corrected_force_base]
    before_torque_norms = [_norm(vector) for vector in before_torque_tool]
    corrected_torque_norms = [_norm(vector) for vector in corrected_torque_tool]
    before_force_mean = statistics.fmean(before_force_norms)
    gravity_fraction = abs(signed_weight_n) / max(before_force_mean, 1e-9)
    warnings = []
    if gravity_fraction < 0.25:
        warnings.append(
            "The pose-dependent gravity term is small compared with the constant "
            "force bias; treat mass/CoM as unreliable and use this as an empirical "
            "no-contact bias correction only."
        )
    return {
        "mode": "tool_gravity_model",
        "source": source,
        "euler_order": order,
        "records": len(records),
        "warnings": warnings,
        "force_bias_tool_n": force_bias,
        "signed_weight_n": signed_weight_n,
        "gravity_fraction_of_mean_force": gravity_fraction,
        "mass_kg_abs": abs(signed_weight_n) / GRAVITY,
        "weighted_com_n_m": weighted_com,
        "com_tool_m": com_m,
        "torque_bias_tool_nm": torque_bias,
        "before_force_norm_mean_n": before_force_mean,
        "before_force_norm_max_n": max(before_force_norms),
        "corrected_force_norm_mean_n": statistics.fmean(corrected_force_norms),
        "corrected_force_norm_max_n": max(corrected_force_norms),
        "corrected_force_norm_rms_n": math.sqrt(
            statistics.fmean(value * value for value in corrected_force_norms)
        ),
        "before_torque_norm_mean_nm": statistics.fmean(before_torque_norms),
        "before_torque_norm_max_nm": max(before_torque_norms),
        "corrected_torque_norm_mean_nm": statistics.fmean(corrected_torque_norms),
        "corrected_torque_norm_max_nm": max(corrected_torque_norms),
        "corrected_torque_norm_rms_nm": math.sqrt(
            statistics.fmean(value * value for value in corrected_torque_norms)
        ),
        "mean_corrected_force_base_n": _mean_vector(corrected_force_base),
        "per_record": per_record,
    }


def _fit_work_offset_model(records: list[dict[str, object]], source: str) -> dict[str, object]:
    measured = []
    per_record = []
    for record in records:
        pose = [float(value) for value in record["pose_mean"]]
        wrench = _record_wrench(record, source)
        force_base = wrench[:3]
        measured.append(force_base)
        per_record.append(
            {
                "index": int(record["index"]),
                "position_base": pose[:3],
                "measured_force_base": force_base,
            }
        )
    offset = _mean_vector(measured)
    corrected = []
    for item in per_record:
        corrected_force = [
            item["measured_force_base"][i] - offset[i] for i in range(3)
        ]
        item["corrected_force_base"] = corrected_force
        corrected.append(corrected_force)
    before_norms = [_norm(vector) for vector in measured]
    corrected_norms = [_norm(vector) for vector in corrected]
    return {
        "mode": "base_constant_offset",
        "source": source,
        "records": len(records),
        "warnings": [
            "This model subtracts a constant base-frame offset only; it does not "
            "estimate payload mass or center of mass."
        ],
        "force_offset_base_n": offset,
        "before_force_norm_mean_n": statistics.fmean(before_norms),
        "before_force_norm_max_n": max(before_norms),
        "corrected_force_norm_mean_n": statistics.fmean(corrected_norms),
        "corrected_force_norm_max_n": max(corrected_norms),
        "corrected_force_norm_rms_n": math.sqrt(
            statistics.fmean(value * value for value in corrected_norms)
        ),
        "mean_corrected_force_base_n": _mean_vector(corrected),
        "per_record": per_record,
    }


def _fit_model(records: list[dict[str, object]], args) -> dict[str, object]:
    if args.source == "work":
        return _fit_work_offset_model(records, args.source)
    if args.euler_order == "auto":
        fits = [_fit_tool_model(records, args.source, order) for order in EULER_ORDERS]
        fits.sort(key=lambda fit: fit["corrected_force_norm_rms_n"])
        best = fits[0]
        best["candidate_orders"] = [
            {
                "euler_order": fit["euler_order"],
                "corrected_force_norm_rms_n": fit["corrected_force_norm_rms_n"],
                "corrected_torque_norm_rms_nm": fit["corrected_torque_norm_rms_nm"],
                "mass_kg_abs": fit["mass_kg_abs"],
            }
            for fit in fits
        ]
        return best
    return _fit_tool_model(records, args.source, args.euler_order)


def _project(point: list[float], scale: float, origin_x: float, origin_y: float) -> tuple[float, float]:
    x, y, z = point
    return (
        origin_x + scale * (0.866 * x - 0.866 * y),
        origin_y + scale * (0.5 * x + 0.5 * y - z),
    )


def _arrow_svg(
    start: list[float],
    vector: list[float],
    *,
    force_scale: float,
    coord_scale: float,
    origin_x: float,
    origin_y: float,
    color: str,
    label: str,
) -> str:
    end = [start[i] + vector[i] * force_scale for i in range(3)]
    x1, y1 = _project(start, coord_scale, origin_x, origin_y)
    x2, y2 = _project(end, coord_scale, origin_x, origin_y)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 7.0
    left = (x2 - head * math.cos(angle - 0.45), y2 - head * math.sin(angle - 0.45))
    right = (x2 - head * math.cos(angle + 0.45), y2 - head * math.sin(angle + 0.45))
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="2" />'
        f'<polygon points="{x2:.1f},{y2:.1f} {left[0]:.1f},{left[1]:.1f} '
        f'{right[0]:.1f},{right[1]:.1f}" fill="{color}" />'
        f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="2.2" fill="#222" />'
        f'<text x="{x1 + 4:.1f}" y="{y1 - 4:.1f}" font-size="9" fill="#333">'
        f'{html.escape(label)}</text>'
    )


def _panel_svg(
    title: str,
    per_record: list[dict[str, object]],
    vector_key: str,
    *,
    color: str,
) -> str:
    positions = [item["position_base"] for item in per_record]
    vectors = [item[vector_key] for item in per_record]
    all_points = positions + [
        [pos[i] + vec[i] * 0.035 for i in range(3)]
        for pos, vec in zip(positions, vectors)
    ]
    min_values = [min(point[i] for point in all_points) for i in range(3)]
    max_values = [max(point[i] for point in all_points) for i in range(3)]
    spans = [max(0.001, max_values[i] - min_values[i]) for i in range(3)]
    center = [(min_values[i] + max_values[i]) * 0.5 for i in range(3)]
    scale = 300.0 / max(spans)
    origin_x = 250.0 - scale * (0.866 * center[0] - 0.866 * center[1])
    origin_y = 230.0 - scale * (0.5 * center[0] + 0.5 * center[1] - center[2])
    arrows = []
    for item in per_record:
        arrows.append(
            _arrow_svg(
                item["position_base"],
                item[vector_key],
                force_scale=0.035,
                coord_scale=scale,
                origin_x=origin_x,
                origin_y=origin_y,
                color=color,
                label=str(item["index"]),
            )
        )
    axes = []
    axis_defs = [
        ([0.0, 0.0, 0.0], [0.18, 0.0, 0.0], "#d33", "X"),
        ([0.0, 0.0, 0.0], [0.0, 0.18, 0.0], "#2a2", "Y"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.18], "#36c", "Z"),
    ]
    for start, vector, axis_color, axis_label in axis_defs:
        axes.append(
            _arrow_svg(
                start,
                vector,
                force_scale=1.0,
                coord_scale=scale,
                origin_x=origin_x,
                origin_y=origin_y,
                color=axis_color,
                label=axis_label,
            )
        )
    return (
        '<div class="panel">'
        f"<h2>{html.escape(title)}</h2>"
        '<svg viewBox="0 0 500 330" width="500" height="330" role="img">'
        '<rect x="0" y="0" width="500" height="330" fill="#fafafa" />'
        f'{"".join(axes)}'
        f'{"".join(arrows)}'
        '</svg></div>'
    )


def _write_html(path: Path, fit: dict[str, object], input_path: Path) -> None:
    before = _panel_svg(
        "Measured base-frame force",
        fit["per_record"],
        "measured_force_base",
        color="#cc3333",
    )
    after = _panel_svg(
        "Corrected base-frame force",
        fit["per_record"],
        "corrected_force_base",
        color="#2a8f5a",
    )
    if fit["mode"] == "tool_gravity_model":
        model_rows = [
            ("mode", fit["mode"]),
            ("source", fit["source"]),
            ("euler_order", fit["euler_order"]),
            ("mass_kg_abs", f"{fit['mass_kg_abs']:.5f}"),
            ("signed_weight_n", f"{fit['signed_weight_n']:.5f}"),
            ("com_tool_m", _format(fit["com_tool_m"], precision=5)),
            ("force_bias_tool_n", _format(fit["force_bias_tool_n"], precision=5)),
        ]
    else:
        model_rows = [
            ("mode", fit["mode"]),
            ("source", fit["source"]),
            ("force_offset_base_n", _format(fit["force_offset_base_n"], precision=5)),
        ]
    model_rows.extend(
        [
            ("before_force_norm_mean_n", f"{fit['before_force_norm_mean_n']:.5f}"),
            ("before_force_norm_max_n", f"{fit['before_force_norm_max_n']:.5f}"),
            ("corrected_force_norm_mean_n", f"{fit['corrected_force_norm_mean_n']:.5f}"),
            ("corrected_force_norm_max_n", f"{fit['corrected_force_norm_max_n']:.5f}"),
            ("corrected_force_norm_rms_n", f"{fit['corrected_force_norm_rms_n']:.5f}"),
        ]
    )
    for index, warning in enumerate(fit.get("warnings", []), start=1):
        model_rows.append((f"warning_{index}", warning))
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in model_rows
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RM65 Force Calibration Visualization</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
.panel {{ border: 1px solid #ddd; padding: 12px; }}
h1 {{ font-size: 22px; margin: 0 0 10px; }}
h2 {{ font-size: 15px; margin: 0 0 8px; }}
table {{ border-collapse: collapse; margin-top: 18px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f4f4f4; }}
.note {{ color: #555; font-size: 13px; max-width: 980px; }}
</style>
</head>
<body>
<h1>RM65 Base-Frame 3D Force Visualization</h1>
<p class="note">Input: {html.escape(str(input_path))}. Arrows are plotted in the base frame.
Measured vectors are shown before correction; corrected vectors are residuals after the fitted no-contact model is subtracted.</p>
<div class="grid">{before}{after}</div>
<table>{rows}</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an offline RM65 no-contact force correction model from recorded poses "
            "and write a base-frame 3D force visualization."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="")
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--expected-tool", default="", help="Alias for --tool-label in standard calibration commands.")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--source", choices=FORCE_SOURCES, default="tool")
    parser.add_argument(
        "--euler-order",
        choices=("auto",) + EULER_ORDERS,
        default="auto",
        help="Only used for raw/zero/tool sources; work source is assumed to be base/work frame.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-html", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_json:
        input_path = Path(args.input_json)
    elif args.arm and (args.tool_label or args.expected_tool):
        input_path = force_sensor_calib4_measurements_path(args.arm, args.tool_label or args.expected_tool)
    else:
        input_path = _default_input_path()
    output_json = (
        Path(args.output_json)
        if args.output_json
        else RESULTS_DIR / f"{input_path.stem}_calibration_fit.json"
    )
    output_html = (
        Path(args.output_html)
        if args.output_html
        else RESULTS_DIR / f"{input_path.stem}_force_base_visualization.html"
    )
    try:
        payload, records = _load_records(input_path)
        fit = _fit_model(records, args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result = {
        "schema_version": 1,
        "generated_by": "hardware/rm65/utils/scripts/rm65_force_payload_calibrate.py",
        "input_json": str(input_path),
        "arm": payload.get("arm"),
        "ip": payload.get("ip"),
        "tool_label": payload.get("tool_label"),
        "input_data_type": payload.get("data_type", "legacy_combined_pose_force_records"),
        "gravity_base_dir": GRAVITY_BASE,
        "fit": fit,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_html(output_html, fit, input_path)

    print(f"records={fit['records']} source={fit['source']} mode={fit['mode']}")
    if fit["mode"] == "tool_gravity_model":
        print(f"euler_order={fit['euler_order']}")
        print(f"mass_abs={fit['mass_kg_abs']:.5f}kg signed_weight={fit['signed_weight_n']:.5f}N")
        print(f"gravity_fraction={fit['gravity_fraction_of_mean_force']:.5f}")
        print(f"com_tool_m={_format(fit['com_tool_m'])}")
        print(f"force_bias_tool_n={_format(fit['force_bias_tool_n'])}")
        print(f"torque_bias_tool_nm={_format(fit['torque_bias_tool_nm'])}")
    else:
        print(f"force_offset_base_n={_format(fit['force_offset_base_n'])}")
    print(
        "force_norm before mean/max="
        f"{fit['before_force_norm_mean_n']:.4f}/{fit['before_force_norm_max_n']:.4f}N"
    )
    print(
        "force_norm corrected mean/rms/max="
        f"{fit['corrected_force_norm_mean_n']:.4f}/"
        f"{fit['corrected_force_norm_rms_n']:.4f}/"
        f"{fit['corrected_force_norm_max_n']:.4f}N"
    )
    print(f"wrote {output_json}")
    print(f"wrote {output_html}")
    for warning in fit.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
