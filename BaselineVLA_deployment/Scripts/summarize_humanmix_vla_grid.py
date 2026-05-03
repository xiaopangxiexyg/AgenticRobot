from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "Outputs" / "suites" / "humanmix_vla_action_history_grid"
HISTORY_ORDER = {
    "no_his": 0,
    "his1_s2": 1,
    "his1_s4": 2,
    "his1_s8": 3,
    "his2_s2": 4,
    "his2_s4": 5,
    "his2_s8": 6,
}
METHOD_ORDER = {
    "vanilla-tactile": 0,
    "fulltrain": 1,
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_param_id(value: str) -> Dict[str, Any]:
    text = str(value)
    match = re.search(r"ac(?P<action>\d+)_(?P<history>no_his|his(?P<history_frames>\d+)_s(?P<stride>\d+))", text)
    if match is not None:
        history_setting = str(match.group("history"))
        history_frames = 0 if history_setting == "no_his" else int(match.group("history_frames"))
        history_stride = 1 if history_setting == "no_his" else int(match.group("stride"))
        return {
            "action_chunk": int(match.group("action")),
            "history_setting": history_setting,
            "history_frames": int(history_frames),
            "state_history": int(history_frames) + 1,
            "history_stride": int(history_stride),
            "history_order": int(HISTORY_ORDER.get(history_setting, 10**6)),
        }

    legacy_match = re.search(r"ac(?P<action>\d+)_h(?P<state_history>\d+)_s(?P<stride>\d+)", text)
    if legacy_match is None:
        return {
            "action_chunk": None,
            "history_setting": None,
            "history_frames": None,
            "state_history": None,
            "history_stride": None,
            "history_order": 10**9,
        }
    state_history = int(legacy_match.group("state_history"))
    history_stride = int(legacy_match.group("stride"))
    history_setting = "no_his" if state_history <= 1 else f"his{state_history - 1}_s{history_stride}"
    return {
        "action_chunk": int(legacy_match.group("action")),
        "history_setting": history_setting,
        "history_frames": max(0, state_history - 1),
        "state_history": state_history,
        "history_stride": history_stride,
        "history_order": int(HISTORY_ORDER.get(history_setting, 10**6)),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference in (None, 0.0):
        return None
    return (float(candidate) - float(reference)) / float(reference) * 100.0


def _extract_rows(summary_dir: Path) -> List[Dict[str, Any]]:
    results_payload = _load_json(summary_dir / "results.json")
    planned_payload = _load_json(summary_dir / "planned_runs.json")
    result_rows = list(results_payload.get("rows", []))
    if result_rows:
        source_rows = result_rows
    else:
        source_rows = [
            {
                **planned_row,
                "status": "missing",
                "evaluation_status": "missing",
            }
            for planned_row in planned_payload.get("runs", [])
        ]

    rows: List[Dict[str, Any]] = []
    for row in source_rows:
        method_id = str(row.get("method_id", ""))
        run_name = str(row.get("run_name", ""))
        parsed = _parse_param_id(method_id)
        if parsed["action_chunk"] is None:
            parsed = _parse_param_id(run_name)
        rows.append(
            {
                **parsed,
                "method": str(row.get("method_kind", row.get("experiment", ""))),
                "status": str(row.get("status", "")),
                "evaluation_status": row.get("evaluation_status"),
                "best_val_action_loss": _float_or_none(row.get("best_val_action_loss")),
                "best_val_total_loss": _float_or_none(row.get("best_val_total_loss")),
                "evaluation_val_action_rmse": _float_or_none(row.get("evaluation_val_action_rmse")),
                "evaluation_val_action_mae": _float_or_none(row.get("evaluation_val_action_mae")),
                "evaluation_val_med_rmse": _float_or_none(row.get("evaluation_val_med_rmse")),
                "run_name": run_name,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            10**9 if row["action_chunk"] is None else int(row["action_chunk"]),
            int(row.get("history_order", 10**9)),
            str(row.get("history_setting") or ""),
            int(METHOD_ORDER.get(str(row["method"]), 10**6)),
            str(row["method"]),
        ),
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action_chunk",
        "history_setting",
        "history_frames",
        "state_history",
        "history_stride",
        "method",
        "status",
        "evaluation_status",
        "best_val_action_loss",
        "best_val_total_loss",
        "evaluation_val_action_rmse",
        "evaluation_val_action_mae",
        "evaluation_val_med_rmse",
        "run_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _append_table(lines: List[str], headers: List[str], table_rows: List[List[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    if not table_rows:
        lines.append("| " + " | ".join("-" for _ in headers) + " |")
        return
    for row in table_rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")


def _write_markdown(path: Path, rows: List[Dict[str, Any]], summary_dir: Path) -> None:
    lines = [
        "# HumanMix VLA Action/History Grid",
        "",
        f"- Source: `{summary_dir}`",
        "- Lower `evaluation_val_action_rmse` is better.",
        "- `history_frames` counts frames before the current frame; `state_history = history_frames + 1`.",
        "- `delta FullTrain vs Vanilla-tactile` is positive when FullTrain is worse.",
        "",
        "## Pivot",
        "",
    ]

    lookup: Dict[tuple[int, str, str], Dict[str, Any]] = {}
    for row in rows:
        if row["action_chunk"] is None or row["history_setting"] is None:
            continue
        lookup[(int(row["action_chunk"]), str(row["history_setting"]), str(row["method"]))] = row

    param_keys = sorted(
        {key[:2] for key in lookup},
        key=lambda key: (int(key[0]), int(HISTORY_ORDER.get(str(key[1]), 10**6)), str(key[1])),
    )
    pivot_rows: List[List[Any]] = []
    for action_chunk, history_setting in param_keys:
        vanilla = lookup.get((action_chunk, history_setting, "vanilla-tactile"))
        fulltrain = lookup.get((action_chunk, history_setting, "fulltrain"))
        reference = vanilla or fulltrain or {}
        vanilla_rmse = None if vanilla is None else vanilla.get("evaluation_val_action_rmse")
        fulltrain_rmse = None if fulltrain is None else fulltrain.get("evaluation_val_action_rmse")
        pivot_rows.append(
            [
                action_chunk,
                history_setting,
                reference.get("history_frames", "-"),
                reference.get("state_history", "-"),
                reference.get("history_stride", "-"),
                _fmt(vanilla_rmse),
                _fmt(fulltrain_rmse),
                "-" if _pct(fulltrain_rmse, vanilla_rmse) is None else f"{_pct(fulltrain_rmse, vanilla_rmse):+.2f}%",
                "-" if vanilla is None else vanilla.get("status", "-"),
                "-" if fulltrain is None else fulltrain.get("status", "-"),
            ]
        )
    _append_table(
        lines,
        [
            "action_chunk",
            "history_setting",
            "history_frames",
            "state_history",
            "history_stride",
            "Vanilla-tactile RMSE",
            "FullTrain RMSE",
            "delta FullTrain vs Vanilla",
            "Vanilla status",
            "FullTrain status",
        ],
        pivot_rows,
    )

    lines.extend(["", "## Detailed Rows", ""])
    detail_rows = [
        [
            row["action_chunk"],
            row["history_setting"],
            row["history_frames"],
            row["state_history"],
            row["history_stride"],
            row["method"],
            row["status"],
            row.get("evaluation_status") or "-",
            _fmt(row.get("best_val_action_loss")),
            _fmt(row.get("evaluation_val_action_rmse")),
            _fmt(row.get("evaluation_val_action_mae")),
            _fmt(row.get("evaluation_val_med_rmse")),
            row["run_name"],
        ]
        for row in rows
    ]
    _append_table(
        lines,
        [
            "action_chunk",
            "history_setting",
            "history_frames",
            "state_history",
            "history_stride",
            "method",
            "status",
            "eval",
            "best val action",
            "eval action RMSE",
            "eval action MAE",
            "eval med RMSE",
            "run_name",
        ],
        detail_rows,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HumanMix VLA action/history grid suite results.")
    parser.add_argument("--summary-dir", type=str, default=str(DEFAULT_SUMMARY_DIR))
    parser.add_argument("--output-md", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir).expanduser().resolve()
    rows = _extract_rows(summary_dir)
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else summary_dir / "humanmix_vla_grid_table.md"
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else summary_dir / "humanmix_vla_grid_table.csv"
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, summary_dir=summary_dir)
    print(f"wrote {output_md}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
