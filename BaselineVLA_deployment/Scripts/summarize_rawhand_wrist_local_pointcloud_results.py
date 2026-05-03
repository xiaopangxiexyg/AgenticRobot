from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CAMERA_PATHS = {
    "genmed": PROJECT_ROOT
    / "Outputs"
    / "genmed"
    / "rawhand_phase2_vla_genmed_pc_f1_8_pq_seed7"
    / "evaluation"
    / "DeskTaskTacRawHand_Point_Real_val"
    / "summary.json",
    "pipeline": PROJECT_ROOT / "Outputs" / "comparisons" / "rawhand_pipeline_vs_vanilla" / "pc_f1_8_pn" / "summary.json",
    "fulltrain": PROJECT_ROOT
    / "Outputs"
    / "comparisons"
    / "rawhand_fulltrain_vs_vanilla"
    / "evaluations"
    / "fulltrain_pc_f1_8_pq_pn"
    / "summary.json",
    "vanilla": PROJECT_ROOT
    / "Outputs"
    / "comparisons"
    / "rawhand_fulltrain_vs_vanilla"
    / "evaluations"
    / "vanilla"
    / "summary.json",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    return json.loads(path.read_text())


def _get(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _eval_metric(summary: Dict[str, Any], key: str, metric: str) -> float | None:
    value = _get(summary, "results", "val", "metrics", key, metric)
    return None if value is None else float(value)


def _eval_samples(summary: Dict[str, Any]) -> int | None:
    value = _get(summary, "results", "val", "samples")
    return None if value is None else int(value)


def _pipeline_metric(summary: Dict[str, Any], mode: str, key: str, metric: str) -> float | None:
    value = _get(summary, "results", mode, "val", "metrics", key, metric)
    return None if value is None else float(value)


def _pipeline_samples(summary: Dict[str, Any], mode: str) -> int | None:
    value = _get(summary, "results", mode, "val", "samples")
    return None if value is None else int(value)


def _pct(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference in (None, 0.0):
        return None
    return (float(candidate) - float(reference)) / float(reference) * 100.0


def _fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"`{value:.{digits}f}`"


def _fmt_pct(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"`{value:+.{digits}f}%`"


def build_summary(output_root: Path) -> Dict[str, Any]:
    wrist_paths = {
        "genmed": output_root / "evaluations" / "genmed_wrist_pc_f1_8_pq" / "summary.json",
        "pipeline": output_root / "pipeline_wrist_pc_f1_8_pn" / "summary.json",
        "fulltrain": output_root / "evaluations" / "fulltrain_wrist_pc_f1_8_pq_pn" / "summary.json",
    }
    loaded = {
        "camera": {key: _load_json(path) for key, path in DEFAULT_CAMERA_PATHS.items()},
        "wrist_local": {key: _load_json(path) for key, path in wrist_paths.items()},
    }

    records: Dict[str, Any] = {}
    for frame_name, summaries in loaded.items():
        genmed = summaries.get("genmed", {})
        pipeline = summaries.get("pipeline", {})
        fulltrain = summaries.get("fulltrain", {})
        vanilla_eval = summaries.get("vanilla", {})

        vanilla_rmse = _pipeline_metric(pipeline, "vanilla", "action", "rmse")
        if vanilla_rmse is None:
            vanilla_rmse = _eval_metric(vanilla_eval, "action", "rmse")
        oracle_rmse = _pipeline_metric(pipeline, "gtmed_ground_truth_med", "action", "rmse")
        predicted_rmse = _pipeline_metric(pipeline, "gtmed_with_genmed_predicted_med", "action", "rmse")
        fulltrain_rmse = _eval_metric(fulltrain, "action", "rmse")

        records[frame_name] = {
            "genmed": {
                "path": str(wrist_paths["genmed"] if frame_name == "wrist_local" else DEFAULT_CAMERA_PATHS["genmed"]),
                "samples": _eval_samples(genmed),
                "med_rmse": _eval_metric(genmed, "med", "rmse"),
                "med_mae": _eval_metric(genmed, "med", "mae"),
                "med_mse": _eval_metric(genmed, "med", "mse"),
            },
            "pipeline": {
                "path": str(wrist_paths["pipeline"] if frame_name == "wrist_local" else DEFAULT_CAMERA_PATHS["pipeline"]),
                "samples": _pipeline_samples(pipeline, "gtmed_with_genmed_predicted_med"),
                "vanilla_action_rmse": vanilla_rmse,
                "oracle_action_rmse": oracle_rmse,
                "predicted_action_rmse": predicted_rmse,
                "predicted_vs_vanilla_pct": _pct(predicted_rmse, vanilla_rmse),
                "predicted_vs_oracle_pct": _pct(predicted_rmse, oracle_rmse),
            },
            "fulltrain": {
                "path": str(wrist_paths["fulltrain"] if frame_name == "wrist_local" else DEFAULT_CAMERA_PATHS["fulltrain"]),
                "samples": _eval_samples(fulltrain),
                "vanilla_action_rmse": vanilla_rmse,
                "action_rmse": fulltrain_rmse,
                "med_rmse": _eval_metric(fulltrain, "med", "rmse"),
                "vs_vanilla_pct": _pct(fulltrain_rmse, vanilla_rmse),
            },
        }

    camera_fulltrain = records["camera"]["fulltrain"]["action_rmse"]
    wrist_fulltrain = records["wrist_local"]["fulltrain"]["action_rmse"]
    records["wrist_local"]["fulltrain"]["vs_camera_fulltrain_pct"] = _pct(wrist_fulltrain, camera_fulltrain)

    camera_pipeline = records["camera"]["pipeline"]["predicted_action_rmse"]
    wrist_pipeline = records["wrist_local"]["pipeline"]["predicted_action_rmse"]
    records["wrist_local"]["pipeline"]["vs_camera_pipeline_pct"] = _pct(wrist_pipeline, camera_pipeline)

    return {
        "output_root": str(output_root),
        "records": records,
    }


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    records = summary["records"]
    lines = [
        "# RawHand Wrist-Local Point-Cloud VLA Results",
        "",
        "> Purpose: compare the existing camera-frame point-cloud med with a wrist-local point-cloud med.",
        "> Scope: `pc_f1_8`, `basic_vla`, `task=false`, `seed=7`, `2 epoch`, `1024 train / 1024 val`.",
        "> Wrist-local transform: `p_local = R_eef^T (p_camera - t_eef)` in column-vector notation.",
        "",
        "## GenMed Predictability",
        "",
        "| frame | samples | med RMSE | med MAE | med MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for frame_name in ("camera", "wrist_local"):
        record = records[frame_name]["genmed"]
        lines.append(
            f"| `{frame_name}` | "
            f"{_fmt(record['samples'])} | "
            f"{_fmt(record['med_rmse'])} | "
            f"{_fmt(record['med_mae'])} | "
            f"{_fmt(record['med_mse'])} |"
        )

    lines.extend(
        [
            "",
            "## Cached Pipeline Action",
            "",
            "| frame | samples | Vanilla RMSE | GTMed oracle RMSE | GenMed->GTMed RMSE | predicted vs Vanilla | predicted vs oracle |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for frame_name in ("camera", "wrist_local"):
        record = records[frame_name]["pipeline"]
        lines.append(
            f"| `{frame_name}` | "
            f"{_fmt(record['samples'])} | "
            f"{_fmt(record['vanilla_action_rmse'])} | "
            f"{_fmt(record['oracle_action_rmse'])} | "
            f"{_fmt(record['predicted_action_rmse'])} | "
            f"{_fmt_pct(record['predicted_vs_vanilla_pct'])} | "
            f"{_fmt_pct(record['predicted_vs_oracle_pct'])} |"
        )
    lines.append("")
    lines.append(
        "Wrist-local pipeline vs existing camera pipeline: "
        f"{_fmt_pct(records['wrist_local']['pipeline'].get('vs_camera_pipeline_pct'))}."
    )

    lines.extend(
        [
            "",
            "## FullTrain Action",
            "",
            "| frame | samples | Vanilla RMSE | FullTrain RMSE | med RMSE | FullTrain vs Vanilla |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for frame_name in ("camera", "wrist_local"):
        record = records[frame_name]["fulltrain"]
        lines.append(
            f"| `{frame_name}` | "
            f"{_fmt(record['samples'])} | "
            f"{_fmt(record['vanilla_action_rmse'])} | "
            f"{_fmt(record['action_rmse'])} | "
            f"{_fmt(record['med_rmse'])} | "
            f"{_fmt_pct(record['vs_vanilla_pct'])} |"
        )
    lines.append("")
    lines.append(
        "Wrist-local FullTrain vs existing camera FullTrain: "
        f"{_fmt_pct(records['wrist_local']['fulltrain'].get('vs_camera_fulltrain_pct'))}."
    )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Wrist-local is useful if it improves FullTrain action RMSE over the existing camera-frame point-cloud line and does not degrade the cached pipeline.",
            "- If wrist-local improves GenMed med RMSE but not action RMSE, the representation is easier to predict but not yet better for control.",
            "- If wrist-local FullTrain beats Vanilla, point-cloud should remain a serious structured alternative to `vec_f1_8`.",
            "",
            "## Source Files",
            "",
        ]
    )
    for frame_name in ("camera", "wrist_local"):
        for block_name in ("genmed", "pipeline", "fulltrain"):
            lines.append(f"- `{frame_name}/{block_name}`: `{records[frame_name][block_name]['path']}`")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize wrist-local point-cloud RawHand VLA experiments.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "comparisons" / "rawhand_wrist_local_pointcloud",
    )
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    summary = build_summary(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, output_root / "summary.md")
    print(f"saved {output_root / 'summary.json'}")
    print(f"saved {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
