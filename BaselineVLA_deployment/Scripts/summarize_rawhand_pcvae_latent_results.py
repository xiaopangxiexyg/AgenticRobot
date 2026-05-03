from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_POINT_SUMMARY = PROJECT_ROOT / "Outputs" / "comparisons" / "rawhand_wrist_local_pointcloud" / "summary.json"


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


def _metric(summary: Dict[str, Any], key: str, metric_name: str) -> float | None:
    value = _get(summary, "results", "val", "metrics", key, metric_name)
    return None if value is None else float(value)


def _samples(summary: Dict[str, Any]) -> int | None:
    value = _get(summary, "results", "val", "samples")
    return None if value is None else int(value)


def _pipeline_metric(summary: Dict[str, Any], mode: str, key: str, metric_name: str) -> float | None:
    value = _get(summary, "results", mode, "val", "metrics", key, metric_name)
    return None if value is None else float(value)


def _pipeline_samples(summary: Dict[str, Any], mode: str) -> int | None:
    value = _get(summary, "results", mode, "val", "samples")
    return None if value is None else int(value)


def _latest_val(metrics: Dict[str, Any], key: str) -> float | None:
    history = list(metrics.get("history", []))
    if not history:
        return None
    value = history[-1].get("val", {}).get(key)
    return None if value is None else float(value)


def _best_epoch(metrics: Dict[str, Any]) -> int | None:
    best = metrics.get("best_val_total_loss")
    if best is None:
        return None
    for row in metrics.get("history", []):
        total = row.get("val", {}).get("total_loss")
        if total is not None and abs(float(total) - float(best)) < 1e-12:
            return int(row.get("epoch", 0))
    return None


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


def _case_paths(output_root: Path, frame: str) -> Dict[str, Path]:
    tag = "wristlocal" if frame == "wrist_local" else "camera"
    return {
        "pcvae_metrics": PROJECT_ROOT / "Outputs" / "pcvae" / f"rawhand_pcvae_{tag}_f1_8_latent32_seed7" / "metrics.json",
        "genmed_eval": output_root / "evaluations" / f"genmed_{tag}_pcvae_latent_f1_8" / "summary.json",
        "pipeline": output_root / f"pipeline_{tag}_pcvae_latent_f1_8" / "summary.json",
        "fulltrain_eval": output_root / "evaluations" / f"fulltrain_{tag}_pcvae_latent_f1_8" / "summary.json",
    }


def build_summary(output_root: Path) -> Dict[str, Any]:
    raw_summary = _load_json(RAW_POINT_SUMMARY)
    records: Dict[str, Any] = {}

    for frame in ("wrist_local", "camera"):
        paths = _case_paths(output_root, frame)
        pcvae_metrics = _load_json(paths["pcvae_metrics"])
        genmed_eval = _load_json(paths["genmed_eval"])
        pipeline = _load_json(paths["pipeline"])
        fulltrain_eval = _load_json(paths["fulltrain_eval"])
        raw_record = _get(raw_summary, "records", frame) or {}

        vanilla_rmse = _pipeline_metric(pipeline, "vanilla", "action", "rmse")
        oracle_rmse = _pipeline_metric(pipeline, "gtmed_ground_truth_med", "action", "rmse")
        predicted_rmse = _pipeline_metric(pipeline, "gtmed_with_genmed_predicted_med", "action", "rmse")
        raw_pipeline_rmse = _get(raw_record, "pipeline", "predicted_action_rmse")
        raw_fulltrain_rmse = _get(raw_record, "fulltrain", "action_rmse")
        fulltrain_rmse = _metric(fulltrain_eval, "action", "rmse")

        records[frame] = {
            "paths": {key: str(value) for key, value in paths.items()},
            "raw_reference": raw_record,
            "pcvae": {
                "epochs": pcvae_metrics.get("epochs"),
                "best_epoch": _best_epoch(pcvae_metrics),
                "best_val_total_loss": pcvae_metrics.get("best_val_total_loss"),
                "final_val_total_loss": _latest_val(pcvae_metrics, "total_loss"),
                "final_val_reconstruction_loss": _latest_val(pcvae_metrics, "reconstruction_loss"),
                "final_val_xyz_loss": _latest_val(pcvae_metrics, "xyz_loss"),
                "final_val_value_loss": _latest_val(pcvae_metrics, "value_loss"),
                "final_val_kl_loss": _latest_val(pcvae_metrics, "kl_loss"),
            },
            "genmed": {
                "samples": _samples(genmed_eval),
                "med_rmse": _metric(genmed_eval, "med", "rmse"),
                "med_mae": _metric(genmed_eval, "med", "mae"),
                "med_mse": _metric(genmed_eval, "med", "mse"),
            },
            "pipeline": {
                "samples": _pipeline_samples(pipeline, "gtmed_with_genmed_predicted_med"),
                "vanilla_action_rmse": vanilla_rmse,
                "oracle_action_rmse": oracle_rmse,
                "predicted_action_rmse": predicted_rmse,
                "predicted_vs_vanilla_pct": _pct(predicted_rmse, vanilla_rmse),
                "predicted_vs_oracle_pct": _pct(predicted_rmse, oracle_rmse),
                "predicted_vs_raw_point_pct": _pct(predicted_rmse, raw_pipeline_rmse),
                "raw_point_predicted_action_rmse": raw_pipeline_rmse,
            },
            "fulltrain": {
                "samples": _samples(fulltrain_eval),
                "action_rmse": fulltrain_rmse,
                "action_mae": _metric(fulltrain_eval, "action", "mae"),
                "action_mse": _metric(fulltrain_eval, "action", "mse"),
                "med_rmse": _metric(fulltrain_eval, "med", "rmse"),
                "vs_vanilla_pct": _pct(fulltrain_rmse, vanilla_rmse),
                "vs_raw_point_fulltrain_pct": _pct(fulltrain_rmse, raw_fulltrain_rmse),
                "raw_point_fulltrain_action_rmse": raw_fulltrain_rmse,
            },
        }

    return {
        "output_root": str(output_root),
        "raw_point_summary": str(RAW_POINT_SUMMARY),
        "records": records,
    }


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    records = summary["records"]
    lines = [
        "# RawHand PCVAE-Latent Point-Cloud VLA Results",
        "",
        "> Purpose: test whether frame-level point-cloud VAE compression fixes the high-dimensional raw point-cloud bottleneck.",
        "> Scope: `pc_f1_8`, `latent_dim=32`, `basic_vla`, `task=false`, `seed=7`, `2 epoch`, `1024 train / 1024 val`.",
        "> Evaluation order: wrist-local first, then camera-frame.",
        "",
        "## PCVAE Pretraining",
        "",
        "| frame | epochs | best epoch | best val total | final reconstruction | final xyz | final value | final kl |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for frame in ("wrist_local", "camera"):
        row = records[frame]["pcvae"]
        lines.append(
            f"| `{frame}` | "
            f"{row.get('epochs', '-')} | "
            f"{_fmt(row.get('best_epoch'))} | "
            f"{_fmt(row.get('best_val_total_loss'))} | "
            f"{_fmt(row.get('final_val_reconstruction_loss'))} | "
            f"{_fmt(row.get('final_val_xyz_loss'))} | "
            f"{_fmt(row.get('final_val_value_loss'))} | "
            f"{_fmt(row.get('final_val_kl_loss'))} |"
        )

    lines.extend(
        [
            "",
            "## GenMed Latent Predictability",
            "",
            "| frame | samples | latent med RMSE | latent med MAE | latent med MSE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for frame in ("wrist_local", "camera"):
        row = records[frame]["genmed"]
        lines.append(
            f"| `{frame}` | "
            f"{_fmt(row.get('samples'))} | "
            f"{_fmt(row.get('med_rmse'))} | "
            f"{_fmt(row.get('med_mae'))} | "
            f"{_fmt(row.get('med_mse'))} |"
        )

    lines.extend(
        [
            "",
            "## Cached Pipeline Action",
            "",
            "| frame | samples | Vanilla RMSE | GTMed oracle RMSE | GenMed->GTMed RMSE | vs Vanilla | vs oracle | vs raw point pipeline |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for frame in ("wrist_local", "camera"):
        row = records[frame]["pipeline"]
        lines.append(
            f"| `{frame}` | "
            f"{_fmt(row.get('samples'))} | "
            f"{_fmt(row.get('vanilla_action_rmse'))} | "
            f"{_fmt(row.get('oracle_action_rmse'))} | "
            f"{_fmt(row.get('predicted_action_rmse'))} | "
            f"{_fmt_pct(row.get('predicted_vs_vanilla_pct'))} | "
            f"{_fmt_pct(row.get('predicted_vs_oracle_pct'))} | "
            f"{_fmt_pct(row.get('predicted_vs_raw_point_pct'))} |"
        )

    lines.extend(
        [
            "",
            "## FullTrain Action",
            "",
            "| frame | samples | action RMSE | action MAE | action MSE | latent med RMSE | vs Vanilla | vs raw point FullTrain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for frame in ("wrist_local", "camera"):
        row = records[frame]["fulltrain"]
        lines.append(
            f"| `{frame}` | "
            f"{_fmt(row.get('samples'))} | "
            f"{_fmt(row.get('action_rmse'))} | "
            f"{_fmt(row.get('action_mae'))} | "
            f"{_fmt(row.get('action_mse'))} | "
            f"{_fmt(row.get('med_rmse'))} | "
            f"{_fmt_pct(row.get('vs_vanilla_pct'))} | "
            f"{_fmt_pct(row.get('vs_raw_point_fulltrain_pct'))} |"
        )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- PCVAE latent is useful if pipeline action RMSE is close to raw point-cloud oracle/pipeline and FullTrain improves over raw point-cloud FullTrain.",
            "- If latent GenMed is easy but GTMed oracle is weak, the VAE compressed away action-relevant point-cloud information.",
            "- If GTMed oracle is good but FullTrain is weak, the remaining bottleneck is still joint optimization or conditioning.",
            "",
            "## Source Files",
            "",
        ]
    )
    for frame in ("wrist_local", "camera"):
        for key, path in records[frame]["paths"].items():
            lines.append(f"- `{frame}/{key}`: `{path}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RawHand PCVAE latent VLA experiments.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "comparisons" / "rawhand_pcvae_latent_vla",
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
