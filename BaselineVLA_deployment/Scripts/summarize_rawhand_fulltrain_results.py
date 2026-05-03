from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CASES = {
    "vanilla": {
        "label": "Vanilla-VLA",
        "eval_dir": "vanilla",
        "train_metrics": PROJECT_ROOT / "Outputs" / "vanilla" / "rawhand_phase3_vla_fulltrain_vanilla_seed7" / "metrics.json",
    },
    "fulltrain_vec_f1_8": {
        "label": "FullTrain-VLA vec_f1_8",
        "eval_dir": "fulltrain_vec_f1_8",
        "train_metrics": PROJECT_ROOT / "Outputs" / "fulltrain" / "rawhand_phase3_vla_fulltrain_vec_f1_8_seed7" / "metrics.json",
    },
    "fulltrain_pc_f1_8_pq_pn": {
        "label": "FullTrain-VLA pc_f1_8_pq_pn",
        "eval_dir": "fulltrain_pc_f1_8_pq_pn",
        "train_metrics": PROJECT_ROOT / "Outputs" / "fulltrain" / "rawhand_phase3_vla_fulltrain_pc_f1_8_pq_pn_seed7" / "metrics.json",
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return json.loads(path.read_text())


def _metric(summary: Dict[str, Any], key: str, metric_name: str) -> float | None:
    value = summary.get("results", {}).get("val", {}).get("metrics", {}).get(key, {}).get(metric_name)
    return None if value is None else float(value)


def _latest_train_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    metrics = _load_json(path)
    history = list(metrics.get("history", []))
    latest = history[-1] if history else {}
    return {
        "epochs": metrics.get("epochs"),
        "best_val_total_loss": metrics.get("best_val_total_loss"),
        "final_val_total_loss": latest.get("val", {}).get("total_loss"),
        "final_val_action_loss": latest.get("val", {}).get("action_loss"),
        "final_val_med_loss": latest.get("val", {}).get("med_loss"),
    }


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"`{value:.{digits}f}`"


def _fmt_pct(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"`{value:+.{digits}f}%`"


def build_summary(output_root: Path) -> Dict[str, Any]:
    eval_root = output_root / "evaluations"
    records: Dict[str, Dict[str, Any]] = {}
    for case_key, case_info in CASES.items():
        eval_summary = _load_json(eval_root / str(case_info["eval_dir"]) / "summary.json")
        records[case_key] = {
            "label": case_info["label"],
            "eval_summary": str(eval_root / str(case_info["eval_dir"]) / "summary.json"),
            "checkpoint": eval_summary.get("checkpoint"),
            "action_rmse": _metric(eval_summary, "action", "rmse"),
            "action_mae": _metric(eval_summary, "action", "mae"),
            "action_mse": _metric(eval_summary, "action", "mse"),
            "med_rmse": _metric(eval_summary, "med", "rmse"),
            "med_mae": _metric(eval_summary, "med", "mae"),
            "med_mse": _metric(eval_summary, "med", "mse"),
            "samples": eval_summary.get("results", {}).get("val", {}).get("samples"),
            "training": _latest_train_metrics(Path(case_info["train_metrics"])),
        }

    vanilla_rmse = records["vanilla"]["action_rmse"]
    for case_key, record in records.items():
        action_rmse = record["action_rmse"]
        if case_key == "vanilla" or vanilla_rmse in (None, 0.0) or action_rmse is None:
            record["rmse_vs_vanilla_pct"] = 0.0 if case_key == "vanilla" else None
        else:
            record["rmse_vs_vanilla_pct"] = (float(action_rmse) - float(vanilla_rmse)) / float(vanilla_rmse) * 100.0
    return {
        "output_root": str(output_root),
        "records": records,
    }


def write_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    records = summary["records"]
    lines = [
        "# RawHand FullTrain-VLA Results",
        "",
        "> **题注**",
        "> 数据集：`RawHandPointData`",
        "> 评测目标：验证真实训练形态 `obs -> predicted med -> action`，其中 `med` 与 `action` 在 `FullTrain` 中联合优化。",
        "> 比较口径：同一 `seed=7`、`basic_vla`、`task=false`、`2 epoch`、`1024 train / 1024 val`、完整 `val=1024` 评估。",
        "> Baseline：重新训练的 `Vanilla-VLA`，不是历史 `5 epoch` baseline。",
        "",
        "## 一览",
        "",
        "| run | samples | action RMSE | action MAE | action MSE | med RMSE | RMSE vs Vanilla |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_key in ("vanilla", "fulltrain_vec_f1_8", "fulltrain_pc_f1_8_pq_pn"):
        record = records[case_key]
        lines.append(
            "| "
            f"`{record['label']}` | "
            f"{record.get('samples', '-')} | "
            f"{_fmt(record['action_rmse'])} | "
            f"{_fmt(record['action_mae'])} | "
            f"{_fmt(record['action_mse'])} | "
            f"{_fmt(record['med_rmse'])} | "
            f"{_fmt_pct(record['rmse_vs_vanilla_pct'])} |"
        )

    lines.extend(
        [
            "",
            "## 训练侧指标",
            "",
            "| run | epochs | best val total loss | final val total | final val action | final val med |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case_key in ("vanilla", "fulltrain_vec_f1_8", "fulltrain_pc_f1_8_pq_pn"):
        record = records[case_key]
        training = record.get("training", {})
        lines.append(
            "| "
            f"`{record['label']}` | "
            f"{training.get('epochs', '-')} | "
            f"{_fmt(training.get('best_val_total_loss'))} | "
            f"{_fmt(training.get('final_val_total_loss'))} | "
            f"{_fmt(training.get('final_val_action_loss'))} | "
            f"{_fmt(training.get('final_val_med_loss'))} |"
        )

    lines.extend(
        [
            "",
            "## 结果文件",
            "",
        ]
    )
    for case_key in ("vanilla", "fulltrain_vec_f1_8", "fulltrain_pc_f1_8_pq_pn"):
        record = records[case_key]
        lines.append(f"- `{record['label']}`: `{record['eval_summary']}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RawHand FullTrain-VLA evaluation results.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "comparisons" / "rawhand_fulltrain_vs_vanilla",
    )
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    summary = build_summary(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_markdown(summary, output_root / "summary.md")
    print(f"saved {output_root / 'summary.json'}")
    print(f"saved {output_root / 'summary.md'}")

    records = summary["records"]
    print("| run | action_rmse | med_rmse | rmse_vs_vanilla |")
    print("| --- | ---: | ---: | ---: |")
    for case_key in ("vanilla", "fulltrain_vec_f1_8", "fulltrain_pc_f1_8_pq_pn"):
        record = records[case_key]
        print(
            f"| {record['label']} | "
            f"{_fmt(record['action_rmse'])} | "
            f"{_fmt(record['med_rmse'])} | "
            f"{_fmt_pct(record['rmse_vs_vanilla_pct'])} |"
        )


if __name__ == "__main__":
    main()
