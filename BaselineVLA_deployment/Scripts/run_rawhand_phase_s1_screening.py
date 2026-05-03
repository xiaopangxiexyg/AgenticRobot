from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "ExperimentSuites" / "rawhand_explainable" / "rawhand_phase_s1_screening.json"


def _sanitize(text: str) -> str:
    normalized = []
    for char in str(text).lower():
        normalized.append(char if char.isalnum() else "_")
    collapsed = "".join(normalized).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed


def _summary_dir_name(*, only: str | None, max_runs: int | None) -> str:
    summary_name = "rawhand_phase_s1_screening"
    if only:
        summary_name = f"{summary_name}__only_{_sanitize(only)}"
    if max_runs is not None:
        summary_name = f"{summary_name}__max_{int(max_runs)}"
    return summary_name


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "Scripts" / "run_experiment_suite.py"),
        str(SUITE_PATH),
    ]
    if args.max_runs is not None:
        command.extend(["--max-runs", str(args.max_runs)])
    if args.only:
        command.extend(["--only", str(args.only)])
    if args.dry_run:
        command.append("--dry-run")
    if args.skip_existing:
        command.append("--skip-existing")
    if args.force:
        command.append("--force")
    if args.summarize_existing_only:
        command.append("--summarize-existing-only")
    return command


def _format_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def _format_percent(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}%"


def _write_phase_summary(summary_dir: Path) -> Path | None:
    results_path = summary_dir / "results.json"
    if not results_path.exists():
        return None

    rows = json.loads(results_path.read_text(encoding="utf-8")).get("rows", [])
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    vanilla_row = next(
        (
            row
            for row in ok_rows
            if row.get("method_kind") == "vanilla" and row.get("evaluation_val_action_rmse") is not None
        ),
        None,
    )
    gtmed_rows = [
        row
        for row in ok_rows
        if row.get("method_kind") == "gtmed" and row.get("evaluation_val_action_rmse") is not None
    ]
    gtmed_rows.sort(key=lambda row: float(row["evaluation_val_action_rmse"]))

    lines: list[str] = []
    lines.append("# Phase S1 Screening Summary")
    lines.append("")
    lines.append("- Ranking metric: `evaluation_val_action_rmse`")
    lines.append("- Dataset: `RawHandPointData`")
    lines.append("- Backbone: `minimal_mlp`")
    lines.append("")

    if vanilla_row is None:
        lines.append("Vanilla baseline evaluation is missing, so relative gains cannot be computed yet.")
        lines.append("")
    else:
        lines.append(
            f"- Vanilla baseline: `{vanilla_row['run_name']}` "
            f"(eval RMSE={_format_float(vanilla_row.get('evaluation_val_action_rmse'))})"
        )
        lines.append("")

    lines.append("| rank | med_target | eval_action_rmse | best_val_action_loss | delta_vs_vanilla | relative_gain | run_name |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    if not gtmed_rows:
        lines.append("| - | - | - | - | - | - | - |")
    else:
        baseline_rmse = None if vanilla_row is None else float(vanilla_row["evaluation_val_action_rmse"])
        for rank, row in enumerate(gtmed_rows, start=1):
            eval_rmse = float(row["evaluation_val_action_rmse"])
            delta = None if baseline_rmse is None else baseline_rmse - eval_rmse
            relative_gain = None if baseline_rmse in (None, 0.0) else delta / baseline_rmse
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        str(row.get("med_target_id") or "-"),
                        _format_float(eval_rmse),
                        _format_float(row.get("best_val_action_loss")),
                        _format_float(delta),
                        _format_percent(relative_gain),
                        str(row["run_name"]),
                    ]
                )
                + " |"
            )

    if gtmed_rows:
        top_rows = gtmed_rows[:2]
        lines.append("")
        lines.append("## Recommended Next Step")
        lines.append("")
        lines.append(
            "进入 Phase S2 时，优先保留："
            + "、".join(f"`{row.get('med_target_id')}`" for row in top_rows)
            + "。"
        )

    output_path = summary_dir / "phase_s1_screening_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase S1 RawHand tactile representation screening as a single integrated suite."
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Only execute the first N planned runs.")
    parser.add_argument("--only", type=str, default=None, help="Optional substring filter on run_name.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have metrics.json.")
    parser.add_argument("--force", action="store_true", help="Ignore suite-level skip_existing and rerun selected specs.")
    parser.add_argument(
        "--summarize-existing-only",
        action="store_true",
        help="Do not launch training; summarize existing outputs only and mark incomplete/missing runs.",
    )
    args = parser.parse_args()

    command = build_command(args)
    summary_dir = PROJECT_ROOT / "Outputs" / "suites" / _summary_dir_name(only=args.only, max_runs=args.max_runs)

    print(f"suite: {SUITE_PATH}", flush=True)
    print(f"summary_dir: {summary_dir}", flush=True)
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)

    if not args.dry_run:
        phase_summary_path = _write_phase_summary(summary_dir)
        print(f"report: {summary_dir / 'report.md'}", flush=True)
        print(f"results: {summary_dir / 'results.csv'}", flush=True)
        if phase_summary_path is not None:
            print(f"phase_summary: {phase_summary_path}", flush=True)


if __name__ == "__main__":
    main()
