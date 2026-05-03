from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_config import load_local_env
from utils.runtime import ensure_dir, save_json


load_local_env()

DEFAULT_RESOURCE_RETRY_PROFILES = [
    {
        "label": "default",
        "overrides": [],
    },
    {
        "label": "reduced_workers",
        "overrides": [
            "data.raw.loader.num_workers=1",
            "data.raw.loader.prefetch_factor=1",
            "data.raw.loader.persistent_workers=false",
        ],
    },
    {
        "label": "single_process_loader",
        "overrides": [
            "data.raw.loader.num_workers=0",
            "data.raw.loader.prefetch_factor=null",
            "data.raw.loader.persistent_workers=false",
        ],
    },
]

DEFAULT_RUNNER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

LOCAL_ROOT_ENV_BY_ROOT_DIR = {
    "Data/RawData/Human/DeskTaskTacRawHand": "MANIPREP_DESKTASK_RAWHAND_ROOT",
    "Data/RawData/Human/DeskTaskAprilTag": "MANIPREP_DESKTASK_APRILTAG_ROOT",
    "Data/RawData/Human/RawHandPointData": "MANIPREP_RAWHAND_POINT_ROOT",
    "Data/RawData/Human/HumanMix": "MANIPREP_HUMANMIX_ROOT",
    "Data/RawData/RealRobot/chips_med": "MANIPREP_REALROBOT_CHIPS_ROOT",
    "Data/RawData/RealRobot/paper_fold": "MANIPREP_REALROBOT_PAPER_FOLD_ROOT",
    "Data/RawData/RealRobot/2026-04-30_heavy_light": "MANIPREP_REALROBOT_HEAVY_LIGHT_ROOT",
}


@dataclass
class RunSpec:
    suite_name: str
    run_name: str
    dataset_id: str
    dataset_label: str
    method_id: str | None
    method_label: str | None
    experiment: str
    model: str
    family: str
    method_kind: str
    med_target_id: str | None
    med_target_label: str | None
    seed: int
    task_enabled: bool
    med_window_id: str | None
    med_future_start: int | None
    med_future_end: int | None
    med_stride: int | None
    root_dir: str
    dataset_name_filter: str | None
    overrides: List[str]
    output_root: str = "Outputs"

    @property
    def run_dir(self) -> Path:
        return PROJECT_ROOT / self.output_root / self.experiment / self.run_name

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"


def _sanitize(text: str) -> str:
    normalized = []
    for char in str(text).lower():
        normalized.append(char if char.isalnum() else "_")
    collapsed = "".join(normalized).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: Any, precision: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100.0:.{precision}f}%"
    except (TypeError, ValueError):
        return str(value)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        file_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_from_history(metrics: Dict[str, Any], split: str, metric_name: str, reduce_mode: str) -> float | None:
    history = metrics.get("history", [])
    values: List[float] = []
    for epoch_record in history:
        split_record = epoch_record.get(split, {})
        value = split_record.get(metric_name)
        if value is None and metric_name == "action_loss":
            value = split_record.get("total_loss")
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    if reduce_mode == "min":
        return min(values)
    return values[-1]


def _resolve_primary_metric_name(experiment_name: str) -> str:
    normalized = str(experiment_name).lower()
    if normalized == "genmed":
        return "med_loss"
    if normalized in {"vanilla", "vanilla-tactile", "gtmed"}:
        return "action_loss"
    return "total_loss"


def _metric_or_fallback(metrics: Dict[str, Any], split: str, metric_name: str, reduce_mode: str) -> float | None:
    value = _metric_from_history(metrics, split=split, metric_name=metric_name, reduce_mode=reduce_mode)
    if value is not None:
        return value
    if metric_name != "total_loss":
        return _metric_from_history(metrics, split=split, metric_name="total_loss", reduce_mode=reduce_mode)
    return None


def _summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    experiment_name = str(metrics.get("experiment", "")).lower()
    primary_metric_name = _resolve_primary_metric_name(experiment_name)
    return {
        "primary_metric_name": primary_metric_name,
        "best_val_total_loss": metrics.get("best_val_total_loss"),
        "best_val_action_loss": _metric_from_history(metrics, split="val", metric_name="action_loss", reduce_mode="min"),
        "best_val_med_loss": _metric_from_history(metrics, split="val", metric_name="med_loss", reduce_mode="min"),
        "best_val_primary_metric": _metric_or_fallback(metrics, split="val", metric_name=primary_metric_name, reduce_mode="min"),
        "final_val_total_loss": _metric_from_history(metrics, split="val", metric_name="total_loss", reduce_mode="last"),
        "final_val_action_loss": _metric_from_history(metrics, split="val", metric_name="action_loss", reduce_mode="last"),
        "final_val_med_loss": _metric_from_history(metrics, split="val", metric_name="med_loss", reduce_mode="last"),
        "final_val_primary_metric": _metric_or_fallback(metrics, split="val", metric_name=primary_metric_name, reduce_mode="last"),
        "best_train_total_loss": _metric_from_history(metrics, split="train", metric_name="total_loss", reduce_mode="min"),
        "best_train_action_loss": _metric_from_history(metrics, split="train", metric_name="action_loss", reduce_mode="min"),
        "best_train_med_loss": _metric_from_history(metrics, split="train", metric_name="med_loss", reduce_mode="min"),
        "best_train_primary_metric": _metric_or_fallback(metrics, split="train", metric_name=primary_metric_name, reduce_mode="min"),
        "epochs": metrics.get("epochs"),
    }


def _aggregate_rows(rows: List[Dict[str, Any]], group_keys: List[str], metric_keys: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated_rows: List[Dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        aggregate = {group_key: value for group_key, value in zip(group_keys, key)}
        aggregate["num_runs"] = len(members)
        for metric_key in metric_keys:
            values = [float(member[metric_key]) for member in members if member.get(metric_key) is not None]
            if not values:
                aggregate[f"{metric_key}_mean"] = None
                aggregate[f"{metric_key}_min"] = None
                aggregate[f"{metric_key}_max"] = None
                continue
            aggregate[f"{metric_key}_mean"] = sum(values) / len(values)
            aggregate[f"{metric_key}_min"] = min(values)
            aggregate[f"{metric_key}_max"] = max(values)
        aggregated_rows.append(aggregate)
    return aggregated_rows


def _format_run_name(template: str, context: Dict[str, Any]) -> str:
    return _sanitize(str(template).format(**context))


def _is_configured(value: Any) -> bool:
    return value is not None and str(value).strip() and str(value).strip().lower() != "null"


def _dataset_local_root_overrides(dataset_spec: Dict[str, Any]) -> List[str]:
    local_root_dirs = dataset_spec.get("local_root_dirs")
    if isinstance(local_root_dirs, list) and local_root_dirs:
        configured = [str(value) for value in local_root_dirs if _is_configured(value)]
        if configured:
            return [f"data.raw.local_root_dirs=[{','.join(configured)}]"]

    local_root_dir = dataset_spec.get("local_root_dir")
    if _is_configured(local_root_dir):
        return [f"data.raw.local_root_dirs=[{local_root_dir}]"]

    env_name = dataset_spec.get("local_root_env")
    if not _is_configured(env_name):
        env_name = LOCAL_ROOT_ENV_BY_ROOT_DIR.get(str(dataset_spec.get("root_dir", "")))
    if _is_configured(env_name):
        return [f"data.raw.local_root_dirs=[${{oc.env:{env_name},null}}]"]

    return ["data.raw.local_root_dirs=[]"]


def _build_legacy_run_specs(suite: Dict[str, Any]) -> List[RunSpec]:
    suite_name = str(suite["suite_name"])
    defaults = suite["defaults"]
    shared_overrides = list(defaults.get("shared_overrides", []))
    data_config = str(defaults.get("data_config", "desktasktac_real"))
    train_config = str(defaults.get("train_config", "real_debug"))
    seeds = [int(seed) for seed in suite.get("seeds", [7])]

    def _common_overrides(dataset_spec: Dict[str, Any], method_spec: Dict[str, Any], seed: int) -> List[str]:
        overrides = [
            f"data={data_config}",
            f"train={train_config}",
            f"experiment={method_spec['experiment']}",
            f"model={method_spec['model']}",
            f"project.seed={seed}",
            f"data.raw.root_dirs=[{dataset_spec['root_dir']}]",
        ]
        overrides.extend(_dataset_local_root_overrides(dataset_spec))
        dataset_name_filter = dataset_spec.get("dataset_name_filter")
        if dataset_name_filter is None:
            overrides.append("data.raw.filters.dataset_name=null")
        else:
            overrides.append(f"data.raw.filters.dataset_name={dataset_name_filter}")
        task_enabled = bool(method_spec.get("task_enabled", False))
        overrides.append(f"model.task.enabled={'true' if task_enabled else 'false'}")
        overrides.extend(shared_overrides)
        overrides.extend(dataset_spec.get("extra_overrides", []))
        overrides.extend(method_spec.get("extra_overrides", []))
        if not any(str(override).startswith("train.wandb.group=") for override in overrides):
            overrides.append(f"train.wandb.group={suite_name}")
        return overrides

    run_specs: List[RunSpec] = []
    for dataset_spec in suite["datasets"]:
        dataset_id = str(dataset_spec["id"])
        dataset_label = str(dataset_spec.get("label", dataset_id))
        for seed in seeds:
            for method_spec in suite["vanilla_methods"]:
                method_id = str(method_spec["id"])
                run_name = _sanitize(f"{suite_name}_{dataset_id}_{method_id}_vanilla_seed{seed}")
                overrides = _common_overrides(dataset_spec, method_spec, seed)
                overrides.append(f"train.run_name={run_name}")
                run_specs.append(
                    RunSpec(
                        suite_name=suite_name,
                        run_name=run_name,
                        dataset_id=dataset_id,
                        dataset_label=dataset_label,
                        method_id=method_id,
                        method_label=str(method_spec.get("label", method_id)),
                        experiment=str(method_spec["experiment"]),
                        model=str(method_spec["model"]),
                        family=str(method_spec["family"]),
                        method_kind="vanilla",
                        med_target_id=None,
                        med_target_label=None,
                        seed=seed,
                        task_enabled=bool(method_spec.get("task_enabled", False)),
                        med_window_id=None,
                        med_future_start=None,
                        med_future_end=None,
                        med_stride=None,
                        root_dir=str(dataset_spec["root_dir"]),
                        dataset_name_filter=dataset_spec.get("dataset_name_filter"),
                        overrides=overrides,
                    )
                )

            for method_spec in suite["gtmed_methods"]:
                method_id = str(method_spec["id"])
                for med_window in suite["med_windows"]:
                    med_id = str(med_window["id"])
                    run_name = _sanitize(f"{suite_name}_{dataset_id}_{method_id}_gtmed_{med_id}_seed{seed}")
                    overrides = _common_overrides(dataset_spec, method_spec, seed)
                    overrides.extend(
                        [
                            f"data.raw.sequence.med_future_start={int(med_window['med_future_start'])}",
                            f"data.raw.sequence.med_future_end={int(med_window['med_future_end'])}",
                            f"data.raw.sequence.med_stride={int(med_window.get('med_stride', 1))}",
                            f"train.run_name={run_name}",
                        ]
                    )
                    run_specs.append(
                        RunSpec(
                            suite_name=suite_name,
                            run_name=run_name,
                            dataset_id=dataset_id,
                            dataset_label=dataset_label,
                            method_id=method_id,
                            method_label=str(method_spec.get("label", method_id)),
                            experiment=str(method_spec["experiment"]),
                            model=str(method_spec["model"]),
                            family=str(method_spec["family"]),
                            method_kind="gtmed",
                            med_target_id=med_id,
                            med_target_label=str(med_window.get("label", med_id)),
                            seed=seed,
                            task_enabled=bool(method_spec.get("task_enabled", False)),
                            med_window_id=med_id,
                            med_future_start=int(med_window["med_future_start"]),
                            med_future_end=int(med_window["med_future_end"]),
                            med_stride=int(med_window.get("med_stride", 1)),
                            root_dir=str(dataset_spec["root_dir"]),
                            dataset_name_filter=dataset_spec.get("dataset_name_filter"),
                            overrides=overrides,
                        )
                    )
    return run_specs


def _expand_method_parameter_grid(
    methods: List[Dict[str, Any]],
    suite_parameter_grid: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    expanded_methods: List[Dict[str, Any]] = []
    for method_spec in methods:
        method_grid = list(method_spec.get("parameter_grid", suite_parameter_grid))
        if not method_grid:
            expanded_methods.append(dict(method_spec))
            continue
        for parameter_spec in method_grid:
            param_id = str(parameter_spec["id"])
            param_label = str(parameter_spec.get("label", param_id))
            expanded = dict(method_spec)
            expanded["id"] = _sanitize(f"{method_spec['id']}_{param_id}")
            expanded["label"] = f"{method_spec.get('label', method_spec['id'])} / {param_label}"
            expanded["param_id"] = param_id
            expanded["param_label"] = param_label
            expanded["param_context"] = dict(parameter_spec.get("context", {}))
            expanded["extra_overrides"] = [
                *list(method_spec.get("extra_overrides", [])),
                *list(parameter_spec.get("extra_overrides", [])),
            ]
            if "run_name_override" in method_spec:
                expanded["run_name_override"] = method_spec["run_name_override"]
            expanded_methods.append(expanded)
    return expanded_methods


def _build_generic_run_specs(suite: Dict[str, Any]) -> List[RunSpec]:
    suite_name = str(suite["suite_name"])
    defaults = suite["defaults"]
    shared_overrides = list(defaults.get("shared_overrides", []))
    data_config = str(defaults.get("data_config", "desktasktac_real"))
    train_config = str(defaults.get("train_config", "real_debug"))
    seeds = [int(seed) for seed in suite.get("seeds", [7])]
    methods = _expand_method_parameter_grid(
        methods=list(suite.get("methods", [])),
        suite_parameter_grid=list(suite.get("parameter_grid", [])),
    )
    med_targets = list(suite.get("med_targets", []))

    if not methods:
        raise ValueError("Generic suite format requires a non-empty `methods` list.")

    def _common_overrides(dataset_spec: Dict[str, Any], method_spec: Dict[str, Any], seed: int) -> List[str]:
        overrides = [
            f"data={data_config}",
            f"train={train_config}",
            f"experiment={method_spec['experiment']}",
            f"model={method_spec['model']}",
            f"project.seed={seed}",
            f"data.raw.root_dirs=[{dataset_spec['root_dir']}]",
        ]
        overrides.extend(_dataset_local_root_overrides(dataset_spec))
        dataset_name_filter = dataset_spec.get("dataset_name_filter")
        if dataset_name_filter is None:
            overrides.append("data.raw.filters.dataset_name=null")
        else:
            overrides.append(f"data.raw.filters.dataset_name={dataset_name_filter}")
        task_enabled = bool(method_spec.get("task_enabled", False))
        overrides.append(f"model.task.enabled={'true' if task_enabled else 'false'}")
        overrides.extend(shared_overrides)
        overrides.extend(dataset_spec.get("extra_overrides", []))
        overrides.extend(method_spec.get("extra_overrides", []))
        if not any(str(override).startswith("train.wandb.group=") for override in overrides):
            overrides.append(f"train.wandb.group={suite_name}")
        return overrides

    run_specs: List[RunSpec] = []
    for dataset_spec in suite["datasets"]:
        dataset_id = str(dataset_spec["id"])
        dataset_label = str(dataset_spec.get("label", dataset_id))
        for seed in seeds:
            for method_spec in methods:
                method_id = str(method_spec["id"])
                method_label = str(method_spec.get("label", method_id))
                method_kind = str(method_spec.get("method_kind", method_spec.get("experiment", "unknown"))).lower()
                use_med_target = bool(method_spec.get("use_med_target", method_kind in {"gtmed", "fulltrain", "genmed"}))
                target_specs = med_targets if use_med_target else [None]
                if use_med_target and not target_specs:
                    raise ValueError(
                        f"Suite {suite_name} method id={method_id} requires a med target, but `med_targets` is empty."
                    )

                for target_spec in target_specs:
                    med_target_id = None if target_spec is None else str(target_spec["id"])
                    med_target_label = None if target_spec is None else str(target_spec.get("label", med_target_id))
                    context = {
                        "suite_name": suite_name,
                        "dataset_id": dataset_id,
                        "dataset_label": dataset_label,
                        "method_id": method_id,
                        "method_label": method_label,
                        "method_kind": method_kind,
                        "experiment": str(method_spec["experiment"]),
                        "model": str(method_spec["model"]),
                        "family": str(method_spec["family"]),
                        "seed": seed,
                        "med_target_id": med_target_id or "none",
                        "param_id": str(method_spec.get("param_id", "none")),
                        "param_label": str(method_spec.get("param_label", "none")),
                    }
                    context.update(dict(method_spec.get("param_context", {})))
                    run_name_template = method_spec.get("run_name_override")
                    if run_name_template is None:
                        suffix = "" if med_target_id is None else f"_{med_target_id}"
                        run_name = _sanitize(f"{suite_name}_{dataset_id}_{method_id}_{method_kind}{suffix}_seed{seed}")
                    else:
                        run_name = _format_run_name(str(run_name_template), context)

                    overrides = _common_overrides(dataset_spec, method_spec, seed)
                    if target_spec is not None:
                        overrides.extend(target_spec.get("extra_overrides", []))
                    overrides.append(f"train.run_name={run_name}")

                    run_specs.append(
                        RunSpec(
                            suite_name=suite_name,
                            run_name=run_name,
                            dataset_id=dataset_id,
                            dataset_label=dataset_label,
                            method_id=method_id,
                            method_label=method_label,
                            experiment=str(method_spec["experiment"]),
                            model=str(method_spec["model"]),
                            family=str(method_spec["family"]),
                            method_kind=method_kind,
                            med_target_id=med_target_id,
                            med_target_label=med_target_label,
                            seed=seed,
                            task_enabled=bool(method_spec.get("task_enabled", False)),
                            med_window_id=med_target_id,
                            med_future_start=None,
                            med_future_end=None,
                            med_stride=None,
                            root_dir=str(dataset_spec["root_dir"]),
                            dataset_name_filter=dataset_spec.get("dataset_name_filter"),
                            overrides=overrides,
                        )
                    )
    return run_specs


def _build_run_specs(suite: Dict[str, Any]) -> List[RunSpec]:
    if "methods" in suite:
        return _build_generic_run_specs(suite)
    return _build_legacy_run_specs(suite)


def _normalize_runner_profiles(runner_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    profiles = runner_config.get("resource_retry_profiles")
    if not profiles:
        profiles = DEFAULT_RESOURCE_RETRY_PROFILES
    normalized: List[Dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        normalized.append(
            {
                "label": str(profile.get("label", f"profile_{profile_index}")),
                "overrides": [str(override) for override in profile.get("overrides", [])],
            }
        )
    return normalized


def _build_runner_env(runner_config: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    if bool(runner_config.get("limit_cpu_thread_env", True)):
        env.update(DEFAULT_RUNNER_ENV)
    for key, value in dict(runner_config.get("env", {})).items():
        env[str(key)] = str(value)
    return env


def _should_retry(result: Dict[str, Any], attempt_index: int, total_attempts: int, runner_config: Dict[str, Any]) -> bool:
    if attempt_index >= total_attempts - 1:
        return False
    if not bool(runner_config.get("retry_on_resource_failure", True)):
        return False
    return result["status"] in {"failed", "timeout"}


def _run_single_attempt(
    run_spec: RunSpec,
    *,
    entrypoint: str,
    base_config: str,
    extra_overrides: List[str],
    env: Dict[str, str],
    timeout_sec: float | None,
) -> Dict[str, Any]:
    command = [sys.executable, entrypoint, f"--config-name={base_config}", *run_spec.overrides, *extra_overrides]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            env=env,
            timeout=None if timeout_sec is None else float(timeout_sec),
        )
        duration = time.time() - started
        result = {
            "status": "ok" if completed.returncode == 0 else "failed",
            "return_code": int(completed.returncode),
            "duration_sec": duration,
            "command": command,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - started
        result = {
            "status": "timeout",
            "return_code": None,
            "duration_sec": duration,
            "command": command,
            "timed_out": True,
        }

    if result["status"] == "ok" and run_spec.metrics_path.exists():
        metrics = _load_json(run_spec.metrics_path)
        result.update(_summarize_metrics(metrics))
    return result


def _run_single_spec(run_spec: RunSpec, entrypoint: str, base_config: str, runner_config: Dict[str, Any]) -> Dict[str, Any]:
    runner_profiles = _normalize_runner_profiles(runner_config)
    runner_env = _build_runner_env(runner_config)
    timeout_sec = runner_config.get("subprocess_timeout_sec")
    sleep_between_attempts_sec = float(runner_config.get("sleep_between_attempts_sec", 2.0))

    final_result: Dict[str, Any] | None = None
    for attempt_index, profile in enumerate(runner_profiles):
        if attempt_index > 0:
            print(
                f"  retry attempt {attempt_index + 1}/{len(runner_profiles)} "
                f"with profile={profile['label']} overrides={profile['overrides']}"
            )
        attempt_result = _run_single_attempt(
            run_spec,
            entrypoint=entrypoint,
            base_config=base_config,
            extra_overrides=list(profile["overrides"]),
            env=runner_env,
            timeout_sec=timeout_sec,
        )
        attempt_result["resource_profile"] = profile["label"]
        attempt_result["attempt_index"] = attempt_index + 1
        attempt_result["applied_retry_overrides"] = list(profile["overrides"])
        final_result = attempt_result
        if attempt_result["status"] == "ok":
            break
        if not _should_retry(attempt_result, attempt_index, len(runner_profiles), runner_config):
            break
        time.sleep(sleep_between_attempts_sec)

    assert final_result is not None
    final_result["attempt_count"] = int(final_result["attempt_index"])
    return final_result


def _normalize_evaluation_config(suite: Dict[str, Any]) -> Dict[str, Any] | None:
    evaluation = suite.get("evaluation")
    if not evaluation:
        return None
    normalized = dict(evaluation)
    if not bool(normalized.get("enabled", True)):
        return None
    normalized["script"] = str(normalized.get("script", "TestScripts/evaluate_checkpoint.py"))
    normalized["checkpoint_kind"] = str(normalized.get("checkpoint_kind", "best"))
    normalized["split"] = str(normalized.get("split", "val"))
    normalized["loader_mode"] = str(normalized.get("loader_mode", "safe"))
    normalized["visualize"] = str(normalized.get("visualize", "all"))
    normalized["output_tag"] = _sanitize(str(normalized.get("output_tag", "suite_eval")))
    normalized["skip_existing"] = bool(normalized.get("skip_existing", True))
    normalized["set"] = [str(item) for item in normalized.get("set", [])]
    return normalized


def _evaluation_output_dir(run_spec: RunSpec, evaluation_config: Dict[str, Any]) -> Path:
    return run_spec.run_dir / "evaluation" / str(evaluation_config["output_tag"])


def _extract_evaluation_summary_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {
        "evaluation_requested_split": summary.get("requested_split"),
        "evaluation_data_name": summary.get("data_name"),
        "evaluation_loader_mode": summary.get("loader_mode"),
        "evaluation_visualize": summary.get("visualize"),
    }
    for split_name, split_summary in dict(summary.get("results", {})).items():
        extracted[f"evaluation_{split_name}_samples"] = split_summary.get("samples")
        extracted[f"evaluation_{split_name}_batches"] = split_summary.get("batches")
        for prediction_key, metrics in dict(split_summary.get("metrics", {})).items():
            for metric_name in ("mse", "rmse", "mae"):
                metric_value = dict(metrics).get(metric_name)
                if metric_value is not None:
                    extracted[f"evaluation_{split_name}_{prediction_key}_{metric_name}"] = metric_value
    return extracted


def _empty_evaluation_result(status: str | None = None) -> Dict[str, Any]:
    return {
        "evaluation_status": status,
        "evaluation_return_code": None,
        "evaluation_duration_sec": 0.0,
        "evaluation_output_dir": None,
        "evaluation_summary_path": None,
        "evaluation_used_cache": False,
    }


def _build_existing_or_missing_evaluation_result(run_spec: RunSpec, evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = _evaluation_output_dir(run_spec, evaluation_config)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {
            **_empty_evaluation_result(status="missing"),
            "evaluation_output_dir": str(output_dir),
            "evaluation_summary_path": str(summary_path),
        }
    summary = _load_json(summary_path)
    return {
        **_empty_evaluation_result(status="ok"),
        "evaluation_return_code": 0,
        "evaluation_output_dir": str(output_dir),
        "evaluation_summary_path": str(summary_path),
        "evaluation_used_cache": True,
        **_extract_evaluation_summary_metrics(summary),
    }


def _run_evaluation_for_spec(
    run_spec: RunSpec,
    *,
    evaluation_config: Dict[str, Any],
    env: Dict[str, str],
    skip_existing: bool,
) -> Dict[str, Any]:
    output_dir = _evaluation_output_dir(run_spec, evaluation_config)
    summary_path = output_dir / "summary.json"
    if skip_existing and summary_path.exists():
        summary = _load_json(summary_path)
        return {
            **_empty_evaluation_result(status="ok"),
            "evaluation_return_code": 0,
            "evaluation_output_dir": str(output_dir),
            "evaluation_summary_path": str(summary_path),
            "evaluation_used_cache": True,
            **_extract_evaluation_summary_metrics(summary),
        }

    command = [
        sys.executable,
        str(evaluation_config["script"]),
        str(run_spec.run_dir),
        "--checkpoint-kind",
        str(evaluation_config["checkpoint_kind"]),
        "--split",
        str(evaluation_config["split"]),
        "--loader-mode",
        str(evaluation_config["loader_mode"]),
        "--visualize",
        str(evaluation_config["visualize"]),
        "--output-dir",
        str(output_dir),
    ]
    optional_args = {
        "--device": evaluation_config.get("device"),
        "--data-config": evaluation_config.get("data_config"),
        "--batch-size": evaluation_config.get("batch_size"),
        "--num-workers": evaluation_config.get("num_workers"),
        "--max-batches": evaluation_config.get("max_batches"),
        "--save-samples": evaluation_config.get("save_samples"),
        "--sample-index": evaluation_config.get("sample_index"),
        "--max-points": evaluation_config.get("max_points"),
        "--projection-point-radius": evaluation_config.get("projection_point_radius"),
        "--projection-frame-height": evaluation_config.get("projection_frame_height"),
        "--projection-max-steps": evaluation_config.get("projection_max_steps"),
        "--camera-serial": evaluation_config.get("camera_serial"),
        "--camera-intrinsics-json": evaluation_config.get("camera_intrinsics_json"),
    }
    for flag, value in optional_args.items():
        if value in (None, "", "null"):
            continue
        command.extend([flag, str(value)])
    for override in evaluation_config.get("set", []):
        command.extend(["--set", str(override)])

    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            env=env,
            timeout=None if evaluation_config.get("subprocess_timeout_sec") in (None, "null") else float(evaluation_config["subprocess_timeout_sec"]),
        )
        duration = time.time() - started
        status = "ok" if completed.returncode == 0 else "failed"
        result = {
            **_empty_evaluation_result(status=status),
            "evaluation_return_code": int(completed.returncode),
            "evaluation_duration_sec": duration,
            "evaluation_output_dir": str(output_dir),
            "evaluation_summary_path": str(summary_path),
            "evaluation_used_cache": False,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - started
        return {
            **_empty_evaluation_result(status="timeout"),
            "evaluation_duration_sec": duration,
            "evaluation_output_dir": str(output_dir),
            "evaluation_summary_path": str(summary_path),
            "evaluation_used_cache": False,
        }

    if result["evaluation_status"] == "ok" and summary_path.exists():
        result.update(_extract_evaluation_summary_metrics(_load_json(summary_path)))
    return result


def _load_existing_result(run_spec: RunSpec) -> Dict[str, Any] | None:
    if not run_spec.metrics_path.exists():
        return None
    metrics = _load_json(run_spec.metrics_path)
    return {
        "status": "ok",
        "return_code": 0,
        "duration_sec": 0.0,
        "command": [],
        **_summarize_metrics(metrics),
    }


def _inspect_run_artifacts(run_spec: RunSpec) -> Dict[str, Any]:
    run_dir = run_spec.run_dir
    metrics_exists = run_spec.metrics_path.exists()
    artifact_flags = {
        "run_dir_exists": run_dir.exists(),
        "metrics_exists": metrics_exists,
        "epoch_metrics_exists": (run_dir / "epoch_metrics.jsonl").exists(),
        "latest_epoch_metrics_exists": (run_dir / "latest_epoch_metrics.json").exists(),
        "best_checkpoint_exists": (run_dir / "best.pt").exists(),
        "last_checkpoint_exists": (run_dir / "last.pt").exists(),
        "wandb_exists": (run_dir / "wandb").exists(),
    }
    if metrics_exists:
        artifact_flags["artifact_status"] = "complete"
    elif artifact_flags["run_dir_exists"]:
        artifact_flags["artifact_status"] = "partial"
    else:
        artifact_flags["artifact_status"] = "missing"
    return artifact_flags


def _build_existing_or_missing_result(run_spec: RunSpec) -> Dict[str, Any]:
    cached = _load_existing_result(run_spec)
    artifacts = _inspect_run_artifacts(run_spec)
    if cached is not None:
        return {
            **cached,
            "used_cache": True,
            "return_code": 0,
            "duration_sec": 0.0,
            "command": [],
            "timed_out": False,
            **artifacts,
        }
    return {
        "status": "partial" if artifacts["artifact_status"] == "partial" else "missing",
        "return_code": None,
        "duration_sec": 0.0,
        "command": [],
        "timed_out": False,
        "used_cache": True,
        "best_val_total_loss": None,
        "best_val_action_loss": None,
        "best_val_med_loss": None,
        "best_val_primary_metric": None,
        "final_val_total_loss": None,
        "final_val_action_loss": None,
        "final_val_med_loss": None,
        "final_val_primary_metric": None,
        "best_train_total_loss": None,
        "best_train_action_loss": None,
        "best_train_med_loss": None,
        "best_train_primary_metric": None,
        "primary_metric_name": None,
        "epochs": None,
        **artifacts,
    }


def _annotate_result_row(run_spec: RunSpec, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **asdict(run_spec),
        **result,
        **_inspect_run_artifacts(run_spec),
    }


def _build_comparison_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vanilla_lookup: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for row in result_rows:
        if row.get("status") != "ok" or row.get("method_kind") != "vanilla":
            continue
        vanilla_lookup[(str(row["dataset_id"]), str(row["family"]), int(row["seed"]))] = row

    comparisons: List[Dict[str, Any]] = []
    for row in result_rows:
        if row.get("status") != "ok" or row.get("method_kind") != "gtmed":
            continue
        baseline = vanilla_lookup.get((str(row["dataset_id"]), str(row["family"]), int(row["seed"])))
        if baseline is None:
            continue
        baseline_metric = baseline.get("best_val_action_loss")
        gtmed_metric = row.get("best_val_action_loss")
        if baseline_metric is None or gtmed_metric is None:
            continue
        absolute_gain = float(baseline_metric) - float(gtmed_metric)
        relative_gain = None if float(baseline_metric) == 0.0 else absolute_gain / float(baseline_metric)
        comparisons.append(
            {
                "dataset_id": row["dataset_id"],
                "dataset_label": row["dataset_label"],
                "family": row["family"],
                "seed": row["seed"],
                "med_window_id": row["med_window_id"],
                "med_future_start": row["med_future_start"],
                "med_future_end": row["med_future_end"],
                "med_stride": row["med_stride"],
                "vanilla_run_name": baseline["run_name"],
                "gtmed_run_name": row["run_name"],
                "vanilla_best_val_action_loss": baseline_metric,
                "gtmed_best_val_action_loss": gtmed_metric,
                "absolute_gain": absolute_gain,
                "relative_gain": relative_gain,
            }
        )
    return comparisons


def _build_status_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in result_rows:
        rows.append(
            {
                "run_name": row["run_name"],
                "dataset_id": row["dataset_id"],
                "family": row["family"],
                "method_kind": row["method_kind"],
                "med_target_id": row.get("med_target_id"),
                "med_window_id": row["med_window_id"],
                "status": row["status"],
                "artifact_status": row.get("artifact_status"),
                "metrics_exists": row.get("metrics_exists"),
                "epoch_metrics_exists": row.get("epoch_metrics_exists"),
                "best_checkpoint_exists": row.get("best_checkpoint_exists"),
                "last_checkpoint_exists": row.get("last_checkpoint_exists"),
                "wandb_exists": row.get("wandb_exists"),
                "evaluation_status": row.get("evaluation_status"),
                "evaluation_summary_path": row.get("evaluation_summary_path"),
            }
        )
    return rows


def _build_summary_metadata(suite: Dict[str, Any], result_rows: List[Dict[str, Any]], comparison_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    analysis_target = str(suite.get("analysis_target", "best_val_action_loss"))
    status_counts: Dict[str, int] = {}
    dataset_counts: Dict[str, Dict[str, int]] = {}
    evaluation_status_counts: Dict[str, int] = {}
    for row in result_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        dataset_id = str(row["dataset_id"])
        dataset_status_counts = dataset_counts.setdefault(dataset_id, {})
        dataset_status_counts[status] = dataset_status_counts.get(status, 0) + 1
        evaluation_status = row.get("evaluation_status")
        if evaluation_status is not None:
            evaluation_status = str(evaluation_status)
            evaluation_status_counts[evaluation_status] = evaluation_status_counts.get(evaluation_status, 0) + 1

    best_by_dataset: Dict[str, Dict[str, Any]] = {}
    for row in result_rows:
        metric_value = row.get(analysis_target)
        if row.get("status") != "ok" or metric_value is None:
            continue
        dataset_id = str(row["dataset_id"])
        best_row = best_by_dataset.get(dataset_id)
        if best_row is None or float(metric_value) < float(best_row[analysis_target]):
            best_by_dataset[dataset_id] = row

    best_gain_by_dataset: Dict[str, Dict[str, Any]] = {}
    for row in comparison_rows:
        if row.get("absolute_gain") is None:
            continue
        dataset_id = str(row["dataset_id"])
        best_row = best_gain_by_dataset.get(dataset_id)
        if best_row is None or float(row["absolute_gain"]) > float(best_row["absolute_gain"]):
            best_gain_by_dataset[dataset_id] = row

    return {
        "suite_name": suite["suite_name"],
        "analysis_target": analysis_target,
        "planned_runs": len(result_rows),
        "status_counts": status_counts,
        "dataset_status_counts": dataset_counts,
        "evaluation_status_counts": evaluation_status_counts,
        "completed_ratio": 0.0 if not result_rows else status_counts.get("ok", 0) / len(result_rows),
        "best_by_dataset": {
            dataset_id: {
                "run_name": row["run_name"],
                "family": row["family"],
                "method_kind": row["method_kind"],
                "med_target_id": row.get("med_target_id"),
                "med_window_id": row["med_window_id"],
                analysis_target: row[analysis_target],
            }
            for dataset_id, row in best_by_dataset.items()
        },
        "best_gain_by_dataset": {
            dataset_id: {
                "gtmed_run_name": row["gtmed_run_name"],
                "family": row["family"],
                "med_window_id": row["med_window_id"],
                "absolute_gain": row["absolute_gain"],
                "relative_gain": row["relative_gain"],
            }
            for dataset_id, row in best_gain_by_dataset.items()
        },
    }


def _append_markdown_table(lines: List[str], headers: List[str], rows: List[List[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    if not rows:
        lines.append("| " + " | ".join("-" for _ in headers) + " |")
        return
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")


def _resolve_suite_epochs(suite: Dict[str, Any]) -> int | None:
    defaults = dict(suite.get("defaults", {}))
    shared_overrides = [str(item) for item in defaults.get("shared_overrides", [])]
    for override in shared_overrides:
        if not override.startswith("train.epochs="):
            continue
        _, value = override.split("=", 1)
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _write_markdown_report(
    path: Path,
    suite: Dict[str, Any],
    result_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
) -> None:
    analysis_target = str(suite.get("analysis_target", "best_val_action_loss"))
    ok_rows = [row for row in result_rows if row.get("status") == "ok"]
    partial_rows = [row for row in result_rows if row.get("status") == "partial"]
    missing_rows = [row for row in result_rows if row.get("status") == "missing"]
    evaluated_rows = [row for row in result_rows if row.get("evaluation_status") is not None]
    sorted_ok_rows = sorted(
        ok_rows,
        key=lambda row: (
            float("inf") if row.get(analysis_target) is None else float(row[analysis_target]),
            str(row["run_name"]),
        ),
    )
    best_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted_ok_rows:
        best_by_dataset.setdefault(str(row["dataset_id"]), []).append(row)

    sorted_comparisons = sorted(
        comparison_rows,
        key=lambda row: (
            float("inf") if row.get("absolute_gain") is None else -float(row["absolute_gain"]),
            str(row["gtmed_run_name"]),
        ),
    )
    best_comparisons_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted_comparisons:
        best_comparisons_by_dataset.setdefault(str(row["dataset_id"]), []).append(row)

    lines: List[str] = []
    lines.append(f"# Suite Report: {suite['suite_name']}")
    lines.append("")
    lines.append(f"- Description: {suite.get('description', '-')}")
    lines.append(f"- Analysis target: `{analysis_target}` (lower is better)")
    lines.append(f"- Planned runs: {len(result_rows)}")
    lines.append(f"- Completed runs: {len(ok_rows)}")
    lines.append(f"- Partial runs: {len(partial_rows)}")
    lines.append(f"- Missing runs: {len(missing_rows)}")
    lines.append(f"- Completion ratio: {_format_percent(len(ok_rows) / len(result_rows) if result_rows else 0.0)}")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    dataset_ids = sorted({str(row['dataset_id']) for row in result_rows})
    coverage_rows: List[List[Any]] = []
    for dataset_id in dataset_ids:
        dataset_rows = [row for row in result_rows if str(row["dataset_id"]) == dataset_id]
        coverage_rows.append(
            [
                dataset_id,
                len(dataset_rows),
                sum(1 for row in dataset_rows if row.get("status") == "ok"),
                sum(1 for row in dataset_rows if row.get("status") == "partial"),
                sum(1 for row in dataset_rows if row.get("status") == "missing"),
            ]
        )
    _append_markdown_table(lines, ["dataset", "planned", "ok", "partial", "missing"], coverage_rows)
    lines.append("")

    lines.append("## Best Available Runs")
    lines.append("")
    best_rows: List[List[Any]] = []
    for dataset_id in dataset_ids:
        for row in best_by_dataset.get(dataset_id, [])[:5]:
            best_rows.append(
                [
                    dataset_id,
                    row["family"],
                    row["method_kind"],
                    row.get("med_target_id") or row["med_window_id"] or "-",
                    _format_float(row.get(analysis_target)),
                    _format_float(row.get("best_train_primary_metric")),
                    row["run_name"],
                ]
            )
    _append_markdown_table(
        lines,
        ["dataset", "family", "method", "med_target", analysis_target, "best_train_primary_metric", "run_name"],
        best_rows,
    )
    lines.append("")

    lines.append("## GTMed vs Vanilla")
    lines.append("")
    comparison_table: List[List[Any]] = []
    for dataset_id in dataset_ids:
        dataset_rows = [row for row in best_comparisons_by_dataset.get(dataset_id, []) if row.get("absolute_gain") is not None]
        dataset_rows.sort(key=lambda row: float(row["absolute_gain"]), reverse=True)
        for row in dataset_rows[:6]:
            comparison_table.append(
                [
                    dataset_id,
                    row["family"],
                    row["med_window_id"] or "-",
                    _format_float(row["vanilla_best_val_action_loss"]),
                    _format_float(row["gtmed_best_val_action_loss"]),
                    _format_float(row["absolute_gain"]),
                    _format_percent(row["relative_gain"]),
                ]
            )
    _append_markdown_table(
        lines,
        ["dataset", "family", "med_window", "vanilla", "gtmed", "absolute_gain", "relative_gain"],
        comparison_table,
    )
    lines.append("")

    if evaluated_rows:
        lines.append("## Post-Train Evaluation")
        lines.append("")
        evaluation_table: List[List[Any]] = []
        for row in evaluated_rows:
            evaluation_table.append(
                [
                    row["dataset_id"],
                    row["method_kind"],
                    row.get("med_target_id") or "-",
                    row.get("evaluation_status") or "-",
                    _format_float(row.get("evaluation_val_action_rmse")),
                    _format_float(row.get("evaluation_val_med_rmse")),
                    row["run_name"],
                ]
            )
        _append_markdown_table(
            lines,
            ["dataset", "method", "med_target", "eval_status", "val_action_rmse", "val_med_rmse", "run_name"],
            evaluation_table,
        )
        lines.append("")

    lines.append("## Incomplete Or Missing Runs")
    lines.append("")
    missing_table: List[List[Any]] = []
    for row in [*partial_rows, *missing_rows]:
        missing_table.append(
            [
                row["dataset_id"],
                row["family"],
                row["method_kind"],
                row.get("med_target_id") or row["med_window_id"] or "-",
                row["status"],
                "yes" if row.get("run_dir_exists") else "no",
                "yes" if row.get("wandb_exists") else "no",
                row["run_name"],
            ]
        )
    _append_markdown_table(
        lines,
        ["dataset", "family", "method", "med_window", "status", "run_dir", "wandb", "run_name"],
        missing_table,
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `metrics.json` is the authoritative completion signal for a run in this report.")
    if evaluated_rows:
        lines.append("- Post-train evaluation summaries come from `TestScripts/evaluate_checkpoint.py` and are stored under each run directory.")
    else:
        lines.append("- Current training artifacts store `train` and `val` histories; no separate persisted `test` metrics were found for this suite.")
    configured_epochs = _resolve_suite_epochs(suite)
    if configured_epochs == 1:
        lines.append("- This suite is configured with `train.epochs=1`, so `best_*` and `final_*` validation metrics are identical whenever a run completed.")
    elif configured_epochs is not None:
        lines.append(f"- This suite is configured with `train.epochs={configured_epochs}`, so `best_*` and `final_*` validation metrics may differ.")
    else:
        lines.append("- `best_*` metrics summarize the best validation checkpoint, while `final_*` metrics summarize the last completed epoch.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a JSON-defined ManipRep experiment suite and summarize validation results.")
    parser.add_argument("suite_json", type=str, help="Path to a suite JSON file.")
    parser.add_argument("--max-runs", type=int, default=None, help="Only execute the first N planned runs.")
    parser.add_argument("--only", type=str, default=None, help="Optional substring filter on run_name.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Append a Hydra dotlist override to every planned run.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override post-train evaluation batch size.")
    parser.add_argument("--eval-num-workers", type=int, default=None, help="Override post-train evaluation loader worker count.")
    parser.add_argument("--eval-max-batches", type=int, default=None, help="Cap post-train evaluation batches per split.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have metrics.json.")
    parser.add_argument("--force", action="store_true", help="Ignore suite-level skip_existing and rerun selected specs.")
    parser.add_argument(
        "--summarize-existing-only",
        action="store_true",
        help="Do not launch training; summarize existing outputs only and mark incomplete/missing runs.",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite_json).expanduser().resolve()
    suite = _load_json(suite_path)
    run_specs = _build_run_specs(suite)
    if args.only:
        needle = str(args.only).lower()
        run_specs = [run_spec for run_spec in run_specs if needle in run_spec.run_name.lower()]
    if args.max_runs is not None:
        run_specs = run_specs[: int(args.max_runs)]
    if args.overrides:
        for run_spec in run_specs:
            run_spec.overrides.extend(str(override) for override in args.overrides)

    if args.dry_run:
        for run_spec in run_specs:
            print(run_spec.run_name)
        print(f"planned_runs={len(run_specs)}")
        return

    summary_dir_name = str(suite["suite_name"])
    if args.only:
        summary_dir_name = f"{summary_dir_name}__only_{_sanitize(args.only)}"
    if args.max_runs is not None:
        summary_dir_name = f"{summary_dir_name}__max_{int(args.max_runs)}"

    summary_root = ensure_dir(PROJECT_ROOT / "Outputs" / "suites" / summary_dir_name)
    save_json(summary_root / "suite_spec.json", suite)
    save_json(summary_root / "planned_runs.json", {"runs": [asdict(run_spec) for run_spec in run_specs]})

    runner_config = dict(suite.get("runner", {}))
    evaluation_config = _normalize_evaluation_config(suite)
    if evaluation_config is not None:
        if args.eval_batch_size is not None:
            evaluation_config["batch_size"] = int(args.eval_batch_size)
        if args.eval_num_workers is not None:
            evaluation_config["num_workers"] = int(args.eval_num_workers)
        if args.eval_max_batches is not None:
            evaluation_config["max_batches"] = int(args.eval_max_batches)
    skip_existing = False if args.force else bool(args.skip_existing or runner_config.get("skip_existing", False))
    fail_fast = bool(runner_config.get("fail_fast", False))
    entrypoint = str(suite.get("entrypoint", "Training/train.py"))
    base_config = str(suite.get("base_config", "train_runner"))
    runner_env = _build_runner_env(runner_config)

    result_rows: List[Dict[str, Any]] = []
    if args.summarize_existing_only:
        for run_spec in run_specs:
            result = _build_existing_or_missing_result(run_spec)
            evaluation_result = _empty_evaluation_result()
            if evaluation_config is not None:
                evaluation_result = _build_existing_or_missing_evaluation_result(run_spec, evaluation_config)
            result_rows.append(_annotate_result_row(run_spec, {**result, **evaluation_result}))
    else:
        for run_index, run_spec in enumerate(run_specs, start=1):
            print(f"[{run_index}/{len(run_specs)}] {run_spec.run_name}")
            if skip_existing:
                cached = _load_existing_result(run_spec)
                if cached is not None:
                    evaluation_result = _empty_evaluation_result()
                    if evaluation_config is not None:
                        evaluation_result = _run_evaluation_for_spec(
                            run_spec,
                            evaluation_config=evaluation_config,
                            env=runner_env,
                            skip_existing=bool(evaluation_config.get("skip_existing", True)),
                        )
                    row = _annotate_result_row(run_spec, {**cached, "used_cache": True, **evaluation_result})
                    result_rows.append(row)
                    print(f"  reused {run_spec.metrics_path}")
                    continue

            result = _run_single_spec(run_spec, entrypoint=entrypoint, base_config=base_config, runner_config=runner_config)
            evaluation_result = _empty_evaluation_result()
            if evaluation_config is not None:
                if result["status"] == "ok":
                    evaluation_result = _run_evaluation_for_spec(
                        run_spec,
                        evaluation_config=evaluation_config,
                        env=runner_env,
                        skip_existing=bool(skip_existing and evaluation_config.get("skip_existing", True)),
                    )
                else:
                    evaluation_result = _empty_evaluation_result(status="skipped")
            row = _annotate_result_row(run_spec, {**result, "used_cache": False, **evaluation_result})
            result_rows.append(row)
            if result["status"] != "ok" and fail_fast:
                break

    comparison_rows = _build_comparison_rows(result_rows)
    aggregate_rows = _aggregate_rows(
        result_rows,
        group_keys=["dataset_id", "family", "method_kind", "med_target_id", "med_window_id"],
        metric_keys=[
            "best_val_total_loss",
            "best_val_action_loss",
            "best_val_med_loss",
            "best_val_primary_metric",
            "final_val_total_loss",
            "final_val_action_loss",
            "final_val_med_loss",
            "final_val_primary_metric",
            "evaluation_val_action_rmse",
            "evaluation_val_med_rmse",
        ],
    )
    aggregate_comparisons = _aggregate_rows(
        [{**row, "status": "ok"} for row in comparison_rows],
        group_keys=["dataset_id", "family", "med_window_id"],
        metric_keys=["gtmed_best_val_action_loss", "vanilla_best_val_action_loss", "absolute_gain", "relative_gain"],
    )
    status_rows = _build_status_rows(result_rows)
    summary_metadata = _build_summary_metadata(suite, result_rows, comparison_rows)

    save_json(summary_root / "results.json", {"rows": result_rows})
    save_json(summary_root / "comparisons.json", {"rows": comparison_rows})
    save_json(summary_root / "aggregated_results.json", {"rows": aggregate_rows})
    save_json(summary_root / "aggregated_comparisons.json", {"rows": aggregate_comparisons})
    save_json(summary_root / "status.json", {"rows": status_rows})
    save_json(summary_root / "summary.json", summary_metadata)
    _write_csv(summary_root / "results.csv", result_rows)
    _write_csv(summary_root / "comparisons.csv", comparison_rows)
    _write_csv(summary_root / "aggregated_results.csv", aggregate_rows)
    _write_csv(summary_root / "aggregated_comparisons.csv", aggregate_comparisons)
    _write_csv(summary_root / "status.csv", status_rows)
    _write_markdown_report(summary_root / "report.md", suite=suite, result_rows=result_rows, comparison_rows=comparison_rows)

    ok_count = sum(1 for row in result_rows if row["status"] == "ok")
    print(f"finished suite={suite['suite_name']} ok={ok_count}/{len(result_rows)} summary_dir={summary_root}")


if __name__ == "__main__":
    main()
