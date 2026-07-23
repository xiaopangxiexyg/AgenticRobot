from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.arm_init import (
    ArmQposTransitionConfig,
    load_qpos_by_side_file,
    plan_slow_qpos_transition,
    qpos_transition_plan_to_dict,
)

from .qpos_init import DEFAULT_QPOS_TARGET_PATH
from ..constants import QPOS_UNIT
from ..qpos_layout import (
    normalize_sides,
    validate_qpos_by_side,
    validate_qpos_within_absolute_limits,
)


QPOS_TEST_SEQUENCE_DATA_TYPE = "rm65_arm_init_qpos_test_sequence"
QPOS_TEST_SEQUENCE_REPORT_DATA_TYPE = "rm65_arm_init_qpos_test_sequence_dry_run_report"
SCHEMA_VERSION = 1

DEFAULT_DATASET_ROOT = Path("datasets/2026-07-05_vive_rm65_real_teleop")
DEFAULT_SEQUENCE_DIR = Path("robot_motion/rm65/arm_init/qpos_test_sequences")
DEFAULT_EPISODE_GLOB = "data/chunk-*/episode_*.parquet"
DEFAULT_SAMPLES_PER_EPISODE = 3
DEFAULT_RANDOM_SEED = 0

DEFAULT_QPOS_COLUMN_CANDIDATES = {
    "left": (
        "observation.arm.left.current_qpos",
        "observation.left_arm.qpos",
        "observation.left_arm.current_qpos",
        "observation.arm.left.actual_qpos",
        "observation.arm.left.actual_hardware_qpos",
    ),
    "right": (
        "observation.arm.right.current_qpos",
        "observation.right_arm.qpos",
        "observation.right_arm.current_qpos",
        "observation.arm.right.actual_qpos",
        "observation.arm.right.actual_hardware_qpos",
    ),
}

DEFAULT_STATE_SLICES = {
    "left": slice(0, 6),
    "right": slice(19, 25),
}


@dataclass(frozen=True)
class QposSequenceEntry:
    index: int
    name: str
    qpos_by_side: dict[str, np.ndarray]
    qpos_units: str = QPOS_UNIT
    source: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = int(self.index)
        if index < 0:
            raise ValueError("qpos sequence entry index must be >= 0.")
        unit = str(self.qpos_units or QPOS_UNIT)
        if unit != QPOS_UNIT:
            raise ValueError(f"qpos_units must be {QPOS_UNIT!r}, got {unit!r}.")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "name", str(self.name or f"target_{index:06d}"))
        object.__setattr__(self, "qpos_by_side", validate_qpos_by_side(self.qpos_by_side))
        object.__setattr__(self, "qpos_units", unit)
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": int(self.index),
            "name": self.name,
            "qpos_by_side": _qpos_mapping_to_lists(self.qpos_by_side),
            "qpos_units": self.qpos_units,
        }
        if self.source:
            payload["source"] = _json_safe(self.source)
        if self.metadata:
            payload["metadata"] = _json_safe(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QposSequenceEntry":
        if not isinstance(payload, Mapping):
            raise ValueError("qpos sequence entry must be a mapping.")
        qpos_by_side = payload.get("qpos_by_side")
        if not isinstance(qpos_by_side, Mapping):
            raise ValueError("qpos sequence entry must contain qpos_by_side.")
        return cls(
            index=int(payload.get("index", 0)),
            name=str(payload.get("name") or ""),
            qpos_by_side=dict(qpos_by_side),
            qpos_units=str(payload.get("qpos_units") or QPOS_UNIT),
            source=dict(payload.get("source", {})) if isinstance(payload.get("source", {}), Mapping) else {},
            metadata=(
                dict(payload.get("metadata", {}))
                if isinstance(payload.get("metadata", {}), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class QposTestSequence:
    entries: tuple[QposSequenceEntry, ...]
    name: str = "rm65_qpos_test_sequence"
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ValueError(f"unsupported qpos sequence schema_version={self.schema_version!r}.")
        entries = tuple(
            entry if isinstance(entry, QposSequenceEntry) else QposSequenceEntry.from_dict(entry)
            for entry in self.entries
        )
        if not entries:
            raise ValueError("qpos test sequence must contain at least one entry.")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "name", str(self.name or "rm65_qpos_test_sequence"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QposTestSequence":
        if not isinstance(payload, Mapping):
            raise ValueError("qpos test sequence payload must be a mapping.")
        data_type = str(payload.get("data_type") or QPOS_TEST_SEQUENCE_DATA_TYPE)
        if data_type != QPOS_TEST_SEQUENCE_DATA_TYPE:
            raise ValueError(f"unsupported qpos test sequence data_type={data_type!r}.")
        entries_payload = payload.get("entries")
        if not isinstance(entries_payload, Sequence) or isinstance(entries_payload, (str, bytes)):
            raise ValueError("qpos test sequence payload must contain an entries list.")
        return cls(
            entries=tuple(QposSequenceEntry.from_dict(entry) for entry in entries_payload),
            name=str(payload.get("name") or "rm65_qpos_test_sequence"),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "data_type": QPOS_TEST_SEQUENCE_DATA_TYPE,
            "schema_version": int(self.schema_version),
            "name": self.name,
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if self.metadata:
            payload["metadata"] = _json_safe(self.metadata)
        return payload


def load_qpos_test_sequence(path: str | Path) -> QposTestSequence:
    return QposTestSequence.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_qpos_test_sequence(sequence: QposTestSequence, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sequence.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def qpos_test_sequence_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "qpos_test_sequence.schema.json"


def build_qpos_test_sequence_from_dataset(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    samples_per_episode: int = DEFAULT_SAMPLES_PER_EPISODE,
    seed: int = DEFAULT_RANDOM_SEED,
    active_sides: Sequence[str] = ("left", "right"),
    max_episodes: int = 0,
    episode_glob: str = DEFAULT_EPISODE_GLOB,
    qpos_columns_by_side: Mapping[str, str] | None = None,
    state_column: str = "observation.state",
    state_slices_by_side: Mapping[str, slice] | None = None,
    allow_short_episodes: bool = False,
    name: str | None = None,
) -> QposTestSequence:
    root = Path(dataset_root)
    sides = normalize_sides(tuple(active_sides))
    samples_per_episode = int(samples_per_episode)
    if samples_per_episode < 1:
        raise ValueError("samples_per_episode must be >= 1.")
    paths = _episode_paths(root, episode_glob=episode_glob)
    if int(max_episodes) > 0:
        paths = paths[: int(max_episodes)]
    if not paths:
        raise FileNotFoundError(f"no episode parquet files found under {root} with glob {episode_glob!r}.")

    rng = random.Random(int(seed))
    qpos_columns = {str(key): str(value) for key, value in dict(qpos_columns_by_side or {}).items() if value}
    state_slices = dict(DEFAULT_STATE_SLICES)
    if state_slices_by_side:
        state_slices.update({str(side): value for side, value in state_slices_by_side.items()})

    entries: list[QposSequenceEntry] = []
    episode_summaries: list[dict[str, Any]] = []
    for path in paths:
        records, summary = sample_episode_qpos_records(
            path=path,
            samples_per_episode=samples_per_episode,
            rng=rng,
            active_sides=sides,
            qpos_columns_by_side=qpos_columns,
            state_column=state_column,
            state_slices_by_side=state_slices,
            allow_short_episode=allow_short_episodes,
        )
        episode_summaries.append(summary)
        for record in records:
            entry_index = len(entries)
            source = dict(record["source"])
            source["dataset_root"] = str(root)
            source["episode_path"] = str(path.relative_to(root) if path.is_relative_to(root) else path)
            entries.append(
                QposSequenceEntry(
                    index=entry_index,
                    name=_entry_name(source, fallback_index=entry_index),
                    qpos_by_side=record["qpos_by_side"],
                    source=source,
                )
            )

    return QposTestSequence(
        name=name or f"{root.name}_{samples_per_episode}_per_episode_seed{int(seed)}",
        entries=tuple(entries),
        metadata={
            "dataset_root": str(root),
            "episode_glob": episode_glob,
            "episode_count": len(paths),
            "samples_per_episode": samples_per_episode,
            "seed": int(seed),
            "active_sides": list(sides),
            "qpos_columns_by_side": qpos_columns,
            "state_column": state_column,
            "state_slices_by_side": {side: _slice_to_text(state_slices[side]) for side in sides},
            "allow_short_episodes": bool(allow_short_episodes),
            "episode_summaries": episode_summaries,
        },
    )


def sample_episode_qpos_records(
    *,
    path: str | Path,
    samples_per_episode: int,
    rng: random.Random,
    active_sides: Sequence[str] = ("left", "right"),
    qpos_columns_by_side: Mapping[str, str] | None = None,
    state_column: str = "observation.state",
    state_slices_by_side: Mapping[str, slice] | None = None,
    allow_short_episode: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pq = _require_pyarrow_parquet()
    episode_path = Path(path)
    sides = normalize_sides(tuple(active_sides))
    samples_per_episode = int(samples_per_episode)
    if samples_per_episode < 1:
        raise ValueError("samples_per_episode must be >= 1.")

    parquet_file = pq.ParquetFile(episode_path)
    available_columns = set(parquet_file.schema_arrow.names)
    state_slices = dict(DEFAULT_STATE_SLICES)
    if state_slices_by_side:
        state_slices.update({str(side): value for side, value in state_slices_by_side.items()})
    sources_by_side = _resolve_qpos_sources(
        available_columns=available_columns,
        active_sides=sides,
        qpos_columns_by_side=qpos_columns_by_side or {},
        state_column=state_column,
        state_slices_by_side=state_slices,
    )
    metadata_columns = [
        column for column in ("episode_index", "frame_index", "timestamp") if column in available_columns
    ]
    qpos_columns = [
        source["column"]
        for source in sources_by_side.values()
        if source["kind"] == "column"
    ]
    state_columns = [
        source["column"]
        for source in sources_by_side.values()
        if source["kind"] == "state_slice"
    ]
    columns = _unique_preserve_order(metadata_columns + qpos_columns + state_columns)
    table = pq.read_table(episode_path, columns=columns)
    data = table.to_pydict()

    valid_records: list[dict[str, Any]] = []
    finite_qpos_row_count = 0
    absolute_limit_rejected_row_count = 0
    for row_index in range(table.num_rows):
        qpos_by_side: dict[str, np.ndarray] = {}
        valid = True
        for side in sides:
            qpos = _qpos_from_source(data, row_index=row_index, source=sources_by_side[side])
            if qpos is None:
                valid = False
                break
            qpos_by_side[side] = qpos
        if not valid:
            continue
        finite_qpos_row_count += 1
        try:
            qpos_by_side = validate_qpos_within_absolute_limits(
                qpos_by_side,
                label=f"{episode_path}:row_{row_index}",
            )
        except ValueError:
            absolute_limit_rejected_row_count += 1
            continue
        source = {
            "row_index": int(row_index),
            "qpos_source_by_side": {
                side: _source_to_json(sources_by_side[side]) for side in sides
            },
        }
        for column in metadata_columns:
            source[column] = _json_safe(data[column][row_index])
        valid_records.append({"qpos_by_side": qpos_by_side, "source": source})

    if len(valid_records) < samples_per_episode and not allow_short_episode:
        raise ValueError(
            f"{episode_path} has only {len(valid_records)} valid qpos rows, "
            f"but {samples_per_episode} samples were requested."
        )
    sample_count = min(samples_per_episode, len(valid_records))
    sampled = sorted(rng.sample(valid_records, sample_count), key=lambda item: int(item["source"]["row_index"]))
    summary = {
        "episode_path": str(episode_path),
        "row_count": int(table.num_rows),
        "finite_qpos_row_count": finite_qpos_row_count,
        "absolute_limit_rejected_row_count": absolute_limit_rejected_row_count,
        "valid_qpos_row_count": len(valid_records),
        "sample_count": sample_count,
        "sampled_row_indices": [int(record["source"]["row_index"]) for record in sampled],
        "qpos_source_by_side": {
            side: _source_to_json(sources_by_side[side]) for side in sides
        },
    }
    return sampled, summary


def dry_run_qpos_test_sequence(
    *,
    sequence: QposTestSequence,
    current_qpos_by_side: Mapping[str, Any],
    active_sides: Sequence[str] | None = None,
    config: ArmQposTransitionConfig | None = None,
    max_targets: int = 0,
    include_per_target_plans: bool = False,
    max_steps_json: int = 0,
) -> dict[str, Any]:
    cfg = config if config is not None else ArmQposTransitionConfig()
    current = validate_qpos_within_absolute_limits(
        current_qpos_by_side,
        label="current_qpos_by_side",
    )
    sides = _resolve_active_sides(sequence, current, active_sides)
    results: list[dict[str, Any]] = []
    total_steps = 0
    total_duration = 0.0
    entries = sequence.entries[: int(max_targets)] if int(max_targets) > 0 else sequence.entries

    for entry in entries:
        target = {side: entry.qpos_by_side[side] for side in sides}
        target = validate_qpos_within_absolute_limits(
            target,
            label=f"sequence_entry_{entry.index}",
        )
        plan = plan_slow_qpos_transition(
            start_qpos_by_side={side: current[side] for side in sides},
            target_qpos_by_side=target,
            config=cfg,
            side_order=sides,
        )
        summary = plan.summary()
        result: dict[str, Any] = {
            "target_index": int(entry.index),
            "target_name": entry.name,
            "source": _json_safe(entry.source),
            "summary": _json_safe(summary),
        }
        if include_per_target_plans:
            result["plan"] = qpos_transition_plan_to_dict(
                plan,
                include_steps=True,
                max_steps=int(max_steps_json),
            )
        results.append(result)
        total_steps += len(plan.steps)
        total_duration += float(summary["duration_sec"])
        current = plan.final_qpos_by_side

    return {
        "data_type": QPOS_TEST_SEQUENCE_REPORT_DATA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "sequence_name": sequence.name,
        "target_count": len(entries),
        "completed_count": sum(1 for result in results if result["summary"].get("completed")),
        "total_step_count": int(total_steps),
        "total_duration_sec": float(total_duration),
        "sides": list(sides),
        "config": {
            "dt_sec": float(cfg.dt_sec),
            "joint_velocity_profile_name": cfg.joint_velocity_profile_name,
            "max_steps_per_side": int(cfg.max_steps_per_side),
        },
        "final_qpos_by_side": _qpos_mapping_to_lists(current),
        "results": results,
    }


def default_sequence_output_path(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    samples_per_episode: int = DEFAULT_SAMPLES_PER_EPISODE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    root = Path(dataset_root)
    return DEFAULT_SEQUENCE_DIR / f"{root.name}_{int(samples_per_episode)}_per_episode_seed{int(seed)}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and dry-run RM65 qpos test sequences. These commands are read-only and "
            "do not connect to hardware."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Sample qpos targets from a LeRobot-style dataset.")
    build.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    build.add_argument("--output", default="", help="Output sequence JSON path.")
    build.add_argument("--samples-per-episode", type=int, default=DEFAULT_SAMPLES_PER_EPISODE)
    build.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    build.add_argument("--arm", default="both", help="left, right, or both.")
    build.add_argument("--max-episodes", type=int, default=0, help="0 means all episodes.")
    build.add_argument("--episode-glob", default=DEFAULT_EPISODE_GLOB)
    build.add_argument("--left-qpos-column", default="")
    build.add_argument("--right-qpos-column", default="")
    build.add_argument("--state-column", default="observation.state")
    build.add_argument("--state-left-slice", default=_slice_to_text(DEFAULT_STATE_SLICES["left"]))
    build.add_argument("--state-right-slice", default=_slice_to_text(DEFAULT_STATE_SLICES["right"]))
    build.add_argument("--allow-short-episodes", action="store_true")
    build.add_argument("--name", default="")

    dry_run = subparsers.add_parser("dry-run", help="Plan sequential slow init moves over a qpos sequence.")
    dry_run.add_argument("--sequence", required=True, help="Qpos test sequence JSON.")
    dry_run.add_argument(
        "--current-qpos",
        default=str(DEFAULT_QPOS_TARGET_PATH),
        help="Starting qpos JSON. Defaults to the measured init profile.",
    )
    dry_run.add_argument("--arm", default="both", help="left, right, or both.")
    dry_run.add_argument("--qpos-dt-sec", type=float, default=0.05)
    dry_run.add_argument("--joint-velocity-profile", default="rm65_init_slow")
    dry_run.add_argument("--max-targets", type=int, default=0, help="0 means all targets.")
    dry_run.add_argument("--output", default="", help="Optional report JSON path.")
    dry_run.add_argument("--include-per-target-plans", action="store_true")
    dry_run.add_argument("--max-steps-json", type=int, default=0, help="0 means all serialized steps.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        return _main_build(args)
    if args.command == "dry-run":
        return _main_dry_run(args)
    raise ValueError(f"unsupported command {args.command!r}.")


def _main_build(args: argparse.Namespace) -> int:
    qpos_columns_by_side = {
        side: value
        for side, value in {
            "left": str(args.left_qpos_column).strip(),
            "right": str(args.right_qpos_column).strip(),
        }.items()
        if value
    }
    sequence = build_qpos_test_sequence_from_dataset(
        dataset_root=args.dataset_root,
        samples_per_episode=int(args.samples_per_episode),
        seed=int(args.seed),
        active_sides=normalize_sides(str(args.arm)),
        max_episodes=int(args.max_episodes),
        episode_glob=str(args.episode_glob),
        qpos_columns_by_side=qpos_columns_by_side,
        state_column=str(args.state_column),
        state_slices_by_side={
            "left": _parse_slice(str(args.state_left_slice), label="--state-left-slice"),
            "right": _parse_slice(str(args.state_right_slice), label="--state-right-slice"),
        },
        allow_short_episodes=bool(args.allow_short_episodes),
        name=str(args.name).strip() or None,
    )
    output = Path(args.output) if str(args.output).strip() else default_sequence_output_path(
        dataset_root=args.dataset_root,
        samples_per_episode=int(args.samples_per_episode),
        seed=int(args.seed),
    )
    save_qpos_test_sequence(sequence, output)
    print(
        "rm65-qpos-sequence-build "
        f"output={output} "
        f"entries={len(sequence.entries)} "
        f"episodes={sequence.metadata.get('episode_count')} "
        f"samples_per_episode={sequence.metadata.get('samples_per_episode')} "
        f"seed={sequence.metadata.get('seed')}"
    )
    return 0


def _main_dry_run(args: argparse.Namespace) -> int:
    sequence = load_qpos_test_sequence(args.sequence)
    current = load_qpos_by_side_file(args.current_qpos)
    report = dry_run_qpos_test_sequence(
        sequence=sequence,
        current_qpos_by_side=current,
        active_sides=normalize_sides(str(args.arm)),
        config=ArmQposTransitionConfig(
            dt_sec=float(args.qpos_dt_sec),
            joint_velocity_profile_name=str(args.joint_velocity_profile),
        ),
        max_targets=int(args.max_targets),
        include_per_target_plans=bool(args.include_per_target_plans),
        max_steps_json=int(args.max_steps_json),
    )
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    if str(args.output).strip():
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(
        "rm65-qpos-sequence-dry-run "
        f"sequence={sequence.name} "
        f"targets={report['target_count']} "
        f"completed={report['completed_count']} "
        f"steps={report['total_step_count']} "
        f"duration_sec={report['total_duration_sec']:.3f}"
    )
    if not str(args.output).strip():
        print(text, end="")
    return 0


def _episode_paths(root: Path, *, episode_glob: str) -> list[Path]:
    return sorted(root.glob(episode_glob), key=_episode_sort_key)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if match:
        return int(match.group(1)), str(path)
    return 10**9, str(path)


def _resolve_qpos_sources(
    *,
    available_columns: set[str],
    active_sides: Sequence[str],
    qpos_columns_by_side: Mapping[str, str],
    state_column: str,
    state_slices_by_side: Mapping[str, slice],
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for side in active_sides:
        explicit_column = str(qpos_columns_by_side.get(side, "")).strip()
        if explicit_column:
            if explicit_column not in available_columns:
                raise ValueError(f"requested {side} qpos column {explicit_column!r} is not in the parquet file.")
            sources[side] = {"kind": "column", "column": explicit_column}
            continue
        column = next(
            (candidate for candidate in DEFAULT_QPOS_COLUMN_CANDIDATES[side] if candidate in available_columns),
            None,
        )
        if column is not None:
            sources[side] = {"kind": "column", "column": column}
            continue
        if state_column in available_columns and side in state_slices_by_side:
            sources[side] = {
                "kind": "state_slice",
                "column": state_column,
                "slice": state_slices_by_side[side],
            }
            continue
        raise ValueError(
            f"could not find a qpos source for {side}. Tried columns "
            f"{DEFAULT_QPOS_COLUMN_CANDIDATES[side]} and state column {state_column!r}."
        )
    return sources


def _qpos_from_source(
    data: Mapping[str, Sequence[Any]],
    *,
    row_index: int,
    source: Mapping[str, Any],
) -> np.ndarray | None:
    column = str(source["column"])
    value = data[column][row_index]
    if source["kind"] == "column":
        return _finite_vector(value, length=6)
    if source["kind"] == "state_slice":
        state = _finite_vector(value)
        qpos_slice = source["slice"]
        if state is None or state.size < int(qpos_slice.stop or 0):
            return None
        return _finite_vector(state[qpos_slice], length=6)
    raise ValueError(f"unsupported qpos source kind {source['kind']!r}.")


def _finite_vector(value: Any, *, length: int | None = None) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if length is not None and arr.shape != (int(length),):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float, copy=True)


def _resolve_active_sides(
    sequence: QposTestSequence,
    current_qpos_by_side: Mapping[str, np.ndarray],
    active_sides: Sequence[str] | None,
) -> tuple[str, ...]:
    if active_sides is not None:
        sides = normalize_sides(tuple(active_sides))
    else:
        sides = tuple(side for side in ("left", "right") if side in sequence.entries[0].qpos_by_side)
    missing_current = [side for side in sides if side not in current_qpos_by_side]
    missing_targets = [
        (entry.index, side)
        for entry in sequence.entries
        for side in sides
        if side not in entry.qpos_by_side
    ]
    if missing_current:
        raise ValueError(f"current qpos is missing sides {missing_current}.")
    if missing_targets:
        raise ValueError(f"sequence target qpos is missing sides {missing_targets[:5]}.")
    return tuple(sides)


def _entry_name(source: Mapping[str, Any], *, fallback_index: int) -> str:
    episode_index = _int_or_none(source.get("episode_index"))
    frame_index = _int_or_none(source.get("frame_index"))
    if episode_index is not None and frame_index is not None:
        return f"episode_{episode_index:06d}_frame_{frame_index:06d}"
    row_index = _int_or_none(source.get("row_index"))
    if row_index is not None:
        return f"target_{fallback_index:06d}_row_{row_index:06d}"
    return f"target_{fallback_index:06d}"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_slice(text: str, *, label: str) -> slice:
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"{label} must use start:stop format, got {text!r}.")
    try:
        start = int(parts[0])
        stop = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{label} must use integer start:stop format, got {text!r}.") from exc
    if start < 0 or stop <= start:
        raise ValueError(f"{label} must satisfy 0 <= start < stop, got {text!r}.")
    return slice(start, stop)


def _slice_to_text(value: slice) -> str:
    return f"{int(value.start or 0)}:{int(value.stop or 0)}"


def _source_to_json(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"kind": str(source["kind"]), "column": str(source["column"])}
    if source["kind"] == "state_slice":
        payload["slice"] = _slice_to_text(source["slice"])
    return payload


def _qpos_mapping_to_lists(qpos_by_side: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        side: np.asarray(qpos, dtype=float).reshape(-1).astype(float).tolist()
        for side, qpos in qpos_by_side.items()
    }


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Building a qpos test sequence from parquet requires pyarrow. "
            "Run this command in the dexteleop310 environment or install pyarrow."
        ) from exc
    return pq


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "QPOS_TEST_SEQUENCE_DATA_TYPE",
    "QPOS_TEST_SEQUENCE_REPORT_DATA_TYPE",
    "QposSequenceEntry",
    "QposTestSequence",
    "build_qpos_test_sequence_from_dataset",
    "default_sequence_output_path",
    "dry_run_qpos_test_sequence",
    "load_qpos_test_sequence",
    "qpos_test_sequence_schema_path",
    "sample_episode_qpos_records",
    "save_qpos_test_sequence",
]


if __name__ == "__main__":
    raise SystemExit(main())
