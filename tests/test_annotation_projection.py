from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from collection_runtime.progress.retrospective_store import (
    commit_retrospective_progress_episode,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import TaskStageProfileV1
from dataset_contracts.retrospective_progress import (
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveProgressWindowV1,
    RetrospectiveSourceBindingV1,
    RetrospectiveVideoBindingV1,
)
from dataset_interface.episode_curation import (
    source_dataset_base_fingerprint,
    source_episode_manifest_hash,
)
from dataset_interface.retrospective_progress import (
    retrospective_sample_axis_hash,
)
from dataset_interface.sample_axis import EpisodeSampleAxis


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CHECKER = _REPOSITORY_ROOT / "scripts/check_annotation_projection.py"
_MANIFEST = _REPOSITORY_ROOT / "release/annotation_projection.toml"
_LOCAL_PACKAGE_ROOTS = (
    "collection_runtime",
    "control_station",
    "dataset_contracts",
    "dataset_interface",
)


def _checker_payload() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(_CHECKER), "--json"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _runtime_closure_modules() -> set[str]:
    source = r"""
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
import control_station.refinement_annotation.cli

print(json.dumps(sorted(
    name
    for name, module in sys.modules.items()
    if getattr(module, "__file__", None)
    and (
        Path(module.__file__).resolve() == root
        or root in Path(module.__file__).resolve().parents
    )
)))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPOSITORY_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return set(json.loads(result.stdout))


def _materialize_projection(root: Path, payload: dict[str, object]) -> None:
    included_paths = payload["included_paths"]
    assert isinstance(included_paths, list)
    for relative_value in included_paths:
        assert isinstance(relative_value, str)
        relative = Path(relative_value)
        source = _REPOSITORY_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


_PROJECTED_ENTRYPOINT_RUNNER = r"""
import os
from pathlib import Path
import runpy
import sys

projection_root = Path(sys.argv[1]).resolve()
repository_root = Path(os.environ.pop("DEXTELEOP_REPOSITORY_ROOT")).resolve()
entrypoint_args = sys.argv[2:]
clean_path = []
for value in sys.path:
    resolved = Path(value or Path.cwd()).resolve()
    if resolved in {repository_root, projection_root}:
        continue
    clean_path.append(value)
sys.path[:] = [str(projection_root), *clean_path]
sys.argv = ["control_station", *entrypoint_args]

exit_code = 0
try:
    runpy.run_module("control_station", run_name="__main__")
except SystemExit as exc:
    exit_code = 0 if exc.code is None else int(exc.code)

local_roots = {
    "collection_runtime",
    "control_station",
    "dataset_contracts",
    "dataset_interface",
}
escaped = []
for name, module in sys.modules.items():
    if name.split(".", 1)[0] not in local_roots:
        continue
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        continue
    resolved = Path(module_path).resolve()
    if resolved != projection_root and projection_root not in resolved.parents:
        escaped.append((name, str(resolved)))
if escaped:
    raise RuntimeError(f"projection imported repository files: {escaped}")
raise SystemExit(exit_code)
"""


def _run_projected_entrypoint(
    projection_root: Path,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["DEXTELEOP_REPOSITORY_ROOT"] = str(_REPOSITORY_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _PROJECTED_ENTRYPOINT_RUNNER,
            str(projection_root),
            *args,
        ],
        cwd=projection_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


_PROJECTED_GUI_SMOKE_RUNNER = r"""
import os
from pathlib import Path
import sys
from urllib.parse import urljoin
from urllib.request import build_opener, ProxyHandler

projection_root = Path(sys.argv[1]).resolve()
repository_root = Path(os.environ.pop("DEXTELEOP_REPOSITORY_ROOT")).resolve()
clean_path = []
for value in sys.path:
    resolved = Path(value or Path.cwd()).resolve()
    if resolved in {repository_root, projection_root}:
        continue
    clean_path.append(value)
sys.path[:] = [str(projection_root), *clean_path]

from control_station.refinement_annotation.server import RefinementAnnotationGui

gui = RefinementAnnotationGui(None, open_browser=False)
try:
    url = gui.start()
    opener = build_opener(ProxyHandler({}))
    payloads = []
    for relative_path in ("", "app.js", "styles.css"):
        with opener.open(urljoin(url, relative_path), timeout=5.0) as response:
            payloads.append(response.read().decode("utf-8"))
    assert "DexTeleop" in payloads[0]
    assert all(payload.strip() for payload in payloads[1:])
finally:
    gui.close()

local_roots = {
    "collection_runtime",
    "control_station",
    "dataset_contracts",
    "dataset_interface",
}
escaped = []
for name, module in sys.modules.items():
    if name.split(".", 1)[0] not in local_roots:
        continue
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        continue
    resolved = Path(module_path).resolve()
    if resolved != projection_root and projection_root not in resolved.parents:
        escaped.append((name, str(resolved)))
if escaped:
    raise RuntimeError(f"projection imported repository files: {escaped}")
"""


def _run_projected_gui_smoke(
    projection_root: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
    ):
        environment.pop(name, None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["DEXTELEOP_REPOSITORY_ROOT"] = str(_REPOSITORY_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _PROJECTED_GUI_SMOKE_RUNNER,
            str(projection_root),
        ],
        cwd=projection_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _retrospective_profiles() -> tuple[
    TaskProgressProfileV1,
    TaskStageProfileV1,
]:
    progress_profile = TaskProgressProfileV1.from_payload(
        {
            "schema": "dexteleop.task_progress_profile.v1",
            "profile_id": "annotation_projection_fixture",
            "profile_version": "1",
            "task_id": "annotation_projection_fixture",
            "steps": [
                {
                    "schema": "dexteleop.progress_step_definition.v1",
                    "progress_step_id": "fixture_step",
                    "semantic_type": "FIXTURE_STEP",
                    "display_name": "Fixture step",
                    "ordinal": 1,
                    "allowed_next_step_ids": [],
                }
            ],
            "initial_step_id": "fixture_step",
            "terminal_step_ids": ["fixture_step"],
            "topology_kind": "LINEAR",
            "allow_retry_after_failure": False,
        }
    )
    stage_profile = TaskStageProfileV1.from_payload(
        {
            "schema": "dexteleop.task_stage_profile.v1",
            "profile_id": "annotation_projection_fixture_stages",
            "profile_version": "1",
            "task_progress_profile_hash": (
                progress_profile.semantic_content_hash
            ),
            "stages": [
                {
                    "schema": "dexteleop.progress_stage_definition.v1",
                    "progress_step_id": "fixture_step",
                    "stage_id": "fixture_step.stage",
                    "display_name": "Fixture stage",
                    "description": "Portable retrospective fixture stage.",
                    "ordinal": 1,
                }
            ],
        }
    )
    return progress_profile, stage_profile


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrospective_workspace(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    (source_root / "meta").mkdir(parents=True)
    (source_root / "data/chunk-000").mkdir(parents=True)
    (source_root / "videos/front/chunk-000").mkdir(parents=True)
    info_path = source_root / "meta/info.json"
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v2.3",
                "dexteleop_dataset_schema": "dexteleop_dataset_v2.3",
                "episode_count": 1,
                "frame_count": 5,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parquet_relative = "data/chunk-000/episode_000000.parquet"
    parquet_path = source_root / parquet_relative
    pq.write_table(
        pa.table({"timestamp": pa.array((0.0, 0.1, 0.2, 0.3, 0.4))}),
        parquet_path,
    )
    video_relative = "videos/front/chunk-000/episode_000000.mp4"
    video_path = source_root / video_relative
    video_path.write_bytes(b"prevalidated-video-fixture")
    (source_root / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "episode_000000",
                "episode_index": 0,
                "frame_count": 5,
                "parquet_path": parquet_relative,
                "video_paths": {"front": video_relative},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    identity_hash = "a" * 64
    progress_profile, stage_profile = _retrospective_profiles()
    axis = EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(0.0, 0.1, 0.2, 0.3, 0.4),
    )
    progress_binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=identity_hash,
        task_progress_profile_hash=progress_profile.semantic_content_hash,
        bound_clock_domain_id=axis.clock_domain_id,
        recording_scope="PROGRESS_STEP_WINDOW",
        entry_step_id="fixture_step",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    source_binding = RetrospectiveSourceBindingV1(
        episode_capture_identity_hash=identity_hash,
        source_dataset_base_fingerprint=source_dataset_base_fingerprint(
            source_root
        ),
        source_dataset_locator=str(source_root),
        source_schema="dexteleop_dataset_v2.3",
        source_batch_id="annotation-projection-fixture",
        source_episode_index=0,
        provenance_source_episode_index=1,
        source_info_hash=_file_sha256(info_path),
        source_episode_record_hash=source_episode_manifest_hash(source_root, 0),
        source_parquet_relative_path=parquet_relative,
        source_parquet_hash=_file_sha256(parquet_path),
        source_parquet_byte_count=parquet_path.stat().st_size,
        admitted_videos=(
            RetrospectiveVideoBindingV1(
                modality_id="front_rgb",
                source_relative_path=video_relative,
                content_hash=_file_sha256(video_path),
                byte_count=video_path.stat().st_size,
                declared_frame_count=axis.sample_count,
                decoded_frame_count=axis.sample_count,
            ),
        ),
    )
    window = RetrospectiveProgressWindowV1(
        episode_capture_identity_hash=identity_hash,
        source_binding_hash=source_binding.semantic_content_hash,
        task_progress_profile_hash=progress_profile.semantic_content_hash,
        progress_episode_binding_hash=progress_binding.semantic_content_hash,
        task_stage_profile_hash=stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        sample_count=axis.sample_count,
        clock_domain_id=axis.clock_domain_id,
        progress_step_id="fixture_step",
        attempt_id="retrospective:fixture:attempt:1",
        start_at=axis.cut_times[0],
        end_at=axis.cut_times[-1],
    )
    manifest = RetrospectiveProgressMigrationManifestV1(
        episode_capture_identity_hash=identity_hash,
        source_binding_hash=source_binding.semantic_content_hash,
        retrospective_window_hash=window.semantic_content_hash,
        task_progress_profile_hash=progress_profile.semantic_content_hash,
        progress_episode_binding_hash=progress_binding.semantic_content_hash,
        task_stage_profile_hash=stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        migration_actor_id="annotation-projection-fixture",
        migration_created_at=1.0,
        migration_clock_domain_id="annotation_clock",
    )
    workspace = tmp_path / "workspace"
    commit_retrospective_progress_episode(
        workspace_root=workspace,
        progress_profile=progress_profile,
        progress_binding=progress_binding,
        stage_profile=stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    return workspace, source_root


def test_annotation_projection_manifest_covers_current_import_closure() -> None:
    payload = _checker_payload()

    assert payload["manifest"] == str(_MANIFEST.relative_to(_REPOSITORY_ROOT))
    closure_modules = payload["closure_modules"]
    assert isinstance(closure_modules, list)
    assert "control_station.refinement_annotation.cli" in closure_modules
    assert set(closure_modules) == _runtime_closure_modules()
    assert {
        module_name.split(".", 1)[0]
        for module_name in closure_modules
    } == set(_LOCAL_PACKAGE_ROOTS)


def test_annotation_projection_is_self_contained_for_cli_and_readers(
    tmp_path: Path,
) -> None:
    payload = _checker_payload()
    projection_root = tmp_path / "projection"
    _materialize_projection(projection_root, payload)

    help_result = _run_projected_entrypoint(
        projection_root,
        ["refinement-annotation", "--help"],
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "Annotate ordered fine stages" in help_result.stdout

    gui_result = _run_projected_gui_smoke(projection_root)
    assert gui_result.returncode == 0, gui_result.stdout + gui_result.stderr

    native_root = tmp_path / "empty-native-dataset"
    native_root.mkdir()
    native_result = _run_projected_entrypoint(
        projection_root,
        ["refinement-annotation", str(native_root), "--dry-run"],
    )
    assert native_result.returncode == 0, (
        native_result.stdout + native_result.stderr
    )
    assert "eligible_finalized_episodes=0" in native_result.stdout

    workspace, source_root = _retrospective_workspace(
        tmp_path / "retrospective"
    )
    retrospective_result = _run_projected_entrypoint(
        projection_root,
        [
            "refinement-annotation",
            str(workspace),
            "--source-root",
            str(source_root),
            "--dry-run",
        ],
    )
    assert retrospective_result.returncode == 0, (
        retrospective_result.stdout + retrospective_result.stderr
    )
    assert "eligible_finalized_episodes=1" in retrospective_result.stdout
