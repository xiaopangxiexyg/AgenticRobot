# 0804 DexTeleop dataset loader

This is a portable, read-only example loader for the dataset documented in
[`DATASET_CONTENT_AND_HANDOFF.md`](../../docs/data/v25p2/DATASET_CONTENT_AND_HANDOFF.md):

```text
2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage
```

It does not import DexTeleop runtime packages and never connects to a robot or
modifies the dataset. It is an integration example for recipients, not a
replacement for the authoritative DexTeleop validators.

## 1. Supported reads

- `meta/info.json`, `modality_validity.json`, and `episodes.jsonl`;
- selected model-facing columns from each Episode Parquet;
- the RGB frame referenced by a formal sample and camera key;
- external `binary_stream_v1` tactile arrays;
- bit-packed wrist depth-range masks;
- canonical Progress profile, Stage profile, sealed Episode facts, and
  Stage annotation sets.

The loader rejects absolute artifact paths, missing files, paths or symlinks
escaping the dataset root, duplicate Episode indices, row-count disagreement,
unsupported tactile/depth formats, and Stage annotation hash-reference
disagreement.

When an Episode has more than one Stage annotation set, the loader raises
`SelectionRequiredError`. It never picks a directory based on name or mtime.

## 2. Environment

Python 3.10 or newer is recommended.

```bash
cd release/phase_dataset_loader
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`opencv-python-headless` is only needed for `load_rgb_frame`. Metadata,
canonical artifacts, Parquet, tactile, and depth loading do not initialize a
camera or other device.

## 3. Run the example

```bash
python examples/load_episode.py \
  "/absolute/path/to/2026-08-04_dmtac_rm65_bimanual_position_v2_writing_4_stage" \
  --episode 0 \
  --frame 0
```

For an Episode with multiple annotation sets, first inspect the IDs printed by
the script and then choose one explicitly:

```bash
python examples/load_episode.py DATASET_ROOT \
  --episode 123 \
  --frame 10 \
  --annotation-set-id stage-EXPLICIT-ID
```

Use `--skip-rgb`, `--skip-tactile`, or `--skip-depth` when a reduced-modality
release intentionally omits that data. Skipping a read does not make the
missing modality training-valid.

## 4. Python API

```python
from dexteleop_dataset_loader import DexTeleopReleaseDataset

dataset = DexTeleopReleaseDataset(DATASET_ROOT)
print(dataset.summary())

episode = dataset.episode(0)
table = dataset.load_parquet(0)  # only DEFAULT_MODEL_COLUMNS
action = table.column("action")[0].as_py()

front_rgb = dataset.load_rgb_frame(0, "front", 0)  # RGB uint8 HWC
left_tactile = dataset.load_tactile_frame(0, "left", 0)
left_depth_mask = dataset.load_wrist_depth_mask(0, "left_wrist", 0)

progress = dataset.load_progress_episode(0)
annotation_ids = dataset.stage_annotation_set_ids(0)
annotation = dataset.load_stage_annotation(
    0,
    annotation_set_id=annotation_ids[0],  # choose from a release profile
)
```

`load_tactile_frame` and `load_wrist_depth_mask` return `None` when the formal
sample has no mapped frame. They do not repeat a previous value.

Pass `columns=None` to `load_parquet` only when all audit columns are truly
needed. The physical Parquet has hundreds of audit/diagnostic columns; they are
not all model inputs.

## 5. Validation boundary

This bundle performs strict structural reads and canonical-domain checks. It
does not reimplement every DexTeleop semantic hash, timeline, camera-role,
training eligibility, or prepared-view rule. Before training, the dataset
owner still needs to provide:

- the authoritative `validate-v23` result;
- the selected annotation-set IDs or prepared-view manifest;
- success/aborted/failure and abnormal exclusions;
- required camera/tactile constraints;
- policy-I/O, pose-frame, causal snapshot, Progress, and Stage hashes.

Do not turn loader success into an assertion that every Episode or modality is
training eligible.

## 6. Test the bundle

From this directory, with the dependencies installed:

```bash
python -m pytest -q tests
```

The tests build a synthetic two-frame dataset. They never read the real 0804
dataset and never access hardware.
