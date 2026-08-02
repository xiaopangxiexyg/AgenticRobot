# Wrist RGB media incident, 2026-07-30

## Status

The following remote source datasets are blocked for corrected wrist-view
training until their wrist RGB videos are restored:

- `/home/user/zc/DexTeleop/datasets/sorted_v22/new_cube_front_ready`
- `/home/user/zc/DexTeleop/datasets/writing_phase1_grap_pen`

Front-camera videos were not selected for rewriting and are not implicated by
this incident.

## What happened

A one-episode geometry-mask diagnostic populated unselected output videos with
hardlinks to source videos. A later full run used `overwrite=True` and opened
those existing output paths directly. Because source and output shared an
inode, OpenCV truncated and rewrote the source wrist RGB video as well.

Observed remote evidence:

- `new_cube_front_ready`: 1,450 of 1,452 selected wrist videos decode 2 frames
  instead of their episode-declared frame counts;
- `writing_phase1_grap_pen`: 1,004 of 1,006 selected wrist videos decode 2
  frames instead of their episode-declared frame counts;
- affected source and output paths had the same inode, and their modification
  time matched the mask-generation run;
- a search across 76 available dataset metadata trees, including Trash and
  known pre-merge backups, found complete alternate wrist RGB for episode 0
  only. The backups share the affected hardlinked inodes.

The invalid generated table-mask roots were retained but moved to:

- `datasets/masks/table_keep_mask/new_cube_front_ready.invalid-hardlink-write-through-20260730`
- `datasets/masks/table_keep_mask/writing_phase1_grap_pen.invalid-hardlink-write-through-20260730`

## Remediation

Video-transform writers now:

1. write to a fresh temporary inode in the destination directory;
2. require the decoded frame count to equal the episode contract;
3. remove the temporary output on failure;
4. atomically replace the destination only after validation.

This applies to both wrist geometry-mask generation and real-image
undistortion, which had the same partial-selection/hardlink hazard.

`dataset_interface.media_integrity` now validates each selected camera video's
declared frame count and final-frame decoding. OpenPI configurations with the
`-corrected-v1` suffix enable this check before parquet loading or the first
training step. Frozen historical configuration names retain their old lazy
media behavior and checkpoint identity.

## Recovery requirement

The original wrist RGB content cannot be reconstructed from the depth-only raw
streams or the two-frame rewritten files. Restore these two dataset roots from
an independent copy or recollect them, then run the strict media validator
before generating table/final masks.
