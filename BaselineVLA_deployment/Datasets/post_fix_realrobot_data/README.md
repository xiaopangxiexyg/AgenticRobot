# Paper-Fold Edge Stuck QC

This utility checks the paper-fold LeRobot datasets for still segments at the
beginning or end of each episode. It does not modify the raw parquet or video
files. It writes per-episode trim recommendations and random visual QC images.

Default rule:

- Pose source: right arm end-effector pose, `observation.right_arm.pose`.
- A boundary is considered still while both conditions hold relative to the
  first or last pose: `xyz <= 0.01 m` and `quat angle <= 2 deg`.
- A still boundary is recommended for trimming only when it lasts at least
  `0.5 s`.
- The script keeps at least `2.0 s` of each episode.

Run from the repository root:

```bash
/home/zc/anaconda3/envs/HumanData/bin/python \
  BaselineVLA_deployment/Datasets/post_fix_realrobot_data/paper_fold_edge_stuck_qc.py \
  --dataset-root BaselineVLA_deployment/Data/RawData/RealRobot/paper_fold \
  --sample-count 10 \
  --sample-seed 17
```

Main outputs are written under:

```text
BaselineVLA_deployment/Datasets/post_fix_realrobot_data/outputs/paper_fold_edge_stuck_qc/
```

Files:

- `paper_fold_edge_trim_report.csv`: per-episode trim seconds and keep frame
  range.
- `paper_fold_edge_trim_report.jsonl`: same data in JSONL.
- `summary.json`: totals by dataset.
- `visualizations/*.png`: random visual checks with right-arm xyz curves,
  trim spans, and sampled camera frames.

Export clipped copies without modifying the source datasets:

```bash
/home/zc/anaconda3/envs/HumanData/bin/python \
  BaselineVLA_deployment/Datasets/post_fix_realrobot_data/export_paper_fold_clipped.py \
  --dataset-root BaselineVLA_deployment/Data/RawData/RealRobot/paper_fold
```

This writes a separate root directory named `paper_fold_clipped` next to
`paper_fold`, with child datasets named `<source_dataset_dir>_clipped`. Parquet
rows and front-camera videos are clipped to the recommended keep ranges.
Episode indices are preserved; per-episode `frame_index`, global `index`, and
`observation.images.front.video_frame_index` are reset for the clipped copy.

The clipped-only training data config is:

```text
BaselineVLA_deployment/Configs/data/realrobot_paper_fold_clipped.yaml
```

The original `realrobot_paper_fold.yaml` discovery glob is kept on the
non-clipped directories so it does not accidentally mix original and clipped
copies after export.

Append 8 synthetic return-motion frames per episode:

```bash
/home/zc/anaconda3/envs/HumanData/bin/python \
  BaselineVLA_deployment/Datasets/post_fix_realrobot_data/export_paper_fold_return8.py \
  --source-root BaselineVLA_deployment/Data/RawData/RealRobot/paper_fold_clipped
```

The default output root is `Data/RawData/RealRobot/paper_fold_clipped_return8`.
Each source episode is assigned a random target start pose within the same
subset via a deranged permutation, so every target episode is selected exactly
once. The generated 8 frames follow the first 8 samples of a virtual
end-to-start interpolation whose translation and rotation speeds do not exceed
the source episode's historical maximum. Non-motion fields are filled from the
last two source frames; front video repeats the last source frame.

The return8-only training data config is:

```text
BaselineVLA_deployment/Configs/data/realrobot_paper_fold_clipped_return8.yaml
```
