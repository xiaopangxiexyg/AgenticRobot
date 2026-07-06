# Dataset Postprocess

DexTeleop dataset 后处理工具集合。默认以 dry-run / 写新目录为主，不原地修改原始数据集。

## 常用命令

使用带 `pyarrow` 的环境：

```bash
conda activate dexteleop310
```

检测 `FAULT/ESTOP + action 全 NaN` 的 episode 尾段：

```bash
python -m dataset_postprocess plan-fault-nan datasets/2026-07-05_vive_rm65_real_teleop
```

检测时间间隔过长的分段，默认阈值建议从 `0.2s` 开始：

```bash
python -m dataset_postprocess plan-stutters datasets/2026-07-05_vive_rm65_real_teleop --max-gap-sec 0.2
```

按长 gap 切分 episode，生成没有内部长卡顿的新 dataset。输出会重建全局
`index`，并把每个 segment 的 `frame_index` 从 0 重新编号：

```bash
python -m dataset_postprocess clean-stutters \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_init_clean_stutter_clean \
  --max-gap-sec 0.2 \
  --min-segment-frames 15 \
  --apply
```

删除指定 episode。默认只生成计划；真正执行必须加 `--apply`，并写到新目录：

```bash
python -m dataset_postprocess delete-episodes \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --episodes 0,3,5-8 \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_clean \
  --apply
```

`--episodes` 使用 dataset 中的 `episode_index`，从 0 开始编号；例如 `episode_000000` 对应 `0`。选择器支持逗号和区间，例如 `1,3,5-8`。

自动裁掉旧数据中 `FAULT/ESTOP + action 全 NaN` 的尾段：

```bash
python -m dataset_postprocess trim-nan-tails \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_trim_nan \
  --apply
```

把指定 episode 从原 dataset 中抽出为新 dataset，同时生成删除后的 clean dataset：

```bash
python -m dataset_postprocess split-episodes \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --episodes 1,17,28,30,43,48,61,62,91,100,105,143,146 \
  --clean-output-root datasets/2026-07-05_vive_rm65_real_teleop_no_collision \
  --extract-output-root datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --apply
```

抽取后的 dataset 保留原始 `episode_index`，方便回查原数据。

给已有 dataset 补 episode 级力/力矩 hard-stop 标签。默认只扫描；真正写入必须加 `--apply`：

```bash
python -m dataset_postprocess label-force-collisions \
  datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --apply
```

如果某个 dataset 是人工抽出的碰撞/超限负样本集，可显式覆盖为全 True：

```bash
python -m dataset_postprocess label-force-collisions \
  datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --override all-detected \
  --apply
```

将相机和触觉引用改成因果对齐：每行选择同一 episode 内
`timestamp <= action timestamp` 的最近帧。输出会重建全局 `index`。默认只生成计划；真正执行写到新目录：

```bash
python -m dataset_postprocess plan-causal-align \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean

python -m dataset_postprocess causal-align \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_init_clean_causal
```

如果同时做卡顿清理和因果对齐，建议先 `clean-stutters`，再对输出运行
`causal-align`，避免对齐跨过已切开的长 gap。

## 相机相关工具

RealSense 去畸变：

```bash
python -m dataset_postprocess.real_undistort dataset \
  --dataset-root datasets/source_dataset \
  --output-root datasets/source_dataset_undistorted
```

腕部相机 mask / depth 后处理：

```bash
python -m dataset_postprocess.wrist_cam_adjust --help
```

## 目录说明

- `plan-fault-nan`：找出故障后继续录制产生的 NaN 尾段。
- `trim-nan-tails`：复制 dataset，并自动裁掉故障后的 NaN 尾段。
- `plan-stutters`：按 timestamp gap 规划分段。
- `clean-stutters`：按 timestamp gap 切分 episode，并重建输出 dataset 的 `index`。
- `delete-episodes`：复制式删除指定 episode。
- `split-episodes`：同时生成 clean dataset 和抽取 dataset。
- `label-force-collisions`：从 parquet 诊断文本聚合 episode 级力/力矩 hard-stop 标签。
- `plan-causal-align` / `causal-align`：把相机和触觉索引改成 latest <= sample timestamp。
- `real_undistort/`：RealSense 视频去畸变。
- `wrist_cam_adjust/`：腕部相机 mask、depth 可视化和后处理。
