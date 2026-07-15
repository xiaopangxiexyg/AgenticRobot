# Dataset Postprocess

DexTeleop dataset 后处理工具集合。默认以 dry-run / 写新目录为主，不原地修改原始数据集。
唯一受支持的包路径是 `utils.dataset_postprocess`。原顶层
`dataset_postprocess` 包已移除；脚本、测试和外部调用应统一使用新路径。

## 常用命令

使用带 `pyarrow` 的环境：

```bash
conda activate dexteleop310
```

## 推荐汇总 Pipeline

大批量数据建议按下面顺序处理。所有会改 dataset 结构的步骤都写到新目录，
默认不会原地覆盖原始数据。

```text
raw dataset
  -> trim-nan-tails / clean-stutters / delete-episodes / split-episodes
  -> label-force-collisions
  -> causal-align
  -> wrist table keep-mask
  -> wrist depth mask postprocess
  -> final wrist keep-mask = postprocessed depth OR table keep-mask
  -> dataset_interface training-time masked augmentation
```

默认 mask 行为：

```text
final_keep_mask = depth_range_mask_postprocessed OR table_geometry_keep_mask
final 合成阶段不额外添加 proxy/body mask
```

默认 table keep-mask 应只包含桌面；如果上游 table keep-mask 已经渲入 painted body mask，
final keep-mask 会按输入保留这部分像素。

不建议把 random background 离线扩成完整新 dataset。正式训练时读取
`final_keep_mask`，在 DataLoader worker 内在线替换背景；离线视频只用于抽样检查。

结构类工具已经按“大文件 hardlink、可变 meta/data copy”的方式派生数据集：
`trim-nan-tails`、`causal-align` 会复制 `meta/` 和 `data/`，视频/触觉/depth
等大文件 hardlink；`clean-stutters` 会重写 `data/`；`delete-episodes` 和
`split-episodes` 会优先 hardlink 保留的 artifact。输出目录不能放在源 dataset 内部。

## v2.1 固定 q6 到 DexTeleop v2.2

转换必须提供显式 `PolicyIOLayout` JSON；工具不会猜测 q6 padding 对应夹爪还是灵巧手。
无 `--apply` 时只做完整路径/episode/Parquet 预检并打印计划：

```bash
python -m utils.dataset_postprocess v21-to-v22 datasets/legacy_task \
  --output-root datasets/legacy_task_v22 \
  --layout-json /abs/path/policy_io_layout.json

python -m utils.dataset_postprocess v21-to-v22 datasets/legacy_task \
  --output-root datasets/legacy_task_v22 \
  --layout-json /abs/path/policy_io_layout.json \
  --apply
```

转换永不覆盖源目录或已存在的输出目录。输出的
`meta/v21_to_v22_conversion.json` 记录 source info SHA-256、conversion id、layout hash 和
全部 Parquet 相对路径；原 q6 列保留用于追溯，新的模型字段按 v2.2 layout 生成。

检测 `FAULT/ESTOP + action 全 NaN` 的 episode 尾段：

```bash
python -m utils.dataset_postprocess plan-fault-nan datasets/2026-07-05_vive_rm65_real_teleop
```

检测时间间隔过长的分段，默认阈值建议从 `0.2s` 开始：

```bash
python -m utils.dataset_postprocess plan-stutters datasets/2026-07-05_vive_rm65_real_teleop --max-gap-sec 0.2
```

按长 gap 切分 episode，生成没有内部长卡顿的新 dataset。输出会重建全局
`index`，并把每个 segment 的 `frame_index` 从 0 重新编号：

```bash
python -m utils.dataset_postprocess clean-stutters \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_init_clean_stutter_clean \
  --max-gap-sec 0.2 \
  --min-segment-frames 15 \
  --apply
```

删除指定 episode。默认只生成计划；真正执行必须加 `--apply`，并写到新目录：

```bash
python -m utils.dataset_postprocess delete-episodes \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --episodes 0,3,5-8 \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_clean \
  --apply
```

`--episodes` 使用 dataset 中的 `episode_index`，从 0 开始编号；例如 `episode_000000` 对应 `0`。选择器支持逗号和区间，例如 `1,3,5-8`。

自动裁掉旧数据中 `FAULT/ESTOP + action 全 NaN` 的尾段：

```bash
python -m utils.dataset_postprocess trim-nan-tails \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_trim_nan \
  --apply
```

把指定 episode 从原 dataset 中抽出为新 dataset，同时生成删除后的 clean dataset：

```bash
python -m utils.dataset_postprocess split-episodes \
  datasets/2026-07-05_vive_rm65_real_teleop \
  --episodes 1,17,28,30,43,48,61,62,91,100,105,143,146 \
  --clean-output-root datasets/2026-07-05_vive_rm65_real_teleop_no_collision \
  --extract-output-root datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --apply
```

抽取后的 dataset 保留原始 `episode_index`，方便回查原数据。

给已有 dataset 补 episode 级力/力矩 hard-stop 标签。默认只扫描；真正写入必须加 `--apply`：

```bash
python -m utils.dataset_postprocess label-force-collisions \
  datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --apply
```

如果某个 dataset 是人工抽出的碰撞/超限负样本集，可显式覆盖为全 True：

```bash
python -m utils.dataset_postprocess label-force-collisions \
  datasets/2026-07-05_vive_rm65_real_teleop_collision \
  --override all-detected \
  --apply
```

将相机和触觉引用改成因果对齐：每行选择同一 episode 内
`timestamp <= action timestamp` 的最近帧。输出会重建全局 `index`。默认只生成计划；真正执行写到新目录：

```bash
python -m utils.dataset_postprocess plan-causal-align \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean

python -m utils.dataset_postprocess causal-align \
  datasets/2026-07-05_vive_rm65_real_teleop_init_clean \
  --output-root datasets/2026-07-05_vive_rm65_real_teleop_init_clean_causal
```

如果同时做卡顿清理和因果对齐，建议先 `clean-stutters`，再对输出运行
`causal-align`，避免对齐跨过已切开的长 gap。

## 派生数据集组织与 mask metadata 迁移

`organize_episodes.py` 用 JSON manifest 从多个源 dataset 选择 episode，生成连续的
`episode_index`，并执行受限的 schema migration。源 dataset 保持不变；构建过程先写入
staging 目录，成功后才发布，不能把输出目录放在源 dataset 内。大文件默认使用 hardlink，
会被重写的 metadata/Parquet 文件写入新文件。已有 manifest 示例：

- `manifests/dexteleop_2026_07_sort.json`
- `manifests/bimanual_cube_new_success_wrist_mask_cleanup.json`
- `manifests/bimanual_cube_new_failure_case_wrist_mask_cleanup.json`

该工具目前作为 Python API 使用，可先规划再构建：

```python
from utils.dataset_postprocess.organize_episodes import build_datasets, plan_dataset_build

plan = plan_dataset_build("/abs/path/manifest.json", dataset_names=["example"])
result = build_datasets("/abs/path/manifest.json", dataset_names=["example"])
```

`relocate_mask_metadata.py` 用于目录迁移后重写 mask JSON/JSONL 中记录的绝对路径，
不会修改 bitpack 等二进制文件。默认是 dry-run；确认输出后才使用 `--apply`：

```bash
python -m utils.dataset_postprocess.relocate_mask_metadata \
  --root /abs/path/masks \
  --replace /old/masks=/new/masks

python -m utils.dataset_postprocess.relocate_mask_metadata \
  --root /abs/path/masks \
  --replace /old/masks=/new/masks \
  --apply
```

## 相机相关工具

RealSense 去畸变：

```bash
python -m utils.dataset_postprocess.real_undistort dataset \
  --dataset-root datasets/source_dataset \
  --output-root datasets/source_dataset_undistorted
```

腕部相机 mask / depth 后处理：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust --help
```

典型 wrist final mask 批处理：

先用 `wrist_cam_adjust dataset --output-kind keep-mask` 生成 `TABLE_MASK_ROOT`；
完整 table keep-mask 命令见 `utils/dataset_postprocess/wrist_cam_adjust/README.md`。

```bash
SESSION=datasets/2026-07-09_vive_rm65_real_teleop
TABLE_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/${SESSION##*/}_table_keep_mask
DEPTH_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess
FINAL_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask

python -m utils.dataset_postprocess.wrist_cam_adjust postprocess-depth-mask \
  --session "$SESSION" \
  --all-episodes \
  --output-root "$DEPTH_MASK_ROOT" \
  --jobs 4 \
  --no-preview \
  --overwrite

python -m utils.dataset_postprocess.wrist_cam_adjust compose-final-mask \
  --session "$SESSION" \
  --table-mask-root "$TABLE_MASK_ROOT" \
  --depth-mask-root "$DEPTH_MASK_ROOT" \
  --output-root "$FINAL_MASK_ROOT" \
  --jobs 4 \
  --overwrite
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
