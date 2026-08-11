# Writing 第一子任务 retrospective 三阶段标注

> 状态：dataset-local semantic layer、pilot campaign 回执和 GUI 入口已实现并通过本地合同测试。
> 正式发布必须使用 `--publish-dataset-local`；`--workspace-root/--source-root` 仅保留为兼容原型。

本功能只负责一个 `dexteleop_dataset_v2.3` 数据集的 writing 第一子任务后标注。程序不会启动
设备或机器人控制，不会改写 Parquet、视频、`meta/info.json` 或 `meta/episodes.jsonl`；
retrospective Progress、草稿和 immutable Stage annotation set 全部追加到该数据集自己的
named semantic layer。跨日期组合、抽象数据集、视角训练约束、split 和混训不属于本标注
入口，由上层 dataset workspace/compiler 处理。

## 1. 启动前检查

目标 v2.3 数据集应保持：

```text
<v2.3-dataset>/
  meta/info.json
  meta/episodes.jsonl
  data/...
  videos/...
  semantic_layers/writing_stage1_retrospective_v1/
    profile.cjson
    stages.cjson
    campaigns/pilot_v1/definition.cjson
    campaigns/pilot_v1/publication_receipt.cjson
    episodes/<episode_identity_hash>/...
    meta/progress/stage_annotation_drafts/...
    meta/progress/stage_annotations/...
```

先只读验证严格 479 + 240 cohort 和确定性 6 + 6 pilot：

```bash
python -m utils.dataset_postprocess writing-stage1-pilot \
  --front-manifest /absolute/path/to/front_479.jsonl \
  --three-view-manifest /absolute/path/to/three_view_323.jsonl \
  --three-view-provenance-root /absolute/path/to/0724_source_dataset \
  --per-batch 6
```

确认 JSON 报告中 `cohort_counts`、`pilot_count=12` 和每条 source/hash 后，发布到 7.23、7.24
各自的数据集内 layer。`migration-created-at` 默认自动生成并写入 campaign definition；同一
campaign 重跑会恢复，不需要手工填写“固定 Unix 时间戳”：

```bash
python -m utils.dataset_postprocess writing-stage1-pilot \
  --front-manifest /absolute/path/to/front_479.jsonl \
  --three-view-manifest /absolute/path/to/three_view_323.jsonl \
  --three-view-provenance-root /absolute/path/to/0724_source_dataset \
  --per-batch 6 \
  --publish-dataset-local \
  --semantic-layer-id writing_stage1_retrospective_v1 \
  --campaign-id pilot_v1 \
  --migration-actor-id writing-stage1-migration-v1
```

发布完成后，7.23 和 7.24 分别做不加锁 dry-run；每次只打开一个源数据集：

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/one_v23_dataset \
  --writing-v23-semantic-layer writing_stage1_retrospective_v1 \
  --dry-run
```

发布命令只创建 semantic layer 内容，不复制 raw Parquet/视频。7.23 和 7.24 分别标注；
两者的组合仍由后续上层 workspace 完成。

Dataset-local source binding 使用相对 owning-dataset 定位，不保存当前机器的绝对数据集目录；
因此整个 v2.3 数据集连同 `semantic_layers` 一起移动或更换挂载点后仍可加载。GUI 只显示已经
由有效 `definition.cjson + publication_receipt.cjson` 完整封存的 Episode；发布中断时已经写入
但尚未形成 receipt 的 Episode 不可见，重跑同一 campaign 后再整体进入标注列表。

发布前还会强制验证：7.23 恰好对应一个 front-only owner，7.24 恰好对应另一个 three-view
owner，每侧恰好选择 6 条；`semantic_layers` 和目标 layer 均不得是 symbolic link。

### Pilot 不是测试数据集

12 条 pilot 只是一组 annotation campaign selection：

```text
7.23 source dataset
  semantic layer
    eligible: 479
    pilot_v1: 6

7.24 source dataset
  semantic layer
    eligible: 240
    pilot_v1: 6
```

不会复制这 12 条的 Parquet/视频，不创建 `pilot_dataset`，也不提前把它们分配为训练的 test
split。pilot 通过后，后续 Episode 继续追加到相同 layer；这 12 条的 sealed annotations 原地
保留，并与后续标注采用相同的 Stage/Skill 合同。每个追加批次有独立 campaign receipt 和创建
时间，同一 campaign 重跑时才恢复其原 timestamp。

## 2. 启动普通一审

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/one_v23_dataset \
  --writing-v23-semantic-layer writing_stage1_retrospective_v1 \
  --annotator-id annotator_01
```

页面中应明确显示：

```text
RETROSPECTIVE_MIGRATED · PROGRESS_STEP_WINDOW · FULL TASK N/A
```

front-only Episode 只显示 front；three-view Episode 显示 front、left wrist、right wrist。
切换视角只改变观察画面，三个视角始终共享同一个 accepted-sample cursor、B1/B2 和草稿。

## 3. 每条 Episode 的操作

首次打开时，程序会把完整 Attempt 临时均分成三个可编辑 Stage。这只是 bootstrap draft，
不能直接当作正式边界：

1. 先播放到 Episode 尾，确认毛笔仍带笔盖、右手稳定持笔，并已送到左手可取盖位置；
2. 回到 front 视角寻找 B1：右夹爪开始有意闭合/接触并建立抓取的第一个正式 sample；
3. 将 `approach -> grasp` 边界移动到 B1；
4. 寻找 B2：抓取已经稳定，毛笔开始随右手离开支撑并进入提送运动的第一个正式 sample；
5. 将 `grasp -> lift_and_present` 边界移动到 B2；
6. three-view 数据可切 wrist 复核遮挡或接触，但不要建立第二套边界；
7. 检查三段顺序、正时长、无重叠，并完整覆盖 accepted sample axis；
8. 中途退出使用“保存草稿”；确认无误后使用“校验并封存”。

三个 Stage 都描述双臂协同：

- `right_acquire_brush.approach`：双臂进入协同区域，右手靠近带盖毛笔；
- `right_acquire_brush.grasp`：右夹爪建立稳定抓取，左臂可以继续协同移动；
- `right_acquire_brush.lift_and_present`：右手提起并送笔至左手可取盖位置。

若开头/结尾有无关复位动作，使用异常区间，不要移动 Stage 来吞入无关动作。若终点不满足
任务定义、B1/B2 因遮挡无法可靠判断，或视觉与低维信号明显冲突，应标记“需要二次复核”，
不要按三等分猜测后封存。

## 4. 二次复核

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/one_v23_dataset \
  --writing-v23-semantic-layer writing_stage1_retrospective_v1 \
  --annotator-id annotator_02 \
  --second-review
```

打开待复核 Episode 会基于上一版 immutable set 创建新的工作草稿；旧版本不会被覆盖。只有
重新检查并封存 `SECOND_REVIEW` set 后，后续 selection manifest 才能显式选择该复核版本。

## 5. 写入边界

标注程序允许写入：

```text
<v2.3-dataset>/semantic_layers/<layer-id>/meta/progress/stage_annotation_drafts/...
<v2.3-dataset>/semantic_layers/<layer-id>/meta/progress/stage_annotations/...
```

标注入口对 raw/capture 区域只读，只能向选定 semantic layer 写入。不要复制或 hardlink 原始
Parquet/视频。front-full、front-common240 和 three-view 等训练产品由上层 workspace 从同一
批 sealed annotation-set hash 派生，不由 GUI 分别组织或发布。
