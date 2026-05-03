# RawHand Tactile Representation Batch Plan

## 1. 目标与约束

目标不是直接找“最强策略”，而是先找 **对动作有用、又能被稳定预测** 的触觉中间表示，再做小范围迁移验证。

这套仓库里最合理的搜索顺序是：

1. `GTMed` 先测表示上限。
2. `GenMed` 再测表示可预测性。
3. `evaluate_gtmed_pipeline.py` 最后测 “预测 med 是否还能保住动作收益”。
4. `FullTrain` 只做最终确认，不拿来做第一轮筛选。

单次训练不超过 `2 epoch`。在当前数据量下，**第一轮必须做 sample-capped screening**，否则批量实验会非常慢。

## 2. 我对当前框架的理解

- 统一入口是 `Training/train.py`，通过 Hydra 组装 `data / model / train / experiment`。
- 数据入口对 `RawHandPointData` 走 `DeskTaskTac_MetaManifest -> DeskTaskTacSequenceDataset`。
- `RawHandPointData` 现在是软链接到 `/media/zc/新加卷/HumanHandData/DeskTaskTacRawHand_spatial_tactile`。
- 数据集规模：`745` 个 episode，`1,018,462` 帧，`24` 个 task，平均 episode 长度约 `1367` 帧。
- 当前默认 sample 组成：
  - `state`: 4-step history
  - `action`: 8-step future chunk
  - `video`: `video.top_view` 的 4-frame clip
  - `med`: 由 `data.raw.sequence.med.*` 和 `data.med.provider_type` 决定
- `vector med` 会被 `FutureSequence_Med` 自动展平成一维。
- `point-cloud med` 默认保持结构化 shape，不强制展平；当前右手点云默认是 `[T, 137, 4] = (x,y,z,value)`。
- `GTMed / GenMed / FullTrain` 共用同一套 `med_spec`，所以表示比较应尽量在同一数据预算、同一 action 设置下进行。

## 3. 当前数据与已有结果给出的直接结论

- 默认 `RawHandPointData` 配置下，短窗口 `future[1,4]`、`future[1,8]`、`future[4,8]` 的**可用样本数几乎相同**，因为样本有效性主要被 `action_chunk=8, action_stride=3` 限制，而不是短 med window 限制。
- 当前默认长窗口 `future_start=1, future_end=40, stride=5` 会得到 8 个 med step，样本数只比短窗口略少。
- 已有 `rawhand_explainable_point_cloud` 结果表明：
  - `GTMed(point cloud)` 比 `Vanilla` 有提升，但幅度不大。
  - `GenMed(point cloud) -> GTMed` 的 pipeline 相对 oracle `GTMed` 的动作 RMSE 劣化约 `1.33%`，说明点云 med **至少是可预测且可用的**。
- 已有 `rawhand_explainable_vector` suite 训练失败，但我做了最小 config/forward 检查，**vector 路径本身可以正常 compose 和前向**；所以这更像运行时或环境问题，不像设计层面不可行。

## 4. 推荐的批量实验框架

### Phase S0: Smoke

目的：先把 `vector / point cloud / pointnet / point_query` 四条关键链路都打通。

- 预算：`1 epoch`
- 建议数据预算：
  - `max_episodes_per_dataset=2`
  - `max_frames_per_episode=4`
  - `max_total_samples=24`
  - `batch_size=2`
- 必跑：
  - `Vanilla + minimal_mlp`
  - `GTMed + vector + mlp projector`
  - `GTMed + point_cloud + mlp projector`
  - `GTMed + point_cloud + pointnet projector`
  - `GenMed + vector + mlp predictor`
  - `GenMed + point_cloud + mlp predictor`
  - `GenMed + point_cloud + point_query predictor`

如果 S0 没全绿，不进入下一阶段。

### Phase S1: Representation Screening

目的：先只看 “这个 med 对动作到底有没有帮助”，因此主指标用 `GTMed vs Vanilla`，不用 `FullTrain`。

- 预算：`2 epoch`
- backbone：`minimal_mlp`
- `task_enabled=false`
- 建议数据预算：
  - `max_episodes_per_dataset=64`
  - `max_frames_per_episode=16`
  - `max_total_samples=1024`
  - `batch_size=64`
  - `train.validation_ratio=1.0`

建议筛这 6 个候选表示：

| id | med | window | branch |
| --- | --- | --- | --- |
| `vec_f1_4` | vector | `future[1,4], stride=1` | mlp |
| `vec_f1_8` | vector | `future[1,8], stride=1` | mlp |
| `pc_f1_4_mlp` | point cloud | `future[1,4], stride=1` | mlp projector |
| `pc_f1_8_mlp` | point cloud | `future[1,8], stride=1` | mlp projector |
| `pc_f1_4_pn` | point cloud | `future[1,4], stride=1` | pointnet projector |
| `pc_f1_8_pn` | point cloud | `future[1,8], stride=1` | pointnet projector |

可选加一个 tie-breaker：

- `pc_f1_40_s5_pn`: `future[1,40], stride=5`，只在 `pc_f1_4` 和 `pc_f1_8` 很接近时再跑。

这一阶段只需要：

- 1 个共享 `Vanilla-MLP` baseline
- 6 个 `GTMed-MLP`

筛选标准：

1. 主指标：`evaluation_val_action_rmse`
2. 次指标：`best_val_action_loss`
3. 决策规则：优先看相对 `Vanilla` 的提升幅度，不看绝对最小 loss 一项

## 5. 第二轮验证

### Phase S2: Predictability Validation

目的：在 S1 的前 2 名表示上，验证 `GenMed` 是否能把它们预测出来。

- 预算：`2 epoch`
- 仍然用 `minimal_mlp`
- 数据预算与 S1 保持一致

建议只跑：

- `GenMed + top1`
- `GenMed + top2`

如果 top 候选是 point cloud，再额外比较：

- `mlp predictor`
- `point_query predictor`

这一阶段主看：

1. `evaluation_val_med_rmse`
2. 单样本点云/向量可视化是否合理
3. 训练是否稳定，`best` 和 `final` 不要漂太大

### Phase S3: GTMed Pipeline Validation

目的：验证 “预测 med” 是否还能保住 `GTMed` 的动作收益。这是最关键的部署前判断。

对 S2 每个候选都跑：

- `GTMed ground-truth med`
- `GenMed -> GTMed predicted med`

用 `TestScripts/evaluate_gtmed_pipeline.py`，主看：

1. `action.rmse` 的相对劣化
2. `action.mae` 的相对劣化

建议阈值：

- `predicted-med` 相对 `ground-truth med` 的动作 RMSE 劣化 `< 5%`
- 如果 `< 2%`，可以认为这个表示已经很扎实

## 6. 最终确认

### Phase S4: Transfer Check

目的：确认结论不是 `minimal_mlp` 的偶然产物。

- backbone：`basic_vla`
- `task_enabled=true`
- 预算：`2 epoch`
- 建议数据预算：
  - `max_episodes_per_dataset=128`
  - `max_frames_per_episode=16`
  - `max_total_samples=2048` 或 `4096`
  - `batch_size=32`

只跑：

- 1 个 `Vanilla-VLA`
- 2 个 `GTMed-VLA`（S1/S3 的前两名表示）

如果 winner 在 `VLA` 上仍稳定优于 `Vanilla-VLA`，这个表示基本可以定下来。

### Phase S5: FullTrain Confirmation

只对最终 winner 跑一次：

- `FullTrain + winner representation`
- `2 epoch`

这里不拿来选 winner，只看：

- `FullTrain` 是否明显优于 `Vanilla`
- `FullTrain` 是否至少逼近 `GTMed`，而不是离 `Vanilla` 更近

## 7. 统一评测规则

为了让批量结果真的可比，建议统一遵守下面 5 条：

1. **最终比较不要再用 `max_batches=4`。**
2. **最终比较时把 `train.validation_ratio` 设成 `1.0`。**
3. `Vanilla / GTMed / GenMed` 必须共用完全一致的数据预算。
4. point cloud 默认保持 `data.med.flatten_sequence=auto`，不要手动改成展平版再去比 `pointnet`。
5. 所有最终排名都以 `evaluation_*` 指标为准，不以训练期 batch-average loss 为准。

## 8. 推荐的最终决策逻辑

最终 winner 不是 “某个 loss 最低” 就算，而应同时满足：

1. `GTMed` 相对 `Vanilla` 有稳定收益。
2. `GenMed` 能稳定预测该表示。
3. `GTMed pipeline` 的动作退化足够小。
4. 在 `basic_vla` 上结论不反转。

如果出现下面两种情况，优先级建议如下：

- `point cloud` 上限更高，但 pipeline 退化明显：优先选 `vector`
- `point cloud` 和 `vector` 收益接近，但 `pointnet` 明显优于 `mlp projector`：优先保留 `point cloud`

## 9. 执行建议

- 批量执行优先用 `Scripts/run_experiment_suite.py`，不要手写几十条训练命令。
- 建议按阶段拆成 3 个 suite：
  - `rawhand_rep_smoke`
  - `rawhand_rep_screen`
  - `rawhand_rep_validate`
- 先用单 seed `7` 做 S1/S2，最后只对前 2 名补 `seed=11` 做稳定性复核。
- 如果当前 Hydra 环境对新字段是 struct 模式，`data.raw.limits.*` 可能需要写成 `+data.raw.limits.*`。
