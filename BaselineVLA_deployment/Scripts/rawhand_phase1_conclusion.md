# RawHand Phase 1 结论

## 结论

S1 的设置是固定 `minimal_mlp + seed=7 + 5 epoch + 1024 train / 1024 val`。在这个预算下，**没有任何触觉中间表示打赢 Vanilla baseline**。

可采纳的排序已经足够清楚：

| rank | candidate | val action RMSE | vs Vanilla |
| --- | --- | ---: | ---: |
| 1 | `pc_f1_8_pn` | `0.027383` | `+0.36%` |
| 2 | `vec_f1_8` | `0.027481` | `+0.73%` |
| 3 | `pc_f1_4_mlp` | `0.027587` | `+1.11%` |
| baseline | `vanilla` | `0.027283` | `0%` |

其中 `Vanilla` 的 `0.027283` 来自 `metrics.json` 的 `best_val_action_loss=0.000744377` 开方。之所以没有直接用 suite 里的评测值，是因为 `evaluate_checkpoint.py --split val` 在 `vanilla` 路径上会触发 `Training/VanillaTrain/build.py:33` 的 `KeyError: 'train'`。

## 方向判断

当前最重要的不是继续铺开搜索，而是立即收敛：

- **主线保留**：`point cloud + future[1,8] + PointNet`，也就是 `pc_f1_8_pn`
- **对照保留**：`vector + future[1,8] + MLP`，也就是 `vec_f1_8`
- **停止推进**：`future[1,4]` 和 point-cloud 的 `mlp projector`

原因很直接：

1. 触觉 med 还没有在小预算下打赢 `vanilla`，所以 Phase 2 不应继续做大范围“找更优表示”，而应转成“验证最强候选能否被稳定预测”。
2. 在已完成结果里，`pc_f1_8_pn` 是最接近 `vanilla` 的触觉方案，也是唯一值得继续作为主线验证的点云表示。
3. `vec_f1_8` 虽然略弱，但它是最合理的紧凑对照组；如果后续 pipeline 更稳、退化更小，它仍有可能成为更可部署的方案。
4. 三个未完成 run 当前没有救的必要：
   - `pc_f1_4_pn` 最新只到 `epoch 2`，`val RMSE=0.028845`
   - `pc_f1_8_mlp` 最新只到 `epoch 2`，`val RMSE=0.031140`
   - `vec_f1_4` 最新只到 `epoch 1`，`val RMSE=0.042345`

## 解读注意项

- 在这份结论对应的原始分析时，`Vanilla` 的单独 `val` 评测存在脚本问题；这个问题现在已经修复，但不影响这里对当时结果的判断。
- 在这份结论对应的原始分析时，`Training/engine.py` 保存 `best.pt` 时没有深拷贝 `state_dict()`；这个问题现在也已经修复。S1 的绝对数值仍应谨慎解读，但**排序仍然可以用来收敛方向**。

## Phase 2 目标

Phase 2 不再问 “哪个 med 上限更高”，而是问：

1. `GenMed` 能否把 `pc_f1_8` 和 `vec_f1_8` 预测出来？
2. 预测出来的 med 进入 `GTMed` 后，动作 RMSE 会退化多少？

建议只跑两条：

- `GenMed + vec_f1_8 + mlp predictor`
- `GenMed + pc_f1_8 + point_query predictor`

并统一使用更稳的单进程 loader，避免再次因为资源问题卡死。

后续新任务默认统一收紧为 `2 epoch`，因为从实际训练曲线看，`2 epoch` 已经足够做方向判断。

## Phase 2 运行命令

下面命令默认在仓库根目录 `/home/zc/HandMotion/ManipRep` 执行。

### 1. 训练 `GenMed + vec_f1_8`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=genmed \
  model=minimal_mlp \
  project.seed=7 \
  model.task.enabled=false \
  train.epochs=2 \
  train.device=cuda \
  train.wandb.enabled=true \
  train.wandb.mode=offline \
  train.validation_ratio=1.0 \
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData] \
  data.raw.filters.dataset_name=RawHandPointData \
  data.raw.loader.batch_size=32 \
  data.raw.loader.num_workers=0 \
  data.raw.loader.prefetch_factor=null \
  data.raw.loader.persistent_workers=false \
  data.raw.loader.val.num_workers=0 \
  data.raw.loader.val.prefetch_factor=null \
  data.raw.loader.val.persistent_workers=false \
  ++data.raw.limits.max_datasets=1 \
  ++data.raw.limits.max_episodes_per_dataset=64 \
  ++data.raw.limits.max_frames_per_episode=16 \
  ++data.raw.limits.max_total_samples=1024 \
  data.raw.sequence.med.kind=vector \
  data.raw.sequence.med.side=right \
  data.raw.sequence.med.intermediate_key=future_right_tactile \
  data.raw.sequence.med.future_start=1 \
  data.raw.sequence.med.future_end=8 \
  data.raw.sequence.med.stride=1 \
  data.raw.sequence.med.vector.field=tactile.right_tacglove \
  data.med.flatten_sequence=auto \
  model.med_branch.predictor.type=mlp \
  train.wandb.group=rawhand_phase_s2_predictability \
  train.run_name=rawhand_phase_s2_genmed_vec_f1_8_seed7
```

### 2. 训练 `GenMed + pc_f1_8 + point_query`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=genmed \
  model=minimal_mlp \
  project.seed=7 \
  model.task.enabled=false \
  train.epochs=2 \
  train.device=cuda \
  train.wandb.enabled=true \
  train.wandb.mode=offline \
  train.validation_ratio=1.0 \
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData] \
  data.raw.filters.dataset_name=RawHandPointData \
  data.raw.loader.batch_size=32 \
  data.raw.loader.num_workers=0 \
  data.raw.loader.prefetch_factor=null \
  data.raw.loader.persistent_workers=false \
  data.raw.loader.val.num_workers=0 \
  data.raw.loader.val.prefetch_factor=null \
  data.raw.loader.val.persistent_workers=false \
  ++data.raw.limits.max_datasets=1 \
  ++data.raw.limits.max_episodes_per_dataset=64 \
  ++data.raw.limits.max_frames_per_episode=16 \
  ++data.raw.limits.max_total_samples=1024 \
  data.raw.sequence.med.kind=point_cloud \
  data.raw.sequence.med.side=right \
  data.raw.sequence.med.intermediate_key=future_right_point_cloud \
  data.raw.sequence.med.future_start=1 \
  data.raw.sequence.med.future_end=8 \
  data.raw.sequence.med.stride=1 \
  data.med.flatten_sequence=auto \
  model.med_branch.predictor.type=point_query \
  train.wandb.group=rawhand_phase_s2_predictability \
  train.run_name=rawhand_phase_s2_genmed_pc_f1_8_pq_seed7
```

### 3. 评估两个 `GenMed` checkpoint

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/genmed/rawhand_phase_s2_genmed_vec_f1_8_seed7 \
  --checkpoint-kind best \
  --split val \
  --loader-mode safe \
  --batch-size 32 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/genmed/rawhand_phase_s2_genmed_pc_f1_8_pq_seed7 \
  --checkpoint-kind best \
  --split val \
  --loader-mode safe \
  --batch-size 32 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

### 4. 跑 `GTMed pipeline` 验证

`vec_f1_8`:

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_gtmed_pipeline.py \
  Outputs/gtmed/rawhand_phase_s1_screening_rawhand_point_gtmed_mlp_gtmed_vec_f1_8_seed7 \
  --genmed-checkpoint Outputs/genmed/rawhand_phase_s2_genmed_vec_f1_8_seed7 \
  --gtmed-checkpoint-kind best \
  --genmed-checkpoint-kind best \
  --split val \
  --pipeline-mode cached \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda \
  --genmed-device cpu
```

`pc_f1_8_pn`:

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_gtmed_pipeline.py \
  Outputs/gtmed/rawhand_phase_s1_screening_rawhand_point_gtmed_mlp_gtmed_pc_f1_8_pn_seed7 \
  --genmed-checkpoint Outputs/genmed/rawhand_phase_s2_genmed_pc_f1_8_pq_seed7 \
  --gtmed-checkpoint-kind best \
  --genmed-checkpoint-kind best \
  --split val \
  --pipeline-mode cached \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda \
  --genmed-device cpu
```

## Phase 2 的判定标准

- 先看 `GenMed` 的 `evaluation_val_med_rmse`
- 再看 pipeline 里 `gtmed_with_genmed_predicted_med` 相对 `gtmed_ground_truth_med` 的 `action.rmse` 劣化

建议阈值：

- 劣化 `< 5%`：可以进入 Phase 3 / Phase 4
- 劣化 `< 2%`：说明这个中间表示已经很稳
- 劣化 `> 5%`：说明这个表示虽然 oracle 有用，但可预测性还不够
