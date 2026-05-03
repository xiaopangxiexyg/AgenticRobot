# RawHand Phase 2 GenMed-VLA Commands

## 修复状态

在开始 Phase 2 之前，当前仓库里两个会影响评测口径的问题已经处理完：

1. `Vanilla --split val` 评测 bug 已修复  
   修复位置：[Training/VanillaTrain/build.py](/home/zc/HandMotion/ManipRep/Training/VanillaTrain/build.py)
   原因是这里把模型输入维度硬编码成了 `data_summary["train"]`，导致 `evaluate_checkpoint.py --split val` 只构建 `val` 时会报 `KeyError: 'train'`。

2. `best.pt` 不是严格 best-epoch 快照的问题已修复  
   修复位置：[Training/engine.py](/home/zc/HandMotion/ManipRep/Training/engine.py)
   现在新的 run 会把当时的 `model/optimizer state_dict` 深拷贝成 checkpoint 快照，而不是继续引用会被后续训练更新的张量。

注意：

- 这两个修复只对**之后新跑的实验**生效。
- 你之前已经跑完的 `GTMed-VLA` 老结果不会自动变成“正确的 best checkpoint”。

因此，**Phase 2 开始前，建议先重跑两条 GTMed-VLA oracle 作为新基线**。

## Phase 2 设计原则

本轮 Phase 2 不再回到 `minimal_mlp`，而是直接切到已经验证过有效的 **VLA 主线**。

为了不把变量重新搅混，建议仍然保持：

- `basic_vla`
- `model.task.enabled=false`
- `2 epoch`
- `1024 train / 1024 val`
- `RawHandPointData`

原因：

- 你上一轮证明的是 “更强 backbone 可以让 med 开始产生净收益”
- 这一步应该继续验证 “这些 med 是否能被预测、以及预测后还能否保住动作收益”
- 如果现在把 `task text` 重新打开，就又把变量混回去了

## 推荐执行顺序

按这个顺序跑：

1. 重跑 `GTMed-VLA` oracle 基线
2. 训练 `GenMed-VLA`
3. 评估 `GenMed-VLA`
4. 跑 `GTMed-VLA pipeline`

## 统一预算

下面命令统一采用：

- `seed=7`
- `train.epochs=2`
- `train.validation_ratio=1.0`
- `++data.raw.limits.max_datasets=1`
- `++data.raw.limits.max_episodes_per_dataset=64`
- `++data.raw.limits.max_frames_per_episode=16`
- `++data.raw.limits.max_total_samples=1024`
- `data.raw.loader.num_workers=0`
- `data.raw.loader.val.num_workers=0`

我这里把训练 `batch_size` 设成 `16`，是为了优先保证 `GenMed-VLA` 的稳定性。  
如果你确认显存和内存都够，再统一改成 `32`。

## A. 先重跑 GTMed-VLA Oracle

这一步不是重复劳动，而是为了让后面的 pipeline 使用**修复后的** `best.pt`。

### A1. GTMed-VLA + `vec_f1_8`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=gtmed \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=false \
  train.epochs=2 \
  train.device=cuda \
  train.wandb.enabled=true \
  train.wandb.mode=offline \
  train.validation_ratio=1.0 \
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData] \
  data.raw.filters.dataset_name=RawHandPointData \
  data.raw.loader.batch_size=16 \
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
  model.med_branch.projector.type=mlp \
  train.wandb.group=rawhand_phase2_vla_oracle \
  train.run_name=rawhand_phase2_vla_oracle_gtmed_vec_f1_8_seed7
```

### A2. GTMed-VLA + `pc_f1_8_pn`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=gtmed \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=false \
  train.epochs=2 \
  train.device=cuda \
  train.wandb.enabled=true \
  train.wandb.mode=offline \
  train.validation_ratio=1.0 \
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData] \
  data.raw.filters.dataset_name=RawHandPointData \
  data.raw.loader.batch_size=16 \
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
  model.med_branch.projector.type=pointnet \
  train.wandb.group=rawhand_phase2_vla_oracle \
  train.run_name=rawhand_phase2_vla_oracle_gtmed_pc_f1_8_pn_seed7
```

## B. 训练 GenMed-VLA

### B1. GenMed-VLA + `vec_f1_8`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=genmed \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=false \
  train.epochs=2 \
  train.device=cuda \
  train.wandb.enabled=true \
  train.wandb.mode=offline \
  train.validation_ratio=1.0 \
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData] \
  data.raw.filters.dataset_name=RawHandPointData \
  data.raw.loader.batch_size=16 \
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
  train.wandb.group=rawhand_phase2_vla_genmed \
  train.run_name=rawhand_phase2_vla_genmed_vec_f1_8_seed7
```

### B2. GenMed-VLA + `pc_f1_8` + `point_query`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=genmed \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=false \
  train.epochs=2 \
  train.device=cuda \
  train.wandb.enabled=true \
  train.wandb.mode=offline \
  train.validation_ratio=1.0 \
  data.raw.root_dirs=[Data/RawData/Human/RawHandPointData] \
  data.raw.filters.dataset_name=RawHandPointData \
  data.raw.loader.batch_size=16 \
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
  train.wandb.group=rawhand_phase2_vla_genmed \
  train.run_name=rawhand_phase2_vla_genmed_pc_f1_8_pq_seed7
```

## C. 评估 GenMed-VLA

### C1. 评估 `GenMed-VLA + vec_f1_8`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/genmed/rawhand_phase2_vla_genmed_vec_f1_8_seed7 \
  --checkpoint-kind best \
  --split val \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

### C2. 评估 `GenMed-VLA + pc_f1_8 + point_query`

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/genmed/rawhand_phase2_vla_genmed_pc_f1_8_pq_seed7 \
  --checkpoint-kind best \
  --split val \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

## D. 跑 GTMed-VLA Pipeline

### D1. `vec_f1_8` pipeline

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_gtmed_pipeline.py \
  Outputs/gtmed/rawhand_phase2_vla_oracle_gtmed_vec_f1_8_seed7 \
  --genmed-checkpoint Outputs/genmed/rawhand_phase2_vla_genmed_vec_f1_8_seed7 \
  --gtmed-checkpoint-kind best \
  --genmed-checkpoint-kind best \
  --split val \
  --pipeline-mode cached \
  --loader-mode safe \
  --batch-size 8 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda \
  --genmed-device cpu
```

### D2. `pc_f1_8_pn` pipeline

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_gtmed_pipeline.py \
  Outputs/gtmed/rawhand_phase2_vla_oracle_gtmed_pc_f1_8_pn_seed7 \
  --genmed-checkpoint Outputs/genmed/rawhand_phase2_vla_genmed_pc_f1_8_pq_seed7 \
  --gtmed-checkpoint-kind best \
  --genmed-checkpoint-kind best \
  --split val \
  --pipeline-mode cached \
  --loader-mode safe \
  --batch-size 8 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda \
  --genmed-device cpu
```

## 可选 Ablation

如果点云主线效果接近 `vector`，而你又想判断 `point_query` 是否真的必要，再补这条：

- `GenMed-VLA + pc_f1_8 + mlp predictor`

但这不是第一优先级。  
第一轮只跑上面的 4 组核心实验已经足够判断方向。

## Phase 2 的判定标准

先看 `GenMed`：

- `evaluation -> results -> val -> metrics -> med -> rmse`

再看 pipeline：

- `gtmed_ground_truth_med` 的 `action.rmse`
- `gtmed_with_genmed_predicted_med` 的 `action.rmse`

决策规则：

- 如果 `predicted-med` 相对 `ground-truth med` 的动作 RMSE 劣化 `< 5%`，这条 med 可以继续进入下一阶段
- 如果 `< 2%`，说明这条表示已经很稳
- 如果 `vector` 和 `point-cloud` 都过线，就两条都保留
- 如果只有一条过线，就直接收敛到那一条
