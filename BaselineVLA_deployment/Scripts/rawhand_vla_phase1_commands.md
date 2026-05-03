# RawHand VLA Phase 1 Commands

## 原则

如果你的假设是 **`minimal_mlp` 容量太小导致触觉 med 没法发挥**，那最干净的第一轮 VLA 对比应该先跑：

- 同一数据预算
- 同一 epoch
- 同一 seed
- 同一 med 候选
- 同一评测方式
- **只换 backbone**

因此我建议把 VLA Phase 1 分成两层：

1. **严格容量对比版**
   `basic_vla + model.task.enabled=false`
   这一步只回答“更强 backbone 会不会让 med 变得有用”。
2. **真实 VLA 确认版**
   `basic_vla + model.task.enabled=true`
   这一步才回答“带 task text 的完整 VLA 结论是否一致”。

## 建议先跑哪些

先不要把 S1 的 6 个候选全部搬到 VLA 上。那样会把“表示搜索”和“容量验证”混在一起。

第一轮只跑 3 条最有信息量的线：

- `Vanilla-VLA`
- `GTMed-VLA + vec_f1_8`
- `GTMed-VLA + pc_f1_8_pn`

这样最科学，因为它直接检验：

- 更大 backbone 下，触觉 med 是否开始超过 `Vanilla`
- `point cloud` 和 `vector` 哪条线更值得进入后续 `GenMed`

## 统一预算

下面所有命令统一采用：

- `2 epoch`
- `seed=7`
- `1024 train / 1024 val`
- `validation_ratio=1.0`
- `RawHandPointData`
- 单进程 loader，避免再次卡死

说明：

- 我把 `batch_size` 设成 `32`，这是更稳的 VLA 设置。
- 这会让每个 epoch 的更新步数比 MLP S1 多一些，所以它不是“逐步数完全相同”的复刻，而是“同数据预算、同时间预算、同 backbone 家族内部可比”的对比。
- 如果你的显存足够稳定，也可以把下面所有命令里的 `data.raw.loader.batch_size=32` 统一改成 `64`，这样更接近 MLP S1。

## A. 严格容量对比版

### A1. Vanilla-VLA

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=vanilla \
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
  train.wandb.group=rawhand_phase1_vla_iso \
  train.run_name=rawhand_phase1_vla_iso_shared_vanilla_seed7
```

### A2. GTMed-VLA + `vec_f1_8`

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
  model.med_branch.projector.type=mlp \
  train.wandb.group=rawhand_phase1_vla_iso \
  train.run_name=rawhand_phase1_vla_iso_gtmed_vec_f1_8_seed7
```

### A3. GTMed-VLA + `pc_f1_8_pn`

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
  model.med_branch.projector.type=pointnet \
  train.wandb.group=rawhand_phase1_vla_iso \
  train.run_name=rawhand_phase1_vla_iso_gtmed_pc_f1_8_pn_seed7
```

## B. 真实 VLA 确认版

如果 A 组已经显示 `GTMed-VLA` 开始明显超过 `Vanilla-VLA`，再补这一组。

注意：

- 这组会引入 task text 编码，不再是纯容量对比。
- 如果 `ContactRep` 环境里缺 `transformers` 或 `sentencepiece`，这组会直接失败。

### B1. Vanilla-VLA + task

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=vanilla \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=true \
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
  train.wandb.group=rawhand_phase1_vla_task \
  train.run_name=rawhand_phase1_vla_task_shared_vanilla_seed7
```

### B2. GTMed-VLA + `vec_f1_8` + task

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=gtmed \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=true \
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
  model.med_branch.projector.type=mlp \
  train.wandb.group=rawhand_phase1_vla_task \
  train.run_name=rawhand_phase1_vla_task_gtmed_vec_f1_8_seed7
```

### B3. GTMed-VLA + `pc_f1_8_pn` + task

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python Training/train.py --config-name=train_runner \
  data=desktasktac_rawhand_point_real \
  train=real_debug \
  experiment=gtmed \
  model=basic_vla \
  project.seed=7 \
  model.task.enabled=true \
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
  model.med_branch.projector.type=pointnet \
  train.wandb.group=rawhand_phase1_vla_task \
  train.run_name=rawhand_phase1_vla_task_gtmed_pc_f1_8_pn_seed7
```

## 评测命令

由于 `Vanilla` 路径当前存在 `--split val` 会触发 `KeyError: 'train'` 的问题，VLA 评测统一建议用 `--split both`。最终只比较其中的 `val` 指标。

### A 组评测

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/vanilla/rawhand_phase1_vla_iso_shared_vanilla_seed7 \
  --checkpoint-kind best \
  --split both \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/gtmed/rawhand_phase1_vla_iso_gtmed_vec_f1_8_seed7 \
  --checkpoint-kind best \
  --split both \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

```bash
/home/zc/anaconda3/envs/ContactRep/bin/python TestScripts/evaluate_checkpoint.py \
  Outputs/gtmed/rawhand_phase1_vla_iso_gtmed_pc_f1_8_pn_seed7 \
  --checkpoint-kind best \
  --split both \
  --loader-mode safe \
  --batch-size 16 \
  --num-workers 0 \
  --save-samples 0 \
  --visualize none \
  --device cuda
```

### B 组评测

把上面 3 条命令里的 run 名分别替换成：

- `rawhand_phase1_vla_task_shared_vanilla_seed7`
- `rawhand_phase1_vla_task_gtmed_vec_f1_8_seed7`
- `rawhand_phase1_vla_task_gtmed_pc_f1_8_pn_seed7`

## 比较口径

第一排序指标：

- `evaluation -> results -> val -> metrics -> action -> rmse`

第二排序指标：

- `metrics.json` 里的 `best_val_action_loss`

判断逻辑：

1. 如果 A 组里 `GTMed-VLA` 明显超过 `Vanilla-VLA`，说明你关于“MLP capacity cap 过低”的怀疑是成立的。
2. 如果 A 组里仍然没有 med 收益，那问题更可能不在 backbone 容量，而在 med 本身的信息密度、训练预算，或当前 GTMed 训练方式。
3. 如果 A 组成立、B 组也成立，就可以直接把 `pc_f1_8_pn` 和 `vec_f1_8` 送去做 `GenMed + pipeline`。

## 我建议你先跑哪组

先跑 **A 组**。

这是最干净、最能回答你当前问题的一组。  
如果你愿意只先启动 1 条最关键的 med 线，那就先跑：

- `rawhand_phase1_vla_iso_shared_vanilla_seed7`
- `rawhand_phase1_vla_iso_gtmed_pc_f1_8_pn_seed7`

这两条已经足够告诉我们：

- 更强 backbone 下，point-cloud med 是否开始产生净收益
