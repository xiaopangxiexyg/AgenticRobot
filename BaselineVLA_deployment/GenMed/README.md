# GenMed

`GenMed` 现在包含两层职责：

- `provider`：从原始样本中提取或组织 ground-truth med target
- `experiment=genmed`：训练 `observation -> med` 模型，不预测 action

## 当前支持的 med 模式

- 137 维触觉向量
  - 数据侧通常来自 `data.raw.sequence.med.vector.field`
  - target shape 可以是 `[T * 137]` 或 `[T, 137]`
- 点云触觉
  - 数据侧通常来自 `data.raw.sequence.med.point_cloud.*`
  - target shape 通常是 `[T, N, C]`
  - 当前 `N` 通常是 `137`，但代码不硬编码这个值
- VAE 压缩后的点云触觉 latent
  - provider: `TactilePointVAE_Latent_Med`
  - raw source 仍来自点云触觉 `[T, 137, 5]`
  - 输出 target shape 通常是 `[T, 32]`
  - 当前推荐沿用 `pc_f1_8`，即 `[8, 32]`

## 训练入口

统一入口：

```bash
python Training/train.py experiment=genmed
```

兼容壳入口：

```bash
python Training/GenMed/train.py
```

常见示例：

```bash
python Training/train.py \
  experiment=genmed \
  data=desktasktac_real \
  model=minimal_mlp \
  train=train_template \
  train.run_name=genmed_tactile_test
```

```bash
python Training/train.py \
  experiment=genmed \
  data=desktasktac_rawhand_point_real \
  model=minimal_mlp \
  model.med_branch.projector.type=pointnet \
  model.med_branch.predictor.type=point_query \
  train=real_debug \
  train.run_name=genmed_point_smoke
```

使用预训练 VAE latent 作为 med target：

```bash
python Training/train.py --config-name=genmed \
  data=desktasktac_rawhand_point_vae_latent \
  data.med.checkpoint=Outputs/pcvae/pcvae_camera_f1_8_latent32_seed7/best.pt \
  model=minimal_mlp \
  train=real_debug \
  train.run_name=genmed_pcvae_latent_camera_f1_8_seed7
```

wrist-local checkpoint 必须配 wrist-local latent preset：

```bash
python Training/train.py --config-name=genmed \
  data=desktasktac_rawhand_point_vae_latent_wrist_local \
  data.med.checkpoint=Outputs/pcvae/pcvae_wristlocal_f1_8_latent32_seed7/best.pt \
  model=minimal_mlp \
  train=real_debug \
  train.run_name=genmed_pcvae_latent_wristlocal_f1_8_seed7
```

## 设计约定

- `genmed` 只消费 `observations`
- `genmed` 只监督 `targets["med"]`
- `med` 的 shape 与 kind 统一由 `data.raw.sequence.med -> data.med -> med_spec` 推断
- `mlp / vla / diffusion` 共用同一套 `med_spec` 与 target 还原逻辑
- point-cloud target 默认保留结构化 shape，不强制在数据层扁平化
- tactile point VAE latent 默认是 `[T, 32]`，作为 vector med 接入现有 `GTMed / GenMed / FullTrain`
- `model.family` 只表示 observation backbone
- `pointnet` 只表示 point-cloud med 的 projector
- `point_query` 只表示 point-cloud med 的 predictor
- camera 和 wrist-local VAE checkpoint 不允许混用；provider 默认会做 strict coordinate-frame 校验

## 关键文件

- provider 注册：[`registry.py`](./registry.py)
- 通用 provider：[`future_sequence_med.py`](./future_sequence_med.py)
- tactile provider：[`future_tactile_med.py`](./future_tactile_med.py)
- point-cloud provider：[`future_point_cloud_med.py`](./future_point_cloud_med.py)
- tactile point VAE latent provider：[`tactile_point_vae_latent_med.py`](./tactile_point_vae_latent_med.py)
- experiment builder：[`Training/GenMed/build.py`](../Training/GenMed/build.py)
- 可视化脚本：[`visualize_genmed_predictions.py`](../TestScripts/visualize_genmed_predictions.py)

## 配置框架

- `data.raw.sequence.med.kind=vector|point_cloud` 选择目标 med 形态
- `data.raw.sequence.med.side=right|left` 可用于自动推默认字段名
- `data.med.provider_type=FutureSequence_Med` 统一组织 target
- `model.med_branch.projector.type` 控制 `gtmed/fulltrain` 的 med 输入编码
- `model.med_branch.predictor.type` 控制 `genmed/fulltrain` 的 med 输出头

## 可视化测试

训练完成后可以对 checkpoint 做一次简单的 target/pred 可视化：

```bash
python TestScripts/visualize_genmed_predictions.py \
  --checkpoint Outputs/genmed/<run_name>/best.pt
```

脚本会：

- 取一个 batch 做前向
- 保存原始 `pred/target` 张量
- 向量 med 输出 line plot 或 heatmap
- 点云 med 输出 target/pred 的 2D scatter 对比
