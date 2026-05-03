# Point-Cloud Models

本目录放点云表示学习组件。当前主要新增的是单帧触觉点云 VAE：

- [`tactile_point_vae.py`](./tactile_point_vae.py)
- [`pointnext_style.py`](./pointnext_style.py)

## TactilePointVAE

`TactilePointVAE` 的训练单位是单帧触觉点云：

```text
[137, 4] -> latent_dim=32 -> [137, 4]
```

其中 4 维重建目标固定为：

```text
(x, y, z, tactile_value)
```

训练数据 preset 会额外把 valid mask 拼到输入特征末尾，因此模型实际输入通常是：

```text
[137, 5] = (x, y, z, tactile_value, valid_mask)
```

但 loss 只监督前 4 维，并且只在 `valid_mask=true` 的点上计算。invalid 点不会贡献 xyz/value 重建 loss。

## Temporal Convention

VAE 本身只处理单帧。为了保持和现有 `pc_f1_8` 点云实验一致，训练和下游 provider 会使用同一组 future offsets：

```text
future_start=1
future_end=40
stride=5
```

因此一个 policy 样本里的 raw point med 是：

```text
[8, 137, 5]
```

VAE 训练时会在模型内部展开为单帧 batch：

```text
[B, 8, 137, 5] -> [B*8, 137, 5]
```

下游使用 latent provider 时，每帧独立编码：

```text
[8, 137, 5] -> [8, 32]
```

这样可以直接对照现有 `pc_f1_8` raw point-cloud med，只把中间表示从 raw points 换成 VAE latent。

## Coordinate Frames

camera 和 wrist-local 是两套不同语义的 VAE，不允许混用。

- camera preset:
  - `Configs/data/desktasktac_rawhand_point_vae.yaml`
  - `Configs/data/desktasktac_rawhand_point_vae_latent.yaml`
- wrist-local preset:
  - `Configs/data/desktasktac_rawhand_point_vae_wrist_local.yaml`
  - `Configs/data/desktasktac_rawhand_point_vae_latent_wrist_local.yaml`

VAE checkpoint、latent provider config、dataset sample metadata 都会记录 `coordinate_frame`。默认 `strict_coordinate_frame=true`，三者不一致会直接报错。

## PointNeXt Backend

`ThirdParty/PointNeXt/openpoints` 已作为第三方代码放在仓库中，但 OpenPoints 依赖 `multimethod` 和 pointnet2 CUDA 扩展。为了先保证 ContactRep 环境里的主链路可运行，当前默认使用 [`pointnext_style.py`](./pointnext_style.py) 里的纯 PyTorch PointNeXt-style 局部聚合模块。

后续如果 OpenPoints 环境和 CUDA op 安装完整，可以新增 adapter 并保持 `TactilePointVAE` 的外部接口不变。
