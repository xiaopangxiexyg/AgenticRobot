# Training 目录说明

本目录负责整个项目的模型训练流程。

## 核心结构

*   **`train.py`**: 训练入口脚本，使用 `@hydra.main()` 挂载配置。
*   **`runtime.py`**: `run_experiment()`，初始化 Dataloader、Tracker，并执行训练流。
*   **`engine.py`**: 基础 Trainer (`SupervisedExperimentTrainer`)，封装了标准的 `fit` 训练循环及日志记录。
*   **`registry.py`**: 实验注册表（`ExperimentSpec`），管理各个子模块特化配置。

### 子模块

1.  **`VanillaTrain/`**: 基础训练逻辑（Observation to Action）。
2.  **`GTMed/`**: Ground-Truth Manipulation Embedding 训练逻辑。
3.  **`FullTrain/`**: 多模态和混合表示的整体训练逻辑。
4.  **`GenMed/`**: 仅预测中间表示（Observation to Med），不预测 Action。
5.  **`PointCloudVAE/`**: 单帧触觉点云 VAE 预训练。用于把 raw tactile point-cloud med 压缩成 latent med，再接回 `GTMed / GenMed / FullTrain`。

---

## 启动训练与配置覆盖

本项目使用 Hydra 管理 `Configs/` 下的配置。可以直接在命令行中替换配置文件或覆盖指定变量（通过 `.` 指定层级）。

### 1. 基础启动

```bash
# 启动 Vanilla 配置
python Training/train.py --config-name=vanilla

# 或指定实验组合
python Training/train.py experiment=gtmed

# 只训练 Med 生成
python Training/train.py experiment=genmed
```

### 2. 训练触觉点云 VAE

camera-frame 版本：

```bash
python Training/train.py --config-name=pcvae \
  data=desktasktac_rawhand_point_vae \
  model=tactile_point_vae \
  train=real_debug \
  train.run_name=pcvae_camera_f1_8_latent32_seed7
```

wrist-local 版本：

```bash
python Training/train.py --config-name=pcvae \
  data=desktasktac_rawhand_point_vae_wrist_local \
  model=tactile_point_vae \
  train=real_debug \
  train.run_name=pcvae_wristlocal_f1_8_latent32_seed7
```

注意：VAE 是单帧模型，但数据窗口仍沿用 `pc_f1_8` 的 `[8, 137, 5]`。训练时模型内部会展开成 `[B*8, 137, 5]`。第 5 维是 valid mask，只用于 mask loss；重建目标仍是 `(x, y, z, tactile_value)`。

### 3. 真机 Chips VLA 图像增强

真机 vanilla / vanilla-tactile 脚本默认不启用图像增强。启用后只作用于 train split，在生成 RGB mean vision token 前执行；增强包含对称亮度扰动、对比度/饱和度扰动、轻微随机裁剪缩放、噪声、轻微模糊和随机遮挡，不包含翻转或大角度旋转。随机遮挡默认避开画面中心 `1/2 x 1/2` 区域。

默认亮度增强是对称采样：`IMAGE_AUGMENT_BRIGHTNESS=0.35` 表示亮度因子从 `[0.65, 1.35]` 均匀采样，既会变暗也会变亮。各增强模块按独立概率触发，强度在连续区间内采样；噪声标准差从 `[0, IMAGE_AUGMENT_NOISE_STD]` 采样，模糊 `sigma` 从 `[0, IMAGE_AUGMENT_BLUR_SIGMA]` 采样。因此一个 batch 里会自然混合几乎没增强、弱增强和多项增强叠加后的强增强样本。默认模糊较轻，`IMAGE_AUGMENT_BLUR_PROB=0.05`、`IMAGE_AUGMENT_BLUR_SIGMA=0.5`。

```bash
IMAGE_AUGMENT=true \
bash Scripts/train_realrobot_chips_vanilla_tactile_vla.sh
```

常用强度可以用环境变量覆盖，例如：

```bash
IMAGE_AUGMENT=true \
IMAGE_AUGMENT_BRIGHTNESS=0.35 \
IMAGE_AUGMENT_CONTRAST=0.10 \
IMAGE_AUGMENT_CROP_PROB=0.3 \
IMAGE_AUGMENT_NOISE_STD=0.005 \
bash Scripts/train_realrobot_chips_vanilla_vla.sh
```

### 4. 真机 Paper Fold 三子集训练

`paper_fold` 数据配置会从 `Data/RawData/RealRobot/paper_fold/*/meta/info.json` 发现三个子数据集，并用 concat 方式共同训练；默认不启用 validation/test 训练流程。

```bash
bash Scripts/train_realrobot_paper_fold_vanilla_vla.sh
```

```bash
bash Scripts/train_realrobot_paper_fold_vanilla_tactile_vla.sh
```

tactile min/max normalize 版本：

```bash
CONFIG_NAME=realrobot_paper_fold_vanilla_tactile_norm \
RUN_NAME=realrobot_paper_fold_vtactile_vla_ac16_nohis_vis_tacnorm \
bash Scripts/train_realrobot_paper_fold_vanilla_tactile_vla.sh
```

图像增强可以继续用 `IMAGE_AUGMENT=true` 打开。

### 5. 实时参数覆盖 (Override)

通过命令行动态覆盖 `Configs/` 下任意嵌套变量：

*   **覆盖数据和训练参数：**
    ```bash
    python Training/train.py data.loader.batch_size=64 train.learning_rate=1e-4
    ```

*   **覆盖设备与 Epoch：**
    ```bash
    python Training/train.py train.device="cuda:0" train.epochs=100
    ```

*   **覆盖模型结构的嵌套参数：**
    ```bash
    python Training/train.py model.num_attention_heads=8 model.task.enabled=true
    ```
