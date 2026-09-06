# DexTeleop 环境与复现指南

本文是环境依赖、安装细节和复现约束的权威说明；可直接复制的最短安装/检查命令见
[`docs/setup/README.md`](./docs/setup/README.md)。项目主环境统一使用 Python 3.10 conda 环境
`dexteleop310`；SAM2、Grounding DINO、VGN、GraspNet 作为抓姿扩展安装到同一个环境。RPent
属于独立的上层项目环境，不是本仓抓姿/仿真运行的前置条件。`core`、`test`、`dataset`、
`annotation`、`sim`、`model`、`openpi`、
`vla_scratch`、`hardware` 是依赖分组，不是九个 conda 环境。其中 `annotation` 还支持
单独安装到 `.venv-annotation`，供不连接真机的异机标注使用。

快速路径：只做后期标注看第 2 节；完整环境首次安装看第 1、3 节；验证主环境看第 5 节；
Isaac Gym 单独看第 6 节；MuJoCo 和冻结保留的 SAPIEN 兼容依赖看第 7 节；只有设备权限问题才进入第 8 节。项目运行命令不在本文维护，完成
环境后回到 [运行文档索引](./docs/README.md)。

## 0. 抓姿 Skill 必读：源码、权重与一键环境

point-grasp 离线 CLI 与仿真评测是同一 `dexteleop310` Python 进程内的直接调用：
MuJoCo/相机数据、SAM2、GraspNet、IK 与运动规划不通过模型 server-client 通信。不要为它新建
`graspnet` 或其他第二 Conda 环境。

基础运行资产和可选感知/抓姿资产包的下载、解压及共享根配置统一见项目根
[`README.md`](./README.md) 的“运行资产”章节；本节只保留环境、源码和复现约束。

### 0.1 源码在哪里？

| 组件 | 当前 worktree 内路径 |
| --- | --- |
| SAM2 | `model/perception/sam2` |
| Grounding DINO | `model/perception/GroundingDINO` |
| VGN | `model/perception/vgn` |
| GraspNet baseline | `model/perception/graspnet-baseline` |
| PointNet2 CUDA 扩展 | `model/perception/graspnet-baseline/pointnet2` |

这四个目录都是 Git submodule；PointNet2 属于 GraspNet 源码，且没有独立权重。

### 0.2 权重在哪里？

权重不提交 Git，不由安装器下载，必须放在共享资产根 `${DEXTELEOP_ASSET_ROOT}` 下：

| 模型 | 相对 `${DEXTELEOP_ASSET_ROOT}` 的必需文件 |
| --- | --- |
| SAM2 | `perception/sam2_grounding_dino/sam2.1_hiera_large.pt` |
| Grounding DINO | `perception/sam2_grounding_dino/groundingdino_swint_ogc.pth` |
| VGN | `perception/sam2_vgn_point_grasp/vgn_conv.pth` |
| GraspNet | `perception/graspnet-point-grasp/checkpoint-rs.tar` |

每个目录还需对应的 `manifest.json` 及真实 SHA-256。使用 worktree 默认 assets 映射时，根就是本仓
`assets/`；下载和解压规则以项目根 [`README.md`](./README.md) 的“运行资产”章节为准。

### 0.3 新机器如何一键安装？

先取得包含权重的 assets，再在 AgenticRobot 根目录运行一条命令。RTX 4090 使用 `8.9`；其他 GPU
把它换成 `python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.major}.{p.minor}")'`
的输出。

```bash
# 纯仿真/感知开发，没有 DM-Tac SDK 时：
bash scripts/install_point_grasp_env.sh \
  --skip-vendor-sdk \
  --cuda-arch-list 8.9

# 真机开发且本机已有 assets/DM_Tac_SDK_Publish_1.2.10 时，去掉 --skip-vendor-sdk。
```

该 wrapper 创建 `dexteleop310`、初始化四个 submodule，并安装/编译 SAM2、Grounding DINO、VGN、
GraspNet 和 PointNet2；它不会下载模型权重。真机且DM-Tac SDK 已就绪时去掉 `--skip-vendor-sdk`。权重必须事先通过 assets 提供。

### 0.4 已有基础 DexTeleop 如何一键更新？

先更新到包含该功能的 AgenticRobot revision，然后运行：

```bash
git pull --ff-only
bash scripts/install_point_grasp_env.sh \
  --update \
  --skip-vendor-sdk \
  --cuda-arch-list 8.9
```

已有 DM-Tac SDK 的真机开发者同样去掉 `--skip-vendor-sdk`。wrapper 会自动执行
`git submodule update --init --recursive`；更新完成后可用以下只读检查确认源码 commit、CUDA 算子和
包导入：

```bash
bash scripts/install_point_grasp_models.sh \
  --backend cuda --with-graspnet --verify-only
```

## 1. 环境边界与唯一来源

| 环境 | 用途 | 是否属于主环境 |
| --- | --- | --- |
| `dexteleop310` / Python 3.10 | 项目代码、测试、数据集、MuJoCo、冻结的 SAPIEN 历史兼容、OpenPI client、真机硬件（含按需安装的 reBotArm RS 依赖）；按需追加 SAM2/Grounding DINO/VGN/GraspNet CUDA 推理 | 是，唯一 DexTeleop 主环境 |
| `.venv-annotation` / Python 3.10 | 封存 Episode 的细粒度阶段标注；无设备、控制、仿真和训练能力 | 否，独立轻量环境 |
| `isaacgym38` / Python 3.8 | Isaac Gym Preview 4 real2sim 检查 | 否，因上游 Python 限制独立 |
| OpenPI server / GPU | 模型加载和 JAX/CUDA 推理服务 | 否，使用 OpenPI 自己的 `uv` 环境 |

项目环境的机器可读来源如下：

- [`pyproject.toml`](./pyproject.toml)：项目元数据、Python 版本和九个 PyPI extras；
- [`environment.yml`](./environment.yml)：conda 基础层，包括 Python、NumPy、SciPy 和 Pinocchio；
- [`uv.lock`](./uv.lock)：九个 extras 的 PyPI 版本、来源与 hash 锁定结果；
- [`scripts/install_conda_env.sh`](./scripts/install_conda_env.sh)：创建、更新和验收完整环境的入口。
- [`scripts/dev_environment_doctor.py`](./scripts/dev_environment_doctor.py)：只读检查当前 worktree 的
  源码解析、Conda 依赖、锁文件和可选 SDK；
- [`scripts/verify_dev_base.sh`](./scripts/verify_dev_base.sh)：多人开发基座的统一无硬件门禁；
- [`scripts/install_annotation_env.sh`](./scripts/install_annotation_env.sh)：创建或同步独立标注环境的入口。

Isaac Gym、OpenPI server 和真实设备的系统驱动不属于 `uv.lock` 的可移植范围；它们的边界和安装方式
在本文后面或对应模块文档中单独说明。

## 2. 只安装后期标注环境

标注机器不需要安装完整 conda 环境，也不需要 Pinocchio、Pink、机器人厂商 SDK、
RealSense、OpenCV、MuJoCo/SAPIEN、PyTorch 或模型部署依赖。前置条件只有：

- Linux；
- 仓库 checkout；
- `uv` 可执行；
- Python 3.10。若本机没有 Python 3.10，`uv` 默认可以下载受管理的解释器。
- 系统 `ffmpeg`/`ffprobe` 可执行文件。当前采集默认的 `mp4v` 视频不是 Chromium 的
  原生可播编码；标注服务首次打开时会在临时目录生成 H.264 兼容缓存。

从仓库根执行：

```bash
cd /path/to/DexTeleop
bash scripts/install_annotation_env.sh
source .venv-annotation/bin/activate
python -m control_station refinement-annotation
```

GUI 打开后可在数据集路径旁点击“浏览目录”通过页面内目录浏览器选目录，再填写标注员 ID。
进入标注后也可以点击顶部“更换数据集”切换到其他目录。数据集可以位于本地磁盘或已经
挂载的共享文件系统，但必须能被标注进程按原路径读写。服务只绑定 loopback；H.264 MP4 会由浏览器
直接解码，旧的 `mp4v`/其他不兼容编码会由系统 ffmpeg 转为临时 H.264 MP4。转码缓存不在
数据集目录内，原始 Parquet 和视频不会被改写；当前视频会自动预读，下一个多阶段 Episode
会后台预热，单阶段 Episode 则默认按需加载；因此 Python 环境仍不需要 OpenCV。

多个升级后的标注程序可以从不同机器通过 SSHFS/NFS 同时打开同一数据集；程序使用共享
数据集上的原子 Episode 锁目录保证同一 Episode 只有一个编辑者。共享数据集的所有标注机
必须使用同一新版标注代码，不能在不确认跨机 `flock` 语义的情况下混用旧版整数据集锁。
进程或网络异常留下的 Episode 锁不会按超时自动抢占，处理方式见
[`LONG_HORIZON_STAGE_ANNOTATION.md`](./docs/real/LONG_HORIZON_STAGE_ANNOTATION.md)。

默认环境目录是仓库内被 git 忽略的 `.venv-annotation`。可以显式指定其他位置或 Python：

```bash
bash scripts/install_annotation_env.sh \
  --venv-path /opt/dexteleop-annotation \
  --python /usr/bin/python3.10
source /opt/dexteleop-annotation/bin/activate
```

重复运行安装脚本会按照冻结的 `uv.lock` 同步环境，不会安装其他 extras。该环境当前仅包含
项目代码、NumPy、PyArrow 及其传递依赖；包根兼容导出按需加载，不会因为导入标注入口而
安装或载入控制、机器人、硬件、仿真、部署和训练依赖。

只读检查某个数据集能否形成标注工作项：

```bash
python -m control_station refinement-annotation \
  /absolute/path/to/dataset \
  --dry-run
```

标注源码继续以 DexTeleop 主仓为唯一来源。需要维护稀疏 checkout 或检查异机标注所需
文件时，使用 [`release/annotation_projection.toml`](./release/annotation_projection.toml)
声明的原路径投影；它不是独立 package，也不复制数据合同。每次修改标注 import 后先运行：

```bash
python scripts/check_annotation_projection.py
```

该检查会重新计算标注入口的本地 Python import closure，拒绝控制、机器人、硬件、仿真、
部署或训练依赖，并确认所有闭包文件和静态资源仍在投影中。数据集、draft、annotation set
和视频转码缓存不属于源码投影。

需要生成可直接复制发布的源码目录时，为每次发布选择一个新的空目录：

```bash
annotation_release_dir=dist/refinement_annotation_20260827_candidate_01
python scripts/export_annotation_projection.py \
  --output "$annotation_release_dir"
python scripts/export_annotation_projection.py \
  --verify "$annotation_release_dir"
```

导出器按清单保留原项目结构，并生成代码文件摘要 receipt；摘要仅用于验证发布代码完整性，
不参与数据集、Episode 或频繁变化配置的准入判断。输出目录已存在时导出器会拒绝覆盖，更新
时应在已忽略的 `dist/` 下使用新的空目录，验证后再替换部署指向。需要版本化的发布快照
必须使用带日期和 source hash 的目录名，并提交自验证 receipt。

## 3. 创建、更新与重建主环境

前置条件：

- Linux，且 `conda` 已安装并可执行；
- 完整真机环境需要 DM-Tac SDK，默认路径为
  `assets/DM_Tac_SDK_Publish_1.2.10`；没有 SDK 时可使用 `--skip-vendor-sdk`；
- 以下命令从仓库根目录执行。

DexTeleop 源码不会安装到共享 Conda 环境中。多个 worktree 可以共享同一套依赖，但每个终端
必须在自己认领的 worktree 根目录运行项目命令，具体协作约定见
[`CONTRIBUTING.md`](./CONTRIBUTING.md)。

首次创建完整环境：

```bash
cd /path/to/DexTeleop
export DEXTELEOP_ROOT="$PWD"
bash scripts/install_conda_env.sh
export DEXTELEOP_ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
eval "$(conda shell.bash hook)"
conda activate "$DEXTELEOP_ENV_NAME"
```

`/path/to/DexTeleop` 必须替换为本机 clone 路径。本文后续用 `DEXTELEOP_ROOT` 表示该目录；
另开 shell 时需要重新导出它。

`conda shell.bash hook` 根据当前 `conda` 命令自动解析安装目录；长期命令不要写死
`/home/.../miniconda3` 或 `/home/.../anaconda3`。如果安装时使用了
`--env-name NAME`，在后续终端中设置相同的 `DEXTELEOP_ENV_NAME=NAME` 即可。

如果 SDK 位于仓库外：

```bash
DEXTELEOP_DMTAC_SDK_ROOT=/path/to/DM_Tac_SDK_Publish_1.2.10 \
  bash scripts/install_conda_env.sh
```

没有 DM-Tac SDK 时，只安装软件、仿真和测试栈：

```bash
bash scripts/install_conda_env.sh --skip-vendor-sdk
```

环境规格发生增删或版本变更时，先修改 `pyproject.toml` / `environment.yml`，再重新生成锁文件并更新
环境：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
uv lock --python 3.10
bash scripts/install_conda_env.sh --update
```

`--update` 会按新的 `environment.yml` 更新并清理 conda 包，再按新的 `uv.lock` 安装 PyPI 包；它不会
自动卸载环境中曾经手工安装、但已不在锁文件中的其他 PyPI 包；它会移除旧的 DexTeleop
editable 安装，避免共享环境固定指向某个 worktree。若需要彻底清除其他残留，建议删除并重建环境：

```bash
export DEXTELEOP_ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
conda deactivate
conda env remove --name "$DEXTELEOP_ENV_NAME"
bash scripts/install_conda_env.sh --env-name "$DEXTELEOP_ENV_NAME"
```

已有同名环境不会被覆盖；也可以通过 `--env-name NAME`、`--sdk-root PATH` 覆盖默认环境名或 SDK 路径。
安装脚本会依次完成：

1. 创建 Python 3.10、NumPy、SciPy、conda-forge Pinocchio 基础环境；
2. 使用 `uv.lock` 的 hash 锁定结果安装全部九个 PyPI 依赖分组；
3. 移除可能把共享环境固定到单个 checkout 的旧 DexTeleop editable 安装；
4. 以 `--no-deps` 安装 `pin-pink==4.2.0`，避免错误拉取 pip 包 `pin`；
5. 以 editable 方式安装仓库外共享的 `dmrobotics==1.2.10` SDK（除非使用
   `--skip-vendor-sdk`）；
6. 执行核心依赖导入检查和共享 worktree 环境诊断。

### 3.1 在 `dexteleop310` 中安装点击抓姿扩展

当前实现没有单独的 GraspNet 模型服务。仿真评测入口
[`experiments/sim_point_grasp/evaluation.py`](./experiments/sim_point_grasp/evaluation.py)
在同一个 Python 进程里依次创建 SAM2 localizer、GraspNet planner、Gateway provider、MuJoCo
环境和运动执行 kernel；每帧 RGB-D 及点击在内存中传递，不经过 HTTP、ZMQ 或其他 client/server
传输。因此复现该评测时，MuJoCo、SAM2、GraspNet、IK 和运动规划全部运行在同一个
`dexteleop310` 环境。Agent Gateway 的 Skill/transport 合同是上层调用边界，不等于模型推理已经
拆成另一个服务。

只有将来实现了明确的模型 RPC 进程、序列化 RGB-D/标定/点击合同、超时与生命周期管理之后，
才应考虑 `dexteleop310` server 加独立模型环境的部署。当前代码没有这个边界，所以不要创建
`graspnet` 环境，也不要把现有 direct-call 测试描述成 server/client。此前本机使用过的额外联合
环境只是 `dexteleop310` 基础层加下述包的重复副本，不是项目合同，合作者无需创建。

从 AgenticRobot worktree 安装或更新唯一 DexTeleop 主环境：

```bash
cd /absolute/path/to/AgenticVLA/AgenticRobot

# 1. 取得父仓锁定的外部源码；模型权重仍由 assets 单独提供。
git submodule update --init --recursive -- \
  model/perception/sam2 \
  model/perception/GroundingDINO \
  model/perception/vgn \
  model/perception/graspnet-baseline

# 2. 首次创建 AgenticRobot 完整主环境。若本机有 DM-Tac SDK，去掉 skip flag。
bash scripts/install_conda_env.sh --skip-vendor-sdk
# 已有 dexteleop310 时改用：
# bash scripts/install_conda_env.sh --update --skip-vendor-sdk

eval "$(conda shell.bash hook)"
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"

# 3. 从本仓 submodule 安装真实感知和点击抓姿 CUDA 栈。
bash scripts/install_point_grasp_models.sh \
  --backend cuda \
  --with-graspnet \
  --cuda-arch-list 8.9
```

主环境更新会先按 `environment.yml`/`uv.lock` 同步并清理基础层，因此更新后重新执行第 3 步，
恢复模型扩展。当前抓姿 CLI、未注册的 Gateway provider 原型和仿真评测都不依赖 RPent；上层 planner 的环境与
部署由 RPent 项目单独维护。RTX 4090 的 compute capability 是 8.9；其他
GPU 应替换为 `python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.major}.{p.minor}")'`
的输出。`install_point_grasp_models.sh` 不下载权重。

RPent 的 core、DexTeleop integration、RLinf、LIBERO、OpenPI server、SAM3 等依赖仍由 RPent
项目各自的环境/profile 管理；不要为了运行本仓 Skill 把它们塞入 `dexteleop310`。AgenticRobot
的全部九个 extras 都由第二步安装，因此仿真、测试、数据、硬件 import 和模型 client 依赖均存在。

安装脚本 fail-closed 固定的关键版本如下；这张表本身不构成模型质量、真机或端到端部署证据：

| 层 | 版本/来源 | 特殊处理 |
| --- | --- | --- |
| Python / CUDA wheel | Python 3.10；PyTorch `2.10.0+cu126`；torchvision `0.25.0+cu126` | NVIDIA 驱动由系统提供；不要把系统 CUDA 驱动写入 Conda lock |
| SAM2 | `model/perception/sam2` @ `aa9b8722d0585b661ded4b3dff1bd103540554ae` | editable，`SAM2_BUILD_CUDA=0`；推理仍使用 PyTorch CUDA |
| Grounding DINO | `model/perception/GroundingDINO` @ `856dde20aee659246248e20734ef9ba5214f5e44` | editable，编译 `_C`；Torch 2.10 兼容头位于 `deployments/perception/` |
| VGN | `model/perception/vgn` @ `d7af0622433f52ae88ebe81533f12b46b33e951a` | editable，只使用 headless inference；安装 `catkin-pkg==1.0.0` |
| GraspNet | `model/perception/graspnet-baseline` @ `280c215129f759ed8649cb4e89fc5dfee55f4f80` | 编译 `pointnet2`，用 `.pth` 暴露 `models/`、`utils/`、`pointnet2/` |
| 数值兼容 | NumPy `1.26.4`；SciPy `1.13.1` | 与 `environment.yml`/`pyproject.toml` 主环境锁定值一致；VGN 未声明更高最低版本 |
| 检测依赖 | transformers `4.39.2`、timm `0.9.16`、supervision `0.22.0`、pycocotools `2.0.8` | 由模型安装脚本锁定 |
| 图像/CUDA build | opencv-contrib-python `4.10.0.84`、ninja `1.11.1.4` | 最后恢复 contrib wheel，保留 `cv2.aruco` |

AgenticRobot 自身不 editable 安装，运行时从当前 worktree 根解析源码。四个模型目录是父仓跟踪的 Git submodule，不再位于 Conda
前缀的 `src/`。每次切分支或 pull 后先执行上述 `git submodule update`，再用安装脚本的
`--verify-only` 检查源码 commit、import 来源与 CUDA 算子：

```bash
bash scripts/install_point_grasp_models.sh \
  --backend cuda --with-graspnet --verify-only

python - <<'PY'
import torch
import sam2, groundingdino, vgn
import pointnet2._ext

assert torch.__version__ == "2.10.0+cu126"
assert torch.cuda.is_available()
print("dexteleop310 point-grasp environment ok", torch.cuda.get_device_name(0))
PY
```

## 4. 依赖分组与特殊包

| 分组 | 内容 | 主环境中是否安装 |
| --- | --- | --- |
| `core` | NumPy、SciPy、Pink/QP 求解和控制循环工具（`pin-pink` 由脚本特殊安装） | 是 |
| `test` | pytest、pytest-timeout | 是 |
| `dataset` | Pandas、PyArrow、Pillow | 是 |
| `annotation` | 独立标注所需的 NumPy、SciPy、OSQP、PyArrow | 是；也可单独安装 |
| `sim` | MuJoCo、Gymnasium，以及冻结保留入口所需的 SAPIEN 兼容依赖 | 是 |
| `model` | PyTorch 2.10、临时 PPO 训练与 checkpoint 推理 | 是 |
| `openpi` | DexTeleop OpenPI websocket/msgpack client | 是 |
| `vla_scratch` | 直接运行 vla-scratch 原生 ZMQ client 所需的 `pyzmq`；不包含模型/server | 是 |
| `hardware` | RealSense、串口、Modbus、USB、DM-Tac SDK 运行库 | 是 |
| perception source | SAM2、Grounding DINO、VGN、GraspNet 固定源码和 CUDA 扩展 | 按需安装到 `dexteleop310`；从 `model/perception/` submodule 安装，权重不在源码中 |

extras 是生成锁文件和安装完整依赖时使用的依赖分组，不要求把 DexTeleop 打包或 editable
安装。需要轻量后期标注时使用第 2 节的独立脚本；其余共享开发、复现和真机环境统一使用
本章的 Conda 安装脚本，避免多个 worktree 争抢 editable 映射。

### B601-RS/reBotArm 有线 CAN 依赖

`reBotArm_control_py` 是独立的 reBotArm 源码/gitlink。它的 RS 配置使用 Python 3.10 和
`motorbridge`；该依赖属于真机硬件附加项，不应让没有 reBotArm gitlink 的基础环境隐式初始化
设备。当前按上游 `pyproject.toml` 的 `motorbridge>=0.4.9` 要求，在 `dexteleop310` 中验证的
兼容版本为 `0.4.9`。上游已提交的 `uv.lock` 仍记录旧的 `0.4.6`，不能把该旧锁定值当作当前
RS 依赖依据。

当目标 worktree 已包含经审查的 `hardware/reBotArm_control_py` 提交，并且确实需要 RS 真机
backend 时，在同一个主环境中执行：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python -m pip install --no-deps motorbridge==0.4.9
```

`--no-deps` 是为了复用本环境已有的 Python/NumPy/Pinocchio 基础层，避免 pip 改写项目锁定的
数值栈；若以后把 reBotArm RS 依赖正式纳入 DexTeleop 主安装流程，应同时更新主项目
`pyproject.toml`、重新生成 `uv.lock` 并由独立 Harness Task 验证，不能手工编辑锁文件。
安装后的无动作 API 检查：

```bash
PYTHONNOUSERSITE=1 python -c \
  'from motorbridge import CallError, Controller, Mode; print("motorbridge import ok", Controller, Mode.MIT, CallError)'
```

该检查不会创建 `Controller` 实例、打开 CAN 或发送电机命令。

### Perception 模型源码与 CUDA 依赖

上表最后一个可选行不是 `pyproject.toml` extra：它是 `dexteleop310` 的后安装层，
其可复现命令、固定 commit 和特殊 wheel 见第 3.1 节。不要把第三方模型 package 加进本仓普通
`uv.lock` 并让不运行感知的用户被迫编译 CUDA；也不要重新 clone 到 Conda prefix。所谓“可选”只表示
基础遥操作/标注开发不必安装；一旦要运行该抓姿 Skill 或仿真评测，它们仍安装在同一个
`dexteleop310`，不是第二个 Conda 环境。

不要安装 pip 包 `pin`。`pip check` 可能报告 `pin-pink requires pin`，这是 `pin-pink` 上游 metadata
与 conda-forge Pinocchio 命名不一致造成的已知例外；安装脚本已经刻意使用 `--no-deps`。

这里的 `openpi` 分组只指 DexTeleop 的 OpenPI client；真正加载模型的 OpenPI server 仍按其 GPU/JAX
版本在独立的 OpenPI `uv` 环境运行。详细的 server/client 启动流程见
[`deployments/openpi/README.md`](./deployments/openpi/README.md)。

`vla_scratch` 同样只覆盖 DexTeleop 侧直接使用原生 ZMQ client 的传输依赖。vla-scratch
server、模型权重和训练依赖仍留在其自己的环境；采用 HTTP、websocket 或外部 relay 时不需要在
DexTeleop 侧安装 `pyzmq`。

## 5. 主环境验证

激活环境后，从当前 worktree 根目录执行只读环境诊断：

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
cd "${DEXTELEOP_ROOT:?set DEXTELEOP_ROOT}"

PYTHONNOUSERSITE=1 python scripts/dev_environment_doctor.py
```

该命令不会导入项目 runtime 或初始化设备。默认情况下，缺少 DM-Tac SDK 只会警告；真机开发
可以设置 `DEXTELEOP_DMTAC_SDK_ROOT` 并加 `--require-vendor-sdk`。若发现历史 editable 映射，
诊断器会同时确认当前根目录仍解析到本 worktree；重新运行安装脚本可移除该映射。

检查关键包没有从 `~/.local` user-site 加载：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import importlib.util

for name in ("numpy", "pandas", "pyarrow", "pinocchio", "hppfcl", "pytest"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f"missing package: {name}")
    print(name, spec.origin)
PY
```

项目测试统一关闭 user-site 和外部 pytest 插件：

`run_pytest.sh` 会显式加载锁定的 `pytest-timeout`：单个测试默认超过 60 秒时先输出
Python 线程栈，超过 120 秒则失败并继续保留 Harness 的外层限时作为兜底。这两个阈值不会
改变测试选择、marker、分片或覆盖率；确认某个已审查测试确实需要更长时间时，可仅对该次
调用设置正数秒值：

```bash
DEXTELEOP_PYTEST_FAULTHANDLER_TIMEOUT_SECONDS=60 \
DEXTELEOP_PYTEST_TIMEOUT_SECONDS=240 \
  bash scripts/run_pytest.sh -q tests/test_known_slow_case.py
```

```bash
# 共享开发基座门禁：环境、锁、源码解析和核心包边界，不连接硬件。
bash scripts/verify_dev_base.sh

# 日常快速反馈：保留普通单元测试，显式排除已审查的长时仿真/IK 闭环。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -m 'unit and not optional and not slow_sim' -q

# 日常并行快速反馈：单元与集成两层并行运行；单元层默认按历史耗时
# 加权分成两个文件 shard，各自使用独立 cache/JUnit，最后合并证据。
# 若当前 worktree 没有 assets/ 链接，先设置绝对路径的共享只读 assets 根。
DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets \
  bash scripts/run_fast_feedback.sh

# 机器资源较小时可退回单一 unit 进程；当前 32-CPU 开发机已验证 3 shard，
# 4 shard 仍属实验配置，不作为集成默认。
DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS=1 \
  DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets \
  bash scripts/run_fast_feedback.sh

# 完整单元回归：合并、提交和推送前仍需执行，不排除 slow_sim。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -m 'unit and not optional' -q

# 不含可选软件栈的完整回归。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -m 'not optional' -q

# 完整环境中的非硬件验收：默认四个耗时加权 shard，逐进程隔离临时目录、
# Python/pytest/MuJoCo/XDG 缓存，并聚合 JUnit；用 SHARDS=1 串行复现。
DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets \
  bash scripts/run_full_acceptance.sh --durations=50
DEXTELEOP_FULL_ACCEPTANCE_SHARDS=1 \
  DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets \
  bash scripts/run_full_acceptance.sh
```

`verify_dev_base.sh` 与 `run_pytest.sh` 都通过 `scripts/resolve_dev_python.sh`
选择 `dexteleop310`（或 `DEXTELEOP_PYTHON` 显式指定的 Python 3.10），因此从未激活
Conda 的 shell 调用时也不会误用 RPent 或系统 Python。

测试按 `unit`、`contract`、`integration`、`optional`、`slow_sim` 和 `hardware` marker 分层；
`slow_sim` 是与执行层正交的耗时标记，仅用于显式快速选择，完整单元/集成门禁仍会执行它。
mocked 硬件行为仍属于 `unit`/`integration`，`hardware` 仅留给未来必须连接真实设备的测试。
`pyarrow` 是正式 parquet 数据写盘和 integration 测试的依赖，完整环境会安装它。

基础数值栈和双臂 IK：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import importlib.metadata as metadata
import numpy as np
import scipy
import pinocchio as pin
import qpsolvers
import pink

assert np.__version__ == "1.26.4", np.__version__
assert scipy.__version__ == "1.13.1", scipy.__version__
assert "osqp" in qpsolvers.available_solvers, qpsolvers.available_solvers
try:
    metadata.version("pin")
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError("Remove pip package 'pin'. Use conda-forge pinocchio only.")

print("numpy", np.__version__)
print("scipy", scipy.__version__)
print("pinocchio", pin.__version__)
print("pink/qpsolvers", "ok")
PY
```

项目和硬件 backend smoke test：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import collection_runtime
import control_station
from hardware.rm65.rm_api2_adapter import Arm, RM65
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ImpedanceArmBackend
from teleop_runtime.backends.grippers import DMTacGripperBackend
from teleop_runtime.backends.hands import InspireTacHandBackend

print("RM65 SDK", RM65, Arm.__module__)
print("collection_runtime import ok", collection_runtime.__name__)
print("control_station import ok", control_station.__name__)
print("teleop_runtime imports ok")
PY
```

DM-Tac 和 RealSense 只做导入/帮助检查，不会连接设备：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import pyrealsense2 as rs
print("pyrealsense2", getattr(rs, "__version__", "unknown"))
PY

cd "${DEXTELEOP_ROOT:?set DEXTELEOP_ROOT}/hardware/DMTAC_gripper"
python run_gripper.py --help
```

## 6. 安装 Isaac Gym 环境

Isaac Gym Preview 4 用于后续 real2sim 和相机位姿检查。不要安装到 `dexteleop310`：包内
`setup.py` 声明只支持 Python `>=3.6,<3.9`，而 `dexteleop310` 是 Python 3.10。建议单独创建
`isaacgym38` 环境，和真机采集环境隔离。

Isaac Gym Preview 4 需要从 NVIDIA Developer 下载离线包。下载后设置包路径；若已经解压过，直接设置
`ISAACGYM_ROOT` 即可：

```bash
export ISAACGYM_ARCHIVE=/path/to/IsaacGym_Preview_4_Package.tar.gz
export ISAACGYM_ROOT=/path/to/isaacgym
```

如果 `ISAACGYM_ROOT` 不存在，先解压：

```bash
mkdir -p "$(dirname "$ISAACGYM_ROOT")"
tar -xzf "$ISAACGYM_ARCHIVE" -C "$(dirname "$ISAACGYM_ROOT")"
```

基础要求：

- Ubuntu 18.04/20.04 风格系统；新系统通常也可运行，但要以实际 smoke test 为准。
- NVIDIA driver 版本不低于 470。
- Python 3.6/3.7/3.8。推荐 Python 3.8，因为包内有 `gym_38.so`。

检查 GPU/驱动：

```bash
nvidia-smi
```

创建独立环境并安装：

```bash
conda create -n isaacgym38 python=3.8 -y
conda activate isaacgym38

python -m pip install --upgrade pip setuptools wheel

# 使用 Isaac Gym Preview 4 官方 rlgpu yml 对应的老版本栈，避免 pip 拉到不兼容 Python 3.8 的新版 Torch。
conda install -c pytorch -c conda-forge -y \
  "pytorch=1.8.1" \
  "torchvision=0.9.1" \
  "cudatoolkit=11.1" \
  "numpy>=1.16.4" \
  "scipy>=1.5.0" \
  "pyyaml>=5.3.1" \
  "tensorboard>=2.2.1"

python -m pip install --no-cache-dir pillow imageio ninja

cd "$ISAACGYM_ROOT/python"
python -m pip install --no-deps -e .
```

验证 Isaac Gym Python package：

```bash
conda activate isaacgym38

python - <<'PY'
from isaacgym import gymapi
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
print("isaacgym import ok; sim created:", sim is not None)
if sim is not None:
    gym.destroy_sim(sim)
PY
```

验证 viewer 和 asset 加载。该示例必须在 Isaac Gym 的 examples 目录运行，因为示例使用相对资产路径：

```bash
cd "$ISAACGYM_ROOT/python/examples"
python joint_monkey.py
```

如果 viewer 找不到 NVIDIA Vulkan 设备，可先指定 NVIDIA ICD：

```bash
export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
```

官方脚本可作为替代方案：

```bash
cd "$ISAACGYM_ROOT"
./create_conda_env_rlgpu.sh
conda activate rlgpu
```

注意：官方脚本会删除并重建名为 `rlgpu` 的环境；已有同名环境时不要直接运行。

后续 real2sim 相机检查优先使用这里生成的双臂 URDF：

```bash
export DEXTELEOP_REAL2SIM_URDF="${DEXTELEOP_ROOT:?set DEXTELEOP_ROOT}/assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
```

该 URDF 是自包含相对 mesh 路径，已包含：

- `world`、`table`、`camera_pole`
- 左右 RM65 基座固定外参
- 左右 `arm_link6` 末端挂载的法兰、D405、`left_camera_rgb` / `right_camera_rgb`
- 左右 gripper 几何

## 7. MuJoCo 验证与冻结保留的 SAPIEN 兼容依赖

MuJoCo 是当前 canonical 仿真路线。完整安装脚本使用 `--all-extras`，因此仍会安装冻结历史入口
所需的 SAPIEN；这只是为了让保留代码可导入和复现旧诊断，不表示 SAPIEN 仍是当前仿真、验证或
发布能力，也不应为它新增功能或调用方。官方 Python binding 包内已经包含 MuJoCo library，不需要
另行下载 MuJoCo 二进制，也不要在主环境中再执行一套不受项目元数据约束的 `pip install`。

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python - <<'PY'
import gymnasium
import mujoco
import sapien

print("gymnasium", gymnasium.__version__)
print("mujoco", mujoco.__version__)
print("sapien", sapien.__version__)
PY
```

验证 viewer：

```bash
python -m mujoco.viewer
```

### 7.1 MuJoCo Gymnasium API

`sim.mujoco.tasks.cube_lift.CubeLiftEnv` 使用 MuJoCo 作为物理后端、使用
Gymnasium 的标准 `reset/step` API 和 `Box` spaces。环境实现本身不导入训练框架；为运行
`sim.mujoco_temp_pipeline` 的临时 PPO，主环境的 `model` extra 固定安装官方 PyTorch 2.10
CUDA 12.6 wheel。未安装 NVIDIA driver 的 debug 机器会自动使用 CPU。

```bash
cd "${DEXTELEOP_ROOT:?set DEXTELEOP_ROOT}"
PYTHONPATH=. python - <<'PY'
import gymnasium
import mujoco
import torch

print("gymnasium", gymnasium.__version__)
print("mujoco", mujoco.__version__)
print("torch", torch.__version__)
print("torch wheel CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
PY
```

项目通过官方 direct wheel URL 固定 `torch==2.10.0+cu126` 的 CPython 3.10 Linux x86_64
构建；不安装 Stable-Baselines3。直接 URL 使 `uv export` 后的 hash requirements 在新机器上
不依赖额外 index 参数。CUDA 12.x minor compatibility 要求 Linux driver 至少为
525.60.13；最终采样服务器仍需执行 `nvidia-smi` 和上面的 CUDA smoke。PyTorch wheel 自带所需
CUDA runtime，通常不需要在 conda 环境重复安装完整 CUDA Toolkit。训练框架不是
`CubeLiftEnv` 自身的运行时依赖。

后续 real2sim 相机检查仍然优先使用同一个双臂 URDF：

```bash
export DEXTELEOP_REAL2SIM_URDF="${DEXTELEOP_ROOT:?set DEXTELEOP_ROOT}/assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
```

先做最小加载检查：

```bash
python - <<'PY'
import os
import mujoco

model_path = os.environ["DEXTELEOP_REAL2SIM_URDF"]
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("mujoco model ok")
print("nbody", model.nbody, "ngeom", model.ngeom, "ncam", model.ncam)
for name in ("left_camera_rgb", "right_camera_rgb"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    print(name, "body_id", body_id)
PY
```

如果该 URDF 不能被 MuJoCo 直接编译，下一步应保留 URDF 作为真实装配基准，再生成一个 MJCF wrapper 或转换后的
MJCF 用于 MuJoCo 检查，避免为了仿真破坏原始坐标定义。

参考：[MuJoCo 官方 Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)。本项目已在
`pyproject.toml` 的 `sim` 分组中固定 MuJoCo 版本。

## 8. 设备权限

运行 GUI 遥操作前，先按 [硬件入口](./docs/hardware/README.md)检查 USB、相机、串口和配置。

### B601-RS/CAN 只读预检

reBotArm RS 当前配置的默认通道是 `can0`。以下命令只枚举主机状态，不改变接口、不打开
电机、不发送 CAN 帧；执行前仍应确认没有其他控制进程占用设备：

```bash
lsusb
ip -brief link
ip -details link show type can
find /dev/serial/by-id -maxdepth 1 -type l -ls 2>/dev/null
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' -o -name 'can*' \) -ls 2>/dev/null
```

只有看到预期的 USB-CAN 适配器和 `can0`（或经过硬件负责人批准并在配置中明确的其他
通道）后，才能进入 reBotArm 的 H1 反馈预检。缺少接口时不要执行
`example/2_zero_and_read.py`、`example/3_mit_control.py` 或任何 `connect/enable/mode/send_*`
路径；这些不是只读连接测试。

如果提示权限不足，执行：

```bash
sudo -E "$CONDA_PREFIX/bin/python" \
  -m teleop_runtime.configure_devices --fix-permissions
```

如果新接 DM-Tac gripper 后 `/dev/ttyUSB*` 不出现，或 `lsusb -t` 看到 CH340/CP210x 是
`Driver=usbfs`，优先按 USB-Serial 抢占处理。若这台机器不需要盲文终端，可执行：

```bash
sudo -E "$CONDA_PREFIX/bin/python" \
  -m teleop_runtime.configure_devices --fix-serial-conflicts
```

之后重新拔插 gripper USB/HUB，再运行 `--fix-permissions`。如果机器需要盲文终端，不要禁用
`brltty`，改用设备专用 udev 例外。

长期建议安装 udev rules，或把当前用户加入 `dialout`、`video` 后重新登录并重新插拔设备。

## 9. 运行文档索引

环境准备完成后，按用途阅读：

- [项目运行文档](./docs/README.md)：按 setup/real/sim/hardware/data 领域查找简洁命令与
  详细说明；
- [根目录项目入口](./README.md)：项目边界、安全原则和各领域导航；
- [`teleop_runtime/README.md`](./teleop_runtime/README.md)：teleop runtime、标定和 dry-run；
- [`hardware/DMTAC_gripper/README.md`](./hardware/DMTAC_gripper/README.md)：DM-Tac USB 配置、触觉和夹爪调试；
- [`sim/mujoco/README.md`](./sim/mujoco/README.md)：MuJoCo real2sim、任务和数据回放；
- [`deployments/openpi/README.md`](./deployments/openpi/README.md)：OpenPI server/client、预检和部署。

## 10. 故障速查

- `ModuleNotFoundError: pinocchio`：确认已用 conda 安装 `pinocchio=3.7.0`。
- `cmeel-boost requires numpy>=2`：环境被 pip `pin` 污染，建议重建环境。
- `pymodbus` import 路径错误：确认版本是 `pymodbus==2.5.3`。
- RealSense 找不到：检查 librealsense/udev 规则和相机是否被其他进程占用。
- Isaac Gym import 失败：确认在 `isaacgym38` 或 `rlgpu` 环境中，且 Python 版本是 3.6/3.7/3.8。
- Isaac Gym viewer 黑屏或无法创建设备：先检查 `nvidia-smi`，再尝试设置
  `VK_ICD_FILENAMES=/path/to/nvidia_icd.json`。
- Isaac Gym 报 `libpython3.x` 或 `libstdc++`：优先设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`，再重新运行 smoke test。
- MuJoCo import 失败：确认已激活 `dexteleop310`，然后重新执行 `bash scripts/install_conda_env.sh --update`。
- MuJoCo viewer 无法打开：先确认当前会话有桌面显示权限；远程或无显示环境下优先做 import 和模型加载检查。
- MuJoCo 直接加载 URDF 失败：保留 URDF 作为装配真值，单独生成 MJCF 检查文件，不要在 URDF 里混入 MuJoCo 专用 workaround。
- `ModuleNotFoundError: motorbridge`：确认在 `dexteleop310` 中按第 4 节安装 `motorbridge==0.4.9`；这只解决 Python API 缺失，不代表 CAN 或电机已连接。
- 串口不可读写：先运行 `configure_devices --fix-permissions`。
- `/dev/ttyUSB*` 不出现或 `Driver=usbfs`：先运行 `configure_devices --fix-serial-conflicts`，重新拔插后再修权限。
- GUI 卡顿且日志出现多个 DM-Tac reset：确认没有误加 `--dmtac-with-tactile` 或触觉可视化。
