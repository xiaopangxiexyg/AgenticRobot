# DexTeleop 环境与复现指南

本文是环境依赖、安装细节和复现约束的权威说明；可直接复制的最短安装/检查命令见
[`docs/setup/README.md`](./docs/setup/README.md)。项目主环境统一使用 Python 3.10 conda 环境
`dexteleop310`；`core`、`test`、`dataset`、`annotation`、`sim`、`model`、`openpi`、
`vla_scratch`、`hardware` 是依赖分组，不是九个 conda 环境。其中 `annotation` 还支持
单独安装到 `.venv-annotation`，供不连接真机的异机标注使用。

快速路径：只做后期标注看第 2 节；完整环境首次安装看第 1、3 节；验证主环境看第 5 节；
Isaac Gym 单独看第 6 节；MuJoCo 和冻结保留的 SAPIEN 兼容依赖看第 7 节；只有设备权限问题才进入第 8 节。项目运行命令不在本文维护，完成
环境后回到 [运行文档索引](./docs/README.md)。

## 1. 环境边界与唯一来源

| 环境 | 用途 | 是否属于主环境 |
| --- | --- | --- |
| `dexteleop310` / Python 3.10 | 项目代码、测试、数据集、MuJoCo、冻结的 SAPIEN 历史兼容、OpenPI client、真机硬件 | 是 |
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
python scripts/export_annotation_projection.py \
  --output release/refinement_annotation
python scripts/export_annotation_projection.py \
  --verify release/refinement_annotation
```

导出器按清单保留原项目结构，并生成代码文件摘要 receipt；摘要仅用于验证发布代码完整性，
不参与数据集、Episode 或频繁变化配置的准入判断。输出目录已存在时导出器会拒绝覆盖，更新
时应使用新的版本目录，验证后再替换部署指向。

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

## 4. 依赖分组与特殊包

| 分组 | 内容 | 主环境中是否安装 |
| --- | --- | --- |
| `core` | NumPy、SciPy、Pink/QP 求解和控制循环工具（`pin-pink` 由脚本特殊安装） | 是 |
| `test` | pytest | 是 |
| `dataset` | Pandas、PyArrow、Pillow | 是 |
| `annotation` | 独立标注所需的 NumPy、SciPy、OSQP、PyArrow | 是；也可单独安装 |
| `sim` | MuJoCo、Gymnasium，以及冻结保留入口所需的 SAPIEN 兼容依赖 | 是 |
| `model` | PyTorch 2.10、临时 PPO 训练与 checkpoint 推理 | 是 |
| `openpi` | DexTeleop OpenPI websocket/msgpack client | 是 |
| `vla_scratch` | 直接运行 vla-scratch 原生 ZMQ client 所需的 `pyzmq`；不包含模型/server | 是 |
| `hardware` | RealSense、串口、Modbus、USB、DM-Tac SDK 运行库 | 是 |

extras 是生成锁文件和安装完整依赖时使用的依赖分组，不要求把 DexTeleop 打包或 editable
安装。需要轻量后期标注时使用第 2 节的独立脚本；其余共享开发、复现和真机环境统一使用
本章的 Conda 安装脚本，避免多个 worktree 争抢 editable 映射。

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

```bash
# 共享开发基座门禁：环境、锁、源码解析和核心包边界，不连接硬件。
bash scripts/verify_dev_base.sh

# 日常快速反馈：保留普通单元测试，显式排除已审查的长时仿真/IK 闭环。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -m 'unit and not optional and not slow_sim' -q

# 完整单元回归：合并、提交和推送前仍需执行，不排除 slow_sim。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -m 'unit and not optional' -q

# 不含可选软件栈的完整回归。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -m 'not optional' -q

# 完整环境中的全部测试。
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash scripts/run_pytest.sh -q
```

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
from hardware.rm65.robotic_arm_package.robotic_arm import RM65, dllPath
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ImpedanceArmBackend
from teleop_runtime.backends.grippers import DMTacGripperBackend
from teleop_runtime.backends.hands import InspireTacHandBackend

print("RM65 SDK", RM65, dllPath)
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
- 串口不可读写：先运行 `configure_devices --fix-permissions`。
- `/dev/ttyUSB*` 不出现或 `Driver=usbfs`：先运行 `configure_devices --fix-serial-conflicts`，重新拔插后再修权限。
- GUI 卡顿且日志出现多个 DM-Tac reset：确认没有误加 `--dmtac-with-tactile` 或触觉可视化。
