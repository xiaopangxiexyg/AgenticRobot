# 环境安装与基础检查

从仓库根执行。依赖、更新、完整测试层级、独立环境和权限说明统一见
[`README_ENV.md`](../../README_ENV.md)。多人并行修改先按
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) 认领独立 worktree。

## 1. 创建或修复环境

只进行封存数据的后期阶段标注时，不要安装完整真机环境：

```bash
cd /path/to/DexTeleop
bash scripts/install_annotation_env.sh
source .venv-annotation/bin/activate
python -m control_station refinement-annotation
```

该轻量环境不安装硬件 SDK、机器人控制、仿真、训练或部署依赖。详细边界与自定义安装路径见
[`README_ENV.md` 的标注环境章节](../../README_ENV.md#2-只安装后期标注环境)。

完整项目或真机机器才使用：

```bash
cd /path/to/DexTeleop
bash scripts/install_conda_env.sh
export DEXTELEOP_ROOT="$PWD"
export DEXTELEOP_ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
eval "$(conda shell.bash hook)"
conda activate "$DEXTELEOP_ENV_NAME"
python scripts/dev_environment_doctor.py
```

`/path/to/DexTeleop` 必须替换为本机 clone 路径。`conda shell.bash hook` 只初始化当前 shell，
不依赖 Conda 的安装目录，也不修改 shell 配置；持久化设置见
[`README_ENV.md`](../../README_ENV.md)。安装时若使用了 `--env-name NAME`，把
`DEXTELEOP_ENV_NAME` 设置为同一个名称即可。

多个 worktree 可以共享同一个 Conda 依赖环境，但所有项目命令都必须从各自 worktree 根目录
运行；安装脚本不会把 DexTeleop editable 绑定到其中某一个 worktree。

解压的机器人模型等大文件也不应复制到每个 worktree。按
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) 配置一次 development asset store，并通过标准
认领工具创建 worktree。RM65 仿真或 IK 启动前执行：

```bash
python scripts/dev_environment_doctor.py --require-development-assets
```

该检查只验证开发资产根和 presence-only bundle requirements；运行时资产内容身份仍以
TargetRuntime 的版本化 hash 合同为准。

离线数据集、采集记录、运行证据、模型权重和机器本地配置不属于 asset store。使用
[`LOCAL_RESOURCES.md`](./LOCAL_RESOURCES.md) 的分类和只读工具盘点或规划 worktree 迁移：

```bash
python scripts/dev_local_resources.py audit
python scripts/dev_local_resources.py plan \
  --source-worktree /absolute/path/to/source \
  --target-worktree /absolute/path/to/target
```

该工具不会复制或链接数据。只有 `assets/` 允许标准自动链接；可变记录保持 writer 隔离，mixed
tracked/ignored 根目录不得整体替换。

## 2. 入口检查（R0）

```bash
python -m teleop_runtime.debug.run_vive_rm65_real_teleop --help
python -m teleop_runtime.collection --help
python -m teleop_runtime.collection umi --help
python -m teleop_runtime.replay --help
python -m teleop_runtime.deploy --help
python -m sim.mujoco.replay --help
```

`teleop_runtime.collection` 保留 headless/兼容 parser；它的顶层 `--help` 不枚举 UMI，因此
UMI 子命令需要单独检查。

## 3. 设备只读预检（H1）

环境检查通过后，按 [硬件入口的第 1 节](../hardware/README.md)运行唯一的全设备只读检查。

## 4. 最快单元测试（R0）

共享开发基座的环境与包边界检查：

```bash
bash scripts/verify_dev_base.sh
```

只运行默认单元测试：

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -m 'unit and not optional' -q
```

## 5. 兼容与开发工具说明

- [`ENVIRONMENT.md`](./ENVIRONMENT.md) 只保留旧链接兼容；新增环境说明应更新本页或仓库根
  [`README_ENV.md`](../../README_ENV.md)，不要再建立第三份安装步骤。
- [`CODEX_BEINGBEYOND_REMOTE.md`](./CODEX_BEINGBEYOND_REMOTE.md) 是特定远端开发工具和
  provider 的运维说明，不定义 DexTeleop 的 Python 依赖、运行环境或部署合同。使用其中命令时仍应遵守
  密钥不入仓库、远端目标显式选择和只读优先的边界。
