# 环境安装与基础检查

从仓库根执行。依赖、更新、完整测试层级、独立环境和权限说明统一见
[`README_ENV.md`](../../README_ENV.md)。

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
```

`/path/to/DexTeleop` 必须替换为本机 clone 路径。`conda shell.bash hook` 只初始化当前 shell，
不依赖 Conda 的安装目录，也不修改 shell 配置；持久化设置见
[`README_ENV.md`](../../README_ENV.md)。安装时若使用了 `--env-name NAME`，把
`DEXTELEOP_ENV_NAME` 设置为同一个名称即可。

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

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -m 'unit and not optional' -q
```
