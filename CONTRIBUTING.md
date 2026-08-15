# DexTeleop 多人开发约定

本仓库以同一源码仓中的 Conda 开发环境为交付边界，不要求把 DexTeleop 构建成可安装产品。
依赖由 `environment.yml`、`pyproject.toml`、`uv.lock` 和安装脚本共同锁定；源码始终从当前
worktree 根目录运行。

## 1. 开始任务前认领 worktree

开始修改前先确认当前 worktree 和并行状态，再为任务建立独立分支及目录：

```bash
git worktree list
git status --short --branch
git worktree add -b task/<task-name> /absolute/path/to/worktree <baseline>
```

向协作者说明 worktree 绝对路径、分支、baseline commit 和计划修改的目录。一个任务只在已
认领的 worktree 中修改；不要在主 worktree 临时改完后再搬运，也不要 reset、clean 或 stash
其他人的未提交工作。需要修改共享热点文件时，先缩小文件范围并确认没有重叠认领。

个人任务列表和临时推演可以放在根目录 `todos/`，该目录不进入 Git。需要跨 worktree 评审或
被正式文档引用的计划应放入实际 owner 的 `docs/**/plans/`，可复用脚本则放入 `scripts/`
或所属 package，不能让 tracked 代码、测试或文档依赖本地 `todos/`。

## 2. 共享 Conda 环境

创建或修复完整依赖环境：

```bash
bash scripts/install_conda_env.sh --skip-vendor-sdk
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
python scripts/dev_environment_doctor.py
```

同一台机器的多个 worktree 可以共享这个 Conda 环境。安装脚本不会把 DexTeleop 本身以
editable 方式绑定到某个 worktree；每个终端必须先 `cd` 到自己认领的 worktree 根目录再运行
`python -m ...` 或测试。诊断器若发现旧 editable 映射会给出警告，但仍会验证根目录运行时
是否确实解析到当前 worktree；重新运行安装脚本会移除该旧映射。

厂商 DM-Tac SDK 不随 linked worktree 复制。需要硬件开发时，把共享 SDK 放在仓库之外并设置：

```bash
export DEXTELEOP_DMTAC_SDK_ROOT=/absolute/path/to/DM_Tac_SDK_Publish_1.2.10
bash scripts/install_conda_env.sh --update
```

不要在仍被其他任务使用时手工增删共享环境中的包。依赖变更应同时更新声明和 `uv.lock`，由
一名认领者运行安装脚本验证；其他 worktree 随后统一更新环境。

## 3. 提交前的无硬件检查

从当前 worktree 根目录执行：

```bash
bash scripts/verify_dev_base.sh
```

该检查只覆盖环境、包边界和静态/进程内测试，不连接设备，也不授权真机、仿真运动或数据
改写。修改具体模块时，还应运行该模块最小相关测试。真实设备测试必须继续遵守
[`tests/README.md`](./tests/README.md) 的显式硬件门禁。

## 4. 合并交接

交接时说明分支、baseline、修改范围、已运行检查和未验证边界。合并者只集成已认领的提交，
不要用整目录复制覆盖并行 worktree。发现冲突时依据当前代码和合同重新判断，不把旧计划或
个人 `todos` 当作高于实现与测试的规范。
