# DexTeleop 多人开发约定

本仓库以同一源码仓中的 Conda 开发环境为交付边界，不要求把 DexTeleop 构建成可安装产品。
依赖由 `environment.yml`、`pyproject.toml`、`uv.lock` 和安装脚本共同锁定；源码始终从当前
worktree 根目录运行。

## 1. 开始任务前认领 worktree

开始修改前先确认当前 worktree 和并行状态。纯代码、测试和文档开发不要求 Harness 或
机器人资产，可直接认领独立 worktree。只有本次任务需要仿真、模型或设备资产时，才把
解压后的共享 `assets/` 根目录记录到机器本地 Git 配置：

```bash
git worktree list
git status --short --branch
export DEXTELEOP_ASSET_ROOT=/absolute/path/to/assets
python scripts/dev_asset_store.py configure \
  --asset-root "$DEXTELEOP_ASSET_ROOT"
```

以后统一通过认领工具创建 worktree；它会写入所有权描述，并在资产可用时建立被 Git 忽略的
`assets` 兼容链接。缺少资产只产生警告，不阻止纯代码任务；资产是本次验证前提时增加
`--require-assets`，工具会在创建分支前 fail closed：

```bash
python scripts/claim_worktree.py \
  --branch task/<owner>/<task-name>-<YYYYMMDD> \
  --path /absolute/path/to/worktree \
  --base <baseline> \
  --owner <owner> \
  --write-scope '<planned files or packages>' \
  --semantic-scope '<public contract or semantics affected>' \
  --validation '<planned offline validation>'
```

向协作者说明 worktree 绝对路径、分支、baseline commit 和计划修改的目录。一个任务只在已
认领的 worktree 中修改；不要在主 worktree 临时改完后再搬运，也不要 reset、clean 或 stash
其他人的未提交工作。需要修改共享热点文件时，先缩小文件范围并确认没有重叠认领。

`claim_worktree.py` 不会覆盖已有目录、已有分支或已有 `assets` 路径；初始化后半段失败时也
不会擅自删除已经建立的 worktree。用下面的只读命令诊断当前绑定，或为早期手工创建的
worktree 补齐链接：

```bash
python scripts/dev_asset_store.py status
python scripts/dev_asset_store.py link-worktree --worktree /absolute/path/to/worktree
```

共享链接是路径兼容层，不是文件系统只读证明。普通开发不得修改共享资产；需要编辑或生成
资产时，应另行认领明确的可写资产工作区并完成 provenance/hash 评审，不能通过任一 worktree
的 `assets` 链接直接改 canonical store。Runtime 的资产身份与 hash 仍由 TargetRuntime 合同
负责，开发机 Git 配置和绝对路径不得进入 Skill、profile 或提交。

个人任务列表和临时推演可以放在根目录 `todos/`，该目录不进入 Git。需要跨 worktree 评审或
被正式文档引用的计划应放入实际 owner 的 `docs/**/plans/`，可复用脚本则放入 `scripts/`
或所属 package，不能让 tracked 代码、测试或文档依赖本地 `todos/`。

`assets/` 之外的离线数据、运行记录、模型权重、秘密配置和生成输出不能假定会随 worktree
自动出现。开始开发或迁移前运行：

```bash
python scripts/dev_local_resources.py audit
python scripts/dev_local_resources.py plan \
  --source-worktree /absolute/path/to/source \
  --target-worktree /absolute/path/to/target
```

完整分类和迁移边界见
[`docs/setup/LOCAL_RESOURCES.md`](./docs/setup/LOCAL_RESOURCES.md)。其中只有 reviewed、read-mostly
的 `assets/` 使用自动软链接；`runtime_artifacts/` 和采集记录保持 writer 隔离；
`teleop_records/`、`model/`、`hardware/VIVE/triad_openvr/` 含 tracked 例外，禁止整体链接或覆盖。
修改本地资源相关 `.gitignore` 规则时，必须同步更新版本化清单、说明和 focused test。

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

诊断器默认把开发资产缺失报告为 `WARN`，因为纯数据或文档任务不一定需要 RM65 模型。要运行
RM65 仿真、IK 或相关验收时使用 `python scripts/dev_environment_doctor.py
--require-development-assets`，缺失或错绑会在启动前 fail closed。

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
