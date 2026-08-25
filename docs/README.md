# DexTeleop 文档

文档按运行领域组织。每个领域目录的 `README.md` 是该领域**当前主命令的唯一汇总入口**；
同目录其他文件解释参数、数据语义、安全边界、验收和特殊变体，不再复制同一条主命令。

| 领域 | 简洁运行入口 | 详细文档 |
|---|---|---|
| 环境安装与基础检查 | [setup](./setup/README.md) | [环境与复现权威说明](../README_ENV.md) |
| 真机遥操、数采、Replay、Deploy | [real](./real/README.md) | [参数数组](./real/teleop_args.sh)、[遥操数采与长程 Progress Profile](./real/TELEOP_COLLECTION.md)、[Replay](./real/REPLAY.md)、[Deploy](./real/DEPLOY.md)、[UMI](./real/UMI.md)、[OpenPI](./real/OPENPI.md) |
| MuJoCo 与仿真 | [sim](./sim/README.md) | [MuJoCo/仿真说明](./sim/MUJOCO.md)、[State-BC Sim-HIL 计划](./sim/plans/state-bc-sim-hil.md) |
| 外部 Agent Skill 接入 | [agent_gateway](./agent_gateway/README.md) | [局域网 SSH 只读接入](./agent_gateway/SSH_READ_ONLY.md)、[协议与 package 技术说明](../agent_gateway/README.md) |
| Sim human-in-the-loop | [HIL](./HIL/README.md) | [profile 速览、安全边界、owner 导航与综合操作指南](./HIL/README.md) |
| 设备、USB、标定 | [hardware](./hardware/README.md) | [设备与标定](./hardware/DEVICES_AND_CALIBRATION.md)、[DMTAC/Q6](./hardware/DMTAC.md) |
| 数据格式、后处理、分析、测试、验收 | [data](./data/README.md) | [v2.5-p2 数据格式](./data/v25p2/README.md)、[数据集内容与交付](./data/v25p2/DATASET_CONTENT_AND_HANDOFF.md)、[数据集系统盘点](./data/DATASET_INVENTORY_20260820.md)、[长程任务细粒度阶段标注](./real/LONG_HORIZON_STAGE_ANNOTATION.md)、[后处理与测试](./data/POSTPROCESS_AND_TESTING.md)、[v2.3 验收](./data/ACCEPTANCE_V23.md) |
| 跨层端到端指南 | [showcase](./showcase/README.md) | 已落地合同、训练接口和部署链路的贯通说明；不作为新的运行入口 |
| 执行模式与模块边界 | [architecture](./architecture/README.md) | [单源直通、多源组合与运行域边界](./architecture/runtime_domain_boundaries.md)；不作为新的运行入口 |
| 被替代命令和历史报告 | [legacy](./legacy/README.md) | 仅用于追溯和复现实验；[VIVE role diagnostics wrapper 退役报告](./legacy/reports/vive_role_diagnostics_wrapper_retirement_20260825.md) |

跨模块的只读架构与 RTC 图示集中在 [architecture](./architecture/README.md)；它们不是运行入口，也不新增实施契约。
当前 deploy、human teleop 和 replay 的默认模式各自直达既有 target owner；只有显式多源 profile 才加载
`control_composition`。被动 trace/shadow 不改变这个分类。

## 文档类型与查找顺序

同一主题可能同时有入口、合同、计划和证据。它们承担不同职责，不应只按文件更新时间判断哪个更权威：

| 类型 | 位置 | 如何使用 |
|---|---|---|
| 运行入口 | 各领域 `README.md` | 查找当前可复制命令、前置条件和风险等级 |
| 专题说明 | 领域目录中的专题 Markdown | 查找当前参数、数据语义、安全边界和验收方法 |
| 架构与接口合同 | [`architecture/`](./architecture/README.md) | 查找当前模块边界、依赖方向和已实现接口；不替代代码与测试 |
| 共享实施计划 | [`architecture/plans/`](./architecture/plans/README.md)、[`real/plans/`](./real/plans/README.md) | 查找仍未闭合的批次、门禁和回退条件；其中“已完成”通常只描述某个子批次 |
| 报告与 evidence receipt | 各领域 `reports/`、`evidence/` | 读取带日期和范围的历史结果；只证明收据明确覆盖的环境与边界 |
| 历史材料 | [`legacy/`](./legacy/README.md) | 追溯已被替代的命令和结论，不作为当前操作入口 |

遇到冲突时，先核对当前代码、测试和 [`AGENTS.md`](../AGENTS.md)，再看领域入口与专题合同；
计划、报告和历史收据不能反向扩大当前实现或授权范围。

## 使用方式

1. 先进入对应领域的 `README.md` 复制当前命令。
2. 需要理解参数或排查问题时，打开同目录专题文件。
3. Package 内 README 仍是模块接口与局部示例的技术权威，但不是另一套正式运行配方；
   涉及多个模块组合时以这里的领域入口为准。
4. 历史命令只进入 `docs/legacy/`，不与当前入口并列。

## 命令约定

- 除非页面另有说明，命令都从仓库根目录、在 Bash 中执行，并假定已经激活项目环境。
  不要把 `/home/.../miniconda3/envs/.../bin/python` 写进长期文档；不同机器的 Conda
  安装目录可能不同。
- 新终端统一使用下面的可移植前缀。默认环境名是 `dexteleop310`；若安装时用了
  `scripts/install_conda_env.sh --env-name NAME`，只需在激活前覆盖
  `DEXTELEOP_ENV_NAME=NAME`：

  ```bash
  cd /path/to/DexTeleop
  export DEXTELEOP_ROOT="$PWD"
  export DEXTELEOP_ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
  eval "$(conda shell.bash hook)"
  conda activate "$DEXTELEOP_ENV_NAME"
  ```

  已经执行过 `conda init` 的交互式 shell 可以省略 `eval` 行。后续一律使用当前环境中的
  `python`、`python -m pytest` 和 `python -m pip`；只有需要把解释器路径传给子进程时，才用
  `DEXTELEOP_PYTHON="$(command -v python)"` 动态取得，不能猜 Conda 安装前缀。
- OpenPI 模型 server 不属于这个 Conda 环境，继续使用它自己的 `uv` / `.venv`；相关文档用
  `OPENPI_ROOT` 表达路径。`.venv-annotation` 和 `isaacgym38` 也按各自环境章节激活，不能与
  `dexteleop310` 混用。
- `/path/to/...` 表示必须替换的本机路径；`${NAME:?set NAME}` 表示必须先设置的环境变量，
  未设置时 Bash 会在连接设备前停止。
- `latest` 是当前站点保存的 assignment，不是跨机器通用配置。换机器、换 USB 口或换设备后
  必须重新做对应的只读检查/assignment。
- `--help`、dry-run 和离线命令也可能加载较重的软件依赖，但不应连接真实执行器；若实际行为
  与风险等级不符，立即停止并修正文档。

## 常用命令速查

先执行上面的环境前缀，再按用途进入对应领域文档：

| 用途 | 命令 | 等级 |
|---|---|---|
| Canonical 入口检查 | `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --help` | R0 |
| Headless collection 检查 | `python -m teleop_runtime.collection --help` | R0 |
| Policy deploy 检查 | `python -m teleop_runtime.deploy --help` | R0 |
| Direct replay 入口检查 | `python -m teleop_runtime.replay --help` | R0 |
| MuJoCo replay 检查 | `python -m sim.mujoco.replay --help` | R0 |
| Agent Gateway 入口检查 | `python -m agent_gateway --help` | R0 |
| 最快单元测试 | `PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -m 'unit and not optional' -q` | R0 |
| 统一设备只读检查 | `python -m teleop_runtime.configure_devices --check` | H1 |

真机遥操、正式数采、replay 和动作 deploy 的完整命令不放在速查表中，避免跳过 profile、
设备和安全前置条件；从 [real](./real/README.md) 进入。数据写入类命令先使用对应页面的
dry-run，再显式应用。

## 现有 README 的归属

仓库里不会为了“集中命令”而清空模块自己的 README。它们按下表归属，避免把局部示例误当作
正式整机命令：

| 原始说明位置 | 负责内容 | 当前跨模块入口 |
|---|---|---|
| `teleop_runtime/**/README.md` | 单个 teleop/collection/replay/deploy 模块接口 | [real](./real/README.md) |
| `sim/**/README.md` | 仿真 backend、任务和调试参数 | [sim](./sim/README.md) |
| `agent_gateway/README.md` | Skill/Gateway 协议、profile 和 transport 技术契约 | [agent_gateway](./agent_gateway/README.md) |
| `hardware/**/README.md` | 单设备枚举、配置和厂商工具 | [hardware](./hardware/README.md) |
| `utils/dataset_postprocess/**`、`utils/statistic/**` | 单个数据工具参数 | [data](./data/README.md) |
| `README_ENV.md` | 环境依赖与安装细节 | [setup](./setup/README.md) |

判断标准很简单：能够单独验证某个模块的命令可以留在模块 README；用于正式遥操、数采、
回放或部署的组合命令只在对应领域 `README.md` 维护。

## 运行等级

| 等级 | 含义 |
|---|---|
| R0 | 纯解析、import、dry-run 或静态检查 |
| R1 | 本地计算、离线数据、仿真或 GUI，不连接真实执行器 |
| H1 | 只读连接真实设备，可能占用端口但不改变配置/目标 |
| H2 | 修改设备 assignment、标定、权限或持久配置 |
| M | 可能向机械臂、手或夹爪发送运动/力控制命令 |

同一命令同时满足多个等级时按副作用最高者标注。H2/M 命令不能因为出现在文档中就直接运行，
仍需现场安全检查和明确授权。

## 文档维护边界

- 新增当前运行方式：更新领域 `README.md`。
- 参数、验收或安全语义变化：更新同领域专题文件。
- 实验命令通过验收后再提升；旧说明迁入 `docs/legacy/`，保留替代关系。根
  `legacy/` 严格保持本地且不进入 Git。
- 真机遥操参数数组只在 [`docs/real/teleop_args.sh`](./real/teleop_args.sh)维护。
- 带日期、数据来源和源码哈希的评估结果放入所属领域的 `reports/`，不与当前运行入口并列。
- 新增、移动或删除文档后运行 `python scripts/check_docs_navigation.py`；检查器只验证
  repository-local 链接、顶层入口可达性、evidence 索引和 `todos/` 边界，不访问外部链接或执行文档命令。
